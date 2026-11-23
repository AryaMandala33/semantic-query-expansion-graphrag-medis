"""
build_testset.py — Pemetaan label acuan set uji ke subset PrimeKG sistem utama
(Bab III 3.4.5). Jalankan setelah set uji diverifikasi dosen.

Untuk setiap kueri:
  1. Konsep acuan (gold) dan pengecoh dipetakan ke node pada database sistem utama
     dengan pencocokan nama yang sama seperti sistem (tanpa membedakan huruf besar/kecil).
       exact      -> nama dan tipe node sesuai draf
       label_diff -> nama ada, tetapi tipe node berbeda dari draf     (PERLU DICEK)
       not_found  -> tidak ada pada subset                            (kueri harus diganti)
  2. Node bernama identik pada ontologi lain (mis. MONDO 'gout' dan HPO 'Gout')
     ditambahkan sebagai konsep acuan setara (auto_equivalent).
  3. Himpunan node relevan = node acuan + tetangga 1-hop melalui relasi relevan
     (relevant_relations) — dasar Precision@5 dan MRR (3.6.1–3.6.2).
  4. Pemeriksaan kebocoran (nama jawaban tertulis di kueri) dan tumpang tindih
     dengan contoh few-shot / set dev.

    python build_testset.py
Input : data/ambiguity_testset_draft.json
Output: data/testset_utama.json, reports/utama/testset_report.txt
"""
import json
import re
import sys
from pathlib import Path

from core.config import RELATIONS
from core.graph_retriever import GraphRetriever
from core.llm_expander import FEW_SHOT_EXAMPLES

INPUT = Path("data/ambiguity_testset_draft.json")
OUTPUT = Path("data/testset_utama.json")
REPORT = Path("reports/utama/testset_report.txt")

# Istilah set dev (tests/test_expander_v2.py) — tidak boleh muncul di set uji.
DEV_TERMS = {"failure", "pe", "thyroid", "pulmonary embolism", "pleural effusion",
             "acute kidney injury", "hyperthyroidism", "osteoporosis", "bone pain"}
STOP = {"of", "the", "and", "with", "to", "in", "s", "a", "an", "or", "for", "by", "on", "at"}


def norm(t):
    return " ".join(str(t).lower().strip().split())


def toks(t):
    return {w for w in re.findall(r"[a-z0-9]+", norm(t)) if w not in STOP}


def few_shot_terms():
    terms = set()
    for ex in FEW_SHOT_EXAMPLES:
        out = ex["output"]
        terms.add(norm(out["ambiguous_term"]))
        terms.update(norm(s) for s in out.get("senses_considered", []))
        terms.update(norm(c["name"]) for c in out.get("candidates", []))
    return terms


class Tee:
    def __init__(self, p):
        p.parent.mkdir(parents=True, exist_ok=True)
        self.f, self.o = open(p, "w", encoding="utf-8"), sys.stdout

    def write(self, s):
        self.o.write(s)
        self.f.write(s)

    def flush(self):
        self.o.flush()
        self.f.flush()


def map_name(r, name, label):
    rows = r.find_by_name(name)
    same = [x for x in rows if x["label"] == label]
    pick, status = (same[0], "exact") if same else ((rows[0], "label_diff") if rows else (None, "not_found"))
    if pick is None:
        return {"status": status, "node_index": None, "kg_name": None, "kg_label": None, "kg_id": None}
    return {"status": status, "node_index": pick["node_index"], "kg_name": pick["name"],
            "kg_label": pick["label"], "kg_id": f"{pick['source']}:{pick['id']}"}


def equivalents(r, gold):
    known = {g["node_index"] for g in gold if g.get("node_index") is not None}
    out = []
    for g in list(gold):
        if not g.get("kg_name"):
            continue
        base = norm(g["kg_name"]).removesuffix(" (disease)")
        for row in r.find_by_name(base):
            if row["node_index"] in known or row["name"].lower().startswith("obsolete"):
                continue
            known.add(row["node_index"])
            out.append({"name": row["name"], "label": row["label"], "status": "equivalent",
                        "auto_equivalent": True, "node_index": row["node_index"],
                        "kg_name": row["name"], "kg_label": row["label"],
                        "kg_id": f"{row['source']}:{row['id']}"})
    return out


def main():
    sys.stdout = Tee(REPORT)
    items = json.loads(INPUT.read_text(encoding="utf-8"))
    fs = few_shot_terms()
    r = GraphRetriever()
    summary = {"OK": 0, "REVIEW": 0}

    for it in items:
        issues = []
        for group in ("gold", "distractors"):
            it[group] = [g for g in it[group] if not g.get("auto_equivalent")]
            for g in it[group]:
                g.update(map_name(r, g["name"], g.get("label")))
                if group == "gold" and g["status"] != "exact":
                    issues.append(f"gold '{g['name']}' -> {g['status']}")
                if group == "distractors" and g["status"] == "not_found":
                    issues.append(f"pengecoh '{g['name']}' tidak ada di subset")
        if not any(g["status"] == "exact" for g in it["gold"]):
            issues.append("TIDAK ADA konsep acuan pada subset -> kueri perlu diganti (3.4.5)")
        it["gold"].extend(equivalents(r, it["gold"]))

        rels = it.get("relevant_relations") or []
        bad = [x for x in rels if x not in RELATIONS]
        if not rels or bad:
            issues.append(f"relevant_relations tidak valid: {rels}")
        gold_idx = sorted({g["node_index"] for g in it["gold"] if g.get("node_index") is not None})
        relevant = set(gold_idx)
        for gi in gold_idx:
            relevant.update(r.relevant_neighbors(gi, [x for x in rels if x in RELATIONS]))
        it["gold_node_indices"] = gold_idx
        it["relevant_node_indices"] = sorted(relevant)
        if len(relevant) == len(gold_idx):
            issues.append(f"konsep acuan tidak punya tetangga melalui {rels}")

        q = toks(it["query"])
        for g in it["gold"]:
            gt = toks(g["name"])
            if gt and gt <= q:
                issues.append(f"KEBOCORAN: nama jawaban '{g['name']}' tertulis di kueri")
        terms = {norm(it["ambiguous_term"])} | {norm(g["name"]) for g in it["gold"] + it["distractors"]}
        overlap = terms & (fs | DEV_TERMS)
        if overlap:
            issues.append(f"tumpang tindih dengan few-shot/dev: {sorted(overlap)}")

        it["mapping_status"] = "OK" if not issues else "REVIEW"
        it["mapping_issues"] = issues
        summary[it["mapping_status"]] += 1

        print("\n" + "-" * 78)
        print(f"{it['query_id']} [{it['ambiguity_type']}] '{it['ambiguous_term']}' -> {it['mapping_status']}")
        print(f"  kueri: {it['query']}")
        for g in it["gold"]:
            print(f"  GOLD {g['status']:10s} {str(g.get('kg_id')):24.24s} {g['name']} [{g.get('kg_label')}]")
        for d in it["distractors"]:
            print(f"  DIST {d['status']:10s} {str(d.get('kg_id')):24.24s} {d['name']} [{d.get('kg_label')}]")
        print(f"  relasi relevan {rels}: {len(relevant)} node relevan")
        for iss in issues:
            print(f"  ! {iss}")

    OUTPUT.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "=" * 78)
    print(f"Item OK: {summary['OK']}   REVIEW: {summary['REVIEW']}")
    print(f"Disimpan ke {OUTPUT}")


if __name__ == "__main__":
    main()
