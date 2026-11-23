"""
baseline_embedding.py — Baseline tambahan: pencocokan embedding kueri ke nama node.

Ditambahkan atas saran dosen pembimbing (revisi artikel, 23 Nov 2026) sebagai
baseline yang lebih kuat daripada pencocokan kata kunci. Rancangan ditetapkan
SEBELUM dijalankan dan tidak disetel berdasarkan hasil (lihat BASELINE_EMBEDDING.md).

Protokol (identik dengan baseline kata kunci, hanya cara memilih node awal yang berbeda):
  1. Semua node subset PrimeKG (kecuali nama "obsolete ...") diubah menjadi teks
     "<nama> (<tipe>)" dan di-embed dengan all-mpnet-base-v2 (model yang sama
     dengan Disambiguation Ranker).
  2. Lima node dengan cosine similarity tertinggi terhadap kueri asli dipakai
     sebagai node awal (setara batas 5 node pada baseline kata kunci).
  3. Tetangga 1-hop diambil dengan templat Cypher dan batas tetangga yang sama.
  4. Seluruh node diurutkan dengan Disambiguation Ranker yang identik.
  5. Interpretasi = node berperingkat pertama.
Metrik dan uji statistik sama dengan evaluate.py. Pembanding sistem utama diambil
dari reports/utama/final/items.csv (hasil evaluasi akhir, TIDAK dijalankan ulang).

    python baseline_embedding.py
Output: reports/baseline_embedding/items.csv, details.json, summary.txt
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from core.config import DATA_DIR
from core.disambiguation_ranker import DisambiguationRanker, neighbor_text, node_label_text
from core.graph_retriever import GraphRetriever
from core.metrics import (mcnemar_exact, mean, paired_randomization_test, paired_t_test,
                          precision_at_5, reciprocal_rank)

TESTSET = Path("data/testset_utama.json")
FINAL_ITEMS = Path("reports/utama/final/items.csv")
OUT = Path("reports/baseline_embedding")
NODE_CACHE = DATA_DIR / "cache" / "node_name_embeddings.npz"
N_SEEDS = 5
TYPES = ("acronym", "polysemy", "specificity_gap")


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


def load_nodes(retriever):
    rows = retriever._q("""
        MATCH (n:Entity)
        RETURN n.node_index AS node_index, n.name AS name,
               [l IN labels(n) WHERE l <> 'Entity'][0] AS label
        ORDER BY n.node_index
    """)
    return [r for r in rows if not (r["name"] or "").lower().startswith("obsolete")]


def node_matrix(ranker, nodes):
    """Embedding seluruh teks node, disimpan di cache agar tidak dihitung ulang."""
    idx = np.array([n["node_index"] for n in nodes])
    if NODE_CACHE.exists():
        z = np.load(NODE_CACHE)
        if z["idx"].shape == idx.shape and (z["idx"] == idx).all():
            return z["vec"]
    print(f"Membuat embedding {len(nodes)} node (sekali saja, beberapa menit)...")
    model = ranker._model()
    vec = model.encode([node_label_text(n) for n in nodes], normalize_embeddings=True,
                       batch_size=128, show_progress_bar=True).astype(np.float32)
    NODE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(NODE_CACHE, idx=idx, vec=vec)
    return vec


def main():
    if not TESTSET.exists():
        sys.exit(f"{TESTSET} tidak ditemukan.")
    if not FINAL_ITEMS.exists():
        sys.exit(f"{FINAL_ITEMS} tidak ditemukan. Skrip ini membandingkan dengan hasil "
                 "evaluasi akhir yang sudah ada, bukan menjalankan ulang evaluasi.")
    items = json.loads(TESTSET.read_text(encoding="utf-8"))
    final = {r["query_id"]: r for r in csv.DictReader(open(FINAL_ITEMS, encoding="utf-8"))}

    retriever = GraphRetriever()
    # Cache embedding terpisah agar cache sistem utama tidak berubah.
    ranker = DisambiguationRanker(cache_path=OUT / "ranker_cache.pkl")
    nodes = load_nodes(retriever)
    mat = node_matrix(ranker, nodes)

    sys.stdout = Tee(OUT / "summary.txt")
    print(f"BASELINE EMBEDDING (kueri -> nama node) — {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"Node kandidat: {len(nodes)} | node awal per kueri: {N_SEEDS}")

    rows, details = [], []
    for it in items:
        gold = set(it["gold_node_indices"])
        relevant = set(it["relevant_node_indices"])
        t0 = time.time()
        q = ranker.embed([it["query"]])[0]
        sims = mat @ q
        top = np.argsort(-sims, kind="stable")[:N_SEEDS]
        seeds = [{**nodes[i], "seed_similarity": round(float(sims[i]), 4)} for i in top]

        pool = []
        for s in seeds:
            node = {k: s[k] for k in ("node_index", "name", "label")}
            pool.append({**node, "relation": None, "display_relation": None, "via": None,
                         "text": node_label_text(node)})
            for n in retriever.neighbors(s["node_index"]):
                pool.append({"node_index": n["node_index"], "name": n["name"],
                             "label": n["label"], "relation": n["relation"],
                             "display_relation": n["display_relation"], "via": s["name"],
                             "text": neighbor_text(n, s["name"])})
        ranked = ranker.rank_nodes(it["query"], pool) if pool else []
        rank_idx = [n["node_index"] for n in ranked]
        interp = ranked[0] if ranked else None
        f = final[it["query_id"]]
        row = {
            "query_id": it["query_id"], "type": it["ambiguity_type"],
            "emb_da": int(bool(interp and interp["node_index"] in gold)),
            "emb_p5": precision_at_5(rank_idx, relevant),
            "emb_rr": round(reciprocal_rank(rank_idx, relevant), 4),
            "emb_seed1_da": int(seeds[0]["node_index"] in gold) if seeds else 0,
            "emb_gold_in_seeds": int(any(s["node_index"] in gold for s in seeds)),
            "utama_da": int(f["utama_da"]),
            "utama_da_tanpa_simulasi": int(f["utama_da_tanpa_simulasi"]),
            "utama_p5": float(f["utama_p5"]), "utama_rr": float(f["utama_rr"]),
            "base_da": int(f["base_da"]), "base_p5": float(f["base_p5"]),
            "base_rr": float(f["base_rr"]),
            "emb_interpretasi": interp["name"] if interp else "",
            "latency_emb_s": round(time.time() - t0, 2),
        }
        rows.append(row)
        details.append({**row, "query": it["query"], "gold": [g["name"] for g in it["gold"]],
                        "seeds": [f"{s['name']} [{s['label']}] {s['seed_similarity']}"
                                  for s in seeds],
                        "top5": " | ".join(f"{n['name']} [{n['label']}]" for n in ranked[:5])})
        print(f"  {it['query_id']} DA={row['emb_da']} P@5={row['emb_p5']:.1f} "
              f"RR={row['emb_rr']:.2f} | {row['emb_interpretasi']} | seed1: {seeds[0]['name']}")
    ranker.save_cache()

    with open(OUT / "items.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (OUT / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2),
                                      encoding="utf-8")

    def table(sub, title):
        print(f"\n{title} (n={len(sub)})")
        print(f"  {'Sistem':34s} {'DA':>6s} {'P@5':>6s} {'MRR':>6s}")
        print(f"  {'Baseline kata kunci':34s} {mean(r['base_da'] for r in sub):6.3f} "
              f"{mean(r['base_p5'] for r in sub):6.3f} {mean(r['base_rr'] for r in sub):6.3f}")
        print(f"  {'Baseline embedding':34s} {mean(r['emb_da'] for r in sub):6.3f} "
              f"{mean(r['emb_p5'] for r in sub):6.3f} {mean(r['emb_rr'] for r in sub):6.3f}")
        print(f"  {'Sistem utama (dengan simulasi)':34s} {mean(r['utama_da'] for r in sub):6.3f} "
              f"{mean(r['utama_p5'] for r in sub):6.3f} {mean(r['utama_rr'] for r in sub):6.3f}")
        print(f"  {'Sistem utama, DA tanpa simulasi':34s} "
              f"{mean(r['utama_da_tanpa_simulasi'] for r in sub):6.3f}")

    print("\n" + "=" * 64)
    table(rows, "SELURUH KUERI")
    for t in TYPES:
        sub = [r for r in rows if r["type"] == t]
        if sub:
            table(sub, t.upper())

    print("\nMETRIK PENDUKUNG BASELINE EMBEDDING")
    print(f"  DA jika interpretasi = node awal teratas : {mean(r['emb_seed1_da'] for r in rows):.3f}")
    print(f"  Konsep acuan ada di 5 node awal          : {mean(r['emb_gold_in_seeds'] for r in rows):.3f}")
    print(f"  Rata-rata waktu (s)                      : {mean(r['latency_emb_s'] for r in rows):.2f}")

    print("\nUJI SIGNIFIKANSI: sistem utama vs baseline embedding (n = 30, alpha = 0,05)")
    for m in ("p5", "rr"):
        a = [r[f"emb_{m}"] for r in rows]
        b = [r[f"utama_{m}"] for r in rows]
        t = paired_t_test(a, b)
        rnd = paired_randomization_test(a, b)
        print(f"  {m.upper():4s} paired t: t({t['df']}) = {t['t']:.3f}, p = {t['p']:.6f} | "
              f"randomization: selisih = {rnd['observed_diff']:+.3f}, p = {rnd['p']:.6f}")
    for col, label in (("utama_da", "DA dengan simulasi"),
                       ("utama_da_tanpa_simulasi", "DA tanpa simulasi")):
        mc = mcnemar_exact([r["emb_da"] for r in rows], [r[col] for r in rows])
        print(f"  {label:20s} McNemar eksak: b = {mc['b']}, c = {mc['c']}, p = {mc['p']:.6f}")
    print(f"\nFile: {OUT}/items.csv, details.json, summary.txt")


if __name__ == "__main__":
    main()
