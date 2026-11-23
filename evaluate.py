"""
evaluate.py — Evaluasi sistem utama vs baseline (Bab III 3.6).

Metrik per kueri (untuk kedua sistem):
  Disambiguation Accuracy (DA) : interpretasi utama termasuk konsep acuan (3.6.3)
  Precision@5                  : proporsi node relevan di 5 teratas (3.6.1)
  Reciprocal rank (MRR)        : 1/peringkat node relevan pertama (3.6.2)

Kueri specificity gap (kategori menurut label acuan): pemilihan kandidat hasil
klarifikasi disimulasikan berdasarkan label acuan (3.6.3). Jika kandidat yang
sesuai konsep acuan ada di keluaran LLM (dan terpetakan), kandidat itu dipilih;
jika tidak, DA dinyatakan salah dan retrieval memakai pilihan ranker.

Metrik pendukung (tidak menggantikan metrik utama):
  DA tanpa simulasi (ranker untuk semua kueri), akurasi deteksi tipe Query Service
  dan LLM, tingkat pemetaan kandidat, candidate recall, waktu proses.

    python evaluate.py                 # set uji BELUM diverifikasi -> hasil praverifikasi
    python evaluate.py --final         # hanya jalan bila seluruh kueri verified = true
    python evaluate.py --limit 3       # uji cepat
Output: reports/utama/<tag>/items.csv, details.json, summary.txt
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

from core.baseline import BaselineSystem
from core.metrics import mean, precision_at_5, reciprocal_rank
from core.pipeline import MainSystem

TYPES = ("acronym", "polysemy", "specificity_gap")
TESTSET = Path("data/testset_utama.json")


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


def simulate_choice(state, gold):
    """Kandidat terpetakan pertama (urutan keluaran LLM) yang termasuk konsep acuan."""
    for m in sorted(state.get("mapped") or [], key=lambda m: m["llm_rank"]):
        if m["node"]["node_index"] in gold:
            return m["node"]["node_index"]
    return None


def brief(nodes, k=5):
    return " | ".join(f"{n['name']} [{n['label']}]" for n in nodes[:k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if not TESTSET.exists():
        sys.exit(f"{TESTSET} belum ada. Jalankan: python build_testset.py")
    items = json.loads(TESTSET.read_text(encoding="utf-8"))
    unverified = [it["query_id"] for it in items if not it.get("verified")]
    review = [it["query_id"] for it in items if it.get("mapping_status") != "OK"]
    if args.final and (unverified or review):
        sys.exit(f"--final ditolak. Belum diverifikasi: {unverified or '-'}; "
                 f"status REVIEW: {review or '-'}")
    tag = "final" if args.final else "praverifikasi"
    if args.limit:
        items = items[: args.limit]
        tag += f"_limit{args.limit}"
    out_dir = Path("reports/utama") / tag

    ms = MainSystem()
    # Pemeriksaan awal: pastikan API LLM dapat diakses sebelum evaluasi dimulai,
    # agar kegagalan teknis (API key salah, koneksi) tidak tercatat sebagai hasil.
    try:
        ms.expander.client.models.list()
    except Exception as e:
        sys.exit(f"LLM tidak dapat diakses ({type(e).__name__}: {e}). "
                 "Periksa LLM_API_KEY di .env. Evaluasi tidak dijalankan.")
    sys.stdout = Tee(out_dir / "summary.txt")

    print(f"EVALUASI SISTEM UTAMA ({tag}) — {time.strftime('%Y-%m-%d %H:%M')}")
    if not args.final:
        print("PERINGATAN: set uji belum seluruhnya diverifikasi; hasil ini hanya untuk persiapan.")

    base = BaselineSystem(retriever=ms.retriever, ranker=ms.ranker)
    rows, details = [], []
    try:
        for it in items:
            gold = set(it["gold_node_indices"])
            relevant = set(it["relevant_node_indices"])
            t0 = time.time()
            state = ms.process(it["query"])
            if str(state.get("error", "")).startswith("Gagal memanggil LLM"):
                # Galat API/koneksi = kegagalan teknis, bukan keluaran sistem.
                # Evaluasi dihentikan tanpa menyimpan hasil.
                sys.stdout = sys.__stdout__
                sys.exit(f"\nEVALUASI DIHENTIKAN pada {it['query_id']}: {state['error']}\n"
                         "Hasil tidak disimpan. Perbaiki masalah teknis, lalu jalankan ulang.")
            if state.get("error"):
                print(f"  ! {it['query_id']}: {state['error']}")
            cand_recall = any(m["node"]["node_index"] in gold for m in state.get("mapped", []))

            # --- sistem utama ---
            ranker_res = ms.finalize(state)
            ranker_da = bool(ranker_res["interpretation"]
                             and ranker_res["interpretation"]["node"]["node_index"] in gold)
            simulated = it["ambiguity_type"] == "specificity_gap"
            if simulated:
                choice = simulate_choice(state, gold)
                main_res = ms.finalize(state, choice) if choice is not None else ranker_res
                main_da = choice is not None
            else:
                main_res, main_da = ranker_res, ranker_da
            latency = round(time.time() - t0, 2)
            main_rank = [n["node_index"] for n in main_res["ranked_nodes"]]

            # --- baseline ---
            b = base.run(it["query"])
            base_rank = [n["node_index"] for n in b["ranked_nodes"]]
            base_da = bool(b["interpretation"] and b["interpretation"]["node_index"] in gold)

            exp = state.get("expansion") or {}
            row = {
                "query_id": it["query_id"], "type": it["ambiguity_type"],
                "seen_in_development": it.get("seen_in_development", False),
                "qs_type": state.get("query_service", {}).get("ambiguity_type"),
                "llm_type": exp.get("detected_type"),
                "n_candidates": len(exp.get("candidates", [])),
                "n_mapped": len(state.get("mapped", [])),
                "cand_recall": int(cand_recall), "simulated": int(simulated),
                "utama_da": int(main_da), "utama_da_tanpa_simulasi": int(ranker_da),
                "utama_p5": precision_at_5(main_rank, relevant),
                "utama_rr": round(reciprocal_rank(main_rank, relevant), 4),
                "base_da": int(base_da),
                "base_p5": precision_at_5(base_rank, relevant),
                "base_rr": round(reciprocal_rank(base_rank, relevant), 4),
                "utama_interpretasi": (main_res["interpretation"]["node"]["name"]
                                       if main_res.get("interpretation") else ""),
                "base_interpretasi": b["interpretation"]["name"] if b["interpretation"] else "",
                "latency_utama_s": latency, "latency_base_s": b["latency_s"],
                "llm_cached": int(bool((exp.get("meta") or {}).get("cached"))),
            }
            rows.append(row)
            details.append({
                **row, "query": it["query"], "gold": [g["name"] for g in it["gold"]],
                "candidates": exp.get("candidates", []),
                "unmapped": state.get("unmapped", []),
                "scored": [{"candidate": m["candidate"]["name"], "node": m["node"]["name"],
                            "label": m["node"]["label"], "similarity": m["similarity"],
                            "subgraph_text": m["subgraph_text"]} for m in state.get("scored", [])],
                "utama_top5": brief(main_res["ranked_nodes"]),
                "base_keywords": b["keywords"],
                "base_matched": [m["name"] for m in b["matched_nodes"]],
                "base_top5": brief(b["ranked_nodes"]),
            })
            print(f"  {it['query_id']} DA utama={row['utama_da']} base={row['base_da']} | "
                  f"P@5 {row['utama_p5']:.1f}/{row['base_p5']:.1f} | "
                  f"RR {row['utama_rr']:.2f}/{row['base_rr']:.2f} | "
                  f"{row['utama_interpretasi'] or '-'}")
    finally:
        ms.ranker.save_cache()

    with open(out_dir / "items.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    def table(subset, title):
        print(f"\n{title} (n={len(subset)})")
        print(f"  {'Sistem':22s} {'DA':>6s} {'P@5':>6s} {'MRR':>6s}")
        print(f"  {'Baseline':22s} {mean(r['base_da'] for r in subset):6.3f} "
              f"{mean(r['base_p5'] for r in subset):6.3f} {mean(r['base_rr'] for r in subset):6.3f}")
        print(f"  {'Sistem utama':22s} {mean(r['utama_da'] for r in subset):6.3f} "
              f"{mean(r['utama_p5'] for r in subset):6.3f} {mean(r['utama_rr'] for r in subset):6.3f}")

    print("\n" + "=" * 60)
    table(rows, "SELURUH KUERI")
    for t in TYPES:
        sub = [r for r in rows if r["type"] == t]
        if sub:
            table(sub, t.upper())

    print("\nMETRIK PENDUKUNG")
    print(f"  DA sistem utama tanpa simulasi klarifikasi : {mean(r['utama_da_tanpa_simulasi'] for r in rows):.3f}")
    print(f"  Akurasi deteksi tipe Query Service (aturan) : {mean(r['qs_type'] == r['type'] for r in rows):.3f}")
    print(f"  Akurasi deteksi tipe LLM                    : {mean(r['llm_type'] == r['type'] for r in rows):.3f}")
    n_c = sum(r["n_candidates"] for r in rows)
    print(f"  Kandidat terpetakan ke graf                 : {sum(r['n_mapped'] for r in rows)}/{n_c}")
    print(f"  Candidate recall (acuan ada di kandidat)    : {mean(r['cand_recall'] for r in rows):.3f}")
    print(f"  Rata-rata waktu sistem utama / baseline (s) : "
          f"{mean(r['latency_utama_s'] for r in rows):.2f} / {mean(r['latency_base_s'] for r in rows):.2f}"
          f"  (respons LLM dari cache: {sum(r['llm_cached'] for r in rows)} kueri)")
    print(f"\nFile: {out_dir}/items.csv, details.json, summary.txt")
    print(f"Analisis statistik: python analyze.py --dir {out_dir}")


if __name__ == "__main__":
    main()
