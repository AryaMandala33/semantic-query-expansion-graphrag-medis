"""
analyze.py — Analisis hasil evaluasi (Bab III 3.6.5 dan 3.7).

  3.7.1 Analisis komparatif   : nilai tiap metrik baseline vs sistem utama + selisih
  3.7.2 Per kategori          : deskriptif (tabel + grafik batang)
  3.7.4 Uji signifikansi      : paired t-test (P@5, RR), uji randomization berpasangan,
                                McNemar exact (DA); seluruh kueri, p < 0,05
  Tabel 3.3                   : perbandingan dengan target keberhasilan
  (3.7.3 User Satisfaction dianalisis terpisah: python analyze_survey.py)

    python analyze.py --dir reports/utama/praverifikasi
    python analyze.py --dir reports/utama/final
Output di folder yang sama: analysis.txt, tabel_bab4.csv, grafik_per_kategori.png
"""
import argparse
import csv
import sys
from pathlib import Path

from core.metrics import (mcnemar_exact, mean, paired_randomization_test, paired_t_test)

TYPES = [("acronym", "Akronim"), ("polysemy", "Polisemi"), ("specificity_gap", "Specificity gap")]
METRICS = [("p5", "Precision@5", 0.70), ("rr", "MRR", 0.65), ("da", "Disambiguation Accuracy", 0.75)]
ALPHA = 0.05


class Tee:
    def __init__(self, p):
        self.f, self.o = open(p, "w", encoding="utf-8"), sys.stdout

    def write(self, s):
        self.o.write(s)
        self.f.write(s)

    def flush(self):
        self.o.flush()
        self.f.flush()


def load(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("utama_da", "base_da", "utama_da_tanpa_simulasi"):
            r[k] = int(r[k])
        for k in ("utama_p5", "base_p5", "utama_rr", "base_rr"):
            r[k] = float(r[k])
    return rows


def sig(p):
    return "signifikan" if p < ALPHA else "tidak signifikan"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="reports/utama/praverifikasi")
    args = ap.parse_args()
    d = Path(args.dir)
    rows = load(d / "items.csv")
    sys.stdout = Tee(d / "analysis.txt")
    n = len(rows)

    print("=" * 74)
    print(f"1. ANALISIS KOMPARATIF (n = {n} kueri)")
    print("=" * 74)
    print(f"  {'Metrik':26s} {'Baseline':>9s} {'Utama':>9s} {'Selisih':>9s}")
    table = []
    for key, label, target in METRICS:
        b, u = mean(r[f"base_{key}"] for r in rows), mean(r[f"utama_{key}"] for r in rows)
        print(f"  {label:26s} {b:9.3f} {u:9.3f} {u - b:+9.3f}")
        table.append({"kategori": "Seluruh kueri", "metrik": label, "baseline": round(b, 3),
                      "utama": round(u, 3), "selisih": round(u - b, 3)})

    print("\n" + "=" * 74)
    print("2. PER KATEGORI AMBIGUITAS (deskriptif)")
    print("=" * 74)
    per_cat = {}
    for t, tl in TYPES:
        sub = [r for r in rows if r["type"] == t]
        if not sub:
            continue
        print(f"\n  {tl} (n = {len(sub)})")
        for key, label, _ in METRICS:
            b, u = mean(r[f"base_{key}"] for r in sub), mean(r[f"utama_{key}"] for r in sub)
            per_cat[(tl, label)] = (b, u)
            print(f"    {label:26s} {b:7.3f} -> {u:7.3f}  ({u - b:+.3f})")
            table.append({"kategori": tl, "metrik": label, "baseline": round(b, 3),
                          "utama": round(u, 3), "selisih": round(u - b, 3)})

    print("\n" + "=" * 74)
    print(f"3. UJI SIGNIFIKANSI (seluruh kueri, alpha = {ALPHA})")
    print("=" * 74)
    for key, label in (("p5", "Precision@5"), ("rr", "Reciprocal rank")):
        a = [r[f"base_{key}"] for r in rows]
        b = [r[f"utama_{key}"] for r in rows]
        t = paired_t_test(a, b)
        rnd = paired_randomization_test(a, b)
        print(f"  {label}:")
        print(f"    paired t-test       t({t['df']}) = {t['t']:.3f}, p = {t['p']:.4f}  ({sig(t['p'])})")
        print(f"    uji randomization   selisih = {rnd['observed_diff']:+.3f}, p = {rnd['p']:.4f}  "
              f"({sig(rnd['p'])}; {rnd['iterations']:,} permutasi)")
    m = mcnemar_exact([bool(r["base_da"]) for r in rows], [bool(r["utama_da"]) for r in rows])
    print("  Disambiguation Accuracy:")
    print(f"    McNemar exact       b = {m['b']} (benar di baseline saja), "
          f"c = {m['c']} (benar di sistem utama saja), p = {m['p']:.4f}  ({sig(m['p'])})")

    print("\n" + "=" * 74)
    print("4. TARGET KEBERHASILAN (Tabel 3.3) — sistem utama")
    print("=" * 74)
    for key, label, target in METRICS:
        u = mean(r[f"utama_{key}"] for r in rows)
        print(f"  {label:26s} {u:6.3f}  target >= {target:.2f}  "
              f"{'TERCAPAI' if u >= target else 'TIDAK TERCAPAI'}")
    print("  User Satisfaction Score    lihat analyze_survey.py (target >= 4,0)")

    print("\n" + "=" * 74)
    print("5. CATATAN PENDUKUNG")
    print("=" * 74)
    sim = [r for r in rows if r["type"] == "specificity_gap"]
    if sim:
        print(f"  DA specificity gap dengan simulasi klarifikasi : {mean(r['utama_da'] for r in sim):.3f}")
        print(f"  DA specificity gap tanpa simulasi (ranker)     : "
              f"{mean(r['utama_da_tanpa_simulasi'] for r in sim):.3f}")
        print("  Hasil dengan simulasi merupakan estimasi optimistis (Bab III 3.6.3).")
    seen = [r for r in rows if r.get("seen_in_development") in ("True", "1", "true")]
    unseen = [r for r in rows if r not in seen]
    if seen:
        print(f"  DA sistem utama — kueri yang terlihat saat pengembangan (n={len(seen)}): "
              f"{mean(r['utama_da'] for r in seen):.3f}; lainnya (n={len(unseen)}): "
              f"{mean(r['utama_da'] for r in unseen):.3f}")

    with open(d / "tabel_bab4.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cats = [tl for _, tl in TYPES]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
        for ax, (key, label, target) in zip(axes, METRICS):
            bvals = [per_cat.get((c, label), (0, 0))[0] for c in cats]
            uvals = [per_cat.get((c, label), (0, 0))[1] for c in cats]
            x = range(len(cats))
            ax.bar([i - 0.2 for i in x], bvals, width=0.4, label="Baseline", color="#9aa5b1")
            ax.bar([i + 0.2 for i in x], uvals, width=0.4, label="Sistem yang diusulkan", color="#1f4e79")
            ax.axhline(target, color="#b03a2e", linestyle="--", linewidth=1, label="Target")
            ax.set_xticks(list(x), cats, fontsize=9)
            ax.set_title(label, fontsize=10)
            ax.set_ylim(0, 1)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(d / "grafik_per_kategori.png", dpi=200)
        print(f"\nGrafik : {d / 'grafik_per_kategori.png'}")
    except Exception as e:
        print(f"\n(Grafik tidak dibuat: {e})")
    print(f"Tabel  : {d / 'tabel_bab4.csv'}")


if __name__ == "__main__":
    main()
