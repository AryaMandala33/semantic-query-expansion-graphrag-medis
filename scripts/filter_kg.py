"""
scripts/filter_kg.py — Pemfilteran PrimeKG (Bab III 3.4.2)

Membaca data/kg.csv secara bertahap (chunk) dan menyimpan baris dengan enam
relasi terpilih ke data/kg_subset.csv. Distribusi yang diharapkan (Bab III):
  total 676.286 baris; disease_phenotype_positive 300.634, disease_protein 160.822,
  drug_effect 129.568, contraindication 61.350, indication 18.776, off-label use 5.136.

    python scripts/filter_kg.py
    python scripts/filter_kg.py --input data/kg.csv --output data/kg_subset.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import RELATIONS  # noqa: E402

EXPECTED = {"disease_phenotype_positive": 300634, "disease_protein": 160822,
            "drug_effect": 129568, "contraindication": 61350, "indication": 18776,
            "off-label use": 5136}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/kg.csv")
    ap.add_argument("--output", default="data/kg_subset.csv")
    ap.add_argument("--chunksize", type=int, default=500_000)
    args = ap.parse_args()

    out = Path(args.output)
    if out.exists():
        out.unlink()
    counts, header = {}, True
    for i, chunk in enumerate(pd.read_csv(args.input, chunksize=args.chunksize,
                                          dtype=str, low_memory=False), 1):
        sub = chunk[chunk["relation"].isin(RELATIONS)]
        for rel, n in sub["relation"].value_counts().items():
            counts[rel] = counts.get(rel, 0) + int(n)
        sub.to_csv(out, mode="a", header=header, index=False)
        header = False
        print(f"  chunk {i}: {len(sub):,} baris disimpan")

    total = sum(counts.values())
    print(f"\nTersimpan di {out}: {total:,} baris")
    for rel in RELATIONS:
        got, exp = counts.get(rel, 0), EXPECTED[rel]
        print(f"  {rel:28s} {got:>9,}   (Bab III: {exp:,}) {'OK' if got == exp else 'BERBEDA'}")


if __name__ == "__main__":
    main()
