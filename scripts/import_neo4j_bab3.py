"""
scripts/import_neo4j_bab3.py — Impor kg_subset.csv ke Neo4j (Bab III 3.4.3–3.4.4)

- Label node: Entity + Disease | Drug | Phenotype | Protein.
- Relasi bertipe RELATION; jenis relasi (relation) dan nama relasi
  (display_relation) disimpan sebagai properti.
- Impor bertahap (batch) dengan MERGE agar node dan relasi tidak terduplikasi.
  PrimeKG menyimpan setiap hubungan dua kali (x->y dan y->x); hanya arah kanonik
  yang diimpor (obat -> kondisi, penyakit -> fenotipe/protein).
- Indeks pada properti id dan name (serta name_lower untuk pencocokan tanpa
  membedakan huruf besar/kecil); node_index dijadikan kunci unik karena id
  PrimeKG tidak unik lintas sumber.

    python scripts/import_neo4j_bab3.py                # database: NEO4J_DATABASE_MAIN
    python scripts/import_neo4j_bab3.py --reset        # kosongkan database tujuan dulu
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import (NEO4J_DATABASE_MAIN, NEO4J_PASSWORD, NEO4J_URI,  # noqa: E402
                         NEO4J_USER, RELATIONS, TYPE_TO_LABEL)

PROTECTED = {"database-clean", "primekgv2", "neo4j", "system"}
# Tipe sumber kanonik untuk tiap relasi (arah yang disimpan).
CANONICAL_SOURCE = {
    "indication": "drug", "contraindication": "drug", "off-label use": "drug",
    "drug_effect": "drug", "disease_phenotype_positive": "disease",
    "disease_protein": "disease",
}
BATCH = 5000


def run(session, query, **params):
    return list(session.run(query, **params))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/kg_subset.csv")
    ap.add_argument("--database", default=NEO4J_DATABASE_MAIN)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    if args.database in PROTECTED:
        sys.exit(f"Menolak menulis ke database '{args.database}'. Pakai database baru.")
    if not NEO4J_PASSWORD:
        sys.exit("NEO4J_PASSWORD tidak ditemukan di .env")

    df = pd.read_csv(args.input, dtype=str)
    df = df[df["relation"].isin(RELATIONS)]
    print(f"Baris kg_subset.csv            : {len(df):,}")
    canon = df[df["x_type"] == df["relation"].map(CANONICAL_SOURCE)].copy()
    canon = canon.drop_duplicates(subset=["x_index", "y_index", "relation"])
    print(f"Relasi unik (arah kanonik)     : {len(canon):,}")

    nodes = pd.concat([
        canon[["x_index", "x_id", "x_type", "x_name", "x_source"]].set_axis(
            ["node_index", "id", "type", "name", "source"], axis=1),
        canon[["y_index", "y_id", "y_type", "y_name", "y_source"]].set_axis(
            ["node_index", "id", "type", "name", "source"], axis=1),
    ]).drop_duplicates("node_index")
    nodes["node_index"] = nodes["node_index"].astype(int)
    nodes["name_lower"] = nodes["name"].str.lower().str.strip()
    nodes["label"] = nodes["type"].map(TYPE_TO_LABEL)
    print(f"Node unik                      : {len(nodes):,}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session(database="system") as s:
            try:
                run(s, f"CREATE DATABASE `{args.database}` IF NOT EXISTS WAIT")
            except Exception as e:
                print(f"(Tidak dapat membuat database otomatis: {e}. "
                      f"Buat database '{args.database}' lewat Neo4j Desktop, lalu ulangi.)")

        with driver.session(database=args.database) as s:
            if args.reset:
                print("Mengosongkan database tujuan ...")
                run(s, "MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS")
            run(s, "CREATE CONSTRAINT entity_node_index IF NOT EXISTS "
                   "FOR (n:Entity) REQUIRE n.node_index IS UNIQUE")
            run(s, "CREATE INDEX entity_id IF NOT EXISTS FOR (n:Entity) ON (n.id)")
            run(s, "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)")
            run(s, "CREATE INDEX entity_name_lower IF NOT EXISTS FOR (n:Entity) ON (n.name_lower)")
            run(s, "CALL db.awaitIndexes(300)")

            for label, group in nodes.groupby("label"):
                rows = group[["node_index", "id", "type", "name", "name_lower", "source"]]\
                    .to_dict("records")
                for i in range(0, len(rows), BATCH):
                    run(s, f"""
                        UNWIND $rows AS row
                        MERGE (n:Entity {{node_index: row.node_index}})
                        SET n:{label}, n.id = row.id, n.type = row.type, n.name = row.name,
                            n.name_lower = row.name_lower, n.source = row.source
                    """, rows=rows[i:i + BATCH])
                print(f"  node {label:10s}: {len(rows):,}")

            canon["x_index"] = canon["x_index"].astype(int)
            canon["y_index"] = canon["y_index"].astype(int)
            rels = canon[["x_index", "y_index", "relation", "display_relation"]].to_dict("records")
            for i in range(0, len(rels), BATCH):
                run(s, """
                    UNWIND $rows AS row
                    MATCH (a:Entity {node_index: row.x_index})
                    MATCH (b:Entity {node_index: row.y_index})
                    MERGE (a)-[r:RELATION {relation: row.relation}]->(b)
                    SET r.display_relation = row.display_relation
                """, rows=rels[i:i + BATCH])
                if (i // BATCH) % 10 == 0:
                    print(f"  relasi {i + len(rels[i:i + BATCH]):,}/{len(rels):,}")

            # ---- Verifikasi (3.4.4) ----
            print("\nVERIFIKASI")
            for r in run(s, "MATCH (n:Entity) RETURN [l IN labels(n) WHERE l <> 'Entity'][0] "
                            "AS label, count(*) AS n ORDER BY n DESC"):
                print(f"  {r['label']:10s} {r['n']:>8,}")
            for r in run(s, "MATCH ()-[r:RELATION]->() RETURN r.relation AS rel, count(*) AS n "
                            "ORDER BY n DESC"):
                print(f"  {r['rel']:28s} {r['n']:>8,}")
            dup = run(s, """MATCH (a)-[r:RELATION]->(b)
                            WITH a, b, r.relation AS rel, count(*) AS c WHERE c > 1
                            RETURN count(*) AS n""")[0]["n"]
            print(f"  relasi duplikat: {dup}")
            ex = run(s, """MATCH (d:Disease {name_lower: 'multiple sclerosis'})-[r:RELATION]-(m)
                           RETURN r.relation AS rel, count(m) AS n ORDER BY n DESC""")
            print("  contoh 'multiple sclerosis':", {r["rel"]: r["n"] for r in ex})
    finally:
        driver.close()


if __name__ == "__main__":
    main()
