"""
Uji asap koneksi Neo4j + LangChain untuk sistem utama (jalankan setelah impor).
    python -m tests.smoke_neo4j
"""
from core.baseline import BaselineSystem
from core.graph_retriever import GraphRetriever


def main():
    r = GraphRetriever()
    print("1. Pemetaan kandidat (nama, jenis konsep) -> status")
    for name, typ in [("multiple sclerosis", "disease"), ("Multiple Sclerosis", "disease"),
                      ("Alzheimer's disease", "disease"), ("myocardial infarction", "disease"),
                      ("Mitral stenosis", "phenotype"), ("Mitral stenosis", "disease"),
                      ("natalizumab", "drug"), ("tidak ada di graf", "disease")]:
        node, status = r.map_candidate(name, typ)
        print(f"   {name!r:26s} {typ:9s} -> {status:13s} {node['name'] + ' [' + node['label'] + ']' if node else ''}")

    node, _ = r.map_candidate("multiple sclerosis", "disease")
    nb = r.neighbors(node["node_index"])
    by_rel = {}
    for n in nb:
        by_rel[n["relation"]] = by_rel.get(n["relation"], 0) + 1
    print(f"\n2. Tetangga 1-hop 'multiple sclerosis' (maks. 30/relasi): {by_rel}")
    print("   contoh:", [f"{n['name']} ({n['display_relation']})" for n in nb[:5]])
    print(f"\n3. Node relevan via indication: {len(r.relevant_neighbors(node['node_index'], ['indication']))}")
    rows = r.keyword_nodes(["optic", "neuritis"])
    print(f"\n4. Node dengan kata kunci 'optic'/'neuritis': {len(rows)} — contoh "
          f"{[x['name'] for x in BaselineSystem.select_matched(rows)]}")
    print("\nSemua kueri Cypher berjalan.")


if __name__ == "__main__":
    main()
