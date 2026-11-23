"""
core/graph_retriever.py — GraphRAG Retriever (Bab III 3.3.4)

- Basis data graf Neo4j, diakses melalui LangChain (langchain-neo4j, Neo4jGraph).
- Setiap kandidat dipetakan ke node PrimeKG lewat pencocokan nama konsep tanpa
  membedakan huruf besar/kecil, dibatasi pada label node sesuai jenis konsep.
- Kueri Cypher berparameter mengambil tetangga 1-hop melalui enam relasi subset.
  Jumlah tetangga dibatasi sama untuk sistem utama dan baseline.
- Kandidat yang tidak ditemukan dicatat sebagai tidak terpetakan.

Skema database (scripts/import_neo4j_bab3.py):
  (:Entity:Disease|Drug|Phenotype|Protein {node_index, id, name, name_lower, type, source})
  (a)-[:RELATION {relation, display_relation}]->(b)
"""
import re

from core.config import (CONCEPT_TYPE_TO_LABEL, NEIGHBORS_PER_RELATION, NEO4J_DATABASE_MAIN,
                         NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, RELATIONS)

_NODE_FIELDS = """n.node_index AS node_index, n.name AS name, n.id AS id, n.source AS source,
                  [l IN labels(n) WHERE l <> 'Entity'][0] AS label"""


def name_variants(name):
    """Varian nama untuk pencocokan: huruf kecil, tanpa bentuk posesif
    ("Alzheimer's disease" -> "alzheimer disease", konvensi MONDO/HPO), dan
    varian berakhiran " (disease)" yang dipakai MONDO untuk membedakan penyakit
    dari fenotipe bernama sama."""
    base = " ".join(re.sub(r"['’]s\b", "", name or "").lower().split())
    if not base:
        return []
    out = [base]
    if not base.endswith(" (disease)"):
        out.append(base + " (disease)")
    return out


class GraphRetriever:
    def __init__(self, graph=None, database=None):
        if graph is not None:              # disuntikkan (mis. uji)
            self.graph = graph
            return
        from langchain_neo4j import Neo4jGraph
        if not NEO4J_PASSWORD:
            raise ValueError("NEO4J_PASSWORD tidak ditemukan di .env")
        self.graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD,
                                database=database or NEO4J_DATABASE_MAIN,
                                refresh_schema=False)

    def _q(self, cypher, **params):
        return self.graph.query(cypher, params=params)

    # ------------------------------------------------------------------
    # Pemetaan kandidat ke node
    # ------------------------------------------------------------------
    def map_candidate(self, name, concept_type):
        """Kembalikan (node, status). status: 'mapped' | 'type_mismatch' | 'not_found'.
        'type_mismatch' = nama ada di graf tetapi dengan label lain (tetap tidak
        terpetakan, dicatat untuk analisis)."""
        variants = name_variants(name)
        label = CONCEPT_TYPE_TO_LABEL.get((concept_type or "").lower())
        if not variants or not label:
            return None, "not_found"
        rows = self._q(f"""
            MATCH (n:Entity) WHERE n.name_lower IN $variants
            RETURN {_NODE_FIELDS}, n.name_lower = $primary AS primary_match
        """, variants=variants, primary=variants[0])
        same = [r for r in rows if r["label"] == label]
        if same:
            # Nama persis didahulukan daripada varian "(disease)"; lalu node_index terkecil.
            same.sort(key=lambda r: (not r["primary_match"], r["node_index"]))
            node = {k: same[0][k] for k in ("node_index", "name", "id", "source", "label")}
            return node, "mapped"
        if rows:
            return None, "type_mismatch"
        return None, "not_found"

    # ------------------------------------------------------------------
    # Tetangga 1-hop
    # ------------------------------------------------------------------
    def neighbors(self, node_index, per_relation=NEIGHBORS_PER_RELATION):
        """Tetangga 1-hop per relasi, masing-masing paling banyak `per_relation`.
        Urutan dalam satu relasi: degree terkecil dulu (paling spesifik), lalu nama."""
        rows = self._q("""
            MATCH (c:Entity {node_index: $idx})-[r:RELATION]-(n:Entity)
            WHERE r.relation IN $relations
            WITH r, n, COUNT { (n)--() } AS degree
            ORDER BY r.relation, degree ASC, n.name ASC
            WITH r.relation AS relation, r.display_relation AS display_relation,
                 collect({node_index: n.node_index, name: n.name, id: n.id, source: n.source,
                          label: [l IN labels(n) WHERE l <> 'Entity'][0],
                          degree: degree}) AS nodes
            RETURN relation, display_relation, nodes[..$k] AS nodes
        """, idx=node_index, relations=RELATIONS, k=per_relation)
        out = []
        for row in rows:
            for n in row["nodes"]:
                out.append({**n, "relation": row["relation"],
                            "display_relation": row["display_relation"] or row["relation"]})
        return out

    def relevant_neighbors(self, node_index, relations):
        """Seluruh tetangga 1-hop melalui relasi relevan (tanpa batas), untuk
        himpunan node relevan pada label acuan (3.4.5, 3.6.1)."""
        rows = self._q("""
            MATCH (c:Entity {node_index: $idx})-[r:RELATION]-(n:Entity)
            WHERE r.relation IN $relations
            RETURN collect(DISTINCT n.node_index) AS idx
        """, idx=node_index, relations=list(relations))
        return rows[0]["idx"] if rows else []

    def node_by_index(self, node_index):
        rows = self._q(f"MATCH (n:Entity {{node_index: $idx}}) RETURN {_NODE_FIELDS}",
                       idx=node_index)
        return rows[0] if rows else None

    def find_by_name(self, name):
        """Semua node dengan nama (case-insensitive, termasuk varian) — untuk build_testset."""
        return self._q(f"""
            MATCH (n:Entity) WHERE n.name_lower IN $variants
            RETURN {_NODE_FIELDS}, n.name_lower = $primary AS primary_match
            ORDER BY primary_match DESC, n.node_index
        """, variants=name_variants(name), primary=(name_variants(name) or [""])[0])

    # ------------------------------------------------------------------
    # Baseline (3.5.2): node yang namanya mengandung kata kunci
    # ------------------------------------------------------------------
    def keyword_nodes(self, keywords):
        """Node yang namanya memuat kata kunci sebagai kata utuh."""
        if not keywords:
            return []
        patterns = [f".*\\b{re.escape(k)}\\b.*" for k in keywords]
        rows = self._q(f"""
            UNWIND range(0, size($patterns) - 1) AS i
            MATCH (n:Entity) WHERE n.name_lower =~ $patterns[i]
            RETURN {_NODE_FIELDS}, collect($keywords[i]) AS matched_keywords
        """, patterns=patterns, keywords=list(keywords))
        return rows
