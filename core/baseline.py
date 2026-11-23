"""
core/baseline.py — Sistem baseline tanpa Semantic Query Expansion (Bab III 3.5)

  1. Normalisasi kueri dan ekstraksi kata kunci (hapus kata henti ID & EN).
  2. Node yang namanya memuat kata kunci diambil (maks. BASELINE_MAX_MATCHED_NODES),
     beserta tetangga 1-hop dengan templat Cypher dan batas tetangga yang SAMA
     dengan sistem utama.
  3. Seluruh node diurutkan dengan Disambiguation Ranker yang identik.
  4. Interpretasi baseline = node berperingkat pertama.
"""
import time

from core.config import BASELINE_MAX_MATCHED_NODES
from core.disambiguation_ranker import DisambiguationRanker, neighbor_text, node_label_text
from core.text_utils import keywords as extract_keywords


class BaselineSystem:
    def __init__(self, retriever=None, ranker=None):
        if retriever is None:
            from core.graph_retriever import GraphRetriever
            retriever = GraphRetriever()
        self.retriever = retriever
        self.ranker = ranker or DisambiguationRanker()

    @staticmethod
    def select_matched(rows, limit=BASELINE_MAX_MATCHED_NODES):
        """Urutan: jumlah kata kunci yang dimuat nama (terbanyak), nama terpendek,
        node_index. Nama 'obsolete ...' dilewati."""
        rows = [r for r in rows if not (r["name"] or "").lower().startswith("obsolete")]
        rows.sort(key=lambda r: (-len(set(r["matched_keywords"])), len(r["name"]),
                                 r["node_index"]))
        return rows[:limit]

    def run(self, query):
        t0 = time.time()
        kws = extract_keywords(query)
        matched = self.select_matched(self.retriever.keyword_nodes(kws))
        items = []
        for m in matched:
            node = {k: m[k] for k in ("node_index", "name", "label")}
            items.append({**node, "relation": None, "display_relation": None,
                          "via": None, "text": node_label_text(node)})
            for n in self.retriever.neighbors(m["node_index"]):
                items.append({"node_index": n["node_index"], "name": n["name"],
                              "label": n["label"], "relation": n["relation"],
                              "display_relation": n["display_relation"], "via": m["name"],
                              "text": neighbor_text(n, m["name"])})
        ranked = self.ranker.rank_nodes(query, items) if items else []
        return {"query": query, "keywords": kws, "matched_nodes": matched,
                "ranked_nodes": ranked,
                "interpretation": ranked[0] if ranked else None,
                "latency_s": round(time.time() - t0, 2)}
