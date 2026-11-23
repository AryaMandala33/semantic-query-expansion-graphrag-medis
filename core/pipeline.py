"""
core/pipeline.py — Alur sistem utama (Bab III 3.3, Gambar 3.1)

  kueri -> Query Service -> LLM Expander -> pemetaan kandidat & sub-graf (Retriever)
        -> Disambiguation Ranker memilih interpretasi
           (khusus specificity gap: Clarification Module, pengguna memilih kandidat)
        -> pengurutan node hasil retrieval -> penjelasan

Dua tahap agar klarifikasi dapat dilakukan sebelum penelusuran graf final:
  state  = MainSystem.process(query)
  result = MainSystem.finalize(state, chosen_node_index=None)
"""
import time

from core.disambiguation_ranker import DisambiguationRanker, neighbor_text, node_label_text


class MainSystem:
    def __init__(self, query_service=None, expander=None, retriever=None, ranker=None):
        if query_service is None:
            from core.query_service import QueryService
            query_service = QueryService()
        if expander is None:
            from core.llm_expander import LLMExpander
            expander = LLMExpander()
        if retriever is None:
            from core.graph_retriever import GraphRetriever
            retriever = GraphRetriever()
        self.qs = query_service
        self.expander = expander
        self.retriever = retriever
        self.ranker = ranker or DisambiguationRanker()

    # ------------------------------------------------------------------
    def process(self, query):
        t0 = time.time()
        qs = self.qs.process(query)
        if qs.get("error"):
            return {"query": query, "query_service": qs, "error": qs["error"]}

        expansion = self.expander.expand(query, qs)
        state = {"query": query, "query_service": qs, "expansion": expansion,
                 "mapped": [], "unmapped": []}
        if expansion.get("error"):
            state["error"] = expansion["error"]
            return state

        seen = set()
        for rank_llm, c in enumerate(expansion["candidates"], 1):
            node, status = self.retriever.map_candidate(c["name"], c["entity_type"])
            if node is None:
                state["unmapped"].append({**c, "status": status})
                continue
            if node["node_index"] in seen:          # dua kandidat -> node yang sama
                continue
            seen.add(node["node_index"])
            state["mapped"].append({"candidate": c, "llm_rank": rank_llm, "node": node,
                                    "neighbors": self.retriever.neighbors(node["node_index"])})

        state["scored"] = self.ranker.score_interpretations(query, state["mapped"])
        state["final_type"] = expansion["detected_type"]
        state["needs_clarification"] = (state["final_type"] == "specificity_gap"
                                        and len(state["scored"]) > 0)
        state["clarifying_question"] = expansion.get("clarifying_question")
        state["latency_process_s"] = round(time.time() - t0, 2)
        return state

    # ------------------------------------------------------------------
    def finalize(self, state, chosen_node_index=None):
        """Pilih interpretasi (dari pengguna/simulasi bila diberikan, jika tidak
        dari ranker), lalu urutkan node hasil retrieval."""
        t0 = time.time()
        scored = state.get("scored") or []
        if not scored:
            return {**state, "interpretation": None, "ranked_nodes": [],
                    "selection": "none", "explanation": None}

        if chosen_node_index is not None:
            interp = next((m for m in scored if m["node"]["node_index"] == chosen_node_index), None)
            if interp is None:
                raise ValueError("Node pilihan tidak ada di antara kandidat terpetakan.")
            selection = "clarification"
        else:
            interp = scored[0]
            selection = "ranker"

        concept = interp["node"]
        items = [{"node_index": concept["node_index"], "name": concept["name"],
                  "label": concept["label"], "relation": None, "display_relation": None,
                  "text": node_label_text(concept)}]
        for n in interp["neighbors"]:
            items.append({"node_index": n["node_index"], "name": n["name"], "label": n["label"],
                          "relation": n["relation"], "display_relation": n["display_relation"],
                          "text": neighbor_text(n, concept["name"])})
        ranked = self.ranker.rank_nodes(state["query"], items)
        return {**state, "interpretation": interp, "selection": selection,
                "ranked_nodes": ranked,
                "explanation": self.ranker.explain(interp, ranked),
                "latency_finalize_s": round(time.time() - t0, 2)}

    def run(self, query, chosen_node_index=None):
        return self.finalize(self.process(query), chosen_node_index)
