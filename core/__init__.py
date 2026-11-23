"""
core/ — SISTEM UTAMA, diimplementasikan sesuai rancangan Bab III skripsi.

    query_service.py         3.3.1 Query Service (normalisasi + deteksi tipe berbasis aturan)
    llm_expander.py          3.3.2 LLM Expander
    graph_retriever.py       3.3.4 GraphRAG Retriever (Neo4j + LangChain)
    disambiguation_ranker.py 3.3.5 Disambiguation Ranker (all-mpnet-base-v2 + cosine)
    pipeline.py              Alur sistem utama, termasuk Clarification Module (3.3.3)
    baseline.py              3.5   Sistem baseline tanpa Semantic Query Expansion
    metrics.py               3.6–3.7 Metrik evaluasi dan uji statistik

Komponen tambahan (versi pengembangan) ada di modules/ dan tidak dipanggil oleh
sistem utama.
"""
