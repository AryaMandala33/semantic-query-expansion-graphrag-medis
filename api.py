"""
api.py — Antarmuka FastAPI untuk komponen sistem utama (Bab III 3.3.1).

Setiap komponen dapat dipanggil dan diuji secara independen:
    POST /query-service     normalisasi + deteksi tipe ambiguitas (berbasis aturan)
    POST /expand            Query Service + LLM Expander
    POST /process           alur sistem utama sampai pemetaan & skor kandidat
    POST /disambiguate      alur lengkap (opsional: chosen_node_index hasil klarifikasi)
    POST /baseline          sistem baseline
    GET  /health

Jalankan dari folder Project:
    uvicorn api:app --reload
Dokumentasi interaktif: http://127.0.0.1:8000/docs
"""
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.query_service import QueryService

app = FastAPI(title="Semantic Query Expansion — Sistem Utama",
              description="Disambiguasi leksikal istilah medis pada GraphRAG (PrimeKG).")


class QueryIn(BaseModel):
    query: str


class DisambiguateIn(BaseModel):
    query: str
    chosen_node_index: Optional[int] = None


@lru_cache(maxsize=1)
def query_service():
    return QueryService()


@lru_cache(maxsize=1)
def main_system():
    from core.pipeline import MainSystem
    return MainSystem(query_service=query_service())


@lru_cache(maxsize=1)
def baseline_system():
    from core.baseline import BaselineSystem
    ms = main_system()
    return BaselineSystem(retriever=ms.retriever, ranker=ms.ranker)


def _public(state):
    """Buang daftar tetangga mentah agar respons ringkas."""
    out = {k: v for k, v in state.items() if k not in ("mapped",)}
    if out.get("scored"):
        out["scored"] = [{k: v for k, v in m.items() if k != "neighbors"} for m in out["scored"]]
    if out.get("interpretation"):
        out["interpretation"] = {k: v for k, v in out["interpretation"].items() if k != "neighbors"}
    if out.get("ranked_nodes"):
        out["ranked_nodes"] = out["ranked_nodes"][:20]
    return out


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query-service")
def run_query_service(body: QueryIn):
    result = query_service().process(body.query)
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/expand")
def run_expand(body: QueryIn):
    qs = run_query_service(body)
    return {"query_service": qs, "expansion": main_system().expander.expand(body.query, qs)}


@app.post("/process")
def run_process(body: QueryIn):
    return _public(main_system().process(body.query))


@app.post("/disambiguate")
def run_disambiguate(body: DisambiguateIn):
    ms = main_system()
    state = ms.process(body.query)
    if state.get("error"):
        raise HTTPException(status_code=502, detail=state["error"])
    try:
        return _public(ms.finalize(state, body.chosen_node_index))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/baseline")
def run_baseline(body: QueryIn):
    res = baseline_system().run(body.query)
    res["ranked_nodes"] = res["ranked_nodes"][:20]
    return res
