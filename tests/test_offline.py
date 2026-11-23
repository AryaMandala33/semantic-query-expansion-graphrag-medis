"""
Uji offline sistem utama (tanpa Neo4j/LLM/model embedding).
    python -m tests.test_offline
"""
import json
import tempfile
from pathlib import Path

import core.llm_expander as le
from core.baseline import BaselineSystem
from core.disambiguation_ranker import DisambiguationRanker, subgraph_text
from core.llm_expander import LLMExpander
from core.metrics import (mcnemar_exact, paired_randomization_test, paired_t_test,
                          precision_at_5, reciprocal_rank)
from core.pipeline import MainSystem
from core.query_service import QueryService
from core.text_utils import keywords, normalize
from tests.fakes import FakeEncoder, FakeLLMClient, FakeRetriever

PASSED = []


def check(cond, msg):
    assert cond, msg
    PASSED.append(msg)


def llm_output(term, typ, cands, question=None):
    return {"ambiguous_term": term, "detected_type": typ, "type_reason": "x",
            "senses_considered": [c[0] for c in cands],
            "candidates": [{"name": n, "entity_type": t, "confidence": c, "reason": "r"}
                           for n, t, c in cands],
            "clarifying_question": question, "uncertainty": None}


def make_system(outputs, tmp):
    le.LLM_CACHE_DIR = Path(tmp) / "llm"
    exp = LLMExpander(client=FakeLLMClient(outputs))
    ranker = DisambiguationRanker(encoder=FakeEncoder(), cache_path=None)
    return MainSystem(query_service=QueryService(), expander=exp, retriever=FakeRetriever(),
                      ranker=ranker)


def main():
    # --- Query Service --------------------------------------------------
    qs = QueryService()
    check(normalize("  Obat untuk MS?!  ") == "obat untuk ms", "normalisasi")
    r = qs.process("Obat untuk MS pada pasien")
    check(r["ambiguity_type"] == "acronym" and r["trigger_term"] == "MS", "aturan akronim")
    check(qs.process("obat untuk ms")["ambiguity_type"] != "acronym",
          "akronim harus huruf kapital pada kueri asli")
    check(qs.process("depression setelah morfin")["ambiguity_type"] == "polysemy", "aturan polisemi")
    check(qs.process("sakit jantung")["ambiguity_type"] == "specificity_gap", "aturan specificity gap")
    check(qs.process("nyeri lutut setelah jatuh")["ambiguity_type"] == "none", "tidak terdeteksi")
    check(keywords("Which drugs are used for MS in a 60-year-old woman?") ==
          ["drugs", "used", "ms", "year", "old", "woman"], "ekstraksi kata kunci baseline")

    with tempfile.TemporaryDirectory() as tmp:
        # --- LLM Expander: ulang sekali bila JSON tidak valid ---------------
        le.LLM_CACHE_DIR = Path(tmp) / "llm"
        good = llm_output("MS", "acronym", [("multiple sclerosis", "disease", 0.9)])
        client = FakeLLMClient(["{not json", good])
        out = LLMExpander(client=client).expand("MS dengan neuritis optik", qs.process("MS"))
        check(client.calls == 2 and not out.get("error"), "retry satu kali saat JSON tidak valid")
        client = FakeLLMClient(["{bad", "{bad again"])
        out = LLMExpander(client=client, use_cache=False).expand("tes lain")
        check(out.get("error") and client.calls == 2, "berhenti setelah 2 percobaan")
        bad_type = llm_output("x", "polysemy", [("a", "gene", 0.5)])
        client = FakeLLMClient([bad_type, good])
        out = LLMExpander(client=client, use_cache=False).expand("tes tipe")
        check(client.calls == 2 and not out.get("error"), "entity_type di luar skema -> retry")
        many = llm_output("x", "polysemy", [(f"c{i}", "disease", 0.1) for i in range(8)])
        out = LLMExpander(client=FakeLLMClient([many]), use_cache=False).expand("tes maks")
        check(len(out["candidates"]) == 5, "maksimal lima kandidat")

        # --- Pipeline: akronim dipilih ranker ------------------------------
        ms = make_system([llm_output("MS", "acronym", [
            ("multiple sclerosis", "disease", 0.9), ("Mitral stenosis", "phenotype", 0.1),
            ("metabolic syndrome", "disease", 0.05)])], tmp)
        q = "Which drugs are used for MS with optic neuritis and paresthesia"
        state = ms.process(q)
        check(len(state["mapped"]) == 2 and state["unmapped"][0]["status"] == "not_found",
              "pemetaan kandidat + kandidat tidak terpetakan dicatat")
        check(not state["needs_clarification"], "akronim tidak butuh klarifikasi")
        res = ms.finalize(state)
        check(res["interpretation"]["node"]["name"] == "multiple sclerosis", "ranker memilih MS")
        check(res["ranked_nodes"][0]["rank"] == 1 and res["selection"] == "ranker", "node diurutkan")
        check(subgraph_text(res["interpretation"]["node"], res["interpretation"]["neighbors"])
              .startswith("multiple sclerosis (disease), phenotype present:"), "templat teks sub-graf")

        # --- Pipeline: tipe node berbeda -> tidak terpetakan ------------------
        ms2 = make_system([llm_output("MS", "acronym", [("Mitral stenosis", "disease", 0.9)])], tmp)
        st2 = ms2.process("MS dengan opening snap")
        check(st2["unmapped"][0]["status"] == "type_mismatch", "tipe berbeda dicatat type_mismatch")

        # --- Specificity gap: klarifikasi ------------------------------------
        ms3 = make_system([llm_output("heart disease", "specificity_gap", [
            ("coronary artery disease", "disease", 0.6), ("heart failure", "disease", 0.4)],
            "Penyakit jantung yang mana?")], tmp)
        st3 = ms3.process("heart disease with ankle swelling and orthopnea")
        check(st3["needs_clarification"], "specificity gap -> klarifikasi")
        chosen = ms3.finalize(st3, 6)
        check(chosen["interpretation"]["node"]["name"] == "heart failure"
              and chosen["selection"] == "clarification", "pilihan pengguna dipakai")

        # --- Baseline ----------------------------------------------------------
        base = BaselineSystem(retriever=FakeRetriever(), ranker=ms.ranker)
        b = base.run("heart failure with edema")
        check(b["interpretation"] is not None and b["keywords"] == ["heart", "failure", "edema"],
              "baseline berjalan")

    # --- Metrik ------------------------------------------------------------
    check(precision_at_5([1, 2, 3], {1, 3}) == 0.4, "P@5 penyebut tetap 5")
    check(reciprocal_rank([5, 6, 1], {1}) == 1 / 3 and reciprocal_rank([5], {1}) == 0,
          "reciprocal rank")
    check(mcnemar_exact([1, 1, 0, 0], [1, 1, 1, 1])["p"] == 0.5, "McNemar exact")
    t = paired_t_test([0.2, 0.4, 0.2, 0.0], [0.6, 0.8, 0.4, 0.6])
    check(t["p"] < 0.05, "paired t-test")
    rnd = paired_randomization_test([0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], iterations=5000)
    check(rnd["p"] < 0.05, "uji randomization")

    # --- FastAPI -------------------------------------------------------------
    from fastapi.testclient import TestClient
    from api import app
    c = TestClient(app)
    resp = c.post("/query-service", json={"query": "obat untuk MS"})
    check(resp.status_code == 200 and resp.json()["ambiguity_type"] == "acronym", "endpoint /query-service")
    check(c.post("/query-service", json={"query": "  "}).status_code == 400, "kueri kosong ditolak")

    print(json.dumps(PASSED, ensure_ascii=False, indent=1))
    print(f"\nSEMUA UJI LULUS ({len(PASSED)})")


if __name__ == "__main__":
    main()
