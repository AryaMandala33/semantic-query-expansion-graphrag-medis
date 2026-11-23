"""
core/llm_expander.py — LLM Expander (Bab III 3.3.2)

- Model utama GPT-4o mini melalui API; Mistral-7B dapat dipakai sebagai
  alternatif lokal lewat endpoint yang kompatibel OpenAI (mis. Ollama), cukup
  dengan mengubah LLM_BASE_URL dan LLM_MODEL_NAME di .env.
- temperature 0 dan versi prompt dikunci (PROMPT_VERSION) agar reprodusibel.
- Prompt: peran sistem, definisi tiga tipe ambiguitas, instruksi memanfaatkan
  konteks kalimat kueri, format keluaran JSON, dan contoh few-shot yang tidak
  diambil dari kueri evaluasi.
- Setiap kandidat: nama konsep (bahasa Inggris), jenis konsep (disease,
  phenotype, drug effect, drug), confidence, dan alasan singkat. Maksimal 5.
- Untuk specificity gap, keluaran memuat pertanyaan klarifikasi.
- Keluaran diperiksa terhadap skema JSON; permintaan diulang SATU kali bila
  tidak valid.
- Label dari Query Service diberikan kepada LLM sebagai petunjuk.
"""
import hashlib
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from core.config import LLM_CACHE_DIR, LLM_MODEL_NAME, MAX_CANDIDATES

load_dotenv()

PROMPT_VERSION = "utama-v1 (2026-10-29)"
VALID_TYPES = ("polysemy", "acronym", "specificity_gap", "none")
CONCEPT_TYPES = ("disease", "phenotype", "drug effect", "drug")

SYSTEM_PROMPT = (
    "You are a biomedical terminology expert supporting a GraphRAG system built on the "
    "PrimeKG knowledge graph. In PrimeKG, diseases follow MONDO naming, phenotypes "
    "(symptoms, clinical signs, test findings) and drug effects follow HPO naming, and "
    "drugs follow DrugBank naming. Your task is to resolve lexical ambiguity of medical "
    "terms in a user query. Never invent identifiers. Respond with one valid JSON object only."
)

# Contoh few-shot. Istilahnya (tension, CP, penyakit kulit) tidak dipakai di set
# uji maupun set dev, sehingga tidak terjadi kebocoran data uji.
FEW_SHOT_EXAMPLES = [
    {
        "query": "tension pada pasien trauma dada dengan sesak berat dan trakea bergeser",
        "qs": {"ambiguity_type": "none", "trigger_term": None},
        "output": {
            "ambiguous_term": "tension",
            "detected_type": "polysemy",
            "type_reason": "'Tension' can denote a type of pneumothorax, a headache type, or muscle tension.",
            "senses_considered": ["Tension pneumothorax", "tension-type headache", "Muscle stiffness"],
            "candidates": [
                {"name": "Tension pneumothorax", "entity_type": "phenotype", "confidence": 0.9,
                 "reason": "Chest trauma, severe dyspnea and tracheal deviation are classic signs."},
                {"name": "tension-type headache", "entity_type": "disease", "confidence": 0.03,
                 "reason": "No headache is described; inconsistent with acute chest trauma."},
                {"name": "Muscle stiffness", "entity_type": "phenotype", "confidence": 0.02,
                 "reason": "Nothing in the query points to muscle symptoms."}],
            "clarifying_question": None, "uncertainty": None,
        },
    },
    {
        "query": "CP pada anak 3 tahun dengan kekakuan otot dan terlambat berjalan",
        "qs": {"ambiguity_type": "acronym", "trigger_term": "CP"},
        "output": {
            "ambiguous_term": "CP",
            "detected_type": "acronym",
            "type_reason": "CP can expand to cerebral palsy, chest pain, cleft palate or chronic pancreatitis.",
            "senses_considered": ["cerebral palsy", "Chest pain", "Cleft palate", "chronic pancreatitis"],
            "candidates": [
                {"name": "cerebral palsy", "entity_type": "disease", "confidence": 0.9,
                 "reason": "A young child with spasticity and delayed walking fits cerebral palsy."},
                {"name": "Cleft palate", "entity_type": "phenotype", "confidence": 0.05,
                 "reason": "Pediatric, but explains neither spasticity nor delayed walking."},
                {"name": "Chest pain", "entity_type": "phenotype", "confidence": 0.02,
                 "reason": "No chest symptoms are described."}],
            "clarifying_question": None, "uncertainty": None,
        },
    },
    {
        "query": "penyakit kulit dengan bercak merah bersisik tebal di siku dan lutut",
        "qs": {"ambiguity_type": "specificity_gap", "trigger_term": "penyakit kulit"},
        "output": {
            "ambiguous_term": "penyakit kulit",
            "detected_type": "specificity_gap",
            "type_reason": "'Penyakit kulit' (skin disease) is a general term covering many specific conditions.",
            "senses_considered": ["psoriasis", "contact dermatitis", "seborrheic dermatitis", "lichen planus"],
            "candidates": [
                {"name": "psoriasis", "entity_type": "disease", "confidence": 0.85,
                 "reason": "Thick red scaly plaques on elbows and knees are typical of plaque psoriasis."},
                {"name": "contact dermatitis", "entity_type": "disease", "confidence": 0.15,
                 "reason": "Red and scaly, but usually limited to areas exposed to an irritant."},
                {"name": "seborrheic dermatitis", "entity_type": "disease", "confidence": 0.1,
                 "reason": "Scaly, but usually on scalp and face, not elbows and knees."}],
            "clarifying_question": "Penyakit kulit mana yang Anda maksud: psoriasis, dermatitis kontak, atau dermatitis seboroik?",
            "uncertainty": None,
        },
    },
]


def _qs_line(qs):
    if qs and qs.get("ambiguity_type") in ("acronym", "polysemy", "specificity_gap"):
        return (f'label "{qs["ambiguity_type"]}", triggered by "{qs.get("trigger_term")}"')
    return "no ambiguity type detected"


def _few_shot_block():
    lines = ["EXAMPLES:"]
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        lines.append(f'\nExample {i}\nQUERY: "{ex["query"]}"\n'
                     f"RULE-BASED DETECTOR: {_qs_line(ex['qs'])}")
        lines.append("OUTPUT: " + json.dumps(ex["output"], ensure_ascii=False))
    return "\n".join(lines) + "\n\nNOW THE REAL TASK.\n"


USER_TEMPLATE = """Perform semantic query expansion for the medical query below.

DEFINITIONS OF AMBIGUITY TYPES
- "polysemy": one term with several distinct medical meanings (e.g. a disorder vs. a physiological sign vs. a test finding).
- "acronym": an abbreviation with several possible medical expansions.
- "specificity_gap": a general term that is too broad to identify one specific medical concept.
- "none": the query contains no ambiguous medical term.
{examples}
QUERY: "{query}"
RULE-BASED DETECTOR: {qs_line}

INSTRUCTIONS
1. Use the whole sentence of the query as context: the symptoms, signs, findings and drugs it mentions decide which meaning is intended.
2. ambiguous_term: the ambiguous medical term, copied as written in the query.
3. detected_type: one of polysemy, acronym, specificity_gap, none. The rule-based detector label is a hint only; decide yourself. Give type_reason in one sentence.
4. senses_considered: list the distinct medical senses of the ambiguous term before choosing candidates.
5. candidates: at most {max_candidates} interpretations, most plausible first.
   - name: standard English medical name as used in MONDO/HPO/DrugBank, also when the query is in Indonesian. Never an abbreviation. No qualifiers in parentheses.
   - entity_type: one of "disease", "phenotype", "drug effect", "drug".
   - confidence: 0-1, based on the context of the query.
   - reason: one short sentence.
   For polysemy and acronym, include competing meanings that the context argues against (with low confidence).
   For specificity_gap, list more specific concepts that fall under the general term.
6. clarifying_question: for specificity_gap, a short question in the SAME language as the query asking which specific concept the user means; otherwise null.
7. uncertainty: a short note on what remains uncertain, or null.

Return JSON with exactly these keys:
{{
  "ambiguous_term": "",
  "detected_type": "",
  "type_reason": "",
  "senses_considered": [],
  "candidates": [{{"name": "", "entity_type": "", "confidence": 0.0, "reason": ""}}],
  "clarifying_question": null,
  "uncertainty": null
}}"""


class SchemaError(ValueError):
    pass


def _str(v):
    return v.strip() if isinstance(v, str) and v.strip() else None


def validate_output(data, query):
    """Periksa keluaran terhadap skema. SchemaError -> permintaan diulang sekali."""
    if not isinstance(data, dict):
        raise SchemaError("keluaran bukan objek JSON")
    for key in ("ambiguous_term", "detected_type", "candidates"):
        if key not in data:
            raise SchemaError(f"kunci '{key}' tidak ada")
    if not isinstance(data["candidates"], list):
        raise SchemaError("'candidates' bukan list")
    detected = (data.get("detected_type") or "").strip().lower()
    if detected not in VALID_TYPES:
        raise SchemaError(f"detected_type tidak valid: {detected!r}")

    candidates, seen = [], set()
    for c in data["candidates"]:
        if not isinstance(c, dict) or not _str(c.get("name")):
            raise SchemaError("kandidat tanpa nama")
        name = _str(c["name"])
        if name.lower() in seen:
            continue
        etype = (c.get("entity_type") or "").strip().lower()
        if etype not in CONCEPT_TYPES:
            raise SchemaError(f"entity_type tidak valid: {etype!r}")
        try:
            conf = min(1.0, max(0.0, float(c.get("confidence"))))
        except (TypeError, ValueError):
            raise SchemaError("confidence bukan angka")
        seen.add(name.lower())
        candidates.append({"name": name, "entity_type": etype,
                           "confidence": round(conf, 3), "reason": _str(c.get("reason")) or ""})
    candidates = candidates[:MAX_CANDIDATES]

    senses = data.get("senses_considered") or []
    return {
        "query": query,
        "ambiguous_term": _str(data.get("ambiguous_term")) or query,
        "detected_type": detected,
        "type_reason": _str(data.get("type_reason")),
        "senses_considered": [s for s in (_str(x) for x in senses) if s],
        "candidates": candidates,
        "clarifying_question": _str(data.get("clarifying_question")),
        "uncertainty": _str(data.get("uncertainty")),
    }


class LLMExpander:
    def __init__(self, api_key=None, base_url=None, model=None, temperature=0.0,
                 use_cache=True, client=None):
        self.model_name = model or LLM_MODEL_NAME
        self.temperature = temperature
        self.use_cache = use_cache
        if client is not None:            # dipakai uji offline
            self.client = client
            return
        api_key = api_key or os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("LLM_API_KEY tidak ditemukan di .env")
        cfg = {"api_key": api_key}
        base_url = base_url or os.getenv("LLM_BASE_URL") or None
        if base_url:
            cfg["base_url"] = base_url
        self.client = OpenAI(**cfg)

    def build_messages(self, query, qs=None):
        user = USER_TEMPLATE.format(examples=_few_shot_block(), query=query.strip(),
                                    qs_line=_qs_line(qs), max_candidates=MAX_CANDIDATES)
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]

    def _cache_path(self, messages):
        key = json.dumps({"model": self.model_name, "temperature": self.temperature,
                          "messages": messages}, sort_keys=True, ensure_ascii=False)
        return LLM_CACHE_DIR / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json")

    def _complete(self, messages):
        resp = self.client.chat.completions.create(
            model=self.model_name, messages=messages, temperature=self.temperature,
            response_format={"type": "json_object"})
        return resp.choices[0].message.content or ""

    def expand(self, query, qs=None):
        base = {"query": query, "detected_type": "none", "candidates": [],
                "clarifying_question": None, "prompt_version": PROMPT_VERSION}
        if not isinstance(query, str) or not query.strip():
            return {**base, "error": "Kueri tidak boleh kosong."}

        messages = self.build_messages(query, qs)
        path = self._cache_path(messages)
        if self.use_cache and path.exists():
            rec = json.loads(path.read_text(encoding="utf-8"))
            out = validate_output(json.loads(rec["raw"]), query.strip())
            return {**out, "prompt_version": PROMPT_VERSION,
                    "meta": {**rec["meta"], "cached": True}}

        t0 = time.time()
        attempts, last_error = [], None
        for attempt in (1, 2):            # percobaan awal + satu kali ulang
            try:
                msgs = messages if attempt == 1 else messages + [
                    {"role": "assistant", "content": attempts[-1]},
                    {"role": "user", "content":
                        f"Your previous output was invalid ({last_error}). "
                        "Return one valid JSON object that follows the required keys exactly."}]
                raw = self._complete(msgs)
                attempts.append(raw)
                out = validate_output(json.loads(raw), query.strip())
                meta = {"model": self.model_name, "latency_s": round(time.time() - t0, 2),
                        "attempts": attempt, "cached": False}
                if self.use_cache:
                    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"raw": raw, "meta": meta}, ensure_ascii=False,
                                               indent=2), encoding="utf-8")
                return {**out, "prompt_version": PROMPT_VERSION, "meta": meta}
            except (json.JSONDecodeError, SchemaError) as e:
                last_error = f"{type(e).__name__}: {e}"
                if not attempts:
                    attempts.append("")
            except Exception as e:        # galat API / koneksi: tidak diulang
                return {**base, "error": f"Gagal memanggil LLM: {type(e).__name__}: {e}"}
        return {**base, "error": f"Keluaran LLM tidak valid setelah 2 percobaan ({last_error})"}
