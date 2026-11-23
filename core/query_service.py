"""
core/query_service.py — Query Service (Bab III 3.3.1)

Menerima kueri pengguna, melakukan normalisasi teks, lalu mendeteksi tipe
ambiguitas dengan rule-based classifier tiga aturan yang diperiksa berurutan:

  1. Akronim      : kueri memuat token yang ada di kamus akronim medis ambigu
                    (Medical Abbreviation and Acronym Meta-Inventory).
  2. Polisemi     : kueri memuat istilah dari daftar istilah polisemi medis
                    (NLM WSD Test Collection).
  3. Specificity  : kueri memuat kata umum ("sakit", "penyakit", "disease", ...)
     gap            yang disertai nama organ/sistem tubuh.

Kueri yang tidak memenuhi aturan apa pun diberi label "none" (tidak terdeteksi)
dan tetap diteruskan ke LLM Expander sebagai kueri tanpa tipe.

Keluaran berupa objek JSON (dict) berisi kueri asli, kueri ternormalisasi,
label tipe ambiguitas, istilah pemicu, dan aturan yang terpenuhi.
"""
import json
import re

from core.config import KAMUS_DIR
from core.text_utils import normalize

# Akronim harus ditulis dengan huruf kapital pada kueri asli (mis. "MS"), agar
# kata biasa seperti "is", "as", "at" — yang juga tercatat sebagai singkatan
# di Meta-Inventory — tidak terdeteksi sebagai akronim.
_ACRONYM_TOKEN = re.compile(r"\b[A-Z][A-Z0-9]{1,5}\b")


def _load_json(name):
    return json.loads((KAMUS_DIR / name).read_text(encoding="utf-8"))


class QueryService:
    LABELS = ("acronym", "polysemy", "specificity_gap", "none")

    def __init__(self):
        self.acronyms = _load_json("akronim_meta_inventory.json")["acronyms"]
        self.polysemy_terms = sorted(_load_json("polisemi_nlm_wsd.json")["terms"],
                                     key=len, reverse=True)
        spec = _load_json("specificity_gap_rules.json")
        organs = spec["organs"]["id"] + spec["organs"]["en"]
        organ_alt = "|".join(sorted((re.escape(o) for o in organs), key=len, reverse=True))
        before = "|".join(spec["general_words_before"])
        after = "|".join(spec["general_words_after"])
        # "penyakit jantung", "sakit perut"  |  "heart disease", "kidney problem"
        self._spec_patterns = [
            re.compile(rf"\b(?:{before})\s+(?:{organ_alt})\b"),
            re.compile(rf"\b(?:{organ_alt})\s+(?:{after})\b"),
        ]

    # ------------------------------------------------------------------
    def _rule_acronym(self, original):
        for m in _ACRONYM_TOKEN.finditer(original):
            tok = m.group(0)
            if tok in self.acronyms:
                return tok
        return None

    def _rule_polysemy(self, normalized):
        for term in self.polysemy_terms:
            if re.search(rf"\b{re.escape(term)}\b", normalized):
                return term
        return None

    def _rule_specificity(self, normalized):
        for pat in self._spec_patterns:
            m = pat.search(normalized)
            if m:
                return m.group(0)
        return None

    # ------------------------------------------------------------------
    def process(self, query):
        original = (query or "").strip()
        normalized = normalize(original)
        result = {"query": original, "normalized_query": normalized,
                  "ambiguity_type": "none", "trigger_term": None, "rule": None}
        if not normalized:
            result["error"] = "Kueri tidak boleh kosong."
            return result

        checks = [
            ("acronym", "aturan 1: kamus akronim (Meta-Inventory)",
             lambda: self._rule_acronym(original)),
            ("polysemy", "aturan 2: daftar istilah polisemi (NLM WSD)",
             lambda: self._rule_polysemy(normalized)),
            ("specificity_gap", "aturan 3: istilah umum + organ/sistem tubuh",
             lambda: self._rule_specificity(normalized)),
        ]
        for label, rule, fn in checks:
            trigger = fn()
            if trigger:
                result.update(ambiguity_type=label, trigger_term=trigger, rule=rule)
                break
        return result

    def acronym_senses(self, short_form, n=5):
        entry = self.acronyms.get(short_form.upper())
        return entry["examples"][:n] if entry else []
