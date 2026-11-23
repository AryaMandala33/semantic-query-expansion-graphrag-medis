"""core/text_utils.py — Normalisasi teks dan kata henti (dipakai Query Service & baseline)."""
import re

from core.config import KAMUS_DIR

_PUNCT = re.compile(r"[^\w\s\-/']", flags=re.UNICODE)
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def normalize(text):
    """Bab III 3.3.1: huruf kecil, hapus tanda baca yang tidak diperlukan, rapikan spasi.
    Tanda hubung, garis miring, dan apostrof dipertahankan (mis. 'st-segment')."""
    t = (text or "").lower()
    t = _PUNCT.sub(" ", t)
    return " ".join(t.split())


def tokens(text):
    return _TOKEN.findall(normalize(text))


def _load_words(name):
    path = KAMUS_DIR / name
    return set(path.read_text(encoding="utf-8").split()) if path.exists() else set()


STOPWORDS = _load_words("stopwords_en.txt") | _load_words("stopwords_id.txt")


def keywords(text):
    """Bab III 3.5.1: kata kunci = token kueri tanpa kata henti Indonesia & Inggris.
    Token yang seluruhnya angka (mis. '60' dari '60-year-old') ikut dibuang."""
    out = []
    for tok in tokens(text):
        for part in tok.split("-"):
            if part and part not in STOPWORDS and not part.isdigit() and part not in out:
                out.append(part)
    return out
