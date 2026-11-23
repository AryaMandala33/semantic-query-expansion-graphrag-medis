"""
core/config.py — Konfigurasi sistem utama.

Seluruh nilai di sini ditetapkan sebelum evaluasi akhir dan tidak disetel
berdasarkan hasil set uji (lihat KEPUTUSAN_IMPLEMENTASI.md).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
KAMUS_DIR = DATA_DIR / "kamus"

# --- Neo4j -----------------------------------------------------------------
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
# Database sistem utama (subset 6 relasi, Bab III 3.4). Sengaja memakai variabel
# terpisah dari NEO4J_DATABASE yang dipakai komponen tambahan (primekgv2).
NEO4J_DATABASE_MAIN = os.getenv("NEO4J_DATABASE_MAIN", "primekgbab3")

# --- Subset PrimeKG (Bab III 3.4.2) ------------------------------------------
RELATIONS = [
    "disease_phenotype_positive",
    "indication",
    "contraindication",
    "off-label use",
    "disease_protein",
    "drug_effect",
]

TYPE_TO_LABEL = {
    "disease": "Disease",
    "drug": "Drug",
    "effect/phenotype": "Phenotype",
    "gene/protein": "Protein",
}
LABEL_WORD = {"Disease": "disease", "Drug": "drug", "Phenotype": "phenotype",
              "Protein": "protein"}

# Jenis konsep kandidat LLM (Bab III 3.3.2) -> label node PrimeKG (3.3.4).
# Di PrimeKG, efek obat disimpan sebagai node effect/phenotype.
CONCEPT_TYPE_TO_LABEL = {
    "disease": "Disease",
    "phenotype": "Phenotype",
    "drug effect": "Phenotype",
    "drug": "Drug",
}

# --- Parameter retrieval dan pemeringkatan ------------------------------------
MAX_CANDIDATES = 5                 # 3.3.2: maksimal lima kandidat per kueri
NEIGHBORS_PER_RELATION = 30        # 3.3.4: batas tetangga, sama untuk sistem utama & baseline
BASELINE_MAX_MATCHED_NODES = 5     # 3.5.2: node yang cocok kata kunci (setara 5 kandidat)
TEXT_NEIGHBORS_PER_RELATION = 8    # 3.3.5: jumlah nama tetangga per relasi dalam teks sub-graf

# Urutan relasi pada teks sub-graf: fenotipe lebih dulu karena paling sering
# menjadi petunjuk klinis pada kueri; protein terakhir.
TEXT_RELATION_ORDER = [
    "disease_phenotype_positive",
    "drug_effect",
    "indication",
    "contraindication",
    "off-label use",
    "disease_protein",
]

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
EMBEDDING_CACHE = DATA_DIR / "cache" / "embeddings.pkl"

# --- LLM (3.3.2) ----------------------------------------------------------------
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")
LLM_CACHE_DIR = DATA_DIR / "cache" / "llm_utama"
