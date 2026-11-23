"""Komponen tiruan untuk uji offline (tanpa Neo4j, LLM, dan model embedding)."""
import hashlib
import json

import numpy as np

# Graf mini: node_index -> (name, label)
NODES = {
    1: ("multiple sclerosis", "Disease"), 2: ("Mitral stenosis", "Phenotype"),
    3: ("natalizumab", "Drug"), 4: ("Optic neuritis", "Phenotype"),
    5: ("Paresthesia", "Phenotype"), 6: ("heart failure", "Disease"),
    7: ("Edema", "Phenotype"), 8: ("Orthopnea", "Phenotype"),
    9: ("coronary artery disease", "Disease"), 10: ("Angina pectoris", "Phenotype"),
    11: ("furosemide", "Drug"), 12: ("rheumatic fever", "Disease"),
}
EDGES = [  # (a, b, relation, display)
    (3, 1, "indication", "indication"), (1, 4, "disease_phenotype_positive", "phenotype present"),
    (1, 5, "disease_phenotype_positive", "phenotype present"),
    (12, 2, "disease_phenotype_positive", "phenotype present"),
    (6, 7, "disease_phenotype_positive", "phenotype present"),
    (6, 8, "disease_phenotype_positive", "phenotype present"),
    (11, 6, "indication", "indication"),
    (9, 10, "disease_phenotype_positive", "phenotype present"),
]


def node(i):
    name, label = NODES[i]
    return {"node_index": i, "name": name, "label": label, "id": str(i), "source": "TEST"}


class FakeRetriever:
    def map_candidate(self, name, concept_type):
        from core.config import CONCEPT_TYPE_TO_LABEL
        from core.graph_retriever import name_variants
        label = CONCEPT_TYPE_TO_LABEL.get(concept_type)
        hits = [i for i, (n, _) in NODES.items() if n.lower() in name_variants(name)]
        same = [i for i in hits if NODES[i][1] == label]
        if same:
            return node(same[0]), "mapped"
        return None, ("type_mismatch" if hits else "not_found")

    def neighbors(self, idx, per_relation=30):
        out = []
        for a, b, rel, disp in EDGES:
            if idx in (a, b):
                other = b if a == idx else a
                out.append({**node(other), "degree": 1, "relation": rel, "display_relation": disp})
        return out

    def relevant_neighbors(self, idx, relations):
        return [n["node_index"] for n in self.neighbors(idx) if n["relation"] in relations]

    def keyword_nodes(self, keywords):
        rows = []
        for i, (name, label) in NODES.items():
            words = set(name.lower().split())
            hit = [k for k in keywords if k in words]
            if hit:
                rows.append({**node(i), "matched_keywords": hit})
        return rows


class FakeEncoder:
    """Bag-of-words ter-hash -> vektor 256 dimensi ternormalisasi."""
    def encode(self, texts, normalize_embeddings=True, **kw):
        out = []
        for t in texts:
            v = np.zeros(256, dtype=np.float32)
            for w in t.lower().replace(",", " ").replace(":", " ").split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
            n = np.linalg.norm(v)
            out.append(v / n if n else v)
        return np.array(out)


class _Msg:
    def __init__(self, c):
        self.content = c


class _Choice:
    def __init__(self, c):
        self.message = _Msg(c)


class _Resp:
    def __init__(self, c):
        self.choices = [_Choice(c)]


class FakeLLMClient:
    """Mengembalikan respons berurutan dari daftar `outputs` (string JSON/mentah)."""
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls += 1
        out = self.outputs.pop(0)
        return _Resp(out if isinstance(out, str) else json.dumps(out))
