"""
core/disambiguation_ranker.py — Disambiguation Ranker (Bab III 3.3.5)

Model all-mpnet-base-v2 (Sentence Transformers) + cosine similarity untuk:
  1. Memilih interpretasi final: cosine(kueri asli, teks sub-graf kandidat).
     Templat teks sub-graf: "<nama> (<tipe>), <relasi>: <tetangga>, ...; <relasi>: ..."
  2. Mengurutkan node hasil retrieval: cosine(kueri, teks node), teks node berisi
     nama, tipe, dan relasinya terhadap konsep terpilih.
Ranker yang sama dipakai baseline (3.5.3).
"""
import pickle

import numpy as np

from core.config import (EMBEDDING_CACHE, EMBEDDING_MODEL, LABEL_WORD,
                         TEXT_NEIGHBORS_PER_RELATION, TEXT_RELATION_ORDER)


def _type_word(label):
    return LABEL_WORD.get(label, (label or "entity").lower())


def node_label_text(node):
    return f"{node['name']} ({_type_word(node.get('label'))})"


def subgraph_text(node, neighbors, per_relation=TEXT_NEIGHBORS_PER_RELATION):
    """Contoh: 'multiple sclerosis (disease), indication: natalizumab, ...; ...'"""
    parts = [node_label_text(node)]
    by_rel = {}
    for n in neighbors:
        by_rel.setdefault(n["relation"], []).append(n)
    for rel in TEXT_RELATION_ORDER:
        group = by_rel.get(rel)
        if group:
            display = group[0].get("display_relation") or rel
            names = ", ".join(n["name"] for n in group[:per_relation])
            parts.append(f"{display}: {names}")
    return parts[0] + (", " + "; ".join(parts[1:]) if len(parts) > 1 else "")


def neighbor_text(neighbor, concept_name):
    """Contoh: 'natalizumab (drug), indication: multiple sclerosis'"""
    display = neighbor.get("display_relation") or neighbor.get("relation")
    return f"{node_label_text(neighbor)}, {display}: {concept_name}"


class DisambiguationRanker:
    def __init__(self, model_name=EMBEDDING_MODEL, encoder=None, cache_path=EMBEDDING_CACHE):
        self.model_name = model_name
        self._encoder = encoder           # dapat disuntikkan untuk uji offline
        self.cache_path = cache_path
        self._cache = {}
        if cache_path and cache_path.exists():
            try:
                stored = pickle.loads(cache_path.read_bytes())
                if stored.get("model") == model_name:
                    self._cache = stored["vectors"]
            except Exception:
                self._cache = {}
        self._dirty = False

    def _model(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def embed(self, texts):
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vecs = self._model().encode(missing, normalize_embeddings=True,
                                        batch_size=64, show_progress_bar=False)
            for t, v in zip(missing, vecs):
                self._cache[t] = np.asarray(v, dtype=np.float32)
            self._dirty = True
        return np.vstack([self._cache[t] for t in texts]) if texts else np.zeros((0, 1))

    def save_cache(self):
        if self._dirty and self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_bytes(pickle.dumps({"model": self.model_name,
                                                      "vectors": self._cache}))
            self._dirty = False

    def cosine(self, query_text, texts):
        if not texts:
            return []
        q = self.embed([query_text])[0]
        m = self.embed(list(texts))
        return [float(x) for x in m @ q]   # vektor sudah dinormalisasi -> dot = cosine

    # ------------------------------------------------------------------
    def score_interpretations(self, query_text, mapped):
        """mapped: list dict berisi 'node', 'neighbors'. Menambahkan 'subgraph_text'
        dan 'similarity', lalu mengurutkan dari tertinggi."""
        for m in mapped:
            m["subgraph_text"] = subgraph_text(m["node"], m["neighbors"])
        sims = self.cosine(query_text, [m["subgraph_text"] for m in mapped])
        for m, s in zip(mapped, sims):
            m["similarity"] = round(s, 4)
        return sorted(mapped, key=lambda m: (-m["similarity"], m["node"]["node_index"]))

    def rank_nodes(self, query_text, items):
        """items: list dict dengan 'node_index' dan 'text'. Duplikat node_index
        disatukan dengan skor tertinggi. Kembalikan list terurut + 'rank'."""
        sims = self.cosine(query_text, [it["text"] for it in items])
        best = {}
        for it, s in zip(items, sims):
            cur = best.get(it["node_index"])
            if cur is None or s > cur["similarity"]:
                best[it["node_index"]] = {**it, "similarity": round(s, 4)}
        ranked = sorted(best.values(), key=lambda x: (-x["similarity"], x["node_index"]))
        for i, r in enumerate(ranked, 1):
            r["rank"] = i
        return ranked

    @staticmethod
    def explain(interp, ranked_nodes, top=5):
        """Penjelasan singkat: konsep terpilih, skor kemiripan, relasi utama."""
        rels = {}
        for n in ranked_nodes[:top]:
            if n.get("relation"):
                rels[n.get("display_relation") or n["relation"]] = rels.get(
                    n.get("display_relation") or n["relation"], 0) + 1
        main_rel = ", ".join(f"{k} ({v})" for k, v in sorted(rels.items(), key=lambda x: -x[1]))
        return {"concept": interp["node"]["name"], "label": interp["node"]["label"],
                "similarity": interp.get("similarity"),
                "main_relations": main_rel or "-"}
