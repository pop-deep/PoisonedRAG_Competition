"""Pluggable retriever backends.

Three backends are provided; selection is driven by ``config/default.yaml``:

* ``tfidf`` (default) - sklearn TfidfVectorizer, offline, deterministic, no
  downloads. Vocabulary is fit on the global clean corpus once; per-sample
  documents (including injected poison) are transformed against it.
* ``contriever`` - facebook/contriever via transformers (faithful to
  PoisonedRAG). Requires ``transformers`` + ``torch`` and a model download.
* ``sentence_transformers`` - any sentence-transformers model.

If a requested backend's dependencies are missing, the factory falls back to
``tfidf`` with a warning so the pipeline always runs.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from core.config import resolve_path


class RetrieverBase:
    """Abstract retriever: embed texts and rank documents for a query."""

    backend_name = "base"

    def __init__(self, score_function: str = "dot"):
        self.score_function = score_function

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def search(self, query: str, docs: list[dict], top_k: int) -> list[dict]:
        """Return top_k docs ranked by similarity to ``query``.

        ``docs`` is a list of ``{"id":..., "text":...}`` dicts. Returns a list
        of ``{"id":..., "text":..., "score":...}`` sorted descending.
        """
        if not docs:
            return []
        texts = [d["text"] for d in docs]
        q_emb = self.embed([query])[0]
        d_emb = self.embed(texts)
        scores = self._score(q_emb, d_emb)
        order = np.argsort(-scores)
        results = []
        for i in order[:top_k]:
            results.append({**docs[i], "score": float(scores[i])})
        return results

    def _score(self, query_emb: np.ndarray, doc_emb: np.ndarray) -> np.ndarray:
        if self.score_function == "cos_sim":
            qn = query_emb / (np.linalg.norm(query_emb) + 1e-9)
            dn = doc_emb / (np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-9)
            return dn @ qn
        # default: dot product
        return doc_emb @ query_emb


# ---------------------------------------------------------------------------
# TF-IDF backend (default, offline)
# ---------------------------------------------------------------------------

class TfidfRetriever(RetrieverBase):
    backend_name = "tfidf"

    def __init__(self, score_function: str = "dot", corpus_texts: Optional[list[str]] = None,
                 ngram_range: tuple = (1, 2)):
        super().__init__(score_function)
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,
            norm=None,            # keep raw tf-idf so dot product mirrors PoisonedRAG
            dtype=np.float32,
        )
        self._fit_texts = corpus_texts or []
        if self._fit_texts:
            self._vectorizer.fit(self._fit_texts)
        self._cache: dict[int, np.ndarray] = {}

    def fit(self, corpus_texts: list[str]) -> None:
        self._vectorizer.fit(corpus_texts)
        self._fit_texts = corpus_texts
        self._cache.clear()

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self._vectorizer.vocabulary_) or 1), dtype=np.float32)
        return np.asarray(self._vectorizer.transform(texts).todense(), dtype=np.float32)


# ---------------------------------------------------------------------------
# Contriever backend (faithful to PoisonedRAG) - optional
# ---------------------------------------------------------------------------

class ContrieverRetriever(RetrieverBase):
    backend_name = "contriever"

    def __init__(self, score_function: str = "dot", model_name: str = "facebook/contriever",
                 cache_dir: Optional[str] = None):
        super().__init__(score_function)
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.eval()
        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self._device)

    def embed(self, texts: list[str]) -> np.ndarray:
        import torch
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        embs = []
        with torch.no_grad():
            for t in texts:
                inp = self.tok(t, padding=True, truncation=True, max_length=512,
                               return_tensors="pt").to(self._device)
                out = self.model(**inp)
                # mean pooling (contriever) over non-pad tokens
                mask = inp["attention_mask"].unsqueeze(-1).float()
                emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
                embs.append(emb.squeeze(0).cpu().numpy())
        return np.stack(embs).astype(np.float32)


# ---------------------------------------------------------------------------
# Sentence-Transformers backend - optional
# ---------------------------------------------------------------------------

class SentenceTransformerRetriever(RetrieverBase):
    backend_name = "sentence_transformers"

    def __init__(self, score_function: str = "cos_sim", model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_dir: Optional[str] = None):
        super().__init__(score_function)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return np.asarray(self.model.encode(texts, convert_to_numpy=True,
                                             normalize_embeddings=False), dtype=np.float32)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_retriever(config: dict, corpus_texts: Optional[list[str]] = None) -> RetrieverBase:
    """Build a retriever from the ``retriever`` section of default.yaml.

    Falls back to TfidfRetriever if the requested backend's deps are missing.
    """
    rcfg = config if isinstance(config, dict) else {}
    backend = rcfg.get("backend", "tfidf")
    score = rcfg.get("score_function", "dot")
    cache_dir = rcfg.get("model_cache_dir")
    if cache_dir:
        cache_dir = str(resolve_path(cache_dir))

    if backend == "contriever":
        try:
            return ContrieverRetriever(
                score_function=score,
                model_name=rcfg.get("contriever_model", "facebook/contriever"),
                cache_dir=cache_dir,
            )
        except Exception as e:
            print(f"[retriever] contriever backend unavailable ({e!r}); falling back to tfidf.")

    if backend == "sentence_transformer":
        try:
            return SentenceTransformerRetriever(
                score_function=score,
                model_name=rcfg.get("st_model", "sentence-transformers/all-MiniLM-L6-v2"),
                cache_dir=cache_dir,
            )
        except Exception as e:
            print(f"[retriever] sentence_transformers backend unavailable ({e!r}); falling back to tfidf.")

    return TfidfRetriever(score_function=score, corpus_texts=corpus_texts)
