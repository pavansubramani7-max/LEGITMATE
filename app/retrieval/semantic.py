"""Semantic retrieval for LEGITMATE.

Backend selection (automatic, priority order):
  1. Neural  — sentence-transformers + torch installed.
               Model: intfloat/multilingual-e5-small (384-dim, Apache-2.0).
               Supports English, Hindi, Kannada and 100+ other languages.
               Vectors are persisted per-document in data/processed/<doc_id>.npz
               and reloaded on subsequent queries; the full corpus is NOT
               re-encoded on every request.
               Cache is invalidated when the model identifier changes.
  2. LSA     — scikit-learn TruncatedSVD over TF-IDF matrix (128 components).
               Corpus-driven; rebuilt in memory on every query.
               Fallback when sentence-transformers / torch is absent.
  3. TF-IDF  — pure-numpy cosine similarity.
               Fallback when scikit-learn is also absent.

E5 prompt convention (required for correct retrieval quality):
  Queries  → prefix "query: "
  Passages → prefix "passage: "

Persistence format (.npz):
  texts      — object array of str (chunk text)
  chunk_ids  — object array of str
  vectors    — float32 array shape (n, 384)  [neural only]
  model_id   — scalar str, e.g. "intfloat/multilingual-e5-small"
               Used to invalidate cache when the model changes.
  If vectors or model_id are absent the file is treated as a legacy text cache
  and vectors are re-encoded on next access.
"""
from __future__ import annotations

import re
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ALL_STOP_WORDS
from ..store import workspace_processed

# ── optional imports ──────────────────────────────────────────────────────────
try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    _SKLEARN = True
except ImportError:  # pragma: no cover
    _SKLEARN = False

try:
    from sentence_transformers import SentenceTransformer as _ST
    _SBERT = True
except Exception:  # pragma: no cover
    _SBERT = False

_LSA_COMPONENTS = 128
_NEURAL_MODEL_ID = os.environ.get("LEGITMATE_EMBEDDING_MODEL", "").strip() or "intfloat/multilingual-e5-small"
_model_singleton: Any = None  # loaded once, reused across requests


def _get_model() -> Any:
    global _model_singleton
    if _model_singleton is None:
        _model_singleton = _ST(_NEURAL_MODEL_ID)
    return _model_singleton


# ── tokeniser (shared with BM25) ──────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[\w'-]+", text.casefold())
            if len(t) > 1 and t not in ALL_STOP_WORDS]


# ── TF-IDF fallback (numpy only) ─────────────────────────────────────────────

class _TFIDFIndex:
    def __init__(self, texts: list[str]) -> None:
        tokenized = [_tokenize(t) for t in texts]
        vocab: dict[str, int] = {}
        for tokens in tokenized:
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self._vocab = vocab
        n, v = len(texts), len(vocab)
        if n == 0 or v == 0:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            self._idf = np.zeros(0, dtype=np.float32)
            return
        df = np.zeros(v, dtype=np.float32)
        for tokens in tokenized:
            for t in set(tokens):
                if t in vocab:
                    df[vocab[t]] += 1
        idf = np.log((n + 1) / (df + 1)) + 1.0
        mat = np.zeros((n, v), dtype=np.float32)
        for i, tokens in enumerate(tokenized):
            tf = Counter(tokens)
            for t, count in tf.items():
                if t in vocab:
                    mat[i, vocab[t]] = count / max(len(tokens), 1)
        mat = mat * idf
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = mat / norms
        self._idf = idf

    def query(self, text: str, top_k: int) -> list[tuple[int, float]]:
        if self._matrix.shape[0] == 0:
            return []
        tokens = _tokenize(text)
        if not tokens:
            return []
        v = len(self._vocab)
        qvec = np.zeros(v, dtype=np.float32)
        tf = Counter(tokens)
        for t, count in tf.items():
            if t in self._vocab:
                qvec[self._vocab[t]] = count / max(len(tokens), 1)
        qvec = qvec * self._idf
        norm = np.linalg.norm(qvec)
        if norm == 0:
            return []
        qvec /= norm
        scores = self._matrix @ qvec
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    @property
    def backend(self) -> str:
        return "tfidf"


# ── LSA index (scikit-learn) ──────────────────────────────────────────────────

class _LSAIndex:
    def __init__(self, texts: list[str]) -> None:
        n = len(texts)
        if n < 2:
            self._fallback = _TFIDFIndex(texts)
            self._svd = None
            return
        self._fallback = None
        n_components = min(_LSA_COMPONENTS, n - 1)
        self._vec = TfidfVectorizer(
            tokenizer=_tokenize, token_pattern=None, sublinear_tf=True, min_df=1,
        )
        tfidf_mat = self._vec.fit_transform(texts)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        lsa_mat = self._svd.fit_transform(tfidf_mat)
        norms = np.linalg.norm(lsa_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = (lsa_mat / norms).astype(np.float32)

    def query(self, text: str, top_k: int) -> list[tuple[int, float]]:
        if self._fallback is not None:
            return self._fallback.query(text, top_k)
        if self._matrix.shape[0] == 0:
            return []
        qvec = self._svd.transform(self._vec.transform([text]))[0].astype(np.float32)
        norm = np.linalg.norm(qvec)
        if norm == 0:
            return []
        qvec /= norm
        scores = self._matrix @ qvec
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    @property
    def backend(self) -> str:
        return "tfidf" if self._fallback is not None else "lsa"


# ── Neural index (sentence-transformers) ─────────────────────────────────────

class _NeuralIndex:
    """Dense cosine index using intfloat/multilingual-e5-small.

    Vectors are passed in at construction time (already encoded and cached).
    The query is encoded on the fly with the "query: " prefix.
    """

    def __init__(self, matrix: np.ndarray) -> None:
        # matrix: float32 (n, 384), already L2-normalised
        self._matrix = matrix

    def query(self, text: str, top_k: int) -> list[tuple[int, float]]:
        if self._matrix.shape[0] == 0:
            return []
        model = _get_model()
        qvec = model.encode([f"query: {text}"], normalize_embeddings=True)[0].astype(np.float32)
        scores = self._matrix @ qvec
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]

    @property
    def backend(self) -> str:
        return "neural"


# ── persistence helpers ───────────────────────────────────────────────────────

def _npz_path(doc_id: str) -> Path:
    return workspace_processed() / f"{doc_id}.npz"


def save_embeddings(doc_id: str, chunks: list[dict[str, Any]]) -> None:
    """Encode and persist chunk vectors (neural) or text cache (LSA/TF-IDF).

    Neural backend: encodes passages with "passage: " prefix, stores float32
    vectors alongside texts and chunk_ids.  model_id is stored so stale caches
    can be detected when the model changes.

    LSA/TF-IDF fallback: stores text strings and chunk_ids only (no vectors).
    """
    workspace_processed().mkdir(parents=True, exist_ok=True)
    texts = np.array([c["text"] for c in chunks], dtype=object)
    chunk_ids = np.array([c["chunk_id"] for c in chunks], dtype=object)
    if _SBERT:
        model = _get_model()
        passages = [f"passage: {c['text']}" for c in chunks]
        vecs = model.encode(passages, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        np.savez(
            _npz_path(doc_id),
            texts=texts,
            chunk_ids=chunk_ids,
            vectors=vecs,
            model_id=np.array(_NEURAL_MODEL_ID),
        )
    else:
        np.savez(_npz_path(doc_id), texts=texts, chunk_ids=chunk_ids)


def load_embeddings(doc_id: str) -> tuple[list[str], list[str]] | None:
    """Return (texts, chunk_ids) from the cache, or None if absent."""
    p = _npz_path(doc_id)
    if not p.exists():
        return None
    data = np.load(p, allow_pickle=True)
    return list(data["texts"]), list(data["chunk_ids"])


def _load_vectors(doc_id: str) -> np.ndarray | None:
    """Return cached float32 vectors for doc_id if they exist and match the current model.

    Returns None if the file is absent, has no vectors, or was built with a
    different model (cache invalidation).
    """
    p = _npz_path(doc_id)
    if not p.exists():
        return None
    data = np.load(p, allow_pickle=True)
    if "vectors" not in data or "model_id" not in data:
        return None
    if str(data["model_id"]) != _NEURAL_MODEL_ID:
        return None  # model changed — caller must re-encode
    return data["vectors"].astype(np.float32)


def delete_embeddings(doc_id: str) -> None:
    """Remove the cached file for a deleted document."""
    p = _npz_path(doc_id)
    if p.exists():
        p.unlink()


# ── corpus-level vector assembly ─────────────────────────────────────────────

def _get_corpus_matrix(chunks: list[dict[str, Any]]) -> np.ndarray:
    """Return a float32 matrix (n_chunks, dim) of passage vectors.

    Loads cached vectors per document; encodes any chunk whose document has no
    valid cache.  Does NOT write new cache files here — that is done at upload
    time by save_embeddings().  This function only fills gaps (e.g. legacy docs
    uploaded before the neural backend was available).
    """
    # Group chunks by document so we can load per-doc caches
    doc_order: dict[str, list[int]] = {}  # doc_id -> list of chunk indices
    for i, c in enumerate(chunks):
        doc_order.setdefault(c["document_id"], []).append(i)

    result = np.zeros((len(chunks), 384), dtype=np.float32)
    model = _get_model()

    for doc_id, indices in doc_order.items():
        cached = _load_vectors(doc_id)
        if cached is not None and len(cached) == len(indices):
            for local_i, global_i in enumerate(indices):
                result[global_i] = cached[local_i]
        else:
            # Encode on the fly (legacy doc or cache miss)
            passages = [f"passage: {chunks[i]['text']}" for i in indices]
            vecs = model.encode(passages, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
            for local_i, global_i in enumerate(indices):
                result[global_i] = vecs[local_i]

    return result


# ── public interface ──────────────────────────────────────────────────────────

def _build_index(texts: list[str]) -> _LSAIndex | _TFIDFIndex:
    if _SKLEARN:
        return _LSAIndex(texts)
    return _TFIDFIndex(texts)  # pragma: no cover


def rank(query: str, chunks: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    """Return top_k chunks ranked by semantic similarity.

    Uses neural embeddings (intfloat/multilingual-e5-small) when available,
    falling back to LSA or TF-IDF.  Each result carries a `retrieval` key.
    """
    if not chunks:
        return []

    if _SBERT:
        matrix = _get_corpus_matrix(chunks)
        index: Any = _NeuralIndex(matrix)
    else:
        index = _build_index([c["text"] for c in chunks])

    results = index.query(query, top_k)
    backend = index.backend
    return [
        {**chunks[i], "score": round(score, 4), "retrieval": backend}
        for i, score in results
    ]


def backend_info() -> dict[str, str | bool]:
    """Describe available semantic-search dependencies without loading a model."""
    if _SBERT:
        try:
            from huggingface_hub.constants import HF_HUB_CACHE
            cache = Path(HF_HUB_CACHE)
        except Exception:
            cache = Path.home() / ".cache" / "huggingface" / "hub"
        local_model = Path(_NEURAL_MODEL_ID).expanduser()
        repo_cache = cache / f"models--{_NEURAL_MODEL_ID.replace('/', '--')}" / "snapshots"
        try:
            cached = (local_model.is_dir() and any(local_model.iterdir())) or (repo_cache.is_dir() and any(path.is_dir() for path in repo_cache.iterdir()))
        except OSError:
            cached = False
        return {
            "backend": "neural",
            "label": "E5 neural model cached locally" if cached else "E5 neural package installed; model is not cached",
            "model": _NEURAL_MODEL_ID,
            "package_available": True,
            "cached": cached,
            "detail": ("Multilingual embeddings load from this computer; cached passage vectors are reused."
                       if cached else "Model loading is lazy and requires network access the first time; cached passage vectors are reused afterward."),
        }
    if _SKLEARN:
        return {
            "backend": "lsa",
            "label": "LSA / TF-IDF fallback active",
            "model": "scikit-learn TruncatedSVD + TF-IDF",
            "package_available": False,
            "cached": False,
            "detail": "Corpus-based fallback; no pretrained neural model is active.",
        }
    return {
        "backend": "tfidf",
        "label": "TF-IDF fallback active",
        "model": "local NumPy TF-IDF",
        "package_available": False,
        "cached": False,
        "detail": "Lightweight lexical-semantic fallback; no pretrained neural model is active.",
    }
