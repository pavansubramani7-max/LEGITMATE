"""Hybrid retrieval for LEGITMATE.

Combines BM25 (lexical) and LSA/TF-IDF cosine (semantic) scores using
Reciprocal Rank Fusion (RRF).

RRF score: sum(1 / (k + rank_i)) for each retriever i
k=60 is the standard constant from Cormack et al. (2009).

This gives a corpus-independent fusion that does not require score
normalisation across retrievers with different score scales.

Each result carries:
  score                 — RRF fusion score (rank-position based)
  best_underlying_score — max(bm25_score, semantic_score) for this chunk;
                          used by the abstention gate in qa/answer.py
  retrieval             — "hybrid-lsa" or "hybrid-tfidf" depending on which
                          semantic backend was active
"""
from __future__ import annotations

from typing import Any

from ..search.lexical import rank as bm25_rank
from .semantic import rank as semantic_rank

_RRF_K = 60


def rank(query: str, chunks: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    """Return top_k chunks fused from BM25 + LSA/TF-IDF cosine via RRF."""
    if not chunks:
        return []

    bm25_hits = bm25_rank(query, chunks, top_k=min(top_k * 2, len(chunks)))
    sem_hits = semantic_rank(query, chunks, top_k=min(top_k * 2, len(chunks)))

    # Detect which semantic backend was used (from first sem hit, default tfidf)
    sem_backend = sem_hits[0].get("retrieval", "tfidf") if sem_hits else "tfidf"

    rrf: dict[str, float] = {}
    best_raw: dict[str, float] = {}

    for rank_pos, hit in enumerate(bm25_hits, start=1):
        cid = hit["chunk_id"]
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (_RRF_K + rank_pos)
        best_raw[cid] = max(best_raw.get(cid, 0.0), hit["score"])
    for rank_pos, hit in enumerate(sem_hits, start=1):
        cid = hit["chunk_id"]
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (_RRF_K + rank_pos)
        best_raw[cid] = max(best_raw.get(cid, 0.0), hit["score"])

    chunk_by_id: dict[str, dict[str, Any]] = {c["chunk_id"]: c for c in chunks}

    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            **chunk_by_id[cid],
            "score": round(score, 6),
            "best_underlying_score": round(best_raw.get(cid, 0.0), 6),
            "retrieval": f"hybrid-{sem_backend}",
        }
        for cid, score in ranked
        if cid in chunk_by_id
    ]
