"""Optional local cross-encoder reranking for LEGITMATE search results.

The cross-encoder scores each question/passage pair after first-stage retrieval.
It only reorders a small candidate set; original retrieval scores and abstention
threshold decisions remain intact. The model is loaded from the local cache only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-2-v2"
_model: Any = None
_load_error: str | None = None


def _cache_dir() -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def model_cached() -> bool:
    repo = _cache_dir() / f"models--{MODEL_ID.replace('/', '--')}"
    snapshots = repo / "snapshots"
    try:
        return snapshots.is_dir() and any(p.is_dir() for p in snapshots.iterdir())
    except OSError:
        return False


def reranker_info() -> dict[str, Any]:
    cached = model_cached()
    if _load_error:
        return {"available": False, "cached": cached, "active": False, "model": MODEL_ID,
                "detail": "The local reranker could not load; search will use the existing hybrid ranking."}
    return {
        "available": cached,
        "cached": cached,
        "active": _model is not None,
        "model": MODEL_ID,
        "detail": ("Cross-encoder model is cached locally and ready for query/passage reranking."
                   if cached else "Cross-encoder model is not cached; existing hybrid search remains available."),
    }


def rerank(question: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rerank hybrid candidates if the public model is already cached locally."""
    global _model, _load_error
    if not hits or not model_cached() or _load_error:
        return hits
    if _model is None:
        try:
            import torch
            from sentence_transformers import CrossEncoder
            _model = CrossEncoder(
                MODEL_ID,
                local_files_only=True,
                device="cpu",
                max_length=512,
                activation_fn=torch.nn.Sigmoid(),
            )
        except Exception:
            _load_error = "load_failed"
            return hits
    try:
        pairs = [(question, str(hit.get("text", ""))) for hit in hits]
        scores = _model.predict(pairs, batch_size=8, show_progress_bar=False)
        ranked = []
        for hit, score in zip(hits, scores):
            ranked.append({**hit, "rerank_score": round(float(score), 6), "reranked": True})
        return sorted(ranked, key=lambda hit: hit["rerank_score"], reverse=True)
    except Exception:
        _load_error = "inference_failed"
        return hits
