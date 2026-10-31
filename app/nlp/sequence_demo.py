"""Tiny local GRU + additive-attention classifier for teaching only."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.config import ROOT

CASES_PATH = ROOT / "data" / "evaluation" / "sequence_demo_cases_v1.json"
WEIGHTS_PATH = ROOT / "models" / "gru-sequence-demo.pt"
_TOKEN_RE = re.compile(r"[\w]+(?:['’/-][\w]+)*", re.UNICODE)
_CACHE: tuple[Any, dict[str, int], list[str], float] | None = None


def _load_cases() -> dict[str, Any]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def model_status() -> dict[str, Any]:
    return {
        "name": "GRU sequence classifier + additive attention",
        "trained_weights": WEIGHTS_PATH.is_file(),
        "model_path": str(WEIGHTS_PATH.relative_to(ROOT)) if WEIGHTS_PATH.is_file() else None,
        "training_data": "Hand-labeled synthetic classroom snippets; not legal case data.",
        "task": "Six-way topic cue classification with token-level attention display.",
    }


def _build_model(vocab_size: int, class_count: int):
    import torch
    from torch import nn

    class GRUAttentionClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 24, padding_idx=0)
            self.gru = nn.GRU(24, 24, batch_first=True)
            self.attention = nn.Sequential(nn.Linear(24, 16), nn.Tanh(), nn.Linear(16, 1))
            self.classifier = nn.Linear(24, class_count)

        def forward(self, token_ids, mask):
            encoded, _ = self.gru(self.embedding(token_ids))
            scores = self.attention(encoded).squeeze(-1).masked_fill(mask == 0, -1e9)
            weights = torch.softmax(scores, dim=1)
            pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
            return self.classifier(pooled), weights

    return GRUAttentionClassifier()


def _encode(tokens: list[str], vocab: dict[str, int], max_length: int = 48) -> list[int]:
    return [vocab.get(token, vocab.get("<unk>", 1)) for token in tokens[:max_length]] or [1]


def _load_trained_model():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not WEIGHTS_PATH.is_file():
        raise RuntimeError("The GRU demo checkpoint is missing. Run the documented GRU training command first.")
    import torch

    torch.set_num_threads(1)
    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    vocab = {str(k): int(v) for k, v in checkpoint["vocab"].items()}
    labels = [str(label) for label in checkpoint["labels"]]
    model = _build_model(len(vocab), len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    _CACHE = (model, vocab, labels, float(checkpoint.get("validation_accuracy", 0.0)))
    return _CACHE


def classify_sequence(text: str) -> dict[str, Any]:
    """Classify a short sentence and return normalized token attention weights."""
    import torch

    tokens = [token.casefold() for token in _TOKEN_RE.findall(str(text or ""))][:48]
    if not tokens:
        raise ValueError("Enter a short sentence containing at least one word.")
    model, vocab, labels, validation_accuracy = _load_trained_model()
    ids = _encode(tokens, vocab)
    token_tensor = torch.tensor([ids], dtype=torch.long)
    mask = torch.ones_like(token_tensor)
    with torch.inference_mode():
        logits, attention = model(token_tensor, mask)
        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_index = int(torch.argmax(probabilities).item())
        weights = attention[0].tolist()
    return {
        "prediction": labels[predicted_index],
        "score": round(float(probabilities[predicted_index]), 4),
        "tokens": [{"text": token, "attention": round(float(weight), 4)} for token, weight in zip(tokens, weights)],
        "labels": labels,
        "validation_accuracy": round(validation_accuracy, 4),
        "model": "Small GRU with additive attention",
        "notice": "Teaching demo trained on 48 synthetic snippets with 12 held-out synthetic examples. Attention weights are model signals, not explanations, evidence, or legal confidence.",
    }
