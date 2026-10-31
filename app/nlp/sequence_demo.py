"""Small local neural architecture comparison demo for teaching only."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.config import ROOT

CASES_PATH = ROOT / "data" / "evaluation" / "sequence_demo_cases_v1.json"
ARCHITECTURES = {
    "gru": ("GRU", "gru-sequence-demo.pt"),
    "cnn": ("CNN", "cnn-sequence-demo.pt"),
    "rnn": ("vanilla RNN", "rnn-sequence-demo.pt"),
    "lstm": ("LSTM", "lstm-sequence-demo.pt"),
}
WEIGHTS_PATH = ROOT / "models" / ARCHITECTURES["gru"][1]
_TOKEN_RE = re.compile(r"[\w]+(?:['’/-][\w]+)*", re.UNICODE)
_CACHE: dict[str, tuple[Any, dict[str, int], list[str], float]] | None = None


def _load_cases() -> dict[str, Any]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def model_status() -> dict[str, Any]:
    models = []
    for key, (label, filename) in ARCHITECTURES.items():
        path = ROOT / "models" / filename
        models.append({"architecture": key, "name": label, "trained_weights": path.is_file(), "model_path": str(path.relative_to(ROOT)) if path.is_file() else None})
    return {
        "name": "Neural sequence architecture teaching demos",
        "trained_weights": all(item["trained_weights"] for item in models),
        "task_connected": any(item["trained_weights"] for item in models),
        "architectures": models,
        "training_data": "Hand-labeled synthetic classroom snippets; not legal case data.",
        "task": "Six-way topic cue classification; GRU includes additive token-attention display.",
    }


def _build_model(vocab_size: int, class_count: int, architecture: str = "gru"):
    import torch
    from torch import nn

    if architecture not in ARCHITECTURES:
        raise ValueError("Choose GRU, CNN, vanilla RNN, or LSTM.")

    class SequenceClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 24, padding_idx=0)
            if architecture == "gru":
                self.encoder = nn.GRU(24, 24, batch_first=True)
            elif architecture == "rnn":
                self.encoder = nn.RNN(24, 24, batch_first=True, nonlinearity="tanh")
            elif architecture == "lstm":
                self.encoder = nn.LSTM(24, 24, batch_first=True)
            else:
                self.encoder = nn.Conv1d(24, 24, kernel_size=3, padding=1)
            self.attention = nn.Sequential(nn.Linear(24, 16), nn.Tanh(), nn.Linear(16, 1))
            self.classifier = nn.Linear(24, class_count)

        def forward(self, token_ids, mask):
            embedded = self.embedding(token_ids)
            if architecture == "cnn":
                encoded = torch.nn.functional.gelu(self.encoder(embedded.transpose(1, 2))).transpose(1, 2)
            else:
                encoded, _ = self.encoder(embedded)
            scores = self.attention(encoded).squeeze(-1).masked_fill(mask == 0, -1e9)
            weights = torch.softmax(scores, dim=1)
            pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
            return self.classifier(pooled), weights

    return SequenceClassifier()


def _encode(tokens: list[str], vocab: dict[str, int], max_length: int = 48) -> list[int]:
    return [vocab.get(token, vocab.get("<unk>", 1)) for token in tokens[:max_length]] or [1]


def _load_trained_model(architecture: str = "gru"):
    global _CACHE
    cache_key = architecture
    if isinstance(_CACHE, dict) and cache_key in _CACHE:
        return _CACHE[cache_key]
    if architecture not in ARCHITECTURES:
        raise ValueError("Choose GRU, CNN, vanilla RNN, or LSTM.")
    weights_path = ROOT / "models" / ARCHITECTURES[architecture][1]
    if not weights_path.is_file():
        raise RuntimeError(f"The {ARCHITECTURES[architecture][0]} demo checkpoint is missing. Run the sequence-demo training command.")
    import torch

    torch.set_num_threads(1)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    vocab = {str(k): int(v) for k, v in checkpoint["vocab"].items()}
    labels = [str(label) for label in checkpoint["labels"]]
    model = _build_model(len(vocab), len(labels), architecture)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    result = (model, vocab, labels, float(checkpoint.get("validation_accuracy", 0.0)))
    if not isinstance(_CACHE, dict):
        _CACHE = {}
    _CACHE[cache_key] = result
    return result


def classify_sequence(text: str, architecture: str = "gru") -> dict[str, Any]:
    """Classify a short sentence and return normalized token attention weights."""
    import torch

    tokens = [token.casefold() for token in _TOKEN_RE.findall(str(text or ""))][:48]
    if not tokens:
        raise ValueError("Enter a short sentence containing at least one word.")
    model, vocab, labels, validation_accuracy = _load_trained_model(architecture)
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
        "architecture": architecture,
        "model": f"Small {ARCHITECTURES[architecture][0]} sequence classifier" + (" with additive attention" if architecture == "gru" else " with attention pooling"),
        "notice": "Teaching demo trained on a small synthetic set with a held-out synthetic validation split. Attention weights are model signals, not explanations, evidence, or legal confidence; these are not legal classifiers.",
    }
