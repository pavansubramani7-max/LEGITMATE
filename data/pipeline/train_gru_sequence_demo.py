"""Train the tiny synthetic GRU + attention teaching model on CPU."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.nlp.sequence_demo import CASES_PATH, WEIGHTS_PATH, _build_model

TOKEN_RE = re.compile(r"[\w]+(?:['’/-][\w]+)*", re.UNICODE)
MAX_LENGTH = 48


def tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)][:MAX_LENGTH]


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(3250)
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    labels = list(cases["labels"])
    label_ids = {label: index for index, label in enumerate(labels)}
    vocab = {"<pad>": 0, "<unk>": 1}
    for row in cases["training"]:
        for token in tokens(row["text"]):
            if token not in vocab:
                vocab[token] = len(vocab)

    def make_batch(rows):
        sequences = [[vocab.get(token, 1) for token in tokens(row["text"])] or [1] for row in rows]
        max_len = max(map(len, sequences))
        input_ids = torch.zeros((len(rows), max_len), dtype=torch.long)
        mask = torch.zeros_like(input_ids)
        for index, seq in enumerate(sequences):
            input_ids[index, :len(seq)] = torch.tensor(seq)
            mask[index, :len(seq)] = 1
        targets = torch.tensor([label_ids[row["label"]] for row in rows], dtype=torch.long)
        return input_ids, mask, targets

    train_x, train_mask, train_y = make_batch(cases["training"])
    val_x, val_mask, val_y = make_batch(cases["validation"])
    model = _build_model(len(vocab), len(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.025, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 120
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(train_x, train_mask)
        loss = loss_fn(logits, train_y)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        val_logits, _ = model(val_x, val_mask)
        val_accuracy = float((val_logits.argmax(dim=-1) == val_y).float().mean().item())
        train_logits, _ = model(train_x, train_mask)
        train_accuracy = float((train_logits.argmax(dim=-1) == train_y).float().mean().item())
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "labels": labels,
        "validation_accuracy": val_accuracy,
        "training_accuracy": train_accuracy,
        "training_examples": len(cases["training"]),
        "validation_examples": len(cases["validation"]),
        "epochs": epochs,
        "seed": 3250,
        "benchmark": cases["benchmark"],
    }, WEIGHTS_PATH)
    print(f"Model: GRU with additive attention; CPU training complete ({epochs} epochs).")
    print(f"Synthetic examples: {len(cases['training'])} train / {len(cases['validation'])} held-out validation.")
    print(f"Accuracy: train={train_accuracy:.3f}, held-out validation={val_accuracy:.3f}.")
    print(f"Saved checkpoint: {WEIGHTS_PATH.relative_to(ROOT)}")
    print("These metrics measure only the tiny synthetic classroom set, not legal-document performance.")


if __name__ == "__main__":
    main()
