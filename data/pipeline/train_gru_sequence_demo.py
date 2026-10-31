"""Train small synthetic GRU, CNN, vanilla RNN, and LSTM teaching demos on CPU."""
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

from app.nlp.sequence_demo import ARCHITECTURES, CASES_PATH, _build_model

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
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    for architecture, (label, filename) in ARCHITECTURES.items():
        torch.manual_seed(3250)
        model = _build_model(len(vocab), len(labels), architecture)
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
        checkpoint = ROOT / "models" / filename
        torch.save({
            "architecture": architecture,
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
        }, checkpoint)
        print(f"{label}: train={train_accuracy:.3f}, held-out synthetic validation={val_accuracy:.3f}; saved {checkpoint.relative_to(ROOT)}")
    print("Metrics describe only this tiny synthetic classroom set; none measure legal-document performance.")


if __name__ == "__main__":
    main()
