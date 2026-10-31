"""Fine-tune cached multilingual E5 on LegalBenchRAG-CUAD retrieval pairs.

The script uses one positive per unique query for training so valid evidence
spans for the same query are not incorrectly treated as in-batch negatives.
Validation and test use query-to-passage ranking over each split's distinct
positive passages. Non-matching passages are treated as negatives, so these
metrics are a useful benchmark but not a substitute for manually labeled
negative passages from LEGITMATE's target domain.

Run from the repository root with the project's virtual environment.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import random
import time
from collections import defaultdict

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed" / "cuad"
MODEL_OUT = ROOT / "models" / "e5-legal"
BASELINE_REPORT = PROCESSED / "base_retrieval_metrics.json"
BASE_MODEL = "intfloat/multilingual-e5-small"
BATCH_SIZE = 16
EPOCHS = 1
MAX_TRAIN_QUERIES = 512  # CPU-friendly pilot; increase only after reviewing its metrics.
WARMUP_FRAC = 0.1
SEED = 42


def load_jsonl(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required prepared split is missing: {path}")
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = [
        {"query": str(row["query"]).strip(), "positive": str(row["positive"]).strip()}
        for row in rows
        if row.get("query", "").strip() and row.get("positive", "").strip()
    ]
    if not rows:
        raise ValueError(f"Prepared split is empty: {path}")
    return rows


def _stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def make_ir_evaluator(rows: list[dict[str, str]], name: str) -> InformationRetrievalEvaluator:
    """Group multiple correct spans by query and rank the split's passage pool."""
    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    relevant: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        query_id = _stable_id(row["query"])
        passage_id = _stable_id(row["positive"])
        queries[query_id] = row["query"]
        corpus[passage_id] = row["positive"]
        relevant[query_id].add(passage_id)

    if len(queries) < 2 or len(corpus) < 2:
        raise ValueError(f"{name} split needs at least two distinct queries and passages")

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=dict(relevant),
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 3, 5, 10],
        precision_recall_at_k=[1, 3, 5, 10],
        map_at_k=[10],
        batch_size=32,
        name=name,
        write_csv=False,
        query_prompt="query: ",
        corpus_prompt="passage: ",
    )


def _evaluate(model: SentenceTransformer, evaluator: InformationRetrievalEvaluator) -> dict[str, float]:
    return {key: float(value) for key, value in evaluator(model).items()}


def _metric(metrics: dict[str, float], metric_name: str) -> float:
    suffix = metric_name.casefold()
    for key, value in metrics.items():
        if key.casefold().endswith(suffix):
            return value
    return float("nan")


def _unique_query_examples(rows: list[dict[str, str]]) -> list[InputExample]:
    positives_by_query: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        positives_by_query[row["query"]].append(row["positive"])

    rng = random.Random(SEED)
    examples = []
    for query, positives in sorted(positives_by_query.items()):
        # A single positive per query prevents other valid spans for the same
        # query from becoming false negatives in MultipleNegativesRankingLoss.
        positive = sorted(set(positives))[rng.randrange(len(set(positives)))]
        examples.append(InputExample(texts=[f"query: {query}", f"passage: {positive}"]))
    if len(examples) < 2:
        raise ValueError("Training requires at least two distinct query examples")
    return examples


def main() -> None:
    train_rows = load_jsonl(PROCESSED / "train.jsonl")
    val_rows = load_jsonl(PROCESSED / "val.jsonl")
    test_rows = load_jsonl(PROCESSED / "test.jsonl")
    all_train_examples = _unique_query_examples(train_rows)
    if len(all_train_examples) > MAX_TRAIN_QUERIES:
        rng = random.Random(SEED)
        train_examples = rng.sample(all_train_examples, MAX_TRAIN_QUERIES)
    else:
        train_examples = all_train_examples
    val_evaluator = make_ir_evaluator(val_rows, "cuad_val")
    test_evaluator = make_ir_evaluator(test_rows, "cuad_test")

    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    existing_weights = list(MODEL_OUT.glob("**/model.safetensors")) + list(MODEL_OUT.glob("**/pytorch_model.bin"))
    if existing_weights:
        raise FileExistsError(
            f"Refusing to overwrite an existing fine-tuned model in {MODEL_OUT}. "
            "Back it up or choose a new output path first."
        )

    print(
        f"Training rows: {len(train_rows)}; unique queries available: {len(all_train_examples)}; "
        f"pilot examples used: {len(train_examples)}"
    )
    print(f"Validation rows/queries/passages: {len(val_rows)}/{len(val_evaluator.queries)}/{len(val_evaluator.corpus)}")
    print(f"Test rows/queries/passages: {len(test_rows)}/{len(test_evaluator.queries)}/{len(test_evaluator.corpus)}")
    print("Loading cached base model in offline mode...")
    model = SentenceTransformer(BASE_MODEL, local_files_only=True)

    if BASELINE_REPORT.is_file():
        baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
        base_val = baseline["validation"]
        base_test = baseline["test"]
        print(f"Loaded saved base-model metrics from {BASELINE_REPORT}")
    else:
        print("Measuring unchanged base model on validation and test splits...")
        base_val = _evaluate(model, val_evaluator)
        base_test = _evaluate(model, test_evaluator)
        BASELINE_REPORT.write_text(
            json.dumps({"validation": base_val, "test": base_test}, indent=2),
            encoding="utf-8",
        )
    print(f"Base validation MRR@10: {_metric(base_val, 'mrr@10'):.4f}")
    print(f"Base test MRR@10: {_metric(base_test, 'mrr@10'):.4f}")

    loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=BATCH_SIZE,
        collate_fn=lambda batch: batch,
        generator=torch.Generator().manual_seed(SEED),
    )
    loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = max(1, math.ceil(len(loader) * EPOCHS * WARMUP_FRAC))
    total_steps = len(loader) * EPOCHS
    print(f"Fine-tuning for {EPOCHS} epochs; batch={BATCH_SIZE}; warmup={warmup_steps}")
    started = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    def lr_scale(step: int) -> float:
        if step < warmup_steps:
            return max(1e-6, (step + 1) / warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    epoch_losses: list[float] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for batch in loader:
            query_texts = [example.texts[0] for example in batch]
            passage_texts = [example.texts[1] for example in batch]
            features = [
                model.preprocess(query_texts),
                model.preprocess(passage_texts),
            ]
            features = [
                {
                    key: value.to(model.device) if isinstance(value, torch.Tensor) else value
                    for key, value in feature.items()
                }
                for feature in features
            ]
            optimizer.zero_grad(set_to_none=True)
            batch_loss = loss(features, labels=None)
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(batch_loss.detach().cpu())

        average_loss = running_loss / max(1, len(loader))
        epoch_losses.append(average_loss)
        print(f"Epoch {epoch}/{EPOCHS}: loss={average_loss:.4f}")

    elapsed = time.time() - started
    model.save(str(MODEL_OUT))
    tuned = SentenceTransformer(str(MODEL_OUT), local_files_only=True)
    print("Evaluating fine-tuned model on validation and held-out test splits...")
    tuned_val = _evaluate(tuned, val_evaluator)
    tuned_test = _evaluate(tuned, test_evaluator)
    report = {
        "base_model": BASE_MODEL,
        "dataset": "adityaraghuvanshi999/legalbenchrag-cuad-retrieval-dataset",
        "license": "CC BY 4.0",
        "training_objective": "MultipleNegativesRankingLoss with one positive per unique query",
        "evaluation_method": "query-grouped ranking over unique positive passages within each held-out split; non-matching passages treated as negatives",
        "train_pairs": len(train_rows),
        "train_unique_queries_available": len(all_train_examples),
        "train_unique_queries_used": len(train_examples),
        "training_scope": "CPU-friendly 512-query one-epoch pilot; not a production-ready fine-tune",
        "validation_pairs": len(val_rows),
        "test_pairs": len(test_rows),
        "epochs": EPOCHS,
        "epoch_average_losses": epoch_losses,
        "batch_size": BATCH_SIZE,
        "warmup_steps": warmup_steps,
        "elapsed_seconds": round(elapsed, 1),
        "base_validation_metrics": base_val,
        "tuned_validation_metrics": tuned_val,
        "base_test_metrics": base_test,
        "tuned_test_metrics": tuned_test,
        "output": str(MODEL_OUT),
        "caveat": "CUAD contract retrieval is not a substitute for evaluation on property/civil documents; manually labeled target-domain queries and negative passages are still needed.",
    }
    report_path = MODEL_OUT / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nTraining completed in {elapsed / 60:.1f} minutes")
    print(f"Model saved to {MODEL_OUT}")
    print(f"Training report saved to {report_path}")
    print(f"Base test MRR@10: {_metric(base_test, 'mrr@10'):.4f}")
    print(f"Fine-tuned test MRR@10: {_metric(tuned_test, 'mrr@10'):.4f}")
    print(f"Base test NDCG@10: {_metric(base_test, 'ndcg@10'):.4f}")
    print(f"Fine-tuned test NDCG@10: {_metric(tuned_test, 'ndcg@10'):.4f}")
    print("Keep the fine-tuned model out of production/default retrieval until target-domain evaluation confirms an improvement.")


if __name__ == "__main__":
    main()
