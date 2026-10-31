"""Train a research-only E5 candidate on CUAD plus Indian legal QA pairs.

Never changes the app's default encoder. Requires a GPU unless --allow-cpu is
explicitly supplied. The QA answers are weak supervision, not cited evidence.

Example (after preparing data):
  python data/pipeline/finetune_e5_kaggle_mix.py --allow-model-download
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import time
from collections import defaultdict

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CUAD = ROOT / "data" / "processed" / "cuad"
DEFAULT_INDQA = ROOT / "data" / "processed" / "kaggle_legal" / "indian_legal_qa"
DEFAULT_OUT = ROOT / "models" / "e5-kaggle-mix-candidate"
BASE_MODEL = "intfloat/multilingual-e5-small"
SEED = 42
BATCH_SIZE = 16


def _load(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared training split not found: {path}. Run prepare_kaggle_legal.py first.")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            query, positive = str(row.get("query", "")).strip(), str(row.get("positive", "")).strip()
            if query and positive:
                rows.append({"query": query, "positive": positive})
    if not rows:
        raise ValueError(f"No query-positive pairs found in {path}")
    return rows


def _id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def _evaluator(rows: list[dict[str, str]], name: str) -> InformationRetrievalEvaluator:
    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    relevant: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        qid, pid = _id(row["query"]), _id(row["positive"])
        queries[qid] = row["query"]
        corpus[pid] = row["positive"]
        relevant[qid].add(pid)
    if len(queries) < 2 or len(corpus) < 2:
        raise ValueError(f"{name} split has too few distinct queries/passages")
    return InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=dict(relevant),
        mrr_at_k=[10], ndcg_at_k=[10], accuracy_at_k=[1, 3, 5, 10],
        precision_recall_at_k=[1, 3, 5, 10], map_at_k=[10], batch_size=32,
        name=name, write_csv=False, query_prompt="query: ", corpus_prompt="passage: ",
    )


def _evaluate(model: SentenceTransformer, evaluator: InformationRetrievalEvaluator) -> dict[str, float]:
    return {str(k): float(v) for k, v in evaluator(model).items()}


def _metric(metrics: dict[str, float], end: str) -> float:
    return next((v for k, v in metrics.items() if k.casefold().endswith(end.casefold())), float("nan"))


def _examples(cuad: list[dict[str, str]], indqa: list[dict[str, str]], maximum: int) -> list[InputExample]:
    rng = random.Random(SEED)
    # Keep contract retrieval dominant; the Indian QA answers are weaker labels.
    qa_cap = min(len(indqa), max(1, maximum // 5))
    legal_cap = min(len(cuad), maximum - qa_cap)
    selected = rng.sample(cuad, legal_cap) + rng.sample(indqa, qa_cap)
    if len(selected) < maximum and len(cuad) > legal_cap:
        remaining = [r for r in cuad if r not in selected]
        selected.extend(rng.sample(remaining, min(maximum - len(selected), len(remaining))))
    rng.shuffle(selected)
    seen: set[str] = set()
    result = []
    for row in selected:
        if row["query"] in seen:
            continue
        seen.add(row["query"])
        result.append(InputExample(texts=[f"query: {row['query']}", f"passage: {row['positive']}"]))
    if len(result) < 2:
        raise ValueError("Need at least two distinct training queries")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuad-dir", type=pathlib.Path, default=DEFAULT_CUAD)
    parser.add_argument("--indqa-dir", type=pathlib.Path, default=DEFAULT_INDQA)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--max-train-queries", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--allow-cpu", action="store_true", help="Allow a slow CPU run; GPU is recommended.")
    parser.add_argument("--allow-model-download", action="store_true", help="Permit downloading the base model if not cached.")
    args = parser.parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("No CUDA GPU is available. Use Kaggle GPU or pass --allow-cpu only if you accept a much slower run.")
    if args.max_train_queries < 2 or args.epochs < 1:
        raise SystemExit("Training needs at least 2 queries and 1 epoch.")

    train_cuad = _load(args.cuad_dir / "train.jsonl")
    train_indqa = _load(args.indqa_dir / "train.jsonl")
    val_cuad = _load(args.cuad_dir / "val.jsonl")
    test_cuad = _load(args.cuad_dir / "test.jsonl")
    val_indqa = _load(args.indqa_dir / "val.jsonl")
    test_indqa = _load(args.indqa_dir / "test.jsonl")
    val_eval = {"cuad": _evaluator(val_cuad, "cuad_val"), "indian_legal_qa": _evaluator(val_indqa, "indqa_val")}
    test_eval = {"cuad": _evaluator(test_cuad, "cuad_test"), "indian_legal_qa": _evaluator(test_indqa, "indqa_test")}
    examples = _examples(train_cuad, train_indqa, args.max_train_queries)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty candidate directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(BASE_MODEL, device=device, local_files_only=not args.allow_model_download)

    print(f"Device: {device}; CUAD pairs: {len(train_cuad)}; India QA weak pairs: {len(train_indqa)}; selected: {len(examples)}")
    print("Evaluating unchanged base encoder on each held-out dataset...")
    base_val = {name: _evaluate(model, evaluator) for name, evaluator in val_eval.items()}
    base_test = {name: _evaluate(model, evaluator) for name, evaluator in test_eval.items()}

    loader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE, collate_fn=lambda batch: batch,
                        generator=torch.Generator().manual_seed(SEED))
    objective = losses.MultipleNegativesRankingLoss(model)
    warmup = max(1, math.ceil(len(loader) * args.epochs * 0.1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=max(1, warmup))
    started = time.time()
    epoch_losses = []
    for epoch in range(args.epochs):
        model.train(); loss_sum = 0.0
        for batch in loader:
            q = [ex.texts[0] for ex in batch]
            p = [ex.texts[1] for ex in batch]
            features = [model.preprocess(q), model.preprocess(p)]
            features = [{k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in f.items()} for f in features]
            optimizer.zero_grad(set_to_none=True)
            loss = objective(features, labels=None)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step(); scheduler.step()
            loss_sum += float(loss.detach().cpu())
        epoch_losses.append(loss_sum / max(1, len(loader)))
        print(f"Epoch {epoch + 1}/{args.epochs}: mean loss={epoch_losses[-1]:.4f}")

    model.save(str(args.output_dir))
    candidate = SentenceTransformer(str(args.output_dir), device=device, local_files_only=True)
    tuned_val = {name: _evaluate(candidate, evaluator) for name, evaluator in val_eval.items()}
    tuned_test = {name: _evaluate(candidate, evaluator) for name, evaluator in test_eval.items()}
    report = {
        "base_model": BASE_MODEL,
        "candidate_model": str(args.output_dir),
        "training": {"device": device, "epochs": args.epochs, "selected_queries": len(examples),
                     "batch_size": BATCH_SIZE, "loss": "MultipleNegativesRankingLoss", "epoch_mean_loss": epoch_losses,
                     "elapsed_seconds": round(time.time() - started, 1), "qa_mix": "up to 20% Indian QA weak answer pairs; remaining examples sampled from CUAD"},
        "datasets": {
            "cuad": {"license": "CC BY 4.0", "train_pairs": len(train_cuad), "val_pairs": len(val_cuad), "test_pairs": len(test_cuad)},
            "indian_legal_qa": {"license": "CC BY 4.0", "train_pairs": len(train_indqa), "val_pairs": len(val_indqa), "test_pairs": len(test_indqa),
                                 "caveat": "Q/A answers are weak labels, not verified source passages."},
        },
        "base_validation": base_val, "candidate_validation": tuned_val,
        "base_test": base_test, "candidate_test": tuned_test,
        "promotion_policy": "Research candidate only. Do not change the app default unless held-out test MRR@10/NDCG@10 improve on CUAD without an unacceptable regression on Indian QA and a separate property/civil gold set.",
    }
    (args.output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"base_test_mrr10": {k: round(_metric(v, "mrr@10"), 4) for k,v in base_test.items()},
                      "candidate_test_mrr10": {k: round(_metric(v, "mrr@10"), 4) for k,v in tuned_test.items()},
                      "report": str(args.output_dir / "training_report.json")}, indent=2))
    print("Candidate remains separate from the app's default model; evaluate it before promotion.")


if __name__ == "__main__":
    main()
