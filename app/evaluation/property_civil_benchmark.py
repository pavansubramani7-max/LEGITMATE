"""Offline retrieval benchmark for the bundled synthetic property/civil PDFs.

This is a small prototype benchmark, not external-domain validation. It reads
the sample PDFs directly and never writes to data/index.json or embeddings.
Gold query/evidence labels live in data/evaluation/property_civil_cases_v2.json.
Run from the project root with the virtual environment:
    .venv/Scripts/python.exe -m app.evaluation.property_civil_benchmark
Model loading is forced offline; no weights are downloaded.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from app.config import ROOT
from app.evidence.knowledge_graph import graph_augmented_rank
from app.extraction.entities import extract_entities
from app.extraction.dates import extract_dates
from app.extraction.claims import extract_claims
from app.ingestion.chunker import make_chunks
from app.ingestion.pdf_reader import extract_pages
from app.qa.answer import build_answer
from app.retrieval.hybrid import rank as hybrid_rank
from app.retrieval.reranker import rerank as rerank_hits, reranker_info
from app.retrieval.semantic import rank as semantic_rank, backend_info
from app.search.lexical import rank as bm25_rank

SAMPLE_DIR = ROOT / "data" / "sample"
CASE_FILE = ROOT / "data" / "evaluation" / "property_civil_cases_v2.json"
OUTPUT = ROOT / "data" / "processed" / "property_civil_benchmark" / "latest.json"
TOP_K = 5

# Human-auditable query and exact-span labels live in data/evaluation.  Load
# them separately from generated reports so benchmark runs never overwrite gold.
CASES: list[dict[str, Any]] = json.loads(CASE_FILE.read_text(encoding="utf-8"))


def _normal(text: Any) -> str:
    return " ".join(str(text or "").casefold().split())


def _load_sample_index() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunks: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for filename in ("sample_sale_deed.pdf", "sample_dispute_letter.pdf", "sample_lease_agreement.pdf"):
        path = SAMPLE_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Bundled benchmark file is missing: {path}")
        pages = extract_pages(path.read_bytes())
        document_id = f"benchmark:{path.stem}"
        for chunk in make_chunks(document_id, filename, pages):
            chunk["citation_unit"] = "page"
            chunks.append(chunk)
        for page in pages:
            page_items = extract_entities(document_id, filename, int(page["page"]), str(page["text"]))
            page_items += extract_dates(document_id, filename, int(page["page"]), str(page["text"]))
            page_items += extract_claims(document_id, filename, int(page["page"]), str(page["text"]))
            for item in page_items:
                item["citation_unit"] = "page"
            items.extend(page_items)
    return chunks, items


def _source_link(item: dict[str, Any], chunks: list[dict[str, Any]], *, strict_value: bool = False) -> dict[str, Any]:
    normalize = lambda text: " ".join(str(text or "").casefold().split())
    candidates = [chunk for chunk in chunks if chunk.get("document_id") == item.get("document_id") and chunk.get("page") == item.get("page")]
    value, context = normalize(item.get("value")), normalize(item.get("context"))
    needles = [(value, "claim_text" if strict_value else "value")]
    if not strict_value and context and context != value:
        needles.insert(0, (context, "context"))
    for needle, match_type in needles:
        if needle:
            for chunk in candidates:
                if needle in normalize(chunk.get("text")):
                    return {"source_match": True, "source_match_type": match_type, "source_chunk_id": chunk.get("chunk_id")}
    return {"source_match": False, "source_match_type": None, "source_chunk_id": None}


def _gold_ids(case: dict[str, Any], chunks: list[dict[str, Any]]) -> set[str]:
    relevant: set[str] = set()
    for filename, page, phrase in case["evidence"]:
        target = _normal(phrase)
        matches = [
            chunk for chunk in chunks
            if chunk.get("document") == filename
            and chunk.get("page") == page
            and target in _normal(chunk.get("text"))
        ]
        if not matches:
            raise ValueError(f"Gold evidence not found for {case['id']}: {filename}, page {page}, {phrase!r}")
        relevant.update(chunk["chunk_id"] for chunk in matches)
    return relevant


def _rank(mode: str, question: str, chunks: list[dict[str, Any]], items: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if mode == "bm25":
        return bm25_rank(question, chunks, TOP_K)
    if mode == "semantic":
        return semantic_rank(question, chunks, TOP_K)
    if mode == "hybrid":
        return hybrid_rank(question, chunks, TOP_K)
    if mode == "graph":
        return graph_augmented_rank(
            question, {"chunks": chunks, "items": items or []}, TOP_K,
            hybrid_rank=hybrid_rank, lexical_rank=bm25_rank, source_link=_source_link,
        )
    candidates = hybrid_rank(question, chunks, min(len(chunks), TOP_K * 2))
    return rerank_hits(question, candidates)[:TOP_K]


def _reciprocal_rank(ids: list[str], gold: set[str]) -> float:
    for position, chunk_id in enumerate(ids, start=1):
        if chunk_id in gold:
            return 1.0 / position
    return 0.0


def _ndcg(ids: list[str], gold: set[str]) -> float:
    dcg = sum(1.0 / math.log2(position + 1) for position, chunk_id in enumerate(ids, start=1) if chunk_id in gold)
    ideal_count = min(TOP_K, len(gold))
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_mode(mode: str, chunks: list[dict[str, Any]], items: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    positive_rows: list[dict[str, float]] = []
    negative_rows: list[dict[str, Any]] = []
    supported_answers: list[bool] = []
    cases_out: list[dict[str, Any]] = []
    category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in CASES:
        gold = _gold_ids(case, chunks)
        hits = _rank(mode, case["question"], chunks, items)
        hit_ids = [str(hit.get("chunk_id", "")) for hit in hits]
        answer = build_answer(case["question"], hits, conflicts=[])
        answer_ids = {str(item.get("chunk_id", "")) for item in answer.get("citations", [])}
        if gold:
            positive_rows.append({
                "hit_at_5": float(bool(set(hit_ids[:TOP_K]) & gold)),
                "mrr_at_5": _reciprocal_rank(hit_ids[:TOP_K], gold),
                "recall_at_5": len(set(hit_ids[:TOP_K]) & gold) / len(gold),
                "ndcg_at_5": _ndcg(hit_ids[:TOP_K], gold),
            })
            supported_answers.append(bool(answer_ids & gold))
        else:
            negative_rows.append({"abstained": bool(answer.get("abstained")), "top_score": float(hits[0].get("score", 0.0)) if hits else 0.0})
        category_rows[str(case.get("category", "uncategorized"))].append({
            "supported": bool(gold),
            "retrieval_hit": bool(set(hit_ids[:TOP_K]) & gold) if gold else None,
            "abstained": bool(answer.get("abstained")),
        })
        cases_out.append({
            "id": case["id"], "category": case.get("category", "uncategorized"), "question": case["question"],
            "gold_evidence_count": len(gold), "ranked": [
                {"document": hit.get("document"), "page": hit.get("page"), "chunk_id": hit.get("chunk_id"),
                 "score": hit.get("score"), "rerank_score": hit.get("rerank_score"),
                 "graph_path": hit.get("graph_path"),
                 "gold": str(hit.get("chunk_id", "")) in gold}
                for hit in hits
            ],
            "answer_abstained": bool(answer.get("abstained")),
            "answer_citation_supported": bool(answer_ids & gold) if gold else None,
        })
    mean = lambda key: round(sum(row[key] for row in positive_rows) / max(1, len(positive_rows)), 4)
    category_metrics = {}
    for category, rows in sorted(category_rows.items()):
        positive = [row for row in rows if row["supported"]]
        negative = [row for row in rows if not row["supported"]]
        category_metrics[category] = {
            "queries": len(rows), "supported_queries": len(positive), "unsupported_queries": len(negative),
            "supported_retrieval_hit_rate_at_5": round(sum(row["retrieval_hit"] for row in positive) / len(positive), 4) if positive else None,
            "unsupported_abstention_rate": round(sum(row["abstained"] for row in negative) / len(negative), 4) if negative else None,
        }
    return {
        "positive_queries": len(positive_rows),
        "negative_queries": len(negative_rows),
        "hit_rate_at_5": mean("hit_at_5"),
        "mrr_at_5": mean("mrr_at_5"),
        "recall_at_5": mean("recall_at_5"),
        "ndcg_at_5": mean("ndcg_at_5"),
        "supported_answer_rate": round(sum(supported_answers) / max(1, len(supported_answers)), 4),
        "negative_abstention_rate": round(sum(row["abstained"] for row in negative_rows) / max(1, len(negative_rows)), 4),
        "negative_false_answer_rate": round(sum(not row["abstained"] for row in negative_rows) / max(1, len(negative_rows)), 4),
        "by_category": category_metrics,
        "cases": cases_out,
    }


def compare_graph_to_hybrid(methods: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Paired, query-level GraphRAG ablation on the shared answerable set."""
    hybrid_cases = {row["id"]: row for row in methods["hybrid"]["cases"]}
    graph_cases = {row["id"]: row for row in methods["graph"]["cases"]}
    shared = [case for case in CASES if case["evidence"]]
    paired: list[dict[str, Any]] = []
    for case in shared:
        hrow, grow = hybrid_cases[case["id"]], graph_cases[case["id"]]
        hmrr = _reciprocal_rank([str(x["chunk_id"]) for x in hrow["ranked"]],
                                {str(x["chunk_id"]) for x in hrow["ranked"] if x["gold"]})
        gmrr = _reciprocal_rank([str(x["chunk_id"]) for x in grow["ranked"]],
                                {str(x["chunk_id"]) for x in grow["ranked"] if x["gold"]})
        paired.append({"id": case["id"], "hybrid_mrr_at_5": hmrr, "graph_mrr_at_5": gmrr,
                       "graph_path_in_top_5": any(x.get("graph_path") for x in grow["ranked"][:TOP_K])})
    deltas = [row["graph_mrr_at_5"] - row["hybrid_mrr_at_5"] for row in paired]
    return {
        "comparison": "Graph expansion versus hybrid retrieval on the same answerable synthetic queries",
        "answerable_queries": len(paired),
        "graph_mrr_better": sum(delta > 1e-12 for delta in deltas),
        "tied": sum(abs(delta) <= 1e-12 for delta in deltas),
        "graph_mrr_worse": sum(delta < -1e-12 for delta in deltas),
        "mean_mrr_at_5_delta_graph_minus_hybrid": round(sum(deltas) / len(deltas), 4) if deltas else None,
        "graph_path_present_in_top_5_queries": sum(row["graph_path_in_top_5"] for row in paired),
        "query_results": paired,
        "interpretation_limit": "Exploratory paired comparison on a small synthetic set; no confidence interval or real-world generalization is claimed.",
    }


def run() -> dict[str, Any]:
    chunks, items = _load_sample_index()
    # Validate every gold span before loading any model so a typo cannot look
    # like a model failure.
    for case in CASES:
        _gold_ids(case, chunks)
    backend = backend_info()
    methods = {mode: evaluate_mode(mode, chunks, items) for mode in ("bm25", "semantic", "hybrid", "graph", "reranked")}
    reranker = reranker_info()
    report = {
        "benchmark": "LEGITMATE synthetic property/civil retrieval benchmark",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus": "Three bundled synthetic PDFs; six pages; never reads or changes data/index.json.",
        "case_file": str(CASE_FILE.relative_to(ROOT)),
        "queries": len(CASES), "positive_queries": sum(bool(case["evidence"]) for case in CASES),
        "negative_queries": sum(not case["evidence"] for case in CASES),
        "top_k": TOP_K,
        "semantic_backend": backend,
        "reranker": reranker,
        "methods": methods,
        "graph_vs_hybrid_paired": compare_graph_to_hybrid(methods),
        "limitations": [
            "Synthetic examples are only a smoke benchmark; they do not establish real-world legal accuracy.",
            "Questions and gold passages are small and hand-authored, not a held-out property/civil dataset.",
            "The cross-encoder is a general passage ranker; its scores are not calibrated legal-confidence probabilities.",
            "Graph mode is a deterministic exact-entity expansion prototype; this synthetic benchmark does not establish entity-linking quality on real names or aliases.",
            "No model is fine-tuned by this script. Model weights are loaded offline only.",
        ],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Benchmark: {report['benchmark']}")
    print(f"Corpus: {report['corpus']}")
    print(f"Semantic model: {backend.get('model', backend.get('label', 'unknown'))}; reranker cached={reranker.get('cached')}, active={reranker.get('active')}")
    for mode, metrics in methods.items():
        print(f"{mode.upper():8s} MRR@5={metrics['mrr_at_5']:.3f} nDCG@5={metrics['ndcg_at_5']:.3f} hit@5={metrics['hit_rate_at_5']:.3f} supported={metrics['supported_answer_rate']:.3f} negative-abstain={metrics['negative_abstention_rate']:.3f}")
    comparison = report["graph_vs_hybrid_paired"]
    print(f"GRAPH vs HYBRID (paired): MRR better/tied/worse={comparison['graph_mrr_better']}/{comparison['tied']}/{comparison['graph_mrr_worse']}; mean delta={comparison['mean_mrr_at_5_delta_graph_minus_hybrid']}; graph path in top 5={comparison['graph_path_present_in_top_5_queries']}/{comparison['answerable_queries']}")
    print(f"Saved: {OUTPUT}")
    print("Caution: small synthetic benchmark; not legal-domain validation.")
    return report


if __name__ == "__main__":
    run()
