"""Evaluate LEGITMATE against independently adjudicated JSONL page records.

This evaluator reports aggregate metrics only; it does not write source text,
gold spans, or extracted values to its output report. Keep private documents
and reviewed labels access-controlled.

Run from the project root:
  .venv/Scripts/python.exe -m app.evaluation.reviewed_corpus_benchmark --input path/to/test.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from app.config import ROOT
from app.evidence.knowledge_graph import graph_augmented_rank
from app.extraction.claims import extract_claims
from app.extraction.dates import extract_dates
from app.extraction.entities import extract_entities
from app.ingestion.chunker import make_chunks
from app.nlp.lab import evaluate_text
from app.qa.answer import build_answer
from app.retrieval.hybrid import rank as hybrid_rank
from app.search.lexical import rank as bm25_rank

_KINDS = ("person", "org", "property", "date", "amount", "claim")
_DEFAULT_REPORT = ROOT / "data" / "processed" / "reviewed_corpus" / "latest.json"


def _norm(value: Any) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(value or "").casefold()).split())


def _read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
        _validate_record(record, line_number)
        records.append(record)
    if not records:
        raise ValueError("The annotation file contains no records.")
    case_ids = [str(record["case_id"]) for record in records]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_id values must be unique within the annotation file.")
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for record in records:
        splits_by_group[str(record["document_group"])].add(str(record["split"]))
    leaked = [group for group, splits in splits_by_group.items() if len(splits) > 1]
    if leaked:
        raise ValueError("Matter leakage across splits: one document_group appears in multiple splits.")
    if not any(record["split"] == "test" for record in records):
        raise ValueError("No test records found. Keep all pages from a matter in one split and mark held-out records split='test'.")
    return records


def _validate_record(record: Any, line_number: int) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"Line {line_number} must contain a JSON object.")
    required = ("case_id", "document_group", "split", "source_file", "source_sha256", "page", "language", "ocr_status", "text", "annotators", "adjudicated_by")
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"Line {line_number} is missing required fields: {', '.join(missing)}")
    for field in ("case_id", "document_group", "source_file", "language", "adjudicated_by"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"Line {line_number}: {field} must be a non-empty string.")
    if record["split"] not in {"train", "dev", "test"}:
        raise ValueError(f"Line {line_number}: split must be train, dev, or test.")
    if record["ocr_status"] not in {"native_text", "ocr", "mixed"}:
        raise ValueError(f"Line {line_number}: ocr_status must be native_text, ocr, or mixed.")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(record["source_sha256"])):
        raise ValueError(f"Line {line_number}: source_sha256 must be a 64-character SHA-256 hex digest.")
    if not isinstance(record["text"], str) or not record["text"].strip():
        raise ValueError(f"Line {line_number}: text must contain the source page/section text.")
    if not isinstance(record["annotators"], list) or any(not isinstance(value, str) or not value.strip() for value in record["annotators"]) or len(set(record["annotators"])) < 2:
        raise ValueError(f"Line {line_number}: at least two independent annotator IDs are required.")
    if not str(record["adjudicated_by"]).strip():
        raise ValueError(f"Line {line_number}: adjudicated_by is required.")
    if not isinstance(record["page"], int) or record["page"] < 1:
        raise ValueError(f"Line {line_number}: page/section number must be a positive integer.")
    for field in ("entities", "dates", "amounts", "claims", "intervals", "retrieval_queries", "answer_references"):
        if field in record and not isinstance(record[field], list):
            raise ValueError(f"Line {line_number}: {field} must be a list.")
    for field in ("entities", "dates", "amounts", "claims"):
        for annotation in record.get(field, []):
            if not isinstance(annotation, dict):
                raise ValueError(f"Line {line_number}: each {field} annotation must be an object.")
            start, end, value = annotation.get("start"), annotation.get("end"), annotation.get("text")
            if not (isinstance(start, int) and isinstance(end, int) and isinstance(value, str)
                    and 0 <= start < end <= len(record["text"]) and record["text"][start:end] == value):
                raise ValueError(f"Line {line_number}: {field} annotations need exact text with valid Unicode offsets.")
    for entity in record.get("entities", []):
        if entity.get("kind") not in {"person", "org", "property"}:
            raise ValueError(f"Line {line_number}: entity kind must be person, org, or property.")
    for interval in record.get("intervals", []):
        if not isinstance(interval, dict):
            raise ValueError(f"Line {line_number}: each interval must be an object.")
        if interval.get("relation") not in {"from_to", "between_and"}:
            raise ValueError(f"Line {line_number}: interval relation must be from_to or between_and.")
        for field in ("start_span", "end_span"):
            span = interval.get(field, [])
            if not (isinstance(span, list) and len(span) == 2 and all(isinstance(x, int) for x in span)
                    and 0 <= span[0] < span[1] <= len(record["text"])):
                raise ValueError(f"Line {line_number}: interval {field} must be a valid [start, end] text span.")
    for query in record.get("retrieval_queries", []):
        if not isinstance(query, dict) or not str(query.get("question", "")).strip():
            raise ValueError(f"Line {line_number}: each retrieval query needs a question.")
        if not isinstance(query.get("supported"), bool):
            raise ValueError(f"Line {line_number}: every retrieval query needs an adjudicated boolean supported label.")
        for start, end in query.get("evidence_spans", []):
            if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(record["text"])):
                raise ValueError(f"Line {line_number}: a retrieval evidence span is outside the source text.")
        if query.get("supported") is False and query.get("evidence_spans"):
            raise ValueError(f"Line {line_number}: unsupported queries cannot include gold evidence spans.")
        if query.get("supported") is True and not query.get("evidence_spans"):
            raise ValueError(f"Line {line_number}: supported queries need adjudicated gold evidence spans.")
    query_ids = [str(query.get("question_id", "")) for query in record.get("retrieval_queries", [])]
    if any(not query_id for query_id in query_ids) or len(query_ids) != len(set(query_ids)):
        raise ValueError(f"Line {line_number}: retrieval question_id values must be present and unique within a page record.")
    for reference in record.get("answer_references", []):
        if not isinstance(reference, dict):
            raise ValueError(f"Line {line_number}: each answer reference must be an object.")
        if not str(reference.get("question_id", "")) or not str(reference.get("reference", "")).strip():
            raise ValueError(f"Line {line_number}: answer references need question_id and reference text.")
        if str(reference["question_id"]) not in query_ids:
            raise ValueError(f"Line {line_number}: answer reference question_id must match a retrieval query on this record.")


def _gold_values(record: dict[str, Any]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {kind: set() for kind in _KINDS}
    for entity in record.get("entities", []):
        if entity.get("kind") in {"person", "org", "property"}:
            values[entity["kind"]].add(_norm(entity.get("text")))
    for kind, field in (("date", "dates"), ("amount", "amounts"), ("claim", "claims")):
        values[kind].update(_norm(item.get("text")) for item in record.get(field, []) if item.get("text"))
    return values


def _predicted_values(items: list[dict[str, Any]]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {kind: set() for kind in _KINDS}
    for item in items:
        kind = item.get("kind")
        if kind in values:
            value = _norm(item.get("value"))
            if value:
                values[kind].add(value)
    return values


def _score_sets(predicted: set[str], expected: set[str]) -> dict[str, Any]:
    tp = len(predicted & expected)
    fp, fn = len(predicted - expected), len(expected - predicted)
    precision = tp / (tp + fp) if tp + fp else (0.0 if expected else None)
    recall = tp / (tp + fn) if tp + fn else (0.0 if predicted else None)
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ) if precision is not None and recall is not None else None
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
    }


def _source_link(item: dict[str, Any], chunks: list[dict[str, Any]], *, strict_value: bool = False) -> dict[str, Any]:
    candidates = [c for c in chunks if c.get("document_id") == item.get("document_id") and c.get("page") == item.get("page")]
    value = _norm(item.get("value"))
    for chunk in candidates:
        if value and value in _norm(chunk.get("text")):
            return {"source_match": True, "source_chunk_id": chunk.get("chunk_id")}
    return {"source_match": False, "source_chunk_id": None}


def _retrieve(mode: str, question: str, chunks: list[dict[str, Any]], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if mode == "bm25":
        return bm25_rank(question, chunks, 5)
    if mode == "hybrid":
        return hybrid_rank(question, chunks, 5)
    if mode == "graph":
        return graph_augmented_rank(question, {"chunks": chunks, "items": items}, 5, hybrid_rank, bm25_rank, _source_link)
    raise ValueError(f"Unsupported mode {mode!r}; use bm25, hybrid, or graph.")


def _gold_chunks(query: dict[str, Any], record: dict[str, Any], chunks: list[dict[str, Any]], document_id: str) -> set[str]:
    expected = [record["text"][start:end] for start, end in query.get("evidence_spans", [])]
    if not expected:
        return set()
    return {
        str(chunk["chunk_id"])
        for chunk in chunks
        if chunk.get("document_id") == document_id and chunk.get("page") == record["page"]
        and any(_norm(span) in _norm(chunk.get("text")) for span in expected if _norm(span))
    }


def _query_rows(records: list[dict[str, Any]], chunks: list[dict[str, Any]], record_doc: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        doc_id = record_doc[str(record["case_id"])]
        for query in record.get("retrieval_queries", []):
            supported = bool(query["supported"])
            gold = _gold_chunks(query, record, chunks, doc_id)
            if supported and not gold:
                raise ValueError(f"Gold evidence did not map to a text chunk for query {query.get('question_id', record['case_id'])}.")
            rows.append({"record": record, "query": query, "supported": supported, "gold": gold, "doc_id": doc_id})
    return rows


def _calibrate_threshold(mode: str, rows: list[dict[str, Any]], chunks: list[dict[str, Any]], items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose a score threshold on dev queries only; never use test labels here."""
    if not rows or not any(row["supported"] for row in rows) or not any(not row["supported"] for row in rows):
        return None
    scored = []
    for row in rows:
        hits = _retrieve(mode, str(row["query"]["question"]), chunks, items)
        quality = max((float(hit.get("best_underlying_score", hit.get("score", 0.0)) or 0.0) for hit in hits), default=0.0)
        scored.append((quality, bool(row["supported"])))
    scores = [score for score, _ in scored]
    candidates = sorted({0.0, *scores, min(scores) - 1e-12, max(scores) + 1e-12})
    best: tuple[float, float] | None = None
    for threshold in candidates:
        positive = [score >= threshold for score, supported in scored if supported]
        negative = [score < threshold for score, supported in scored if not supported]
        balanced_accuracy = (sum(positive) / len(positive) + sum(negative) / len(negative)) / 2
        candidate = (round(balanced_accuracy, 8), threshold)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return {"threshold": best[1], "dev_balanced_accuracy": best[0],
            "dev_supported_queries": sum(supported for _, supported in scored),
            "dev_unsupported_queries": sum(not supported for _, supported in scored),
            "selection": "balanced answerability accuracy; ties prefer the more conservative higher threshold"}


def _run(input_path: Path, output_path: Path, modes: list[str]) -> dict[str, Any]:
    records = _read_records(input_path)
    test_records = [record for record in records if record["split"] == "test"]
    dev_records = [record for record in records if record["split"] == "dev"]
    chunks_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    items_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    record_doc: dict[str, str] = {}
    extraction_by_stratum: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(lambda: {k: {"pred": set(), "gold": set()} for k in _KINDS})
    temporal_by_stratum: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "date_normalization_correct": 0, "date_normalization_total": 0,
        "event_type_correct": 0, "event_type_total": 0,
        "interval_pred": set(), "interval_gold": set(),
    })

    for record in records:
        doc_id = f"{record['document_group']}::{record['source_file']}"
        record_doc[str(record["case_id"])] = doc_id
        page = {"page": int(record["page"]), "text": record["text"]}
        chunks_by_split[str(record["split"])].extend(make_chunks(doc_id, str(record["source_file"]), [page]))
        found = extract_entities(doc_id, str(record["source_file"]), int(record["page"]), record["text"])
        found += extract_dates(doc_id, str(record["source_file"]), int(record["page"]), record["text"])
        found += extract_claims(doc_id, str(record["source_file"]), int(record["page"]), record["text"])
        for item in found:
            item["citation_unit"] = record.get("citation_unit", "page")
        items_by_split[str(record["split"])].extend(found)
        if record["split"] != "test":
            continue
        pred, gold = _predicted_values(found), _gold_values(record)
        pred = {kind: {f"{record['case_id']}\u0000{value}" for value in values} for kind, values in pred.items()}
        gold = {kind: {f"{record['case_id']}\u0000{value}" for value in values} for kind, values in gold.items()}
        strata = {"language": str(record["language"]), "ocr_status": str(record["ocr_status"])}
        for dimension, label in strata.items():
            bucket = extraction_by_stratum[f"{dimension}:{label}"]
            for kind in _KINDS:
                bucket[kind]["pred"].update(pred[kind])
                bucket[kind]["gold"].update(gold[kind])
        bucket = extraction_by_stratum["overall"]
        for kind in _KINDS:
            bucket[kind]["pred"].update(pred[kind])
            bucket[kind]["gold"].update(gold[kind])

        by_surface = {_norm(item.get("value")): item for item in found if item.get("kind") == "date"}
        interval_gold = set()
        for date in record.get("dates", []):
            if not date.get("text"):
                continue
            predicted = by_surface.get(_norm(date["text"]))
            buckets = [temporal_by_stratum["overall"], *(temporal_by_stratum[f"{dimension}:{label}"] for dimension, label in strata.items())]
            for bucket in buckets:
                if date.get("normalized"):
                    bucket["date_normalization_total"] += 1
                    bucket["date_normalization_correct"] += int(bool(predicted) and str(predicted.get("event_date")) == str(date["normalized"]))
                if date.get("event_type"):
                    bucket["event_type_total"] += 1
                    bucket["event_type_correct"] += int(bool(predicted) and str(predicted.get("event_type")) == str(date["event_type"]))
        for interval in record.get("intervals", []):
            start_span, end_span = interval.get("start_span"), interval.get("end_span")
            if start_span and end_span:
                start_text = record["text"][int(start_span[0]):int(start_span[1])]
                end_text = record["text"][int(end_span[0]):int(end_span[1])]
                interval_gold.add(f"{record['case_id']}\u0000{_norm(start_text)}\u0000{_norm(end_text)}\u0000{interval.get('relation', '')}")
        predicted_intervals: dict[str, dict[str, Any]] = defaultdict(dict)
        for item in found:
            if item.get("kind") == "date" and item.get("interval_id") and item.get("interval_role"):
                predicted_intervals[str(item["interval_id"])][str(item["interval_role"])] = item
        interval_pred = {
            f"{record['case_id']}\u0000{_norm(pair['start'].get('value'))}\u0000{_norm(pair['end'].get('value'))}\u0000{pair['start'].get('interval_relation', '')}"
            for pair in predicted_intervals.values() if pair.get("start") and pair.get("end")
        }
        temporal_by_stratum["overall"]["interval_pred"].update(interval_pred)
        temporal_by_stratum["overall"]["interval_gold"].update(interval_gold)
        for dimension, label in strata.items():
            temporal_by_stratum[f"{dimension}:{label}"]["interval_pred"].update(interval_pred)
            temporal_by_stratum[f"{dimension}:{label}"]["interval_gold"].update(interval_gold)

    extraction = {
        stratum: {kind: _score_sets(values["pred"], values["gold"]) for kind, values in per_kind.items()}
        for stratum, per_kind in extraction_by_stratum.items()
    }
    temporal = {}
    for stratum, counts in temporal_by_stratum.items():
        temporal[stratum] = {
            "date_normalization_accuracy": round(counts["date_normalization_correct"] / counts["date_normalization_total"], 4) if counts["date_normalization_total"] else None,
            "date_normalization_labeled": counts["date_normalization_total"],
            "event_type_accuracy": round(counts["event_type_correct"] / counts["event_type_total"], 4) if counts["event_type_total"] else None,
            "event_type_labeled": counts["event_type_total"],
            "explicit_interval_labeled": len(counts["interval_gold"]),
            "explicit_interval_links": _score_sets(counts["interval_pred"], counts["interval_gold"]) if counts["interval_gold"] or counts["interval_pred"] else None,
        }

    chunks, items = chunks_by_split["test"], items_by_split["test"]
    dev_chunks, dev_items = chunks_by_split["dev"], items_by_split["dev"]
    query_rows = _query_rows(test_records, chunks, record_doc)
    dev_query_rows = _query_rows(dev_records, dev_chunks, record_doc)
    if not query_rows:
        raise ValueError("Test records contain no retrieval_queries.")

    retrieval: dict[str, Any] = {}
    answer_references = {
        (str(record["case_id"]), str(reference["question_id"])): str(reference["reference"])
        for record in records for reference in record.get("answer_references", [])
    }
    for mode in modes:
        calibration = _calibrate_threshold(mode, dev_query_rows, dev_chunks, dev_items)
        supported_metrics: list[dict[str, float]] = []
        answer_overlap_rows: list[dict[str, float]] = []
        unsupported_abstentions: list[bool] = []
        calibrated_supported_answers: list[bool] = []
        calibrated_supported_citations: list[bool] = []
        calibrated_unsupported_abstentions: list[bool] = []
        by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in query_rows:
            record, query, gold = row["record"], row["query"], row["gold"]
            hits = _retrieve(mode, str(query["question"]), chunks, items)
            ranked_ids = [str(hit.get("chunk_id", "")) for hit in hits]
            answer = build_answer(str(query["question"]), hits, conflicts=[])
            calibrated_answer = build_answer(
                str(query["question"]), hits, conflicts=[],
                abstain_threshold=calibration["threshold"] if calibration else None,
            ) if calibration else None
            if row["supported"]:
                reference = answer_references.get((str(record["case_id"]), str(query.get("question_id", ""))))
                if reference:
                    answer_overlap_rows.append(evaluate_text(reference, str(answer.get("answer", ""))))
                reciprocal = next((1 / rank for rank, cid in enumerate(ranked_ids, 1) if cid in gold), 0.0)
                dcg = sum(1 / math.log2(rank + 1) for rank, cid in enumerate(ranked_ids, 1) if cid in gold)
                ideal_n = min(5, len(gold))
                idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_n + 1))
                metrics = {
                    "hit_at_5": float(bool(set(ranked_ids[:5]) & gold)),
                    "recall_at_5": len(set(ranked_ids[:5]) & gold) / max(1, len(gold)),
                    "mrr_at_5": reciprocal,
                    "ndcg_at_5": dcg / idcg if idcg else 0.0,
                    "gold_citation_covered": float(bool({str(c.get("chunk_id", "")) for c in answer.get("citations", [])} & gold)),
                }
                supported_metrics.append(metrics)
                row_metrics = {**metrics, "abstained": bool(answer.get("abstained"))}
                if calibrated_answer is not None:
                    calibrated_supported_answers.append(not bool(calibrated_answer.get("abstained")))
                    calibrated_supported_citations.append(bool(
                        {str(c.get("chunk_id", "")) for c in calibrated_answer.get("citations", [])} & gold
                    ))
            else:
                abstained = bool(answer.get("abstained"))
                unsupported_abstentions.append(abstained)
                row_metrics = {"abstained": abstained}
                if calibrated_answer is not None:
                    calibrated_unsupported_abstentions.append(bool(calibrated_answer.get("abstained")))
            for dimension, value in (("language", record["language"]), ("ocr_status", record["ocr_status"])):
                by_stratum[f"{dimension}:{value}"].append({"supported": row["supported"], **row_metrics})
        mean = lambda key: round(sum(row[key] for row in supported_metrics) / len(supported_metrics), 4) if supported_metrics else None
        retrieval[mode] = {
            "queries": len(query_rows), "supported_queries": len(supported_metrics),
            "unsupported_queries": len(unsupported_abstentions),
            "hit_at_5": mean("hit_at_5"), "recall_at_5": mean("recall_at_5"),
            "mrr_at_5": mean("mrr_at_5"), "ndcg_at_5": mean("ndcg_at_5"),
            "gold_citation_coverage": mean("gold_citation_covered"),
            "answer_text_overlap": {
                key: round(sum(float(row.get(key, 0.0)) for row in answer_overlap_rows) / len(answer_overlap_rows), 4)
                for key in ("bleu", "rouge_l", "precision", "recall", "f1")
            } if answer_overlap_rows else None,
            "answer_references_scored": len(answer_overlap_rows),
            "unsupported_abstention_rate": round(sum(unsupported_abstentions) / len(unsupported_abstentions), 4) if unsupported_abstentions else None,
            "dev_threshold_calibration": calibration,
            "calibrated_test_metrics": ({
                "supported_answer_rate": round(sum(calibrated_supported_answers) / len(calibrated_supported_answers), 4) if calibrated_supported_answers else None,
                "supported_gold_citation_coverage": round(sum(calibrated_supported_citations) / len(calibrated_supported_citations), 4) if calibrated_supported_citations else None,
                "unsupported_abstention_rate": round(sum(calibrated_unsupported_abstentions) / len(calibrated_unsupported_abstentions), 4) if calibrated_unsupported_abstentions else None,
            } if calibration else None),
            "by_stratum": {
                key: {
                    "queries": len(values),
                    "supported_queries": sum(bool(v["supported"]) for v in values),
                    "unsupported_queries": sum(not bool(v["supported"]) for v in values),
                    "supported_hit_at_5": round(sum(v.get("hit_at_5", 0.0) for v in values if v["supported"]) / max(1, sum(bool(v["supported"]) for v in values)), 4) if any(v["supported"] for v in values) else None,
                    "unsupported_abstention_rate": round(sum(bool(v["abstained"]) for v in values if not v["supported"]) / max(1, sum(not bool(v["supported"]) for v in values)), 4) if any(not v["supported"] for v in values) else None,
                } for key, values in sorted(by_stratum.items())
            },
        }

    report = {
        "benchmark": "Human-reviewed property/civil corpus benchmark",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "test_records": len(test_records),
        "test_matter_groups": len({str(r["document_group"]) for r in test_records}),
        "dev_records": len(dev_records),
        "dev_queries": len(dev_query_rows),
        "languages": dict(sorted({str(r["language"]): sum(str(x["language"]) == str(r["language"]) for x in test_records) for r in test_records}.items())),
        "ocr_status_counts": {status: sum(r["ocr_status"] == status for r in test_records) for status in ("native_text", "ocr", "mixed")},
        "extraction": extraction,
        "temporal": temporal,
        "retrieval": retrieval,
        "privacy_note": "Aggregate metrics only; source text, evidence spans, and extracted values are not included.",
        "limitations": ["Metrics apply only to these adjudicated held-out records.", "Text overlap is not legal correctness.", "Group separation is validated from document_group and split fields; maintain your signed-off split manifest."],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Adjudicated UTF-8 JSONL page/section records")
    parser.add_argument("--output", type=Path, default=_DEFAULT_REPORT, help=f"Aggregate report path (default: {_DEFAULT_REPORT})")
    parser.add_argument("--modes", nargs="+", choices=("bm25", "hybrid", "graph"), default=["bm25", "hybrid", "graph"])
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    report = _run(args.input, args.output, args.modes)
    print(f"Human-reviewed corpus benchmark: {report['test_records']} held-out records, {report['test_matter_groups']} matter groups")
    print(f"Input SHA-256: {report['input_sha256']} · aggregate report: {args.output}")
    for stratum, metrics in report["extraction"].items():
        print(f"Extraction {stratum}: " + ", ".join(f"{kind} F1={score['f1']:.3f}" if score["f1"] is not None else f"{kind} F1=n/a" for kind, score in metrics.items()))
    for stratum, metrics in report["temporal"].items():
        interval = metrics["explicit_interval_links"]
        interval_score = f"{interval['f1']:.3f}" if interval and interval["f1"] is not None else "not measured"
        print(f"Temporal {stratum}: date-normalization={metrics['date_normalization_accuracy']} event-type={metrics['event_type_accuracy']} interval-F1={interval_score}")
    for mode, metrics in report["retrieval"].items():
        overlap = metrics["answer_text_overlap"]
        rouge = f"{overlap['rouge_l']:.3f}" if overlap else "not measured"
        metric = lambda key: f"{metrics[key]:.3f}" if metrics[key] is not None else "n/a"
        print(f"{mode}: MRR@5={metric('mrr_at_5')} nDCG@5={metric('ndcg_at_5')} citation={metric('gold_citation_coverage')} ROUGE-L={rouge} unsupported-abstain={metrics['unsupported_abstention_rate']}")


if __name__ == "__main__":
    main()
