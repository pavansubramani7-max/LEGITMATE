"""Measure extraction on the bundled, manually labeled synthetic property/civil PDFs.

Run from the project root with `.venv/Scripts/python.exe -m app.evaluation.extraction_benchmark`.
The benchmark reads only the bundled sample PDFs and writes one generated report;
it never reads or changes the live user document index.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import ROOT
from app.evidence.conflicts import detect_conflicts
from app.extraction.claims import extract_claims
from app.extraction.dates import extract_dates
from app.extraction.entities import extract_entities
from app.ingestion.pdf_reader import extract_pages

GOLD_PATH = ROOT / "data" / "evaluation" / "extraction_cases_v1.json"
REPORT_PATH = ROOT / "data" / "processed" / "property_civil_benchmark" / "extraction_latest.json"


def _norm(value: Any) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(value or "").casefold()).split())


def _scores(predicted: set[str], expected: set[str]) -> dict[str, Any]:
    correct = predicted & expected
    return {
        "predicted": len(predicted), "expected": len(expected), "matched": len(correct),
        "precision": round(len(correct) / len(predicted), 4) if predicted else 0.0,
        "recall": round(len(correct) / len(expected), 4) if expected else 0.0,
        "false_positives": sorted(predicted - expected),
        "missed": sorted(expected - predicted),
    }


def run() -> dict[str, Any]:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    all_items: list[dict[str, Any]] = []
    entities: dict[str, set[str]] = defaultdict(set)
    dates: set[str] = set()
    amounts: set[str] = set()
    claims_by_source: dict[tuple[str, int], list[str]] = defaultdict(list)

    for filename in gold["source_files"]:
        source = ROOT / "data" / "sample" / filename
        if not source.is_file():
            raise FileNotFoundError(f"Bundled benchmark PDF is missing: {source}")
        for page in extract_pages(source.read_bytes()):
            number, text = int(page["page"]), str(page["text"])
            extracted = extract_entities(source.stem, filename, number, text)
            extracted += extract_dates(source.stem, filename, number, text)
            extracted += extract_claims(source.stem, filename, number, text)
            all_items.extend(extracted)
            for item in extracted:
                value = _norm(item.get("value"))
                if item.get("kind") in {"person", "org", "property"}:
                    entities[str(item["kind"])].add(value)
                elif item.get("kind") == "date":
                    dates.add(value)
                elif item.get("kind") == "amount":
                    amounts.add(value)
                elif item.get("kind") == "claim":
                    claims_by_source[(filename, number)].append(_norm(value))

    entity_scores = {
        kind: _scores(entities[kind], {_norm(value) for value in values})
        for kind, values in gold["entities"].items()
    }
    date_scores = _scores(dates, {_norm(value) for value in gold["dates"]})
    amount_scores = _scores(amounts, {_norm(value) for value in gold["amounts"]})

    gold_claims = [
        (filename, int(page), _norm(text))
        for filename, page, text in gold["claims"]
    ]
    claim_found = [
        any(expected in predicted for predicted in claims_by_source[(filename, page)])
        for filename, page, expected in gold_claims
    ]
    matched_predictions = 0
    total_predictions = sum(len(values) for values in claims_by_source.values())
    false_positive_claims: list[dict[str, Any]] = []
    for (filename, page), predictions in claims_by_source.items():
        expected = [text for file_name, page_number, text in gold_claims if file_name == filename and page_number == page]
        for prediction in predictions:
            if any(gold_text in prediction for gold_text in expected):
                matched_predictions += 1
            else:
                false_positive_claims.append({"document": filename, "page": page, "value": prediction})
    claims = {
        "predicted": total_predictions, "expected": len(gold_claims), "matched": sum(claim_found),
        "precision": round(matched_predictions / total_predictions, 4) if total_predictions else 0.0,
        "recall": round(sum(claim_found) / len(gold_claims), 4) if gold_claims else 0.0,
        "missed": [gold["claims"][index] for index, found in enumerate(claim_found) if not found],
        "false_positive_count": total_predictions - matched_predictions,
    }

    item_by_id = {str(item["item_id"]): item for item in all_items}
    conflicts = detect_conflicts(all_items)
    known_conflict_results = []
    for expected in gold["known_conflicts"]:
        wanted = {_norm(value) for value in expected["values"]}
        found = False
        for conflict in conflicts:
            if conflict.get("kind") != expected["kind"]:
                continue
            evidence = [item_by_id[item_id] for item_id in conflict.get("items", []) if item_id in item_by_id]
            values = {_norm(item.get("value")) for item in evidence}
            documents = {item.get("document_id") for item in evidence}
            if wanted <= values and len(documents) >= 2:
                found = True
                break
        known_conflict_results.append({"topic": expected["topic"], "kind": expected["kind"], "found": found})
    conflict_details = []
    for conflict in conflicts:
        evidence = [item_by_id[item_id] for item_id in conflict.get("items", []) if item_id in item_by_id]
        conflict_details.append({
            "kind": conflict.get("kind"), "label": conflict.get("label"),
            "evidence": [{"document": item.get("document"), "page": item.get("page"), "value": item.get("value"), "context": item.get("context")} for item in evidence],
        })

    report = {
        "benchmark": gold["benchmark"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_corpus": gold["corpus"],
        "documents": len(gold["source_files"]),
        "pages": 6,
        "entities": entity_scores,
        "dates": date_scores,
        "amounts": amount_scores,
        "claims": claims,
        "false_positive_claims": false_positive_claims,
        "conflicts": {"detected_count": len(conflicts), "known_cases": known_conflict_results, "records": conflict_details},
        "limitations": [
            "The documents are synthetic and the gold set is small; these scores do not establish performance on real legal files.",
            "Claim matching accepts a gold sentence as a substring of an extracted span to allow layout whitespace differences.",
            "OCR is not measured here; these bundled PDFs contain selectable text.",
        ],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Benchmark: {report['benchmark']}")
    print(f"Corpus: {report['documents']} synthetic PDFs / {report['pages']} pages")
    for kind, metrics in entity_scores.items():
        print(f"{kind:8s} precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} ({metrics['matched']}/{metrics['expected']} gold; {metrics['predicted']} predicted)")
    print(f"dates    precision={date_scores['precision']:.3f} recall={date_scores['recall']:.3f}")
    print(f"amounts  precision={amount_scores['precision']:.3f} recall={amount_scores['recall']:.3f}")
    print(f"claims   precision={claims['precision']:.3f} recall={claims['recall']:.3f} ({claims['matched']}/{claims['expected']} gold)")
    print(f"conflicts: {len(conflicts)} flagged; known cases={sum(row['found'] for row in known_conflict_results)}/{len(known_conflict_results)}")
    print(f"Saved: {REPORT_PATH}")
    return report


if __name__ == "__main__":
    run()
