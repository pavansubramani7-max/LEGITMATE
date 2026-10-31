"""Prepare separate, provenance-preserving Kaggle legal research corpora.

Does not modify data/index.json, uploaded documents, or the existing CUAD splits.
Run from the LEGITMATE repository root:
    .venv\\Scripts\\python.exe data\\pipeline\\prepare_kaggle_legal.py
"""
from __future__ import annotations

import csv
import gzip
import json
import pathlib
import random
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "processed" / "kaggle_legal"
INDQA = ROOT / "data" / "raw" / "indqa" / "legal_qa.csv"
ICLEC = ROOT / "data" / "raw" / "iclec" / "indian_supreme_court_judgments.csv"
OPENNYAYA = ROOT / "data" / "raw" / "opennyaya" / "nyai-supreme-court-clean"
SEED = 42


def _split_cases(case_ids: list[str]) -> dict[str, str]:
    groups = sorted(set(case_ids))
    random.Random(SEED).shuffle(groups)
    n_test = max(1, round(len(groups) * 0.10)) if len(groups) >= 10 else 1
    n_val = max(1, round(len(groups) * 0.10)) if len(groups) >= 10 else 1
    if n_test + n_val >= len(groups):
        n_test, n_val = 1, 0
    val_ids = set(groups[:n_val])
    test_ids = set(groups[n_val:n_val + n_test])
    return {case: "val" if case in val_ids else "test" if case in test_ids else "train" for case in groups}


def _write_jsonl(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_indian_qa() -> dict[str, Any]:
    if not INDQA.is_file():
        return {"status": "missing", "expected_file": str(INDQA.relative_to(ROOT))}
    with INDQA.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    duplicate_rows = 0
    for row in rows:
        case = " ".join((row.get("case_name") or "").split())
        query = " ".join((row.get("question") or "").split())
        answer = " ".join((row.get("answer") or "").split())
        if not (case and query and answer):
            continue
        key = (case.casefold(), query.casefold(), answer.casefold())
        if key in unique:
            duplicate_rows += 1
            continue
        date = (row.get("judgement_date") or "").strip()
        has_ref = str(row.get("has_reference_pdf") or "0").strip() == "1"
        unique[key] = {
            "query": query,
            "positive": answer,
            "case_name": case,
            "judgement_date": date if date.casefold() not in {"nan", "nat", "none"} else "",
            "source_dataset": "yogeshm01/indian-legal-qa-dataset-10k-questions",
            "source_license": "CC BY 4.0",
            "supervision": "weak_qa_answer_pair; not a source passage or citation",
            "has_reference_pdf": has_ref,
        }
    records = list(unique.values())
    case_splits = _split_cases([r["case_name"] for r in records])
    out_dir = OUT / "indian_legal_qa"
    split_counts: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        selected = []
        for record in records:
            if case_splits[record["case_name"]] == split:
                selected.append({**record, "split": split})
        _write_jsonl(out_dir / f"{split}.jsonl", selected)
        split_counts[split] = len(selected)
    pdf_count = sum(1 for r in records if r["has_reference_pdf"])
    date_count = sum(1 for r in records if r["judgement_date"])
    return {
        "status": "prepared",
        "dataset": "Indian Legal QA Dataset (10K+ Questions)",
        "kaggle_ref": "yogeshm01/indian-legal-qa-dataset-10k-questions",
        "license": "CC BY 4.0",
        "source_rows": len(rows),
        "prepared_pairs": len(records),
        "duplicate_pairs_removed": duplicate_rows,
        "case_groups": len(case_splits),
        "pairs_with_reference_pdf_flag": pdf_count,
        "pairs_with_judgement_date": date_count,
        "split_strategy": "deterministic 80/10/10 case-group split; no case appears in more than one split",
        "split_counts": dict(split_counts),
        "use_limit": "Weak question-answer supervision for separate research/benchmarking. The answer is not a verified passage; never inject this dataset into user-document evidence or citations.",
    }


def prepare_iclec() -> dict[str, Any]:
    if not ICLEC.is_file():
        return {"status": "missing", "expected_file": str(ICLEC.relative_to(ROOT))}
    with ICLEC.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows = [r for r in rows if (r.get("judgment_text") or "").strip()]
    splits = _split_cases([r.get("id", "") for r in rows])
    out_dir = OUT / "iclec"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    # Full text remains separate from the private uploaded-document index.
    with gzip.open(out_dir / "judgments.jsonl.gz", "wt", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            case_id = (row.get("id") or "unknown").strip()
            split = splits[case_id]
            record = {
                "case_id": case_id,
                "split": split,
                "petitioner": (row.get("petitioner") or "").strip(),
                "respondent": (row.get("respondent") or "").strip(),
                "date_of_judgment": (row.get("date_of_judgment") or "").strip(),
                "act": (row.get("act") or "").strip(),
                "headnote": (row.get("headnote") or "").strip(),
                "citations": (row.get("citations") or "").strip(),
                "text": (row.get("judgment_text") or "").strip(),
                "source_dataset": "hrithikraj2537/indian-case-law-evaluation-corpus-iclec",
                "source_license": "CC0-1.0",
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[split] += 1
    return {
        "status": "prepared",
        "dataset": "Indian Case Law Evaluation Corpus (ICLEC)",
        "kaggle_ref": "hrithikraj2537/indian-case-law-evaluation-corpus-iclec",
        "license": "CC0-1.0",
        "judgment_rows": len(rows),
        "case_groups": len(splits),
        "split_strategy": "deterministic case-ID grouping; repeated rows for a case remain together",
        "split_counts": dict(counts),
        "metadata_completeness": "Many judgment rows lack act/headnote fields; full judgment text is present in the packaged rows.",
        "output": str((out_dir / "judgments.jsonl.gz").relative_to(ROOT)),
    }


def inspect_open_nyaya() -> dict[str, Any]:
    if not OPENNYAYA.is_dir():
        return {"status": "missing", "expected_directory": str(OPENNYAYA.relative_to(ROOT))}
    files = sorted(OPENNYAYA.rglob("*.md"))
    years = Counter(p.parent.name for p in files)
    total_bytes = sum(p.stat().st_size for p in files)
    return {
        "status": "available_raw_corpus",
        "dataset": "OpenNyaya Indian Supreme Court Judgement Clean",
        "kaggle_ref": "gaurav41/opennyaya-indian-supreme-court-judgement-clean",
        "license": "CC BY 4.0",
        "judgment_files": len(files),
        "uncompressed_text_bytes": total_bytes,
        "year_directories": dict(sorted(years.items())),
        "root": str(OPENNYAYA.relative_to(ROOT)),
        "quality_note": "The Kaggle card describes OCR-cleaned judgments; spot checks still show OCR artifacts. Keep exact source file citations and do not treat text as authoritative without review.",
        "use_limit": "Separate research/training corpus only; it is not added to the application's user-document index.",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "existing_cuad": {
            "source": "adityaraghuvanshi999/legalbenchrag-cuad-retrieval-dataset",
            "license": "CC BY 4.0",
            "prepared_stats": str((ROOT / "data/processed/cuad/stats.json").relative_to(ROOT)),
            "status": "preserved; this pipeline does not rewrite CUAD files",
        },
        "indian_legal_qa": prepare_indian_qa(),
        "iclec": prepare_iclec(),
        "opennyaya": inspect_open_nyaya(),
        "training_policy": "Do not promote a fine-tuned encoder to default based on training loss. Compare unchanged base and candidate on CUAD and case-grouped Indian legal held-out data; keep uploaded-document retrieval isolated from research corpora.",
    }
    path = OUT / "dataset_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved report: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
