"""Prepare and adjudicate blind, matter-grouped annotations for LEGITMATE.

Usage:
  python -m app.evaluation.annotation_workbench prepare --source pages.jsonl --source-root authorized-originals --reviewer-a alice --reviewer-b bob --out-dir reviews
  python -m app.evaluation.annotation_workbench compare --reviewer-a reviews/reviewer-a.jsonl --reviewer-b reviews/reviewer-b.jsonl --output reviews/adjudication.jsonl
  # An adjudicator fills adjudication.labels in every row of adjudication.jsonl.
  python -m app.evaluation.annotation_workbench finalize --worksheet reviews/adjudication.jsonl --adjudicator carol --source-root authorized-originals --output adjudicated.jsonl
  python -m app.evaluation.reviewed_corpus_benchmark --input adjudicated.jsonl

Review files contain source text and must remain access-controlled. The tool
never invents labels and does not call an external model or provider.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

_BASE = ("case_id", "document_group", "split", "source_file", "source_sha256", "page", "citation_unit", "language", "script", "ocr_status", "text")
_LABELS = ("entities", "dates", "amounts", "claims", "intervals", "retrieval_queries", "answer_references")
_MAX_REVIEWER_ID_LENGTH = 128


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{n}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{n}: each line must be a JSON object.")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}: no records found.")
    return rows


def _write(path: Path, rows: list[dict[str, Any]], *, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {path}; choose another path or pass --force.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _base(row: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in _BASE if key not in row]
    if missing:
        raise ValueError(f"{row.get('case_id', '<unknown>')}: source record missing {', '.join(missing)}")
    return {key: row[key] for key in _BASE}


def _unique(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id = {str(row.get("case_id", "")): row for row in rows}
    if "" in by_id or len(by_id) != len(rows):
        raise ValueError("case_id values must be present and unique.")
    return by_id


def _labels_signature(values: list[Any]) -> list[str]:
    """Compare unordered annotation rows without treating row order as disagreement."""
    return sorted(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) for value in values)


def _verify_sources(rows: list[dict[str, Any]], source_root: Path) -> None:
    root = source_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("--source-root must be a folder containing the authorized source documents.")
    for row in rows:
        relative = Path(str(row["source_file"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{row['case_id']}: source_file must be a relative path inside --source-root.")
        try:
            source = (root / relative).resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"{row['case_id']}: source file could not be opened under --source-root.") from exc
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f"{row['case_id']}: source_file is not a regular file inside --source-root.")
        hasher = hashlib.sha256()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        if not hmac.compare_digest(hasher.hexdigest(), str(row["source_sha256"]).casefold()):
            raise ValueError(f"{row['case_id']}: source file SHA-256 does not match the annotation manifest.")


def _check_matter_splits(rows: list[dict[str, Any]]) -> None:
    groups: dict[str, set[str]] = {}
    for row in rows:
        groups.setdefault(str(row["document_group"]), set()).add(str(row["split"]))
    if any(len(splits) != 1 for splits in groups.values()):
        raise ValueError("Matter leakage: one document_group appears in more than one split.")


def prepare(source: Path, reviewer_a: str, reviewer_b: str, out_dir: Path, source_root: Path, force: bool = False) -> list[Path]:
    reviewer_a, reviewer_b = reviewer_a.strip(), reviewer_b.strip()
    if (not reviewer_a or not reviewer_b or reviewer_a == reviewer_b
            or len(reviewer_a) > _MAX_REVIEWER_ID_LENGTH or len(reviewer_b) > _MAX_REVIEWER_ID_LENGTH):
        raise ValueError("Provide two different reviewer IDs of 1 to 128 characters.")
    source_rows = _jsonl(source)
    records = [_base(row) for row in source_rows]
    _verify_sources(records, source_root)
    _unique(records)
    _check_matter_splits(records)
    outputs = []
    for slot, reviewer in zip(("reviewer-a", "reviewer-b"), (reviewer_a, reviewer_b)):
        target = out_dir / f"{slot}.jsonl"
        rows = [{**record, "reviewer_id": reviewer, "labels": {field: [] for field in _LABELS}} for record in records]
        _write(target, rows, force=force)
        outputs.append(target)
    return outputs


def compare(reviewer_a: Path, reviewer_b: Path, output: Path, force: bool = False) -> dict[str, Any]:
    a_rows, b_rows = _jsonl(reviewer_a), _jsonl(reviewer_b)
    a, b = _unique(a_rows), _unique(b_rows)
    if set(a) != set(b):
        raise ValueError("Both reviewers must annotate the same case_id set.")
    ids_a = {str(row.get("reviewer_id", "")).strip() for row in a.values()}
    ids_b = {str(row.get("reviewer_id", "")).strip() for row in b.values()}
    if len(ids_a) != 1 or len(ids_b) != 1 or not next(iter(ids_a)) or not next(iter(ids_b)) or ids_a == ids_b:
        raise ValueError("Each file must identify one distinct reviewer_id.")
    reviewer_ids = [next(iter(ids_a)), next(iter(ids_b))]
    worksheet, agreement = [], {field: 0 for field in _LABELS}
    for case_id in sorted(a):
        left, right = a[case_id], b[case_id]
        if _base(left) != _base(right):
            raise ValueError(f"{case_id}: the source metadata/text differs between reviewer files.")
        labels_left, labels_right = left.get("labels"), right.get("labels")
        if not isinstance(labels_left, dict) or not isinstance(labels_right, dict):
            raise ValueError(f"{case_id}: reviewer labels must be objects.")
        for field in _LABELS:
            if not isinstance(labels_left.get(field, []), list) or not isinstance(labels_right.get(field, []), list):
                raise ValueError(f"{case_id}: {field} labels must be lists.")
            if _labels_signature(labels_left.get(field, [])) == _labels_signature(labels_right.get(field, [])):
                agreement[field] += 1
        worksheet.append({
            **_base(left),
            "reviewer_ids": reviewer_ids,
            "reviewer_labels": {reviewer_ids[0]: labels_left, reviewer_ids[1]: labels_right},
            "adjudication": {"adjudicator_id": "", "labels": None, "private_note": ""},
        })
    _check_matter_splits(worksheet)
    _write(output, worksheet, force=force)
    return {"records": len(worksheet), "reviewers": reviewer_ids, "exact_agreement_by_field": agreement, "worksheet": str(output)}


def finalize(worksheet_path: Path, adjudicator: str, output: Path, source_root: Path, force: bool = False) -> dict[str, Any]:
    from app.evaluation.reviewed_corpus_benchmark import _validate_record

    adjudicator = adjudicator.strip()
    if not adjudicator or len(adjudicator) > _MAX_REVIEWER_ID_LENGTH:
        raise ValueError("An adjudicator ID of 1 to 128 characters is required.")
    worksheet = _jsonl(worksheet_path)
    records = []
    for row in worksheet:
        base = _base(row)
        reviewer_ids = row.get("reviewer_ids")
        reviewer_labels = row.get("reviewer_labels")
        adjudication = row.get("adjudication")
        if (not isinstance(reviewer_ids, list) or len(reviewer_ids) != 2
                or any(not isinstance(value, str) or not value.strip() for value in reviewer_ids)
                or len(set(reviewer_ids)) != 2):
            raise ValueError(f"{base['case_id']}: two distinct reviewer IDs are required.")
        if not isinstance(reviewer_labels, dict) or set(reviewer_labels) != set(reviewer_ids):
            raise ValueError(f"{base['case_id']}: both reviewer label sets must be retained in the adjudication worksheet.")
        for reviewer_id in reviewer_ids:
            labels = reviewer_labels[reviewer_id]
            if not isinstance(labels, dict) or any(not isinstance(labels.get(field), list) for field in _LABELS):
                raise ValueError(f"{base['case_id']}: {reviewer_id}'s worksheet needs list values for every label field.")
        if adjudicator in reviewer_ids:
            raise ValueError(f"{base['case_id']}: adjudicator must not be one of the annotators.")
        if not isinstance(adjudication, dict) or str(adjudication.get("adjudicator_id", "")).strip() != adjudicator:
            raise ValueError(f"{base['case_id']}: set adjudication.adjudicator_id to {adjudicator!r} after review.")
        labels = adjudication.get("labels")
        if not isinstance(labels, dict) or any(not isinstance(labels.get(field), list) for field in _LABELS):
            raise ValueError(f"{base['case_id']}: adjudication.labels must contain list values for every label field: {', '.join(_LABELS)}.")
        record = {**base, "annotators": reviewer_ids, "adjudicated_by": adjudicator, **{field: labels[field] for field in _LABELS}}
        _validate_record(record, len(records) + 1)
        records.append(record)
    _unique(records)
    _check_matter_splits(records)
    _verify_sources(records, source_root)
    splits = {str(row["split"]) for row in records}
    missing_splits = {"train", "dev", "test"} - splits
    if missing_splits:
        raise ValueError(f"Matter-grouped train/dev/test records are required before benchmark evaluation; missing: {', '.join(sorted(missing_splits))}.")
    _write(output, records, force=force)
    return {"records": len(records), "matter_groups": len({row["document_group"] for row in records}), "source_hashes_verified": len(records), "adjudicator": adjudicator, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Create separate blind annotation files from a source manifest.")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--reviewer-a", required=True)
    p.add_argument("--reviewer-b", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True, help="Folder containing authorized originals referenced by source_file; hashes are verified before review files are written.")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("compare", help="Create a private adjudication worksheet and field-level agreement counts.")
    p.add_argument("--reviewer-a", type=Path, required=True)
    p.add_argument("--reviewer-b", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("finalize", help="Validate adjudicated records for the held-out corpus evaluator.")
    p.add_argument("--worksheet", type=Path, required=True)
    p.add_argument("--adjudicator", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True, help="Re-verify each original source file SHA-256 before finalizing the corpus.")
    p.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = {"review_files": [str(path) for path in prepare(args.source, args.reviewer_a, args.reviewer_b, args.out_dir, args.source_root, args.force)]}
    elif args.command == "compare":
        result = compare(args.reviewer_a, args.reviewer_b, args.output, args.force)
    else:
        result = finalize(args.worksheet, args.adjudicator, args.output, args.source_root, args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
