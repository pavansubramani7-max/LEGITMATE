"""LEGITMATE — neuro-symbolic property/civil evidence research prototype."""
from __future__ import annotations

import uuid
import re
import json
import csv
import io
import zipfile
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from starlette.background import BackgroundTask

# Load .env before anything reads os.environ
load_dotenv()

from app.auth import (
    apply_security_headers,
    clear_session,
    get_csrf_token,
    handle_login,
    require_csrf,
    require_session,
    require_admin,
    get_session_user,
    bind_request_workspace,
    reset_request_workspace,
)
from app.accounts import initialize_accounts, list_users, create_customer, set_active, reset_password, change_password
from app.config import MAX_UPLOAD_BYTES, ROOT
from app.store import load, save, workspace_uploads
from app.ingestion.pdf_reader import detect_ocr_language, extract_pages, extract_pages_with_ocr, ocr_quality_signals
from app.ingestion.docx_reader import extract_docx_sections
from app.ingestion.text_readers import extract_office_sections, extract_text_sections
from app.ingestion.chunker import make_chunks
from app.search.lexical import rank
from app.retrieval.semantic import rank as semantic_rank, save_embeddings, delete_embeddings, backend_info
from app.retrieval.hybrid import rank as hybrid_rank
from app.retrieval.reranker import rerank as rerank_hits, reranker_info
from app.extraction.entities import extract_entities
from app.extraction.dates import extract_dates
from app.extraction.claims import extract_claims
from app.evidence.conflicts import detect_conflicts
from app.evidence.audit import record_audit_event, verify_audit_log
from app.evidence.timeline import build_timeline
from app.evidence.knowledge_graph import build_knowledge_graph, graph_augmented_rank
from app.qa.answer import build_answer
from app.qa.summary import build_case_brief
from app.qa.ai_provider import generate_chat_reply, generate_grounded_answer, generate_general_reply, provider_status, is_general_chat
from app.nlp.language import detect_language
from app.nlp.lab import analyze_text, evaluate_text, capability_status
from app.nlp.sequence_demo import classify_sequence
from app.reports import render_document_report
from app.schemas import SearchRequest, ReviewPatch, ManualItemRequest, LanguageRequest, MatterCreateRequest, MatterPatchRequest, DocumentMatterPatch
from app.evaluation.deployment_readiness import readiness_report
from pydantic import BaseModel, Field

app = FastAPI(
    title="LEGITMATE",
    version="0.6.0",
    description="Local neuro-symbolic research prototype for property/civil document understanding, source-linked claim review, temporal graph exploration, and hybrid graph-assisted retrieval. Not legal advice or validated legal reasoning.",
)


# ── security headers on every response ───────────────────────────────────────

@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    workspace_token = bind_request_workspace(request)
    try:
        response = await call_next(request)
        apply_security_headers(response)
        return response
    finally:
        reset_request_workspace(workspace_token)


# ── helpers ───────────────────────────────────────────────────────────────────

_LEGAL_TOPIC_RULES = {
    "Ownership and title": r"\b(title|ownership|owner|sale deed|conveyance|transfer of title)\b",
    "Lease and tenancy": r"\b(lease|lessee|lessor|tenant|rent|tenancy)\b",
    "Payments and amounts": r"\b(price|payment|consideration|amount|rent|deposit|rupees|INR|Rs\.)\b",
    "Property description": r"\b(property|plot|survey|khasra|khata|flat|unit|parcel|land)\b",
    "Possession and delivery": r"\b(possession|handover|vacate|deliver|occupancy)\b",
    "Dispute and notice": r"\b(dispute|notice|claim|allege|litigation|court|breach|objection)\b",
    "Obligations and conditions": r"\b(shall|must|covenant|obligation|condition|warrant|undertake)\b",
    "Registration and records": r"\b(registration|registered|encumbrance|mutation|record|sub-registrar)\b",
}


def _template_diagnostics(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Flag documents with many visibly unfilled form fields; this is a cue, not a classifier."""
    text = "\n".join(str(page.get("text", "")) for page in pages)
    blank_runs = re.findall(r"(?<!_)_{3,}(?!_)|\.{4,}|\[\s*\]", text)
    return {
        "looks_like_unfilled_template": len(blank_runs) >= 6,
        "blank_field_markers": len(blank_runs),
        "method": "Counted long underscore/dot blanks and empty brackets in extracted text; heuristic only.",
    }


def _document_scorecard(pages: list[dict[str, Any]], items: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute reference-free coverage measurements for every source section."""
    page_scores: list[dict[str, Any]] = []
    topic_rollups = {name: {"sections": 0, "occurrences": 0} for name in _LEGAL_TOPIC_RULES}
    valid_locations = {str(page.get("page", index + 1)) for index, page in enumerate(pages)}
    candidate_counts: dict[str, int] = {}
    for item in items:
        location = str(item.get("page", ""))
        candidate_counts[location] = candidate_counts.get(location, 0) + 1
    located_candidates = sum(count for location, count in candidate_counts.items() if location in valid_locations)
    readable_count = 0
    for index, page in enumerate(pages):
        text = str(page.get("text", ""))
        readable = bool(text.strip())
        if readable:
            readable_count += 1
        terms = re.findall(r"\w+(?:['’.-]\w+)*", text, re.UNICODE) if readable else []
        terms = [term.casefold() for term in terms if any(char.isalnum() for char in term)]
        unique_count = len(set(terms))
        cues: list[str] = []
        cue_occurrences = 0
        for topic, pattern in _LEGAL_TOPIC_RULES.items():
            hits = len(re.findall(pattern, text, re.I)) if readable else 0
            if hits:
                cues.append(topic)
                topic_rollups[topic]["sections"] += 1
                topic_rollups[topic]["occurrences"] += hits
                cue_occurrences += hits
        page_ref = page.get("page", index + 1)
        candidate_count = candidate_counts.get(str(page_ref), 0)
        page_scores.append({
            "section": page_ref,
            "readable": readable,
            "characters": len(text),
            "tokens": len(terms),
            "unique_terms": unique_count,
            "lexical_diversity_pct": round(unique_count / len(terms) * 100, 1) if terms else 0.0,
            "topic_cues": cues,
            "topic_coverage_pct": round(len(cues) / len(_LEGAL_TOPIC_RULES) * 100, 1) if readable else 0.0,
            "topic_cue_hits": cue_occurrences,
            "candidates": candidate_count,
            "status": "Scored" if readable else "No readable text",
        })
    total = len(pages)
    topic_slots = readable_count * len(_LEGAL_TOPIC_RULES)
    topic_slots_hit = sum(rollup["sections"] for rollup in topic_rollups.values())
    topic_coverage = [
        {"topic": topic, "sections": values["sections"],
         "coverage_pct": round(values["sections"] / readable_count * 100, 1) if readable_count else 0.0,
         "cue_occurrences": values["occurrences"]}
        for topic, values in topic_rollups.items()
    ]
    return {
        "sections_total": total,
        "sections_readable": readable_count,
        "sections_scored": readable_count,
        "section_coverage_pct": round(readable_count / total * 100, 1) if total else 0.0,
        "topic_coverage_pct": round(topic_slots_hit / topic_slots * 100, 1) if topic_slots else 0.0,
        "candidate_traceability_pct": round(located_candidates / len(items) * 100, 1) if items else None,
        "topics": topic_coverage,
        "per_section": page_scores,
        "note": "These are full-document coverage and text-profile measurements, not legal accuracy or truth scores. Topic percentages show the share of readable sections with matching rule cues.",
    }


def _upgrade_saved_report(report: dict[str, Any], document: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Backfill scorecards for reports saved before document-wide scoring existed."""
    if report.get("coverage_scorecard"):
        return report
    document_id = str(document.get("document_id", ""))
    grouped: dict[str, list[str]] = {}
    for chunk in data.get("chunks", []):
        if str(chunk.get("document_id", "")) == document_id and str(chunk.get("text", "")).strip():
            grouped.setdefault(str(chunk.get("page", "")), []).append(str(chunk["text"]))
    count = max(int(document.get("pages", report.get("pages_or_sections", 0)) or 0), len(grouped))
    pages = [{"page": number, "text": " ".join(grouped.get(str(number), []))} for number in range(1, count + 1)]
    items = [item for item in data.get("items", []) if str(item.get("document_id", "")) == document_id]
    updated = dict(report)
    updated["coverage_scorecard"] = _document_scorecard(pages, items)
    updated["analysis_scope"] = (str(report.get("analysis_scope", "")).strip() + " Saved-document scorecard backfilled from every indexed page/section; sections without indexed text are marked unreadable.").strip()
    return updated

def _run_extraction(doc_id: str, doc_name: str, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for p in pages:
        text = p["text"]
        if not text.strip():
            continue
        items.extend(extract_entities(doc_id, doc_name, p["page"], text))
        items.extend(extract_dates(doc_id, doc_name, p["page"], text))
        items.extend(extract_claims(doc_id, doc_name, p["page"], text))
    for item in items:
        item["source_type"] = "automated"
    return items


_REVIEW_FIELDS = (
    "review_status", "reviewed_at", "note", "event_date",
    "claim_evidence_judgment", "claim_evidence_refs", "claim_evidence_note",
    "evidence_reviewed_at",
)


def _item_review_key(item: dict[str, Any]) -> tuple[str, str, str]:
    value = re.sub(r"[^\w\s]", " ", str(item.get("value", "")).casefold())
    return str(item.get("kind", "")), str(item.get("page", "")), " ".join(value.split())


def _merge_reanalysis_items(previous: list[dict[str, Any]], refreshed: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep stable IDs and human review state for exact source-backed matches."""
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in previous:
        by_key.setdefault(_item_review_key(item), []).append(item)
    used: set[str] = set()
    merged: list[dict[str, Any]] = []
    for item in refreshed:
        candidates = by_key.get(_item_review_key(item), [])
        old = next((candidate for candidate in candidates if str(candidate.get("item_id")) not in used), None)
        if old is not None:
            old_id = str(old.get("item_id", ""))
            used.add(old_id)
            if old_id:
                item["item_id"] = old_id
            for key in _REVIEW_FIELDS:
                if key in old:
                    item[key] = old[key]
        merged.append(item)

    # Unmatched reviewed or manually added records remain visible for follow-up;
    # unmatched automated candidates are replaced by the improved extractor.
    referenced_ids = {
        str(ref_id)
        for item in previous
        if item.get("kind") == "claim" and item.get("claim_evidence_judgment", "unreviewed") != "unreviewed"
        for ref_id in item.get("claim_evidence_refs", [])
    }
    kept_reviewed = 0
    for old in previous:
        old_id = str(old.get("item_id", ""))
        if old_id in used:
            continue
        has_review = (
            old.get("source_type") == "manual"
            or old.get("review_status", "candidate") != "candidate"
            or old.get("note")
            or old.get("claim_evidence_judgment", "unreviewed") != "unreviewed"
            or old.get("claim_evidence_refs")
            or old.get("claim_evidence_note")
            or old_id in referenced_ids
        )
        if has_review:
            merged.append(old)
            kept_reviewed += 1
    return merged, kept_reviewed


def _analysis_sample(pages: list[dict[str, Any]], limit: int = 10000) -> str:
    readable = [str(page.get("text", "")) for page in pages if str(page.get("text", "")).strip()]
    if sum(map(len, readable)) <= limit:
        return "\n".join(text.strip() for text in readable)[:limit]
    slots = min(20, len(readable))
    selected = sorted({round(index * (len(readable) - 1) / max(1, slots - 1)) for index in range(slots)})
    per_page = max(1, limit // len(selected))
    return "\n".join(readable[index][:per_page].strip() for index in selected)[:limit]


def _processing_report(filename: str, suffix: str, pages: list[dict[str, Any]], items: list[dict[str, Any]], language: dict[str, Any], byte_count: int, corpus_chunks: list[str]) -> dict[str, Any]:
    """Create an auditable summary of what local extraction found."""
    topics = [label for label, pattern in _LEGAL_TOPIC_RULES.items()
              if any(re.search(pattern, str(page.get("text", "")), re.I) for page in pages)]
    kinds = {kind: sum(1 for item in items if item.get("kind") == kind) for kind in ("person", "org", "property", "date", "amount", "claim")}
    highlights = [
        {"kind": str(item.get("kind", "candidate")), "value": str(item.get("value", ""))[:240],
         "page": item.get("page"), "citation_unit": {".pptx": "slide", ".xlsx": "sheet"}.get(suffix, "page" if suffix == ".pdf" else "section"),
         "excerpt": str(item.get("context", item.get("value", "")))[:360]}
        for item in items[:100]
    ]
    sample_text = _analysis_sample(pages)
    nlp_analysis = analyze_text(sample_text, corpus_chunks=corpus_chunks)
    scorecard = _document_scorecard(pages, items)
    template_diagnostics = _template_diagnostics(pages)
    warnings = []
    if suffix == ".pdf" and not any(str(page.get("text", "")).strip() for page in pages):
        warnings.append("No selectable PDF text was found. Scanned pages need OCR before analysis.")
    if not topics:
        warnings.append("No topic matched the current legal-document topic cues; inspect the source text manually.")
    if template_diagnostics["looks_like_unfilled_template"]:
        warnings.append(
            "This file appears to be an unfilled template ("
            f"{template_diagnostics['blank_field_markers']} blank-field markers). "
            "Treat names, amounts, and property identifiers as incomplete until confirmed in the source."
        )
    return {
        "filename": filename, "file_type": suffix.lstrip("."), "size_bytes": byte_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "language": language, "pages_or_sections": len(pages),
        "readable_pages_or_sections": sum(bool(str(p.get("text", "")).strip()) for p in pages),
        "topics": topics, "counts": kinds, "candidate_highlights": highlights,
        "coverage_scorecard": scorecard, "template_diagnostics": template_diagnostics,
        "total_candidates": len(items), "nlp_analysis": nlp_analysis,
        "analysis_scope": f"Full-document coverage scoring assessed every readable page/section ({scorecard['sections_scored']} of {len(pages)} records), including token counts, lexical diversity, topic cue coverage and source-located candidate counts. The detailed POS/lemma and term preview remains bounded to {len(sample_text):,} characters for responsiveness; every readable section is included in the scorecard.",
        "warnings": warnings,
        "notice": "Automated extraction is a set of review candidates, not verified facts or legal advice.",
    }


def _refresh_conflicts(data: dict[str, Any]) -> None:
    existing_resolved = {
        c["conflict_id"]: c for c in data.get("conflicts", []) if c.get("status") == "resolved"
    }
    new_conflicts = detect_conflicts(data["items"])
    merged = list(existing_resolved.values())
    for c in new_conflicts:
        if c["conflict_id"] not in existing_resolved:
            merged.append(c)
    data["conflicts"] = merged


_UNASSIGNED_MATTER = "__unassigned__"


def _scoped_data(data: dict[str, Any], matter_id: str | None) -> dict[str, Any]:
    """Return an isolated read view for one matter; never mutate the stored index."""
    if not matter_id:
        return data
    matter = None
    if matter_id == _UNASSIGNED_MATTER:
        documents = [doc for doc in data.get("documents", []) if not doc.get("matter_id")]
    else:
        matter = next((row for row in data.get("matters", []) if row.get("matter_id") == matter_id), None)
        if matter is None:
            raise HTTPException(404, "Matter not found in this account workspace.")
        documents = [doc for doc in data.get("documents", []) if doc.get("matter_id") == matter_id]
    document_ids = {str(doc.get("document_id", "")) for doc in documents}
    items = [item for item in data.get("items", []) if str(item.get("document_id", "")) in document_ids]
    item_ids = {str(item.get("item_id", "")) for item in items}
    conflicts = [
        conflict for conflict in data.get("conflicts", [])
        if conflict.get("items") and set(map(str, conflict.get("items", []))).issubset(item_ids)
    ]
    return {
        **data,
        "documents": documents,
        "chunks": [chunk for chunk in data.get("chunks", []) if str(chunk.get("document_id", "")) in document_ids],
        "items": items,
        "conflicts": conflicts,
        "matters": [matter] if matter else [],
    }


def _csv_safe(value: Any) -> str:
    text = str(value if value is not None else "")
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else text


def _evidence_matrix(data: dict[str, Any]) -> list[dict[str, Any]]:
    items = data.get("items", [])
    chunks = data.get("chunks", [])
    by_id = {str(item.get("item_id", "")): item for item in items}
    conflicts_by_id: dict[str, list[str]] = {}
    for conflict in data.get("conflicts", []):
        for item_id in conflict.get("items", []):
            conflicts_by_id.setdefault(str(item_id), []).append(str(conflict.get("conflict_id", "")))
    rows = []
    for claim in items:
        if claim.get("kind") != "claim":
            continue
        link = _source_link(claim, chunks, strict_value=True)
        source_chunk = next((chunk for chunk in chunks if chunk.get("chunk_id") == link.get("source_chunk_id")), None)
        references = []
        for item_id in claim.get("claim_evidence_refs", []):
            item = by_id.get(str(item_id))
            if not item:
                continue
            ref_source = _source_link(item, chunks)
            references.append({
                "item_id": item.get("item_id"), "kind": item.get("kind"), "value": item.get("value"),
                "document": item.get("document"), "page": item.get("page"),
                "citation_unit": item.get("citation_unit", "page"),
                "source_match": bool(ref_source.get("source_match")),
                "source_chunk_id": ref_source.get("source_chunk_id"),
            })
        rows.append({
            "claim_id": claim.get("item_id"), "claim": claim.get("value"), "context": claim.get("context", ""),
            "document_id": claim.get("document_id"), "document": claim.get("document"),
            "page": claim.get("page"), "citation_unit": claim.get("citation_unit", "page"),
            "source_match": bool(link.get("source_match")), "source_chunk_id": link.get("source_chunk_id"),
            "source_passage": source_chunk.get("text", "") if source_chunk else "",
            "review_status": claim.get("review_status", "candidate"),
            "evidence_judgment": claim.get("claim_evidence_judgment", "unreviewed"),
            "review_note": claim.get("claim_evidence_note", ""),
            "evidence_references": references,
            "conflict_ids": conflicts_by_id.get(str(claim.get("item_id", "")), []),
            "reviewed_at": claim.get("evidence_reviewed_at") or claim.get("reviewed_at"),
        })
    return sorted(rows, key=lambda row: (str(row.get("document", "")).casefold(), int(row.get("page") or 0), str(row.get("claim", ""))))


def _matrix_csv(rows: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    fields = ["claim_id", "claim", "document", "citation", "source_match", "source_chunk_id", "review_status", "evidence_judgment", "evidence_references", "conflict_ids", "review_note"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        row_out = {
            **row,
            "citation": f"{row.get('citation_unit', 'page')} {row.get('page', '?')}",
            "evidence_references": " | ".join(f"{ref.get('kind')}: {ref.get('value')} ({ref.get('document')}, {ref.get('citation_unit')} {ref.get('page')})" for ref in row.get("evidence_references", [])),
            "conflict_ids": " | ".join(row.get("conflict_ids", [])),
        }
        writer.writerow({field: _csv_safe(row_out.get(field)) for field in fields})
    return output.getvalue()


def _source_link(item: dict[str, Any], chunks: list[dict[str, Any]], *, strict_value: bool = False) -> dict[str, Any]:
    """Link an extracted item to a same-document, same-page chunk by exact text."""
    def normalize(text: Any) -> str:
        return " ".join(str(text or "").casefold().split())

    candidates = [
        chunk for chunk in chunks
        if chunk.get("document_id") == item.get("document_id")
        and chunk.get("page") == item.get("page")
    ]
    value = normalize(item.get("value"))
    context = normalize(item.get("context"))
    needles = [(value, "claim_text" if strict_value else "value")]
    if not strict_value and context and context != value:
        needles.insert(0, (context, "context"))

    for needle, match_type in needles:
        if not needle:
            continue
        for chunk in candidates:
            if needle in normalize(chunk.get("text")):
                return {
                    "source_match": True,
                    "source_match_type": match_type,
                    "source_chunk_id": chunk.get("chunk_id"),
                    "source_passage": chunk.get("text"),
                    "citation_unit": chunk.get("citation_unit", item.get("citation_unit", "page")),
                }
    return {
        "source_match": False,
        "source_match_type": None,
        "source_chunk_id": None,
        "source_passage": None,
        "citation_unit": item.get("citation_unit", "page"),
    }


def _document_integrity(document: dict[str, Any]) -> dict[str, Any]:
    """Compare an account-local original with its recorded upload fingerprint."""
    stored_name = Path(str(document.get("stored_filename", ""))).name
    source_path = workspace_uploads() / stored_name
    try:
        resolved = source_path.resolve(strict=True)
        if resolved.parent != workspace_uploads().resolve() or not resolved.is_file():
            return {"status": "path_invalid", "verified": False}
        hasher = hashlib.sha256()
        with resolved.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(block)
        actual = hasher.hexdigest()
    except OSError:
        return {"status": "missing", "verified": False}
    expected = str(document.get("source_sha256", ""))
    if not expected:
        return {"status": "legacy_unverified", "verified": False, "current_sha256": actual}
    matches = expected.casefold() == actual.casefold()
    return {
        "status": "verified" if matches else "changed",
        "verified": matches,
        "expected_sha256": expected,
        "current_sha256": actual,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }


# ── health (public) ───────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "LEGITMATE", "mode": "local prototype"}


@app.get("/api/overview")
def overview(matter_id: str | None = None, _session: str = Depends(require_session)) -> dict[str, Any]:
    """Authenticated, read-only dashboard summary based on the local index."""
    data = _scoped_data(load(), matter_id)
    items = data.get("items", [])
    conflicts = data.get("conflicts", [])
    graph = build_knowledge_graph(data, _source_link)
    return {
        "counts": {
            "documents": len(data.get("documents", [])),
            "pages": sum(int(d.get("pages", 0) or 0) for d in data.get("documents", [])),
            "passages": len(data.get("chunks", [])),
            "candidate_items": sum(1 for i in items if i.get("review_status", "candidate") == "candidate"),
            "claims": sum(1 for i in items if i.get("kind") == "claim"),
            "open_conflicts": sum(1 for c in conflicts if c.get("status") != "resolved"),
            "timeline_events": len(build_timeline(items)),
            "graph_nodes": graph["summary"]["node_count"],
        },
        "documents": sorted(data.get("documents", []), key=lambda d: d.get("uploaded_at", ""), reverse=True)[:5],
        "semantic_backend": backend_info(),
        "reranker": reranker_info(),
        "ai_provider": provider_status(),
        "methods": [
            {"name": "Find matching words (BM25)", "detail": "Breaks the question into searchable words, ignores common words, understands selected legal synonyms, and ranks matching passages."},
            {"name": "Find similar meaning (embeddings)", "detail": backend_info()["detail"]},
            {"name": "Combine search results (RRF)", "detail": "Combines word matching and meaning-based search so either can help find a passage."},
            {"name": "Rerank passages (cross-encoder)", "detail": reranker_info()["detail"]},
            {"name": "Find names and facts (NER + rules)", "detail": "Looks for people, organizations, properties, dates, and amounts using spaCy when available and pattern matching as a fallback."},
            {"name": "Check the source page (evidence grounding)", "detail": "Checks whether a claim appears in the same document and page; found facts still need human review."},
            {"name": "Connect facts over time (knowledge graph)", "detail": "Links findings to source passages, exact repeated multi-word entity names across files, possible differences, and date order within each document. Explicit date ranges get start/end links."},
            {"name": "Follow named facts (GraphRAG prototype)", "detail": "When a question names an extracted person, organization, or property, hybrid search can expand to same-page facts and exact normalized repeated entity mentions in other files. Each expanded result shows its graph path; relevance still comes from the text retriever."},
            {"name": "Detect language locally", "detail": "Uses writing-script checks and common-word cues for several languages. Very short, mixed, or uncommon text can remain uncertain."},
            {"name": "Live chat or grounded answer (optional AI)", "detail": "Live mode sends the message and recent chat turns. Document mode can send the question and up to five retrieved excerpts when selected. Uploaded files remain local."},
        ],
        "notice": "Prototype metrics describe the current local index. Extracted values require human review.",
    }


# ── auth endpoints (public) ───────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str


class CustomerCreateRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    display_name: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=12, max_length=256)


class AccountAdminPatch(BaseModel):
    active: bool | None = None
    password: str | None = Field(default=None, min_length=12, max_length=256)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=12, max_length=256)


class NLPTextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


class NLPCompareRequest(BaseModel):
    reference: str = Field(min_length=1, max_length=10000)
    candidate: str = Field(min_length=1, max_length=10000)


class SequenceDemoRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


@app.post("/api/login")
def login(req: LoginRequest, request: Request, response: Response) -> dict[str, Any]:
    # handle_login raises 401/429 on failure; never logs credentials.
    return handle_login(request, response, req.password, req.username)


@app.get("/api/me")
def current_account(request: Request, _session: str = Depends(require_session)) -> dict[str, Any]:
    user = get_session_user(request)
    return {key: user[key] for key in ("username", "display_name", "role")}


@app.get("/api/deployment-readiness")
def deployment_readiness(_session: str = Depends(require_session)) -> dict[str, Any]:
    """Expose a non-secret checklist for the current local deployment profile."""
    report = readiness_report()
    audit = verify_audit_log()
    report["checks"].append({
        "id": "audit_ledger", "title": "Review audit ledger integrity",
        "status": "blocked" if not audit["valid"] else "review" if audit["entries"] == 0 else "pass",
        "detail": "The account's hash-chained review log failed integrity verification." if not audit["valid"] else f"Hash chain verified across {audit['entries']} events." if audit["entries"] else "No auditable changes have been recorded yet.",
    })
    report["production_ready"] = False
    return report


@app.get("/api/audit-log/integrity")
def audit_log_integrity(_session: str = Depends(require_session)) -> dict[str, Any]:
    """Return only ledger verification metadata, never the raw event log."""
    return verify_audit_log()


@app.get("/api/matters")
def list_matters(_session: str = Depends(require_session)) -> dict[str, Any]:
    data = load()
    matters = []
    for matter in data.get("matters", []):
        doc_ids = {str(doc.get("document_id")) for doc in data.get("documents", []) if doc.get("matter_id") == matter.get("matter_id")}
        matters.append({
            **matter,
            "document_count": len(doc_ids),
            "candidate_count": sum(str(item.get("document_id")) in doc_ids for item in data.get("items", [])),
            "claim_count": sum(str(item.get("document_id")) in doc_ids and item.get("kind") == "claim" for item in data.get("items", [])),
        })
    return {"matters": matters, "unassigned_document_count": sum(not doc.get("matter_id") for doc in data.get("documents", []))}


@app.post("/api/matters")
def create_matter(
    request: MatterCreateRequest,
    _session: str = Depends(require_session),
    _csrf: None = Depends(require_csrf),
) -> dict[str, Any]:
    title = " ".join(request.title.split())
    data = load()
    if any(str(matter.get("title", "")).casefold() == title.casefold() for matter in data.get("matters", [])):
        raise HTTPException(409, "A matter with that title already exists in this workspace.")
    now = datetime.now(timezone.utc).isoformat()
    matter = {"matter_id": uuid.uuid4().hex, "title": title, "description": request.description.strip(), "created_at": now, "updated_at": now}
    data.setdefault("matters", []).append(matter)
    save(data)
    record_audit_event("matter.created", "matter", matter["matter_id"], changes=["matter_created"])
    return {"matter": matter}


@app.patch("/api/matters/{matter_id}")
def update_matter(
    matter_id: str,
    request: MatterPatchRequest,
    _session: str = Depends(require_session),
    _csrf: None = Depends(require_csrf),
) -> dict[str, Any]:
    if request.title is None and request.description is None:
        raise HTTPException(422, "Provide a title or description to update.")
    data = load()
    matter = next((row for row in data.get("matters", []) if row.get("matter_id") == matter_id), None)
    if matter is None:
        raise HTTPException(404, "Matter not found.")
    if request.title is not None:
        title = " ".join(request.title.split())
        if any(row is not matter and str(row.get("title", "")).casefold() == title.casefold() for row in data.get("matters", [])):
            raise HTTPException(409, "A matter with that title already exists in this workspace.")
        matter["title"] = title
    if request.description is not None:
        matter["description"] = request.description.strip()
    matter["updated_at"] = datetime.now(timezone.utc).isoformat()
    save(data)
    record_audit_event("matter.updated", "matter", matter_id, changes=[key for key in ("title", "description") if getattr(request, key) is not None])
    return {"matter": matter}


@app.delete("/api/matters/{matter_id}")
def delete_matter(
    matter_id: str,
    _session: str = Depends(require_session),
    _csrf: None = Depends(require_csrf),
) -> dict[str, Any]:
    data = load()
    before = len(data.get("matters", []))
    data["matters"] = [row for row in data.get("matters", []) if row.get("matter_id") != matter_id]
    if len(data["matters"]) == before:
        raise HTTPException(404, "Matter not found.")
    detached = 0
    for document in data.get("documents", []):
        if document.get("matter_id") == matter_id:
            document["matter_id"] = None
            detached += 1
    save(data)
    record_audit_event("matter.deleted", "matter", matter_id, changes=["documents_detached"])
    return {"status": "deleted", "documents_detached": detached}


@app.patch("/api/documents/{document_id}/matter")
def assign_document_matter(
    document_id: str,
    request: DocumentMatterPatch,
    _session: str = Depends(require_session),
    _csrf: None = Depends(require_csrf),
) -> dict[str, Any]:
    data = load()
    document = next((doc for doc in data.get("documents", []) if doc.get("document_id") == document_id), None)
    if document is None:
        raise HTTPException(404, "Document not found in this workspace.")
    matter_id = request.matter_id
    if matter_id is not None and not any(row.get("matter_id") == matter_id for row in data.get("matters", [])):
        raise HTTPException(404, "Matter not found in this account workspace.")
    document["matter_id"] = matter_id
    save(data)
    record_audit_event("document.matter_assigned", "document", document_id, changes=["matter_assignment_changed"])
    return {"document": document}


@app.get("/api/users")
def admin_list_users(_admin_id: str = Depends(require_admin)) -> dict[str, Any]:
    return {"users": list_users()}


@app.post("/api/users")
def admin_create_user(
    req: CustomerCreateRequest,
    _admin_id: str = Depends(require_admin),
    _csrf: None = Depends(require_csrf),
) -> dict[str, Any]:
    user = create_customer(req.username, req.display_name, req.password)
    record_audit_event("account.created", "account", str(user.get("user_id", "")), changes=["customer_account_created"])
    return {"user": {key: user[key] for key in ("user_id", "username", "display_name", "role", "active", "created_at")}}


@app.patch("/api/users/{user_id}")
def admin_update_user(
    user_id: str,
    req: AccountAdminPatch,
    _admin_id: str = Depends(require_admin),
    _csrf: None = Depends(require_csrf),
) -> dict[str, str]:
    if req.active is None and req.password is None:
        raise HTTPException(422, "Provide active or a new password.")
    if req.active is not None:
        set_active(user_id, req.active, _admin_id)
    if req.password is not None:
        reset_password(user_id, req.password)
    record_audit_event("account.updated", "account", user_id, changes=[key for key in ("active", "password") if getattr(req, key) is not None])
    return {"status": "updated"}


@app.put("/api/me/password")
def update_own_password(
    req: PasswordChangeRequest,
    request: Request,
    _csrf: None = Depends(require_csrf),
) -> dict[str, str]:
    user = get_session_user(request)
    change_password(str(user["user_id"]), req.current_password, req.new_password)
    record_audit_event("account.password_changed", "account", str(user["user_id"]), changes=["password_changed"])
    return {"status": "password_changed"}


@app.post("/api/logout")
def logout(response: Response, _csrf: None = Depends(require_csrf)) -> dict[str, str]:
    clear_session(response)
    return {"status": "logged_out"}


@app.get("/api/csrf")
def csrf_token(request: Request, _session: str = Depends(require_session)) -> dict[str, str]:
    """Return the CSRF token for the current session (used by the frontend on load)."""
    token = get_csrf_token(request)
    return {"csrf": token}


# ── documents ─────────────────────────────────────────────────────────────────

@app.get("/api/documents")
def list_documents(matter_id: str | None = None, _session: str = Depends(require_session)) -> dict[str, Any]:
    data = load()
    changed = False
    for document in data["documents"]:
        report = document.get("processing_report")
        if isinstance(report, dict) and not report.get("coverage_scorecard"):
            document["processing_report"] = _upgrade_saved_report(report, document, data)
            changed = True
    if changed:
        save(data)
    return {"documents": _scoped_data(data, matter_id)["documents"]}


@app.post("/api/detect-language")
def language_endpoint(
    request: LanguageRequest,
    _session: str = Depends(require_session),
) -> dict[str, Any]:
    return {"language": detect_language(request.text)}


@app.get("/api/nlp-lab/status")
def nlp_lab_status(_session: str = Depends(require_session)) -> dict[str, Any]:
    return capability_status()


@app.get("/api/evaluation/property-civil")
def property_civil_evaluation(_session: str = Depends(require_session)) -> dict[str, Any]:
    """Expose the last offline synthetic benchmark summary to the authenticated UI."""
    report_path = ROOT / "data" / "processed" / "property_civil_benchmark" / "latest.json"
    if not report_path.is_file():
        raise HTTPException(404, "No saved property/civil benchmark is available yet.")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, "The saved benchmark report could not be read.") from exc
    method_metrics = {}
    for name, result in report.get("methods", {}).items():
        method_metrics[name] = {
            key: result.get(key)
            for key in (
                "positive_queries", "negative_queries", "hit_rate_at_5", "mrr_at_5",
                "recall_at_5", "ndcg_at_5", "supported_answer_rate",
                "negative_abstention_rate", "negative_false_answer_rate", "by_category",
            )
        }
    return {
        "benchmark": report.get("benchmark"),
        "generated_at": report.get("generated_at"),
        "corpus": report.get("corpus"),
        "queries": report.get("queries", 0),
        "positive_queries": report.get("positive_queries", 0),
        "negative_queries": report.get("negative_queries", 0),
        "semantic_backend": report.get("semantic_backend", {}),
        "reranker": report.get("reranker", {}),
        "methods": method_metrics,
        "limitations": report.get("limitations", []),
        "notice": "Small synthetic sample benchmark. It is not real-world legal accuracy validation.",
    }


@app.post("/api/nlp-lab/analyze")
def nlp_lab_analyze(req: NLPTextRequest, _session: str = Depends(require_session)) -> dict[str, Any]:
    return analyze_text(req.text)


@app.post("/api/nlp-lab/metrics")
def nlp_lab_metrics(req: NLPCompareRequest, _session: str = Depends(require_session)) -> dict[str, Any]:
    return evaluate_text(req.reference, req.candidate)


@app.post("/api/nlp-lab/sequence-demo")
def nlp_lab_sequence_demo(req: SequenceDemoRequest, _session: str = Depends(require_session)) -> dict[str, Any]:
    try:
        return classify_sequence(req.text)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.post("/api/documents")
async def upload_document(
    file: UploadFile = File(...),
    use_local_ocr: bool = Form(default=False),
    ocr_language: str = Form(default="auto"),
    matter_id: str = Form(default=""),
    _csrf: None = Depends(require_csrf),
) -> dict[str, Any]:
    name = Path(file.filename or "").name
    suffix = Path(name).suffix.casefold()
    supported = {".pdf", ".docx", ".docm", ".pptx", ".xlsx", ".txt", ".md", ".csv", ".json", ".html", ".htm", ".rtf"}
    if suffix not in supported:
        raise HTTPException(415, "Supported files: PDF, DOCX/DOCM, PPTX, XLSX, TXT, Markdown, CSV, JSON, HTML, and RTF. Save legacy .doc or scanned files as a supported text-readable format first.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Maximum upload size is 200 MB per file.")

    citation_unit = "page" if suffix in {".pdf", ".pptx", ".xlsx"} else "section"
    ocr_pages: list[int] = []
    ocr_skipped_pages: list[int] = []
    ocr_language_selection: dict[str, Any] | None = None
    if suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise HTTPException(400, "The file does not appear to be a valid PDF.")
        try:
            pages = extract_pages(content)
            needs_ocr = any(len("".join(str(page.get("text", "")).split())) < 40 for page in pages)
            if use_local_ocr and needs_ocr:
                initial_text = "\n".join(str(page.get("text", "")) for page in pages if str(page.get("text", "")).strip())
                ocr_language_selection = detect_ocr_language(content, initial_text[:20000])
                ocr_language = str(ocr_language_selection["language"])
                pages, ocr_pages, ocr_skipped_pages = extract_pages_with_ocr(content, language=ocr_language)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
    elif suffix == ".docx":
        try:
            pages = extract_docx_sections(content)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    elif suffix in {".docm", ".pptx", ".xlsx"}:
        try:
            pages = extract_office_sections(content, suffix)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
    else:
        try:
            pages = extract_text_sections(content, suffix)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    text_pages = [p for p in pages if str(p["text"]).strip()]
    if not text_pages:
        detail = ("No selectable text found. Enable local OCR for this PDF and ensure Tesseract is installed."
                  if suffix == ".pdf" else "No readable text was found in this document.")
        raise HTTPException(422, detail)

    doc_id = str(uuid.uuid4())
    data = load()
    if matter_id and not any(row.get("matter_id") == matter_id for row in data.get("matters", [])):
        raise HTTPException(404, "Selected matter does not exist in this account workspace.")
    uploads = workspace_uploads()
    uploads.mkdir(parents=True, exist_ok=True)
    stored = uploads / f"{doc_id}{suffix}"
    stored.write_bytes(content)

    language = detect_language(_analysis_sample(text_pages, limit=20000))
    record: dict[str, Any] = {
        "document_id": doc_id,
        "matter_id": matter_id or None,
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "provenance_version": 1,
        "ingestion_version": app.version,
        "filename": name,
        "stored_filename": stored.name,
        "file_type": suffix.lstrip("."),
        "citation_unit": citation_unit,
        "language": language,
        "pages": len(pages),
        "text_pages": len(text_pages),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready",
        "ocr_needed": suffix == ".pdf" and len(text_pages) < len(pages),
        "ocr_used": bool(ocr_pages),
        "ocr_language": ocr_language if ocr_pages else None,
        "ocr_language_detection": ocr_language_selection,
        "ocr_pages": ocr_pages,
        "ocr_skipped_pages": ocr_skipped_pages,
    }

    new_chunks = make_chunks(doc_id, name, pages)
    for chunk in new_chunks:
        chunk["citation_unit"] = citation_unit
    data["documents"].append(record)
    data["chunks"].extend(new_chunks)
    save_embeddings(doc_id, new_chunks)
    new_items = _run_extraction(doc_id, name, text_pages)
    for item in new_items:
        item["citation_unit"] = citation_unit
    data["items"].extend(new_items)
    _refresh_conflicts(data)
    corpus_chunks = [str(chunk.get("text", "")) for chunk in data.get("chunks", [])[:300]]
    report = _processing_report(name, suffix, pages, new_items, language, len(content), corpus_chunks)
    report["source_provenance"] = {
        "sha256": record["source_sha256"],
        "hash_stage": "original upload bytes before text extraction/OCR",
        "ingestion_version": record["ingestion_version"],
    }
    if ocr_pages:
        quality_by_page = [
            {"page": page_number, **ocr_quality_signals(str(pages[page_number - 1].get("text", "")))}
            for page_number in ocr_pages
        ]
        report["ocr"] = {
            "used": True,
            "engine": "Tesseract through PyMuPDF (local)",
            "language": ocr_language,
            "language_detection": ocr_language_selection,
            "pages": ocr_pages,
            "skipped_pages": ocr_skipped_pages,
            "quality_signals": quality_by_page,
            "notice": "OCR text can contain recognition errors; verify every finding against the original page image.",
        }
        if ocr_language_selection and ocr_language_selection.get("fallback"):
            report["warnings"].append(
                f"Automatic OCR language detection selected {ocr_language} because the detected document language pack was not installed locally."
            )
        flagged_pages = [str(entry["page"]) for entry in quality_by_page if entry["review_recommended"]]
        if flagged_pages:
            report["warnings"].append(
                "OCR text diagnostics recommend manual image review on pages: " + ", ".join(flagged_pages[:30]) + ". These are heuristic warning signals, not OCR confidence scores."
            )
        if ocr_skipped_pages:
            report["warnings"].append(
                f"OCR was limited to 50 low-text pages. Pages not OCR'd: {', '.join(map(str, ocr_skipped_pages[:20]))}."
            )
    record["processing_report"] = report
    save(data)
    record_audit_event("document.uploaded", "document", doc_id, changes=["source_hash_recorded", "text_extracted", "ocr_used" if ocr_pages else "native_text_path"])

    return {
        "document": record,
        "report": report,
        "extracted_items": len(new_items),
        "message": f"Document indexed. {len(new_items)} candidate items extracted.",
    }


@app.get("/api/documents/{document_id}/report")
def download_document_report(document_id: str, _session: str = Depends(require_session)) -> HTMLResponse:
    data = load()
    document = next((doc for doc in data.get("documents", []) if doc.get("document_id") == document_id), None)
    if document is None:
        raise HTTPException(404, "Document not found.")
    report = document.get("processing_report")
    if not isinstance(report, dict):
        raise HTTPException(404, "This document has no saved NLP report. Upload it again to create one.")
    if not report.get("coverage_scorecard"):
        report = _upgrade_saved_report(report, document, data)
        document["processing_report"] = report
        save(data)
    filename = f"LEGITMATE-report-{document_id[:8]}.html"
    return HTMLResponse(
        render_document_report(report),
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.get("/api/documents/{document_id}/integrity")
def document_integrity(document_id: str, _session: str = Depends(require_session)) -> dict[str, Any]:
    data = load()
    document = next((doc for doc in data.get("documents", []) if doc.get("document_id") == document_id), None)
    if document is None:
        raise HTTPException(404, "Document not found in this account workspace.")
    return {"document_id": document_id, **_document_integrity(document)}


@app.post("/api/documents/{document_id}/reprocess")
def reprocess_document(document_id: str, _csrf: None = Depends(require_csrf)) -> dict[str, Any]:
    """Re-run local extraction on the unchanged stored source without duplicating it."""
    data = load()
    document = next((doc for doc in data.get("documents", []) if doc.get("document_id") == document_id), None)
    if document is None:
        raise HTTPException(404, "Document not found in this account workspace.")
    integrity = _document_integrity(document)
    if not integrity.get("verified"):
        raise HTTPException(409, "Re-analysis requires the original file to match its upload-time SHA-256. Verify or restore the unchanged source first.")

    stored_name = Path(str(document.get("stored_filename", ""))).name
    source_path = workspace_uploads() / stored_name
    content = source_path.read_bytes()
    suffix = "." + str(document.get("file_type", Path(stored_name).suffix.lstrip("."))).casefold().lstrip(".")
    ocr_pages: list[int] = []
    ocr_skipped_pages: list[int] = []
    ocr_language_selection: dict[str, Any] | None = None
    try:
        if suffix == ".pdf":
            source_pages = extract_pages(content)
            if document.get("ocr_used") and any(len("".join(str(page.get("text", "")).split())) < 40 for page in source_pages):
                initial_text = "\n".join(str(page.get("text", "")) for page in source_pages if str(page.get("text", "")).strip())
                ocr_language_selection = detect_ocr_language(content, initial_text[:20000])
                pages, ocr_pages, ocr_skipped_pages = extract_pages_with_ocr(
                    content, language=str(ocr_language_selection["language"])
                )
            else:
                pages = source_pages
        elif suffix == ".docx":
            pages = extract_docx_sections(content)
        elif suffix in {".docm", ".pptx", ".xlsx"}:
            pages = extract_office_sections(content, suffix)
        else:
            pages = extract_text_sections(content, suffix)
    except ValueError as exc:
        raise HTTPException(422, f"Could not re-read the verified source: {exc}") from exc

    text_pages = [page for page in pages if str(page.get("text", "")).strip()]
    if not text_pages:
        raise HTTPException(422, "The verified source has no readable text sections to re-analyze.")

    previous = [item for item in data.get("items", []) if str(item.get("document_id", "")) == document_id]
    refreshed = _run_extraction(document_id, str(document.get("filename", "")), text_pages)
    citation_unit = str(document.get("citation_unit", "page"))
    for item in refreshed:
        item["citation_unit"] = citation_unit
    merged, preserved_reviewed = _merge_reanalysis_items(previous, refreshed)
    data["items"] = [item for item in data.get("items", []) if str(item.get("document_id", "")) != document_id] + merged

    language = detect_language(_analysis_sample(text_pages, limit=20000))
    corpus_chunks = [str(chunk.get("text", "")) for chunk in data.get("chunks", [])[:300]]
    report = _processing_report(str(document.get("filename", "")), suffix, pages, merged, language, len(content), corpus_chunks)
    report["source_provenance"] = {
        "sha256": str(document.get("source_sha256", "")),
        "hash_stage": "verified original upload bytes before text extraction/OCR",
        "ingestion_version": app.version,
    }
    report["reanalysis"] = {
        "performed_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256_verified": True,
        "review_annotations_preserved": preserved_reviewed,
    }
    if ocr_pages:
        quality_by_page = [
            {"page": number, **ocr_quality_signals(str(pages[number - 1].get("text", "")))}
            for number in ocr_pages
        ]
        report["ocr"] = {
            "used": True,
            "engine": "Tesseract through PyMuPDF (local)",
            "language": str(document.get("ocr_language") or "eng"),
            "language_detection": ocr_language_selection,
            "pages": ocr_pages,
            "skipped_pages": ocr_skipped_pages,
            "quality_signals": quality_by_page,
            "notice": "OCR text can contain recognition errors; verify every finding against the original page image.",
        }
        flagged_pages = [str(entry["page"]) for entry in quality_by_page if entry["review_recommended"]]
        if flagged_pages:
            report["warnings"].append("OCR text diagnostics recommend manual image review on pages: " + ", ".join(flagged_pages[:30]) + ". These are heuristic warning signals, not OCR confidence scores.")
        if ocr_skipped_pages:
            report["warnings"].append(f"OCR was limited to 50 low-text pages. Pages not OCR'd: {', '.join(map(str, ocr_skipped_pages[:20]))}.")

    document.update({
        "language": language,
        "pages": len(pages),
        "text_pages": len(text_pages),
        "ocr_needed": suffix == ".pdf" and len(text_pages) < len(pages),
        "ocr_pages": ocr_pages,
        "ocr_skipped_pages": ocr_skipped_pages,
        "ocr_language": str(ocr_language_selection.get("language")) if ocr_language_selection and ocr_pages else document.get("ocr_language"),
        "ocr_language_detection": ocr_language_selection or document.get("ocr_language_detection"),
        "processing_report": report,
        "ingestion_version": app.version,
    })
    _refresh_conflicts(data)
    save(data)
    record_audit_event("document.reanalyzed", "document", document_id, changes=["source_sha256_verified", "candidate_items_refreshed", "review_state_preserved"])
    return {
        "document_id": document_id,
        "filename": document.get("filename"),
        "pages": len(pages),
        "readable_pages": len(text_pages),
        "candidate_items": len(merged),
        "review_annotations_preserved": preserved_reviewed,
        "template_like": bool(report.get("template_diagnostics", {}).get("looks_like_unfilled_template")),
        "warnings": report.get("warnings", []),
        "status": "reprocessed",
    }


@app.delete("/api/documents/{document_id}")
def delete_document(
    document_id: str,
    _csrf: None = Depends(require_csrf),
) -> dict[str, str]:
    data = load()
    document = next((d for d in data["documents"] if d["document_id"] == document_id), None)
    if document is None:
        raise HTTPException(404, "Document not found.")
    data["documents"] = [d for d in data["documents"] if d["document_id"] != document_id]
    data["chunks"] = [c for c in data["chunks"] if c["document_id"] != document_id]
    data["items"] = [i for i in data["items"] if i["document_id"] != document_id]
    _refresh_conflicts(data)
    stored_filename = Path(document.get("stored_filename") or f"{document_id}.pdf").name
    (workspace_uploads() / stored_filename).unlink(missing_ok=True)
    delete_embeddings(document_id)
    save(data)
    record_audit_event("document.deleted", "document", document_id, changes=["source_and_index_removed"])
    return {"status": "deleted"}


# ── search / ask ──────────────────────────────────────────────────────────────

def _retrieve(question: str, chunks: list, top_k: int, mode: str) -> list:
    if mode == "semantic":
        return semantic_rank(question, chunks, top_k)
    if mode == "hybrid":
        return hybrid_rank(question, chunks, top_k)
    return rank(question, chunks, top_k)


def _retrieve_graph(question: str, data: dict[str, Any], top_k: int) -> list:
    return graph_augmented_rank(
        question, data, top_k, hybrid_rank=hybrid_rank, lexical_rank=rank,
        source_link=_source_link,
    )


def _retrieve_reranked(question: str, chunks: list, top_k: int) -> list:
    candidates = hybrid_rank(question, chunks, min(len(chunks), max(top_k * 2, top_k)))
    return rerank_hits(question, candidates)[:top_k]


@app.post("/api/search")
def search(
    request: SearchRequest,
    _session: str = Depends(require_session),
) -> dict[str, Any]:
    data = _scoped_data(load(), request.matter_id)
    if request.retrieval_mode == "graph":
        hits = _retrieve_graph(request.question, data, request.top_k)
    elif request.retrieval_mode == "reranked":
        hits = _retrieve_reranked(request.question, data["chunks"], request.top_k)
    else:
        hits = _retrieve(request.question, data["chunks"], request.top_k, request.retrieval_mode)
    return {"question": request.question, "results": hits, "count": len(hits)}


@app.post("/api/ask")
def ask(
    request: SearchRequest,
    _session: str = Depends(require_session),
) -> dict[str, Any]:
    data = _scoped_data(load(), request.matter_id)
    language = detect_language(request.question)
    if request.chat_mode == "live":
        if not request.use_ai:
            raise HTTPException(400, "Select the live AI chat option to send this message to the configured AI provider.")
        try:
            reply = generate_chat_reply(
                request.question,
                [turn.model_dump() for turn in request.chat_history],
                language,
            )
        except RuntimeError as exc:
            return {
                "question": request.question, "answer": "I couldn't reach the live AI right now.",
                "citations": [], "evidence": [], "abstained": False,
                "question_language": language, "ai_status": "error", "ai_message": str(exc),
            }
        return {
            "question": request.question, "answer": reply, "citations": [], "evidence": [],
            "abstained": False, "question_language": language, "ai_status": "groq_live", "ai_provider": provider_status(),
            "notice": "Live general chat. No document evidence was used; this is not legal advice.",
        }
    if request.retrieval_mode == "graph":
        hits = _retrieve_graph(request.question, data, request.top_k)
    elif request.retrieval_mode == "reranked":
        hits = _retrieve_reranked(request.question, data["chunks"], request.top_k)
    else:
        hits = _retrieve(request.question, data["chunks"], request.top_k, request.retrieval_mode)
    answer = build_answer(request.question, hits, data.get("conflicts", []))
    answer["question_language"] = language
    answer["ai_status"] = "not_requested"
    if request.use_ai:
        if answer.get("abstained"):
            if is_general_chat(request.question):
                try:
                    answer["answer"] = generate_general_reply(request.question, language)
                    answer["abstained"] = False
                    answer["ai_status"] = "groq_general"
                    answer["ai_provider"] = provider_status()
                    answer["notice"] = "General chat response. No document evidence was used; this is not legal advice."
                except RuntimeError as exc:
                    answer["ai_status"] = "error"
                    answer["ai_message"] = str(exc)
            else:
                answer["ai_status"] = "skipped_no_evidence"
                answer["ai_message"] = "No matching document evidence was found. Add or select a relevant document; LEGITMATE will not guess about legal facts."
        elif not provider_status()["configured"]:
            answer["ai_status"] = "not_configured"
            selected = provider_status()["provider"]
            variable = "GEMINI_API_KEY" if selected == "gemini" else "GROQ_API_KEY"
            answer["ai_message"] = f"Add {variable} to the private .env file and restart LEGITMATE to enable AI answers. Showing the local evidence answer."
        else:
            try:
                answer["answer"] = generate_grounded_answer(request.question, answer["evidence"], language)
                answer["ai_status"] = "groq"
                answer["ai_provider"] = provider_status()
            except RuntimeError as exc:
                answer["ai_status"] = "error"
                answer["ai_message"] = str(exc)
    return answer


# ── extracted items ───────────────────────────────────────────────────────────

@app.get("/api/items")
def list_items(
    kind: str | None = None,
    document_id: str | None = None,
    status: str | None = None,
    matter_id: str | None = None,
    _session: str = Depends(require_session),
) -> dict[str, Any]:
    items = _scoped_data(load(), matter_id)["items"]
    if kind:
        items = [i for i in items if i["kind"] == kind]
    if document_id:
        items = [i for i in items if i["document_id"] == document_id]
    if status:
        items = [i for i in items if i["review_status"] == status]
    return {"items": items, "count": len(items)}


@app.patch("/api/items/{item_id}")
def patch_item(
    item_id: str,
    patch: ReviewPatch,
    _csrf: None = Depends(require_csrf),
) -> dict[str, Any]:
    if not any(value is not None for value in patch.model_dump().values()):
        raise HTTPException(422, "Provide at least one review field to update.")
    data = load()
    for item in data["items"]:
        if item["item_id"] == item_id:
            if patch.review_status is not None:
                item["review_status"] = patch.review_status
                item["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            if patch.note is not None:
                item["note"] = patch.note
            if patch.value is not None:
                item["value"] = patch.value
            if patch.event_date is not None:
                item["event_date"] = patch.event_date
            if patch.claim_evidence_judgment is not None or patch.claim_evidence_refs is not None or patch.claim_evidence_note is not None:
                if item.get("kind") != "claim":
                    raise HTTPException(422, "Evidence judgments can only be attached to claim items.")
                if patch.claim_evidence_judgment is not None:
                    item["claim_evidence_judgment"] = patch.claim_evidence_judgment
                if patch.claim_evidence_refs is not None:
                    ref_ids = list(dict.fromkeys(patch.claim_evidence_refs))
                    refs_by_id = {str(candidate["item_id"]): candidate for candidate in data["items"]}
                    documents_by_id = {str(document["document_id"]): document for document in data.get("documents", [])}
                    claim_matter = documents_by_id.get(str(item.get("document_id")), {}).get("matter_id")
                    if item_id in ref_ids or any(
                        ref_id not in refs_by_id
                        or documents_by_id.get(str(refs_by_id[ref_id].get("document_id")), {}).get("matter_id") != claim_matter
                        for ref_id in ref_ids
                    ):
                        raise HTTPException(422, "Evidence references must point to another item in the same matter workspace.")
                    item["claim_evidence_refs"] = ref_ids
                if patch.claim_evidence_note is not None:
                    item["claim_evidence_note"] = patch.claim_evidence_note
                item["evidence_reviewed_at"] = datetime.now(timezone.utc).isoformat()
            _refresh_conflicts(data)
            save(data)
            record_audit_event("review.item_updated", "item", item_id, changes=list(patch.model_dump(exclude_unset=True).keys()))
            return {"item": item}
    raise HTTPException(404, "Item not found.")


@app.post("/api/items")
def add_item(
    req: ManualItemRequest,
    _csrf: None = Depends(require_csrf),
) -> dict[str, Any]:
    data = load()
    if not any(d["document_id"] == req.document_id for d in data["documents"]):
        raise HTTPException(404, "Document not found.")
    item: dict[str, Any] = {
        "item_id": str(uuid.uuid4()),
        "kind": req.kind,
        "value": req.value,
        "context": req.context,
        "document_id": req.document_id,
        "document": req.document,
        "page": req.page,
        "citation_unit": next(d.get("citation_unit", "page") for d in data["documents"] if d["document_id"] == req.document_id),
        "source_type": "manual",
        "review_status": "candidate",
        "event_date": req.event_date,
        "reviewed_at": None,
    }
    data["items"].append(item)
    _refresh_conflicts(data)
    save(data)
    return {"item": item}


@app.delete("/api/items/{item_id}")
def delete_item(
    item_id: str,
    _csrf: None = Depends(require_csrf),
) -> dict[str, str]:
    data = load()
    before = len(data["items"])
    data["items"] = [i for i in data["items"] if i["item_id"] != item_id]
    if len(data["items"]) == before:
        raise HTTPException(404, "Item not found.")
    _refresh_conflicts(data)
    save(data)
    return {"status": "deleted"}


# ── conflicts ─────────────────────────────────────────────────────────────────

@app.get("/api/conflicts")
def list_conflicts(matter_id: str | None = None, _session: str = Depends(require_session)) -> dict[str, Any]:
    data = _scoped_data(load(), matter_id)
    conflicts = data.get("conflicts", [])
    items_by_id = {i["item_id"]: i for i in data["items"]}
    enriched = []
    for c in conflicts:
        enriched.append({
            **c,
            "evidence": [
                {**items_by_id[iid], **_source_link(items_by_id[iid], data["chunks"])}
                for iid in c["items"] if iid in items_by_id
            ],
        })
    return {"conflicts": enriched, "count": len(enriched)}


@app.patch("/api/conflicts/{conflict_id}")
def resolve_conflict(
    conflict_id: str,
    status: str = "resolved",
    _csrf: None = Depends(require_csrf),
) -> dict[str, Any]:
    data = load()
    for c in data.get("conflicts", []):
        if c["conflict_id"] == conflict_id:
            c["status"] = status
            save(data)
            return {"conflict": c}
    raise HTTPException(404, "Conflict not found.")


# ── claims ────────────────────────────────────────────────────────────────────

@app.get("/api/claims")
def list_claims(matter_id: str | None = None, _session: str = Depends(require_session)) -> dict[str, Any]:
    data = _scoped_data(load(), matter_id)
    claims = [i for i in data["items"] if i["kind"] == "claim"]
    open_conflicts = [c for c in data.get("conflicts", []) if c.get("status") != "resolved"]

    item_conflict_map: dict[str, list[str]] = {}
    for c in open_conflicts:
        for iid in c.get("items", []):
            item_conflict_map.setdefault(iid, []).append(c["conflict_id"])

    enriched = []
    items_by_id = {i["item_id"]: i for i in data["items"]}
    for claim in claims:
        cids = item_conflict_map.get(claim["item_id"], [])
        source = _source_link(claim, data["chunks"], strict_value=True)
        enriched.append({
            **claim,
            **source,
            "supported": source["source_match"],
            "evidence_status": "exact_source_text_match" if source["source_match"] else "source_match_unavailable",
            "evidence_assessment": ("The extracted claim text matches an indexed passage on this source page; this confirms traceability, not factual truth."
                                    if source["source_match"] else "No exact indexed source passage was matched on this page; inspect the original document."),
            "conflicted": bool(cids),
            "conflict_ids": cids,
            "claim_evidence_judgment": claim.get("claim_evidence_judgment", "unreviewed"),
            "claim_evidence_items": [
                {**items_by_id[ref_id], **_source_link(items_by_id[ref_id], data["chunks"])}
                for ref_id in claim.get("claim_evidence_refs", []) if ref_id in items_by_id
            ],
        })

    return {"claims": enriched, "count": len(enriched)}


# ── timeline ──────────────────────────────────────────────────────────────────

@app.get("/api/timeline")
def timeline(matter_id: str | None = None, _session: str = Depends(require_session)) -> dict[str, Any]:
    events = build_timeline(_scoped_data(load(), matter_id)["items"])
    return {"events": events, "count": len(events)}


@app.get("/api/knowledge-graph")
def knowledge_graph(matter_id: str | None = None, _session: str = Depends(require_session)) -> dict[str, Any]:
    """Return a read-only graph view assembled from the existing local index."""
    return build_knowledge_graph(_scoped_data(load(), matter_id), _source_link)


@app.get("/api/case-brief")
def case_brief(
    document_id: str | None = None,
    matter_id: str | None = None,
    sentences_per_document: int = Query(default=4, ge=2, le=8),
    _session: str = Depends(require_session),
) -> dict[str, Any]:
    """Generate a local, source-linked extractive brief; does not call an AI provider."""
    data = _scoped_data(load(), matter_id)
    if document_id and not any(doc.get("document_id") == document_id for doc in data.get("documents", [])):
        raise HTTPException(404, "Document not found in the local index.")
    return build_case_brief(
        data, _source_link, document_id=document_id,
        sentences_per_document=sentences_per_document,
    )


@app.get("/api/evidence-matrix")
def evidence_matrix(
    matter_id: str | None = None,
    output: str = Query(default="json", pattern="^(json|csv)$"),
    _session: str = Depends(require_session),
) -> Any:
    data = _scoped_data(load(), matter_id)
    rows = _evidence_matrix(data)
    if output == "csv":
        return Response(
            content="\ufeff" + _matrix_csv(rows),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="LEGITMATE-evidence-matrix.csv"'},
        )
    return {"matter_id": matter_id, "rows": rows, "count": len(rows), "notice": "Source match means traceability, not truth. Evidence judgments are human review notes."}


@app.get("/api/case-bundle")
def download_case_bundle(
    matter_id: str | None = None,
    include_sources: bool = Query(default=False),
    _session: str = Depends(require_session),
) -> FileResponse:
    data = _scoped_data(load(), matter_id)
    matrix = _evidence_matrix(data)
    document_ids = {str(doc.get("document_id", "")) for doc in data.get("documents", [])}
    items = data.get("items", [])
    chunks = data.get("chunks", [])
    timeline_rows = build_timeline(items)
    evidence_rows = []
    for item in items:
        link = _source_link(item, chunks, strict_value=item.get("kind") == "claim")
        evidence_rows.append({
            "item_id": item.get("item_id"), "kind": item.get("kind"), "value": item.get("value"),
            "document": item.get("document"), "document_id": item.get("document_id"),
            "page": item.get("page"), "citation_unit": item.get("citation_unit", "page"),
            "review_status": item.get("review_status", "candidate"),
            "source_match": bool(link.get("source_match")), "source_chunk_id": link.get("source_chunk_id"),
        })
    conflicts = []
    by_id = {str(item.get("item_id")): item for item in items}
    for conflict in data.get("conflicts", []):
        conflicts.append({
            "conflict_id": conflict.get("conflict_id"), "kind": conflict.get("kind"),
            "label": conflict.get("label"), "status": conflict.get("status"),
            "evidence": [
                {"item_id": by_id[item_id].get("item_id"), "value": by_id[item_id].get("value"),
                 "document": by_id[item_id].get("document"), "page": by_id[item_id].get("page"),
                 "citation_unit": by_id[item_id].get("citation_unit", "page")}
                for item_id in map(str, conflict.get("items", [])) if item_id in by_id
            ],
        })
    selected_matter = next((row for row in data.get("matters", []) if row.get("matter_id") == matter_id), None) if matter_id and matter_id != _UNASSIGNED_MATTER else None
    title = selected_matter.get("title") if selected_matter else ("Unassigned documents" if matter_id == _UNASSIGNED_MATTER else "All workspace documents")
    case_data = {
        "project": "LEGITMATE",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "matter": selected_matter or {"matter_id": matter_id, "title": title},
        "notice": "Local research export. Extracted facts and candidate conflicts require human review. Not legal advice.",
        "documents": [{key: value for key, value in doc.items() if key not in {"stored_filename"}} for doc in data.get("documents", [])],
        "case_brief": build_case_brief(data, _source_link),
        "evidence_matrix": matrix,
        "evidence_index": evidence_rows,
        "timeline": timeline_rows,
        "conflicts": conflicts,
        "included_source_files": bool(include_sources),
    }
    export_root = workspace_processed() / "case_exports"
    export_root.mkdir(parents=True, exist_ok=True)
    export_path = export_root / f"case-bundle-{uuid.uuid4().hex}.zip"
    source_manifest: list[dict[str, Any]] = []
    source_bytes = 0
    max_source_bytes = 1_000_000_000
    with zipfile.ZipFile(export_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr("README.txt", (
            f"LEGITMATE case bundle\nMatter: {title}\nExported: {case_data['exported_at']}\n"
            f"Documents: {len(data.get('documents', []))}\nClaims in evidence matrix: {len(matrix)}\n\n"
            "This is a local research export, not legal advice. Candidate findings may be incomplete or incorrect. "
            "Verify every relevant statement against the cited original. CSV files guard spreadsheet-formula prefixes.\n"
            + ("Original source documents are included under sources/.\n" if include_sources else "Original source files are omitted; set include_sources=true to include them.\n")
        ))
        archive.writestr("case-data.json", json.dumps(case_data, ensure_ascii=False, indent=2))
        archive.writestr("evidence-matrix.csv", "\ufeff" + _matrix_csv(matrix))
        conflict_rows = [
            {"conflict_id": row.get("conflict_id"), "kind": row.get("kind"), "label": row.get("label"), "status": row.get("status"),
             "source_statements": " | ".join(f"{source.get('value')} ({source.get('document')}, {source.get('citation_unit')} {source.get('page')})" for source in row.get("evidence", []))}
            for row in conflicts
        ]
        for filename, rows, fields in (
            ("timeline.csv", timeline_rows, ["event_date", "event_type", "value", "document", "citation_unit", "page", "review_status", "interval_role"]),
            ("evidence-index.csv", evidence_rows, ["item_id", "kind", "value", "document", "citation_unit", "page", "review_status", "source_match", "source_chunk_id"]),
            ("conflicts.csv", conflict_rows, ["conflict_id", "kind", "label", "status", "source_statements"]),
        ):
            target = io.StringIO(newline="")
            writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: _csv_safe(row.get(field)) for field in fields})
            archive.writestr(filename, "\ufeff" + target.getvalue())
        source_writer_buffer = io.StringIO(newline="")
        source_writer = csv.DictWriter(source_writer_buffer, fieldnames=["document_id", "filename", "sha256", "included", "note"])
        source_writer.writeheader()
        for doc in data.get("documents", []):
            stored_name = Path(str(doc.get("stored_filename", ""))).name
            source_path = workspace_uploads() / stored_name
            digest = str(doc.get("source_sha256", ""))
            included = False
            note = "Source omitted by export option; ingestion hash retained when available."
            if include_sources:
                try:
                    resolved = source_path.resolve(strict=True)
                    if resolved.parent != workspace_uploads().resolve() or not resolved.is_file():
                        note = "Source file path was not a regular file within this account workspace."
                    else:
                        stat_size = resolved.stat().st_size
                        if source_bytes + stat_size > max_source_bytes:
                            note = "Source omitted because the 1 GB export source cap would be exceeded."
                        else:
                            hasher = hashlib.sha256()
                            with resolved.open("rb") as source:
                                for block in iter(lambda: source.read(1024 * 1024), b""):
                                    hasher.update(block)
                            export_digest = hasher.hexdigest()
                            if digest and export_digest != digest:
                                note = "Source excluded because its SHA-256 does not match the ingestion-time manifest."
                            else:
                                legacy_hash = not digest
                                digest = export_digest
                                safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(str(doc.get("filename", "source"))).name).strip("._") or "source"
                                archive.write(resolved, f"sources/{str(doc.get('document_id', 'doc'))[:8]}_{safe_name}")
                                included = True
                                source_bytes += stat_size
                                note = "Included; legacy source has export-time hash only." if legacy_hash else "Included; ingestion-time SHA-256 verified."
                except OSError:
                    note = "Source file is missing or unreadable in this account workspace."
            source_manifest.append({"document_id": doc.get("document_id"), "filename": Path(str(doc.get("filename", ""))).name, "sha256": digest, "included": included, "note": note})
            source_writer.writerow({key: _csv_safe(value) for key, value in source_manifest[-1].items()})
        archive.writestr("source-manifest.csv", "\ufeff" + source_writer_buffer.getvalue())
    download_name = "LEGITMATE-case-bundle-" + re.sub(r"[^A-Za-z0-9_-]+", "-", str(title)).strip("-")[:50] + ".zip"
    return FileResponse(
        export_path,
        media_type="application/zip",
        filename=download_name,
        background=BackgroundTask(lambda: export_path.unlink(missing_ok=True)),
    )


# ── static ────────────────────────────────────────────────────────────────────

@app.get("/login", include_in_schema=False)
def login_page() -> FileResponse:
    return FileResponse(ROOT / "login.html")


@app.get("/brand/legitmate-mark.png", include_in_schema=False)
def brand_mark() -> FileResponse:
    """Serve the bundled public brand mark used by the login and app headers."""
    return FileResponse(
        ROOT / "static" / "brand" / "legitmate-mark.png",
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/", include_in_schema=False, response_model=None)
def home(request: Request) -> FileResponse | HTMLResponse:
    # If not authenticated, redirect to login
    if get_csrf_token(request) is None:
        return HTMLResponse(
            '<html><head><meta http-equiv="refresh" content="0;url=/login"></head></html>',
            status_code=302,
            headers={"Location": "/login"},
        )
    return FileResponse(ROOT / "index.html")
