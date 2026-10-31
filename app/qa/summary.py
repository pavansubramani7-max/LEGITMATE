"""Source-linked extractive brief generation for the LEGITMATE workspace.

This module ranks sentences that already exist in uploaded-document chunks. It
never writes new factual prose or decides which side of a conflict is correct.
"""
from __future__ import annotations

import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"\b[a-zA-Z][a-zA-Z0-9'-]{2,}\b")
_MAX_DOCS = 40
_MAX_SENTENCES_PER_DOC = 1200
_SUMMARY_SENTENCES = 4


def _rank_sentences(chunks: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        for sentence in _SENTENCE_SPLIT.split(text):
            sentence = " ".join(sentence.split()).strip(" \t\r\n\u2022-")
            if len(sentence) < 35 or len(sentence) > 700:
                continue
            key = sentence.casefold()
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "text": sentence,
                "document": chunk.get("document"),
                "document_id": chunk.get("document_id"),
                "page": chunk.get("page"),
                "citation_unit": chunk.get("citation_unit", "page"),
                "chunk_id": chunk.get("chunk_id"),
                "ordinal": len(candidates),
            })
            if len(candidates) >= _MAX_SENTENCES_PER_DOC:
                break
        if len(candidates) >= _MAX_SENTENCES_PER_DOC:
            break
    if not candidates:
        return []

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=5000,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9'-]+\b",
        )
        matrix = vectorizer.fit_transform([row["text"] for row in candidates])
        lengths = [max(1, len(_WORD.findall(row["text"]))) for row in candidates]
        scores = (matrix.sum(axis=1).A1 / [length ** 0.5 for length in lengths]).tolist()
    except ValueError:
        scores = [0.0] * len(candidates)

    for row, score in zip(candidates, scores):
        row["score"] = round(float(score), 6)
    selected = sorted(candidates, key=lambda row: (-row["score"], row["ordinal"]))[:count]
    return [
        {key: row[key] for key in ("text", "document", "document_id", "page", "citation_unit", "chunk_id", "score")}
        for row in sorted(selected, key=lambda row: row["ordinal"])
    ]


def build_case_brief(
    data: dict[str, Any],
    source_link,
    *,
    document_id: str | None = None,
    sentences_per_document: int = _SUMMARY_SENTENCES,
) -> dict[str, Any]:
    documents = [doc for doc in data.get("documents", []) if not document_id or doc.get("document_id") == document_id]
    if document_id is None:
        documents.sort(key=lambda doc: str(doc.get("uploaded_at", "")), reverse=True)
    if len(documents) > _MAX_DOCS:
        documents = documents[:_MAX_DOCS]
    allowed_ids = {str(doc.get("document_id")) for doc in documents}
    chunks_by_doc: dict[str, list[dict[str, Any]]] = {doc_id: [] for doc_id in allowed_ids}
    for chunk in data.get("chunks", []):
        if str(chunk.get("document_id")) in chunks_by_doc:
            chunks_by_doc[str(chunk["document_id"])].append(chunk)

    items_by_doc: dict[str, list[dict[str, Any]]] = {doc_id: [] for doc_id in allowed_ids}
    items_by_id: dict[str, dict[str, Any]] = {}
    for item in data.get("items", []):
        item_id = str(item.get("item_id", ""))
        items_by_id[item_id] = item
        doc_id = str(item.get("document_id", ""))
        if doc_id in items_by_doc and item.get("review_status") != "rejected":
            items_by_doc[doc_id].append(item)

    doc_sections = []
    for doc in documents:
        doc_id = str(doc.get("document_id", ""))
        facts = []
        for item in items_by_doc.get(doc_id, []):
            source = source_link(item, data.get("chunks", []), strict_value=item.get("kind") == "claim")
            facts.append({
                "kind": item.get("kind"), "value": item.get("value"), "context": item.get("context"),
                "page": item.get("page"), "citation_unit": item.get("citation_unit", "page"),
                "review_status": item.get("review_status", "candidate"),
                "source_match": bool(source.get("source_match")),
                "source_passage": source.get("source_passage"),
                "chunk_id": source.get("source_chunk_id"),
            })
        doc_sections.append({
            "document_id": doc_id,
            "filename": doc.get("filename", "Unknown document"),
            "file_type": doc.get("file_type"),
            "pages": doc.get("pages", 0),
            "uploaded_at": doc.get("uploaded_at"),
            "summary_sentences": _rank_sentences(chunks_by_doc.get(doc_id, []), sentences_per_document),
            "candidate_facts": facts[:60],
        })

    conflicts = []
    for conflict in data.get("conflicts", []):
        if conflict.get("status") == "resolved":
            continue
        evidence = []
        for item_id in conflict.get("items", []):
            item = items_by_id.get(str(item_id))
            if not item or str(item.get("document_id")) not in allowed_ids:
                continue
            source = source_link(item, data.get("chunks", []))
            evidence.append({
                "value": item.get("value"), "document": item.get("document"),
                "page": item.get("page"), "citation_unit": item.get("citation_unit", "page"),
                "source_match": bool(source.get("source_match")),
                "source_passage": source.get("source_passage"),
            })
        if len(evidence) >= 2:
            conflicts.append({"kind": conflict.get("kind"), "label": conflict.get("label"), "evidence": evidence})

    timeline = [
        item for item in data.get("items", [])
        if item.get("kind") in {"date", "event"}
        and item.get("event_date")
        and item.get("review_status") != "rejected"
        and str(item.get("document_id")) in allowed_ids
    ]
    timeline.sort(key=lambda item: (str(item.get("event_date")), str(item.get("document", ""))))

    return {
        "document_count": len(doc_sections),
        "page_or_section_count": sum(int(doc.get("pages", 0) or 0) for doc in doc_sections),
        "summary_method": "Local TF-IDF extractive sentence ranking",
        "documents": doc_sections,
        "open_conflicts": conflicts,
        "timeline": [
            {"date": item.get("event_date"), "value": item.get("value"), "document": item.get("document"),
             "page": item.get("page"), "citation_unit": item.get("citation_unit", "page"),
             "review_status": item.get("review_status", "candidate")}
            for item in timeline[:150]
        ],
        "limited_to_recent_documents": len(data.get("documents", [])) > len(doc_sections) and document_id is None,
        "notice": "Extractive research aid: sentences are selected from source text, not rewritten. Candidate facts require human review; conflicts are shown without choosing a winner.",
    }
