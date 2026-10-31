from __future__ import annotations
import re
from typing import Any

from ..config import ABSTAIN_THRESHOLD, NEURAL_ABSTAIN_THRESHOLD


# ── natural-language answer composer ─────────────────────────────────────────

_QUESTION_PATTERNS = [
    (re.compile(r"\b(sale\s*price|consideration|amount|cost|price)\b", re.I),
     "price"),
    (re.compile(r"\b(date|when|signed|executed|entered)\b", re.I),
     "date"),
    (re.compile(r"\b(part(y|ies)|seller|buyer|lessor|lessee|vendor|purchaser|who)\b", re.I),
     "parties"),
    (re.compile(r"\b(property|plot|survey|flat|unit|address|located|situated)\b", re.I),
     "property"),
    (re.compile(r"\b(term|duration|period|lease|months?|years?)\b", re.I),
     "term"),
    (re.compile(r"\b(payment|pay|deposit|instalment|due)\b", re.I),
     "payment"),
    (re.compile(r"\b(condition|clause|covenant|warrant|obligation|shall|must)\b", re.I),
     "condition"),
]

_INTROS = {
    "price":     "Based on the documents, the relevant consideration or price information is as follows:",
    "date":      "According to the documents, here is what I found regarding the date:",
    "parties":   "The documents identify the following parties:",
    "property":  "The property described in the documents is:",
    "term":      "Regarding the term or duration, the documents state:",
    "payment":   "On the matter of payment, the documents indicate:",
    "condition": "The relevant conditions or obligations stated in the documents are:",
    "default":   "Based on the uploaded documents, here is the most relevant information I found:",
}


def _detect_intent(question: str) -> str:
    for pattern, intent in _QUESTION_PATTERNS:
        if pattern.search(question):
            return intent
    return "default"


def _clean_passage(text: str) -> str:
    """Normalise whitespace and trim passage text."""
    return re.sub(r"\s+", " ", text).strip()


def _source_location(hit: dict[str, Any]) -> str:
    unit = "section" if hit.get("citation_unit") == "section" else "page"
    return f"{unit} {hit.get('page', '?')}"


def _compose_answer(question: str, hits: list[dict[str, Any]]) -> str:
    """Produce a natural, ChatGPT-style answer grounded in retrieved passages."""
    intent = _detect_intent(question)
    intro = _INTROS[intent]

    best = hits[0]
    passage = _clean_passage(best["text"])
    doc = best["document"]
    location = _source_location(best)

    # Build the main answer paragraph
    lines = [intro, ""]
    lines.append(f"“{passage}”")
    lines.append(f"\u2014 {doc}, {location}")

    # If there are more hits from the same or different docs, summarise them
    extra = hits[1:4]  # up to 3 more
    if extra:
        lines.append("")
        lines.append("Additional relevant passages:")
        for h in extra:
            p = _clean_passage(h["text"])
            # Truncate long passages for readability
            if len(p) > 220:
                p = p[:220].rsplit(" ", 1)[0] + "…"
            lines.append(f"\u2022 “{p}” ({h['document']}, {_source_location(h)})")

    lines.append("")
    lines.append(
        "\u26a0\ufe0f This answer is drawn directly from the uploaded documents. "
        "It is a research aid only \u2014 not legal advice. Always verify against the original source."
    )
    return "\n".join(lines)


def _conflict_note(hits: list[dict[str, Any]], conflicts: list[dict[str, Any]]) -> str | None:
    """Return a conflict note if any open conflict involves documents cited in hits."""
    cited_docs = {h["document_id"] for h in hits}

    # Path 1: open conflict in the store touches ≥2 cited documents
    # (requires caller to pass conflicts from the store)
    if conflicts:
        cited_items_by_doc: dict[str, set[str]] = {}
        for h in hits:
            cited_items_by_doc.setdefault(h["document_id"], set())
        for c in conflicts:
            if c.get("status") == "resolved":
                continue
            # conflict record carries item_ids; we only need to know if it
            # spans documents that both appear in the cited set
            # The evidence list on the conflict carries document_id directly
            evidence = c.get("evidence", [])
            conflict_docs = {e["document_id"] for e in evidence} if evidence else set()
            if len(conflict_docs & cited_docs) >= 2:
                return (
                    "Possible conflict — relevant passages found in multiple documents. "
                    "Review each source before drawing conclusions."
                )

    # Path 2: hits themselves span multiple documents (no store needed)
    if len(cited_docs) > 1:
        return (
            "Possible conflict — relevant passages found in multiple documents. "
            "Review each source before drawing conclusions."
        )

    return None


def _above_threshold(hits: list[dict[str, Any]], threshold_override: float | None = None) -> list[dict[str, Any]]:
    """Keep only hits whose relevance meets the abstention threshold.

    Score scales differ by retrieval mode:
      BM25   — typically 0.5–3.0; compared directly against ABSTAIN_THRESHOLD.
      TF-IDF — cosine similarity 0–1; compared directly against ABSTAIN_THRESHOLD.
      Neural — cosine similarity, but compressed into ~0.70–0.85 even for
               unrelated queries.  Uses NEURAL_ABSTAIN_THRESHOLD (0.80).
      Hybrid — RRF position score (~0.01–0.03) carries no absolute relevance
               signal.  Each hybrid hit carries `best_underlying_score` — the
               maximum raw BM25 / TF-IDF / neural score for that chunk.
               The threshold applied is chosen based on the semantic backend.

    A hit is kept only when its effective quality score ≥ the applicable threshold.
    """
    if not hits:
        return []

    def _quality(h: dict[str, Any]) -> float:
        return h.get("best_underlying_score", h["score"])

    if threshold_override is not None:
        return [h for h in hits if _quality(h) >= threshold_override]

    def _threshold(h: dict[str, Any]) -> float:
        retrieval = h.get("retrieval", "")
        if retrieval in ("neural", "hybrid-neural"):
            return NEURAL_ABSTAIN_THRESHOLD
        return ABSTAIN_THRESHOLD

    return [h for h in hits if _quality(h) >= _threshold(h)]


def build_answer(
    question: str,
    hits: list[dict[str, Any]],
    conflicts: list[dict[str, Any]] | None = None,
    abstain_threshold: float | None = None,
) -> dict[str, Any]:
    hits = _above_threshold(hits, abstain_threshold)
    if not hits:
        return {
            "question": question,
            "answer": "Insufficient evidence in the uploaded documents.",
            "citations": [],
            "evidence": [],
            "abstained": True,
        }

    conflict_note = _conflict_note(hits, conflicts or [])

    best = hits[0]
    cited_doc_ids = {h["document_id"] for h in hits}
    answer = _compose_answer(question, hits)
    if len(hits) > 1 and len(cited_doc_ids) > 1:
        answer += f'\n\nI found relevant passages across {len(cited_doc_ids)} documents. Review each source carefully as they may present different perspectives on the same matter.'

    citations = [
        {"document": h["document"], "document_id": h["document_id"], "page": h["page"],
         "citation_unit": h.get("citation_unit", "page"), "chunk_id": h["chunk_id"]}
        for h in hits
    ]

    result: dict[str, Any] = {
        "question": question,
        "answer": answer,
        "citations": citations,
        "evidence": hits,
        "abstained": False,
        "notice": "Research aid only. Verify the cited source; this is not legal advice.",
    }
    if conflict_note:
        result["conflict_note"] = conflict_note
    return result
