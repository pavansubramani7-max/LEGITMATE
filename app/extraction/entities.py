"""Entity extraction for LEGITMATE.

Strategy (in priority order):
1. If spaCy en_core_web_sm is available, use it for PERSON and ORG labels.
2. Otherwise use enhanced regex with honorific-prefix anchoring and a broader
   org-suffix list to reduce false positives from the original capitalised-pair
   pattern.
3. PROPERTY identifiers always use regex (spaCy does not cover Indian legal
   property references).
4. Entities are deduplicated per (document_id, kind, normalised_value) so
   repeated mentions do not flood the review panel.
"""
from __future__ import annotations

import re
import uuid
from typing import Any

from .ner_model import get_nlp

# ── property pattern (regex only) ────────────────────────────────────────────
_PROP_RE = re.compile(
    r"\b(?:Plot|Survey|Khasra|Khata|Flat|Unit|House|Parcel)"
    r"\s*(?:No\.?|Number|#)?\s*[\w\-/]+\b",
    re.IGNORECASE,
)
_PROPERTY_ID_RE = re.compile(r"\bProperty\s+(?:ID|No\.?|Number|#)\s*[\w\-/]+\b", re.IGNORECASE)

# ── enhanced regex fallback ───────────────────────────────────────────────────
# Honorific prefixes common in Indian legal documents
_HONORIFICS = (
    r"(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?|Shri\.?|Smt\.?|Kumari\.?|Sri\.?)\s+"
)
# Name: optional honorific + 2–3 capitalised words
_NAME_RE = re.compile(
    rf"(?:{_HONORIFICS})?([A-Z][a-z]{{1,20}})(?:\s+[A-Z][a-z]{{1,20}}){{1,2}}\b"
)

_ORG_SUFFIXES = (
    r"Ltd|LLC|Inc|Corp|LLP|Trust|Association|Society|Bank|Authority|Council"
    r"|Foundation|Institute|University|College|Hospital|Cooperative|Committee"
)
_ORG_RE = re.compile(
    rf"\b((?:[A-Z][A-Za-z&.-]*\s+){{0,4}}(?:{_ORG_SUFFIXES})\.?)\b"
)
_ORG_SUFFIX_WORDS = frozenset("Ltd LLC Inc Corp LLP Trust Association Society Bank Authority Council Foundation Institute University College Hospital Cooperative Committee".split())

_SKIP_NAMES = frozenset(
    "This That These Those Such Said Above Below Whereas Therefore Herein Hereto The "
    "Schedule Exhibit Annexure Clause Section Article Party Parties Seller Buyer "
    "Lessor Lessee Vendor Purchaser Witness Witnesses Signed Executed From To Subject "
    "Dear Notice Legal Sale Deed Plot Survey Flat Unit House Parcel Property Rupees "
    "Apartments Apartment "
    "Five Fifty Thousand Lakh Crore One Two Three Four Six Seven Eight Nine Ten "
    "Green Lotus Colony Avenue Road Street Layout Sector Jaipur Bangalore Rajasthan".split()
)


def _ctx(text: str, start: int, end: int, window: int = 80) -> str:
    return text[max(0, start - window): end + window].replace("\n", " ").strip()


def _item(kind: str, value: str, ctx: str, doc_id: str, doc_name: str, page: int) -> dict[str, Any]:
    return {
        "item_id": str(uuid.uuid4()),
        "kind": kind,
        "value": value.strip(),
        "context": ctx,
        "document_id": doc_id,
        "document": doc_name,
        "page": page,
        "review_status": "candidate",
    }


def _dedup(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate (document_id, kind, normalised value) entries, keeping first."""
    seen: set[tuple[str, str, str]] = set()
    out = []
    for i in items:
        key = (i["document_id"], i["kind"], re.sub(r"\s+", " ", i["value"].casefold()))
        if key not in seen:
            seen.add(key)
            out.append(i)
    return out


def _extract_spacy(nlp, doc_id: str, doc_name: str, page: int, text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            items.append(_item("person", ent.text, _ctx(text, ent.start_char, ent.end_char), doc_id, doc_name, page))
        elif ent.label_ == "ORG":
            items.append(_item("org", ent.text, _ctx(text, ent.start_char, ent.end_char), doc_id, doc_name, page))
    return items


def _extract_regex(doc_id: str, doc_name: str, page: int, text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for m in _NAME_RE.finditer(text):
        first_cap = m.group(1)
        words = re.findall(r"[A-Za-z]+", m.group(0))
        if (
            first_cap in _SKIP_NAMES
            or any(word in _SKIP_NAMES for word in words[1:])
            or any(word.rstrip(".") in _ORG_SUFFIX_WORDS for word in words)
        ):
            continue
        items.append(_item("person", m.group(0), _ctx(text, m.start(), m.end()), doc_id, doc_name, page))
    for m in _ORG_RE.finditer(text):
        items.append(_item("org", m.group(0), _ctx(text, m.start(), m.end()), doc_id, doc_name, page))
    return items


def extract_entities(doc_id: str, doc_name: str, page: int, text: str) -> list[dict[str, Any]]:
    nlp = get_nlp()
    if nlp is not None:
        items = _extract_spacy(nlp, doc_id, doc_name, page, text)
    else:
        items = _extract_regex(doc_id, doc_name, page, text)

    # PROPERTY always uses regex
    for m in (*_PROP_RE.finditer(text), *_PROPERTY_ID_RE.finditer(text)):
        items.append(_item("property", m.group(0), _ctx(text, m.start(), m.end()), doc_id, doc_name, page))

    return _dedup(items)
