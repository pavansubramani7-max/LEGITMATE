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

# ── property identifier patterns (regex only) ────────────────────────────────
# A property label is not itself an identifier. Require an explicit number/id
# cue or a digit-bearing value so headings such as "SALE DEED FOR FLAT THIS"
# and unfilled fields such as "Plot No. ______" do not become property facts.
_IDENTIFIER = r"(?=[A-Za-z0-9./-]*\d)[A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)*"
_PROP_RE = re.compile(
    rf"\b(?:Plot|Survey|Khasra|Khata|Flat|Unit|House|Parcel)"
    rf"\s+(?:bearing\s+)?(?:No\.?|Number|#)?\s*{_IDENTIFIER}\b",
    re.IGNORECASE,
)
_PROPERTY_ID_RE = re.compile(
    rf"\bProperty\s+(?:ID|No\.?|Number|#)\s*{_IDENTIFIER}\b", re.IGNORECASE
)

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
    rf"\b((?:[A-Z][A-Za-z&.-]*\s+){{1,4}}(?:{_ORG_SUFFIXES})\.?)\b"
)
_ORG_SUFFIX_WORDS = frozenset(word.casefold() for word in "Ltd LLC Inc Corp LLP Trust Association Society Bank Authority Council Foundation Institute University College Hospital Cooperative Committee".split())
_ORG_GENERIC_WORDS = frozenset(
    "housing apartment apartments residents owners welfare cooperative co-operative "
    "municipal village gram office department revenue urban land registration "
    "district association society corporation committee authority council".split()
)

_SKIP_NAMES = frozenset(
    "This That These Those Such Said Above Below Whereas Therefore Herein Hereto The "
    "Schedule Exhibit Annexure Clause Section Article Party Parties Seller Buyer "
    "Lessor Lessee Vendor Purchaser Witness Witnesses Signed Executed From To Subject "
    "Dear Notice Legal Sale Deed Plot Survey Flat Unit House Parcel Property Rupees "
    "Apartments Apartment No Number Ward Block Municipal Corporation Gift Settlement "
    "Document Volume Development Agreement Permit File Gram Panchayat Revenue Records "
    "Indian Stamp Urban Agglomeration Land Ceiling Village Rate Per Market Value "
    "Registration District Society Association Authority Department Government Rules "
    "Act Section Schedule Schedule Property M/S SRO Whereas Hereinafter "
    "Stilt Floor Floors Assigned Lands Muncipal Corportation Municipal Corporation "
    "Plinth Area With Common Built Constructed Superstructure Measurement "
    "Five Fifty Thousand Lakh Crore One Two Three Four Six Seven Eight Nine Ten "
    "Green Lotus Colony Avenue Road Street Layout Sector Jaipur Bangalore Rajasthan".split()
)
_SKIP_NAMES_CASEFOLD = frozenset(value.casefold() for value in _SKIP_NAMES)


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
        if ent.label_ == "PERSON" and not _is_noise_entity(ent.text, "person"):
            items.append(_item("person", ent.text, _ctx(text, ent.start_char, ent.end_char), doc_id, doc_name, page))
        elif ent.label_ == "ORG" and not _is_noise_entity(ent.text, "org"):
            items.append(_item("org", ent.text, _ctx(text, ent.start_char, ent.end_char), doc_id, doc_name, page))
    return items


def _is_noise_entity(value: str, kind: str) -> bool:
    """Reject common legal-form labels and placeholder text as named entities."""
    if re.search(r"_{2,}|\.{4,}|\[\s*\]", value):
        return True
    words = [word.casefold() for word in re.findall(r"[A-Za-z]+", value)]
    if kind == "person":
        return len(words) < 2 or any(word in _SKIP_NAMES_CASEFOLD for word in words)
    if kind == "org":
        # A suffix alone or a phrase made only of generic institution labels is
        # not enough evidence for an organization name.
        return not any(word not in _ORG_GENERIC_WORDS and word not in _ORG_SUFFIX_WORDS for word in words)
    return False


def _extract_regex(doc_id: str, doc_name: str, page: int, text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for m in _NAME_RE.finditer(text):
        words = re.findall(r"[A-Za-z]+", m.group(0))
        if _is_noise_entity(m.group(0), "person") or any(word.casefold() in _ORG_SUFFIX_WORDS for word in words):
            continue
        items.append(_item("person", m.group(0), _ctx(text, m.start(), m.end()), doc_id, doc_name, page))
    for m in _ORG_RE.finditer(text):
        if _is_noise_entity(m.group(0), "org"):
            continue
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
