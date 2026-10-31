from __future__ import annotations
import re
import uuid
from typing import Any

# Sentences containing legal claim keywords
_CLAIM_TRIGGERS = re.compile(
    r"\b(agrees?|agreed|shall|must|warrants?|represents?|covenants?|undertakes?|"
    r"transfers?|conveys?|grants?|assigns?|sells?|purchases?|pays?|paid|owes?|"
    r"claims?|disputes?|alleges?|states?|declares?|confirms?|uses?|maintains?|"
    r"sublets?|assigns?|deliver(?:s|ed)?|occup(?:y|ies)|vacates?|discrepancy|discrepancies)\b|\bnot\s+(?:sublet|assign|transfer)\b",
    re.IGNORECASE,
)

_BOUNDARY_RE = re.compile(r"[.!?]+[\"'”’)]*(?=\s+|$)")
_ABBREVIATIONS = frozenset({"rs", "no", "mr", "mrs", "ms", "dr", "st", "vs", "etc", "i", "e"})


def _sentences(text: str) -> list[str]:
    """Split prose without treating common legal abbreviations or list numbers as sentence ends."""
    out: list[str] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(text):
        punctuation = next((char for char in text[match.start():match.end()] if char in ".!?"), "")
        if punctuation == ".":
            before = text[:match.start()]
            token_match = re.search(r"([\w]+)$", before.rstrip())
            token = token_match.group(1) if token_match else ""
            after = text[match.end():].lstrip()
            line_before_dot = text[:match.start()].splitlines()[-1].strip()
            if token.isdigit() and line_before_dot == token and after[:1].isupper():
                prefix_end = len(before.rstrip())
                token_start = prefix_end - len(token)
                previous = text[start:token_start].strip()
                if previous:
                    out.append(previous)
                start = match.end()
                continue
            # Rs., No., honorifics, initials, decimals, and numbered clauses
            # are internal punctuation, not sentence boundaries.
            if token.casefold() in _ABBREVIATIONS or (len(token) == 1 and token.isupper()):
                continue
            if match.start() > 0 and match.start() + 1 < len(text):
                if text[match.start() - 1].isdigit() and text[match.start() + 1].isdigit():
                    continue
        sentence = text[start:match.end()].strip()
        if sentence:
            out.append(sentence)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _remove_document_preamble(sentence: str) -> str:
    """Drop salutation and standalone all-caps section headings from a claim span."""
    sentence = re.sub(r"(?is)^.*?\bDear\b[^,:\n]*[, :]\s*", "", sentence, count=1)
    lines = sentence.splitlines()
    while len(lines) > 1 and lines[0].strip().isupper() and not lines[0].strip().endswith(":"):
        lines.pop(0)
    return " ".join(line.strip() for line in lines if line.strip())


def extract_claims(doc_id: str, doc_name: str, page: int, text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for sentence in _sentences(text):
        sentence = _remove_document_preamble(sentence)
        if len(sentence) < 20:
            continue
        if sentence.rstrip().endswith(":"):
            continue
        if _CLAIM_TRIGGERS.search(sentence):
            items.append({
                "item_id": str(uuid.uuid4()), "kind": "claim", "value": sentence,
                "context": sentence, "document_id": doc_id, "document": doc_name,
                "page": page, "review_status": "candidate", "event_date": None,
            })
    return items
