from __future__ import annotations
import re
import uuid
from typing import Any

_MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "jun": "06", "jul": "07", "aug": "08", "sep": "09", "oct": "10",
    "nov": "11", "dec": "12",
}

# dd Month yyyy  or  Month dd, yyyy  or  dd/mm/yyyy  or  yyyy-mm-dd
_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[,\s]+(\d{4})\b"
    r"|\b(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})[,\s]+(\d{4})\b"
    r"|\b(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})\b"
    r"|\b(\d{4})[/\-\.](\d{2})[/\-\.](\d{2})\b",
    re.IGNORECASE,
)

_AMOUNT_RE = re.compile(
    r"(?:Rs\.?|INR|USD|\$|£|€)\s*[\d,]+(?:\.\d{1,2})?"
    r"|\b[\d,]+(?:\.\d{1,2})?\s*(?:rupees?|dollars?|lakhs?|crores?)\b",
    re.IGNORECASE,
)


def _iso(m: re.Match) -> str:
    g = m.groups()
    if g[0]:   # dd Month yyyy
        return f"{g[2]}-{_MONTHS[g[1].lower()]}-{int(g[0]):02d}"
    if g[3]:   # Month dd yyyy
        return f"{g[5]}-{_MONTHS[g[3].lower()]}-{int(g[4]):02d}"
    if g[6]:   # dd/mm/yyyy
        return f"{g[8]}-{int(g[7]):02d}-{int(g[6]):02d}"
    if g[9]:   # yyyy-mm-dd
        return f"{g[9]}-{g[10]}-{g[11]}"
    return ""


def infer_event_type(context: str) -> str:
    """Assign a transparent event cue from the words surrounding a date."""
    text = context.casefold()
    patterns = (
        ("execution", r"\b(sign(?:ed|ing)?|execut(?:ed|ion)|entered into|made this agreement)\b"),
        ("payment", r"\b(paid|payment|remit(?:ted)?|transfer(?:red)?|instal?ment|deposit)\b"),
        ("possession", r"\b(possession|handover|handed over|delivery|vacat(?:e|ed|ion)|occup(?:y|ied|ancy))\b"),
        ("notice", r"\b(notice|served|demand|intimation)\b"),
        ("registration", r"\b(registr(?:y|ation|ar)|registered|sub-registrar|mutation)\b"),
        ("lease_start", r"\b(commenc(?:e|es|ed|ement)|effective from|start(?:s|ed)?|begin(?:s|ning)?)\b"),
        ("lease_end", r"\b(expir(?:e|es|ed|y)|terminat(?:e|es|ed|ion)|ending|ends)\b"),
        ("hearing_or_order", r"\b(hearing|judgment|order|decree|court|filed|petition)\b"),
    )
    for label, pattern in patterns:
        if re.search(pattern, text):
            return label
    return "undetermined"


def extract_dates(doc_id: str, doc_name: str, page: int, text: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    matches = list(_DATE_RE.finditer(text))
    interval_roles: dict[int, tuple[str, str, str]] = {}
    # Only label an interval when the source explicitly expresses a bounded
    # range. Two nearby dates alone are not enough to infer one.
    for first, second in zip(matches, matches[1:]):
        between = text[first.end():second.start()]
        prefix = text[max(0, first.start() - 48):first.start()]
        relation = None
        if re.search(r"\bfrom\s*$", prefix, re.IGNORECASE) and re.search(r"\bto\b", between, re.IGNORECASE):
            relation = "from_to"
        elif re.search(r"\bbetween\s*$", prefix, re.IGNORECASE) and re.search(r"\band\b", between, re.IGNORECASE):
            relation = "between_and"
        if relation:
            interval_id = str(uuid.uuid4())
            interval_roles[first.start()] = (interval_id, "start", relation)
            interval_roles[second.start()] = (interval_id, "end", relation)
    for m in matches:
        iso = _iso(m)
        start = max(0, m.start() - 80)
        ctx = text[start: m.end() + 80].replace("\n", " ").strip()
        interval = interval_roles.get(m.start())
        items.append({
            "item_id": str(uuid.uuid4()), "kind": "date", "value": m.group(0),
            "event_date": iso, "event_type": infer_event_type(ctx), "context": ctx,
            "interval_id": interval[0] if interval else None,
            "interval_role": interval[1] if interval else None,
            "interval_relation": interval[2] if interval else None,
            "document_id": doc_id, "document": doc_name,
            "page": page, "review_status": "candidate",
        })
    for m in _AMOUNT_RE.finditer(text):
        start = max(0, m.start() - 80)
        ctx = text[start: m.end() + 80].replace("\n", " ").strip()
        items.append({
            "item_id": str(uuid.uuid4()), "kind": "amount", "value": m.group(0),
            "event_date": None, "context": ctx,
            "document_id": doc_id, "document": doc_name,
            "page": page, "review_status": "candidate",
        })
    return items
