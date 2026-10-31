from __future__ import annotations
import re
import uuid
from itertools import combinations
from typing import Any


def _normalise(value: str) -> str:
    # Strip trailing punctuation before comparing so "Rs. 550000," == "Rs. 550000"
    return re.sub(r"[,;.]+$", "", re.sub(r"\s+", " ", value.casefold().strip()))


def _context_key(item: dict[str, Any]) -> str:
    """Map context to a canonical topic bucket for grouping related facts.

    Only the text *before* the item value is examined so that a date's own
    surrounding sentence does not bleed into the next sentence's keywords.
    """
    ctx_full = item.get("context", "").casefold()
    value = item.get("value", "").casefold()
    # Use only the text before the value in the context window
    idx = ctx_full.find(value)
    ctx = ctx_full[:idx] if idx > 0 else ctx_full
    instrument = f"{ctx} {str(item.get('document', '')).casefold()}"
    kind = item.get("kind")
    if kind == "date":
        if "lease" in instrument:
            if any(kw in instrument for kw in ("commenc", "ending", "term", "expires")):
                return "lease_term"
            if any(kw in instrument for kw in ("execut", "signed", "entered into")):
                return "lease_execution"
        if any(kw in ctx for kw in ("execut", "signed", "sign ", "signing")):
            return "execution"
        # Generic agreement dates often refer to different instruments; keep
        # them apart until the document identity is clear enough to compare.
        return "__any__"
    if kind == "amount":
        if any(kw in ctx for kw in ("security deposit", "deposit")):
            return "security_deposit"
        if any(kw in ctx for kw in ("sale price", "consideration", "sale amount", "agreed")):
            return "consideration"
        if any(kw in ctx for kw in ("monthly rent", "rent")):
            return "rent"
        if any(kw in ctx for kw in ("balance", "outstanding")):
            return "balance"
        if any(kw in ctx for kw in ("payment", "paid", "neft", "transfer")):
            return "payment"
    return "__any__"


def detect_conflicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Detect conflicts between items from different documents.
    Only flags pairs where the same kind of fact (same context keyword) differs
    across documents. Returns minimal conflict records — one per conflicting pair
    or group, not one per year.
    """
    conflicts: list[dict[str, Any]] = []
    seen_pairs: set[frozenset[str]] = set()

    active = [i for i in items if i.get("review_status") != "rejected"]

    # ── Date conflicts ────────────────────────────────────────────────────────
    dates = [i for i in active if i["kind"] == "date" and i.get("event_date")]

    # Deduplicate: keep one representative item per (doc_id, event_date, ctx_key)
    # so multiple extractions of the same date from the same document don't
    # generate combinatorial pairs.
    _seen_date_key: set[tuple[str, str, str]] = set()
    deduped_dates: list[dict[str, Any]] = []
    for d in dates:
        dk = (d["document_id"], d["event_date"], _context_key(d))
        if dk not in _seen_date_key:
            _seen_date_key.add(dk)
            deduped_dates.append(d)

    # Group by context keyword, then check for different dates across documents
    by_ctx: dict[str, list[dict[str, Any]]] = {}
    for d in deduped_dates:
        by_ctx.setdefault(_context_key(d), []).append(d)

    # Only compare dates in contexts that identify a specific document event.
    # "payment" and "__any__" dates are event-specific (e.g. tax-paid date,
    # NEFT transfer date) and can legitimately differ across documents.
    _DATE_CONFLICT_CONTEXTS = frozenset({"execution", "lease_execution"})

    for ctx_key, group in by_ctx.items():
        if ctx_key not in _DATE_CONFLICT_CONTEXTS:
            continue
        # Only consider items from multiple documents
        docs = {i["document_id"] for i in group}
        if len(docs) < 2:
            continue
        # Find pairs from different documents with different ISO dates
        for a, b in combinations(group, 2):
            if a["document_id"] == b["document_id"]:
                continue
            if a["event_date"] == b["event_date"]:
                continue
            pair_key = frozenset([a["item_id"], b["item_id"]])
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            conflicts.append({
                "conflict_id": str(uuid.uuid4()),
                "kind": "date",
                "label": f"Different dates: {a['value']} vs {b['value']}",
                "items": [a["item_id"], b["item_id"]],
                "status": "open",
            })

    # ── Amount conflicts ──────────────────────────────────────────────────────
    amounts = [i for i in active if i["kind"] == "amount"]

    # Deduplicate: keep one representative per (doc_id, normalised_value, ctx_key)
    _seen_amt_key: set[tuple[str, str, str]] = set()
    deduped_amounts: list[dict[str, Any]] = []
    for a in amounts:
        ak = (a["document_id"], _normalise(a["value"]), _context_key(a))
        if ak not in _seen_amt_key:
            _seen_amt_key.add(ak)
            deduped_amounts.append(a)

    by_ctx_amt: dict[str, list[dict[str, Any]]] = {}
    for a in deduped_amounts:
        by_ctx_amt.setdefault(_context_key(a), []).append(a)

    for ctx_key, group in by_ctx_amt.items():
        if ctx_key == "__any__":
            continue
        docs = {i["document_id"] for i in group}
        if len(docs) < 2:
            continue
        for a, b in combinations(group, 2):
            if a["document_id"] == b["document_id"]:
                continue
            if _normalise(a["value"]) == _normalise(b["value"]):
                continue
            pair_key = frozenset([a["item_id"], b["item_id"]])
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            conflicts.append({
                "conflict_id": str(uuid.uuid4()),
                "kind": "amount",
                "label": f"Different amounts: {a['value']} vs {b['value']}",
                "items": [a["item_id"], b["item_id"]],
                "status": "open",
            })

    # Unstructured claim-pair conflict detection is intentionally omitted.
    # Sharing a verb (e.g. "warrants" or "pay") does not prove contradiction;
    # claim comparisons require reviewed semantic labels or an evaluated NLI model.

    return conflicts
