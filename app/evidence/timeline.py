from __future__ import annotations
from typing import Any
from ..extraction.dates import infer_event_type


def build_timeline(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return date/event items sorted by event_date (ISO string).
    Only includes items with a non-empty event_date and not rejected.
    """
    dated = [
        i for i in items
        if i.get("event_date") and i.get("review_status") != "rejected"
        and i["kind"] in ("date", "event")
    ]
    dated.sort(key=lambda x: x["event_date"])
    return [
        {
            "event_date": i["event_date"],
            "event_type": i.get("event_type") or infer_event_type(str(i.get("context", ""))),
            "interval_id": i.get("interval_id"),
            "interval_role": i.get("interval_role"),
            "interval_relation": i.get("interval_relation"),
            "value": i["value"],
            "context": i["context"],
            "document": i["document"],
            "page": i["page"],
            "citation_unit": i.get("citation_unit", "page"),
            "review_status": i["review_status"],
            "item_id": i["item_id"],
        }
        for i in dated
    ]
