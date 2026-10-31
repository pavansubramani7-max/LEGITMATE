"""Account-local, hash-chained audit events for sensitive review actions.

This is tamper-evident application logging, not immutable storage: an operator
with filesystem and signing-key access can rewrite it. Events intentionally
contain identifiers and change categories, never source text or reviewer notes.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import threading
from datetime import datetime, timezone
from typing import Any

from app.auth import current_workspace_id
from app.store import workspace_processed

_LOCK = threading.Lock()
_ZERO_HASH = "0" * 64


def _path():
    return workspace_processed() / "audit" / "events.jsonl"


def verify_audit_log() -> dict[str, Any]:
    path = _path()
    if not path.exists():
        return {"valid": True, "entries": 0, "status": "empty"}
    previous = _ZERO_HASH
    entries = 0
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                claimed = row.pop("chain_hash", None)
                if row.get("previous_hash") != previous:
                    return {"valid": False, "entries": entries, "status": "broken_chain", "line": line_number}
                canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                actual = hashlib.sha256(bytes.fromhex(previous) + canonical).hexdigest()
                if not isinstance(claimed, str) or not hmac.compare_digest(claimed, actual):
                    return {"valid": False, "entries": entries, "status": "hash_mismatch", "line": line_number}
                previous = claimed
                entries += 1
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {"valid": False, "entries": entries, "status": "unreadable"}
    return {"valid": True, "entries": entries, "status": "verified", "head": previous}


def record_audit_event(event_type: str, subject_type: str, subject_id: str, *, changes: list[str] | None = None) -> dict[str, Any]:
    """Append a minimal account-scoped event, verifying the existing chain first."""
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        integrity = verify_audit_log()
        if not integrity["valid"]:
            raise RuntimeError("The local audit ledger failed integrity verification; new review changes are paused.")
        row = {
            "schema_version": 1,
            "event_id": hashlib.sha256(f"{datetime.now(timezone.utc).isoformat()}:{subject_type}:{subject_id}".encode()).hexdigest()[:24],
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "workspace_id": current_workspace_id(),
            "event_type": event_type,
            "subject_type": subject_type,
            "subject_id": subject_id,
            "changes": sorted(set(changes or [])),
            "previous_hash": integrity.get("head", _ZERO_HASH),
        }
        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        row["chain_hash"] = hashlib.sha256(bytes.fromhex(row["previous_hash"]) + canonical).hexdigest()
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            import os
            os.fsync(stream.fileno())
        return {"event_id": row["event_id"], "chain_hash": row["chain_hash"]}
