from __future__ import annotations
import json
from typing import Any
from fastapi import HTTPException
from .config import DATA
from .auth import current_workspace_id


def workspace_root():
    workspace_id = current_workspace_id()
    return DATA if workspace_id == "legacy" else DATA / "workspaces" / workspace_id


def workspace_uploads():
    return workspace_root() / "documents"


def workspace_processed():
    return workspace_root() / "processed"


def load() -> dict[str, Any]:
    root = workspace_root()
    db_path = root / "index.json"
    root.mkdir(parents=True, exist_ok=True)
    workspace_uploads().mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        return {"documents": [], "chunks": [], "items": [], "conflicts": [], "matters": []}
    try:
        data = json.loads(db_path.read_text(encoding="utf-8"))
        # ensure new keys exist in older index files
        data.setdefault("items", [])
        data.setdefault("conflicts", [])
        data.setdefault("matters", [])
        for document in data.get("documents", []):
            document.setdefault("matter_id", None)
        return data
    except (json.JSONDecodeError, OSError):
        raise HTTPException(500, "The account's local document index is unreadable. Back up its workspace before recovery.")


def save(data: dict[str, Any]) -> None:
    root = workspace_root()
    root.mkdir(parents=True, exist_ok=True)
    db_path = root / "index.json"
    tmp = db_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(db_path)
