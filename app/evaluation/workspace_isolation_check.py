"""Verify admin/customer document indexes are isolated using a temporary store.

Run with `.venv/Scripts/python.exe -m app.evaluation.workspace_isolation_check`.
The check creates a temporary SQLite account database and temporary indexes;
it never opens or changes the application's real accounts or document data.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from app import accounts, store
from app.auth import _workspace_id
from app.config import ROOT


def run() -> None:
    original_data = accounts.DATA
    original_db = accounts.DB_PATH
    original_store_data = store.DATA
    old_password = os.environ.get("LEGITMATE_PASSWORD")
    temp_dir = tempfile.TemporaryDirectory(prefix=".legitmate-isolation-", dir=ROOT)
    root = Path(temp_dir.name)
    try:
        accounts.DATA = root
        accounts.DB_PATH = root / "accounts.sqlite3"
        store.DATA = root
        os.environ["LEGITMATE_PASSWORD"] = "temporary-admin-password-123"
        accounts.initialize_accounts()
        customer_a = accounts.create_customer("isolation_a", "Isolation A", "temporary-customer-a-123")
        customer_b = accounts.create_customer("isolation_b", "Isolation B", "temporary-customer-b-123")

        token = _workspace_id.set("legacy")
        try:
            store.save({"documents": [{"doc_id": "admin-only"}], "chunks": [], "items": [], "conflicts": []})
            admin_docs = [row["doc_id"] for row in store.load()["documents"]]
        finally:
            _workspace_id.reset(token)

        for customer, marker in ((customer_a, "customer-a-only"), (customer_b, "customer-b-only")):
            token = _workspace_id.set(customer["workspace_id"])
            try:
                store.save({"documents": [{"doc_id": marker}], "chunks": [], "items": [], "conflicts": []})
            finally:
                _workspace_id.reset(token)

        observed: dict[str, list[str]] = {}
        roots: dict[str, str] = {}
        for name, workspace in (("admin", "legacy"), ("customer_a", customer_a["workspace_id"]), ("customer_b", customer_b["workspace_id"])):
            token = _workspace_id.set(workspace)
            try:
                observed[name] = [row["doc_id"] for row in store.load()["documents"]]
                roots[name] = str(store.workspace_root().resolve())
            finally:
                _workspace_id.reset(token)

        expected = {
            "admin": ["admin-only"],
            "customer_a": ["customer-a-only"],
            "customer_b": ["customer-b-only"],
        }
        assert admin_docs == expected["admin"]
        assert observed == expected, f"Unexpected workspace contents: {observed}"
        assert len(set(roots.values())) == 3, f"Workspace paths are not distinct: {roots}"
        assert accounts.authenticate("isolation_a", "temporary-customer-a-123") is not None
        print("PASS: admin and both customer indexes contain only their own document marker.")
        print("PASS: all three workspace storage paths are distinct.")
        print("PASS: temporary customer authentication succeeds.")
        print("Temporary data removed automatically; live account and document data were not accessed.")
    finally:
        accounts.DATA = original_data
        accounts.DB_PATH = original_db
        store.DATA = original_store_data
        if old_password is None:
            os.environ.pop("LEGITMATE_PASSWORD", None)
        else:
            os.environ["LEGITMATE_PASSWORD"] = old_password
        temp_dir.cleanup()


if __name__ == "__main__":
    run()
