"""Local account store for LEGITMATE's single-admin, isolated-customer setup."""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from fastapi import HTTPException

from .config import DATA

DB_PATH = DATA / "accounts.sqlite3"
_HASH_ROUNDS = 600_000
_MIN_PASSWORD = 12


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    DATA.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=10000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _HASH_ROUNDS)
    return salt.hex(), digest.hex()


def initialize_accounts() -> None:
    """Create schema and migrate the legacy env password to a private admin account."""
    with _db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                display_name TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin','customer')),
                workspace_id TEXT NOT NULL UNIQUE,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1)),
                auth_version INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "auth_version" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN auth_version INTEGER NOT NULL DEFAULT 0")
        if conn.execute("SELECT 1 FROM users LIMIT 1").fetchone():
            return
        password = os.environ.get("LEGITMATE_PASSWORD", "")
        if not password:
            raise RuntimeError("Set LEGITMATE_PASSWORD in .env to initialize the first admin account.")
        username = os.environ.get("LEGITMATE_ADMIN_USERNAME", "admin").strip() or "admin"
        display = os.environ.get("LEGITMATE_ADMIN_NAME", "Workspace administrator").strip() or "Workspace administrator"
        salt, digest = _hash_password(password)
        now = _now()
        conn.execute(
            "INSERT INTO users(user_id,username,display_name,role,workspace_id,password_salt,password_hash,active,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("admin", username, display, "admin", "legacy", salt, digest, 1, now, now),
        )


def _validate_username(username: str) -> str:
    username = " ".join(str(username or "").strip().split())
    if not 3 <= len(username) <= 64 or any(ord(c) < 32 for c in username):
        raise HTTPException(422, "Username must be between 3 and 64 characters.")
    return username


def _validate_password(password: str) -> str:
    if len(password) < _MIN_PASSWORD or len(password) > 256:
        raise HTTPException(422, f"Password must be between {_MIN_PASSWORD} and 256 characters.")
    return password


def authenticate(username: str, password: str) -> dict[str, Any] | None:
    initialize_accounts()
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=? AND active=1", (str(username or "").strip(),)
        ).fetchone()
    if row is None:
        # Keep roughly similar password-hash work when the username is unknown.
        _hash_password(password or "", bytes(16))
        return None
    salt = bytes.fromhex(row["password_salt"])
    candidate = hashlib.pbkdf2_hmac("sha256", (password or "").encode("utf-8"), salt, _HASH_ROUNDS).hex()
    if not hmac.compare_digest(candidate, row["password_hash"]):
        return None
    return public_user(row)


def get_user(user_id: str) -> dict[str, Any] | None:
    initialize_accounts()
    with _db() as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id=? AND active=1", (user_id,)).fetchone()
    return {**public_user(row), "auth_version": int(row["auth_version"])} if row else None


def public_user(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("user_id", "username", "display_name", "role", "workspace_id", "active", "created_at")}


def list_users() -> list[dict[str, Any]]:
    initialize_accounts()
    with _db() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY role, username COLLATE NOCASE").fetchall()
    return [{**public_user(row), "active": bool(row["active"])} for row in rows]


def create_customer(username: str, display_name: str, password: str) -> dict[str, Any]:
    username = _validate_username(username)
    password = _validate_password(password)
    display_name = " ".join(str(display_name or "").strip().split())
    if not display_name or len(display_name) > 100:
        raise HTTPException(422, "Customer name is required and must be at most 100 characters.")
    initialize_accounts()
    user_id = uuid.uuid4().hex
    workspace_id = user_id
    salt, digest = _hash_password(password)
    now = _now()
    try:
        with _db() as conn:
            conn.execute(
                "INSERT INTO users(user_id,username,display_name,role,workspace_id,password_salt,password_hash,active,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (user_id, username, display_name, "customer", workspace_id, salt, digest, 1, now, now),
            )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(409, "That username is already in use.") from exc
    return get_user(user_id) or {}


def set_active(user_id: str, active: bool, actor_id: str) -> None:
    initialize_accounts()
    with _db() as conn:
        row = conn.execute("SELECT role FROM users WHERE user_id=?", (user_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Account not found.")
        if row["role"] == "admin":
            raise HTTPException(403, "The primary admin account cannot be disabled here.")
        if user_id == actor_id:
            raise HTTPException(400, "You cannot disable your own account.")
        conn.execute("UPDATE users SET active=?, auth_version=auth_version+1, updated_at=? WHERE user_id=?", (int(active), _now(), user_id))


def reset_password(user_id: str, password: str) -> None:
    password = _validate_password(password)
    initialize_accounts()
    salt, digest = _hash_password(password)
    with _db() as conn:
        row = conn.execute("SELECT role FROM users WHERE user_id=?", (user_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Account not found.")
        if row["role"] == "admin":
            raise HTTPException(403, "Change the primary admin password through the server's protected settings.")
        conn.execute("UPDATE users SET password_salt=?, password_hash=?, auth_version=auth_version+1, updated_at=? WHERE user_id=?", (salt, digest, _now(), user_id))


def change_password(user_id: str, current_password: str, new_password: str) -> None:
    _validate_password(new_password)
    with _db() as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id=? AND active=1", (user_id,)).fetchone()
        if not row:
            raise HTTPException(401, "Account is no longer active.")
        actual = hashlib.pbkdf2_hmac("sha256", (current_password or "").encode(), bytes.fromhex(row["password_salt"]), _HASH_ROUNDS).hex()
        if not hmac.compare_digest(actual, row["password_hash"]):
            raise HTTPException(401, "Current password is incorrect.")
        salt, digest = _hash_password(new_password)
        conn.execute("UPDATE users SET password_salt=?, password_hash=?, auth_version=auth_version+1, updated_at=? WHERE user_id=?", (salt, digest, _now(), user_id))
