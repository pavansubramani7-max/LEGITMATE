"""Authentication and session management for LEGITMATE.

Local admin/customer accounts with per-account storage isolation. This is still
a localhost research prototype, not a public identity or deployment platform.

Environment variables (set in .env or shell before starting):
  LEGITMATE_PASSWORD   — required once; seeds the first admin account
  LEGITMATE_ADMIN_USERNAME — optional first admin username (default: admin)
  LEGITMATE_SECRET     — required; secret key for signing session cookies (no default)

Sessions:
Signed with itsdangerous.TimestampSigner and linked to an active local account.
Cookie name: lm_session.
  Max age: SESSION_MAX_AGE seconds (default 8 hours).
  HttpOnly, SameSite=Strict. Secure is off for localhost HTTP. Set
  LEGITMATE_COOKIE_SECURE=true only behind HTTPS; plain HTTP will not send it.

CSRF:
  State-changing API routes (POST/PUT/PATCH/DELETE) require the header
  X-CSRF-Token matching the value stored in the session.
  The CSRF token is a random hex string generated at login time.
  GET /api/csrf returns the token for the current session (used by the frontend).

Login throttle:
  After MAX_FAILURES failed attempts within WINDOW_SECONDS, further attempts
  are rejected with 429 for LOCKOUT_SECONDS.  Counts are in-process memory
  only; they reset on server restart.  This is sufficient for a local prototype.

Password comparison:
  hmac.compare_digest is used to prevent timing attacks.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time
from contextvars import ContextVar
from typing import Any

from fastapi import Cookie, HTTPException, Request, Response
from itsdangerous import BadSignature, SignatureExpired, TimestampSigner

from app.accounts import authenticate, get_user

# ── configuration ─────────────────────────────────────────────────────────────

SESSION_MAX_AGE = int(os.environ.get("LEGITMATE_SESSION_MAX_AGE", 8 * 3600))
COOKIE_SECURE = os.environ.get("LEGITMATE_COOKIE_SECURE", "false").strip().casefold() in {"1", "true", "yes", "on"}
_COOKIE_NAME = "lm_session"
_workspace_id: ContextVar[str] = ContextVar("legitmate_workspace_id", default="legacy")

# Throttle: 5 failures in 60 s → locked out for 300 s
MAX_FAILURES = 5
WINDOW_SECONDS = 60
LOCKOUT_SECONDS = 300

# ── in-process throttle state ─────────────────────────────────────────────────
# {ip: [timestamp, ...]}  — only recent failures kept
_failures: dict[str, list[float]] = {}


def _get_secret() -> str:
    s = os.environ.get("LEGITMATE_SECRET", "")
    if not s:
        raise RuntimeError(
            "LEGITMATE_SECRET is not set. "
            "Set it in .env before starting the server."
        )
    return s


def _get_password_hash() -> str:
    pw = os.environ.get("LEGITMATE_PASSWORD", "")
    if not pw:
        raise RuntimeError(
            "LEGITMATE_PASSWORD is not set. "
            "Set it in .env before starting the server."
        )
    return hashlib.sha256(pw.encode()).hexdigest()


def _signer() -> TimestampSigner:
    return TimestampSigner(_get_secret(), salt="lm-session")


# ── session helpers ───────────────────────────────────────────────────────────

def create_session(response: Response, user_id: str = "admin") -> str:
    """Create a signed user session, set the cookie, return the CSRF token."""
    user = get_user(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Account is no longer active.")
    csrf = secrets.token_hex(32)
    payload = f"{csrf}:{user_id}:{user['auth_version']}"
    signed = _signer().sign(payload).decode()
    response.set_cookie(
        _COOKIE_NAME,
        signed,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="strict",
        secure=COOKIE_SECURE,  # false for localhost HTTP; enable behind HTTPS
    )
    return csrf


def _read_session(cookie: str | None) -> tuple[str, str, int] | None:
    """Return CSRF token and user id from a valid, unexpired signed cookie."""
    if not cookie:
        return None
    try:
        raw = _signer().unsign(cookie, max_age=SESSION_MAX_AGE).decode()
        csrf, user_id, auth_version = raw.split(":", 2)
        if not csrf or not user_id:
            return None
        return csrf, user_id, int(auth_version)
    except (BadSignature, SignatureExpired, ValueError):
        return None


def get_csrf_token(request: Request) -> str | None:
    principal = get_session_user(request)
    return principal["csrf"] if principal else None


def get_session_user(request: Request) -> dict[str, Any] | None:
    """Resolve the signed session to an active account and set its workspace context."""
    parsed = _read_session(request.cookies.get(_COOKIE_NAME))
    if parsed is None:
        return None
    csrf, user_id, auth_version = parsed
    user = get_user(user_id)
    if user is None or int(user["auth_version"]) != auth_version:
        return None
    _workspace_id.set(str(user["workspace_id"]))
    request.state.user = user
    return {**user, "csrf": csrf}


def current_workspace_id() -> str:
    return _workspace_id.get()


def bind_request_workspace(request: Request):
    """Bind the authenticated workspace before FastAPI runs sync endpoints in a worker."""
    token = _workspace_id.set("legacy")
    principal = get_session_user(request)
    _workspace_id.set(str(principal["workspace_id"]) if principal else "legacy")
    return token


def reset_request_workspace(token) -> None:
    _workspace_id.reset(token)


def clear_session(response: Response) -> None:
    response.delete_cookie(_COOKIE_NAME, httponly=True, samesite="strict", secure=COOKIE_SECURE)


# ── throttle ──────────────────────────────────────────────────────────────────

def _client_ip(request: Request) -> str:
    # X-Forwarded-For is not trusted here (localhost only)
    return request.client.host if request.client else "unknown"


def _record_failure(ip: str) -> None:
    now = time.monotonic()
    _failures.setdefault(ip, [])
    _failures[ip] = [t for t in _failures[ip] if now - t < WINDOW_SECONDS]
    _failures[ip].append(now)


def _is_locked(ip: str) -> bool:
    now = time.monotonic()
    recent = [t for t in _failures.get(ip, []) if now - t < WINDOW_SECONDS]
    if len(recent) >= MAX_FAILURES:
        # Check if the oldest failure is within lockout window
        oldest = min(recent)
        if now - oldest < LOCKOUT_SECONDS:
            return True
        # Lockout expired — clear
        _failures[ip] = []
    return False


def _clear_failures(ip: str) -> None:
    _failures.pop(ip, None)


# ── public auth helpers ───────────────────────────────────────────────────────

def check_password(candidate: str) -> bool:
    """Legacy environment-password helper retained for local setup diagnostics."""
    candidate_hash = hashlib.sha256(candidate.encode()).hexdigest()
    return hmac.compare_digest(candidate_hash, _get_password_hash())


def require_session(request: Request) -> str:
    """Dependency: validate active user and return its CSRF token."""
    principal = get_session_user(request)
    if principal is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return str(principal["csrf"])


def require_admin(request: Request) -> str:
    """Dependency that restricts account administration to the admin role."""
    require_session(request)
    if request.state.user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access is required.")
    return str(request.state.user["user_id"])


def require_csrf(request: Request) -> None:
    """Dependency: validate X-CSRF-Token header for state-changing requests."""
    principal = get_session_user(request)
    if principal is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    csrf_from_session = str(principal["csrf"])
    csrf_from_header = request.headers.get("X-CSRF-Token", "")
    if not hmac.compare_digest(csrf_from_header, csrf_from_session):
        raise HTTPException(status_code=403, detail="CSRF token invalid.")


# ── security headers middleware ───────────────────────────────────────────────

SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "X-Permitted-Cross-Domain-Policies": "none",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=()",
    "Cache-Control": "no-store",
    # Tight CSP: no external resources, no inline scripts (frontend uses defer+nonce pattern)
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "  # inline JS in single-file HTML
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "object-src 'none'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none';"
    ),
}


def apply_security_headers(response: Response) -> None:
    for k, v in SECURITY_HEADERS.items():
        response.headers[k] = v


# ── login / logout request handling ──────────────────────────────────────────

def handle_login(request: Request, response: Response, password: str, username: str = "admin") -> dict[str, Any]:
    """Validate account credentials, apply throttle, and create an account session."""
    ip = _client_ip(request)
    if _is_locked(ip):
        raise HTTPException(status_code=429, detail="Too many failed attempts. Try again later.")
    user = authenticate(username, password)
    if user is None:
        _record_failure(ip)
        raise HTTPException(status_code=401, detail="Username or password is incorrect.")
    _clear_failures(ip)
    csrf = create_session(response, str(user["user_id"]))
    apply_security_headers(response)
    return {"ok": True, "csrf": csrf, "user": {key: user[key] for key in ("username", "display_name", "role")}}
