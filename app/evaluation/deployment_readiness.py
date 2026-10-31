"""Non-secret local deployment checklist for the LEGITMATE prototype."""
from __future__ import annotations

import os


def readiness_report() -> dict[str, object]:
    """Report configuration posture without returning credentials or file paths."""
    secret = os.environ.get("LEGITMATE_SECRET", "")
    password = os.environ.get("LEGITMATE_PASSWORD", "")
    host = os.environ.get("LEGITMATE_BIND_HOST", "127.0.0.1").strip().casefold()
    provider = os.environ.get("AI_PROVIDER", "off").strip().casefold()
    cookie_secure = os.environ.get("LEGITMATE_COOKIE_SECURE", "false").strip().casefold() in {"1", "true", "yes", "on"}
    local_host = host in {"127.0.0.1", "localhost", "::1"}
    checks = [
        {
            "id": "loopback_binding", "title": "Local network binding",
            "status": "pass" if local_host else "blocked",
            "detail": "Server is configured for loopback-only access." if local_host else "Server host is not loopback. Do not expose this prototype on a shared or public network.",
        },
        {
            "id": "session_secret", "title": "Session signing secret",
            "status": "pass" if len(secret) >= 32 else "blocked",
            "detail": "A sufficiently long signing secret is configured." if len(secret) >= 32 else "Set LEGITMATE_SECRET to a unique random value of at least 32 characters.",
        },
        {
            "id": "secure_cookie", "title": "HTTPS session cookie",
            "status": "pass" if cookie_secure else "review",
            "detail": "Session cookies are marked Secure for HTTPS." if cookie_secure else "Secure is off for localhost HTTP. Set LEGITMATE_COOKIE_SECURE=true only when the app is served through HTTPS; the browser will not send the cookie over HTTP.",
        },
        {
            "id": "admin_password", "title": "Initial administrator password",
            "status": "pass" if len(password) >= 12 else "review",
            "detail": "The configured seed password meets the minimum length." if len(password) >= 12 else "Set a unique administrator seed password of at least 12 characters; existing accounts use their stored password hash.",
        },
        {
            "id": "external_ai", "title": "External AI provider",
            "status": "pass" if provider in {"", "off", "none"} else "review",
            "detail": "External AI is disabled by configuration." if provider in {"", "off", "none"} else "An external AI provider is enabled; selected prompts and excerpts may leave this computer.",
        },
        {
            "id": "reviewed_validation", "title": "Independent legal-domain validation",
            "status": "blocked",
            "detail": "No independently reviewed real-document test set is bundled; synthetic regression scores cannot validate legal-domain accuracy.",
        },
        {
            "id": "at_rest_encryption", "title": "Encryption at rest",
            "status": "blocked",
            "detail": "The application does not encrypt local documents, indexes, or account storage. Use OS-level disk protection; production deployment needs an approved encryption and key-management design.",
        },
        {
            "id": "production_operations", "title": "Production operations",
            "status": "blocked",
            "detail": "The JSON document index, single-process login throttle, local HTTP development server, backup/restore plan, and monitoring are not a production deployment stack.",
        },
    ]
    local_ready = local_host and len(secret) >= 32 and len(password) >= 12
    return {
        "profile": "local research prototype",
        "local_configuration_ready": local_ready,
        "production_ready": False,
        "checks": checks,
        "notice": "This checklist does not inspect or return secrets and is not a security certification. Production readiness requires an environment-specific security review and approved operations plan.",
    }
