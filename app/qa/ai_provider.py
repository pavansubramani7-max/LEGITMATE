"""Choose the configured AI provider while preserving local fallbacks."""
from __future__ import annotations

import os
from typing import Any

from app.qa import gemini, groq


def provider_status() -> dict[str, str | bool]:
    requested = os.environ.get("AI_PROVIDER", "off").strip().lower()
    if requested not in {"off", "auto", "gemini", "groq"}:
        requested = "off"
    if requested == "off":
        return {"provider": "off", "configured": False, "model": "disabled"}
    selected = requested
    if selected == "auto":
        selected = "gemini" if gemini.gemini_status()["configured"] else "groq"
    status = gemini.gemini_status() if selected == "gemini" else groq.groq_status()
    return {"provider": selected, "configured": status["configured"], "model": status["model"]}


def _module():
    status = provider_status()
    if status["provider"] == "off":
        raise RuntimeError("External AI is disabled (AI_PROVIDER=off). Local document search remains available.")
    if not status["configured"]:
        name = "Gemini" if status["provider"] == "gemini" else "Groq"
        env = "GEMINI_API_KEY" if status["provider"] == "gemini" else "GROQ_API_KEY"
        raise RuntimeError(f"{name} is not configured. Add {env} to your private .env file and restart LEGITMATE.")
    return gemini if status["provider"] == "gemini" else groq


def generate_chat_reply(question: str, history: list[dict[str, str]], language: dict[str, Any]) -> str:
    return _module().generate_chat_reply(question, history, language)


def generate_general_reply(question: str, language: dict[str, Any]) -> str:
    return _module().generate_general_reply(question, language)


def generate_grounded_answer(question: str, hits: list[dict[str, Any]], language: dict[str, Any]) -> str:
    return _module().generate_grounded_answer(question, hits, language)


def is_general_chat(question: str) -> bool:
    return groq.is_general_chat(question)
