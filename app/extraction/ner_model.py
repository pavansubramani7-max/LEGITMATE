"""Lazy spaCy NER loader with graceful fallback.

If spaCy or en_core_web_sm is unavailable (e.g. DLL policy blocks compiled
extensions), _nlp() returns None and entities.py falls back to regex.
"""
from __future__ import annotations

_model = None
_attempted = False


def get_nlp():
    """Return a loaded spaCy Language object, or None if unavailable."""
    global _model, _attempted
    if _attempted:
        return _model
    _attempted = True
    try:
        import spacy  # noqa: PLC0415
        _model = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except Exception:
        _model = None
    return _model
