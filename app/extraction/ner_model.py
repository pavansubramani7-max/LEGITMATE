"""Lazy spaCy NER loader with graceful fallback.

If spaCy or en_core_web_sm is unavailable (e.g. DLL policy blocks compiled
extensions), _nlp() returns None and entities.py falls back to regex.
"""
from __future__ import annotations

from functools import lru_cache

_model = None
_attempted = False

_SPACY_MODELS = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "pt": "pt_core_news_sm",
    "it": "it_core_news_sm",
}


@lru_cache(maxsize=6)
def get_nlp_for_language(language_code: str):
    """Load an installed spaCy pipeline for a supported language, if present.

    This is deliberately optional: missing packages, missing model wheels, and
    incompatible local runtimes return ``None`` and leave the rule fallback in
    place. No model is downloaded at request time.
    """
    model_name = _SPACY_MODELS.get(str(language_code or "").split("-")[0])
    if not model_name:
        return None
    try:
        import spacy  # noqa: PLC0415
        return spacy.load(model_name, disable=["parser"])
    except Exception:
        return None


def spacy_model_name(language_code: str) -> str | None:
    """Return the optional model name associated with a language code."""
    return _SPACY_MODELS.get(str(language_code or "").split("-")[0])


def get_nlp():
    """Return a loaded spaCy Language object, or None if unavailable."""
    global _model, _attempted
    if _attempted:
        return _model
    _attempted = True
    _model = get_nlp_for_language("en")
    return _model
