"""Small offline language estimate using script and language-specific word cues."""
from __future__ import annotations

from collections import Counter
import re
import unicodedata

_SCRIPTS = (
    ("Kannada", 0x0C80, 0x0CFF, "kn"), ("Devanagari", 0x0900, 0x097F, "hi"),
    ("Tamil", 0x0B80, 0x0BFF, "ta"), ("Telugu", 0x0C00, 0x0C7F, "te"),
    ("Malayalam", 0x0D00, 0x0D7F, "ml"), ("Bengali", 0x0980, 0x09FF, "bn"),
    ("Gujarati", 0x0A80, 0x0AFF, "gu"), ("Gurmukhi", 0x0A00, 0x0A7F, "pa"),
    ("Arabic", 0x0600, 0x06FF, "ar"), ("Cyrillic", 0x0400, 0x04FF, "ru"),
    ("Han", 0x4E00, 0x9FFF, "zh"),
)

_WORD_CUES = {
    "en": ("English", "the and is are was were with from this that have has your you my what how hi hello hey thanks please document property agreement legal", "hi|hello|hey|thanks|hi there|hello there|good morning|good afternoon|good evening|thank you"),
    "es": ("Spanish", "el la los las de que y es por para con una un del documento contrato propiedad", "hola|buenos dias|buenas tardes|muchas gracias|gracias"),
    "fr": ("French", "le la les des une un et est dans pour avec contrat document propriété", "bonjour|salut|merci beaucoup|merci"),
    "de": ("German", "der die das und ist ein eine mit für von vertrag dokument eigentum", "guten morgen|hallo|danke"),
    "pt": ("Portuguese", "o a os as de que e é para com uma um contrato documento propriedade", "olá|bom dia|obrigado|obrigada"),
    "it": ("Italian", "il lo la gli le di che e è per con un una contratto documento proprietà", "ciao|buongiorno|grazie"),
    "hi-Latn": ("Hindi (Latin-script estimate)", "hai hain kya ka ki ke mein mera meri aap namaste dhanyavaad", "namaste|kaise ho|aap kaise hain"),
}


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def detect_language(text: str) -> dict[str, str | float]:
    sample = " ".join(str(text or "").split())[:20000]
    letters = [c for c in sample if c.isalpha()]
    words = [_fold(w.strip(".,?!;:()[]{}\"'“”‘’")) for w in sample.split()]
    words = [w for w in words if any(ch.isalpha() for ch in w)]
    if not letters:
        return {"code": "und", "label": "Undetermined", "script": "Unknown", "confidence": 0.0}

    counts: Counter[str] = Counter()
    for char in letters:
        codepoint = ord(char)
        for script, start, end, _ in _SCRIPTS:
            if start <= codepoint <= end:
                counts[script] += 1
                break
        else:
            if "LATIN" in unicodedata.name(char, ""):
                counts["Latin"] += 1
            else:
                counts["Other"] += 1

    script, count = counts.most_common(1)[0]
    share = count / len(letters)
    if script != "Latin":
        if share < 0.55:
            return {"code": "und", "label": "Mixed-script text", "script": "Mixed", "confidence": round(share, 2)}
        entry = next(item for item in _SCRIPTS if item[0] == script)
        if script == "Devanagari":
            code, label = "hi", "Hindi (Devanagari-script estimate)"
        elif script in {"Arabic", "Cyrillic", "Han"}:
            code, label = "und", f"{script} script (language uncertain)"
        else:
            code, label = entry[3], f"{script} script (language estimate)"
        return {"code": code, "label": label, "script": script, "confidence": round(min(0.99, max(0.72, share)), 2)}

    folded = " ".join(words)
    scored: list[tuple[float, str, str]] = []
    for code, (label, cues, phrases) in _WORD_CUES.items():
        cue_set = {_fold(word) for word in cues.split()}
        hits = sum(1 for word in words if word in cue_set)
        phrase_hits = sum(1 for phrase in phrases.split("|") if _fold(phrase) in folded)
        # Phrase cues make very short greetings such as "hi there" detectable.
        score = hits + phrase_hits * 2.5
        if code == "hi-Latn" and any(ch in sample.casefold() for ch in "āīūḍṭṇñ"):
            score += 2
        scored.append((score, code, label))
    score, code, label = max(scored)
    runner_up = sorted((entry[0] for entry in scored), reverse=True)[1]
    min_signal = 1.5 if len(words) <= 4 else max(2.0, len(words) * 0.09)
    if score >= min_signal and (score - runner_up >= 0.5 or score >= 3.5):
        confidence = min(0.96, 0.62 + 0.09 * score + (0.08 if len(words) <= 4 else 0))
        script_label = "Latin" if code != "hi-Latn" else "Latin (transliterated)"
        return {"code": code, "label": f"{label} (estimated)", "script": script_label, "confidence": round(confidence, 2)}
    return {"code": "und", "label": "Latin-script language (uncertain)", "script": "Latin", "confidence": round(min(0.6, share), 2)}
