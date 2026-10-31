from __future__ import annotations
from typing import Any
import shutil
import subprocess
import re
import pymupdf as fitz

from app.nlp.language import detect_language


_TESSERACT_BY_CODE = {
    "en": "eng", "hi": "hin", "kn": "kan", "ta": "tam", "te": "tel",
    "ml": "mal", "bn": "ben", "gu": "guj", "pa": "pan", "ar": "ara",
    "ru": "rus", "zh": "chi_sim", "es": "spa", "fr": "fra", "de": "deu",
    "pt": "por", "it": "ita",
}
_TESSERACT_BY_SCRIPT = {
    "Latin": "eng", "Devanagari": "hin", "Kannada": "kan", "Tamil": "tam",
    "Telugu": "tel", "Malayalam": "mal", "Bengali": "ben", "Gujarati": "guj",
    "Gurmukhi": "pan", "Arabic": "ara", "Cyrillic": "rus", "Han": "chi_sim",
}


def _installed_tesseract_languages() -> set[str]:
    executable = shutil.which("tesseract")
    if not executable:
        return set()
    try:
        result = subprocess.run(
            [executable, "--list-langs"], capture_output=True, text=True,
            timeout=8, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    return {line.strip() for line in result.stdout.splitlines()[1:] if re.fullmatch(r"[A-Za-z0-]+", line.strip())}


def _osd_script(content: bytes) -> tuple[str | None, str | None]:
    """Ask local Tesseract OSD to identify the script of a scanned page, if available."""
    executable = shutil.which("tesseract")
    if not executable:
        return None, None
    try:
        with fitz.open(stream=content, filetype="pdf") as pdf:
            for page in pdf:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                result = subprocess.run(
                    [executable, "stdin", "stdout", "--psm", "0"],
                    input=pixmap.tobytes("png"), capture_output=True,
                    timeout=15, check=False,
                )
                diagnostic = result.stderr.decode("utf-8", errors="replace")
                script = re.search(r"(?im)^Script:\s*(.+?)\s*$", diagnostic)
                confidence = re.search(r"(?im)^Script confidence:\s*([\d.]+)", diagnostic)
                if script:
                    return script.group(1).strip(), confidence.group(1) if confidence else None
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
        pass
    return None, None


def detect_ocr_language(content: bytes, readable_text: str = "") -> dict[str, Any]:
    """Choose the best installed local OCR pack from document language or scanned-page script.

    Tesseract's script detector only identifies scripts, not languages. For a scan-only
    file, Latin script therefore falls back to English; the report makes that uncertainty
    explicit rather than presenting it as a reliable language classification.
    """
    if not shutil.which("tesseract"):
        raise ValueError(
            "Local OCR is unavailable because Tesseract is not installed or not on PATH. "
            "Install Tesseract with local language data, restart the server, then retry. "
            "Language selection is automatic."
        )
    installed = _installed_tesseract_languages()
    detected = detect_language(readable_text)
    confidence = float(detected.get("confidence", 0) or 0)
    code = str(detected.get("code", "und"))
    requested = _TESSERACT_BY_CODE.get(code) if confidence >= 0.72 else None
    method = "document text language estimate"
    script = str(detected.get("script", "Unknown"))
    if not requested and confidence >= 0.72:
        requested = _TESSERACT_BY_SCRIPT.get(script)
        if requested:
            method = "document writing-system estimate"
    osd_confidence = None
    if not requested and not readable_text.strip():
        script, osd_confidence = _osd_script(content)
        requested = _TESSERACT_BY_SCRIPT.get(str(script))
        method = "local Tesseract script detection" if requested else "English fallback; scan script uncertain"
    if not requested:
        requested = "eng"
        method = "English fallback; document language uncertain"

    selected = requested
    fallback = False
    if selected not in installed:
        if "eng" in installed:
            selected = "eng"
        else:
            preferred = ["hin", "tam", "tel", "kan", "mal", "ben", "guj", "pan", "ara", "spa", "fra", "deu", "por", "ita", "chi_sim", "rus"]
            selected = next((pack for pack in preferred if pack in installed), "")
        fallback = selected != requested
    if not selected:
        raise ValueError(
            "No usable Tesseract language data is installed. Install a local language pack, "
            "then retry; language selection is automatic."
        )
    return {
        "language": selected,
        "detected_code": code,
        "detected_label": str(detected.get("label", "Undetermined")),
        "script": script,
        "confidence": confidence,
        "method": method,
        "fallback": fallback,
        "installed_pack": selected,
        "osd_script_confidence": osd_confidence,
    }


def ocr_quality_signals(text: str) -> dict[str, Any]:
    """Return inspectable OCR warning signals, not a recognition-confidence score."""
    compact = [char for char in str(text or "") if not char.isspace()]
    words = [word for word in str(text or "").split() if any(char.isalpha() for char in word)]
    letters = sum(char.isalpha() for char in compact)
    alpha_share = letters / len(compact) if compact else 0.0
    replacement_count = str(text or "").count("\ufffd")
    one_letter_share = (sum(1 for word in words if len(word.strip(".,:;!?()[]{}\"'“”‘’")) == 1)
                        / len(words)) if words else 0.0
    flags: list[str] = []
    if len(compact) < 60:
        flags.append("very_little_recognized_text")
    if compact and alpha_share < 0.45:
        flags.append("low_alphabetic_character_share")
    if replacement_count:
        flags.append("replacement_characters_present")
    if len(words) >= 12 and one_letter_share > 0.35:
        flags.append("many_single_character_tokens")
    return {
        "recognized_characters": len(compact),
        "recognized_words": len(words),
        "alphabetic_share": round(alpha_share, 3),
        "replacement_characters": replacement_count,
        "single_character_token_share": round(one_letter_share, 3),
        "review_recommended": bool(flags),
        "flags": flags,
        "method_note": "Heuristic text diagnostics only; PyMuPDF does not expose per-word OCR confidence in this workflow. Compare with the page image.",
    }


def extract_pages(content: bytes) -> list[dict[str, Any]]:
    """Return list of {page, text} dicts. Raises ValueError on failure."""
    try:
        pdf = fitz.open(stream=content, filetype="pdf")
        pages = [{"page": i + 1, "text": page.get_text("text")} for i, page in enumerate(pdf)]
        pdf.close()
        return pages
    except Exception as exc:
        raise ValueError(f"Could not read PDF ({type(exc).__name__}).") from exc


def extract_pages_with_ocr(
    content: bytes,
    *,
    language: str = "eng",
    max_ocr_pages: int = 50,
    dpi: int = 180,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    """OCR only pages with little selectable text, using local Tesseract via PyMuPDF.

    Returns pages, successfully OCR'd page numbers, and page numbers skipped by
    the per-document OCR cap. This never sends document content over the network.
    """
    pdf = None
    try:
        pdf = fitz.open(stream=content, filetype="pdf")
        pages: list[dict[str, Any]] = []
        ocr_done: list[int] = []
        skipped: list[int] = []
        if not any(len("".join(page.get_text("text").split())) < 40 for page in pdf):
            return [{"page": i + 1, "text": page.get_text("text")} for i, page in enumerate(pdf)], [], []
        if not shutil.which("tesseract"):
            raise ValueError(
                "Local OCR is unavailable because Tesseract is not installed or not on PATH. "
                "Install Tesseract and its language data, restart the server, then retry."
            )
        used = 0
        for index, page in enumerate(pdf):
            text = page.get_text("text")
            if len("".join(text.split())) < 40:
                if used >= max_ocr_pages:
                    skipped.append(index + 1)
                else:
                    try:
                        text_page = page.get_textpage_ocr(language=language, dpi=dpi, full=True)
                        ocr_text = page.get_text("text", textpage=text_page)
                    except Exception as exc:
                        raise ValueError(
                            f"OCR failed on page {index + 1}. Check that Tesseract language data '{language}' is installed."
                        ) from exc
                    used += 1
                    if ocr_text.strip():
                        text = ocr_text
                        ocr_done.append(index + 1)
            pages.append({"page": index + 1, "text": text})
        return pages, ocr_done, skipped
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read PDF for OCR ({type(exc).__name__}).") from exc
    finally:
        if pdf is not None:
            pdf.close()
