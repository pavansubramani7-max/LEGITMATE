from __future__ import annotations
from typing import Any
import shutil
import fitz


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
