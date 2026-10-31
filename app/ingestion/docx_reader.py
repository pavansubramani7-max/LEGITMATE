"""Extract readable paragraphs and table rows from modern Word (.docx) files.

Word pagination changes across devices and printers, so each paragraph/table
row is assigned a stable text-section number instead of a guessed page number.
"""
from __future__ import annotations

from io import BytesIO
from zipfile import BadZipFile, ZipFile
import re
import xml.etree.ElementTree as ET

_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_MAX_UNCOMPRESSED_BYTES = 400 * 1024 * 1024


def _paragraph_parts(paragraph: ET.Element) -> list[str]:
    parts = [""]
    for node in paragraph.iter():
        if node.tag == f"{_NS}t":
            parts[-1] += node.text or ""
        elif node.tag == f"{_NS}tab":
            parts[-1] += "\t"
        elif node.tag == f"{_NS}br" and node.attrib.get(f"{_NS}type") == "page":
            parts.append("")
        elif node.tag == f"{_NS}lastRenderedPageBreak":
            parts.append("")
    return [re.sub(r"\s+", " ", part).strip() for part in parts if part.strip()]


def extract_docx_sections(content: bytes) -> list[dict[str, object]]:
    """Read DOCX paragraph/table text into stable numbered citation sections."""
    try:
        with ZipFile(BytesIO(content)) as archive:
            infos = archive.infolist()
            if sum(info.file_size for info in infos) > _MAX_UNCOMPRESSED_BYTES:
                raise ValueError("The Word file expands beyond the supported size limit.")
            if "word/document.xml" not in archive.namelist():
                raise ValueError("This is not a supported Word .docx file.")
            xml = archive.read("word/document.xml")
    except (BadZipFile, OSError, KeyError) as exc:
        raise ValueError("The Word file is invalid or unreadable.") from exc

    if b"<!DOCTYPE" in xml.upper() or b"<!ENTITY" in xml.upper():
        raise ValueError("The Word file contains unsupported XML declarations.")

    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ValueError("The Word document text could not be read.") from exc

    body = root.find(f"{_NS}body")
    if body is None:
        raise ValueError("The Word document does not contain readable body text.")

    blocks: list[str] = []
    for child in body:
        if child.tag == f"{_NS}p":
            blocks.extend(_paragraph_parts(child))
        elif child.tag == f"{_NS}tbl":
            for row in child.findall(f".//{_NS}tr"):
                cells = []
                for cell in row.findall(f"{_NS}tc"):
                    text = " ".join(part for paragraph in cell.findall(f".//{_NS}p")
                                    for part in _paragraph_parts(paragraph))
                    if text.strip():
                        cells.append(re.sub(r"\s+", " ", text).strip())
                if cells:
                    blocks.append(" | ".join(cells))

    if not blocks:
        raise ValueError("No selectable text was found in this Word document.")
    return [{"page": index, "text": text} for index, text in enumerate(blocks, start=1)]
