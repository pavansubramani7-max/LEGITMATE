"""Safe text extraction for common office, web, and plain-text legal files."""
from __future__ import annotations

from html.parser import HTMLParser
from io import BytesIO
from zipfile import BadZipFile, ZipFile
import csv
import json
import re
import xml.etree.ElementTree as ET

_MAX_UNCOMPRESSED_BYTES = 400 * 1024 * 1024


def _decode(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            pass
    return content.decode("utf-8", errors="replace")


def _numbered(text: str, *, csv_mode: bool = False) -> list[dict[str, object]]:
    if csv_mode:
        rows = list(csv.reader(_decode(text).splitlines()))
        blocks = [" | ".join(cell.strip() for cell in row if cell.strip()) for row in rows]
    else:
        blocks = [part.strip() for part in re.split(r"\n\s*\n|(?<=\.)\s{2,}", text) if part.strip()]
        if len(blocks) < 2 and text.strip():
            blocks = [line.strip() for line in text.splitlines() if line.strip()]
    return [{"page": n, "text": block} for n, block in enumerate(blocks, 1)]


class _TextHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "noscript"}:
            self.hidden += 1
        elif tag in {"p", "div", "li", "br", "tr", "h1", "h2", "h3", "section"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.hidden:
            self.hidden -= 1
        elif tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "section"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden and data.strip():
            self.parts.append(data.strip())


def _zip(content: bytes) -> ZipFile:
    archive = ZipFile(BytesIO(content))
    if sum(info.file_size for info in archive.infolist()) > _MAX_UNCOMPRESSED_BYTES:
        archive.close()
        raise ValueError("This Office file expands beyond the safe processing limit.")
    return archive


def _xml(data: bytes) -> ET.Element:
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("This Office file contains unsupported XML declarations.")
    return ET.fromstring(data)


def extract_office_sections(content: bytes, suffix: str) -> list[dict[str, object]]:
    """Extract DOCM/DOCX paragraphs, PPTX slides, or XLSX rows as citations."""
    try:
        with _zip(content) as archive:
            names = set(archive.namelist())
            if suffix in {".docx", ".docm"}:
                path = "word/document.xml"
                if path not in names:
                    raise ValueError("This Word file has no readable document text.")
                root = _xml(archive.read(path))
                ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
                blocks = []
                for paragraph in root.iter(f"{ns}p"):
                    text = "".join(node.text or "" for node in paragraph.iter() if node.tag in {f"{ns}t", f"{ns}instrText"})
                    if text.strip():
                        blocks.append(re.sub(r"\s+", " ", text).strip())
                pages = [{"page": i, "text": text} for i, text in enumerate(blocks, 1)]
            elif suffix == ".pptx":
                slides = sorted((n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)), key=lambda n: int(re.search(r"slide(\d+)", n).group(1)))
                pages = []
                for i, name in enumerate(slides, 1):
                    root = _xml(archive.read(name))
                    text = " ".join(node.text or "" for node in root.iter() if node.tag.endswith("}t"))
                    pages.append({"page": i, "text": re.sub(r"\s+", " ", text).strip()})
            elif suffix == ".xlsx":
                shared: list[str] = []
                if "xl/sharedStrings.xml" in names:
                    root = _xml(archive.read("xl/sharedStrings.xml"))
                    shared = ["".join(t.text or "" for t in item.iter() if t.tag.endswith("}t")) for item in root]
                sheets = sorted((n for n in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)), key=lambda n: int(re.search(r"sheet(\d+)", n).group(1)))
                pages = []
                for i, name in enumerate(sheets, 1):
                    root = _xml(archive.read(name))
                    rows = []
                    for row in (node for node in root.iter() if node.tag.endswith("}row")):
                        cells = []
                        for cell in (node for node in row if node.tag.endswith("}c")):
                            val = next((n.text or "" for n in cell if n.tag.endswith("}v")), "")
                            inline = "".join(n.text or "" for n in cell.iter() if n.tag.endswith("}t"))
                            if cell.attrib.get("t") == "s" and val.isdigit() and int(val) < len(shared):
                                val = shared[int(val)]
                            else:
                                val = inline or val
                            if val.strip():
                                cells.append(val.strip())
                        if cells:
                            rows.append(" | ".join(cells))
                    pages.append({"page": i, "text": "\n".join(rows)})
            else:
                raise ValueError("Unsupported Office document type.")
    except (BadZipFile, OSError, ET.ParseError, KeyError) as exc:
        raise ValueError("This Office file is invalid or its text could not be read.") from exc
    if not any(str(page["text"]).strip() for page in pages):
        raise ValueError("No readable text was found in this file.")
    return pages


def extract_text_sections(content: bytes, suffix: str) -> list[dict[str, object]]:
    text = _decode(content)
    if suffix == ".html" or suffix == ".htm":
        parser = _TextHTML()
        parser.feed(text)
        text = "\n".join(parser.parts)
    elif suffix == ".rtf":
        text = re.sub(r"\\'[0-9a-fA-F]{2}", " ", text)
        text = re.sub(r"\\[a-zA-Z]+-?\d* ?|\\[^a-zA-Z]", " ", text)
        text = text.replace("{", " ").replace("}", " ")
    elif suffix == ".json":
        try:
            obj = json.loads(text)
            text = json.dumps(obj, ensure_ascii=False, indent=2)
        except json.JSONDecodeError as exc:
            raise ValueError("The JSON file is invalid.") from exc
    pages = _numbered(text, csv_mode=suffix == ".csv")
    if not any(str(page["text"]).strip() for page in pages):
        raise ValueError("No readable text was found in this file.")
    return pages
