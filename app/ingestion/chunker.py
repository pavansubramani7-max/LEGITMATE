from __future__ import annotations
import re
import uuid
from typing import Any
from ..config import CHUNK_MAX, CHUNK_SPLIT


def _chunk(doc_id: str, name: str, page: int, text: str) -> dict[str, Any]:
    return {"chunk_id": str(uuid.uuid4()), "document_id": doc_id, "document": name, "page": page, "text": text}


def make_chunks(doc_id: str, name: str, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for page in pages:
        paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", page["text"]) if p.strip()]
        if not paragraphs and page["text"].strip():
            paragraphs = [re.sub(r"\s+", " ", page["text"]).strip()]
        buffer = ""
        for para in paragraphs:
            if len(para) > CHUNK_MAX * 1.3:
                if buffer:
                    output.append(_chunk(doc_id, name, page["page"], buffer))
                    buffer = ""
                for i in range(0, len(para), CHUNK_SPLIT):
                    output.append(_chunk(doc_id, name, page["page"], para[i: i + CHUNK_MAX]))
            elif len(buffer) + len(para) > CHUNK_MAX and buffer:
                output.append(_chunk(doc_id, name, page["page"], buffer))
                buffer = para
            else:
                buffer = (buffer + " " + para).strip()
        if buffer:
            output.append(_chunk(doc_id, name, page["page"], buffer))
    return output
