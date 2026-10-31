from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    file_type: Literal["pdf", "docx"] = "pdf"
    citation_unit: Literal["page", "section"] = "page"
    pages: int
    text_pages: int
    uploaded_at: str
    status: str
    ocr_needed: bool


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    document: str
    page: int
    citation_unit: Literal["page", "section"] = "page"
    text: str


class ScoredChunk(Chunk):
    score: float


class ExtractedItem(BaseModel):
    item_id: str
    kind: Literal["person", "org", "property", "date", "amount", "claim", "event"]
    value: str
    context: str          # surrounding passage text
    document_id: str
    document: str
    page: int
    citation_unit: Literal["page", "section"] = "page"
    review_status: Literal["candidate", "reviewed", "rejected", "unresolved"] = "candidate"
    note: Optional[str] = None
    event_date: Optional[str] = None   # ISO date string for events/dates


class ConflictRecord(BaseModel):
    conflict_id: str
    kind: str             # e.g. "date", "amount", "claim"
    label: str            # human-readable description
    items: list[str]      # item_ids involved
    status: Literal["open", "resolved"] = "open"


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class SearchRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)
    retrieval_mode: Literal["bm25", "semantic", "hybrid", "graph", "reranked"] = "bm25"
    use_ai: bool = False
    chat_mode: Literal["documents", "live"] = "documents"
    chat_history: list[ChatTurn] = Field(default_factory=list, max_length=10)
    matter_id: Optional[str] = Field(default=None, max_length=64)


class MatterCreateRequest(BaseModel):
    title: str = Field(min_length=2, max_length=120)
    description: str = Field(default="", max_length=2000)


class MatterPatchRequest(BaseModel):
    title: Optional[str] = Field(default=None, min_length=2, max_length=120)
    description: Optional[str] = Field(default=None, max_length=2000)


class DocumentMatterPatch(BaseModel):
    matter_id: Optional[str] = Field(default=None, max_length=64)


class LanguageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


class ReviewPatch(BaseModel):
    review_status: Optional[Literal["candidate", "reviewed", "rejected", "unresolved"]] = None
    note: Optional[str] = None
    value: Optional[str] = None
    event_date: Optional[str] = None
    claim_evidence_judgment: Optional[Literal["unreviewed", "supports", "contradicts", "ambiguous", "no_direct_support"]] = None
    claim_evidence_refs: Optional[list[str]] = Field(default=None, max_length=100)
    claim_evidence_note: Optional[str] = Field(default=None, max_length=2000)


class ManualItemRequest(BaseModel):
    kind: Literal["person", "org", "property", "date", "amount", "claim", "event"]
    value: str
    context: str = ""
    document_id: str
    document: str
    page: int
    event_date: Optional[str] = None
