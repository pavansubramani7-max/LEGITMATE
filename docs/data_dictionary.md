# Data Dictionary

## documents[]

| Field | Type | Description |
|-------|------|-------------|
| document_id | string (UUID) | Unique identifier |
| filename | string | Original upload filename |
| file_type | string | `pdf` or `docx` |
| citation_unit | string | `page` for PDFs or `section` for Word files |
| pages | int | Total PDF pages or numbered Word text sections |
| text_pages | int | PDF pages or Word sections with selectable text |
| uploaded_at | ISO datetime | Upload timestamp (UTC) |
| status | string | "ready" |
| ocr_needed | bool | True if some PDF pages have no text |
| ocr_used | bool | True when local OCR produced text for one or more PDF pages |
| ocr_language | string/null | Tesseract language code(s) used for pages actually OCR'd |
| ocr_pages | list[int] | Page numbers where local OCR produced text |
| ocr_skipped_pages | list[int] | Low-text page numbers beyond the 50-page OCR limit |

The upload report's `processing_report.ocr.quality_signals` contains heuristic
per-page OCR text diagnostics. It is not a recognition-confidence score.

## chunks[]

| Field | Type | Description |
|-------|------|-------------|
| chunk_id | string (UUID) | Unique identifier |
| document_id | string | Parent document |
| document | string | Original filename |
| page | int | Source location number (1-based page or Word text section; use citation_unit to interpret) |
| text | string | Passage text |

## items[]

| Field | Type | Description |
|-------|------|-------------|
| item_id | string (UUID) | Unique identifier |
| kind | enum | person, org, property, date, amount, claim, event |
| value | string | Extracted value |
| context | string | Surrounding passage text |
| document_id | string | Source document |
| document | string | Source filename |
| page | int | Source location number (page or Word text section; use citation_unit to interpret) |
| citation_unit | string | `page` or `section` |
| review_status | enum | candidate, reviewed, rejected, unresolved |
| note | string? | Reviewer note |
| event_date | string? | ISO date (for date/event items) |

## conflicts[]

| Field | Type | Description |
|-------|------|-------------|
| conflict_id | string (UUID) | Unique identifier |
| kind | string | date, amount, claim |
| label | string | Human-readable description |
| items | string[] | item_ids involved |
| status | enum | open, resolved |
