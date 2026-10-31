# Architecture

## Overview

LEGITMATE is a single-process FastAPI application. All data is stored locally in `data/`.

```
Browser  ──HTTP──►  FastAPI (app.py / server.py)
                        │
              ┌─────────┼──────────────────┐
              │         │                  │
         ingestion   search/             extraction/
         pdf_reader  ├─ lexical.py (BM25  entities, dates, claims
         chunker     │   + synonym map)        │
              │      ├─ retrieval/         evidence/
              │      │   semantic.py       conflicts, timeline,
              │      │   hybrid.py         knowledge_graph.py
              │      └─ retrieval/         qa/answer
              │          hybrid.py         (cited answer
              │          (BM25 + active     + abstention)
              │           via RRF)               │
              └──────────────────────────────────────┘
                                       data/index.json  (JSON store)
                                       data/processed/*.npz  (vector cache)
```

## Data flow

1. User uploads PDF or DOCX → `pdf_reader.extract_pages` or `docx_reader.extract_docx_sections` → `chunker.make_chunks`
2. Chunks stored in `index.json["chunks"]`
3. `retrieval/semantic.save_embeddings` encodes passage vectors and caches them
   in `data/processed/<doc_id>.npz` (neural backend) or skips (LSA/TF-IDF fallback)
4. `extraction/` modules run over page/section text → items stored in `index.json["items"]`
5. `conflicts.detect_conflicts` runs over all items → stored in `index.json["conflicts"]`
6. On `/api/ask`:
   - `retrieval_mode=bm25`: `lexical.rank` (with synonym expansion) scores chunks
   - `retrieval_mode=semantic`: `retrieval/semantic.rank` uses E5 when available; otherwise LSA or TF-IDF
   - `retrieval_mode=hybrid`: `retrieval/hybrid.rank` fuses BM25 + the active semantic backend via RRF (k=60)
   - `retrieval_mode=reranked`: hybrid retrieval passes its candidate passages through the cached MiniLM cross-encoder, which reorders candidates by query/passage score; original abstention scores and gates are preserved
   - `retrieval_mode=graph`: hybrid retrieval plus bounded symbolic expansion from query-matched entities to same-page facts and exact normalized multi-token entity mentions in other documents
   - `qa/answer.build_answer` applies abstention threshold and constructs cited response
7. `GET /api/knowledge-graph` assembles a read-only graph from the current index. It links extracted items to documents and evidence passages, links exact normalized repeated entity mentions across documents, records conflict edges and human-selected claim-to-evidence references, adds explicit interval edges only when range language is found, and adds date-order edges only within each source document. Date nodes include transparent event-type cues inferred from surrounding text. The graph is rebuilt from indexed data; no separate graph database is used.
8. `GET /api/overview` supplies live corpus counts, recent document metadata, and the semantic backend configuration for the dashboard.

## Storage

Single `data/index.json` with four top-level arrays:
- `documents` — document metadata
- `chunks` — passage chunks with doc/page references
- `items` — extracted candidates (persons, dates, claims, etc.)
- `conflicts` — detected conflicts between items

Uploaded files: `data/documents/{uuid}.{extension}` (supported readable formats, maximum 200 MB per file)
Vector cache: `data/processed/{uuid}.npz` (invalidated if model identifier changes)
For newly uploaded documents, the account index stores an ingestion-time SHA-256. Optional source-file case-bundle export checks the file against that fingerprint and excludes changed sources. Older records without a fingerprint receive an export-time digest only; that is not proof of their upload-time identity.

Sensitive matter, document, account, and item-review actions append metadata-only events to the account-local `processed/audit/events.jsonl` hash chain. The integrity endpoint exposes verification status only. The file is not immutable and remains subject to local filesystem access, backup, and key-management risks.

## Limitations

- Single-user local authentication; session and CSRF protection are implemented for the prototype
- JSON store is not suitable for large corpora (>100 documents)
- Extraction uses regex patterns; accuracy depends on document formatting
- Optional local Tesseract OCR for low-text PDF pages when the selected language pack is installed; OCR output needs visual review
- DOCX citations use numbered text sections, not page numbers, because Word pagination can change across devices
- Legacy binary `.doc` files are not supported; save them as `.docx` or PDF first
- Language and topic labels are heuristic estimates and must be reviewed
- Neural abstention threshold cannot separate relevant from irrelevant queries on a
  small corpus; see `docs/limitations.md` for details
- Graph links are deterministic exact-name, page-co-occurrence, and reviewer-selected evidence-reference links, not alias resolution, a validated legal ontology, or an automatic semantic contradiction model. Graph expansion follows explicit query-matched entity mentions; relevance scores remain separate from graph ranking bonuses. Human claim judgments are reviewer annotations, not legal conclusions.
- Blind annotation preparation/comparison and adjudication validation are available as local CLI tools. They do not authenticate reviewer qualifications or produce legal-domain evidence without independently labeled source records.
