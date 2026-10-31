# LEGITMATE

**A Neuro-Symbolic Legal Intelligence Framework for Property and Civil Document Understanding, Evidence–Claim Reasoning, Temporal Knowledge Graphs, and Hybrid GraphRAG**

LEGITMATE is a local research prototype for exploring evidence in property and civil documents. It extracts readable text, indexes traceable passages, identifies candidate facts, and provides question answering with citations. It also offers review tools for possible conflicts, dates, claims, and connected evidence.

> **Research prototype only. Not legal advice and not production-ready.** Extracted facts and conflicts are candidates for human review. Check every important result in the original document. Use only material you are authorized to process.

## Project abstract

Property and civil document review requires finding relevant statements across files while preserving where each statement came from. LEGITMATE is a local-first neuro-symbolic research prototype for property and civil document understanding, evidence–claim review, temporal graph exploration, and Hybrid GraphRAG. It extracts text and source locations from supported documents, divides text into traceable passages, and retrieves evidence with BM25, multilingual E5 embeddings, reciprocal-rank fusion, optional cross-encoder reranking, and bounded graph expansion. Rule-based components identify candidate people, organizations, property identifiers, dates, amounts, and claims. Claims are labeled as source-text matched or unmatched; a match confirms provenance, not truth. The read-only graph links exact normalized multi-token entity mentions across documents and keeps date-order edges within each source document. Possible conflicts and event cues remain review signals, not legal conclusions. An optional AI provider can draft responses from retrieved excerpts; external AI is disabled by default. Separate local account workspaces isolate each account’s files and indexes inside the application.

The NLP Studio contains foundational NLP demonstrations and selectable CNN, vanilla RNN, GRU, and LSTM classifiers trained on synthetic teaching snippets rather than legal records. Evaluation currently uses synthetic samples: the expanded 50-question retrieval set found a labeled passage in the top five for all 30 answerable questions, but unsupported-query abstention remained poor (0.000–0.200 across modes). The extraction sample matched its small hand-labeled set, but this does not establish performance on real documents. On 12 held-out synthetic examples, GRU and CNN scored 0.750, RNN 0.417, and LSTM 0.583. These models are not connected to document answers. No target-domain fine-tuned legal model is deployed. LEGITMATE demonstrates an auditable evidence workflow, not validated legal reasoning or production-grade accuracy.

## At a glance

| Area | Current implementation |
| --- | --- |
| Document processing | PDF, DOCX/DOCM, PPTX, XLSX, TXT, Markdown, CSV, JSON, HTML, and RTF; up to 200 MB per file |
| Text search | BM25 with legal query synonyms, multilingual E5 semantic retrieval, hybrid RRF, optional cross-encoder reranking |
| Evidence analysis | Source-text claim matching, human claim-to-evidence judgments with selected source-item references, candidate facts, heuristic date/amount conflicts, typed date cues and explicit ranges, timeline, provenance graph, extractive case brief |
| NLP learning | Tokenization, optional spaCy POS/lemma, BoW, TF-IDF, n-grams/perplexity, overlap metrics, CNN/RNN/GRU/LSTM teaching demos |
| Optional services | Local Tesseract OCR for selected installed languages; Gemini or Groq chat/drafting when explicitly configured |
| Data separation | Separate local account folders and indexes for admin and customer workspaces; matter filters within each account |
| Case organization | Matter workspaces scope search, evidence review, conflicts, dates, graph, brief, and export |
| Review and handoff | Claim evidence matrix with human judgments, CSV export, and a ZIP case bundle with optional source files |
| Deployment posture | Authenticated readiness checklist; explicitly reports missing production safeguards and independent validation |
| Evidence integrity | Ingestion-time SHA-256 for new uploads, source verification at export, and an account-local hash-chained metadata audit ledger |
| Research workflow | Blind two-reviewer JSONL preparation, field-level agreement comparison, adjudication validation, and held-out scoring CLI |
| Fine-tuned legal model | None deployed; the neural sequence classifiers are synthetic teaching demos, not part of legal document QA |

## How LEGITMATE works

1. A user signs into an individual workspace and uploads a supported file.
2. The reader extracts text and keeps its page, slide, sheet, or section location. Optional PDF OCR runs locally when selected and Tesseract is installed.
3. The app creates source-linked passages and candidate facts. Candidate extraction uses patterns and optional lightweight NLP components; it does not decide whether a fact is true.
4. A question is searched using the selected retrieval method. Results retain the document and page/section location.
5. The answer cites passages. The user can open the evidence, review candidate facts, compare possible conflicts, and inspect the timeline or graph.
6. If the user explicitly selects AI drafting or Live AI chat, the configured provider receives only the text described under [AI and privacy](#ai-and-privacy). The original file is not sent.

## Main features

- Multi-format local document reading, upload reports, file listing, and deletion; opt-in multilingual local PDF OCR with per-page review warning signals.
- Candidate extraction for people, organizations, property identifiers, dates, amounts, and claims.
- BM25, E5 semantic retrieval, hybrid reciprocal-rank fusion, optional cross-encoder reranking, and bounded graph expansion across exact repeated entity mentions.
- Evidence citations by PDF page, presentation slide, spreadsheet sheet, or numbered section.
- Local extractive answers, an evidence-based abstention gate, and optional AI-drafted responses.
- Source-linked claims, review status, heuristic date/amount conflict records, timeline, read-only evidence graph, and extractive case brief.
- Matter workspaces to group documents; the active matter scopes the analysis panels. Existing documents remain unassigned until deliberately moved.
- Evidence matrix for human claim review, CSV download, and a ZIP case bundle containing a manifest, evidence index, timeline, conflicts, brief, and optional originals.
- New uploads receive a source SHA-256; a source whose contents changed is excluded from case bundles. Account-local review/action events are hash-chained and record identifiers/categories, not legal text or reviewer notes.
- Blind two-reviewer annotation preparation, agreement summaries, adjudication worksheet validation, matter-group split checks, and a handoff to the reviewed-corpus evaluator are available in `app.evaluation.annotation_workbench`.
- Readiness & limits panel reports local configuration and production blockers without exposing credentials.
- Separate admin and customer workspaces, protected account management, signed sessions, and CSRF checks.
- NLP Studio learning examples and a selectable CNN/RNN/GRU/LSTM synthetic sequence-model comparison.

## System outline

```text
Browser UI
  ├─ Sign-in, account controls, document upload, review panels, NLP Studio
  └─ Local API requests
       ├─ Readers → source locations → passages → local workspace index
       ├─ Retrieval → evidence snippets → extractive answer and citations
       ├─ Candidate extraction → review items → conflicts and timeline
       ├─ Knowledge graph → bounded named-fact expansion
       └─ Optional AI provider → draft/chat text only when selected
```

The symbolic parts are the transparent extraction rules, source matching, review states, and graph/conflict links. The neural parts are the pretrained E5 encoder, optional cross-encoder reranker, and separate synthetic sequence-model teaching demos. The app does not use these classroom models to answer questions about uploaded legal documents.

## Requirements

- Windows PowerShell instructions are provided below. The application is Python/FastAPI-based.
- Python 3.10 or newer and `pip`.
- Internet access to install packages and, if not cached, obtain pretrained retrieval weights. Cached models can run offline.
- Optional: Tesseract OCR with a matching language pack for scanned PDF pages.
- Optional: Gemini or Groq API access for Live AI chat or AI-drafted document answers.

## Install

Open PowerShell in the project folder:

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py` is unavailable, use `python -m venv .venv` after installing Python.

## Configure the administrator

Create a private environment file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set a unique administrator password (at least 12 characters) and a random session secret:

```env
LEGITMATE_PASSWORD=choose-a-unique-password-at-least-12-characters
LEGITMATE_ADMIN_USERNAME=admin
LEGITMATE_SECRET=paste-a-random-secret-here
```

Generate the secret with:

```powershell
.venv\Scripts\python.exe -c "import secrets; print(secrets.token_hex(32))"
```

Copy the printed value to `LEGITMATE_SECRET`. Do not commit or share `.env`. The initial credentials seed the first administrator. Once the account exists, changing `LEGITMATE_PASSWORD` does not reset the stored account password; use the protected password controls.

## Start and open the app

From the project folder:

```powershell
.\run_legitmate.ps1
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) and sign in. Keep the terminal open while using the app. Press **Ctrl+C** in that terminal to stop it.

Use `server:app` as the Uvicorn target. `server.py` loads the root `app.py`; the `app/` package directory makes `app:app` the wrong target for this repository.

For a guided five-minute demonstration and reproducible evaluation commands, see [the reviewer walkthrough](docs/REVIEWER_WALKTHROUGH.md).

## Upload and process documents

The maximum upload size is **200 MB per file**. Supported extensions are PDF, DOCX, DOCM, PPTX, XLSX, TXT, Markdown (`.md`), CSV, JSON, HTML, and RTF. Legacy binary `.doc` files must be converted to DOCX or PDF. Encrypted, damaged, image-only, or unusually formatted documents may not extract correctly.

1. Open **Upload & documents** and select a file.
2. For scanned PDFs, optionally enable **Try local OCR on scanned PDF pages**. LEGITMATE selects the OCR language automatically from readable document text or, when only scans are available, a local Tesseract script estimate.
3. Review the language estimate, topic cues, candidate findings, and upload report. Open the report or download it as HTML.
4. Open **Ask a question** and choose a retrieval mode.
5. Open citations to inspect their source passages. Review candidate facts and possible conflicts instead of treating them as legal conclusions.

The processing report samples up to 10,000 characters from up to 20 evenly spaced readable pages/sections for some NLP diagnostics. Topic cues and candidate extraction scan all extracted page/section records. These analyses can miss or misread content.

### Optional local OCR

OCR is opt-in and uses Tesseract on this computer; document content is not sent to an OCR service. Install Tesseract with the needed local language data, put `tesseract.exe` on `PATH`, then restart the terminal and server. The language is not selected manually: readable text drives a local language estimate, and scan-only PDFs use Tesseract's script detector when its OSD data is installed. For uncertain text, LEGITMATE chooses an installed fallback pack and labels that fallback in the report. Script detection cannot reliably distinguish languages that share a script; Latin-script scans default to English unless readable text provides a stronger estimate. The workflow targets low-text PDF pages and handles at most 50 OCR pages per file. The report shows simple per-page text-quality warning signals; these are not Tesseract confidence scores and cannot verify the recognized words. Compare names, dates, and amounts with the original page image. OCR quality and multilingual extraction are not covered by the included benchmark.

## Retrieval modes and pretrained models

| Mode/component | What it does | Important limit |
| --- | --- | --- |
| BM25 | Lexical passage ranking with a transparent legal synonym map | Requires vocabulary overlap with the document |
| E5 semantic | Uses `intfloat/multilingual-e5-small` when installed/cached | Scores are not legal confidence; unrelated passages can rank highly |
| LSA/TF-IDF fallback | Local semantic fallback if E5 is unavailable | Corpus-dependent and weaker for small corpora |
| Hybrid | Fuses BM25 and semantic rankings with reciprocal-rank fusion | Does not guarantee correct abstention |
| Cross-encoder | Optionally reorders hybrid candidates with `cross-encoder/ms-marco-MiniLM-L-2-v2` | General passage ranking, not property-law trained |
| Graph expansion | Adds bounded passages when a query names an extracted entity/property; exact normalized multi-token mentions can connect documents | Exact text matching only; no alias resolution or unrestricted graph traversal. Graph links do not manufacture relevance scores |

E5 and the cross-encoder are pretrained retrieval models. LEGITMATE has no validated fine-tuned app model. The small sequence models described below are separate from document retrieval and question answering.

## NLP Studio and CS3250 syllabus coverage

The NLP Studio contains these local demonstrations:

- Unicode tokenization and legal entity candidates, plus automatic language selection for the optional spaCy POS/lemma pipeline. Each analysis reports whether it used a trained installed spaCy model or the transparent rule fallback.
- Bag-of-Words, TF-IDF, bigrams, and smoothed unigram/bigram perplexity.
- BLEU-4, ROUGE-L, precision, recall, and F1 from user-provided reference/candidate text. These measure text overlap, not legal correctness.
- Indic writing-system counts and a lightweight language estimate.
- A selectable CNN, vanilla RNN, GRU, and LSTM sequence-model comparison. The models predict one broad topic cue from a tiny synthetic classroom set (72 training / 12 held-out validation snippets); current validation accuracies are GRU 0.750, CNN 0.750, RNN 0.417, and LSTM 0.583. Scores and pooling weights are not legal confidence, explanations, evidence, or source citations.

These teaching classifiers are not used for document answers. Word2Vec/GloVe/FastText checkpoints and tasks, separate ELMo/BERT/RoBERTa/ALBERT/GPT/T5 NLP Studio tasks, legal-domain fine-tuning, Whisper/Wav2Vec, CLIP, full IndicNLP, translation, and production abstractive summarization remain incomplete. The Studio separates runtime availability, checkpoint presence, and connected tasks, and gives copyable next steps where setup is meaningful. See [docs/nlp-studio.md](docs/nlp-studio.md) for the module-by-module map.

To retrain the sequence teaching models with their fixed seed:

```powershell
.venv\Scripts\python.exe data\pipeline\train_gru_sequence_demo.py
```

This creates checkpoints for GRU, CNN, vanilla RNN, and LSTM using only the synthetic training split in `data/evaluation/sequence_demo_cases_v1.json`. It does not fine-tune the legal retrieval model.

## Evaluation and measured results

The offline scripts below use bundled synthetic documents or temporary account storage. They do not overwrite the live uploaded-document index.

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.venv\Scripts\python.exe _check.py
.venv\Scripts\python.exe -m app.evaluation.property_civil_benchmark
.venv\Scripts\python.exe -m app.evaluation.extraction_benchmark
.venv\Scripts\python.exe -m app.evaluation.workspace_isolation_check
```

Latest recorded retrieval benchmark: **50 queries over three synthetic PDFs** (30 answerable, 20 unsupported). The older 36-query v1 labels remain preserved.

| Mode | MRR@5 | nDCG@5 | Answerable hit@5 | Supported citations | Unsupported abstention |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.917 | 0.928 | 1.000 | 1.000 | 0.200 |
| Semantic | 0.861 | 0.895 | 1.000 | 1.000 | 0.000 |
| Hybrid | 0.900 | 0.924 | 1.000 | 1.000 | 0.000 |
| Graph | 0.917 | 0.936 | 1.000 | 1.000 | 0.000 |
| Reranked | 0.933 | 0.949 | 1.000 | 1.000 | 0.000 |

Every answerable query retrieved a labeled passage in the top five. BM25 abstained on 4/20 unsupported queries; Semantic, Hybrid, Graph, and Reranked abstained on 0/20. Do not tune a production threshold against this synthetic set.

The extraction benchmark used three synthetic PDFs/six pages. The latest recorded run reported 1.000 precision and recall for the labeled entities, dates, amounts, and claims, and found both labeled conflict cases. This measures agreement with a small hand-labeled sample only. OCR is not evaluated.

The retrieval benchmark v2 adds fourteen unsupported questions that target details absent from the three sample PDFs. Graph mode scored MRR@5 0.917 and nDCG@5 0.936, and answered all twenty unsupported questions. These figures are regression checks only and are not legal-domain validation.

An annotation guide and runnable evaluator for a future independently reviewed corpus are in [data/evaluation/REVIEWED_CORPUS_GUIDE.md](data/evaluation/REVIEWED_CORPUS_GUIDE.md). They define matter-grouped splits, double annotation/adjudication, OCR/language strata, citation spans, and the metrics to report. No reviewed real-world corpus is bundled, and no scores are claimed for it.

Date extraction assigns transparent event cues and links interval endpoints only for explicit “from … to …” or “between … and …” phrasing. Those cues preserve source context; they do not establish that an interval is legally operative. Timeline ordering and graph date links remain exploratory.

The sequence models were trained on 72 synthetic examples and measured on 12 held-out synthetic examples. They reached GRU 0.750, CNN 0.750, RNN 0.417, and LSTM 0.583 accuracy; these are not property/civil accuracy estimates.

The separate legacy API evaluation requires the app server and the three sample PDFs uploaded in the active workspace:

```powershell
.venv\Scripts\python.exe -m app.evaluation.evaluate
```

It is distinct from the offline 50-question synthetic retrieval benchmark and reads the active workspace index. See [docs/evaluation.md](docs/evaluation.md) for measurement details.

## Research datasets and fine-tuning

Kaggle-derived corpora and prepared splits are research data, separate from user documents and the live app index. Dataset locations, reported licenses, label caveats, and preparation commands are listed in [docs/kaggle-datasets.md](docs/kaggle-datasets.md). CUAD is contract-focused; Indian Legal QA answers are weak labels; judgment corpora are not property-document evidence. No candidate fine-tuned model is used by the app. Do not promote a model unless it improves a separate, leakage-safe evaluation on reviewed, appropriately licensed target-domain examples.

## Account storage and privacy

- The first administrator can create, disable, enable, or reset customer accounts. The app does not send invitations; share initial passwords securely.
- Each account has a separate document folder, JSON index, and embedding cache. The isolation check verifies local path/index separation; it is not a penetration test or protection against someone who controls the server or operating-system account.
- Existing initial-admin files remain under `data/documents/` and `data/index.json`. Customer files/indexes are under `data/workspaces/<account-id>/`.
- Account records are stored in `data/accounts.sqlite3`; passwords are salted PBKDF2 hashes. Sessions use signed HttpOnly, SameSite=Strict cookies (8-hour default). State-changing API routes use CSRF tokens.
- Files, indexes, and vector caches are plain local files with no application-level encryption at rest. Use OS account protection and drive encryption for sensitive material. Do not expose this development server publicly.
- For an HTTPS reverse-proxy deployment, set `LEGITMATE_COOKIE_SECURE=true` so session cookies carry the Secure attribute. Leave it unset for the bundled localhost HTTP runner. The readiness panel reports this separately; setting it while using plain HTTP will prevent the browser from sending the session cookie.
- Audit history is stored per account in `processed/audit/events.jsonl`. The chain detects accidental edits and unsophisticated tampering; it is not immutable and does not replace external append-only logging, signed evidence custody, or protected backups.
- Login throttling is in-process and resets on restart. There is no SSO, MFA, invitation delivery, distributed rate limiting, or public multi-tenant hardening.

## API overview

The browser uses the FastAPI service. Most workspace endpoints require a signed-in session; state-changing endpoints also use CSRF protection.

| Route group | Main routes | Purpose |
| --- | --- | --- |
| Health and workspace | `GET /api/health`, `GET /api/overview` | Service and current-workspace summary |
| Authentication | `POST /api/login`, `POST /api/logout`, `GET /api/me`, `GET /api/csrf` | Session and CSRF handling |
| Accounts | `GET/POST /api/users`, `PATCH /api/users/{user_id}`, `PUT /api/me/password` | Admin account controls and password change |
| Documents | `GET/POST /api/documents`, `GET /api/documents/{id}/report`, `GET /api/documents/{id}/integrity`, `DELETE /api/documents/{id}` | Upload, list, report, verify an original SHA-256, and remove a document |
| Search and QA | `POST /api/search`, `POST /api/ask` | Retrieval and cited question answering |
| Evidence review | `GET/PATCH /api/items`, `GET/PATCH /api/conflicts`, `GET /api/claims`, `GET /api/timeline`, `GET /api/knowledge-graph`, `GET /api/case-brief` | Candidate facts, conflicts, claims, chronology, graph, and brief |
| Matter workspaces | `GET/POST /api/matters`, `PATCH/DELETE /api/matters/{matter_id}`, `PATCH /api/documents/{document_id}/matter` | Account-local case organization and document assignment |
| Evidence handoff | `GET /api/evidence-matrix`, `GET /api/case-bundle` | Claim review matrix and downloadable case bundle; pass `matter_id` to scope it; `include_sources=true` adds available originals (up to 1 GB) |
| Deployment posture | `GET /api/deployment-readiness` | Non-secret local configuration checklist; production readiness remains false |
| Audit integrity | `GET /api/audit-log/integrity` | Verifies the account-local event chain; returns integrity metadata only |
| NLP Studio | `GET /api/nlp-lab/status`, `POST /api/nlp-lab/analyze`, `POST /api/nlp-lab/metrics`, `POST /api/nlp-lab/sequence-demo` | Syllabus status, local analysis, overlap metrics, selectable CNN/RNN/GRU/LSTM teaching demo |
| Benchmarks | `GET /api/evaluation/property-civil` | Saved synthetic retrieval benchmark summary |

For request schemas, see the FastAPI interactive docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) after starting the server and signing in where required.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Server does not start | Confirm `.env` has `LEGITMATE_PASSWORD` and `LEGITMATE_SECRET`; use `server:app` as the Uvicorn target |
| Sign-in fails | Use the stored account password; changing `.env` does not reset an existing database account |
| Upload returns unsupported format | Convert legacy `.doc` to DOCX/PDF and use one of the listed extensions |
| Scanned PDF has no text | Install Tesseract + English data, restart the server, and select local OCR for that PDF |
| Semantic model appears unavailable | Check whether its package and weights exist in the active Python environment/cache; BM25 remains available |
| Gemini/Groq is slow or unavailable | Check provider configuration, account quota, network access, and selected model; local document retrieval remains usable |
| Sequence demo checkpoint is missing | Run the training command in the NLP Studio section to create the CNN/RNN/GRU/LSTM teaching checkpoints |
| A result looks wrong | Open the cited passage and inspect the original page; candidate extraction and language detection can be wrong |

## Useful commands

| Task | Command |
| --- | --- |
| Start local server | `.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload` |
| Generate synthetic sample PDFs | `.venv\Scripts\python.exe data\sample\generate_samples.py` |
| Logic checks | `.venv\Scripts\python.exe _check.py` |
| Retrieval benchmark | `.venv\Scripts\python.exe -m app.evaluation.property_civil_benchmark` |
| Extraction benchmark | `.venv\Scripts\python.exe -m app.evaluation.extraction_benchmark` |
| Account isolation check | `.venv\Scripts\python.exe -m app.evaluation.workspace_isolation_check` |
| Train/retrain sequence teaching demos | `.venv\Scripts\python.exe data\pipeline\train_gru_sequence_demo.py` |
| Legacy API evaluation | `.venv\Scripts\python.exe -m app.evaluation.evaluate` |

## Repository map

```text
app.py                              FastAPI routes, uploads, reports, account context
server.py                           Uvicorn entry point
index.html, login.html              Workspace UI and sign-in
app/accounts.py, app/auth.py        Accounts, signed sessions, security checks
app/ingestion/                      File readers and passage chunking
app/extraction/                     Candidate entities, dates, and claims
app/search/, app/retrieval/         BM25, semantic search, fusion, reranking
app/evidence/                       Provenance, conflicts, timeline, graph, case brief
app/nlp/                            NLP Studio and sequence-model teaching demos
app/evaluation/                     Offline retrieval, extraction, and isolation checks
data/sample/                        Synthetic demonstration PDFs
data/evaluation/                    Versioned synthetic benchmark labels
data/pipeline/                      Dataset preparation and sequence-demo training scripts
data/raw/, data/processed/          Separate research corpora and derived outputs
models/gru-sequence-demo.pt         Synthetic GRU teaching checkpoint
docs/                               Architecture, demo, datasets, limitations, evaluation
```

## Legal and safety boundary

LEGITMATE does not determine ownership, validate title, resolve disputes, give legal advice, or guarantee correct extraction. Human reviewers must verify important statements against the original file and consult a qualified legal professional for legal decisions.

## What a user does

LEGITMATE is a document-review workspace, not a legal decision engine. A normal review looks like this:

1. Sign in to the account that should own the files.
2. Upload a document and wait for its processing report.
3. Check the detected language and extracted passages. Open the original file when a scan or layout is unclear.
4. Ask a question whose answer should be in the uploaded material. Choose **Keywords/BM25** for the most predictable demo behavior, or compare it with Semantic and Hybrid retrieval.
5. Open each citation and verify the quoted text and location. Review candidate facts, dates, possible conflicts, and the case brief as leads for human review.
6. Download the report if needed. Delete the document from the same account when it is no longer needed.

If a question is not answered by the documents, treat any returned passage as a retrieval candidate—not proof that the question is supported. Semantic and hybrid retrieval can return weakly related passages, and abstention is not reliable enough to replace review.

## What the main terms mean

| Term | In LEGITMATE |
| --- | --- |
| Passage/chunk | A section of extracted text kept with its document and page/section location |
| Evidence citation | A link from an answer or finding to the passage that should be checked in the source |
| Candidate claim/fact | A statement detected by rules and offered for review; it is not verified as true |
| Conflict | Two extracted values or statements that may differ; the app does not decide which one is correct |
| Temporal view | A chronological arrangement of detected dates and events, not a verified legal timeline |
| Knowledge graph | A small set of extracted entities and links used for navigation and bounded retrieval expansion |
| GraphRAG | Retrieval that can use those links to find additional passages; this prototype does not perform unrestricted legal reasoning |
| Live AI | Optional external-provider text generation. It can be slow or unavailable and does not verify legal facts |
