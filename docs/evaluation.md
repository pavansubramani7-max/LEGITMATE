# Evaluation

## Property/civil retrieval benchmark

The offline benchmark uses the versioned gold set at
`data/evaluation/property_civil_cases_v2.json`; v1 is retained unchanged.
Version 2 contains 50 questions over the three bundled synthetic PDFs: 30
answerable questions with exact document/page/evidence-span labels and 20
deliberately unsupported questions. Fourteen additional unsupported questions
probe omitted property and civil details. The set covers financial terms,
parties, dates, property descriptions, obligations, claims, and cross-document
sale-price/date conflicts. The benchmark reports
retrieval quality and citation support for answerable questions, plus abstention
and false-answer rates for unsupported questions, overall and by category.

Run it from the project root with:

```powershell
.venv\Scripts\python.exe -m app.evaluation.property_civil_benchmark
```

It reads the bundled PDFs and gold set, loads available models offline, and
writes only `data/processed/property_civil_benchmark/latest.json`. It does not
modify the uploaded-document index. The sample corpus is synthetic and small;
this is a regression benchmark, not evidence of real-world legal accuracy.
The per-query ranked results include graph path metadata where graph expansion
contributed a candidate, so those links can be inspected alongside the gold
evidence labels.

Latest local run on the 50-question v2 synthetic set:

| Mode | MRR@5 | nDCG@5 | Positive hit@5 | Supported citations | Unsupported abstention |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.917 | 0.928 | 1.000 | 1.000 | 0.200 |
| Semantic | 0.861 | 0.895 | 1.000 | 1.000 | 0.000 |
| Hybrid | 0.900 | 0.924 | 1.000 | 1.000 | 0.000 |
| Graph | 0.917 | 0.936 | 1.000 | 1.000 | 0.000 |
| Reranked | 0.933 | 0.949 | 1.000 | 1.000 | 0.000 |

This run retrieved at least one labeled passage in the top five for all 30
answerable questions. BM25 abstained on 4/20 unsupported questions; Semantic,
Hybrid, Graph, and Reranked abstained on 0/20. Treat that as a concrete
limitation. The query set is synthetic, and its additional negatives are not
an independently reviewed test set; do not tune a production cutoff against
these examples.

Graph mode expands hybrid results through exact normalized entity mentions. It
scored MRR@5 0.917 and nDCG@5 0.936 in this run, but also answered all twenty
unsupported queries. Inspect each `graph_path` and cited source; graph links
are navigation cues, not evidence that a legal relationship exists.

The saved run's paired Graph-vs-Hybrid comparison over the same 30 answerable
questions found Graph MRR@5 better on 1 query, tied on 29, and worse on 0; the
mean MRR@5 difference was +0.0167. A graph path appeared in the top five for
3/30 answerable queries. This describes behavior on the bundled examples only;
the small, hand-authored set supports no statistical or real-world accuracy
claim. The per-query comparison is in `latest.json` under
`graph_vs_hybrid_paired`.

## Running the evaluation

Run the offline benchmarks (no server and no access to the live upload index):

```powershell
.venv\Scripts\python.exe -m app.evaluation.property_civil_benchmark
.venv\Scripts\python.exe -m app.evaluation.extraction_benchmark
.venv\Scripts\python.exe -m app.evaluation.workspace_isolation_check
```

The legacy API evaluation is separate and requires the app server plus its
sample PDFs to be present in the active workspace:

1. Start the server: `python -m uvicorn server:app --reload`
2. Upload the sample PDFs via the UI or API
3. Run: `.venv\Scripts\python.exe -m app.evaluation.evaluate`

## Extraction benchmark

The versioned labels at `data/evaluation/extraction_cases_v1.json` cover the
three bundled synthetic PDFs (six pages). The latest run measured precision
and recall of 1.000 for the labeled person, organization, property identifier,
date, amount, and claim fields. It detected both of the two manually labeled
conflicts. This small synthetic result is a regression check only; it does not
estimate accuracy on real or scanned legal documents. OCR is not evaluated by
this benchmark.

An annotation protocol and JSONL record format for a future independently
reviewed, matter-grouped property/civil corpus is documented in
[`data/evaluation/REVIEWED_CORPUS_GUIDE.md`](../data/evaluation/REVIEWED_CORPUS_GUIDE.md).
No real-world reviewed corpus is bundled and no such corpus-level score is
claimed. Scanned/OCR and multilingual performance need separately labeled,
stratified evaluation before they can be reported.

The runner accepts that adjudicated JSONL format and reports per-kind
precision/recall/F1, date-normalization and event-cue accuracy, explicit
interval-link precision/recall/F1, retrieval Hit@5/Recall@5/MRR@5/nDCG@5,
gold-citation coverage, supported-answer BLEU/ROUGE-L wording overlap,
unsupported-query abstention, and per-language/per-OCR strata. Wording overlap
does not establish factual correctness. If the development split contains both
supported and unsupported queries, the runner selects a score cutoff on dev
only and reports its answerability and citation results separately on test.
Small development sets can overfit; the cutoff is not a universal confidence
value or a changed production default. The runner requires two annotators,
adjudicator IDs, unique record IDs, matter-grouped
train/dev/test splits, evidence offsets, and source SHA-256 metadata. It writes
aggregate metrics only, without source text or extracted values:

```powershell
.venv\Scripts\python.exe -m app.evaluation.reviewed_corpus_benchmark --input path\to\adjudicated.jsonl
```

Use `--modes bm25 hybrid graph` to select methods and `--output` to choose a
report destination. This evaluator does not create or certify human labels.

### Blind annotation and adjudication workflow

The annotation workbench prepares separate copies for two reviewers, compares
their field-level agreement, creates a private adjudication worksheet, and
validates the adjudicator's final decisions before they reach the benchmark
runner. It rejects different source text/hash metadata between reviewers and
prevents one matter group from crossing train/dev/test splits. It never fills
labels automatically.

Keep the source manifest and reviewer/worksheet files in access-controlled
storage: they contain source text and may contain personal or privileged
information. The compare command reports agreement counts only; its worksheet
contains both reviewers' labels and must not be shared as an aggregate report.

```powershell
.venv\Scripts\python.exe -m app.evaluation.annotation_workbench prepare --source pages.jsonl --source-root authorized-originals --reviewer-a reviewer-01 --reviewer-b reviewer-02 --out-dir private-reviews
.venv\Scripts\python.exe -m app.evaluation.annotation_workbench compare --reviewer-a private-reviews\reviewer-a.jsonl --reviewer-b private-reviews\reviewer-b.jsonl --output private-reviews\adjudication.jsonl
# Adjudicator fills adjudication.adjudicator_id and adjudication.labels for every record.
.venv\Scripts\python.exe -m app.evaluation.annotation_workbench finalize --worksheet private-reviews\adjudication.jsonl --adjudicator adjudicator-01 --source-root authorized-originals --output private-reviews\adjudicated.jsonl
.venv\Scripts\python.exe -m app.evaluation.reviewed_corpus_benchmark --input private-reviews\adjudicated.jsonl
```

The tool cannot establish that reviewer IDs correspond to qualified or
independent people. That is an organizational control and must be attested
outside the software. Keep raw annotations, adjudication worksheets, the
signed-off split manifest, and aggregate benchmark reports under separate
access rules.

## Workspace isolation check

`app.evaluation.workspace_isolation_check` creates a temporary account database
and three temporary document indexes, then verifies that the admin and two
customers each see only their own index. It cleans up its temporary data and
does not access the real account database or uploaded documents. This verifies
local storage routing, not security against a compromised server or operating
system account.

## Test cases

| ID | Question | Expected | Check |
|----|----------|----------|-------|
| TC-01 | What is the sale price? | "500000" on page 1 | Retrieval hit + page citation |
| TC-02 | What date was the agreement signed? | Conflict between docs | Conflict note present |
| TC-03 | Who are the parties? | "Sharma" on page 1 | Retrieval hit |
| TC-04 | What is the zoning classification? | No answer | Abstention |

## What is measured

- **Retrieval hit**: expected keyword appears in top-5 evidence passages
- **Page citation**: cited page matches expected page
- **Conflict surfaced**: `conflict_note` field present in response
- **Abstention**: `abstained: true` when no relevant evidence exists

## What is not measured

- Legal accuracy (the system does not make legal determinations)
- Generalization of extraction precision/recall beyond the tiny synthetic gold set
- OCR quality on scanned, low-resolution, rotated, or multilingual legal documents
- Performance at scale (not tested beyond a few documents)

All reported results come from actually running the evaluation script.
No numbers are invented or estimated.
