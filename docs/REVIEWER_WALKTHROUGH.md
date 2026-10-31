# LEGITMATE reviewer walkthrough

This is a short, reproducible demonstration of the current local research
prototype. Use only the bundled synthetic examples unless you have authority
to process other documents. Findings are review candidates, not legal advice.

## Start

From the project folder in PowerShell:

```powershell
.\run_legitmate.ps1
```

Open `http://127.0.0.1:8000` and sign in with the administrator account already
configured for this local installation. The launcher does not create or reset
accounts. If setup is new, follow the README administrator configuration first.

## Five-minute product walkthrough

1. **Upload:** add one of the synthetic PDFs from `data/sample/`. Show the
   upload report and retained page locations. For an authorized scanned PDF,
   enable local OCR, select a language pack installed in Tesseract, and show
   the per-page OCR warning signals. Explain that these are not confidence
   scores and every OCR finding needs visual source review.
2. **Ask:** ask a question whose answer is stated in the document. Open its
   citation and verify the cited passage and page in the source.
3. **Review a claim:** open the claims/evidence review view. Show source-text
   provenance separately from truth, apply a human evidence judgment, attach
   related evidence items, and save the review state.
4. **Inspect time and links:** open the timeline and knowledge graph. Point out
   date-event cues, explicit interval links, entity connections, and their
   source references. These are navigation and review aids; a graph edge does
   not prove a legal relationship.
5. **Try an unsupported question:** check whether the system abstains and
   inspect the cited passages. The included synthetic benchmark shows that
   abstention is a known weakness, so do not present it as solved.

## Reproduce the included evaluations

Run from the repository root. These commands use the bundled synthetic set and
do not score a real-world legal corpus:

```powershell
.venv\Scripts\python.exe -m app.evaluation.property_civil_benchmark
.venv\Scripts\python.exe -m app.evaluation.extraction_benchmark
.venv\Scripts\python.exe -m app.evaluation.workspace_isolation_check
```

The retrieval report is written to
`data/processed/property_civil_benchmark/latest.json`. It includes paired
Graph-vs-Hybrid results on the same answerable queries, unsupported-query
abstention, per-query rankings, and graph-path metadata. Read
[`evaluation.md`](evaluation.md) for the metrics and their limits.

## What to say about evidence and scores

- A source match means the extracted item can be traced to text; it does not
  establish that the statement is true.
- A support/contradiction label is a human review decision recorded with
  source-item references, not an automated legal conclusion.
- The bundled retrieval and extraction scores are regression checks over
  synthetic examples. They do not estimate accuracy on real contracts, court
  records, scanned files, or other languages.
- BLEU/ROUGE measure wording overlap, not factual correctness. A single source
  document cannot serve as an independent reference answer.
- Real-domain claims require an authorized, de-identified or appropriately
  protected corpus, independent annotations, adjudication, matter-separated
  splits, and reported errors. The project provides the annotation protocol
  and evaluator, but no real reviewed corpus is bundled.
