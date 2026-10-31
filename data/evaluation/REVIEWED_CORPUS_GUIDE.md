# Reviewed property/civil evaluation corpus

This folder contains a format and protocol for a future independently reviewed corpus. The bundled PDFs and current benchmark labels are synthetic; they are not a substitute for reviewed real-world documents. Do not report scores from this template as legal-domain validation.

## Record format

Create one UTF-8 JSON object per line in a new file (keep source documents separately, access-controlled, and authorized for research). Each record should describe one page/section and its gold annotations:

```json
{"case_id":"case-0001-page-1","document_group":"matter-0001","split":"test","source_file":"redacted.pdf","source_sha256":"0000000000000000000000000000000000000000000000000000000000000000","page":1,"citation_unit":"page","language":"en","script":"Latin","ocr_status":"native_text","text":"On 15 March 2023, Example Person paid Rs. 5,00,000. The buyer paid the amount.","annotators":["reviewer-a","reviewer-b"],"adjudicated_by":"reviewer-c","entities":[{"kind":"person","text":"Example Person","start":18,"end":32}],"dates":[{"text":"15 March 2023","start":3,"end":16,"normalized":"2023-03-15","event_type":"payment"}],"amounts":[{"text":"Rs. 5,00,000","start":38,"end":50,"normalized":"500000","currency":"INR"}],"intervals":[],"claims":[{"text":"The buyer paid the amount.","start":52,"end":78,"claim_type":"payment","polarity":"asserted"}],"retrieval_queries":[{"question_id":"q-01","question":"When was the payment made?","supported":true,"evidence_spans":[[3,16]]}],"answer_references":[{"question_id":"q-01","reference":"Payment date stated as 15 March 2023.","acceptable_facts":["2023-03-15"],"must_cite_spans":[[3,16]]}]}
```

The example is illustrative and is not part of any scored dataset; replace its placeholder SHA-256 with the actual hash of the authorized source file. Offsets are zero-based Unicode code-point offsets into `text`, with an exclusive end. Keep the original OCR text and annotate what the application received; preserve a separate corrected transcription if one is needed. Record OCR status as `native_text`, `ocr`, or `mixed`, plus language/script. Use ISO dates only when the text supports an unambiguous normalization; otherwise set `normalized` to `null` and explain in a private adjudication note. Optional `intervals` records use `start_span`, `end_span`, and relation `from_to` or `between_and`. Every retrieval query must include `supported`; use an empty `evidence_spans` list only for an adjudicated unsupported query.

## Review protocol

Use the separate blind annotation workflow in [`docs/evaluation.md`](../../docs/evaluation.md#blind-annotation-and-adjudication-workflow) to prepare two reviewer files, compare fields, and validate adjudication before scoring. Keep the original documents in a private `--source-root` folder and set `source_file` to a relative path beneath it. The workbench verifies every original file's SHA-256 during preparation and finalization, rejects paths outside that folder, and checks matter-group splits. It cannot verify reviewer qualifications or independence. Reviewer files and adjudication worksheets contain the full source text and must remain private.

1. Use documents you are authorized to process; redact personal identifiers before sharing or committing examples. Record source hashes and never commit sensitive originals.
2. Group all pages from the same matter under one `document_group`. Set `split` to `train`, `dev`, or `test`, and split by matter so pages from one dispute cannot leak across splits.
3. Have two reviewers label independently, then adjudicate disagreements. Preserve both annotator IDs and the adjudicator ID. Define a written policy for names, aliases, copied boilerplate, ambiguous dates, OCR errors, and claim boundaries before labeling.
4. Label evidence spans and event types from the text. For contradiction/support, record the exact cited spans and a reviewer judgment; do not infer truth from repetition or source-text matching.
5. Include native, scanned/OCR, and multilingual records in separate reported strata. Report counts by document, page, language/script, and OCR status, as well as precision, recall, F1 and error examples for each extraction kind.
6. For retrieval, keep query and gold evidence spans together. Report Hit@k, Recall@k, MRR/nDCG, citation-span accuracy, and unsupported-query abstention. For answers, use adjudicated references and report text metrics separately from fact/citation accuracy.
7. Publish corpus provenance, license/consent, exclusions, adjudication policy, and split manifest. Never describe synthetic or weakly labeled public QA data as adjudicated property-document evidence.

## Current project status

No larger reviewed legal-document corpus is bundled. Existing benchmark samples are synthetic and small. The runner is available as `python -m app.evaluation.reviewed_corpus_benchmark --input path/to/adjudicated.jsonl`; it requires the matter-grouped test split and reports aggregate extraction/temporal/retrieval metrics, reference-answer BLEU/ROUGE-L wording overlap, citations, and language/OCR strata without writing source text or values into the report. Text-overlap scores do not establish fact correctness. When the dev split includes both supported and unsupported queries, the runner selects an answerability threshold on dev only and reports its results separately on test. Small dev sets can overfit; do not use the measured threshold as a universal confidence value. Run modes can be selected with `--modes bm25 hybrid graph`. Consequently, this project does not claim real-world extraction, multilingual OCR, legal support, or unsupported-answer rates until authorized reviewed records are supplied and evaluated.
