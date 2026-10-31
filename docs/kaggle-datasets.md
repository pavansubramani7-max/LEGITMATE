# Kaggle legal datasets and model plan

## What is present locally

| Dataset | Local location | License reported by Kaggle | How LEGITMATE uses it |
| --- | --- | --- | --- |
| [LegalBenchRAG-CUAD](https://www.kaggle.com/datasets/adityaraghuvanshi999/legalbenchrag-cuad-retrieval-dataset) | `data/raw/cuad/` and existing splits in `data/processed/cuad/` | CC BY 4.0 | Contract retrieval training/evaluation pairs. Existing 80/10/10 query-level splits are preserved. |
| [Indian Legal QA 10K](https://www.kaggle.com/datasets/yogeshm01/indian-legal-qa-dataset-10k-questions) | `data/raw/indqa/` | CC BY 4.0 | Supplementary weak question→answer training/evaluation pairs. Answers are not verified evidence passages. |
| [OpenNyaya Indian Supreme Court judgments](https://www.kaggle.com/datasets/gaurav41/opennyaya-indian-supreme-court-judgement-clean) | `data/raw/opennyaya/` | CC BY 4.0 | Large, separate legal-language research corpus. OCR artifacts remain possible. Not added to user-document RAG. |
| [Indian Case Law Evaluation Corpus (ICLEC)](https://www.kaggle.com/datasets/hrithikraj2537/indian-case-law-evaluation-corpus-iclec) | `data/raw/iclec/` | CC0-1.0 | Separate judgment corpus with case metadata; prepared output is grouped by case ID. |

Attribution, license terms, and upstream dataset pages should be retained with presentations or redistributed artifacts. These datasets do not make LEGITMATE a source of current legal advice.

## Local audit observations

- CUAD contains 6,247 query/evidence pairs across 4,042 unique queries in its existing prepared splits.
- Indian Legal QA has 10,002 source rows; preparation removed 31 exact duplicate case/question/answer pairs and made 9,971 pairs across 1,211 case groups. Only 608 rows have the dataset's `has_reference_pdf` flag set, and 749 have a judgment date. Treat it as weak supervision, not verified source-grounded QA.
- OpenNyaya contains 33,016 Markdown judgment files (about 1.17 GB uncompressed in this local copy). The Kaggle description advertises 1950–2023, while this download's year folders extend through 2026; that discrepancy should be checked before making coverage claims. Spot-checked text still has OCR mistakes.
- ICLEC contains 500 rows and 449 unique case IDs. The full text is present, but fields such as headnote and act are sparse; splits group repeated rows by case ID.

## Prepare reproducible research splits

From the repository root:

```powershell
.venv\Scripts\python.exe data\pipeline\prepare_kaggle_legal.py
```

This writes the case-grouped Indian QA `train.jsonl`, `val.jsonl`, and `test.jsonl`, a compressed ICLEC judgment corpus with case-group splits, and `data/processed/kaggle_legal/dataset_report.json`. It does not alter `data/index.json`, uploaded documents, or the existing CUAD files. The 33,016 OpenNyaya source files remain in `data/raw/` and are not duplicated into the app index.

The authenticated NLP Studio reads this report and shows the corpus sizes, licenses, and caveats under **Kaggle research datasets · separate from your documents**.

## Fine-tuning candidate workflow

The mixed E5 script samples contract pairs plus up to 20% Indian Legal QA answer pairs, then evaluates the base and candidate separately on both held-out datasets:

```powershell
.venv\Scripts\python.exe data\pipeline\finetune_e5_kaggle_mix.py
```

It stops on a CPU-only machine unless `--allow-cpu` is explicitly supplied. A GPU run is recommended. It saves to `models/e5-kaggle-mix-candidate/` and never replaces the app's model automatically. Training loss alone is not proof of improvement. The QA answers are weak labels, so any candidate must improve held-out retrieval while preserving the app's separate property/civil evidence evaluation before it is selected.

After a candidate passes evaluation, set `LEGITMATE_EMBEDDING_MODEL=models/e5-kaggle-mix-candidate` in the private `.env`, then rebuild vector cache files with:

```powershell
.venv\Scripts\python.exe data\pipeline\reindex_embeddings.py
```

The reindex script updates only cached vectors. It does not edit source documents or `data/index.json`. Leave the environment variable unset to keep the pretrained multilingual E5 model as the default.
