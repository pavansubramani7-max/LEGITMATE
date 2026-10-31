# Current model status and limitations

## Search models

- The app uses the pretrained `intfloat/multilingual-e5-small` embedding model when the Python package and model cache are available. It has 384-dimensional embeddings and is used for semantic retrieval.
- The current offline benchmark run loaded E5 and the cached cross-encoder successfully. This confirms those cached models can run in the benchmark environment; it is not a deployment or legal-quality guarantee.
- The optional **Combined + cross-encoder reranker** mode uses `cross-encoder/ms-marco-MiniLM-L-2-v2` from the local Hugging Face cache. The model is Apache-2.0 licensed. It scores query/passage pairs and reorders hybrid candidates; it does not replace the underlying relevance threshold or guarantee an answer.
- The cross-encoder was trained for general English passage ranking, not property-law judgments. On the 36-question synthetic regression set, the lowest best-matching gold-passage score was 0.000698 while an unsupported question's highest score was 0.416018. The scores overlap, so a single global cross-encoder cutoff cannot safely fix abstention on this set.
- If E5 is unavailable, semantic search falls back to LSA or TF-IDF. If the cross-encoder cannot load, reranked mode falls back to ordinary hybrid ranking.
- A local smoke check loaded both model families. On a synthetic sale-price query, the relevant passage ranked first (cross-encoder score 0.999833); this confirms inference works, not legal-domain accuracy.

## Extraction and graph limits

- Person/organisation NER uses spaCy only if `en_core_web_sm` is installed; otherwise the current implementation falls back to regex. Property identifiers, dates, amounts, and claim sentences also rely on patterns and can produce false positives or misses.
- Extracted facts remain **Candidate** until a human reviews them. A source-page match shows where text appears; it does not confirm that the statement is true. Claim review can record a human judgment (`supports`, `contradicts`, `ambiguous`, or `no direct support`) and link selected extracted items. These labels capture a reviewer's assessment; they are not automatic legal support scores or legal conclusions.
- Date and amount conflict detection uses conservative context buckets and can still pair unrelated statements. The sample extraction benchmark checks two known conflicts, but it is synthetic and small. Automatic semantic comparison of claim sentences is disabled until it can be evaluated with reviewed labels. LEGITMATE shows source passages side by side and does not choose which one is legally correct.
- The temporal graph assigns rule-based event cues from nearby words, links range endpoints only for explicit “from … to” or “between … and” wording, and orders dated items only within the same source document. It does not resolve event identity across instruments or establish a legally valid chronology. Date normalization and event-type cues still need reviewed-document evaluation.
- GraphRAG adds same-page facts and exact normalized multi-token entity mentions across documents when a query names an extracted entity. This does not resolve aliases, common-name collisions, or semantic identity and is not unrestricted multi-hop traversal.
- Exact source-text matching on a claim is a provenance check. It does not establish whether the claim is true, legally operative, or contradicted by a different statement.

## Features not included

- Optional local OCR is available for scanned PDF pages when Tesseract and compatible language data are installed. The upload form automatically estimates language from readable text or uses local script detection for scan-only PDFs when Tesseract OSD data is present. If a detected pack is missing, the report records the installed fallback. Script is not the same as language; Latin-script scan-only content falls back to English when no stronger signal is available. Per-page text-quality indicators are heuristics, not OCR confidence values. OCR has not been measured by the bundled text-PDF benchmarks; visually review OCR-dependent findings. Without Tesseract, normal text extraction remains available and scanned pages may produce little or no text.
- Language detection is an offline heuristic: common Latin-script languages and writing systems have cues, but short, mixed-language, transliterated, or uncommon text may be uncertain or mislabeled.
- Automatic topic labels are keyword cues for common property/civil subjects, not a trained legal classifier. Extracted items are candidates and require source review.
- Files larger than 200 MB, legacy binary `.doc`, and unlisted formats are not accepted.
- No custom fine-tuned legal model. The app uses pretrained models; the experimental fine-tuning script has not produced a trained app model.
- Groq requires the user's own API key in `.env`. Live chat sends the message and recent turns; document mode sends the question and retrieved excerpts only after opting in. Generated document answers can still be wrong and must be checked against citations.
- Language labels use local script and common-word heuristics. They are estimates, not a full multilingual language-identification model.
- No POS tagging or lemmatization pipeline.
- Local admin/customer accounts now have isolated workspaces. The prototype still has no SSO, MFA, email verification, distributed rate limiting, or public-deployment hardening.
- PDFs and the JSON index are plain files on disk, without encryption at rest. Use only appropriate local test documents and protect the drive.

## Evaluation status

The last confirmed legacy evaluation was **10/12**: BM25 4/4, semantic 3/4, and hybrid 3/4. Semantic/hybrid abstention failed for an unrelated zoning question on the small sample corpus because irrelevant and relevant embedding scores overlapped.

The expanded 50-question benchmark is a separate, synthetic result (see `docs/evaluation.md` and `data/processed/property_civil_benchmark/latest.json`). Reranking improved ordering on this tiny sample but did **not** fix unsupported-question abstention; Graph, Semantic, Hybrid, and Reranked answered all 20 unsupported questions. Do not tune a cutoff against these synthetic negatives and present it as production confidence; obtain reviewed, appropriately licensed property/civil examples first.

A separate in-memory check using the three sample PDFs (six passages, without writing to `data/index.json`) found that both hybrid and reranked search returned expected keyword/page evidence for TC-01 through TC-03. Both still failed to abstain on TC-04. This is a small smoke evaluation, not a replacement for the project's saved 10/12 evaluation.
