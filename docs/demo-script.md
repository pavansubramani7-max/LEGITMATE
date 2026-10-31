# Demo Script

## Setup (one time)

```powershell
cd legitmate
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python data/sample/generate_samples.py
```

Create `.env` with credentials:

```powershell
copy .env.example .env
# Edit .env: set LEGITMATE_PASSWORD and LEGITMATE_SECRET
python -c "import secrets; print(secrets.token_hex(32))"
```

Start the server:

```powershell
uvicorn server:app --reload
```

Open http://127.0.0.1:8000

> The entry point is `server:app`, not `app:app`.

---

## Demo steps

### 1. Sign in
- Open http://127.0.0.1:8000 — you are redirected to `/login`
- Enter the password set in `LEGITMATE_PASSWORD`
- On success you are redirected to the workspace

### 2. Upload documents
- Click **Upload & documents** in the top navigation
- Click "Choose file" and upload `data/sample/sample_sale_deed.pdf`; open its language, topic, and candidate-finding report. The uploader accepts the common readable formats listed in the Documents page, up to 200 MB per file.
- Wait for "Document indexed" confirmation, then upload:
  - `data/sample/sample_dispute_letter.pdf`
  - `data/sample/sample_lease_agreement.pdf`
- All three appear in the Indexed documents list with page counts; DOCX documents show text-section counts

### 3. Search and cited answer (keyword mode)
- Click **Ask a question** in the top navigation
- Confirm the search mode shows **Keywords — recommended demo**
- Ask: "What is the sale price in the agreement?"
- The answer quotes the passage from page 1 of the sale deed
- Citation chips show document name and page number
- The retrieved passage is shown beneath the answer

### 4. Conflict detection
- Click **Differences** in the top navigation (the badge shows the count of open conflicts)
- Two conflicts appear:
  - Different dates: 15 March 2023 (`sample_sale_deed.pdf`) vs 20 March 2023 (`sample_dispute_letter.pdf`)
  - Different amounts: Rs. 500000 vs Rs. 550000
- Both source passages are shown side by side for each conflict
- The footer states: "LEGITMATE does not determine which statement is legally correct"
- Return to **Ask a question** and ask: "What date was the agreement signed?" (keyword mode)
  — the answer includes a conflict note

### 5. Evidence-linked claims
- Click **Statements** in the top navigation
- Each extracted claim sentence is marked **Supported** or **Unsupported — verify source**
- A claim is Supported only when its exact text is found in an indexed chunk from the same document and page
- Example supported claim: *"The Buyer agrees to pay and the Seller agrees to accept a total sale consideration of Rs. …"* — `sample_sale_deed.pdf`, page 1
- Click "Show matched source passage" to see the exact chunk text
- Claims flagged ⚠ Conflict show both source passages side by side; LEGITMATE does not select a winner

### 6. Timeline
- Click **Dates** in the top navigation
- Dates from all documents appear in chronological order
- Each entry shows the source document and page, and the review status badge

### Case brief — source-linked extractive summary
- Click **Case brief** in the top navigation
- Select all three sample documents, then click **Generate case brief**
- Inspect the chosen sentences and their document/page citations, then expand candidate facts
- Review both cited sides of each open difference and the combined chronology
- Download the readable HTML report or JSON data; the summary is selected from source text locally and is not rewritten by an AI provider

### 7. Review panel
- Click **Review facts** in the top navigation
- Extracted candidates are listed: persons, dates, amounts, claims, properties
- Use the type and status filters to narrow the list
- Mark items as Reviewed, Unresolved, or Rejected using the action buttons
- Items remain **Candidate** until explicitly reviewed; review timestamps are recorded

### 8. Retrieval modes
- Return to **Ask a question**
- Ask: "Who are the parties to the agreement?" in keyword mode
  — the synonym map expands "parties" to seller/buyer/lessor/lessee and returns named parties
- Switch to **Similar meaning — experimental** or **Combine both searches** and ask the same question
- semantic search uses multilingual E5 when available; combined search fuses lexical and semantic rankings
- Optionally select **Combine + cross-encoder reranker** to reorder passages by query/passage relevance. This mode is experimental for this corpus and should not be presented as a proven accuracy improvement.

### 9. Abstention (keyword mode)
- In keyword mode, ask: "What is the zoning classification of the property?"
- Response: "Insufficient evidence in the uploaded documents."
- Note: the last confirmed evaluation found semantic/hybrid modes did not abstain reliably on this small corpus. The reranker mode is newly added and has not yet been evaluated against this case.

### 10. Known limitations to disclose
- **Semantic/hybrid abstention:** `intfloat/multilingual-e5-small` compresses cosine scores into a narrow band on a small corpus; relevant and irrelevant queries score in overlapping ranges. BM25 abstention is reliable; semantic/hybrid is not on this corpus.
- **Candidate extraction:** This environment uses regex fallback for NER because spaCy is not installed; dates, amounts, properties, and claim sentences use patterns. False positives are possible. All items are marked **Candidate** and require human review.
- **No encryption at rest:** Uploaded PDFs and the index are plain files in `data/`. Ensure OS-level drive encryption if storing sensitive documents.
- **Single-user only:** The login is a single shared password. Do not expose the server publicly.

### 11. Sign out
- Click **Sign out** in the top-right header
- You are redirected to the login screen
- The session cookie is cleared

### 12. Delete a document
- Click **Upload & documents** in the top navigation
- Click **Delete** next to a document
- Its chunks, items, and conflicts are removed from the index
