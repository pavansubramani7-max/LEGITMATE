# Sample Data

This folder contains synthetic documents for testing LEGITMATE.
All names, addresses, and amounts are fictional.

## Generate sample PDFs

```powershell
python data/sample/generate_samples.py
```

This creates three PDFs:

| File | Contents |
|------|----------|
| `sample_sale_deed.pdf` | Sale deed dated 15 March 2023, price Rs. 500000 |
| `sample_dispute_letter.pdf` | Dispute letter citing date 20 March 2023 and price Rs. 550000 |
| `sample_lease_agreement.pdf` | Lease agreement dated 1 January 2023 |

## Conflict demonstration

Upload `sample_sale_deed.pdf` and `sample_dispute_letter.pdf`.
The Conflicts panel will flag:
- Different dates (15 March vs 20 March 2023)
- Different amounts (Rs. 500000 vs Rs. 550000)

## Example questions and expected passages

| Question | Expected keyword | Expected page |
|----------|-----------------|---------------|
| What is the sale price? | 500000 | 1 (sale deed) |
| What date was the agreement signed? | March 2023 | 1 (both docs, conflict) |
| Who are the parties? | Sharma / Verma | 1 |
| What is the zoning classification? | — | abstain |

## Evaluation

```powershell
python -m app.evaluation.evaluate
```

Run after uploading the sample PDFs.
