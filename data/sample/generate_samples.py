"""
Generate synthetic sample PDFs for LEGITMATE evaluation.
Uses only PyMuPDF (fitz), which is already a project dependency.

Run from the project root:
    python data/sample/generate_samples.py
"""
from __future__ import annotations
from pathlib import Path
import fitz  # PyMuPDF

OUT = Path(__file__).parent

DOCS = {
    "sample_sale_deed.pdf": [
        # page 1
        """SALE DEED

This Sale Deed is executed on 15 March 2023 between:

SELLER: Rajesh Kumar Sharma, son of Mohan Lal Sharma,
residing at 42 Green Park Colony, Jaipur, Rajasthan.

BUYER: Priya Anand Verma, daughter of Suresh Anand Verma,
residing at 7 Lotus Avenue, Jaipur, Rajasthan.

PROPERTY: Plot No. 118, Survey No. 44/B, situated at
Sector 9, Jaipur Development Authority Layout, Jaipur.

SALE PRICE: The Buyer agrees to pay and the Seller agrees
to accept a total sale consideration of Rs. 500000
(Rupees Five Lakh Only) for the above-described property.

The Seller warrants that the property is free from all
encumbrances, liens, and third-party claims as of the
date of this deed.
""",
        # page 2
        """TERMS AND CONDITIONS

1. The Buyer shall pay the full sale price on or before
   31 March 2023.

2. The Seller shall hand over vacant possession of the
   property on receipt of full payment.

3. The Seller confirms that all property taxes up to
   14 March 2023 have been paid in full.

4. This agreement is governed by the laws of Rajasthan.

Signed and executed at Jaipur on 15 March 2023.

Rajesh Kumar Sharma (Seller)
Priya Anand Verma (Buyer)

Witnesses:
1. Anil Mehta
2. Sunita Rao
"""
    ],
    "sample_dispute_letter.pdf": [
        # page 1
        """LEGAL NOTICE

Date: 10 April 2023

To: Rajesh Kumar Sharma
    42 Green Park Colony, Jaipur

From: Priya Anand Verma
      7 Lotus Avenue, Jaipur

Subject: Dispute regarding Sale Deed dated 20 March 2023
         for Plot No. 118, Survey No. 44/B

Dear Mr. Sharma,

I write to bring to your attention a discrepancy in the
sale deed executed between us. The deed in my possession
states the execution date as 20 March 2023, whereas your
copy states 15 March 2023.

Furthermore, the agreed sale price as per our verbal
agreement was Rs. 550000 (Rupees Five Lakh Fifty Thousand),
not Rs. 500000 as stated in the written deed.

I request you to clarify these discrepancies within
7 days of receipt of this notice, failing which I shall
be constrained to seek appropriate legal remedy.

Yours sincerely,
Priya Anand Verma
""",
        # page 2
        """ANNEXURE A — PAYMENT RECORD

Payment of Rs. 250000 made on 22 March 2023 via
NEFT transfer to account of Rajesh Kumar Sharma.

Balance of Rs. 250000 (as per written deed) or
Rs. 300000 (as per verbal agreement) remains outstanding.

The Buyer disputes the total sale consideration stated
in the written deed and claims the agreed amount was
Rs. 550000.

This document is prepared for the purpose of legal
notice only and does not constitute an admission of
any fact.
"""
    ],
    "sample_lease_agreement.pdf": [
        # page 1
        """LEASE AGREEMENT

This Lease Agreement is entered into on 1 January 2023
between:

LESSOR: Vikram Singh Rathore, owner of Flat No. 5B,
        Sunrise Apartments, MG Road, Bangalore.

LESSEE: Meena Krishnamurthy, employed at InfoTech Ltd,
        Bangalore.

PROPERTY: Flat No. 5B, Sunrise Apartments, MG Road,
          Bangalore — 560001.

LEASE TERM: 12 months commencing 1 January 2023 and
            ending 31 December 2023.

MONTHLY RENT: Rs. 25000 per month payable on or before
              the 5th of each month.

SECURITY DEPOSIT: Rs. 75000 paid by the Lessee to the
                  Lessor on 28 December 2022.
""",
        # page 2
        """COVENANTS

The Lessee shall:
1. Use the premises only for residential purposes.
2. Not sublet or assign the premises without written
   consent of the Lessor.
3. Maintain the premises in good condition.
4. Pay all utility bills during the lease term.

The Lessor warrants that the property is fit for
residential occupation and that Vikram Singh Rathore
holds clear title to the property.

This agreement is executed at Bangalore on 1 January 2023.

Vikram Singh Rathore (Lessor)
Meena Krishnamurthy (Lessee)
"""
    ],
}


def make_pdf(text_pages: list[str]) -> bytes:
    doc = fitz.open()
    for text in text_pages:
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_textbox(
            fitz.Rect(60, 60, 535, 780),
            text,
            fontsize=11,
            fontname="helv",
            color=(0, 0, 0),
        )
    buf = doc.tobytes()
    doc.close()
    return buf


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, pages in DOCS.items():
        path = OUT / filename
        path.write_bytes(make_pdf(pages))
        print(f"Created {path}")
    print("\nSample PDFs ready. Upload them via the LEGITMATE interface to test conflict detection.")


if __name__ == "__main__":
    main()
