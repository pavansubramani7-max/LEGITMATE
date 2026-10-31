"""
LEGITMATE evaluation script.

Usage:
    python -m app.evaluation.evaluate

Requires:
  - Server running:  uvicorn server:app --reload
  - All three sample PDFs uploaded via the UI or API:
      data/sample/sample_sale_deed.pdf
      data/sample/sample_dispute_letter.pdf
      data/sample/sample_lease_agreement.pdf
  (TC-03 BM25 requires the lease doc for its lessor/lessee tokens;
   TC-02 conflict detection requires both sale deed and dispute letter.)

Reports per retrieval mode (bm25 / semantic / hybrid):
  - Retrieval hit: correct passage appears in top-5 results
  - Citation accuracy: cited page matches expected page
  - Conflict surfaced: conflict flag present when expected
  - Abstention: system abstains when no relevant document is uploaded
"""
from __future__ import annotations
import json
import sys
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"

TEST_CASES = [
    {
        "id": "TC-01",
        "question": "What is the sale price in the agreement?",
        "expected_keyword": "500000",
        "expected_page": 1,
        "expect_abstain": False,
        "expect_conflict": False,
        "description": "Sale price retrieval from sample_sale_deed",
    },
    {
        "id": "TC-02",
        "question": "What date was the agreement signed?",
        "expected_keyword": "2023",
        "expected_page": 1,
        "expect_abstain": False,
        "expect_conflict": True,
        "description": "Date conflict between sample_sale_deed and sample_dispute_letter",
    },
    {
        "id": "TC-03",
        "question": "Who are the parties to the agreement?",
        "expected_keyword": "Sharma",
        "expected_page": 1,
        "expect_abstain": False,
        "expect_conflict": False,
        "description": "Party name retrieval",
    },
    {
        "id": "TC-04",
        "question": "What is the zoning classification of the property?",
        "expected_keyword": None,
        "expected_page": None,
        "expect_abstain": True,
        "expect_conflict": False,
        "description": "Abstention: question not answerable from sample docs",
    },
]

RETRIEVAL_MODES = ["bm25", "semantic", "hybrid"]


def _get(path: str) -> dict:
    with urllib.request.urlopen(BASE + path, timeout=15) as resp:
        return json.loads(resp.read())


def _check_claims_endpoint() -> None:
    """TC-05: /api/claims returns claims with document and page citations."""
    print("\n--- TC-05: Evidence-linked claims endpoint ---")
    result = _get("/api/claims")
    claims = result.get("claims", [])
    print(f"  Total claims returned: {len(claims)}")

    if not claims:
        print("  [SKIP] No claims in index — upload sample PDFs first.")
        return

    # Every claim must have document, document_id, page, value, context
    missing_citation = [c for c in claims if not (c.get("document") and c.get("document_id") and c.get("page") and c.get("value"))]
    print(f"  [{'PASS' if not missing_citation else 'FAIL'}] All claims have document+page citation: {len(claims)-len(missing_citation)}/{len(claims)} cited")

    # conflicted flag must be a bool
    bad_flag = [c for c in claims if not isinstance(c.get("conflicted"), bool)]
    print(f"  [{'PASS' if not bad_flag else 'FAIL'}] All claims have conflicted bool flag")

    conflicted = [c for c in claims if c.get("conflicted")]
    print(f"  [INFO] Conflicted claims: {len(conflicted)}/{len(claims)}")
    if conflicted:
        # Each conflicted claim must have at least one conflict_id
        missing_ids = [c for c in conflicted if not c.get("conflict_ids")]
        print(f"  [{'PASS' if not missing_ids else 'FAIL'}] Conflicted claims carry conflict_ids")


def _post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _run_mode(mode: str) -> tuple[int, int]:
    """Run all test cases for one retrieval mode. Returns (passed, total)."""
    passed = 0
    total = len(TEST_CASES)
    print(f"\n--- Mode: {mode.upper()} ---")
    for tc in TEST_CASES:
        print(f"  [{tc['id']}] {tc['description']}")
        result = _post("/api/ask", {"question": tc["question"], "top_k": 5, "retrieval_mode": mode})

        abstained = result.get("abstained", False)
        evidence = result.get("evidence", [])
        conflict_note = result.get("conflict_note", "")

        checks = []

        if tc["expect_abstain"]:
            checks.append(("Abstention", abstained, f"abstained={abstained}"))
        else:
            checks.append(("Not abstained", not abstained, f"abstained={abstained}"))

        if tc["expected_keyword"] and not tc["expect_abstain"]:
            found = any(tc["expected_keyword"] in e.get("text", "") for e in evidence)
            checks.append(("Keyword in evidence", found, f"keyword='{tc['expected_keyword']}'"))

        if tc["expected_page"] and not tc["expect_abstain"]:
            pages = [e.get("page") for e in evidence]
            checks.append(("Page citation", tc["expected_page"] in pages,
                           f"expected page {tc['expected_page']}, got {pages}"))

        if tc["expect_conflict"]:
            checks.append(("Conflict surfaced", bool(conflict_note),
                           f"conflict_note present={bool(conflict_note)}"))

        ok = all(c[1] for c in checks)
        if ok:
            passed += 1
        for label, ok_c, detail in checks:
            print(f"    [{'PASS' if ok_c else 'FAIL'}] {label}: {detail}")

    return passed, total


def run() -> None:
    print("LEGITMATE Evaluation\n" + "=" * 50)
    print("Requires: server running + sample PDFs uploaded.")

    # Check server is up
    try:
        urllib.request.urlopen(BASE + "/api/health", timeout=5)
    except urllib.error.URLError as e:
        print(f"ERROR: Could not reach server — {e}")
        print("Run: uvicorn server:app --reload")
        sys.exit(1)

    grand_passed = 0
    grand_total = 0
    mode_results: list[tuple[str, int, int]] = []

    for mode in RETRIEVAL_MODES:
        try:
            p, t = _run_mode(mode)
        except urllib.error.URLError as e:
            print(f"  ERROR during {mode}: {e}")
            sys.exit(1)
        mode_results.append((mode, p, t))
        grand_passed += p
        grand_total += t

    _check_claims_endpoint()

    print(f"\n{'=' * 50}")
    print("Results by retrieval mode:")
    for mode, p, t in mode_results:
        print(f"  {mode.upper():8s}: {p}/{t}")
    print(f"  {'TOTAL':8s}: {grand_passed}/{grand_total}")
    print()
    print("Notes:")
    print("  TC-02 conflict detection requires sale deed + dispute letter uploaded.")
    print("  TC-03 BM25 requires all three PDFs (lease doc provides lessor/lessee tokens).")
    print("  TC-04 abstention requires NO document containing zoning information.")
    print("  Neural/hybrid abstention uses NEURAL_ABSTAIN_THRESHOLD=0.74.")
    print("  TC-04 fails for semantic/hybrid: irrelevant query scores 0.74-0.78 overlap")
    print("  with relevant query scores 0.75-0.82 on this corpus. No threshold fixes both.")


if __name__ == "__main__":
    run()
