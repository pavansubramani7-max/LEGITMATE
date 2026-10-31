# Matter workspaces and case export

## Matter isolation within an account

Matter workspaces organize a single account's documents. They do not replace account-level isolation. Choose a matter in the workspace selector to scope the dashboard, document list, search and question answering, candidate facts, claims, conflicts, timeline, graph, case brief, and evidence matrix. Uploads made while a matter is selected are assigned to it. Existing documents stay unassigned until a user assigns them.

Deleting a matter detaches its documents and retains those files in the account workspace. Deleting a document remains a separate, destructive action. Search results and evidence references are restricted to the selected matter. Cross-matter evidence references are rejected.

## Evidence matrix

The matrix brings each extracted claim together with its document location, matching source passage, selected evidence references, human review judgment, review note, and related conflict IDs. It is a review aid. A text match means that the text is traceable to an indexed passage; it does not prove the claim. Candidate extraction and conflict detection can miss information or produce false positives.

The CSV export includes claim-level review fields. User text is guarded against common spreadsheet formula prefixes. Treat exported material as sensitive and share it only with authorized reviewers.

## Case bundle

The ZIP export includes:

- `case-data.json` with the matter metadata, evidence matrix, extractive brief, timeline, evidence index, and conflict references;
- CSV versions of the evidence matrix, timeline, evidence index, conflicts, and source manifest;
- `README.txt` with the export timestamp and limitations;
- optionally, original source files under `sources/`.

Source files are excluded by default. The optional originals total is limited to 1 GB. The manifest records a SHA-256 digest for each included file. Missing files and files omitted by the cap are recorded in the manifest. The archive is generated inside the current account's processed folder and removed after the response is sent; configure local backup and cleanup policies for crash recovery and data retention.

## Deployment readiness

The in-app checklist reports non-secret configuration and fixed project limitations. A passing local configuration does not imply production readiness. The current app has no application-level encryption at rest, independent legal-domain evaluation, distributed throttling, production database migration/backup operations, or production deployment controls. Do not expose the local development server to a public or shared network.

## Validation still required

No real-document precision, recall, citation accuracy, or unsupported-answer rate is claimed by the matter workspace or export features. Those require an authorized corpus, independent annotation, a held-out split, documented metrics, and reviewers who can adjudicate disagreements. Scanned and multilingual document evaluation should be reported separately. Do not substitute synthetic examples for this validation.
