# data/

| Folder | Contents |
|--------|----------|
| `documents/` | Bootstrap admin's legacy uploads, stored under generated UUIDs |
| `processed/` | Bootstrap admin's per-document embeddings and benchmark reports |
| `workspaces/<account-id>/` | Each customer account's isolated `documents/`, `processed/`, and `index.json` |
| `accounts.sqlite3` | Local account names, roles, salted password hashes, and active status |
| `sample/` | Synthetic sample PDFs and generation script |
| `index.json` | Local JSON index (documents, chunks, items, conflicts) |

Do not commit real legal documents or personal data to this folder.
The admin's legacy `index.json` is created on first admin upload. Customer indexes
are created inside their own workspace when they first upload a document.
