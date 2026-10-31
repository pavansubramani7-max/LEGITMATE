"""Rebuild only passage-vector caches for the encoder selected in .env.

This never edits data/index.json or uploaded source documents. Run only after a
candidate encoder has been trained and validated against held-out evaluation.
"""
from __future__ import annotations

import pathlib

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from app.retrieval.semantic import backend_info, save_embeddings
from app.store import load


def main() -> None:
    info = backend_info()
    if info.get("backend") != "neural" or not info.get("cached"):
        raise SystemExit(f"Selected neural model is not available locally: {info.get('model')}\n{info.get('detail')}")
    data = load()
    documents = data.get("documents", [])
    chunks = data.get("chunks", [])
    count = 0
    for index, document in enumerate(documents, start=1):
        doc_id = document.get("document_id")
        doc_chunks = [c for c in chunks if c.get("document_id") == doc_id]
        if doc_chunks:
            save_embeddings(str(doc_id), doc_chunks)
            count += len(doc_chunks)
        print(f"Processed {index}/{len(documents)} documents")
    print(f"Rebuilt vectors for {count} passages with {info.get('model')}.")
    print("Uploaded documents and data/index.json were not modified.")


if __name__ == "__main__":
    main()
