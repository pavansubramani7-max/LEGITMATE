from __future__ import annotations
import math
import re
from collections import Counter
from typing import Any
from ..config import ALL_STOP_WORDS, BM25_K1, BM25_B


# Legal synonym map for query expansion.
# Keys are query terms a user might write; values are the terms that actually
# appear in property/civil documents.  Applied only to query tokens — the
# document index is never rewritten.  Each entry is auditable and intentional.
_LEGAL_SYNONYMS: dict[str, frozenset[str]] = {
    "parties":    frozenset({"seller", "buyer", "lessor", "lessee", "vendor", "purchaser"}),
    "party":      frozenset({"seller", "buyer", "lessor", "lessee", "vendor", "purchaser"}),
    "landlord":   frozenset({"lessor"}),
    "tenant":     frozenset({"lessee"}),
    "vendor":     frozenset({"seller"}),
    "purchaser":  frozenset({"buyer"}),
}


def _expand(terms: set[str]) -> set[str]:
    """Return terms union their legal synonyms."""
    expanded = set(terms)
    for t in terms:
        if t in _LEGAL_SYNONYMS:
            expanded |= _LEGAL_SYNONYMS[t]
    return expanded


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[\w'-]+", text.casefold()) if len(t) > 1 and t not in ALL_STOP_WORDS]


def rank(query: str, chunks: list[dict[str, Any]], top_k: int = 5) -> list[dict[str, Any]]:
    qterms = _expand(set(tokenize(query)))
    if not qterms or not chunks:
        return []
    docs = [tokenize(c["text"]) for c in chunks]
    df: Counter[str] = Counter(t for terms in docs for t in set(terms))
    n = len(docs)
    avg_len = sum(map(len, docs)) / max(n, 1)
    scored = []
    for chunk, terms in zip(chunks, docs):
        tf: Counter[str] = Counter(terms)
        score = 0.0
        for term in qterms:
            if tf[term]:
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * (tf[term] * BM25_K1 / (tf[term] + BM25_K1 * (1 - BM25_B + BM25_B * len(terms) / max(avg_len, 1))))
        if score:
            scored.append({**chunk, "score": round(score, 4)})
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
