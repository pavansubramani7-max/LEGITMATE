"""Build an inspectable, evidence-linked graph from the local LEGITMATE index.

This is a deterministic prototype graph, not an ontology or a legal conclusion
engine. Nodes retain document/page provenance so graph expansion can be audited.
"""
from __future__ import annotations

import re
import hashlib
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Callable

from ..search.lexical import tokenize
from ..extraction.dates import infer_event_type

_ENTITY_KINDS = frozenset({"person", "org", "property"})
_RELATED_KINDS = frozenset({"claim", "date", "amount", "event"})
_QUESTION_WORDS = frozenset({
    "about", "did", "does", "find", "give", "show", "state", "tell",
    "what", "when", "where", "which", "who", "whose", "how", "please",
})
_ENTITY_TITLES = frozenset({"mr", "mrs", "ms", "dr", "prof", "shri", "smt", "sri"})


def _canonical_entity(item: dict[str, Any]) -> str | None:
    """Conservative exact-name key; single-token names are deliberately not merged."""
    if item.get("kind") not in _ENTITY_KINDS:
        return None
    value = unicodedata.normalize("NFKC", str(item.get("value", ""))).casefold()
    words = [word for word in re.findall(r"[\w]+", value, re.UNICODE) if word not in _ENTITY_TITLES]
    if len(words) < 2:
        return None
    return f"{item.get('kind', 'entity')}:{' '.join(words)}"


def _entity_node_id(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    return f"entity:{digest}"


def build_knowledge_graph(
    data: dict[str, Any],
    source_link: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """Return document, extracted-item, evidence-passage, conflict and time nodes."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    documents = data.get("documents", [])
    chunks = data.get("chunks", [])
    items = data.get("items", [])
    conflicts = data.get("conflicts", [])
    doc_by_id = {d.get("document_id"): d for d in documents}
    item_by_id = {i.get("item_id"): i for i in items}
    chunk_by_id = {c.get("chunk_id"): c for c in chunks}
    page_items: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    item_source_ids: dict[str, str] = {}
    passage_nodes: dict[str, dict[str, Any]] = {}
    entity_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in items:
        key = _canonical_entity(item)
        if key and item.get("review_status") != "rejected":
            entity_groups[key].append(item)

    for doc in documents:
        doc_id = str(doc.get("document_id", ""))
        nodes.append({"id": f"doc:{doc_id}", "type": "document", "label": doc.get("filename", "Document")})

    cross_document_entities = 0
    for key, mentions in entity_groups.items():
        documents_for_entity = {str(item.get("document_id", "")) for item in mentions}
        nodes.append({
            "id": _entity_node_id(key), "type": "canonical_entity",
            "label": str(mentions[0].get("value", "")),
            "kind": mentions[0].get("kind", "entity"),
            "mention_count": len(mentions), "document_count": len(documents_for_entity),
            "exact_match_only": True,
        })
        if len(documents_for_entity) > 1:
            cross_document_entities += 1

    for item in items:
        item_id = str(item.get("item_id", ""))
        doc_id = str(item.get("document_id", ""))
        page = item.get("page")
        node = {
            "id": f"item:{item_id}", "type": item.get("kind", "fact"),
            "label": str(item.get("value", "")), "document_id": doc_id,
            "document": item.get("document", doc_by_id.get(doc_id, {}).get("filename", "Document")),
            "page": page, "citation_unit": item.get("citation_unit", "page"),
            "review_status": item.get("review_status", "candidate"),
            "event_date": item.get("event_date"),
            "event_type": item.get("event_type") or (infer_event_type(str(item.get("context", ""))) if item.get("kind") == "date" else None),
            "interval_id": item.get("interval_id"),
            "interval_role": item.get("interval_role"),
            "interval_relation": item.get("interval_relation"),
            "claim_evidence_judgment": item.get("claim_evidence_judgment", "unreviewed") if item.get("kind") == "claim" else None,
        }
        nodes.append(node)
        entity_key = _canonical_entity(item)
        if entity_key and item.get("review_status") != "rejected":
            edges.append({"source": f"item:{item_id}", "target": _entity_node_id(entity_key), "relation": "exact_entity_mention"})
        if item.get("kind") == "claim":
            for reference_id in item.get("claim_evidence_refs", []):
                if reference_id in item_by_id:
                    edges.append({
                        "source": f"item:{item_id}", "target": f"item:{reference_id}",
                        "relation": "human_evidence_reference",
                        "judgment": item.get("claim_evidence_judgment", "unreviewed"),
                    })
        if doc_id in doc_by_id:
            edges.append({"source": f"item:{item_id}", "target": f"doc:{doc_id}", "relation": "found_in"})
        if page is not None:
            page_items[(doc_id, page)].append(item)

        if item.get("review_status") != "rejected":
            linked = source_link(item, chunks, strict_value=item.get("kind") == "claim")
            chunk_id = linked.get("source_chunk_id") if linked.get("source_match") else None
            if chunk_id and chunk_id in chunk_by_id:
                item_source_ids[item_id] = chunk_id
                if chunk_id not in passage_nodes:
                    chunk = chunk_by_id[chunk_id]
                    passage_nodes[chunk_id] = {
                        "id": f"passage:{chunk_id}", "type": "evidence",
                        "label": re.sub(r"\s+", " ", str(chunk.get("text", ""))).strip()[:240],
                        "document_id": doc_id, "document": node["document"], "page": page,
                        "citation_unit": chunk.get("citation_unit", item.get("citation_unit", "page")),
                    }
                edges.append({"source": f"item:{item_id}", "target": f"passage:{chunk_id}", "relation": "supported_by"})

    nodes.extend(passage_nodes.values())

    # Co-location edges express only that items were extracted from the same
    # document page; they do not claim a semantic or legal relationship.
    for (doc_id, page), page_group in page_items.items():
        entities = [i for i in page_group if i.get("kind") in _ENTITY_KINDS]
        related = [i for i in page_group if i.get("kind") in _RELATED_KINDS]
        for entity in entities:
            for fact in related:
                edges.append({
                    "source": f"item:{entity['item_id']}", "target": f"item:{fact['item_id']}",
                    "relation": "co_occurs_on_page", "document_id": doc_id, "page": page,
                })

    for conflict in conflicts:
        conflict_id = str(conflict.get("conflict_id", ""))
        nodes.append({
            "id": f"conflict:{conflict_id}", "type": "conflict",
            "label": conflict.get("label", "Possible conflict"), "status": conflict.get("status", "open"),
        })
        for item_id in conflict.get("items", []):
            if item_id in item_by_id:
                edges.append({"source": f"conflict:{conflict_id}", "target": f"item:{item_id}", "relation": "compares"})

    dated_by_document: dict[str, list[dict[str, Any]]] = defaultdict(list)
    interval_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dated = [
        i for i in items if i.get("event_date") and i.get("review_status") != "rejected"
        and i.get("kind") in ("date", "event")
    ]
    for item in dated:
        dated_by_document[str(item.get("document_id", ""))].append(item)
        if item.get("interval_id"):
            interval_groups[str(item["interval_id"])].append(item)
    for interval_id, endpoints in interval_groups.items():
        roles = {item.get("interval_role"): item for item in endpoints}
        start, end = roles.get("start"), roles.get("end")
        if start and end and start.get("document_id") == end.get("document_id"):
            edges.append({"source": f"item:{start['item_id']}", "target": f"item:{end['item_id']}",
                          "relation": "explicit_interval", "interval_id": interval_id,
                          "interval_relation": start.get("interval_relation"),
                          "document_id": start.get("document_id")})
    temporal_sequences = 0
    for document_id, document_dates in dated_by_document.items():
        document_dates.sort(key=lambda i: (str(i.get("event_date")), str(i.get("item_id", ""))))
        for earlier, later in zip(document_dates, document_dates[1:]):
            if earlier.get("event_date") != later.get("event_date"):
                edges.append({"source": f"item:{earlier['item_id']}", "target": f"item:{later['item_id']}",
                              "relation": "precedes_in_document", "document_id": document_id})
                temporal_sequences += 1

    counts = Counter(node["type"] for node in nodes)
    return {
        "nodes": nodes,
        "edges": edges,
        "summary": {"documents": counts["document"], "entities": sum(counts[k] for k in _ENTITY_KINDS),
                    "claims": counts["claim"], "dates_and_events": counts["date"] + counts["event"],
                    "canonical_entities": counts["canonical_entity"],
                    "cross_document_entities": cross_document_entities,
                    "temporal_sequences": temporal_sequences,
                    "explicit_intervals": sum(1 for edge in edges if edge.get("relation") == "explicit_interval"),
                    "evidence_passages": counts["evidence"], "conflicts": counts["conflict"],
                    "node_count": len(nodes), "edge_count": len(edges)},
        "notice": "Repeated entities are linked only by exact normalized text (titles and punctuation ignored); this is not alias resolution. Page co-occurrence is not proof of a legal relationship. Date-order edges are limited to events from the same source document.",
    }


def graph_augmented_rank(
    question: str,
    data: dict[str, Any],
    top_k: int,
    hybrid_rank: Callable[..., list[dict[str, Any]]],
    lexical_rank: Callable[..., list[dict[str, Any]]],
    source_link: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    """Hybrid retrieval plus exact-entity expansion with auditable graph paths."""
    chunks = data.get("chunks", [])
    if not chunks:
        return []
    base = hybrid_rank(question, chunks, max(top_k, min(len(chunks), top_k * 2)))
    candidates: dict[str, dict[str, Any]] = {h["chunk_id"]: {**h, "_graph_rank": 0.0} for h in base if h.get("chunk_id")}

    query_terms = set(tokenize(question)) - _QUESTION_WORDS
    items = data.get("items", [])
    matched_entities: dict[str, dict[str, Any]] = {}
    for item in items:
        if item.get("kind") not in _ENTITY_KINDS or item.get("review_status") == "rejected":
            continue
        entity_terms = set(tokenize(str(item.get("value", ""))))
        overlap = query_terms & entity_terms
        if overlap:
            key = _canonical_entity(item) or f"mention:{item.get('item_id', '')}"
            coverage = len(overlap) / max(len(entity_terms), 1)
            old = matched_entities.get(key)
            if old is None or coverage > old["coverage"]:
                matched_entities[key] = {"item": item, "coverage": coverage}

    if matched_entities:
        chunks_by_page: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for chunk in chunks:
            chunks_by_page[(chunk.get("document_id"), chunk.get("page"))].append(chunk)
        raw_scores = {h["chunk_id"]: h["score"] for h in lexical_rank(question, chunks, len(chunks))}
        for key, match in matched_entities.items():
            seed = match["item"]
            canonical_key = _canonical_entity(seed)
            mentions = [
                item for item in items
                if item.get("kind") in _ENTITY_KINDS and item.get("review_status") != "rejected"
                and ((_canonical_entity(item) == canonical_key) if canonical_key else item.get("item_id") == seed.get("item_id"))
            ]
            related_items: dict[str, dict[str, Any]] = {}
            for mention in mentions:
                related_items[str(mention.get("item_id", ""))] = mention
                for item in items:
                    if (item.get("document_id") == mention.get("document_id")
                            and item.get("page") == mention.get("page")
                            and item.get("kind") in _RELATED_KINDS
                            and item.get("review_status") != "rejected"):
                        related_items[str(item.get("item_id", ""))] = item

            for item in related_items.values():
                source = source_link(item, chunks, strict_value=item.get("kind") == "claim")
                chunk_id = source.get("source_chunk_id") if source.get("source_match") else None
                page_chunks = chunks_by_page.get((item.get("document_id"), item.get("page")), [])
                if chunk_id:
                    page_chunks = [chunk for chunk in page_chunks if chunk.get("chunk_id") == chunk_id]
                cross_document = item.get("document_id") != seed.get("document_id")
                relation = "exact_entity_across_documents" if cross_document else "same_document_page"
                for chunk in page_chunks:
                    cid = chunk.get("chunk_id")
                    if not cid:
                        continue
                    path = {
                        "entity": str(seed.get("value", "")), "relation": relation,
                        "item_kind": item.get("kind"), "seed_document": seed.get("document"),
                        "matched_document": item.get("document"), "exact_entity_key": bool(canonical_key),
                        "query_overlap": round(float(match["coverage"]), 4),
                    }
                    raw_score = float(raw_scores.get(cid, 0.0))
                    graph_rank = min(0.12, float(match["coverage"]) * 0.12)
                    old = candidates.get(cid)
                    if old:
                        old["graph_path"] = old.get("graph_path") or path
                        old["retrieval"] = "hybrid-graph"
                        old["_graph_rank"] = max(float(old.get("_graph_rank", 0.0)), graph_rank)
                        continue
                    candidates[cid] = {
                        **chunk, "score": round(raw_score, 6),
                        # Keep relevance separate from the graph ranking bonus: a
                        # graph path must never manufacture evidence confidence.
                        "best_underlying_score": round(raw_score, 6),
                        "retrieval": "hybrid-graph", "graph_path": path,
                        "_graph_rank": graph_rank,
                    }

    ranked = sorted(candidates.values(), key=lambda h: (
        bool(h.get("graph_path")), h.get("_graph_rank", 0.0),
        h.get("best_underlying_score", h.get("score", 0.0)), h.get("score", 0.0),
    ), reverse=True)
    for hit in ranked:
        hit.pop("_graph_rank", None)
    return ranked[:top_k]
