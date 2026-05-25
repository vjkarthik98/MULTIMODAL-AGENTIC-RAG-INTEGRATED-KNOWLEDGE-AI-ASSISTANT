"""Retrieval quality metrics: recall@k, precision@k, MRR, nDCG, context_precision, hit_rate.

All functions are pure — they take lists of IDs and return MetricResult.
They never call the pipeline; the runner calls the pipeline and passes results here.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set

from app.eval.metrics.base import MetricResult


def _relevant_set(relevant_chunk_ids: object) -> Set[str]:
    """Normalise the various gold-set shapes for relevant_chunk_ids."""
    if not relevant_chunk_ids:
        return set()
    if isinstance(relevant_chunk_ids, str):
        if relevant_chunk_ids.startswith("TODO"):
            return set()
        return {relevant_chunk_ids}
    return {str(r) for r in relevant_chunk_ids if r and not str(r).startswith("TODO")}


def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int = 5,
) -> MetricResult:
    """Fraction of relevant documents retrieved in top-k."""
    rel = _relevant_set(relevant_ids)
    if not rel:
        return MetricResult.empty("recall_at_k", "no relevant_chunk_ids (TODO)")
    ret_k = set(retrieved_ids[:k])
    hit = len(rel & ret_k)
    value = hit / len(rel)
    return MetricResult(
        name=f"recall_at_{k}",
        value=value,
        n=len(rel),
        notes=f"hits={hit}/{len(rel)} in top-{k}",
    )


def precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int = 5,
) -> MetricResult:
    """Fraction of top-k retrieved documents that are relevant."""
    rel = _relevant_set(relevant_ids)
    if not rel:
        return MetricResult.empty("precision_at_k", "no relevant_chunk_ids (TODO)")
    ret_k = retrieved_ids[:k]
    if not ret_k:
        return MetricResult(name=f"precision_at_{k}", value=0.0, n=0, notes="no results returned")
    hit = sum(1 for rid in ret_k if rid in rel)
    value = hit / len(ret_k)
    return MetricResult(
        name=f"precision_at_{k}",
        value=value,
        n=len(ret_k),
        notes=f"relevant={hit}/{len(ret_k)} in top-{k}",
    )


def mrr(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -> MetricResult:
    """Mean Reciprocal Rank — 1/(rank of first relevant hit), 0 if none."""
    rel = _relevant_set(relevant_ids)
    if not rel:
        return MetricResult.empty("mrr", "no relevant_chunk_ids (TODO)")
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in rel:
            return MetricResult(
                name="mrr",
                value=1.0 / i,
                n=len(retrieved_ids),
                notes=f"first hit at rank {i}",
            )
    return MetricResult(name="mrr", value=0.0, n=len(retrieved_ids), notes="no hit in retrieved set")


def ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int = 10,
) -> MetricResult:
    """Normalised Discounted Cumulative Gain at k (binary relevance)."""
    rel = _relevant_set(relevant_ids)
    if not rel:
        return MetricResult.empty("ndcg_at_k", "no relevant_chunk_ids (TODO)")

    def _dcg(ids: List[str]) -> float:
        return sum(
            1.0 / math.log2(i + 2)
            for i, rid in enumerate(ids[:k])
            if rid in rel
        )

    dcg = _dcg(retrieved_ids)
    # Ideal: relevant docs in top positions
    ideal_order = [1] * min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(len(ideal_order)))
    value = dcg / idcg if idcg > 0 else 0.0
    return MetricResult(
        name=f"ndcg_at_{k}",
        value=value,
        n=min(len(retrieved_ids), k),
        notes=f"dcg={dcg:.4f} idcg={idcg:.4f}",
    )


def context_precision(
    retrieved_docs: List[Dict],
    relevant_ids: List[str],
) -> MetricResult:
    """Fraction of retrieved chunks that overlap with any relevant chunk ID.

    retrieved_docs: list of dicts with at least a 'chunk_id' or 'metadata.chunk_id' key.
    """
    rel = _relevant_set(relevant_ids)
    if not rel:
        return MetricResult.empty("context_precision", "no relevant_chunk_ids (TODO)")
    if not retrieved_docs:
        return MetricResult(name="context_precision", value=0.0, n=0, notes="no docs returned")

    def _get_id(doc: Dict) -> Optional[str]:
        cid = doc.get("chunk_id") or doc.get("id")
        if not cid:
            meta = doc.get("metadata") or {}
            cid = meta.get("chunk_id") or meta.get("doc_id")
        return str(cid) if cid else None

    relevant_count = sum(1 for d in retrieved_docs if _get_id(d) in rel)
    value = relevant_count / len(retrieved_docs)
    return MetricResult(
        name="context_precision",
        value=value,
        n=len(retrieved_docs),
        notes=f"relevant={relevant_count}/{len(retrieved_docs)}",
    )


def hit_rate(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -> MetricResult:
    """Binary: 1 if any relevant document was retrieved, 0 otherwise."""
    rel = _relevant_set(relevant_ids)
    if not rel:
        return MetricResult.empty("hit_rate", "no relevant_chunk_ids (TODO)")
    hit = int(bool(rel & set(retrieved_ids)))
    return MetricResult(
        name="hit_rate",
        value=float(hit),
        n=1,
        notes="hit" if hit else "miss",
    )


def aggregate_retrieval_metrics(
    results: List[Dict],
    k5: int = 5,
    k10: int = 10,
) -> Dict[str, MetricResult]:
    """Aggregate retrieval metrics over a list of {retrieved_ids, relevant_ids} dicts.

    results: [{"retrieved_ids": [...], "relevant_ids": [...], "query": "..."}]
    Returns a dict of metric_name → MetricResult with n = number of queries.
    """
    recalls_5, recalls_10, mrrs, ndcgs, cps, hits = [], [], [], [], [], []

    for r in results:
        ret_ids = [str(x) for x in (r.get("retrieved_ids") or [])]
        rel_ids = r.get("relevant_ids") or []

        rec5 = recall_at_k(ret_ids, rel_ids, k=k5)
        rec10 = recall_at_k(ret_ids, rel_ids, k=k10)
        m = mrr(ret_ids, rel_ids)
        n = ndcg_at_k(ret_ids, rel_ids, k=k10)
        cp = context_precision(r.get("retrieved_docs") or [], rel_ids)
        h = hit_rate(ret_ids, rel_ids)

        if rec5.n > 0:
            recalls_5.append(rec5.value)
        if rec10.n > 0:
            recalls_10.append(rec10.value)
        if m.n > 0:
            mrrs.append(m.value)
        if n.n > 0:
            ndcgs.append(n.value)
        if cp.n > 0:
            cps.append(cp.value)
        if h.n > 0:
            hits.append(h.value)

    def _mean(lst: List[float], name: str) -> MetricResult:
        if not lst:
            return MetricResult.empty(name, "all queries had TODO relevant_chunk_ids")
        return MetricResult(name=name, value=sum(lst) / len(lst), n=len(lst))

    return {
        f"recall_at_{k5}": _mean(recalls_5, f"recall_at_{k5}"),
        f"recall_at_{k10}": _mean(recalls_10, f"recall_at_{k10}"),
        "mrr": _mean(mrrs, "mrr"),
        f"ndcg_at_{k10}": _mean(ndcgs, f"ndcg_at_{k10}"),
        "context_precision": _mean(cps, "context_precision"),
        "hit_rate": _mean(hits, "hit_rate"),
    }
