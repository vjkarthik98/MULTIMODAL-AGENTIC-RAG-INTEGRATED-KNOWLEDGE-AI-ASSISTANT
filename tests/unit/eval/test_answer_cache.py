"""Tests for app/eval/answer_cache.py — the per-run cross-sub-suite memo.

Context (measured 2026-08-20): `--suite full` issued 269 /rag/query
round-trips for 164 distinct queries, because the e2e sub-suite re-queries
ALL 105 rows the generation sub-suite already answered. At ~13-15s/row that
is ~26 min of a 180-min CD job cap spent recomputing known answers.

The memo removes the duplication, so what matters here is that it can never
return an answer for a request that would have reached the model
DIFFERENTLY. A miss is always safe (it just costs a query); only a false hit
could distort a gated metric — so every axis of the key is pinned below.
"""

from __future__ import annotations

from app.eval import answer_cache


def setup_function() -> None:
    answer_cache.clear()


RESP = {"answer": "Q3 revenue was $1.2B", "sources": [{"text": "orig"}]}


def _k(**kw):
    base = {"query": "What was Q3 revenue?", "user_id": "tenant-1", "sources": ["a.pdf", "b.pdf"]}
    base.update(kw)
    return answer_cache.make_key(
        base["query"],
        base["user_id"],
        sources=base["sources"],
        force_web=base.get("force_web", False),
    )


def test_roundtrip_hit():
    answer_cache.put(_k(), RESP)
    assert answer_cache.get(_k()) == RESP


def test_scope_order_does_not_matter():
    """`sources` expresses a SET of in-scope files; the same scope written in
    a different order is the same request and must hit, or the memo would
    silently never fire."""
    answer_cache.put(_k(sources=["a.pdf", "b.pdf"]), RESP)
    assert answer_cache.get(_k(sources=["b.pdf", "a.pdf"])) is not None


def test_different_scope_misses():
    answer_cache.put(_k(sources=["a.pdf"]), RESP)
    assert answer_cache.get(_k(sources=["b.pdf"])) is None
    assert answer_cache.get(_k(sources=["a.pdf", "b.pdf"])) is None
    assert answer_cache.get(_k(sources=None)) is None


def test_tenant_is_part_of_the_key():
    """Tenant isolation is enforced at every data layer in this codebase; a
    memo keyed without user_id would hand one tenant another's answer."""
    answer_cache.put(_k(user_id="tenant-1"), RESP)
    assert answer_cache.get(_k(user_id="tenant-2")) is None


def test_force_web_is_part_of_the_key():
    """A force_web row exercises the web path and yields a different answer
    from the same query scoped to KB files."""
    answer_cache.put(_k(force_web=False), RESP)
    assert answer_cache.get(_k(force_web=True)) is None


def test_different_query_misses():
    answer_cache.put(_k(query="What was Q3 revenue?"), RESP)
    assert answer_cache.get(_k(query="What was Q4 revenue?")) is None


def test_get_returns_a_copy():
    """Callers annotate the response dict (latency, scores, resolved chunk
    ids). Handing out the stored object would let one sub-suite's mutations
    leak into another's copy."""
    answer_cache.put(_k(), RESP)
    first = answer_cache.get(_k())
    first["answer"] = "MUTATED"
    first["sources"][0]["text"] = "MUTATED"

    second = answer_cache.get(_k())
    assert second["answer"] == "Q3 revenue was $1.2B"
    assert second["sources"][0]["text"] == "orig"


def test_clear_scopes_the_memo_to_one_run():
    """EvalRunner.run() clears on entry so a long-lived process can never
    carry an answer from one run into the next — the only way this could
    corrupt a gate."""
    answer_cache.put(_k(), RESP)
    answer_cache.clear()
    assert answer_cache.get(_k()) is None
    assert answer_cache.stats()["entries"] == 0


def test_stats_track_hits_and_misses():
    answer_cache.put(_k(), RESP)
    answer_cache.get(_k())  # hit
    answer_cache.get(_k(query="x"))  # miss
    s = answer_cache.stats()
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["entries"] == 1


def test_non_dict_values_are_ignored():
    """Defensive: a runner handing back an unexpected shape must not poison
    the memo for the row that follows."""
    answer_cache.put(_k(), "not-a-dict")  # type: ignore[arg-type]
    assert answer_cache.get(_k()) is None
