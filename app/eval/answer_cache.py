"""Per-RUN memo of /rag/query responses, shared between eval sub-suites.

WHY THIS EXISTS — measured duplication (2026-08-20):

    generation rows : 105
    e2e rows        : 164
    overlap         : 105   <- ALL of generation's rows are re-queried by e2e
    e2e-only        :  59

`--suite full` therefore issues 269 full RAG round-trips through Qwen2.5-14B
where 164 distinct queries exist. At the ~13-15s/row measured on a g6e.xlarge
that is ~26 minutes of wall clock spent re-deriving answers the run already
had, inside a job whose cap is 180 minutes (tier2-eval.yml timeout-minutes).

WHAT THIS IS NOT: it is NOT the server-side answer cache. Both runners keep
sending `no_cache: True`, so the model is still exercised live exactly as
before — the rationale in generation_runner ("eval must measure the live
model, never a stale cache") is preserved. This memo lives only in the eval
process, only for the lifetime of ONE `app.eval.run` invocation, and is
seeded exclusively by responses this same run already collected. Nothing can
leak in from a previous run, which is the failure mode that would actually
corrupt a gate.

CORRECTNESS: `/rag/query` composes an answer from session history
(query_pipeline.py's memory_context path), so two calls are only
interchangeable when they would present the model with the same inputs. The
key below therefore pins query + tenant + retrieval scope + web-forcing, and
any difference at all — a different `sources` list, one row forcing web and
the other not — simply misses and re-queries at today's cost. A miss is
always safe; only a false HIT could distort a number, so the key is
deliberately strict rather than clever.

Each gold row additionally gets its own single-turn session id
(`eval_gen_<id>` vs `eval_e2e_<id>`), so neither call has prior turns to
differentiate them.
"""

from __future__ import annotations

import json
from typing import Any

from app.utils.logger import get_logger

logger = get_logger(__name__)

_CACHE: dict[str, dict[str, Any]] = {}
_hits = 0
_misses = 0


def make_key(
    query: str,
    user_id: str,
    sources: Any = None,
    force_web: bool = False,
) -> str:
    """Stable key over everything that can change the answer.

    `sources` is normalised (sorted, de-duplicated) because it expresses a
    SET of in-scope files — the same scope written in a different order must
    hit, while a genuinely different scope must not.
    """
    if sources:
        try:
            scope: Any = sorted({str(s) for s in sources})
        except TypeError:
            scope = str(sources)
    else:
        scope = None
    return json.dumps(
        {"q": query, "u": user_id, "scope": scope, "web": bool(force_web)},
        sort_keys=True,
        ensure_ascii=True,
    )


def get(key: str) -> dict[str, Any] | None:
    global _hits, _misses
    hit = _CACHE.get(key)
    if hit is None:
        _misses += 1
        return None
    _hits += 1
    # Hand back a copy: callers mutate the response dict (adding latency,
    # scores, row bookkeeping), and a shared mutable value would let one
    # sub-suite's annotations bleed into another's.
    return json.loads(json.dumps(hit)) if _is_json_safe(hit) else dict(hit)


def put(key: str, value: dict[str, Any]) -> None:
    if isinstance(value, dict):
        _CACHE[key] = value


def _is_json_safe(v: Any) -> bool:
    try:
        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


def stats() -> dict[str, int]:
    return {"hits": _hits, "misses": _misses, "entries": len(_CACHE)}


def log_stats(where: str) -> None:
    s = stats()
    logger.info(
        event="eval_answer_cache_stats",
        where=where,
        hits=s["hits"],
        misses=s["misses"],
        entries=s["entries"],
    )


def clear() -> None:
    """Reset. Called at the start of a run so a long-lived process (tests,
    a REPL) can never carry answers across two independent runs."""
    global _hits, _misses
    _CACHE.clear()
    _hits = 0
    _misses = 0
