from __future__ import annotations

import asyncio
import hashlib
import math
import time
import unicodedata
from typing import Any

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from app.guardrails.input_guard import sanitize as _guard_sanitize

    _GUARD_AVAILABLE = True
except Exception:
    _GUARD_AVAILABLE = False

try:
    import pybreaker

    _fusion_breaker = pybreaker.CircuitBreaker(
        fail_max=settings.CIRCUIT_BREAKER_MAX_FAILURES,
        reset_timeout=settings.CIRCUIT_BREAKER_RESET_TIMEOUT,
    )
    _PYBREAKER_AVAILABLE = True
except ImportError:
    _PYBREAKER_AVAILABLE = False

    class _DummyBreaker:
        def __call__(self, fn):
            return fn

    _fusion_breaker = _DummyBreaker()  # type: ignore[assignment]


# BUDGET RATIOS
_SUMMARY_RATIO = 0.30
_HISTORY_RATIO = 0.55
_VECTOR_RATIO = 0.15

# MODALITY RECENCY WEIGHTS
_MODALITY_WEIGHTS: dict[str, float] = {
    "text": 1.0,
    "image": 1.05,
    "audio": 1.1,
    "video": 1.1,
    "table": 1.0,
    "document": 1.0,
}

# ROLE WEIGHTS FOR IMPORTANCE SCORING
_ROLE_WEIGHTS: dict[str, float] = {
    "system": 1.5,
    "assistant": 1.2,
    "user": 1.0,
}


# NORMALIZE TEXT


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or ""))
    return " ".join(text.strip().split())


# TRUNCATE


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    return text[: max(limit, 0)]


# HASH MESSAGE


def _hash_msg(msg: dict[str, Any]) -> str:
    base = f"{msg.get('role', '')}|{str(msg.get('content', ''))[:200]}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


# HASH TEXT PREFIX


def _hash_prefix(text: str, length: int = 200) -> str:
    return hashlib.sha256(text[:length].encode("utf-8")).hexdigest()


# RECENCY SCORE


def _recency_score(ts: Any, now: float) -> float:
    try:
        if ts is None:
            return 1.0
        age = max(now - float(ts), 0.0)
        scale = getattr(settings, "MEMORY_RECENCY_SCALE", 3600)
        return 1.0 / (1.0 + age / scale)
    except Exception:
        return 1.0


# IMPORTANCE SCORE


def _importance_score(msg: dict[str, Any]) -> float:
    try:
        val = float(msg.get("importance", 0.5))
        if math.isnan(val) or math.isinf(val):
            return 0.5
        return max(0.0, min(val, 1.0))
    except Exception:
        return 0.5


# MODALITY WEIGHT


def _modality_weight(msg: dict[str, Any]) -> float:
    return _MODALITY_WEIGHTS.get(str(msg.get("modality", "text")).lower(), 1.0)


# ROLE WEIGHT


def _role_weight(msg: dict[str, Any]) -> float:
    return _ROLE_WEIGHTS.get(str(msg.get("role", "user")).lower(), 1.0)


# COMPOSITE SCORE


def _composite_score(msg: dict[str, Any], now: float) -> float:
    recency = _recency_score(msg.get("timestamp"), now)
    importance = _importance_score(msg)
    modality = _modality_weight(msg)
    role = _role_weight(msg)

    score = 0.40 * recency + 0.30 * importance + 0.20 * role + 0.10 * modality

    if math.isnan(score) or math.isinf(score):
        return 0.0

    return round(score, 5)


# DEDUP MESSAGES


def _dedup_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set = set()
    out: list[dict[str, Any]] = []

    for msg in messages:
        try:
            content = str(msg.get("content", "")).strip()
            if not content or len(content) < 2:
                continue
            h = _hash_msg(msg)
            if h in seen:
                continue
            seen.add(h)
            out.append(msg)
        except Exception:
            continue

    return out


# DEDUP SUMMARY VS HISTORY — REMOVE OVERLAPPING PREFIX


def _dedup_summary_history(summary: str, history: str) -> tuple[str, str]:
    if not summary or not history:
        return summary, history

    if _hash_prefix(summary) == _hash_prefix(history):
        history = history.replace(summary[:200], "").strip()

    return summary, history


# FORMAT SINGLE MESSAGE


def _format_message(msg: dict[str, Any], char_limit: int) -> str:
    try:
        role = str(msg.get("role", "user")).lower()
        if role not in {"user", "assistant", "system"}:
            role = "user"

        content = _normalize(msg.get("content", ""))
        if len(content) < 2:
            return ""

        if _GUARD_AVAILABLE:
            try:
                content = _guard_sanitize(content, surface="memory_context")
            except Exception:
                pass  # pass through on failure

        modality = str(msg.get("modality", "text")).lower()
        ts = msg.get("timestamp")

        label = f"[{role.upper()}]"

        if modality and modality != "text":
            label += f"[{modality.upper()}]"

        if ts:
            try:
                age = time.time() - float(ts)
                if age < 60:
                    label += f"[{int(age)}s ago]"
                elif age < 3600:
                    label += f"[{int(age // 60)}m ago]"
                elif age < 86400:
                    label += f"[{int(age // 3600)}h ago]"
                else:
                    label += f"[{int(age // 86400)}d ago]"
            except Exception:
                pass

        line = f"{label}: {content}"
        return _truncate(line, char_limit)

    except Exception:
        return ""


# FORMAT HISTORY LIST


def _format_history(
    messages: list[dict[str, Any]],
    max_chars: int,
    max_messages: int,
) -> str:
    if not messages:
        return ""

    now = time.time()

    # DEDUP
    messages = _dedup_messages(messages)

    # SCORE AND SORT
    scored = [(_composite_score(m, now), m) for m in messages]
    scored.sort(key=lambda x: x[0], reverse=True)

    # TAKE TOP BY SCORE BUT CAP AT MAX_MESSAGES
    top = [m for _, m in scored[:max_messages]]

    # RESTORE CHRONOLOGICAL ORDER FOR READABILITY
    top_sorted = sorted(
        top,
        key=lambda m: float(m.get("timestamp", 0) or 0),
    )

    per_msg_limit = max(max_chars // max(len(top_sorted), 1), 100)

    parts: list[str] = []
    total = 0

    for msg in top_sorted:
        line = _format_message(msg, per_msg_limit)
        if not line:
            continue
        if total + len(line) > max_chars:
            break
        parts.append(line)
        total += len(line)

    return "\n".join(parts)


# FETCH MONGO SUMMARY — SAFE


def _fetch_mongo_summary(session_id: str, user_id: str | None = None) -> str:
    try:
        from app.core.infra_registry import infra

        mongo = infra.get_mongo()
        if mongo and hasattr(mongo, "get_latest_summary"):

            def _do():
                return mongo.get_latest_summary(session_id, user_id) or ""

            if _PYBREAKER_AVAILABLE:
                return _fusion_breaker(_do)()
            return _do()
    except Exception as exc:
        logger.debug(
            event="memory_fusion_mongo_fetch_failed",
            error=str(exc),
            session_id=session_id,
        )
    return ""


# FORMAT VECTOR MEMORIES


def _format_vector_memories(
    vector_memories: list[dict[str, Any]],
    max_chars: int,
) -> str:
    if not vector_memories:
        return ""

    parts: list[str] = []
    total = 0
    seen: set = set()

    for mem in vector_memories:
        text = _normalize(str(mem.get("text", "") or mem.get("content", "")))
        if not text or len(text) < 5:
            continue

        h = hashlib.sha256(text[:200].encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)

        score = mem.get("score", 0.0)
        modality = (mem.get("metadata", {}) or {}).get("modality", "text")
        label = f"[MEM|{modality.upper()}|score:{round(float(score), 2)}]"
        line = f"{label}: {text}"

        if total + len(line) > max_chars:
            break

        parts.append(line)
        total += len(line)

    return "\n".join(parts)


# MAIN BUILD MEMORY CONTEXT


def build_memory_context(
    summary: str,
    filtered_history: list[dict[str, Any]],
    vector_memories: list[dict[str, Any]] | None = None,
    max_total_chars: int | None = None,
    session_id: str = "default",
    user_id: str | None = None,
) -> str:
    start = time.time()
    max_total_chars = max_total_chars or settings.MEMORY_MAX_CONTEXT_CHARS
    vector_memories = vector_memories or []

    try:
        # NORMALIZE SUMMARY
        summary = _normalize(summary or "")

        # FETCH FROM MONGO IF NOT PROVIDED
        if not summary:
            summary = _normalize(_fetch_mongo_summary(session_id, user_id))

        # FORMAT HISTORY
        history_str = _format_history(
            filtered_history,
            max_chars=int(max_total_chars * _HISTORY_RATIO),
            max_messages=settings.MAX_HISTORY_MESSAGES,
        )

        # FORMAT VECTOR MEMORIES
        vector_str = _format_vector_memories(
            vector_memories,
            max_chars=int(max_total_chars * _VECTOR_RATIO),
        )

        # EARLY EXIT
        if not summary and not history_str and not vector_str:
            logger.debug(
                event="memory_fusion_empty",
                session_id=session_id,
            )
            return ""

        # DEDUP SUMMARY VS HISTORY
        summary, history_str = _dedup_summary_history(summary, history_str)

        # BUDGET ALLOCATION
        summary_budget = int(max_total_chars * _SUMMARY_RATIO)
        history_budget = int(max_total_chars * _HISTORY_RATIO)
        vector_budget = int(max_total_chars * _VECTOR_RATIO)

        # ASSEMBLE PARTS
        parts: list[str] = []

        # HEADER
        parts.append(
            "[MEMORY CONTEXT]\n"
            "Use only if relevant to the current query.\n"
            "Prefer recent interactions over older ones."
        )

        # LONG-TERM SUMMARY
        if summary:
            parts.append("[Long-Term Summary]\n" + _truncate(summary, summary_budget))

        # VECTOR-RETRIEVED MEMORIES
        if vector_str:
            parts.append("[Retrieved Memories]\n" + _truncate(vector_str, vector_budget))

        # RECENT CONVERSATION
        if history_str:
            parts.append("[Recent Conversation]\n" + _truncate(history_str, history_budget))

        # INSTRUCTION FOOTER
        parts.append(
            "[Instructions]\n"
            "- Use memory only if directly relevant\n"
            "- Do not repeat memory verbatim\n"
            "- Prefer recent context over older\n"
            "- Never fabricate memory content"
        )

        result = "\n\n".join(parts).strip()

        # OVERFLOW GUARD
        if len(result) > settings.MAX_PROMPT_CHARS:
            header = parts[0]
            footer = parts[-1]
            fixed_len = len(header) + len(footer) + 20
            allowed = max(settings.MAX_PROMPT_CHARS - fixed_len, 0)

            middle_parts = parts[1:-1]
            middle = "\n\n".join(middle_parts)
            middle = _truncate(middle, allowed)

            result = "\n\n".join([header, middle, footer])

            logger.warning(
                event="memory_context_truncated",
                original_size=len("\n\n".join(parts)),
                truncated_size=len(result),
                session_id=session_id,
            )

        logger.debug(
            event="memory_fusion_success",
            size=len(result),
            has_summary=bool(summary),
            has_history=bool(history_str),
            has_vector=bool(vector_str),
            history_messages=len(filtered_history),
            vector_memories=len(vector_memories),
            latency=round(time.time() - start, 3),
            session_id=session_id,
        )

        return result

    except Exception as exc:
        logger.error(
            event="memory_fusion_failed",
            error=str(exc),
            session_id=session_id,
        )
        return ""


# ASYNC WRAPPER


async def async_build_memory_context(
    summary: str,
    filtered_history: list[dict[str, Any]],
    vector_memories: list[dict[str, Any]] | None = None,
    max_total_chars: int | None = None,
    session_id: str = "default",
    user_id: str | None = None,
) -> str:
    return await asyncio.to_thread(
        build_memory_context,
        summary,
        filtered_history,
        vector_memories,
        max_total_chars,
        session_id,
        user_id,
    )
