from typing import List, Dict
import numpy as np
import time
import hashlib

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


#  NORMALIZE 
def _normalize(query: str) -> str:
    return " ".join(str(query or "").strip().split())


#  HASH 
def _hash(msg: Dict) -> str:
    base = f"{msg.get('role')}|{str(msg.get('content'))[:200]}"
    return hashlib.sha256(base.encode()).hexdigest()


#  COSINE 
def _cosine(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-10
    return float(np.dot(v1, v2) / denom)


#  VALID 
def _valid(vec):
    return (
        isinstance(vec, list) and
        len(vec) in (
            settings.TEXT_EMBEDDING_DIM,
            settings.VISION_EMBEDDING_DIM
        )
    )


#  WEIGHTS 
def _recency(ts, now):
    try:
        if not ts:
            return 1.0
        age = now - float(ts)
        return 1.0 / (1.0 + age / settings.MEMORY_RECENCY_SCALE)
    except Exception:
        return 1.0


def _role(role: str):
    return getattr(settings, "MEMORY_ROLE_WEIGHTS", {}).get(role, 1.0)


def _importance(msg: Dict):
    try:
        return max(0.0, min(float(msg.get("importance", 1.0)), 1.0))
    except Exception:
        return 0.5


#  DEDUP 
def _dedup(history: List[Dict]) -> List[Dict]:

    seen = set()
    out = []

    for m in history:
        try:
            h = _hash(m)
            if h in seen:
                continue
            seen.add(h)
            out.append(m)
        except Exception:
            continue

    return out


#  MAIN 
def filter_relevant_history(
    query: str,
    history: List[Dict],
    embedder,
    top_k: int = None,
    threshold: float = None
) -> List[Dict]:

    if not history:
        return []

    start = time.time()

    query = _normalize(query)
    top_k = top_k or settings.MEMORY_TOP_K
    threshold = threshold or settings.MEMORY_SIM_THRESHOLD

    history = _dedup(history[-settings.MAX_HISTORY_MESSAGES:])

    try:
        query_vec = embedder.embed_query(query)
        now = time.time()

        scored = []

        for msg in history:

            try:
                text = str(msg.get("content", "")).strip()
                if len(text) < 3:
                    continue

                role = msg.get("role", "user")
                text = text[:settings.MAX_PROMPT_CHARS]

                vec = msg.get("embedding")

                if not _valid(vec):
                    vec = embedder.embed_query(text)

                if not _valid(vec):
                    continue

                sim = _cosine(query_vec, vec)

                # adaptive threshold
                if sim < threshold * 0.9:
                    continue

                score = (
                    0.5 * sim +
                    0.2 * _recency(msg.get("timestamp"), now) +
                    0.2 * _role(role) +
                    0.1 * _importance(msg)
                )

                scored.append((score, msg))

            except Exception:
                continue

        if not scored:
            return []

        scores = np.array([s for s, _ in scored], dtype=float)
        max_s = max(scores.max(), 1e-6)

        scored = [(s / max_s, m) for s, m in scored]
        scored.sort(key=lambda x: x[0], reverse=True)

        result = [m for _, m in scored[:top_k]]

        logger.debug(
            event="memory_filter_success",
            selected=len(result),
            latency=round(time.time() - start, 3)
        )

        return result

    except Exception as e:
        logger.error(event="memory_filter_failed", error=str(e))
        return []