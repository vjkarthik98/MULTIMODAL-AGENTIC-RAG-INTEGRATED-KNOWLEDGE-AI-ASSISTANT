from typing import List, Dict
import numpy as np
import time

from app.core.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


#  NORMALIZE QUERY 
def _normalize_query(query: str) -> str:
    return " ".join(query.strip().split())


#  COSINE SIM 
def _cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-10
    return float(np.dot(v1, v2) / denom)


#  RECENCY 
def _recency_weight(timestamp, current_time):
    if not timestamp:
        return 1.0

    try:
        age = current_time - float(timestamp)
        return 1.0 / (1.0 + age / settings.MEMORY_RECENCY_SCALE)
    except Exception:
        return 1.0


#  ROLE 
def _role_weight(role: str):
    weights = getattr(settings, "MEMORY_ROLE_WEIGHTS", {
        "user": 1.3,
        "assistant": 1.0,
        "system": 1.2
    })
    return weights.get(role, 1.0)


#  IMPORTANCE 
def _importance_weight(msg: Dict):
    try:
        return max(0.0, min(float(msg.get("importance", 1.0)), 1.0))
    except Exception:
        return 0.5


#  EMBEDDING VALIDATION 
def _valid_embedding(vec):
    return (
        isinstance(vec, list) and
        len(vec) in (
            settings.TEXT_EMBEDDING_DIM,
            settings.VISION_EMBEDDING_DIM
        )
    )


#  DEDUP 
def _deduplicate(history: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []

    for msg in history:
        key = (
            msg.get("role"),
            str(msg.get("content"))[:200]
        )

        if key not in seen:
            seen.add(key)
            unique.append(msg)

    return unique


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

    query = _normalize_query(query)

    top_k = top_k or settings.MEMORY_TOP_K
    threshold = threshold or settings.MEMORY_SIM_THRESHOLD

    # HARD LIMIT + DEDUP
    history = _deduplicate(history[-settings.MAX_HISTORY_MESSAGES:])

    try:
        logger.debug("[MemoryFilter][START]")

        query_vec = embedder.embed_query(query)

        current_time = time.time()
        scored = []

        for msg in history:

            try:
                text = str(msg.get("content", "")).strip()
                role = msg.get("role", "user")

                if len(text) < 3:
                    continue

                text = text[:settings.MAX_PROMPT_CHARS]

                msg_vec = msg.get("embedding")

                if not _valid_embedding(msg_vec):
                    msg_vec = embedder.embed_query(text)

                if not _valid_embedding(msg_vec):
                    continue

                sim = _cosine_similarity(query_vec, msg_vec)

                if sim < threshold:
                    continue

                recency = _recency_weight(msg.get("timestamp"), current_time)
                role_w = _role_weight(role)
                importance = _importance_weight(msg)

                # STABLE SCORING (WEIGHTED SUM)
                score = (
                    0.5 * sim +
                    0.2 * recency +
                    0.2 * role_w +
                    0.1 * importance
                )

                scored.append((score, msg))

            except Exception:
                continue

        if not scored:
            return []

        # NORMALIZE SCORES
        max_score = max(s for s, _ in scored) or 1.0
        scored = [(s / max_score, m) for s, m in scored]

        scored.sort(key=lambda x: x[0], reverse=True)

        selected = [msg for _, msg in scored[:top_k]]

        latency = round(time.time() - start, 2)

        logger.debug(
            "[MemoryFilter][SUCCESS] selected=%s | latency=%ss",
            len(selected),
            latency
        )

        return selected

    except Exception as e:
        logger.error("[MemoryFilter][FAILED] %s", str(e))
        return []