from typing import List, Dict
import numpy as np
import time

from app.core.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


def _cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-10
    return float(np.dot(v1, v2) / denom)


def _recency_weight(timestamp, current_time):
    if not timestamp:
        return 1.0

    try:
        age = current_time - float(timestamp)
        return 1.0 / (1.0 + age / settings.MEMORY_RECENCY_SCALE)
    except Exception:
        return 1.0


def _role_weight(role: str):
    weights = getattr(settings, "MEMORY_ROLE_WEIGHTS", {
        "user": 1.3,
        "assistant": 1.0,
        "system": 1.2
    })
    return weights.get(role, 1.0)


def _importance_weight(msg: Dict):
    return float(msg.get("importance", 1.0))


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

    top_k = top_k or settings.MEMORY_TOP_K
    threshold = threshold or settings.MEMORY_SIM_THRESHOLD

    # Limit history size
    history = history[-settings.MAX_HISTORY_MESSAGES:]

    try:
        logger.debug("[MemoryFilter][START]")

        # Query embedding
        query_vec = embedder.embed_query(query)

        current_time = time.time()
        scored = []

        for msg in history:
            try:
                text = str(msg.get("content", "")).strip()
                role = msg.get("role", "unknown")

                if not text:
                    continue

                # Truncate text 
                if len(text) > settings.MAX_PROMPT_CHARS:
                    text = text[:settings.MAX_PROMPT_CHARS]

                # Use existing embedding safely
                msg_vec = msg.get("embedding")

                if msg_vec is None:
                    msg_vec = embedder.embed_query(text)

                # Validate embedding dimension
                if not isinstance(msg_vec, list):
                    continue

                if len(msg_vec) not in (
                    settings.TEXT_EMBEDDING_DIM,
                    settings.VISION_EMBEDDING_DIM,
                ):
                    continue

                sim = _cosine_similarity(query_vec, msg_vec)

                if sim < threshold:
                    continue

                recency = _recency_weight(msg.get("timestamp"), current_time)
                role_w = _role_weight(role)
                importance = _importance_weight(msg)

                score = sim * recency * role_w * importance

                scored.append((score, msg))

            except Exception:
                continue

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