from typing import List, Dict
import numpy as np
import time
from app.utils.logger import get_logger

# Logger
logger = get_logger(__name__)


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)

# WEIGHTING FUNCTIONS
def _recency_weight(timestamp, current_time):

    if not timestamp:
        return 1.0
    
    try:
        age = current_time - float(timestamp)
        return 1.0 / (1.0 + age / 300)
    
    except Exception:
        return 1.0
    
def _role_weight(role: str):
    if role == "user":
        return 1.3
    elif role == "assistant":
        return 1.0
    elif role == "system":
        return 1.2
    return 1.0

def _importance_weight(msg: Dict):
    return msg.get("importance", 1.0)

# MAIN FILTER
def filter_relevant_history(
    query: str,
    history: List[Dict],
    embedder,
    top_k: int = 5,
    threshold: float = 0.35
) -> List[Dict]:

    if not history:
        logger.debug("[MemoryFilter] Empty history received")
        return []

    try:
        start_time = time.time()

        logger.debug("[MemoryFilter][START] Filtering")

        # Step 1: Embed query
        query_vec = embedder.embed_query(query)

        current_time = time.time()
    

        scored = []

        # Step 2: Score each message

        for msg in history:
            text = msg.get("content", "").strip()
            role = msg.get("role", "unknown")

            if not text:
                continue
            
            # Reuse embedding if available
            if "embedding" in msg and msg["embedding"] is not None:
                msg_vec = msg["embedding"]

            else:
                msg_vec = embedder.embed_query(text)
                msg["embedding"] = msg_vec

            sim = cosine_similarity(query_vec, msg_vec)

            if sim < threshold:
                continue
        
            # Mutli-Factor Scoring
            recency = _recency_weight(msg.get("timestamp"), current_time)
            role_w = _role_weight(role)
            importance = _importance_weight(msg)

            final_score = sim * recency * role_w * importance

            scored.append((final_score, msg))


        # Step 3: Sort by similarity
        scored.sort(key=lambda x: x[0], reverse=True)

        # Step 4: Select top_k
        selected = [msg for _, msg in scored[:top_k]]

        latency = time.time() - start_time

        logger.info(
            f"[MemoryFilter][SUCCESS] selected={len(selected)}")

        return selected

    except Exception as e:
        logger.error(f"[MemoryFilter][FAIL] | error={str(e)}")
        return []