from typing import List, Dict
import numpy as np

def cosine_similarity(vec1, vec2):
    """compute cosine similiarity"""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)

def filter_relevant_history(
    query: str,
    history: List[Dict],
    embedder,
    top_k: int = 3,
    threshold: float = 0.5
) -> List[Dict]:
    """
    Select relevant chat history based on similarity.
    
    Args:
        query: current usery query
        history: chat history
        embedder: your TextEmbedde instance
        top_k: max messges to return
        threshold: similariity cutoff
    
    Returns:
        filtered history list
    """

    if not history:
        return []
    
    # Step 1: Embed query
    query_vec = embedder.embed_query(query)

    scored = []

    for msg in history:
        text = msg.get("content", "")

        if not text.strip():
            continue

        # Step 2: Embed message
        msg_vec = embedder.embed_query(text)

        # Step 3: Compute similarity
        score = cosine_similarity(query_vec, msg_vec)

        if score >= threshold:
            scored.append((score, msg))

        # Step 4: Sort by similarity
        scored.sort(key=lambda x:x[0], reverse=True)

        # Step 5: Take top_k
        filtered = [msg for _, msg in scored[:top_k]]

        return filtered