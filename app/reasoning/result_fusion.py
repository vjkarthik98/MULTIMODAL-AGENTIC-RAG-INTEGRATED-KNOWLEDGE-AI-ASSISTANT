"""
result_fusion.py

Combines and ranks results from multiple sub-query retrievals.
"""

from typing import List, Dict
import numpy as np


class ResultFusion:

    def __init__(self, top_k: int = 5, similarity_threshold: float = 0.95):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
    
    def _cosine_similarity(self, v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)

        return np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        )

    def fuse(self, results: List[Dict]) -> List[Dict]:
        """
        Rank and filter results.
        """

        if not results:
            return []
        
        # Step 1: Sort by rerank_score if exists
        results = sorted(
            results,
            key=lambda x: x.get("rerank_score", x.get("score", 0)),
            reverse=True
        )

        diverse_results = []

        for candidate in results:
            keep = True

            for selected in diverse_results:
                # Use embeddings if available
                vec1 = candidate.get("embedding")
                vec2 = selected.get("embedding")

                if vec1 is not None and vec2 is not None:
                    sim = self._cosine_similarity(vec1, vec2)

                    if sim > self.similarity_threshold:
                        keep = False
                        break
                else:
                    # fallback -> text comparison
                    if candidate.get("text")[:100] == selected.get("text")[:100]:
                        keep = False
                        break

            if keep:
                diverse_results.append(candidate)

            if len(diverse_results) < 3:
                diverse_results.append(candidate)

            if len(diverse_results) >= self.top_k:
                break

            return diverse_results
        
        # Step 2: Remove weak results
        filtered = []
        for r in results:
            score = r.get("rerank_score", r.get("score", 0))

            # Threshold (tunable)
            if score > 0:
                filtered.append(r)
        # Step 3: Limit top_k
        return filtered[:self.top_k]