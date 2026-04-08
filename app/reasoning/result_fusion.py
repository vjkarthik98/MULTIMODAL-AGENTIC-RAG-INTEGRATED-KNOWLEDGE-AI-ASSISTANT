"""
result_fusion.py

Combines and ranks results from multiple sub-query retrievals.
"""

from typing import List, Dict
import numpy as np
import logging

# ✅ Logger
logger = logging.getLogger(__name__)


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
            logger.debug("[ResultFusion] Empty results received")
            return []

        try:
            logger.debug(f"[ResultFusion] Starting fusion | input_count={len(results)}")

            # Step 1: Sort by score
            results = sorted(
                results,
                key=lambda x: x.get("rerank_score", x.get("score", 0)),
                reverse=True
            )

            diverse_results = []

            # Step 2: Diversity filtering
            for candidate in results:
                keep = True

                for selected in diverse_results:
                    vec1 = candidate.get("embedding")
                    vec2 = selected.get("embedding")

                    if vec1 is not None and vec2 is not None:
                        sim = self._cosine_similarity(vec1, vec2)

                        if sim > self.similarity_threshold:
                            keep = False
                            break
                    else:
                        if candidate.get("text", "")[:100] == selected.get("text", "")[:100]:
                            keep = False
                            break

                if keep:
                    diverse_results.append(candidate)

                if len(diverse_results) >= self.top_k:
                    break

            logger.debug(
                f"[ResultFusion] Diversity filtering done | count={len(diverse_results)}"
            )

            # Step 3: Score filtering
            filtered = []
            for r in diverse_results:
                score = r.get("rerank_score", r.get("score", 0))

                if score > 0:
                    filtered.append(r)

            logger.debug(
                f"[ResultFusion] Score filtering done | final_count={len(filtered)}"
            )

            return filtered[:self.top_k]

        except Exception as e:
            logger.error(f"[ResultFusion] Failed | error={str(e)}")
            return results[:self.top_k]