from typing import List, Dict
import numpy as np
import hashlib
from app.utils.logger import get_logger

# Logger
logger = get_logger(__name__)


class ResultFusion:

    def __init__(self, top_k: int = 5, similarity_threshold: float = 0.85):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    # Main API
    def fuse(self, results: List[Dict]) -> List[Dict]:

        if not results:
            logger.debug("[ResultFusion] Empty results")
            return []

        try:
            logger.info(f"[ResultFusion] Start | input={len(results)}")

            # STEP 1: Normalize Scores
            results = self._normalize_scores(results)

            # STEP 2: Enrich Scores (Multi-Signal)
            results = self._enrich_scores(results)

            # STEP 3: Sort
            results = sorted(
                results,
                key=lambda x: x["final_score"],
                reverse=True
            )

            # STEP 4: Deduplication
            results = self._deduplicate(results)

            # STEP 5: Diversity Filtering
            results = self._diversity_filter(results)

            logger.info(f"[ResultFusion] Final count={len(results)}")

            return results[:self.top_k]
        
        except Exception as e:
            logger.error(f"[ResultFusion] Failed | {str(e)}")
            return results[:self.top_k]
        
    # Score Normalization
    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
            
        scores = [
            r.get("rerank_score", r.get("score", 0))
            for r in results
        ]

        if not scores:
            return results
            
        min_s, max_s = min(scores), max(scores)

        for r in results:
            raw = r.get("rerank_score", r.get("score", 0))

            if max_s - min_s > 1e-6:
                r["norm_score"] = (raw - min_s) / (max_s - min_s)

            else:
                r["norm_score"] = 0.5

        return results
    
    # Multi - Signal Enrichment
    def _enrich_scores(self, results: List[Dict]) -> List[Dict]:
            
        for r in results:

            score = r.get("norm_score", 0)

            # Text richness (longer = better context)
            text_len = len(r.get("text", ""))
            length_boost = min(text_len / 500, 1.0)

            # Modality boost
            modality = r.get("modality", "text")
            modality_boost = 1.0

            if modality == "image":
                    modality_boost = 0.9
            elif modality == "audio":
                    modality_boost = 1.1
            elif modality == "video":
                    modality_boost = 1.15

            # Final Weighted Score
            r["final_score"] = (
                0.7 * score +
                0.2 * length_boost +
                0.1 * modality_boost
            )

        return results
    
    # Deduplication
    def _deduplicate(self, results: List[Dict]) -> List[Dict]:
            
        seen_hashes = set()
        unique = []

        for r in results:
            text = r.get("text", "").strip().lower()

            if not text:
                continue
                 
            h = hashlib.md5(text[:200].encode()).hexdigest()

            if h not in seen_hashes:
                seen_hashes.add(h)
                unique.append(r)

        return unique
    
    # Semantic Diversity
    def _diversity_filter(self, results: List[Dict]) -> List[Dict]:
            
        selected = []

        for candidate in results:

            keep = True

            for s in selected:

                v1 = candidate.get("embedding")
                v2 = s.get("embedding")

                if v1 is not None and v2 is not None:
                    sim = self._cosine_similarity(v1, v2)

                    if sim > self.similarity_threshold:
                        keep = False
                        break

            if keep:
                selected.append(candidate)
            if len(selected) >= self.top_k:
                break

        return selected
        
    # Cosine Similarity
    def _cosine_similarity(self, v1, v2):

        v1 = np.array(v1)
        v2 = np.array(v2)

        return np.dot(v1, v2) / (
            np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        )
                