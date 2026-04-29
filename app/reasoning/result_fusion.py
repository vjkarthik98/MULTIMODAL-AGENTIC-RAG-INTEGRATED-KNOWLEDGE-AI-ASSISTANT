import time
import hashlib
from typing import List, Dict

import numpy as np

from app.core.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__)


class ResultFusion:

    def __init__(self):
        self.top_k = settings.RERANK_TOP_K
        self.similarity_threshold = settings.FUSION_SIMILARITY_THRESHOLD
        self.min_score = getattr(settings, "FUSION_MIN_SCORE", 0.05)

    #  MAIN 
    def fuse(self, results: List[Dict], session_id: str = "default") -> List[Dict]:

        if not results:
            return []

        start = time.time()

        try:
            logger.info("[ResultFusion][START] input=%s", len(results))

            # LIMIT INPUT
            results = results[:settings.FUSION_MAX_INPUT]

            # COPY
            results = [dict(r) for r in results]

            # FILTER LOW QUALITY
            results = self._filter_low_quality(results)

            # NORMALIZE
            results = self._normalize_scores(results)

            # ENRICH
            results = self._enrich_scores(results)

            # SORT
            results.sort(key=lambda x: x.get("final_score", 0), reverse=True)

            # DEDUP
            results = self._deduplicate(results)

            # DIVERSITY
            results = self._diversity_filter(results)

            latency = round(time.time() - start, 2)

            logger.info(
                "[ResultFusion][SUCCESS] output=%s latency=%ss",
                len(results),
                latency
            )

            return results[:self.top_k]

        except Exception as e:
            logger.error("[ResultFusion][FAILED] %s", str(e))
            return results[:self.top_k]

    #  FILTER 
    def _filter_low_quality(self, results: List[Dict]) -> List[Dict]:
        return [
            r for r in results
            if r.get("text") and r.get("score", 0.0) > self.min_score
        ]

    #  NORMALIZE 
    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:

        scores = [r.get("score", 0.0) for r in results]

        if not scores:
            return results

        min_s, max_s = min(scores), max(scores)

        for r in results:
            raw = r.get("score", 0.0)

            if max_s - min_s > 1e-6:
                r["norm_score"] = (raw - min_s) / (max_s - min_s)
            else:
                r["norm_score"] = 0.5

        return results

    #  ENRICH 
    def _enrich_scores(self, results: List[Dict]) -> List[Dict]:

        for r in results:

            score = r.get("norm_score", 0.0)

            metadata = r.get("metadata", {}) or {}
            modality = metadata.get("modality", "text")

            modality_weights = getattr(settings, "FUSION_MODALITY_WEIGHTS", {
                "text": 1.0,
                "image": 0.9,
                "audio": 1.1,
                "video": 1.15,
            })

            modality_boost = modality_weights.get(modality, 1.0)

            # LENGTH QUALITY (NOT JUST LENGTH)
            text = str(r.get("text", ""))
            quality = min(len(text) / settings.FUSION_MAX_TEXT_CHARS, 1.0)

            r["final_score"] = (
                settings.FUSION_SCORE_WEIGHT * score +
                settings.FUSION_QUALITY_WEIGHT * quality +
                settings.FUSION_MODALITY_WEIGHT * modality_boost
            )

        return results

    #  DEDUP 
    def _deduplicate(self, results: List[Dict]) -> List[Dict]:

        seen = set()
        unique = []

        for r in results:
            text = str(r.get("text", "")).strip().lower()
            meta = r.get("metadata", {})

            key = hashlib.md5(
                (
                    text[:200] +
                    str(meta.get("doc_id")) +
                    str(meta.get("chunk_id"))
                ).encode()
            ).hexdigest()

            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique

    #  DIVERSITY 
    def _diversity_filter(self, results: List[Dict]) -> List[Dict]:

        selected = []

        for candidate in results:

            v1 = candidate.get("embedding")

            if not self._valid_embedding(v1):
                selected.append(candidate)
                continue

            similarities = []

            for s in selected:
                v2 = s.get("embedding")

                if not self._valid_embedding(v2):
                    continue

                sim = self._cosine_similarity(v1, v2)
                similarities.append(sim)

            if similarities and max(similarities) > self.similarity_threshold:
                continue

            selected.append(candidate)

            if len(selected) >= self.top_k:
                break

        return selected

    #  UTILS 
    def _valid_embedding(self, emb):
        return (
            isinstance(emb, list) and
            len(emb) in (
                settings.TEXT_EMBEDDING_DIM,
                settings.VISION_EMBEDDING_DIM
            )
        )

    def _cosine_similarity(self, v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)

        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
        return float(np.dot(v1, v2) / denom)