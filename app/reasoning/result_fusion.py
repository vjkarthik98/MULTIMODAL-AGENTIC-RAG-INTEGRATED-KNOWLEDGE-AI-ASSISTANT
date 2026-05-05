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
        self.sim_threshold = settings.FUSION_SIMILARITY_THRESHOLD
        self.min_score = getattr(settings, "FUSION_MIN_SCORE", 0.05)

    #  HASH 
    def _hash(self, text: str, meta: Dict) -> str:
        base = f"{text[:200]}|{meta.get('doc_id')}|{meta.get('chunk_id')}"
        return hashlib.sha256(base.encode()).hexdigest()

    #  MAIN 
    def fuse(self, results: List[Dict], session_id: str = "default") -> List[Dict]:

        if not results:
            return []

        start = time.time()

        try:
            results = results[:settings.FUSION_MAX_INPUT]
            results = [dict(r) for r in results]

            results = self._filter(results)
            results = self._normalize(results)
            results = self._score(results)

            results.sort(key=lambda x: x["final_score"], reverse=True)

            results = self._dedup(results)
            results = self._diversity(results)

            logger.info(
                event="fusion_success",
                output=len(results),
                latency=round(time.time() - start, 2)
            )

            return results[:self.top_k]

        except Exception as e:
            logger.error(event="fusion_failed", error=str(e))
            return results[:self.top_k]

    #  FILTER 
    def _filter(self, results: List[Dict]) -> List[Dict]:
        return [
            r for r in results
            if r.get("text") and r.get("score", 0.0) > self.min_score
        ]

    #  NORMALIZE 
    def _normalize(self, results: List[Dict]):

        scores = np.array([r.get("score", 0.0) for r in results], dtype=float)

        if scores.size == 0:
            return results

        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)

        min_s, max_s = scores.min(), scores.max()

        for i, r in enumerate(results):
            if max_s - min_s > 1e-6:
                r["norm_score"] = (scores[i] - min_s) / (max_s - min_s)
            else:
                r["norm_score"] = 0.5

        return results

    #  SCORE 
    def _score(self, results: List[Dict]):

        modality_weights = getattr(settings, "FUSION_MODALITY_WEIGHTS", {
            "text": 1.0,
            "image": 0.9,
            "audio": 1.1,
            "video": 1.15,
        })

        for r in results:

            base = r.get("norm_score", 0.0)

            meta = r.get("metadata", {}) or {}
            modality = meta.get("modality", "text")

            modality_boost = modality_weights.get(modality, 1.0)

            text = str(r.get("text", ""))
            quality = min(len(text) / settings.FUSION_MAX_TEXT_CHARS, 1.0)

            r["final_score"] = (
                settings.FUSION_SCORE_WEIGHT * base +
                settings.FUSION_QUALITY_WEIGHT * quality +
                settings.FUSION_MODALITY_WEIGHT * modality_boost
            )

        return results

    #  DEDUP 
    def _dedup(self, results: List[Dict]):

        seen = set()
        unique = []

        for r in results:
            h = self._hash(r.get("text", ""), r.get("metadata", {}))

            if h in seen:
                continue

            seen.add(h)
            unique.append(r)

        return unique

    #  DIVERSITY 
    def _diversity(self, results: List[Dict]):

        selected = []

        for r in results:

            v1 = r.get("embedding")

            if not self._valid(v1):
                selected.append(r)
                continue

            if any(
                self._cosine(v1, s.get("embedding")) > self.sim_threshold
                for s in selected
                if self._valid(s.get("embedding"))
            ):
                continue

            selected.append(r)

            if len(selected) >= self.top_k:
                break

        return selected

    #  UTILS 
    def _valid(self, emb):
        return (
            isinstance(emb, list) and
            len(emb) in (
                settings.TEXT_EMBEDDING_DIM,
                settings.VISION_EMBEDDING_DIM
            )
        )

    def _cosine(self, v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)

        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
        return float(np.dot(v1, v2) / denom)