import hashlib
import math
import time
from typing import Dict, List, Optional

import numpy as np

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ResultFusion:

    def __init__(self) -> None:
        self.top_k         = settings.RERANK_TOP_K
        self.sim_threshold = settings.FUSION_SIMILARITY_THRESHOLD
        self.min_score     = settings.FUSION_MIN_SCORE

    # HASH

    def _hash(self, text: str, meta: Dict) -> str:
        base = f"{text[:200]}|{meta.get('doc_id')}|{meta.get('chunk_id')}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    # MAIN

    def fuse(self, results: List[Dict], session_id: str = "default") -> List[Dict]:

        if not results:
            return []

        start       = time.time()
        input_count = len(results)

        try:
            results = results[:settings.FUSION_MAX_INPUT]
            results = [dict(r) for r in results]

            results = self._filter(results)
            filtered_count = len(results)

            if not results:
                return []

            results = self._normalize(results)
            results = self._score(results)

            results.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

            results = self._dedup(results)
            results = self._diversity(results)

            output = results[:self.top_k]

            logger.info(
                event="fusion_success",
                input_count=input_count,
                filtered_count=filtered_count,
                output=len(output),
                modality_breakdown=self._modality_counts(output),
                latency=round(time.time() - start, 2),
                session_id=session_id,
            )

            return output

        except Exception as e:
            logger.error(
                event="fusion_failed",
                error=str(e),
                session_id=session_id,
            )
            return results[:self.top_k] if results else []

    # FILTER

    def _filter(self, results: List[Dict]) -> List[Dict]:
        return [
            r for r in results
            if r.get("text") and r.get("score", 0.0) > self.min_score
        ]

    # NORMALIZE

    def _normalize(self, results: List[Dict]) -> List[Dict]:
        scores = np.array([r.get("score", 0.0) for r in results], dtype=float)

        if scores.size == 0:
            return results

        scores  = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        min_s   = scores.min()
        max_s   = scores.max()

        for i, r in enumerate(results):
            if max_s - min_s > 1e-6:
                r["norm_score"] = float((scores[i] - min_s) / (max_s - min_s))
            else:
                r["norm_score"] = 0.5

        return results

    # SCORE

    def _score(self, results: List[Dict]) -> List[Dict]:
        modality_weights = settings.FUSION_MODALITY_WEIGHTS

        for r in results:
            base     = r.get("norm_score", 0.0)
            meta     = r.get("metadata", {}) or {}
            modality = meta.get("modality", "text")

            modality_boost = modality_weights.get(modality, 1.0)

            text    = str(r.get("text", ""))
            quality = (
                0.1 if len(text) < settings.CHUNK_MIN_SIZE
                else min(len(text) / settings.FUSION_MAX_TEXT_CHARS, 1.0)
            )

            # RECENCY BOOST
            recency = 0.0
            ts = meta.get("timestamp_start") or meta.get("ingestion_time")
            if ts:
                try:
                    age     = max(time.time() - float(ts), 0.0)
                    recency = 1.0 / (1.0 + age / settings.MEMORY_RECENCY_SCALE)
                except Exception:
                    recency = 0.0

            final_score = (
                settings.FUSION_SCORE_WEIGHT    * base +
                settings.FUSION_QUALITY_WEIGHT  * quality +
                settings.FUSION_MODALITY_WEIGHT * modality_boost +
                0.05 * recency
            )

            if math.isnan(final_score) or math.isinf(final_score):
                final_score = 0.0

            r["final_score"] = round(float(final_score), 5)

        return results

    # DEDUP

    def _dedup(self, results: List[Dict]) -> List[Dict]:
        seen:   set       = set()
        unique: List[Dict] = []

        for r in results:
            h = self._hash(r.get("text", ""), r.get("metadata", {}))
            if h in seen:
                continue
            seen.add(h)
            unique.append(r)

        return unique

    # DIVERSITY

    def _diversity(self, results: List[Dict]) -> List[Dict]:
        selected: List[Dict] = []

        for r in results:
            v1 = r.get("embedding")

            if not self._valid_embedding(v1):
                selected.append(r)
                if len(selected) >= self.top_k:
                    break
                continue

            too_similar = any(
                self._cosine(v1, s.get("embedding")) > self.sim_threshold
                for s in selected
                if self._valid_embedding(s.get("embedding"))
            )

            if too_similar:
                continue

            selected.append(r)

            if len(selected) >= self.top_k:
                break

        return selected

    # UTILS

    def _valid_embedding(self, emb) -> bool:
        return (
            isinstance(emb, list) and
            len(emb) in (settings.TEXT_EMBEDDING_DIM, settings.VISION_EMBEDDING_DIM)
        )

    def _cosine(self, v1, v2) -> float:
        a     = np.nan_to_num(np.array(v1, dtype=float))
        b     = np.nan_to_num(np.array(v2, dtype=float))
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    def _modality_counts(self, results: List[Dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in results:
            m = (r.get("metadata", {}) or {}).get("modality", "unknown")
            counts[m] = counts.get(m, 0) + 1
        return counts