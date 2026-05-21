from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import pybreaker
    _text_breaker = pybreaker.CircuitBreaker(
        fail_max=settings.CIRCUIT_BREAKER_MAX_FAILURES,
        reset_timeout=settings.CIRCUIT_BREAKER_RESET_TIMEOUT,
    )
    _vision_breaker = pybreaker.CircuitBreaker(
        fail_max=settings.CIRCUIT_BREAKER_MAX_FAILURES,
        reset_timeout=settings.CIRCUIT_BREAKER_RESET_TIMEOUT,
    )
    _bm25_breaker = pybreaker.CircuitBreaker(
        fail_max=settings.CIRCUIT_BREAKER_MAX_FAILURES,
        reset_timeout=settings.CIRCUIT_BREAKER_RESET_TIMEOUT,
    )
    _PYBREAKER_AVAILABLE = True
except ImportError:
    _PYBREAKER_AVAILABLE = False

    class _DummyBreaker:
        def __call__(self, fn):
            return fn

    _text_breaker = _DummyBreaker()   # type: ignore[assignment]
    _vision_breaker = _DummyBreaker() # type: ignore[assignment]
    _bm25_breaker = _DummyBreaker()   # type: ignore[assignment]


# COLLOQUIAL → KEYWORD HEURISTIC EXPANSION
#
# Cheap, deterministic, no model call. Drops the stopwords/filler tokens that
# common questions wrap technical terms in, producing a tighter BM25 form.
# We DO NOT use this for dense retrieval — sentence-transformers already
# handles synonym/paraphrase well. We use it ONLY to widen the BM25 lane.
# Latency overhead per query: < 1 ms.

_QUERY_STOPWORDS: set = {
    "what", "which", "who", "when", "where", "why", "how",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "done",
    "the", "a", "an", "of", "to", "in", "on", "for", "at", "by",
    "with", "from", "this", "that", "these", "those", "it", "its",
    "as", "and", "or", "but", "so", "if", "no", "not", "any",
    "can", "could", "should", "would", "may", "might", "will",
    "me", "my", "you", "your", "we", "our", "they", "their",
    "about", "into", "than", "then", "also", "tell", "give",
    "explain", "describe", "summarize", "summarise", "show",
    "good", "bad", "make", "makes", "made",
}


def _expand_query_heuristic(q: str) -> List[str]:
    """Return up to 2 distinct query forms for BM25: [original, keywords-only].

    Adds nothing for queries that are already mostly keywords (avoids dups
    feeding RRF and inflating scores on already-good queries).
    """
    if not q:
        return []
    original = q.strip()
    tokens   = [t for t in re.findall(r"[A-Za-z0-9'\-]+", original.lower())]
    if not tokens:
        return [original]

    keep = [t for t in tokens if t not in _QUERY_STOPWORDS and len(t) > 1]

    # If the keyword form is identical (or near-identical) to the original,
    # don't bother emitting a second variant.
    if not keep or len(keep) >= len(tokens) - 1:
        return [original]

    keyword_form = " ".join(keep)
    if keyword_form.lower() == original.lower():
        return [original]

    return [original, keyword_form]


# VISION QUERY KEYWORDS
_VISION_KEYWORDS = {
    "image", "photo", "diagram", "visual", "figure",
    "chart", "graph", "screenshot", "picture", "illustration",
    "drawing", "render", "thumbnail", "frame", "scene",
}

# AUDIO QUERY KEYWORDS
_AUDIO_KEYWORDS = {
    "audio", "sound", "speech", "transcript", "recording",
    "voice", "podcast", "spoken", "listen", "hear",
}

# VIDEO QUERY KEYWORDS
_VIDEO_KEYWORDS = {
    "video", "clip", "footage", "movie", "film",
    "watch", "stream", "playback", "scene", "frame",
}

# RRF CONSTANT
_RRF_K: int = settings.HYBRID_RRF_K


class HybridRetriever:

    def __init__(
        self,
        bm25,
        vector_store,
        embedder,
        clip_text_embedder=None,
    ) -> None:
        self.bm25 = bm25
        self.vector_store = vector_store
        self.embedder = embedder
        self.clip_text_embedder = clip_text_embedder

        self.w_bm25 = settings.HYBRID_WEIGHT_BM25
        self.w_vector = settings.HYBRID_WEIGHT_VECTOR
        self.w_vision = settings.HYBRID_WEIGHT_VISION

        self.candidate_multiplier = settings.HYBRID_CANDIDATES_MULTIPLIER
        self.min_score = settings.HYBRID_MIN_SCORE
        self.mmr_enabled = settings.MMR_ENABLED
        self.mmr_lambda = settings.MMR_LAMBDA

        # LRU EMBEDDING CACHE
        self._embed_cache: OrderedDict = OrderedDict()
        self._embed_cache_max: int = settings.LRU_CACHE_MAXSIZE

        # VISION LRU CACHE
        self._vision_cache: OrderedDict = OrderedDict()
        self._vision_cache_max: int = min(settings.LRU_CACHE_MAXSIZE, 256)

    # HASH

    def _hash(self, text: str, meta: Dict) -> str:
        base = f"{text[:200]}|{meta.get('doc_id', '')}|{meta.get('chunk_id', '')}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    # QUERY NORMALIZATION

    def _normalize_query(self, q: str) -> str:
        import unicodedata
        q = unicodedata.normalize("NFC", str(q or ""))
        return " ".join(q.strip().split())[:settings.MAX_PROMPT_CHARS]

    # MODALITY DETECTION

    def _is_vision_query(self, q: str) -> bool:
        tokens = set(q.lower().split())
        return bool(tokens & _VISION_KEYWORDS)

    def _is_audio_query(self, q: str) -> bool:
        tokens = set(q.lower().split())
        return bool(tokens & _AUDIO_KEYWORDS)

    def _is_video_query(self, q: str) -> bool:
        tokens = set(q.lower().split())
        return bool(tokens & _VIDEO_KEYWORDS)

    # SCORE VALID

    def _valid_score(self, score: float) -> bool:
        return isinstance(score, (int, float)) and not math.isnan(score) and not math.isinf(score)

    # LRU EMBEDDING CACHE — TEXT

    def _embed_query_cached(self, q: str, session_id: str = "") -> List[float]:
        cache_key = hashlib.sha256(q.encode("utf-8")).hexdigest()

        if cache_key in self._embed_cache:
            self._embed_cache.move_to_end(cache_key)
            logger.debug(event="embed_cache_hit", session_id=session_id)
            return self._embed_cache[cache_key]

        vec = self.embedder.embed_query(q, session_id=session_id)
        self._embed_cache[cache_key] = vec

        if len(self._embed_cache) > self._embed_cache_max:
            self._embed_cache.popitem(last=False)

        return vec

    # LRU EMBEDDING CACHE — VISION

    def _embed_vision_cached(self, q: str, session_id: str = "") -> List[float]:
        cache_key = "vis_" + hashlib.sha256(q.encode("utf-8")).hexdigest()

        if cache_key in self._vision_cache:
            self._vision_cache.move_to_end(cache_key)
            logger.debug(event="vision_embed_cache_hit", session_id=session_id)
            return self._vision_cache[cache_key]

        from app.core.model_loader import model_loader
        clip = self.clip_text_embedder or model_loader.get_clip_text_embedder()
        vec = clip.embed_single(q, session_id=session_id)

        self._vision_cache[cache_key] = vec
        if len(self._vision_cache) > self._vision_cache_max:
            self._vision_cache.popitem(last=False)

        return vec

    # SCORE NORMALIZATION

    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        if not results:
            return results
        scores = [r.get("score", 0.0) for r in results]
        max_s = max(scores) if scores else 0.0
        if max_s <= 1e-8:
            return results
        for r in results:
            r["score"] = r.get("score", 0.0) / max_s
        return results

    # RRF FUSION — score = sum(1 / (K + rank_i)) per spec

    def _rrf_score(self, rank: int) -> float:
        return 1.0 / (_RRF_K + rank)

    # FUSE INTO COMBINED MAP

    def _fuse(
        self,
        combined: Dict[str, Dict],
        results: List[Dict],
        weight: float,
        source_tag: str,
    ) -> None:
        for rank, r in enumerate(results, start=1):
            text = r.get("text")
            meta = r.get("metadata", {}) or {}

            if not text:
                continue

            rrf = self._rrf_score(rank) * weight
            combined_score = rrf

            if not self._valid_score(combined_score):
                continue

            h = self._hash(text, meta)

            if h not in combined:
                combined[h] = {
                    "text": text,
                    "metadata": meta,
                    "score": combined_score,
                    "sources": {source_tag},
                    "embedding": r.get("embedding"),
                }
            else:
                combined[h]["score"] += combined_score
                combined[h]["sources"].add(source_tag)

    # METADATA FILTER

    def _apply_filters(
        self,
        results: List[Dict],
        filters: Optional[Dict[str, Any]],
        session_id: Optional[str],
    ) -> List[Dict]:
        # NOTE: session isolation is already enforced upstream (bm25.search and
        # vector_store.search both filter by session_id before results reach here).
        # Re-filtering here would silently drop results when metadata["session_id"]
        # is missing or mismatched due to schema defaults. Only apply extra filters.
        if not filters:
            return results

        filtered = []
        for r in results:
            meta = r.get("metadata", {}) or {}

            if filters:
                if filters.get("modality") and meta.get("modality") != filters["modality"]:
                    continue
                if filters.get("language") and meta.get("language") != filters["language"]:
                    continue
                if filters.get("source_type") and meta.get("source_type") != filters["source_type"]:
                    continue
                if filters.get("date_from") and meta.get("ingestion_time", 0) < filters["date_from"]:
                    continue
                if filters.get("date_to") and meta.get("ingestion_time", float("inf")) > filters["date_to"]:
                    continue
                # Scope query to specific source filenames. Match is a
                # substring check (case-insensitive) so callers can pass
                # the original filename without the SHA prefix the staging
                # layer adds (e.g. "edge_tabular.txt" matches
                # "2da390bdd91e4c0586cc861c82432c8d_edge_tabular.txt").
                allowed_sources = filters.get("sources")
                if allowed_sources:
                    src = str(meta.get("source", "")).lower()
                    if not any(s.lower() in src for s in allowed_sources if s):
                        continue
            filtered.append(r)

        return filtered

    # MMR DIVERSITY

    def _mmr(
        self,
        results: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        if not self.mmr_enabled or not results:
            return results[:top_k]

        selected: List[Dict] = []
        candidates = list(results)

        while candidates and len(selected) < top_k:
            best_idx = 0
            best_score = float("-inf")

            for i, candidate in enumerate(candidates):
                relevance = candidate.get("score", 0.0)

                if selected:
                    max_sim = max(
                        self._text_overlap(
                            candidate.get("text", ""),
                            s.get("text", ""),
                        )
                        for s in selected
                    )
                else:
                    max_sim = 0.0

                mmr_score = (
                    self.mmr_lambda * relevance
                    - (1.0 - self.mmr_lambda) * max_sim
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            selected.append(candidates.pop(best_idx))

        return selected

    # TEXT OVERLAP FOR MMR

    def _text_overlap(self, left: str, right: str) -> float:
        a = set(left.lower().split())
        b = set(right.lower().split())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    # VECTOR SEARCH — TEXT SPACE

    def _vector_search_text(
        self,
        q_vec: List[float],
        candidate_k: int,
        session_id: str,
    ) -> List[Dict]:
        def _do():
            return self.vector_store.search_text(q_vec, candidate_k, session_id)

        try:
            if _PYBREAKER_AVAILABLE:
                results = _text_breaker(_do)()
            else:
                results = _do()
            return self._normalize_scores(results or [])
        except Exception as exc:
            logger.warning(
                event="vector_text_search_failed",
                error=str(exc),
                session_id=session_id,
            )
            return []

    # VECTOR SEARCH — VISION SPACE

    def _vector_search_vision(
        self,
        v_vec: List[float],
        candidate_k: int,
        session_id: str,
    ) -> List[Dict]:
        def _do():
            return self.vector_store.search_vision(v_vec, candidate_k, session_id)

        try:
            if _PYBREAKER_AVAILABLE:
                results = _vision_breaker(_do)()
            else:
                results = _do()
            return self._normalize_scores(results or [])
        except Exception as exc:
            logger.warning(
                event="vector_vision_search_failed",
                error=str(exc),
                session_id=session_id,
            )
            return []

    # BM25 SEARCH

    def _bm25_search(
        self,
        query: str,
        candidate_k: int,
        session_id: str,
        filters: Optional[Dict],
    ) -> List[Dict]:
        def _do():
            return self.bm25.search(
                query,
                session_id=session_id,
                top_k=candidate_k,
                filters=filters,
            )

        try:
            if _PYBREAKER_AVAILABLE:
                results = _bm25_breaker(_do)()
            else:
                results = _do()
            return self._normalize_scores(results or [])
        except Exception as exc:
            logger.warning(
                event="bm25_search_failed",
                error=str(exc),
                session_id=session_id,
            )
            return []

    # MAIN SEARCH

    def search(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        if not query or not session_id:
            return []

        start = time.time()
        query = self._normalize_query(query)
        top_k = top_k or settings.DEFAULT_TOP_K
        candidate_k = min(top_k * self.candidate_multiplier, 50)

        is_vision = self._is_vision_query(query)
        is_audio = self._is_audio_query(query)
        is_video = self._is_video_query(query)

        try:
            # TEXT EMBEDDING
            q_vec: Optional[List[float]] = None
            try:
                q_vec = self._embed_query_cached(query, session_id=session_id)
            except Exception as exc:
                logger.warning(
                    event="text_embed_failed",
                    error=str(exc),
                    session_id=session_id,
                )

            # BM25 SEARCH — run for each heuristic query variant and union
            # results so colloquial queries ("how does the body fight germs?")
            # still hit the keyword-heavy chunks. Cheap (<1ms expansion,
            # +1 BM25 lookup at worst). Dense lane uses the original query
            # only — sentence-transformers already paraphrase well.
            bm25_res: List[Dict] = []
            seen_bm25_keys: set  = set()
            for variant in _expand_query_heuristic(query):
                variant_res = self._bm25_search(variant, candidate_k, session_id, filters)
                for r in variant_res:
                    meta = r.get("metadata") or {}
                    key  = self._hash(r.get("text", ""), meta)
                    if key in seen_bm25_keys:
                        continue
                    seen_bm25_keys.add(key)
                    bm25_res.append(r)

            # VECTOR TEXT SEARCH
            vec_res: List[Dict] = []
            if q_vec:
                vec_res = self._vector_search_text(q_vec, candidate_k, session_id)

            # VISION SEARCH — only when vision_collection has indexed points.
            vis_res: List[Dict] = []
            if self.vector_store.has_vision_data():
                try:
                    v_vec = self._embed_vision_cached(query, session_id=session_id)
                    vis_res = self._vector_search_vision(v_vec, candidate_k, session_id)
                except Exception as exc:
                    logger.warning(
                        event="vision_search_skipped",
                        error=str(exc),
                        session_id=session_id,
                    )

            # EARLY EXIT IF ALL EMPTY (text + vision)
            if not bm25_res and not vec_res and not vis_res:
                logger.warning(
                    event="hybrid_retrieval_empty_no_results_from_any_retriever",
                    query_len=len(query),
                    session_id=session_id,
                )
                return []

            # AUDIO/VIDEO MODALITY FILTER BOOST
            modality_filter: Optional[str] = None
            if is_audio:
                modality_filter = "audio"
            elif is_video:
                modality_filter = "video"

            # RRF FUSION — vision always contributes; boosted when query is vision-cued.
            combined: Dict[str, Dict] = {}
            self._fuse(combined, bm25_res, self.w_bm25, "bm25")
            self._fuse(combined, vec_res, self.w_vector, "dense")

            if vis_res:
                vis_weight = self.w_vision * (1.5 if is_vision else 1.0)
                self._fuse(combined, vis_res, vis_weight, "vision")

            if not combined:
                return []

            # SORT BY FUSED SCORE
            fused = sorted(
                combined.values(),
                key=lambda x: x["score"],
                reverse=True,
            )

            # METADATA FILTERING
            fused = self._apply_filters(fused, filters, session_id)

            # MODALITY BOOST FOR AUDIO/VIDEO QUERIES
            if modality_filter:
                for r in fused:
                    if r.get("metadata", {}).get("modality") == modality_filter:
                        r["score"] = min(r["score"] * 1.2, 1.0)
                fused.sort(key=lambda x: x["score"], reverse=True)

            # NOTE: We deliberately do NOT clip to top_k here. The downstream
            # cross-encoder reranker is the layer that should make the final
            # relevance decision, and it can only re-score what we hand it.
            # Previously we clipped to top_k=5 and applied MMR diversity here,
            # which discarded BM25's strong lexical hits (e.g. chunks that
            # share many tokens with their neighbours) before the reranker
            # ever saw them. Multi-hop queries like "Atacama-3 storage capacity"
            # could lose the very chunk that answers the question.
            #
            # We now pass the full fused candidate pool downstream and let the
            # cross-encoder pick the winners. MMR runs ONLY when explicitly
            # requested and only as a final-stage tie-breaker.
            final = fused[:max(top_k * self.candidate_multiplier, top_k)]

            # SCORE THRESHOLD — skip: RRF scores are inherently small (max ~1/61)
            # and already scoped by session filter; dropping by absolute threshold
            # would discard all valid results when only one retriever has results.

            # CLEAN INTERNAL FIELDS
            for r in final:
                r.pop("sources", None)

            latency = round(time.time() - start, 3)

            if latency * 1000 > settings.LATENCY_TARGET_CROSS_MODAL_MS:
                logger.warning(
                    event="hybrid_search_slo_exceeded",
                    latency_ms=round(latency * 1000, 1),
                    target_ms=settings.LATENCY_TARGET_CROSS_MODAL_MS,
                    session_id=session_id,
                )

            logger.info(
                event="hybrid_search_success",
                results=len(final),
                bm25_count=len(bm25_res),
                vector_count=len(vec_res),
                vision_count=len(vis_res),
                is_vision=is_vision,
                is_audio=is_audio,
                is_video=is_video,
                mmr_enabled=self.mmr_enabled,
                latency=latency,
                session_id=session_id,
            )

            return final

        except Exception as exc:
            logger.error(
                event="hybrid_search_failed",
                error=str(exc),
                session_id=session_id,
            )
            return []

    # ASYNC WRAPPER

    async def async_search(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict]:
        return await asyncio.to_thread(self.search, query, session_id, top_k, filters)

    # HEALTH CHECK

    def health_check(self) -> Dict[str, Any]:
        return {
            "bm25_ready": getattr(self.bm25, "bm25", None) is not None,
            "vector_store_ready": self.vector_store is not None,
            "embedder_ready": self.embedder is not None,
            "clip_embedder_ready": self.clip_text_embedder is not None,
            "mmr_enabled": self.mmr_enabled,
            "circuit_breaker": _PYBREAKER_AVAILABLE,
            "embed_cache_size": len(self._embed_cache),
            "vision_cache_size": len(self._vision_cache),
            "weights": {
                "bm25": self.w_bm25,
                "vector": self.w_vector,
                "vision": self.w_vision,
            },
        }



