"""Jailbreak detector — two-tier: regex rules + embedding similarity.

Tier 1 (always on): compiled regex patterns from policies.yaml.
Tier 2 (optional): cosine similarity of query embedding vs pre-computed
  jailbreak corpus embeddings. Reuses the SentenceTransformer singleton
  already loaded for BM25/dense retrieval — zero extra memory.
  Falls back to Tier 1 only if embedder unavailable or CUDA OOM.

Upgrade hook: replace _semantic_check() with a call to a small open
classifier (e.g. deepset/deberta-v3-base-injection) when available.
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class JailbreakResult:
    is_jailbreak: bool
    confidence: float           # 0.0 = clean, 1.0 = certain jailbreak
    matched_pattern: Optional[str]
    tier: int                   # 1 = regex, 2 = semantic
    degraded: bool = False      # True when Tier 2 fell back due to OOM


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_regex_patterns: List[re.Pattern] = []
_semantic_threshold: float = 0.82
_semantic_enabled: bool = True
_cuda_oom_fallback: bool = True
_corpus_embeddings: Optional[object] = None  # np.ndarray when loaded
_corpus_texts: List[str] = []
_init_lock = threading.Lock()
_initialized = False


def _load_config() -> None:
    global _semantic_threshold, _semantic_enabled, _cuda_oom_fallback
    try:
        from app.guardrails._policy_loader import get_policy
        cfg = get_policy().get("jailbreak", {})
        _semantic_threshold = float(cfg.get("semantic_similarity_threshold", 0.82))
        _semantic_enabled = bool(cfg.get("semantic_enabled", True))
        _cuda_oom_fallback = bool(cfg.get("cuda_oom_fallback", True))
    except Exception as e:
        logger.warning("jailbreak_config_load_failed", error=str(e))


def _compile_injection_patterns() -> List[re.Pattern]:
    """Compile regex injection patterns from policies.yaml."""
    try:
        from app.guardrails._policy_loader import get_policy
        patterns = get_policy().get("injection", {}).get("regex_patterns", [])
        compiled = []
        for entry in patterns:
            try:
                compiled.append(re.compile(entry["pattern"], re.IGNORECASE | re.DOTALL))
            except re.error as e:
                logger.warning("jailbreak_pattern_compile_failed", pattern=entry.get("pattern", ""), error=str(e))
        return compiled
    except Exception as e:
        logger.warning("jailbreak_patterns_load_failed", error=str(e))
        return []


def _load_corpus_embeddings() -> None:
    """Pre-compute embeddings for known jailbreak corpus (Tier 2)."""
    global _corpus_embeddings, _corpus_texts
    try:
        from app.guardrails._policy_loader import get_policy
        cfg = get_policy().get("jailbreak", {})
        corpus_file = cfg.get("corpus_file", "adversarial/red_team_prompts.jsonl")
        jailbreak_tag = cfg.get("corpus_jailbreak_tag", "jailbreak")

        corpus_path = Path(__file__).parent.parent.parent / "tests" / "guardrails" / corpus_file
        if not corpus_path.exists():
            logger.info("jailbreak_corpus_not_found", path=str(corpus_path))
            return

        texts = []
        with open(corpus_path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line.strip())
                    if jailbreak_tag in obj.get("tags", []):
                        texts.append(obj["prompt"])
                except (json.JSONDecodeError, KeyError):
                    continue

        if not texts:
            return

        # Reuse the project's SentenceTransformer singleton
        from app.embeddings.text_embedder import TextEmbedder
        from app.core.config import settings as _cfg
        embedder = TextEmbedder(
            model_name=_cfg.EMBEDDING_MODEL,
            batch_size=_cfg.EMBEDDING_BATCH_SIZE,
            device=_cfg.EMBEDDER_DEVICE,
        )
        import numpy as np
        embeddings = embedder.embed_texts(texts)
        if embeddings is not None and len(embeddings) > 0:
            _corpus_texts = texts
            _corpus_embeddings = np.array(embeddings)
            logger.info("jailbreak_corpus_loaded", n=len(texts))
    except Exception as e:
        logger.warning("jailbreak_corpus_load_failed", error=str(e))


def initialize() -> None:
    """Initialize patterns and corpus. Called at startup."""
    global _regex_patterns, _initialized
    with _init_lock:
        if _initialized:
            return
        _load_config()
        _regex_patterns = _compile_injection_patterns()
        if _semantic_enabled:
            _load_corpus_embeddings()
        _initialized = True
        logger.info(
            "jailbreak_initialized",
            n_patterns=len(_regex_patterns),
            semantic_enabled=_semantic_enabled,
            corpus_size=len(_corpus_texts),
        )


def _regex_check(query: str) -> Optional[tuple[float, str]]:
    """Tier 1: regex check. Returns (confidence, matched_pattern) or None."""
    if not _regex_patterns:
        initialize()
    lower = query.lower()
    for pattern in _regex_patterns:
        if pattern.search(lower):
            return 0.95, pattern.pattern
    return None


def _semantic_check(query: str) -> Optional[float]:
    """Tier 2: embedding similarity against jailbreak corpus.

    Returns cosine similarity score (0–1) or None if unavailable.
    """
    global _corpus_embeddings
    if _corpus_embeddings is None or len(_corpus_texts) == 0:
        return None

    try:
        import numpy as np
        from app.embeddings.text_embedder import TextEmbedder
        from app.core.config import settings as _cfg

        embedder = TextEmbedder(
            model_name=_cfg.EMBEDDING_MODEL,
            batch_size=_cfg.EMBEDDING_BATCH_SIZE,
            device=_cfg.EMBEDDER_DEVICE,
        )
        q_emb = embedder.embed_texts([query])
        if q_emb is None or len(q_emb) == 0:
            return None

        q_vec = np.array(q_emb[0])
        corpus = _corpus_embeddings

        # Cosine similarity
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
        c_norms = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
        sims = c_norms @ q_norm
        return float(np.max(sims))

    except RuntimeError as e:
        if "out of memory" in str(e).lower() and _cuda_oom_fallback:
            logger.warning("jailbreak_cuda_oom_fallback")
            return None
        raise
    except Exception as e:
        logger.warning("jailbreak_semantic_check_failed", error=str(e))
        return None


def check(query: str) -> JailbreakResult:
    """Check query for jailbreak patterns.

    Returns JailbreakResult with full attribution for audit logging.
    """
    if not _initialized:
        initialize()

    # Tier 1 — always fast
    regex_hit = _regex_check(query)
    if regex_hit:
        conf, pattern = regex_hit
        return JailbreakResult(
            is_jailbreak=True,
            confidence=conf,
            matched_pattern=pattern,
            tier=1,
        )

    # Tier 2 — semantic similarity (optional)
    if _semantic_enabled:
        sim = _semantic_check(query)
        if sim is not None:
            is_jb = sim >= _semantic_threshold
            return JailbreakResult(
                is_jailbreak=is_jb,
                confidence=sim,
                matched_pattern=None,
                tier=2,
                degraded=False,
            )
        else:
            # Tier 2 unavailable (OOM, embedder not loaded, etc.) — degrade
            return JailbreakResult(
                is_jailbreak=False,
                confidence=0.0,
                matched_pattern=None,
                tier=1,
                degraded=True,
            )

    return JailbreakResult(
        is_jailbreak=False,
        confidence=0.0,
        matched_pattern=None,
        tier=1,
    )
