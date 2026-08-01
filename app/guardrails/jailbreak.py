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

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class JailbreakResult:
    is_jailbreak: bool
    confidence: float  # 0.0 = clean, 1.0 = certain jailbreak
    matched_pattern: str | None
    tier: int  # 1 = regex, 2 = semantic
    degraded: bool = False  # True when Tier 2 fell back due to OOM


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_regex_patterns: list[re.Pattern] = []
_semantic_threshold: float = 0.82
_semantic_enabled: bool = True
_cuda_oom_fallback: bool = True
_corpus_embeddings: object | None = None  # np.ndarray when loaded
_corpus_texts: list[str] = []
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


def _compile_injection_patterns() -> list[re.Pattern]:
    """Compile regex injection patterns from policies.yaml."""
    try:
        from app.guardrails._policy_loader import get_policy

        patterns = get_policy().get("injection", {}).get("regex_patterns", [])
        compiled = []
        for entry in patterns:
            try:
                compiled.append(re.compile(entry["pattern"], re.IGNORECASE | re.DOTALL))
            except re.error as e:
                logger.warning(
                    "jailbreak_pattern_compile_failed",
                    pattern=entry.get("pattern", ""),
                    error=str(e),
                )
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

        # Lives under app/, not tests/ — this corpus is a live production
        # detection input for the semantic jailbreak check (Tier 2), not
        # just a test fixture, so it must ship in the Docker image (which
        # COPYs app/ but never tests/). Confirmed missing in production via
        # a live Tier-2 eval run: corpus_size=0, detector running on its 46
        # regex patterns only with zero semantic corpus coverage. Directory
        # named "resources", not "data" — .dockerignore excludes any `data/`
        # at any depth (user-data volume mount rule), which would have
        # silently re-broken this the same way.
        corpus_path = Path(__file__).parent / "resources" / corpus_file
        if not corpus_path.exists():
            logger.info("jailbreak_corpus_not_found", path=str(corpus_path))
            return

        texts = []
        with open(corpus_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line.strip())
                    if jailbreak_tag in obj.get("tags", []):
                        texts.append(obj["prompt"])
                except (json.JSONDecodeError, KeyError):
                    continue

        if not texts:
            return

        # Reuse the project's cached TextEmbedder singleton — constructing a
        # fresh TextEmbedder() here reloads the 0.6B SentenceTransformer onto
        # the GPU every call (4-9s), which is the dominant input-guard latency.
        from app.core.model_loader import model_loader

        embedder = model_loader.get_embedder()
        import numpy as np

        embeddings = embedder.embed_texts(texts)
        if embeddings is not None and len(embeddings) > 0:
            _corpus_texts = texts
            # Pre-normalize rows once at load — _semantic_check then only
            # normalizes the query vector instead of the whole corpus per call.
            corpus = np.array(embeddings)
            _corpus_embeddings = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
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


def _regex_check(query: str) -> tuple[float, str] | None:
    """Tier 1: regex check. Returns (confidence, matched_pattern) or None."""
    if not _regex_patterns:
        initialize()
    lower = query.lower()
    for pattern in _regex_patterns:
        if pattern.search(lower):
            return 0.95, pattern.pattern
    return None


def _semantic_check(query: str) -> float | None:
    """Tier 2: embedding similarity against jailbreak corpus.

    Returns cosine similarity score (0–1) or None if unavailable.
    """
    global _corpus_embeddings
    if _corpus_embeddings is None or len(_corpus_texts) == 0:
        return None

    try:
        import numpy as np

        # Cached singleton — see _load_corpus_embeddings(): a fresh TextEmbedder()
        # here would reload the embedding model to GPU on every query.
        from app.core.model_loader import model_loader

        embedder = model_loader.get_embedder()
        q_emb = embedder.embed_texts([query])
        if q_emb is None or len(q_emb) == 0:
            return None

        q_vec = np.array(q_emb[0])
        corpus = _corpus_embeddings

        # Cosine similarity
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
        # corpus rows are pre-normalized at load time
        sims = corpus @ q_norm
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
