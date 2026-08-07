"""GPU admission control — bounds concurrent heavy GPU-bound work across
ingestion AND query, and converts a CUDA OOM into a clean, catchable error
instead of an opaque crash.

Models are loaded once and shared across all requests (correct — you don't
reload a resident model per request), but each concurrent operation still
needs its own slice of GPU memory for activations/KV cache on top of the
resident weights. Real measurement, 2026-07-30 (g6e.xlarge / L40S, 48GB): a
single full multimodal ingestion used ~42GB, leaving ~4GB headroom — not
enough for a second concurrent heavy job.

This is ONE shared gate for both ingestion and query, not two independent
ones — they compete for the same physical VRAM budget, so separate limits
could still sum past what the box can hold (e.g. an ingest limit of 3 plus a
query limit of 3 could mean 6 concurrent heavy jobs against a budget
measured to barely fit one).

Before this module existed: `app/pipeline/ingestion_pipeline.py` already had
a `MAX_CONCURRENT_GPU_JOBS`-based semaphore (`_gpu_semaphore()`), but it only
guarded `IngestionPipeline.process_file_async()` — which nothing in the live
app actually called. The `/upload` route imported the bare module-level
`process_file()` (the synchronous path) and bypassed it entirely, so the
semaphore existed, was correctly designed, and protected nothing. Fixed by
centralizing it here and routing both `/upload` and the query endpoints
through it.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_semaphore: asyncio.Semaphore | None = None

# Count of heavy GPU operations currently inside gpu_slot(). Tracked
# explicitly rather than read off the semaphore, because asyncio.Semaphore
# exposes no public "how many are held" API — only the private `_value` and
# `locked()`, and `locked()` is False whenever ANY slot is free, which would
# be wrong the moment MAX_CONCURRENT_GPU_JOBS is raised above 1.
_inflight: int = 0


def gpu_semaphore() -> asyncio.Semaphore:
    """Lazily created so it binds to the running event loop, not import
    time. Module-level singleton — shared by every caller in the process."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_GPU_JOBS)
    return _semaphore


def gpu_busy() -> bool:
    """True while any heavy GPU operation holds a slot.

    Read by app/core/model_reaper.py, which must never evict a model that an
    in-flight job is part-way through using. Dropping the loader's reference
    mid-job is memory-SAFE (the job holds its own reference, so refcounting
    keeps the weights alive and nothing crashes), but it is wasteful: the
    model would be reloaded on the next stage and both copies briefly coexist
    — the opposite of what an eviction pass is for. Skipping while busy makes
    that impossible rather than merely unlikely.
    """
    return _inflight > 0


class GPUBusyError(Exception):
    """Raised when a heavy GPU operation could not get a slot in time, or
    when the GPU ran out of memory anyway. Callers convert this to a clean,
    user-facing "try again" response — never let it surface as a raw 500 or
    a silently crashed stream."""


@contextlib.asynccontextmanager
async def gpu_slot(operation: str) -> AsyncIterator[None]:
    """Acquire a bounded slot before starting heavy GPU work.

    Waits up to GPU_ADMISSION_TIMEOUT_SEC for a slot rather than queueing
    forever — a caller stuck behind someone else's video ingestion for
    minutes with no feedback is a worse experience than a prompt "try again
    shortly". Also catches a CUDA OOM that happens anyway (e.g. one request
    alone is just too large) and converts it to the same clean error.
    """
    global _inflight
    sem = gpu_semaphore()
    try:
        await asyncio.wait_for(sem.acquire(), timeout=settings.GPU_ADMISSION_TIMEOUT_SEC)
    except asyncio.TimeoutError as exc:
        logger.warning(event="gpu_admission_timeout", operation=operation)
        raise GPUBusyError(
            "The server is busy processing other requests right now. Please try again in a moment."
        ) from exc

    # Incremented only AFTER the slot is acquired and decremented in the same
    # finally that releases it, so the counter can never outlive the slot.
    _inflight += 1
    try:
        yield
    except Exception as exc:
        if _is_cuda_oom(exc):
            logger.error(event="gpu_oom", operation=operation, error=str(exc))
            _recover()
            raise GPUBusyError(
                "The server ran out of GPU memory processing this request. "
                "Please try again — a smaller file or simpler query may help."
            ) from exc
        raise
    finally:
        _inflight -= 1
        sem.release()


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    msg = str(exc).lower()
    return "cuda" in msg and "out of memory" in msg


def _recover() -> None:
    """Best-effort: release cached (not in-use) CUDA memory so the next
    request has a fair shot. Does not fix an OOM caused by one genuinely
    oversized request, only frees fragmentation left by finished work."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
