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


def gpu_semaphore() -> asyncio.Semaphore:
    """Lazily created so it binds to the running event loop, not import
    time. Module-level singleton — shared by every caller in the process."""
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_GPU_JOBS)
    return _semaphore


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
    sem = gpu_semaphore()
    try:
        await asyncio.wait_for(sem.acquire(), timeout=settings.GPU_ADMISSION_TIMEOUT_SEC)
    except asyncio.TimeoutError as exc:
        logger.warning(event="gpu_admission_timeout", operation=operation)
        raise GPUBusyError(
            "The server is busy processing other requests right now. Please try again in a moment."
        ) from exc

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
