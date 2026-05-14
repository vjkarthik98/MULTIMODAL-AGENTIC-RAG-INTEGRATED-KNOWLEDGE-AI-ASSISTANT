import os
import shutil
import time
from pathlib import Path
from typing import Optional

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Gauge, Histogram

from app.core.config import settings

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMETHEUS METRICS
_download_duration = Histogram(
    "model_download_duration_seconds",
    "Model download duration",
    ["model"],
)
_download_errors = Counter(
    "model_download_errors_total",
    "Model download errors by model and error type",
    ["model", "error_type"],
)
_download_size = Gauge(
    "model_download_size_bytes",
    "Downloaded model size in bytes",
    ["model"],
)

# MODEL CONFIG
MODEL_REPO             = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILE             = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_SIZE_ESTIMATE_MB = 4_200
MIN_VALID_SIZE_BYTES   = 1024 * 1024

# DERIVE LOCAL DIR FROM SETTINGS
_model_path = Path(settings.LLM_MODEL_PATH)
LOCAL_DIR   = str(_model_path.parent)
TARGET_PATH = str(_model_path)

MAX_RETRIES = 3


# DISK SPACE GUARD

def _check_disk_space(target_dir: str) -> None:
    try:
        _, _, free = shutil.disk_usage(target_dir)
        free_mb    = free / (1024 * 1024)
        required   = MODEL_SIZE_ESTIMATE_MB * 1.2

        if free_mb < required:
            logger.warning(
                "disk_space_low",
                free_mb=round(free_mb, 1),
                required_mb=round(required, 1),
            )
            raise IOError(
                f"INSUFFICIENT_DISK_SPACE: {free_mb:.1f} MB free, "
                f"minimum required {required:.1f} MB"
            )

        logger.info("disk_space_ok", free_mb=round(free_mb, 1))

    except IOError:
        raise
    except Exception as exc:
        logger.warning("disk_space_check_failed", error=str(exc))


# FILE INTEGRITY VALIDATION

def _validate_download(path: str) -> None:
    if not os.path.exists(path):
        raise RuntimeError(f"DOWNLOAD_MISSING: {path}")

    size = os.path.getsize(path)

    if size < MIN_VALID_SIZE_BYTES:
        raise RuntimeError(
            f"DOWNLOAD_CORRUPT: {size} bytes at {path} "
            f"(minimum {MIN_VALID_SIZE_BYTES} bytes expected)"
        )

    logger.info(
        "download_validated",
        path=path,
        size_mb=round(size / (1024 * 1024), 1),
    )


# SHA-256 CHECKSUM VERIFICATION (OPTIONAL — IF CHECKSUM PROVIDED)

def _verify_checksum(path: str, expected_sha256: Optional[str] = None) -> None:
    if not expected_sha256:
        return

    import hashlib

    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)

    actual = sha256.hexdigest()

    if actual != expected_sha256:
        raise RuntimeError(
            f"CHECKSUM_MISMATCH: expected {expected_sha256}, got {actual}"
        )

    logger.info("checksum_verified", path=path)


# ATOMIC WRITE — DOWNLOAD TO TEMP THEN RENAME

def _atomic_write(
    download_fn,
    temp_path: str,
    final_path: str,
) -> None:
    try:
        download_fn(temp_path)
        os.replace(temp_path, final_path)
    except Exception:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise


# HF TOKEN RESOLUTION

def _resolve_hf_token() -> Optional[str]:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if token:
        logger.info("hf_token_found")
    else:
        logger.info(
            "hf_token_missing",
            hint="Set HF_TOKEN in .env for gated model access",
        )
    return token


# MAIN DOWNLOAD FUNCTION

def download_model(
    repo_id: str = MODEL_REPO,
    filename: str = MODEL_FILE,
    local_dir: str = LOCAL_DIR,
    target_path: str = TARGET_PATH,
    expected_sha256: Optional[str] = None,
    max_retries: int = MAX_RETRIES,
) -> str:

    os.makedirs(local_dir, exist_ok=True)

    with tracer.start_as_current_span("model_download") as span:
        span.set_attribute("model.repo", repo_id)
        span.set_attribute("model.file", filename)
        span.set_attribute("target.path", target_path)

        # SKIP IF ALREADY EXISTS AND VALID
        if os.path.exists(target_path):
            try:
                _validate_download(target_path)
                _verify_checksum(target_path, expected_sha256)

                size_mb = round(os.path.getsize(target_path) / (1024 * 1024), 1)
                _download_size.labels(model=filename).set(
                    os.path.getsize(target_path)
                )

                logger.info(
                    "model_already_exists",
                    path=target_path,
                    size_mb=size_mb,
                )
                span.set_status(Status(StatusCode.OK))
                return target_path

            except RuntimeError as exc:
                logger.warning(
                    "model_corrupt_redownloading",
                    path=target_path,
                    error=str(exc),
                )

        # DISK SPACE PRE-CHECK
        _check_disk_space(local_dir)

        hf_token = _resolve_hf_token()
        start    = time.time()

        for attempt in range(max_retries):
            try:
                logger.info(
                    "model_download_start",
                    repo=repo_id,
                    file=filename,
                    target=target_path,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                )

                from huggingface_hub import hf_hub_download

                temp_path = target_path + f".tmp.{attempt}"

                def _do_download(dest: str) -> None:
                    downloaded = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=local_dir,
                        local_dir_use_symlinks=False,
                        resume_download=True,
                        token=hf_token,
                    )
                    # MOVE DOWNLOADED FILE TO EXPECTED TARGET PATH
                    if downloaded != target_path and os.path.exists(downloaded):
                        shutil.move(downloaded, dest)

                _atomic_write(_do_download, temp_path, target_path)

                _validate_download(target_path)
                _verify_checksum(target_path, expected_sha256)

                size       = os.path.getsize(target_path)
                size_mb    = round(size / (1024 * 1024), 1)
                latency    = round(time.time() - start, 2)
                throughput = round(size / (1024 * 1024) / max(latency, 1e-6), 1)

                _download_duration.labels(model=filename).observe(latency)
                _download_size.labels(model=filename).set(size)

                span.set_attribute("download.size_mb", size_mb)
                span.set_attribute("download.latency", latency)
                span.set_status(Status(StatusCode.OK))

                logger.info(
                    "model_download_success",
                    path=target_path,
                    size_mb=size_mb,
                    latency=latency,
                    throughput_mb_per_sec=throughput,
                    attempt=attempt + 1,
                )

                return target_path

            except Exception as exc:
                backoff    = 2 ** attempt
                error_type = type(exc).__name__

                _download_errors.labels(
                    model=filename,
                    error_type=error_type,
                ).inc()

                logger.warning(
                    "model_download_retry",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    backoff_sec=backoff,
                    error=str(exc),
                    error_type=error_type,
                )

                if attempt >= max_retries - 1:
                    span.set_status(Status(StatusCode.ERROR, str(exc)))
                    span.record_exception(exc)

                    logger.error(
                        "model_download_failed",
                        repo=repo_id,
                        file=filename,
                        error=str(exc),
                        attempts=attempt + 1,
                    )
                    raise

                time.sleep(backoff)

    raise RuntimeError("MODEL_DOWNLOAD_EXHAUSTED")


# ENTRY POINT

if __name__ == "__main__":
    download_model()