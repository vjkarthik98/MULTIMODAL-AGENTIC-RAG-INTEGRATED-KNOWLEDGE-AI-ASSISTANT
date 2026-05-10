import os
import shutil
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# MODEL CONFIG

MODEL_REPO             = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILE             = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_SIZE_ESTIMATE_MB = 4_200
MIN_VALID_SIZE_BYTES   = 1024 * 1024

# DERIVE LOCAL DIR FROM SETTINGS
_model_path = Path(settings.LLM_MODEL_PATH)
LOCAL_DIR   = str(_model_path.parent)
TARGET_PATH = str(_model_path)

MAX_RETRIES = 2


# DISK SPACE CHECK

def _check_disk_space(target_dir: str) -> None:
    try:
        _, _, free = shutil.disk_usage(target_dir)
        free_mb    = free / (1024 * 1024)
        required   = MODEL_SIZE_ESTIMATE_MB * 1.1

        if free_mb < required:
            logger.warning(
                event="disk_space_low",
                free_mb=round(free_mb, 1),
                required_mb=round(required, 1),
            )
        else:
            logger.info(event="disk_space_ok", free_mb=round(free_mb, 1))

    except Exception as e:
        logger.warning(event="disk_space_check_failed", error=str(e))


# FILE VALIDATION

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
        event="download_validated",
        path=path,
        size_mb=round(size / (1024 * 1024), 1),
    )


# MAIN

def download_model() -> str:

    os.makedirs(LOCAL_DIR, exist_ok=True)

    # SKIP IF ALREADY EXISTS AND VALID
    if os.path.exists(TARGET_PATH):
        try:
            _validate_download(TARGET_PATH)
            logger.info(
                event="model_already_exists",
                path=TARGET_PATH,
                size_mb=round(os.path.getsize(TARGET_PATH) / (1024 * 1024), 1),
            )
            return TARGET_PATH
        except RuntimeError as e:
            logger.warning(
                event="model_corrupt_redownloading",
                path=TARGET_PATH,
                error=str(e),
            )

    # DISK SPACE PRE-CHECK
    _check_disk_space(LOCAL_DIR)

    # HF TOKEN
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    if hf_token:
        logger.info(event="hf_token_found")
    else:
        logger.info(
            event="hf_token_missing",
            hint="Set HF_TOKEN in .env for gated model access",
        )

    start = time.time()

    for attempt in range(MAX_RETRIES + 1):
        try:
            logger.info(
                event="model_download_start",
                repo=MODEL_REPO,
                file=MODEL_FILE,
                target=TARGET_PATH,
                attempt=attempt,
            )

            path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILE,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,
                resume_download=True,
                token=hf_token,
            )

            _validate_download(path)

            logger.info(
                event="model_download_success",
                path=path,
                size_mb=round(os.path.getsize(path) / (1024 * 1024), 1),
                latency=round(time.time() - start, 2),
            )

            return path

        except Exception as e:
            backoff = 2 ** attempt

            logger.warning(
                event="model_download_retry",
                attempt=attempt,
                max_retries=MAX_RETRIES,
                backoff_sec=backoff,
                error=str(e),
            )

            if attempt >= MAX_RETRIES:
                logger.error(
                    event="model_download_failed",
                    repo=MODEL_REPO,
                    file=MODEL_FILE,
                    error=str(e),
                )
                raise

            time.sleep(backoff)

    raise RuntimeError("MODEL_DOWNLOAD_EXHAUSTED")


if __name__ == "__main__":
    download_model()