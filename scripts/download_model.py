import os
import time

from huggingface_hub import hf_hub_download

from app.utils.logger import get_logger

logger = get_logger(__name__)


MODEL_REPO = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILE = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
LOCAL_DIR = "models/mistral"

MAX_RETRIES = 2


def download_model():

    os.makedirs(LOCAL_DIR, exist_ok=True)

    target_path = os.path.join(LOCAL_DIR, MODEL_FILE)

    #  SKIP IF EXISTS 
    if os.path.exists(target_path):
        logger.info(event="model_exists", path=target_path)
        return target_path

    start = time.time()

    for attempt in range(MAX_RETRIES + 1):

        try:
            logger.info(event="model_download_start", attempt=attempt)

            path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILE,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,
                resume_download=True
            )

            if not os.path.exists(path):
                raise RuntimeError("DOWNLOAD_FAILED")

            logger.info(
                event="model_download_success",
                latency=round(time.time() - start, 2),
                path=path
            )

            return path

        except Exception as e:
            logger.warning(
                event="model_download_retry",
                attempt=attempt,
                error=str(e)
            )

            if attempt >= MAX_RETRIES:
                logger.error(event="model_download_failed")
                raise

            time.sleep(1)


if __name__ == "__main__":
    download_model()