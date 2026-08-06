from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(override=True)


# HELPERS


def _bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).strip().lower() in ("true", "1", "yes")


def _int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except (ValueError, TypeError):
        return default


def _str(key: str, default: str = "") -> str:
    return os.getenv(key, default) or default


def _opt(key: str) -> str | None:
    val = os.getenv(key, "").strip()
    return val if val else None


def _list(key: str, default: list[str] | None = None) -> list[str]:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default or []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _auto_device() -> str:
    return "cuda" if _cuda_available() else "cpu"


def _auto_whisper_compute() -> str:
    return "int8_float16" if _cuda_available() else "int8"


# SETTINGS CLASS


class Settings:

    # PROJECT ROOT
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

    # MODEL CACHE — permanent .hf_cache (survives instance stop/start; mount as EBS volume on EC2)
    HF_HOME: str = _str("HF_HOME", str(Path(__file__).resolve().parents[2] / ".hf_cache"))
    MODEL_CACHE_REQUIRE_MANIFEST: bool = _bool("MODEL_CACHE_REQUIRE_MANIFEST", False)

    # torch.hub cache (separate mechanism from HF_HOME above; used by Detoxify)
    TORCH_HOME: str = _str("TORCH_HOME", str(Path(__file__).resolve().parents[2] / ".torch_cache"))

    # EasyOCR's own cache (separate mechanism again — a JaidedAI CDN download,
    # not HF Hub or torch.hub). Defaults to EasyOCR's own ~/.EasyOCR unless
    # pointed at the persisted volume here — without this, every
    # easyocr.Reader() call site downloads to the container's ephemeral
    # filesystem, re-fetching from scratch (~2min, confirmed live) on every
    # container recreate instead of once at provisioning time.
    EASYOCR_MODEL_DIR: str = _str(
        "EASYOCR_MODEL_DIR", str(Path(__file__).resolve().parents[2] / ".hf_cache" / "easyocr")
    )

    # CORE APPLICATION
    APP_NAME: str = _str("APP_NAME", "Multimodal Agentic RAG Integrated Knowledge AI Assistant")
    # Keep in sync with /VERSION and the test assertion below by hand — nothing
    # reads /VERSION at runtime, so this drifts silently if only one is bumped.
    APP_VERSION: str = _str("APP_VERSION", "0.30.0")
    APP_DESCRIPTION: str = _str(
        "APP_DESCRIPTION", "Production Multimodal Agentic RAG Integrated AI System"
    )
    ENV: str = _str("ENV", "development")
    DEBUG: bool = _bool("DEBUG", False)

    # SERVER
    HOST: str = _str("HOST", "0.0.0.0")
    PORT: int = _int("PORT", 8000)
    ROOT_PATH: str = _str("ROOT_PATH", "")

    # PATHS
    DATA_DIR: Path = Path(_str("DATA_DIR", str(PROJECT_ROOT / "data")))
    LOG_DIR: Path = Path(_str("LOG_DIR", str(PROJECT_ROOT / "logs")))
    UPLOAD_STAGING_DIR: Path = Path(
        _str("UPLOAD_STAGING_DIR", str(PROJECT_ROOT / "data" / "staging"))
    )
    VIDEO_FRAMES_DIR: Path = Path(
        _str("VIDEO_FRAMES_DIR", str(PROJECT_ROOT / "data" / "temp_frames"))
    )
    PDF_IMAGE_DIR: Path = Path(_str("PDF_IMAGE_DIR", str(PROJECT_ROOT / "data" / "images")))
    TEST_FIXTURES_DIR: Path = Path(
        _str("TEST_FIXTURES_DIR", str(PROJECT_ROOT / "tests" / "fixtures" / "phase24"))
    )
    TEMP_DIR: Path = Path(_str("TEMP_DIR", str(PROJECT_ROOT / "data" / "temp")))
    AUDIT_LOG_PATH: Path = Path(_str("AUDIT_LOG_PATH", str(PROJECT_ROOT / "logs" / "audit.log")))
    CHROOT_BASE: Path = Path(_str("CHROOT_BASE", str(PROJECT_ROOT / "data")))

    # LOGGING
    LOG_LEVEL: str = _str("LOG_LEVEL", "INFO")
    LOG_JSON: bool = _bool("LOG_JSON", False)
    LOG_SHOW_TIMESTAMP: bool = _bool("LOG_SHOW_TIMESTAMP", False)
    ENABLE_FILE_LOGGING: bool = _bool("ENABLE_FILE_LOGGING", True)
    LOG_FILE_NAME: str = _str("LOG_FILE_NAME", "app.log")
    LOG_MAX_BYTES: int = _int("LOG_MAX_BYTES", 10_485_760)
    LOG_BACKUP_COUNT: int = _int("LOG_BACKUP_COUNT", 5)
    CORRELATION_ID_HEADER: str = _str("CORRELATION_ID_HEADER", "X-Correlation-ID")

    # OPENTELEMETRY
    OTEL_ENABLED: bool = _bool("OTEL_ENABLED", False)
    OTEL_EXPORTER_OTLP_ENDPOINT: str = _str("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    OTEL_SERVICE_NAME: str = _str("OTEL_SERVICE_NAME", "MULTIMODAL-AGENTIC-RAG-INTEGRATED-KNOWLEDGE-AI-ASSISTANT")
    OTEL_SAMPLING_RATIO: float = _float("OTEL_SAMPLING_RATIO", 1.0)

    # PROMETHEUS — 9464 is the OpenTelemetry/Prometheus community convention for
    # app-exporter ports; NOT 9090, which is the Prometheus server's own default
    # web port and collides with it the moment both run on the same docker network
    # (see monitoring/prometheus/prometheus.yml, Phase 31).
    PROMETHEUS_ENABLED: bool = _bool("PROMETHEUS_ENABLED", False)
    PROMETHEUS_PORT: int = _int("PROMETHEUS_PORT", 9464)
    PROMETHEUS_METRICS_PATH: str = _str("PROMETHEUS_METRICS_PATH", "/metrics")

    # PERFORMANCE — defaults tuned for AWS g6e.xlarge L40S (48 GB) GPU deployment
    THREAD_POOL_SIZE: int = _int("THREAD_POOL_SIZE", 8)
    ASYNC_SEMAPHORE_WORKERS: int = _int("ASYNC_SEMAPHORE_WORKERS", 10)
    MAX_PARALLEL_REQUESTS: int = _int("MAX_PARALLEL_REQUESTS", 32)
    REQUEST_TIMEOUT_SEC: int = _int("REQUEST_TIMEOUT_SEC", 120)
    FILE_PROCESSING_TIMEOUT_SEC: int = _int("FILE_PROCESSING_TIMEOUT_SEC", 300)
    # Audio/video ingestion (Whisper transcription + pyannote diarization + frame
    # captioning) legitimately runs for many minutes: a 54-min earnings call takes
    # ~6 min, a 1-hour video ~15-20 min. The 300s doc timeout would fire mid-run,
    # mark the job failed, and delete the staging file — after which frame
    # extraction hits VIDEO_NOT_FOUND (vision frames lost) and the upload reports
    # failure even though text partially stored. Media gets a generous cap.
    MEDIA_PROCESSING_TIMEOUT_SEC: int = _int("MEDIA_PROCESSING_TIMEOUT_SEC", 2400)
    LLM_CALL_TIMEOUT_SEC: int = _int("LLM_CALL_TIMEOUT_SEC", 60)
    RETRIEVAL_TIMEOUT: int = _int("RETRIEVAL_TIMEOUT", 10)
    EMBEDDING_TIMEOUT: int = _int("EMBEDDING_TIMEOUT", 5)
    VECTOR_DB_TIMEOUT: int = _int("VECTOR_DB_TIMEOUT", 10)
    MODEL_TIMEOUT_SEC: int = _int("MODEL_TIMEOUT_SEC", 120)
    SLOW_REQUEST_THRESHOLD: float = _float("SLOW_REQUEST_THRESHOLD", 2.0)
    INGESTION_WORKER_COUNT: int = _int("INGESTION_WORKER_COUNT", 6)

    # LLM
    LLM_MODEL_PATH: str = _str(
        "LLM_MODEL_PATH", "./.hf_cache/gguf/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
    )
    LLM_MAX_TOKENS: int = _int("LLM_MAX_TOKENS", 1024)
    CONTEXT_MAX_TOKENS: int = _int("CONTEXT_MAX_TOKENS", 16384)
    # "chatml" (Qwen2.5 family, current default) | "raw" (legacy Mistral plain-
    # completion prompting). Applied once, in app/llm/gguf_model.py, right
    # before the assembled prompt is sent to llama-server — no prompt-assembly
    # call site (prompt_builder.py, reasoning_engine.py) needs to know about it.
    LLM_PROMPT_FORMAT: str = _str("LLM_PROMPT_FORMAT", "chatml")
    LLM_TEMPERATURE: float = _float("LLM_TEMPERATURE", 0.0)
    LLM_TEMPERATURE_FACTUAL: float = _float("LLM_TEMPERATURE_FACTUAL", 0.1)
    LLM_TEMPERATURE_GENERATIVE: float = _float("LLM_TEMPERATURE_GENERATIVE", 0.35)
    # Floor temperature for an explicit user "regenerate this answer" request.
    # The normal path decodes at 0.0-0.1 (greedy/near-greedy) so the same
    # question always yields the same answer — correct for a factual RAG, but
    # it makes the regenerate button a no-op: identical prompt, identical
    # sampling, identical text. See app/llm/regeneration.py. Kept modest on
    # purpose: high enough to explore a genuinely different phrasing/ordering,
    # low enough that the model doesn't start improvising figures.
    LLM_TEMPERATURE_REGENERATE: float = _float("LLM_TEMPERATURE_REGENERATE", 0.4)
    LLM_TOP_P: float = _float("LLM_TOP_P", 0.9)
    LLM_TOP_K_SAMPLING: int = _int("LLM_TOP_K_SAMPLING", 40)
    LLM_MIN_P: float = _float("LLM_MIN_P", 0.05)
    LLM_REPEAT_PENALTY: float = _float("LLM_REPEAT_PENALTY", 1.1)
    MAX_PROMPT_CHARS: int = _int("MAX_PROMPT_CHARS", 14000)
    LLM_GPU_LAYERS: int = _int("LLM_GPU_LAYERS", -1)
    LLM_THREADS: int = _int("LLM_THREADS", 4)
    LLM_N_BATCH: int = _int("LLM_N_BATCH", 1024)
    # mlock pins the GGUF file in host RAM. With n_gpu_layers=-1 the weights are
    # already on the GPU, so mlock just wastes host RAM and can force the kernel
    # to swap the LLM's active runtime pages instead — thrashing generation to a
    # crawl. Keep it OFF; the mmap'd file is reclaimable and the GPU copy is what
    # actually serves inference.
    LLM_USE_MLOCK: bool = _bool("LLM_USE_MLOCK", False)
    # SEPARATE LLM PROCESS — llama.cpp on the GPU must run in its OWN process.
    # In-process, llama.cpp's CUDA init corrupts PyTorch's CUDA context on worker
    # threads (embedding SIGSEGVs at ingest time). Running llama.cpp as a separate
    # llama-server (its own CUDA context) lets the LLM stay on GPU (fast) while
    # PyTorch owns CUDA in the main process. start_server.py launches the server;
    # GGUFModel proxies generate()/stream() to it over HTTP.
    LLM_USE_SERVER: bool = _bool("LLM_USE_SERVER", True)
    LLM_SERVER_HOST: str = _str("LLM_SERVER_HOST", "127.0.0.1")
    LLM_SERVER_PORT: int = _int("LLM_SERVER_PORT", 8081)
    LLM_MAX_RETRIES: int = _int("LLM_MAX_RETRIES", 2)
    LLM_RETRY_WAIT_MIN: int = _int("LLM_RETRY_WAIT_MIN", 1)
    LLM_RETRY_WAIT_MAX: int = _int("LLM_RETRY_WAIT_MAX", 5)

    # EMBEDDINGS
    EMBEDDING_MODEL: str = _str("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    EMBEDDING_BATCH_SIZE: int = _int("EMBEDDING_BATCH_SIZE", 128)
    EMBEDDING_MAX_BATCH_SIZE: int = _int("EMBEDDING_MAX_BATCH_SIZE", 8)
    EMBEDDING_MAX_SEQ_LEN: int = _int("EMBEDDING_MAX_SEQ_LEN", 512)
    INGESTION_MICRO_BATCH: int = _int("INGESTION_MICRO_BATCH", 64)
    # Clear CUDA cache every N micro-batches during ingestion (OOM guard
    # without the per-chunk empty_cache+gc tax that dominated embed latency).
    INGESTION_CACHE_CLEAR_EVERY: int = _int("INGESTION_CACHE_CLEAR_EVERY", 25)
    # Parallelism caps for heavy ML operations — keep within VRAM budget
    VIDEO_CAPTION_CONCURRENCY: int = _int("VIDEO_CAPTION_CONCURRENCY", 3)
    # Shared GPU semaphore (app/core/gpu_admission.py) — max concurrent heavy
    # GPU jobs across BOTH ingestion (embed/Whisper/Qwen2-VL) and query
    # (embed/rerank/generate). One shared gate, not two independent ones —
    # ingestion and query compete for the same physical VRAM budget, so
    # separate limits could still sum past what the box can hold.
    #
    # Real measurement, 2026-07-30 (g6e.xlarge / L40S, 48GB): a single full
    # multimodal ingestion (all 7 modalities in one session) used ~42GB,
    # leaving ~4GB headroom — not enough for a second concurrent heavy job of
    # similar size. This setting was previously 3, deliberately deferred
    # pending exactly this measurement (see git history) — now that it
    # exists, 3 is not conservative, it's a near-guaranteed OOM under any
    # real concurrent load. Set to 1 (fully serialize) until a future
    # headroom measurement justifies raising it.
    MAX_CONCURRENT_GPU_JOBS: int = _int("MAX_CONCURRENT_GPU_JOBS", 1)
    # How long a request waits for a GPU slot before getting a clean "server
    # busy" response instead of queueing silently. Ingestion of a large video
    # can hold the slot for minutes; a query waiting behind it should give up
    # and let the user retry rather than hang with no feedback.
    GPU_ADMISSION_TIMEOUT_SEC: int = _int("GPU_ADMISSION_TIMEOUT_SEC", 45)
    AUDIO_TRANSCRIPTION_WORKERS: int = _int("AUDIO_TRANSCRIPTION_WORKERS", 2)
    PDF_OCR_WORKERS: int = _int("PDF_OCR_WORKERS", 8)
    EMBEDDING_CACHE_TTL: int = _int("EMBEDDING_CACHE_TTL", 2_592_000)
    TEXT_EMBEDDING_DIM: int = _int("TEXT_EMBEDDING_DIM", 1024)
    # Matryoshka Representation Learning truncation length for
    # TextEmbedder.embed_matryoshka() (app/embeddings/base_embedder.py) — BGE-
    # large-en-v1.5 supports MRL, so the first N dims of the full 1024-d
    # vector are themselves a valid (lower-fidelity) embedding. 256 is BGE's
    # own commonly-cited MRL truncation point (~4x smaller, modest recall
    # loss) for a fast first-pass ANN search re-ranked with the full vector.
    # Was previously referenced with no setting defined at all — an
    # AttributeError waiting to happen the first time this method was called.
    MATRYOSHKA_SHORT_DIM: int = _int("MATRYOSHKA_SHORT_DIM", 256)
    # BGE-large requires instruction prefix on queries only (not documents)
    BGE_QUERY_INSTRUCTION: str = _str(
        "BGE_QUERY_INSTRUCTION", "Represent this sentence for searching relevant passages: "
    )

    # MODALITY FEATURE FLAGS — set to false to skip model load and all related processing
    ENABLE_VISION: bool = _bool("ENABLE_VISION", True)
    ENABLE_AUDIO: bool = _bool("ENABLE_AUDIO", True)

    # FINANCE-GRADE PROCESSING FEATURE FLAGS
    PDF_USE_PDFPLUMBER: bool = _bool("PDF_USE_PDFPLUMBER", True)
    VIDEO_SCENE_DETECTION: bool = _bool("VIDEO_SCENE_DETECTION", True)
    EXCEL_SEMANTIC_GROUP: bool = _bool("EXCEL_SEMANTIC_GROUP", True)
    AUDIO_SPEAKER_SUBINDEX_ENABLED: bool = _bool("AUDIO_SPEAKER_SUBINDEX_ENABLED", True)

    # MODEL DEVICE / WARMUP — target deploy is L40S 48 GB (AWS g6e.xlarge)
    # Profiles: "auto" (CUDA → all_gpu, else cpu), "hybrid", "all_gpu", "all_cpu"
    MODELS_DEVICE_PROFILE: str = _str("MODELS_DEVICE_PROFILE", "all_gpu")
    # Per docs/runbooks/phase-30-aws-deployment.md Stage 1 item 4, the g6e.xlarge
    # L40S value is a box-level `.env` override, not a code default — this
    # conservative default merely caps usage (safe either way), so it's left
    # as-is here rather than baked into a shared default that isn't
    # hardware-pinned. Set VRAM_BUDGET_GB=44 (not 46 — see .env.example) in
    # the box's .env on g6e.xlarge: the nominal 48GB card shows as ~44.4GB
    # actually visible to CUDA once the OS/driver take their share
    # (confirmed live). A value above the real total is a silent no-op —
    # device_manager.py's LLM layer-count formula does
    # min(actual_free_vram, VRAM_BUDGET_GB), so 46 never binds and the
    # setting stops doing anything at all.
    VRAM_BUDGET_GB: float = _float("VRAM_BUDGET_GB", 22.0)
    WARMUP_AT_STARTUP: bool = _bool("WARMUP_AT_STARTUP", True)
    # LATENCY: eager-load ONLY the text query/ingest essentials at startup
    # (~110s vs ~280s for all 12). Vision/audio/video models load on first use of
    # that modality via model_registry.ensure_for_modality() (ingest) and
    # ensure_for_query(needs_vision=True) (cross-modal query). The first image/
    # audio/video upload after boot pays a one-time load; cached thereafter.
    # To pre-warm everything again, restore the full list:
    #   text_embedder,llm,reranker,siglip,image_embedder,siglip_text_embedder,blip,trocr,whisper,ner,finbert,diarizer
    WARMUP_MODELS: list[str] = _list("WARMUP_MODELS", ["text_embedder", "llm", "reranker"])
    MODEL_PARALLEL_LOAD: bool = _bool("MODEL_PARALLEL_LOAD", True)

    # IDLE MODEL EVICTION — the unload half of WARMUP_MODELS' load-on-demand.
    #
    # model_loader.unload_idle_models() has existed since 2026-08-01 but its
    # only caller was _oom_guard(loading=...), which runs solely when ANOTHER
    # heavy model is about to load. So eviction was reactive to demand
    # pressure and never to idleness: upload one image, and BLIP2/Qwen2-VL
    # stayed resident in VRAM indefinitely if nothing else was ever uploaded.
    # app/core/model_reaper.py is the background sweeper that makes the
    # idle_seconds argument actually mean something.
    #
    # Only the 8 ingestion-only models in _EVICTABLE_MODELS are affected.
    # The query hot set (llm, text_embedder, reranker, and the SigLIP family
    # used for cross-modal query search) is pinned and never evicted — a
    # live query must never wait on a model reload.
    MODEL_IDLE_EVICTION_ENABLED: bool = _bool("MODEL_IDLE_EVICTION_ENABLED", True)
    # Long enough that a user uploading several files in one sitting doesn't
    # pay a reload (~25s for a VLM) between them; short enough that a box
    # left alone gives its VRAM back promptly.
    MODEL_IDLE_TIMEOUT_SEC: float = _float("MODEL_IDLE_TIMEOUT_SEC", 300.0)
    MODEL_REAPER_INTERVAL_SEC: float = _float("MODEL_REAPER_INTERVAL_SEC", 60.0)
    # Urgent trigger: when free VRAM falls below this, evict least-recently-
    # used evictable models regardless of age. 0 disables pressure eviction
    # and leaves only the idle-TTL sweep.
    MODEL_EVICT_VRAM_WATERMARK_GB: float = _float("MODEL_EVICT_VRAM_WATERMARK_GB", 6.0)
    LLM_GPU_LAYERS_AUTO: bool = _bool("LLM_GPU_LAYERS_AUTO", True)
    LLM_GPU_LAYERS_ALL: int = _int("LLM_GPU_LAYERS_ALL", -1)
    LLM_N_CTX: int = _int("LLM_N_CTX", 16384)
    EMBEDDER_HALF_PRECISION: bool = _bool("EMBEDDER_HALF_PRECISION", _cuda_available())
    VISION_HALF_PRECISION: bool = _bool("VISION_HALF_PRECISION", _cuda_available())
    WHISPER_COMPUTE_TYPE: str = _str("WHISPER_COMPUTE_TYPE", _auto_whisper_compute())
    EMBEDDER_DEVICE: str = _str("EMBEDDER_DEVICE", _auto_device())
    RERANKER_DEVICE: str = _str("RERANKER_DEVICE", _auto_device())
    SIGLIP_DEVICE: str = _str("SIGLIP_DEVICE", _auto_device())
    BLIP_DEVICE: str = _str("BLIP_DEVICE", _auto_device())
    QWEN2_VL_DEVICE: str = _str("QWEN2_VL_DEVICE", _auto_device())
    TROCR_DEVICE: str = _str("TROCR_DEVICE", _auto_device())
    DIARIZER_DEVICE: str = _str("DIARIZER_DEVICE", _auto_device())
    NER_DEVICE: str = _str("NER_DEVICE", _auto_device())
    WHISPER_DEVICE: str = _str("WHISPER_DEVICE", _auto_device())
    LLM_DEVICE_HINT: str = _str("LLM_DEVICE_HINT", _auto_device())

    # VISION MODELS
    SIGLIP_MODEL: str = _str("SIGLIP_MODEL", "google/siglip-so400m-patch14-384")
    VISION_EMBEDDING_DIM: int = _int("VISION_EMBEDDING_DIM", 1152)
    # BLIP — image captioning (Salesforce/blip-image-captioning-large, 1.56 GB fp16)
    BLIP_MODEL: str = _str("BLIP_MODEL", "Salesforce/blip-image-captioning-large")
    BLIP_MAX_TOKENS: int = _int("BLIP_MAX_TOKENS", 100)
    BLIP_NUM_BEAMS: int = _int("BLIP_NUM_BEAMS", 5)
    # Qwen2-VL — video frame + financial chart captioning.
    # Upgraded 2B→7B (INT8, ~9.5 GB): the 7B produces materially richer,
    # more accurate chart/figure descriptions. VRAM verified to fit — image
    # work peaks ~20 GB of the 22 GB budget (device_manager evicts idle models
    # under pressure). Exact numeric chart values come from the deterministic
    # digitizer in image_chunker.py regardless of model size.
    QWEN2_VL_MODEL: str = _str("QWEN2_VL_MODEL", "Qwen/Qwen2-VL-7B-Instruct")
    # Video frame captioning uses a smaller VLM than image chart digitisation: a
    # 1-hour video ingest loads Whisper+pyannote+SigLIP+TrOCR+BGE simultaneously
    # next to the resident llama-server, and the 7B INT8 captioner (~8GB) adds
    # substantial VRAM pressure under a tight budget → risks later GPU allocs
    # OOMing and the upload failing before reaching Qdrant. Frames are a chart +
    # ticker (verbatim text already captured by TrOCR), so 2B is ample and frees
    # ~6GB. Image keeps 7B.
    # NOTE: on the 48GB L40S (g6e.xlarge) this VRAM constraint likely no longer
    # binds, but 2B is deliberately kept here rather than reverted blind: doing so
    # also needs re-validating the separate,
    # latency-motivated VIDEO_CAPTION_MAX_TOKENS=180 cut below (a 7B model at the
    # 400-token still-image budget measured ~80s/frame — unworkable at ~20 frames
    # for a 54-min call). Revisit both together once real hardware is available,
    # per docs/runbooks/phase-30-aws-deployment.md's "don't tune blind" guidance.
    VIDEO_QWEN2_VL_MODEL: str = _str("VIDEO_QWEN2_VL_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
    QWEN2_VL_LOAD_IN_8BIT: bool = _bool("QWEN2_VL_LOAD_IN_8BIT", False)
    QWEN2_VL_MAX_TOKENS: int = _int("QWEN2_VL_MAX_TOKENS", 400)
    # Video frame captioning uses a tighter token budget than still-image
    # captioning: an earnings webcast frame is a chart + ticker overlay, and
    # verbatim on-screen text is captured by TrOCR, so the VLM only needs a
    # short scene/number summary. The 400-token still-image budget made a 7B
    # INT8 model spend ~80s/frame (unworkable at 20 frames for a 54-min call).
    VIDEO_CAPTION_MAX_TOKENS: int = _int("VIDEO_CAPTION_MAX_TOKENS", 180)
    # TrOCR — printed OCR for financial documents
    TROCR_MODEL: str = _str("TROCR_MODEL", "microsoft/trocr-large-printed")
    MAX_IMAGE_DIM: int = _int("MAX_IMAGE_DIM", 1024)
    MAX_IMAGE_SIZE_MP: int = _int("MAX_IMAGE_SIZE_MP", 50)
    THUMBNAIL_WIDTH: int = _int("THUMBNAIL_WIDTH", 256)
    THUMBNAIL_HEIGHT: int = _int("THUMBNAIL_HEIGHT", 256)
    IMAGE_PRIVACY_MODE: bool = _bool("IMAGE_PRIVACY_MODE", False)
    SOLID_COLOR_THRESHOLD: float = _float("SOLID_COLOR_THRESHOLD", 0.95)

    # ASR WHISPER
    WHISPER_MODEL: str = _str("WHISPER_MODEL", "large-v3")
    AUDIO_SAMPLE_RATE: int = _int("AUDIO_SAMPLE_RATE", 16000)
    MAX_AUDIO_DURATION_SEC: int = _int("MAX_AUDIO_DURATION_SEC", 10800)
    MAX_AUDIO_SEGMENTS: int = _int("MAX_AUDIO_SEGMENTS", 500)
    AUDIO_SNR_THRESHOLD_DB: float = _float("AUDIO_SNR_THRESHOLD_DB", 10.0)
    AUDIO_SILENCE_GAP_MS: int = _int("AUDIO_SILENCE_GAP_MS", 200)
    AUDIO_CHUNK_DURATION_SEC: int = _int("AUDIO_CHUNK_DURATION_SEC", 1800)
    # Fine-grained audio chunking (audio-only; video keeps the av_shared defaults
    # of 75/225). Smaller chunks put a specific spoken fact in a focused,
    # retrievable segment instead of a ~1.5-min block of generic prose.
    AUDIO_CHUNK_MIN_WORDS: int = _int("AUDIO_CHUNK_MIN_WORDS", 45)
    AUDIO_CHUNK_MAX_WORDS: int = _int("AUDIO_CHUNK_MAX_WORDS", 130)
    # Fine-grained VIDEO transcript chunking (video-only; the 20 vision-frame
    # chunks are separate and unaffected). Same sweet spot as audio — isolates a
    # spoken fact ("$102.5 billion in revenue") from the coarse disclaimer block.
    VIDEO_CHUNK_MIN_WORDS: int = _int("VIDEO_CHUNK_MIN_WORDS", 45)
    VIDEO_CHUNK_MAX_WORDS: int = _int("VIDEO_CHUNK_MAX_WORDS", 130)
    DIARIZATION_ENABLED: bool = _bool("DIARIZATION_ENABLED", True)
    AUDIO_DIARIZATION_ENABLED: bool = _bool("AUDIO_DIARIZATION_ENABLED", True)
    # Pyannote speaker diarization (requires HF_TOKEN + model access approval)
    DIARIZATION_MODEL: str = _str("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
    # Finance NER
    NER_MODEL: str = _str("NER_MODEL", "dslim/bert-base-NER")
    # FinBERT — finance-domain tone/sentiment classifier (yiyanghkust/finbert-tone)
    FINBERT_MODEL: str = _str("FINBERT_MODEL", "yiyanghkust/finbert-tone")
    FINBERT_ENABLED: bool = _bool("FINBERT_ENABLED", False)
    FINBERT_BATCH_SIZE: int = _int("FINBERT_BATCH_SIZE", 32)
    HF_TOKEN: str | None = _opt("HF_TOKEN")
    WHISPER_DOMAIN_VOCAB: list[str] = _list(
        "WHISPER_DOMAIN_VOCAB",
        [
            "RAG",
            "FAISS",
            "Qdrant",
            "LlamaIndex",
            "LangChain",
            "RAGAS",
            "Pinecone",
            "ChromaDB",
            "Mistral",
            "OpenAI",
            "Anthropic",
            "embedding",
            "retrieval",
            "chunking",
            "reranker",
            "BM25",
            "cosine",
            "upsert",
        ],
    )
    WHISPER_FILLER_WORDS: list[str] = _list("WHISPER_FILLER_WORDS", ["uh", "um", "er", "hmm"])

    # RERANKER
    RERANKER_MODEL: str = _str("RERANKER_MODEL", "BAAI/bge-reranker-large")
    RERANK_TOP_K: int = _int("RERANK_TOP_K", 20)
    RERANK_MAX_INPUT: int = _int("RERANK_MAX_INPUT", 30)
    RERANK_CONTEXT_MAX_CHARS: int = _int("RERANK_CONTEXT_MAX_CHARS", 2048)
    RERANK_MODEL_WEIGHT: float = _float("RERANK_MODEL_WEIGHT", 0.7)
    RERANK_FUSION_WEIGHT: float = _float("RERANK_FUSION_WEIGHT", 0.3)
    RERANK_POSITION_WEIGHT: float = _float("RERANK_POSITION_WEIGHT", 0.1)
    RERANK_SCORE_THRESHOLD: float = _float("RERANK_SCORE_THRESHOLD", 0.1)
    RERANK_BATCH_SIZE: int = _int("RERANK_BATCH_SIZE", 32)
    RERANK_SOURCE_MAX_CHUNKS: int = _int("RERANK_SOURCE_MAX_CHUNKS", 8)
    MMR_ENABLED: bool = _bool("MMR_ENABLED", True)
    # 0.5 picks diverse chunks so cross-doc questions get both DOC-010 and DOC-011
    MMR_LAMBDA: float = _float("MMR_LAMBDA", 0.5)

    RERANK_MODALITY_WEIGHTS: dict[str, float] = {
        "text": 1.0,
        "table": 1.15,
        "image": 0.75,
        "audio": 0.85,
        "video": 0.75,
    }

    # QDRANT
    QDRANT_URL: str | None = _opt("QDRANT_URL")
    QDRANT_API_KEY: str | None = _opt("QDRANT_API_KEY")
    QDRANT_HOST: str = _str("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = _int("QDRANT_PORT", 6333)
    QDRANT_TIMEOUT: int = _int("QDRANT_TIMEOUT", 10)
    TEXT_COLLECTION_NAME: str = _str("TEXT_COLLECTION_NAME", "text_collection")
    VISION_COLLECTION_NAME: str = _str("VISION_COLLECTION_NAME", "vision_collection")
    COLLECTION_NAME: str = _str("COLLECTION_NAME", "rag_collection")
    QDRANT_BATCH_SIZE: int = _int("QDRANT_BATCH_SIZE", 100)
    QDRANT_MAX_DOCS: int = _int("QDRANT_MAX_DOCS", 100_000)
    QDRANT_ALLOW_RECREATE: bool = _bool("QDRANT_ALLOW_RECREATE", True)
    QDRANT_INIT_RETRIES: int = _int("QDRANT_INIT_RETRIES", 3)
    QDRANT_RETRY_DELAY: int = _int("QDRANT_RETRY_DELAY", 2)
    QDRANT_TEXT_MAX_CHARS: int = _int("QDRANT_TEXT_MAX_CHARS", 2000)
    QDRANT_SOFT_DELETE: bool = _bool("QDRANT_SOFT_DELETE", True)
    QDRANT_CB_FAIL_MAX: int = _int("QDRANT_CB_FAIL_MAX", 5)
    QDRANT_CB_RESET_TIMEOUT: int = _int("QDRANT_CB_RESET_TIMEOUT", 60)

    # CIRCUIT BREAKER — SHARED DEFAULTS
    CIRCUIT_BREAKER_MAX_FAILURES: int = _int("CIRCUIT_BREAKER_MAX_FAILURES", 5)
    CIRCUIT_BREAKER_RESET_TIMEOUT: int = _int("CIRCUIT_BREAKER_RESET_TIMEOUT", 60)

    # RETRY — SHARED DEFAULTS
    RETRY_MAX_ATTEMPTS: int = _int("RETRY_MAX_ATTEMPTS", 3)
    RETRY_WAIT_MIN_SEC: float = _float("RETRY_WAIT_MIN_SEC", 1.0)
    RETRY_WAIT_MAX_SEC: float = _float("RETRY_WAIT_MAX_SEC", 10.0)

    # REDIS
    REDIS_URL: str | None = _opt("REDIS_URL")
    REDIS_TOKEN: str | None = _opt("REDIS_TOKEN")
    REDIS_HOST: str = _str("REDIS_HOST", "localhost")
    REDIS_PORT: int = _int("REDIS_PORT", 6379)
    REDIS_DB: int = _int("REDIS_DB", 0)
    REDIS_TIMEOUT: int = _int("REDIS_TIMEOUT", 5)
    REDIS_TTL_SECONDS: int = _int("REDIS_TTL_SECONDS", 86400)
    REDIS_QUERY_CACHE_TTL: int = _int("REDIS_QUERY_CACHE_TTL", 3600)
    REDIS_EMBEDDING_CACHE_TTL: int = _int("REDIS_EMBEDDING_CACHE_TTL", 2_592_000)
    REDIS_KEY_PREFIX: str = _str("REDIS_KEY_PREFIX", "rag")
    USE_REDIS: bool = _bool("USE_REDIS", True)
    USE_MONGO: bool = _bool("USE_MONGO", True)
    REDIS_CB_FAIL_MAX: int = _int("REDIS_CB_FAIL_MAX", 5)
    REDIS_CB_RESET_TIMEOUT: int = _int("REDIS_CB_RESET_TIMEOUT", 30)
    REDIS_MAX_WINDOW_TURNS: int = _int("REDIS_MAX_WINDOW_TURNS", 40)
    REDIS_MAX_CONNECTIONS: int = _int("REDIS_MAX_CONNECTIONS", 10)
    # LOCAL REDIS CACHE — hot-path ephemeral state (job status, embedding cache,
    # token blacklist, rate limits). Always local (host/port) so it never pays the
    # ~200ms Upstash cloud round-trip. Durable conversation memory still uses Upstash.
    LOCAL_CACHE_HOST: str = _str("LOCAL_CACHE_HOST", "localhost")
    LOCAL_CACHE_PORT: int = _int("LOCAL_CACHE_PORT", 6379)
    LOCAL_CACHE_DB: int = _int("LOCAL_CACHE_DB", 1)
    # In-process TTL (seconds) for JWT revocation/generation checks so most authed
    # requests do zero network. Revocation propagates within this window. 0 disables.
    AUTH_REVOCATION_CACHE_TTL: int = _int("AUTH_REVOCATION_CACHE_TTL", 30)

    # MONGODB
    MONGO_URI: str = _str("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME: str = _str("MONGO_DB_NAME", "rag_memory")
    DB_TIMEOUT_MS: int = _int("DB_TIMEOUT_MS", 5000)
    DB_MAX_POOL_SIZE: int = _int("DB_MAX_POOL_SIZE", 20)
    MONGO_CB_FAIL_MAX: int = _int("MONGO_CB_FAIL_MAX", 5)
    MONGO_CB_RESET_TIMEOUT: int = _int("MONGO_CB_RESET_TIMEOUT", 60)
    MONGO_MESSAGES_COLLECTION: str = _str("MONGO_MESSAGES_COLLECTION", "messages")
    MONGO_SUMMARIES_COLLECTION: str = _str("MONGO_SUMMARIES_COLLECTION", "summaries")
    MONGO_MEMORY_COLLECTION: str = _str("MONGO_MEMORY_COLLECTION", "messages")
    MONGO_SUMMARY_COLLECTION: str = _str("MONGO_SUMMARY_COLLECTION", "summaries")
    MONGO_AUDIT_COLLECTION: str = _str("MONGO_AUDIT_COLLECTION", "audit_log")
    MONGO_SESSIONS_COLLECTION: str = _str("MONGO_SESSIONS_COLLECTION", "chat_sessions")
    MONGO_FEEDBACK_COLLECTION: str = _str("MONGO_FEEDBACK_COLLECTION", "feedback")
    MONGO_EVAL_SHADOW_COLLECTION: str = _str("MONGO_EVAL_SHADOW_COLLECTION", "eval_shadow")

    # ONLINE EVAL (Phase 31) — deterministic sampling of live queries into
    # MONGO_EVAL_SHADOW_COLLECTION for app/eval/jobs/online_eval.py to score.
    # 0.0 = disabled (default). Reference-free metrics only (no gold set is
    # available for arbitrary live traffic) — see app/eval/jobs/online_eval.py.
    ONLINE_EVAL_SAMPLE_RATE: float = _float("ONLINE_EVAL_SAMPLE_RATE", 0.0)
    ONLINE_EVAL_ENABLED: bool = _bool("ONLINE_EVAL_ENABLED", False)
    ONLINE_EVAL_INTERVAL_SEC: int = _int("ONLINE_EVAL_INTERVAL_SEC", 1800)

    # TEXT REPAIR — per-pass toggles for the broken-corpus hardening layer
    # (mojibake fix, noise-line strip, whitespace recovery, OCR normalization,
    # footnote strip, placeholder skip, title-mismatch flag, version tagging).
    # All default ON; each pass is a no-op on clean text so the cost is low.
    TEXT_REPAIR_ENABLED: bool = _bool("TEXT_REPAIR_ENABLED", True)
    TEXT_REPAIR_MOJIBAKE: bool = _bool("TEXT_REPAIR_MOJIBAKE", True)
    TEXT_REPAIR_NOISE_LINES: bool = _bool("TEXT_REPAIR_NOISE_LINES", True)
    TEXT_REPAIR_WHITESPACE: bool = _bool("TEXT_REPAIR_WHITESPACE", True)
    TEXT_REPAIR_OCR: bool = _bool("TEXT_REPAIR_OCR", True)
    TEXT_REPAIR_FOOTNOTES: bool = _bool("TEXT_REPAIR_FOOTNOTES", True)
    TEXT_REPAIR_PLACEHOLDERS: bool = _bool("TEXT_REPAIR_PLACEHOLDERS", True)
    TEXT_REPAIR_TITLE_MISMATCH: bool = _bool("TEXT_REPAIR_TITLE_MISMATCH", True)
    TEXT_REPAIR_VERSION_TAG: bool = _bool("TEXT_REPAIR_VERSION_TAG", True)

    # CHUNKING
    CHUNK_SIZE: int = _int("CHUNK_SIZE", 1024)
    CHUNK_OVERLAP: int = _int("CHUNK_OVERLAP", 128)
    CHUNK_HASH_ID: bool = _bool("CHUNK_HASH_ID", True)
    FINANCE_NUMBER_PROTECT: bool = _bool("FINANCE_NUMBER_PROTECT", True)
    CHUNK_TARGET_TOKENS: int = _int("CHUNK_TARGET_TOKENS", 120)
    CHUNK_OVERLAP_SENTENCES: int = _int("CHUNK_OVERLAP_SENTENCES", 1)
    CHUNK_OVERLAP_RATIO: float = _float("CHUNK_OVERLAP_RATIO", 0.15)
    CHUNK_MIN_SIZE: int = _int("CHUNK_MIN_SIZE", 50)
    MAX_CHUNKS: int = _int("MAX_CHUNKS", 1500)
    INGESTION_BATCH_SIZE: int = _int("INGESTION_BATCH_SIZE", 16)
    CHUNK_MAX_TOKENS: int = _int("CHUNK_MAX_TOKENS", 512)
    CHUNK_MINHASH_THRESHOLD: float = _float("CHUNK_MINHASH_THRESHOLD", 0.85)
    CHUNK_MINHASH_PERMUTATIONS: int = _int("CHUNK_MINHASH_PERMUTATIONS", 128)
    CHUNK_ADAPTIVE_ENABLED: bool = _bool("CHUNK_ADAPTIVE_ENABLED", True)
    LATE_CHUNKING_ENABLED: bool = _bool("LATE_CHUNKING_ENABLED", False)
    EXCEL_ROWS_PER_CHUNK: int = _int("EXCEL_ROWS_PER_CHUNK", 25)

    # BM25
    BM25_MAX_DOCS: int = _int("BM25_MAX_DOCS", 10_000)
    BM25_TOP_K: int = _int("BM25_TOP_K", 15)
    BM25_MIN_SCORE: float = _float("BM25_MIN_SCORE", 0.05)
    BM25_MAX_TEXT_CHARS: int = _int("BM25_MAX_TEXT_CHARS", 2000)
    BM25_MAX_TOKENS: int = _int("BM25_MAX_TOKENS", 512)

    BM25_MODALITY_WEIGHTS: dict[str, float] = {
        "text": 1.0,
        "image": 0.9,
        "audio": 1.0,
        "video": 1.1,
    }

    # HYBRID RETRIEVAL
    HYBRID_WEIGHT_BM25: float = _float("HYBRID_WEIGHT_BM25", 0.35)
    HYBRID_WEIGHT_VECTOR: float = _float("HYBRID_WEIGHT_VECTOR", 0.65)
    HYBRID_WEIGHT_VISION: float = _float("HYBRID_WEIGHT_VISION", 0.2)
    HYBRID_CANDIDATES_MULTIPLIER: int = _int("HYBRID_CANDIDATES_MULTIPLIER", 3)
    HYBRID_MIN_SCORE: float = _float("HYBRID_MIN_SCORE", 0.20)
    HYBRID_RRF_K: int = _int("HYBRID_RRF_K", 60)
    HYBRID_MMR_LAMBDA: float = _float("HYBRID_MMR_LAMBDA", 0.7)
    HYBRID_SCORE_THRESHOLD: float = _float("HYBRID_SCORE_THRESHOLD", 0.1)

    # FUSION
    FUSION_SIMILARITY_THRESHOLD: float = _float("FUSION_SIMILARITY_THRESHOLD", 0.7)
    FUSION_SCORE_WEIGHT: float = _float("FUSION_SCORE_WEIGHT", 0.6)
    FUSION_QUALITY_WEIGHT: float = _float("FUSION_QUALITY_WEIGHT", 0.2)
    FUSION_MODALITY_WEIGHT: float = _float("FUSION_MODALITY_WEIGHT", 0.2)
    FUSION_MAX_INPUT: int = _int("FUSION_MAX_INPUT", 60)
    FUSION_MAX_TEXT_CHARS: int = _int("FUSION_MAX_TEXT_CHARS", 1200)
    FUSION_MIN_SCORE: float = _float("FUSION_MIN_SCORE", 0.05)

    FUSION_MODALITY_WEIGHTS: dict[str, float] = {
        "text": 1.0,
        "image": 0.9,
        "audio": 1.1,
        "video": 1.15,
    }

    # RETRIEVAL
    DEFAULT_TOP_K: int = _int("DEFAULT_TOP_K", 20)
    VECTOR_TOP_K: int = _int("VECTOR_TOP_K", 25)
    MAX_CONTEXT_DOCS: int = _int("MAX_CONTEXT_DOCS", 10)
    MAX_CONTEXT_CHARS: int = _int("MAX_CONTEXT_CHARS", 16000)
    RAG_TOP_K: int = _int("RAG_TOP_K", 20)
    RAG_DOC_MAX_CHARS: int = _int("RAG_DOC_MAX_CHARS", 2500)

    # STREAMING — token flush holdback (see RAGPipeline.stream)
    STREAM_PREFIX_GATE_CHARS: int = _int("STREAM_PREFIX_GATE_CHARS", 160)
    STREAM_HOLDBACK_CHARS: int = _int("STREAM_HOLDBACK_CHARS", 48)

    # QUERY DECOMPOSITION
    MAX_SUBQUERIES: int = _int("MAX_SUBQUERIES", 1)
    SUBQUERY_MAX_TOKENS: int = _int("SUBQUERY_MAX_TOKENS", 64)
    DECOMPOSITION_MIN_WORDS: int = _int("DECOMPOSITION_MIN_WORDS", 8)
    # Heuristic gate: only call the LLM decomposer for structurally multi-part
    # queries (≥2 question marks or an explicit compare/multi-part marker).
    # Measured logs show decompose yields 1 subquery ~100% of the time on
    # simple questions — a pure LLM-latency tax on every query otherwise.
    DECOMPOSITION_HEURISTIC_GATE: bool = _bool("DECOMPOSITION_HEURISTIC_GATE", True)
    DECOMPOSITION_MAX_SUBQUERIES: int = _int("DECOMPOSITION_MAX_SUBQUERIES", 3)
    DECOMPOSITION_CONFIDENCE_THRESHOLD: float = _float("DECOMPOSITION_CONFIDENCE_THRESHOLD", 0.5)
    DECOMPOSITION_KEYWORDS: list[str] = _list(
        "DECOMPOSITION_KEYWORDS",
        [
            "and",
            "also",
            "then",
            "but",
            "additionally",
        ],
    )

    # WEB SEARCH
    TAVILY_API_KEY: str = _str("TAVILY_API_KEY", "")
    WEB_MAX_DOCS: int = _int("WEB_MAX_DOCS", 5)
    WEB_MAX_RESULTS: int = _int("WEB_MAX_RESULTS", 10)
    WEB_DOC_MAX_CHARS: int = _int("WEB_DOC_MAX_CHARS", 1000)
    WEB_CONTEXT_MAX_CHARS: int = _int("WEB_CONTEXT_MAX_CHARS", 4000)
    WEB_SEARCH_DEPTH: str = _str("WEB_SEARCH_DEPTH", "advanced")
    WEB_SEARCH_BLOCK_PRIVATE_IPS: bool = _bool("WEB_SEARCH_BLOCK_PRIVATE_IPS", True)
    WEB_SEARCH_DOMAIN_ALLOWLIST: list[str] = _list("WEB_SEARCH_DOMAIN_ALLOWLIST", [])

    # MEMORY
    MAX_HISTORY_MESSAGES: int = _int("MAX_HISTORY_MESSAGES", 40)
    MAX_SYSTEM_MESSAGES: int = _int("MAX_SYSTEM_MESSAGES", 5)
    MEMORY_TOP_K: int = _int("MEMORY_TOP_K", 5)
    MEMORY_MAX_CONTEXT_CHARS: int = _int("MEMORY_MAX_CONTEXT_CHARS", 400)
    MEMORY_SIM_THRESHOLD: float = _float("MEMORY_SIM_THRESHOLD", 0.3)
    MEMORY_RECENCY_SCALE: float = _float("MEMORY_RECENCY_SCALE", 3600.0)
    MEMORY_SUMMARY_MAX_CHARS: int = _int("MEMORY_SUMMARY_MAX_CHARS", 2000)
    MEMORY_SUMMARY_INPUT_CHARS: int = _int("MEMORY_SUMMARY_INPUT_CHARS", 4000)
    MEMORY_SUMMARY_EVERY_N_TURNS: int = _int("MEMORY_SUMMARY_EVERY_N_TURNS", 10)
    MIN_SUMMARY_LENGTH: int = _int("MIN_SUMMARY_LENGTH", 50)
    SLIDING_WINDOW_MAX_TOKENS: int = _int("SLIDING_WINDOW_MAX_TOKENS", 4096)

    MEMORY_ROLE_WEIGHTS: dict[str, float] = {
        "user": 1.0,
        "assistant": 0.9,
        "system": 1.1,
    }

    # AGENT
    AGENT_MAX_STEPS: int = _int("AGENT_MAX_STEPS", 10)
    AGENT_TIMEOUT_SEC: int = _int("AGENT_TIMEOUT_SEC", 60)
    AGENT_TOKEN_BUDGET: int = _int("AGENT_TOKEN_BUDGET", 32000)
    AGENT_STEP_TIMEOUT_SEC: int = _int("AGENT_STEP_TIMEOUT_SEC", 30)
    AGENT_HIGH_CONFIDENCE: float = _float("AGENT_HIGH_CONFIDENCE", 0.7)
    AGENT_LOW_CONFIDENCE: float = _float("AGENT_LOW_CONFIDENCE", 0.4)
    AGENT_TOOL_TIMEOUT: int = _int("AGENT_TOOL_TIMEOUT", 30)
    AGENT_MAX_RETRIES: int = _int("AGENT_MAX_RETRIES", 3)
    AGENT_ENABLE_DECOMPOSITION: bool = _bool("AGENT_ENABLE_DECOMPOSITION", True)
    AGENT_QUERY_EXPANSION_ENABLED: bool = _bool("AGENT_QUERY_EXPANSION_ENABLED", True)
    AGENT_MAX_ITERATIONS: int = _int("AGENT_MAX_ITERATIONS", 20)
    AGENT_PARALLEL_TOOLS: bool = _bool("AGENT_PARALLEL_TOOLS", True)

    # AGENT — ANSWER VERIFICATION LOOP (Phase 32)
    AGENT_VERIFY_ENABLED: bool = _bool("AGENT_VERIFY_ENABLED", True)
    AGENT_VERIFY_RETRIEVAL_MIN: float = _float("AGENT_VERIFY_RETRIEVAL_MIN", 90.0)
    AGENT_VERIFY_GROUNDING_MIN: float = _float("AGENT_VERIFY_GROUNDING_MIN", 90.0)
    AGENT_VERIFY_CITATION_MIN: float = _float("AGENT_VERIFY_CITATION_MIN", 95.0)
    AGENT_VERIFY_OVERALL_MIN: float = _float("AGENT_VERIFY_OVERALL_MIN", 90.0)
    AGENT_VERIFY_MAX_RETRIES: int = _int("AGENT_VERIFY_MAX_RETRIES", 3)
    AGENT_VERIFY_TIMEOUT_SEC: float = _float("AGENT_VERIFY_TIMEOUT_SEC", 30.0)
    AGENT_VERIFY_MIN_IMPROVEMENT_PCT: float = _float("AGENT_VERIFY_MIN_IMPROVEMENT_PCT", 2.0)
    AGENT_VERIFY_MODALITIES: list[str] = _list(
        "AGENT_VERIFY_MODALITIES",
        ["txt", "pdf", "docx", "xlsx", "image", "audio", "video"],
    )

    # VIDEO
    VIDEO_FRAME_INTERVAL_SEC: int = _int("VIDEO_FRAME_INTERVAL_SEC", 2)
    MAX_VIDEO_FRAMES: int = _int("MAX_VIDEO_FRAMES", 20)
    MAX_VIDEO_DURATION_SEC: int = _int("MAX_VIDEO_DURATION_SEC", 7200)
    FFMPEG_PATH: str = _str("FFMPEG_PATH", "ffmpeg")
    FFMPEG_TIMEOUT_SEC: int = _int("FFMPEG_TIMEOUT_SEC", 120)
    SCENE_CHANGE_THRESHOLD: float = _float("SCENE_CHANGE_THRESHOLD", 25.0)
    # PySceneDetect AdaptiveDetector operates on a very different scale than the
    # OpenCV mean-absdiff path (SCENE_CHANGE_THRESHOLD): its sane range is ~1.5-5
    # (default 3.0). Reusing 25.0 here made a static earnings-chart webcast yield
    # a single scene → one frame for the whole call. Separate, correctly-scaled.
    VIDEO_SCENE_ADAPTIVE_THRESHOLD: float = _float("VIDEO_SCENE_ADAPTIVE_THRESHOLD", 3.0)
    # Guarantee timeline coverage: blend scene cuts with uniform interval samples
    # so a near-static chart + persistent ticker still gets frames spread across
    # the whole recording (needed to capture the ticker/chart at multiple points).
    VIDEO_UNIFORM_TIMELINE_COVERAGE: bool = _bool("VIDEO_UNIFORM_TIMELINE_COVERAGE", True)
    # pHash near-duplicate hamming distance for frame dedup. The old literal 8
    # collapsed a whole earnings webcast to one frame: a Benzinga chart+ticker
    # layout is globally similar across time, so temporally-spread coverage
    # frames sit within 8 bits of each other and were wrongly dropped. A tight
    # value keeps only truly-identical frames out while preserving coverage.
    VIDEO_FRAME_DEDUP_HAMMING: int = _int("VIDEO_FRAME_DEDUP_HAMMING", 2)
    FRAME_DARKNESS_THRESHOLD: float = _float("FRAME_DARKNESS_THRESHOLD", 10.0)
    VIDEO_DUPLICATE_FRAME_THRESHOLD: float = _float("VIDEO_DUPLICATE_FRAME_THRESHOLD", 0.98)
    VIDEO_SUBTITLE_EXTRACTION: bool = _bool("VIDEO_SUBTITLE_EXTRACTION", True)
    VIDEO_HDR_TONEMAPPING: bool = _bool("VIDEO_HDR_TONEMAPPING", True)
    VIDEO_DEINTERLACE: bool = _bool("VIDEO_DEINTERLACE", False)
    VIDEO_THUMBNAIL_GRID_ROWS: int = _int("VIDEO_THUMBNAIL_GRID_ROWS", 3)
    VIDEO_THUMBNAIL_GRID_COLS: int = _int("VIDEO_THUMBNAIL_GRID_COLS", 3)
    VIDEO_STREAM_THRESHOLD_BYTES: int = _int("VIDEO_STREAM_THRESHOLD_BYTES", 10_737_418_240)

    # FILE SIZE LIMITS
    MAX_FILE_SIZE_TEXT: int = _int("MAX_FILE_SIZE_TEXT", 10 * 1024 * 1024)
    MAX_FILE_SIZE_PDF: int = _int("MAX_FILE_SIZE_PDF", 100 * 1024 * 1024)
    MAX_FILE_SIZE_DOCX: int = _int("MAX_FILE_SIZE_DOCX", 50 * 1024 * 1024)
    MAX_FILE_SIZE_XLSX: int = _int("MAX_FILE_SIZE_XLSX", 50 * 1024 * 1024)
    MAX_FILE_SIZE_IMAGE: int = _int("MAX_FILE_SIZE_IMAGE", 20 * 1024 * 1024)
    MAX_FILE_SIZE_AUDIO: int = _int("MAX_FILE_SIZE_AUDIO", 200 * 1024 * 1024)
    MAX_FILE_SIZE_VIDEO: int = _int("MAX_FILE_SIZE_VIDEO", 2 * 1024 * 1024 * 1024)
    MAX_FILE_SIZE_MB: int = _int("MAX_FILE_SIZE_MB", 500)
    UPLOAD_CHUNK_SIZE: int = _int("UPLOAD_CHUNK_SIZE", 1_048_576)

    # Matches this Settings class's UPPER_CASE field convention (MAX_FILE_SIZE_*),
    # not a function-naming mistake.
    @property
    def FILE_SIZE_LIMITS(self) -> dict[str, int]:  # noqa: N802
        return {
            "text": self.MAX_FILE_SIZE_TEXT,
            "pdf": self.MAX_FILE_SIZE_PDF,
            "docx": self.MAX_FILE_SIZE_DOCX,
            "xlsx": self.MAX_FILE_SIZE_XLSX,
            "image": self.MAX_FILE_SIZE_IMAGE,
            "audio": self.MAX_FILE_SIZE_AUDIO,
            "video": self.MAX_FILE_SIZE_VIDEO,
        }

    # INPUT VALIDATION
    PREFLIGHT_MAX_MS: int = _int("PREFLIGHT_MAX_MS", 50)
    WAV_HEADER_VALIDATION_MAX_MS: int = _int("WAV_HEADER_VALIDATION_MAX_MS", 5)
    ALLOWED_MIME_TYPES: list[str] = _list("ALLOWED_MIME_TYPES", [])
    ALLOWED_FILE_TYPES: list[str] = _list("ALLOWED_FILE_TYPES", [])
    MIN_FREE_DISK_MB: int = _int("MIN_FREE_DISK_MB", 500)
    STRIP_NULL_BYTES: bool = _bool("STRIP_NULL_BYTES", True)
    STRIP_BOM: bool = _bool("STRIP_BOM", True)
    MAGIC_BYTE_MIME_DETECTION: bool = _bool("MAGIC_BYTE_MIME_DETECTION", True)

    # DEDUP AND SECURITY
    DEDUP_ENABLED: bool = _bool("DEDUP_ENABLED", True)
    CLAMAV_ENABLED: bool = _bool("CLAMAV_ENABLED", False)
    CLAMAV_HOST: str = _str("CLAMAV_HOST", "localhost")
    CLAMAV_PORT: int = _int("CLAMAV_PORT", 3310)
    TEMP_FILE_ENCRYPTION: bool = _bool("TEMP_FILE_ENCRYPTION", False)

    # HALLUCINATION
    HALLUCINATION_THRESHOLD: float = _float("HALLUCINATION_THRESHOLD", 0.5)

    # CROSS-MODAL FUSION
    CROSS_MODAL_FUSION_TIMEOUT: float = _float("CROSS_MODAL_FUSION_TIMEOUT", 5.0)
    CROSS_MODAL_MIN_CONFIDENCE: float = _float("CROSS_MODAL_MIN_CONFIDENCE", 0.1)
    CROSS_MODAL_MAX_CHUNKS_PER_MODALITY: int = _int("CROSS_MODAL_MAX_CHUNKS_PER_MODALITY", 5)

    # LATENCY TARGETS
    LATENCY_TARGET_IMAGE_MS: int = _int("LATENCY_TARGET_IMAGE_MS", 3000)
    LATENCY_TARGET_AUDIO_RTF: float = _float("LATENCY_TARGET_AUDIO_RTF", 0.5)
    LATENCY_TARGET_PDF_MS: int = _int("LATENCY_TARGET_PDF_MS", 10_000)
    LATENCY_TARGET_CROSS_MODAL_MS: int = _int("LATENCY_TARGET_CROSS_MODAL_MS", 5000)
    LATENCY_TARGET_PREFLIGHT_MS: int = _int("LATENCY_TARGET_PREFLIGHT_MS", 50)
    LATENCY_TARGET_CACHE_HIT_MS: int = _int("LATENCY_TARGET_CACHE_HIT_MS", 100)
    LATENCY_TARGET_EMBED_BATCH_MS: int = _int("LATENCY_TARGET_EMBED_BATCH_MS", 2000)

    # SLO TARGETS
    SLO_TEXT_P95_MS: int = _int("SLO_TEXT_P95_MS", 2000)
    SLO_PDF_P95_MS: int = _int("SLO_PDF_P95_MS", 30000)
    SLO_WORD_P95_MS: int = _int("SLO_WORD_P95_MS", 15000)
    SLO_EXCEL_P95_MS: int = _int("SLO_EXCEL_P95_MS", 45000)
    SLO_IMAGE_P95_MS: int = _int("SLO_IMAGE_P95_MS", 10000)
    SLO_AUDIO_P95_MS: int = _int("SLO_AUDIO_P95_MS", 300000)
    SLO_VIDEO_P95_MS: int = _int("SLO_VIDEO_P95_MS", 900000)

    # MULTI-USER
    DEFAULT_DEV_USER_ID: str = _str("DEFAULT_DEV_USER_ID", "a3f9c821-4b2e-11ef-9454-0242ac120002")
    TEST_USER_2_ID: str = _str("TEST_USER_2_ID", "b7d2e109-4b2e-11ef-9454-0242ac120002")

    # JWT AUTHENTICATION — Phase 27
    JWT_SECRET_KEY: str = _str("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")
    JWT_ALGORITHM: str = _str("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = _int("ACCESS_TOKEN_EXPIRE_MINUTES", 30)
    REFRESH_TOKEN_EXPIRE_DAYS: int = _int("REFRESH_TOKEN_EXPIRE_DAYS", 7)
    AUTH_COLLECTION: str = _str("AUTH_COLLECTION", "users")
    PASSWORD_MIN_ZXCVBN_SCORE: int = _int("PASSWORD_MIN_ZXCVBN_SCORE", 2)
    PASSWORD_MIN_LENGTH: int = _int("PASSWORD_MIN_LENGTH", 8)
    AUTH_ENABLED: bool = _bool("AUTH_ENABLED", True)
    # httpOnly cookie attributes for the browser session (access/refresh/device
    # tokens are never exposed to JS — see app/auth/cookies.py). Secure defaults
    # to on outside local dev so cookies are never sent over plain HTTP in any
    # real deployment; SameSite=Lax blocks cross-site form/script forgery while
    # still surviving the Google OAuth redirect landing on our own callback.
    COOKIE_SECURE: bool = _bool("COOKIE_SECURE", ENV != "development")
    COOKIE_SAMESITE: str = _str("COOKIE_SAMESITE", "lax")
    COOKIE_DOMAIN: str | None = _opt("COOKIE_DOMAIN")  # None = host-only cookie
    # The one fixed, publicly-shared demo login (recruiters / hiring managers).
    # /auth/login skips the email OTP for this address — see app/auth/router.py.
    # Matching on the address, not only on the Mongo `is_demo` flag, is
    # deliberate: the flag is a database write that only
    # `python -m app.bin.seed_demo_account` performs, so a fresh environment
    # (or an account that predated the flag) would silently fall back to
    # asking a recruiter for an OTP they can't receive. Set to "" to disable
    # the bypass entirely.
    DEMO_ACCOUNT_EMAIL: str = _str("DEMO_ACCOUNT_EMAIL", "testuser@ragdev.local")

    # EMAIL (Gmail SMTP) — OTP + password-reset
    SMTP_HOST: str = _str("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = _int("SMTP_PORT", 587)
    SMTP_USER: str | None = _opt("SMTP_USER")
    SMTP_PASSWORD: str | None = _opt("SMTP_PASSWORD")
    EMAIL_FROM: str = _str("EMAIL_FROM", "magikaiassistant@gmail.com")
    EMAIL_FROM_NAME: str = _str("EMAIL_FROM_NAME", "MAGIK AI")
    FRONTEND_URL: str = _str("FRONTEND_URL", "http://localhost:5173")

    # OTP / password-reset tuning
    OTP_TTL_SECONDS: int = _int("OTP_TTL_SECONDS", 600)  # 10 min
    OTP_MAX_ATTEMPTS: int = _int("OTP_MAX_ATTEMPTS", 3)
    OTP_LOCKOUT_SECONDS: int = _int("OTP_LOCKOUT_SECONDS", 900)  # 15 min
    OTP_RESEND_COOLDOWN_SECONDS: int = _int(
        "OTP_RESEND_COOLDOWN_SECONDS", 45
    )  # min gap between resends
    OTP_RESEND_MAX_PER_WINDOW: int = _int("OTP_RESEND_MAX_PER_WINDOW", 5)  # resend cap per window
    OTP_RESEND_WINDOW_SECONDS: int = _int("OTP_RESEND_WINDOW_SECONDS", 3600)  # 1 hr resend window
    RESET_TOKEN_TTL_SECONDS: int = _int("RESET_TOKEN_TTL_SECONDS", 3600)  # 1 hr
    # Dev-only: log OTP/reset links to console instead of sending email
    DEV_OTP_LOG: bool = _bool("DEV_OTP_LOG", False)

    # SECURITY
    SECRET_KEY: str = _str("SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")
    DATABASE_URL: str = _str("DATABASE_URL", "sqlite:///./data/rag_users.db")
    # No wildcard default: CORSMiddleware runs with allow_credentials=True (see
    # main.py) so the auth cookies ride every allowed cross-origin request —
    # Starlette's CORS middleware reflects the request's Origin verbatim
    # instead of literally sending "*" whenever allow_credentials=True, so an
    # unset/"*" value here would silently accept credentialed requests from
    # ANY origin. Must be set explicitly per environment (see .env.example).
    CORS_ORIGINS: list[str] = _list("CORS_ORIGINS", [])
    RATE_LIMIT_RPM: int = _int("RATE_LIMIT_RPM", 60)
    # Comma-separated list of trusted reverse-proxy IPs (e.g. an internal ALB).
    # X-Forwarded-For is ONLY trusted when the direct TCP peer is in this list —
    # otherwise it's attacker-controlled and rate limiting keys on it would be
    # trivially bypassable. Empty by default: no proxy is trusted, so the
    # direct connection IP is always used.
    TRUSTED_PROXY_IPS: list[str] = _list("TRUSTED_PROXY_IPS", [])
    AUTH_LOGIN_RATE_LIMIT_PER_MIN: int = _int("AUTH_LOGIN_RATE_LIMIT_PER_MIN", 5)
    AUTH_REGISTER_RATE_LIMIT_PER_HOUR: int = _int("AUTH_REGISTER_RATE_LIMIT_PER_HOUR", 3)
    PII_DETECTION_ENABLED: bool = _bool("PII_DETECTION_ENABLED", False)
    PII_ENTITIES: list[str] = _list(
        "PII_ENTITIES", ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD"]
    )
    PII_REDACTION_OPERATOR: str = _str("PII_REDACTION_OPERATOR", "replace")
    PROMPT_INJECTION_FILTER_ENABLED: bool = _bool("PROMPT_INJECTION_FILTER_ENABLED", False)
    AUDIT_LOG_ENABLED: bool = _bool("AUDIT_LOG_ENABLED", True)
    SSRF_BLOCK_PRIVATE_IPS: bool = _bool("SSRF_BLOCK_PRIVATE_IPS", True)

    # DOCUMENT PROCESSING
    PDF_OCR_TEXT_DENSITY_THRESHOLD: float = _float("PDF_OCR_TEXT_DENSITY_THRESHOLD", 0.1)
    # Supplemental OCR re-runs Tesseract on pages that already have a text layer but
    # low text density (tables, sparse pages). Off by default — it's slow on filings
    # like 10-Ks and pages with text rarely need it. Pages with NO text are always OCR'd.
    PDF_SUPPLEMENTAL_OCR_ENABLED: bool = _bool("PDF_SUPPLEMENTAL_OCR_ENABLED", False)
    LIBREOFFICE_ENABLED: bool = _bool("LIBREOFFICE_ENABLED", True)
    LIBREOFFICE_PATH: str = _str("LIBREOFFICE_PATH", "libreoffice")
    LIBREOFFICE_TIMEOUT_SEC: int = _int("LIBREOFFICE_TIMEOUT_SEC", 60)
    PDF_SIGNATURE_DETECTION: bool = _bool("PDF_SIGNATURE_DETECTION", True)
    PDF_STRIP_JAVASCRIPT: bool = _bool("PDF_STRIP_JAVASCRIPT", True)
    PDF_FORM_EXTRACTION: bool = _bool("PDF_FORM_EXTRACTION", True)
    WORD_TRACK_CHANGES: bool = _bool("WORD_TRACK_CHANGES", True)
    WORD_MACRO_DETECTION: bool = _bool("WORD_MACRO_DETECTION", True)

    # EXCEL PROCESSING
    EXCEL_STREAM_READ_ONLY: bool = _bool("EXCEL_STREAM_READ_ONLY", True)
    EXCEL_MAX_ROWS: int = _int("EXCEL_MAX_ROWS", 1_000_000)

    # KEYWORD EXTRACTION
    KEYWORD_EXTRACTOR: str = _str("KEYWORD_EXTRACTOR", "yake")
    KEYWORD_MAX_PER_CHUNK: int = _int("KEYWORD_MAX_PER_CHUNK", 10)
    KEYBERT_MODEL: str = _str("KEYBERT_MODEL", "all-MiniLM-L6-v2")

    # DSA LAYER
    LRU_CACHE_MAXSIZE: int = _int("LRU_CACHE_MAXSIZE", 1000)
    PRIORITY_QUEUE_TOP_K: int = _int("PRIORITY_QUEUE_TOP_K", 5)
    INVERTED_INDEX_MAX_VOCAB: int = _int("INVERTED_INDEX_MAX_VOCAB", 50_000)

    # TEST FIXTURES
    TEST_AUDIO_SAMPLE_RATE: int = _int("TEST_AUDIO_SAMPLE_RATE", 44100)
    TEST_AUDIO_DURATION_SEC: int = _int("TEST_AUDIO_DURATION_SEC", 10)
    TEST_IMAGE_WIDTH: int = _int("TEST_IMAGE_WIDTH", 800)
    TEST_IMAGE_HEIGHT: int = _int("TEST_IMAGE_HEIGHT", 600)

    # DIRECTORY CREATION
    def _create_directories(self) -> None:
        dirs = [
            self.DATA_DIR,
            self.LOG_DIR,
            self.TEST_FIXTURES_DIR,
            self.AUDIT_LOG_PATH.parent,
        ]
        for d in dirs:
            try:
                d.mkdir(parents=True, exist_ok=True)
            except PermissionError as exc:
                print(f"[config] WARNING: Cannot create directory {d}: {exc}", file=sys.stderr)

    # VALIDATION
    def validate(self) -> bool:
        errors: list[str] = []

        if not self.LLM_MODEL_PATH:
            errors.append("LLM_MODEL_PATH is required")

        if self.LLM_MAX_TOKENS > self.CONTEXT_MAX_TOKENS:
            errors.append(
                f"LLM_MAX_TOKENS ({self.LLM_MAX_TOKENS}) cannot exceed "
                f"CONTEXT_MAX_TOKENS ({self.CONTEXT_MAX_TOKENS})"
            )

        if self.LLM_THREADS < 1:
            errors.append("LLM_THREADS must be >= 1")

        if self.TEXT_EMBEDDING_DIM <= 0:
            errors.append(f"TEXT_EMBEDDING_DIM must be > 0, got {self.TEXT_EMBEDDING_DIM}")

        if self.VISION_EMBEDDING_DIM <= 0:
            errors.append(f"VISION_EMBEDDING_DIM must be > 0, got {self.VISION_EMBEDDING_DIM}")

        if self.EMBEDDING_BATCH_SIZE <= 0:
            errors.append(f"EMBEDDING_BATCH_SIZE must be > 0, got {self.EMBEDDING_BATCH_SIZE}")

        if not self.RERANKER_MODEL:
            errors.append("RERANKER_MODEL is required")

        if not self.QDRANT_URL and not self.QDRANT_HOST:
            errors.append("Either QDRANT_URL or QDRANT_HOST must be set")

        if self.USE_MONGO and not self.MONGO_URI:
            errors.append("MONGO_URI is required when USE_MONGO=true")

        if self.USE_REDIS and not self.REDIS_URL and not self.REDIS_HOST:
            errors.append("Either REDIS_URL or REDIS_HOST must be set")

        if self.MAX_CHUNKS <= 0:
            errors.append(f"MAX_CHUNKS must be > 0, got {self.MAX_CHUNKS}")

        if self.INGESTION_BATCH_SIZE <= 0:
            errors.append(f"INGESTION_BATCH_SIZE must be > 0, got {self.INGESTION_BATCH_SIZE}")

        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            errors.append(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be < CHUNK_SIZE ({self.CHUNK_SIZE})"
            )

        if self.WEB_MAX_DOCS <= 0:
            errors.append(f"WEB_MAX_DOCS must be > 0, got {self.WEB_MAX_DOCS}")

        if self.MEMORY_TOP_K <= 0:
            errors.append(f"MEMORY_TOP_K must be > 0, got {self.MEMORY_TOP_K}")

        if self.RAG_TOP_K <= 0:
            errors.append(f"RAG_TOP_K must be > 0, got {self.RAG_TOP_K}")

        if self.AGENT_MAX_STEPS <= 0:
            errors.append(f"AGENT_MAX_STEPS must be > 0, got {self.AGENT_MAX_STEPS}")

        if self.AGENT_VERIFY_MAX_RETRIES < 0:
            errors.append(
                f"AGENT_VERIFY_MAX_RETRIES must be >= 0, got {self.AGENT_VERIFY_MAX_RETRIES}"
            )

        if self.AGENT_VERIFY_TIMEOUT_SEC <= 0:
            errors.append(
                f"AGENT_VERIFY_TIMEOUT_SEC must be > 0, got {self.AGENT_VERIFY_TIMEOUT_SEC}"
            )

        for _name, _val in (
            ("AGENT_VERIFY_RETRIEVAL_MIN", self.AGENT_VERIFY_RETRIEVAL_MIN),
            ("AGENT_VERIFY_GROUNDING_MIN", self.AGENT_VERIFY_GROUNDING_MIN),
            ("AGENT_VERIFY_CITATION_MIN", self.AGENT_VERIFY_CITATION_MIN),
            ("AGENT_VERIFY_OVERALL_MIN", self.AGENT_VERIFY_OVERALL_MIN),
        ):
            if not (0.0 <= _val <= 100.0):
                errors.append(f"{_name} must be within 0-100, got {_val}")

        fusion_sum = (
            self.FUSION_SCORE_WEIGHT + self.FUSION_QUALITY_WEIGHT + self.FUSION_MODALITY_WEIGHT
        )
        if not (0.95 <= fusion_sum <= 1.05):
            errors.append(f"FUSION weights must sum to ~1.0, got {fusion_sum:.3f}")

        if self.ENV != "development" and self.SECRET_KEY == "CHANGE_ME_IN_PRODUCTION":
            errors.append("SECRET_KEY must be changed from default in non-development environments")

        if self.ENV != "development" and self.JWT_SECRET_KEY == "CHANGE_ME_IN_PRODUCTION":
            errors.append(
                "JWT_SECRET_KEY must be changed from default in non-development environments"
            )

        if self.AUTH_ENABLED and len(self.JWT_SECRET_KEY) < 32:
            errors.append("JWT_SECRET_KEY must be at least 32 characters long")

        if self.CROSS_MODAL_FUSION_TIMEOUT <= 0:
            errors.append("CROSS_MODAL_FUSION_TIMEOUT must be > 0")

        if self.CHUNK_MINHASH_THRESHOLD < 0.0 or self.CHUNK_MINHASH_THRESHOLD > 1.0:
            errors.append(
                f"CHUNK_MINHASH_THRESHOLD must be 0.0–1.0, got {self.CHUNK_MINHASH_THRESHOLD}"
            )

        if self.OTEL_SAMPLING_RATIO < 0.0 or self.OTEL_SAMPLING_RATIO > 1.0:
            errors.append(f"OTEL_SAMPLING_RATIO must be 0.0–1.0, got {self.OTEL_SAMPLING_RATIO}")

        if errors:
            formatted = "\n  - ".join(errors)
            raise ValueError(
                f"Config validation failed with {len(errors)} error(s):\n  - {formatted}"
            )

        return True

    def get_service_status(self) -> dict[str, bool]:
        return {
            "redis": self.USE_REDIS,
            "mongo": self.USE_MONGO,
            "qdrant": bool(self.QDRANT_URL or self.QDRANT_HOST),
        }


# SINGLETON


def _build_settings() -> Settings:
    s = Settings()
    s._create_directories()
    s.validate()
    return s


settings: Settings = _build_settings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return settings


# TESTS

# ============================================================
# TESTS — Phase 24 Upgrade
# Run: pytest app/core/config.py -v
# ============================================================


class TestSettings:

    def test_defaults_are_valid(self):
        s = Settings()
        assert s.APP_VERSION == "0.30.0"
        assert s.TEXT_EMBEDDING_DIM == 1024
        assert s.VISION_EMBEDDING_DIM > 0
        assert s.CHUNK_OVERLAP < s.CHUNK_SIZE
        assert s.AGENT_MAX_STEPS > 0
        assert s.RAG_TOP_K > 0
        assert s.MEMORY_TOP_K > 0

    def test_fusion_weights_sum_to_one(self):
        s = Settings()
        total = s.FUSION_SCORE_WEIGHT + s.FUSION_QUALITY_WEIGHT + s.FUSION_MODALITY_WEIGHT
        assert 0.95 <= total <= 1.05

    def test_file_size_limits_property(self):
        s = Settings()
        limits = s.FILE_SIZE_LIMITS
        assert "text" in limits
        assert "pdf" in limits
        assert "video" in limits
        assert limits["video"] > limits["audio"]
        assert limits["pdf"] > limits["text"]

    def test_chunk_overlap_less_than_chunk_size(self):
        s = Settings()
        assert s.CHUNK_OVERLAP < s.CHUNK_SIZE

    def test_llm_max_tokens_within_context(self):
        s = Settings()
        assert s.LLM_MAX_TOKENS <= s.CONTEXT_MAX_TOKENS

    def test_new_phase24_fields_present(self):
        s = Settings()
        assert hasattr(s, "ASYNC_SEMAPHORE_WORKERS")
        assert hasattr(s, "OTEL_ENABLED")
        assert hasattr(s, "DEDUP_ENABLED")
        assert hasattr(s, "MAGIC_BYTE_MIME_DETECTION")
        assert hasattr(s, "CHUNK_MINHASH_THRESHOLD")
        assert hasattr(s, "HYBRID_RRF_K")
        assert hasattr(s, "QDRANT_SOFT_DELETE")
        assert hasattr(s, "REDIS_EMBEDDING_CACHE_TTL")
        assert hasattr(s, "MONGO_CB_FAIL_MAX")
        assert hasattr(s, "PII_ENTITIES")
        assert hasattr(s, "SSRF_BLOCK_PRIVATE_IPS")
        assert hasattr(s, "LIBREOFFICE_ENABLED")
        assert hasattr(s, "VIDEO_DUPLICATE_FRAME_THRESHOLD")

    def test_circuit_breaker_defaults(self):
        s = Settings()
        assert s.QDRANT_CB_FAIL_MAX == 5
        assert s.REDIS_CB_FAIL_MAX == 5
        assert s.MONGO_CB_FAIL_MAX == 5
        assert s.QDRANT_CB_RESET_TIMEOUT > 0
        assert s.REDIS_CB_RESET_TIMEOUT > 0

    def test_otel_sampling_ratio_range(self):
        s = Settings()
        assert 0.0 <= s.OTEL_SAMPLING_RATIO <= 1.0

    def test_minhash_threshold_range(self):
        s = Settings()
        assert 0.0 <= s.CHUNK_MINHASH_THRESHOLD <= 1.0

    def test_env_development_secret_key_allowed(self):
        s = Settings()
        assert s.ENV == "development"

    def test_paths_are_path_objects(self):
        s = Settings()
        assert isinstance(s.DATA_DIR, Path)
        assert isinstance(s.LOG_DIR, Path)
        assert isinstance(s.TEMP_DIR, Path)
        assert isinstance(s.AUDIT_LOG_PATH, Path)
        assert isinstance(s.CHROOT_BASE, Path)

    def test_whisper_filler_words_list(self):
        s = Settings()
        assert isinstance(s.WHISPER_FILLER_WORDS, list)
        assert "uh" in s.WHISPER_FILLER_WORDS

    def test_pii_entities_list(self):
        s = Settings()
        assert isinstance(s.PII_ENTITIES, list)
        assert "EMAIL_ADDRESS" in s.PII_ENTITIES

    def test_slo_targets_positive(self):
        s = Settings()
        assert s.SLO_TEXT_P95_MS > 0
        assert s.SLO_PDF_P95_MS > 0
        assert s.SLO_VIDEO_P95_MS > s.SLO_TEXT_P95_MS

    def test_validate_raises_on_invalid_chunk_overlap(self):
        s = Settings()
        original = s.CHUNK_OVERLAP
        s.CHUNK_OVERLAP = s.CHUNK_SIZE + 10
        with pytest.raises(ValueError, match="CHUNK_OVERLAP"):
            s.validate()
        s.CHUNK_OVERLAP = original

    def test_validate_raises_on_bad_fusion_weights(self):
        s = Settings()
        original = s.FUSION_SCORE_WEIGHT
        s.FUSION_SCORE_WEIGHT = 0.9
        with pytest.raises(ValueError, match="FUSION weights"):
            s.validate()
        s.FUSION_SCORE_WEIGHT = original

    def test_get_settings_cached(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
