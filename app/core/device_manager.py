"""
Central device + dtype policy for the model layer.

L40S 48 GB VRAM deploy target (AWS g6e.xlarge).

GPU models (all_gpu profile) — ALL models run on GPU:
  - Mistral 7B Q4_K_M          ~4.10 GB  (llama-cpp n_gpu_layers=-1)
  - SigLIP SO400M/14@384px     ~1.76 GB  (float16, 878M params)
  - BLIP large                 ~1.56 GB  (float16, 779M params, captioning)
  - Qwen2-VL-2B-Instruct       ~2.20 GB  (int8, video/chart captioning)
  - Whisper large-v3           ~1.55 GB  (int8_float16)
  - BGE-large-en-v1.5          ~1.35 GB  (float16)
  - BGE-reranker-large         ~1.34 GB  (float16)
  - TrOCR large-printed        ~1.40 GB  (float16)
  Total static ≈ 15.3 GB  — fits in 48 GB with ~32.7 GB headroom
  Peak (LLM KV cache 8192 ctx): ~18 GB
  (figures above predate the Qwen2.5-14B/Qwen2-VL-7B upgrades documented in
  CHANGELOG.md — kept as originally measured; the 48 GB budget has enough
  margin that the exact static total isn't the binding constraint here.)

CPU services (no VRAM needed — not models):
  - FastAPI / Uvicorn
  - Qdrant client (network I/O only)
  - Redis client (network I/O only)
  - MongoDB client (network I/O only)
  - BM25 (in-memory, CPU-bound)

Profiles
--------
- auto      → all_gpu on CUDA, all_cpu otherwise.
- hybrid    → LLM / SigLIP / BLIP / Whisper on CUDA; text embedder + reranker on CPU.
- all_gpu   → every model on CUDA.
- all_cpu   → every model on CPU.

Each per-model env var (EMBEDDER_DEVICE, SIGLIP_DEVICE, …) takes precedence
over the profile when set.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# Models the loader can place on a device. Names match get_<x> methods in ModelLoader.
MODEL_NAMES = (
    "llm",
    "text_embedder",
    "siglip",
    "blip",
    "qwen2_vl",
    "trocr",
    "diarizer",
    "ner",
    "whisper",
    "reranker",
)

# Hybrid profile: heavy attention models on GPU, tiny models on CPU.
_HYBRID_MAP: dict[str, str] = {
    "llm": "cuda",
    "siglip": "cuda",
    "blip": "cuda",
    "qwen2_vl": "cuda",
    "trocr": "cuda",
    "diarizer": "cuda",
    "ner": "cuda",
    "whisper": "cuda",
    "text_embedder": "cpu",
    "reranker": "cpu",
}

_ALL_GPU_MAP: dict[str, str] = {name: "cuda" for name in MODEL_NAMES}
_ALL_CPU_MAP: dict[str, str] = {name: "cpu" for name in MODEL_NAMES}


@dataclass
class DeviceDecision:
    device: str  # "cuda" | "cpu" | "mps"
    dtype: str | None  # "float16" | "float32" | "int8" | None
    notes: str = ""


class DeviceManager:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._torch = None
        self._cuda_ok = False
        self._mps_ok = False
        self._vram_total_gb = 0.0
        self._profile = self._resolve_profile(settings.MODELS_DEVICE_PROFILE)
        self._probe()

        logger.info(
            event="device_manager_initialized",
            profile=self._profile,
            cuda=self._cuda_ok,
            mps=self._mps_ok,
            vram_total_gb=self._vram_total_gb,
            vram_budget_gb=settings.VRAM_BUDGET_GB,
        )

    # PROBE TORCH ONCE — single source of truth for CUDA/MPS availability

    def _probe(self) -> None:
        try:
            import torch  # noqa: F401

            self._torch = torch
            self._cuda_ok = bool(torch.cuda.is_available())
            self._mps_ok = bool(
                getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            )
            if self._cuda_ok:
                try:
                    props = torch.cuda.get_device_properties(0)
                    self._vram_total_gb = round(props.total_memory / (1024**3), 2)
                except Exception:
                    self._vram_total_gb = 0.0
        except Exception:
            self._torch = None

    def _resolve_profile(self, raw: str) -> str:
        profile = (raw or "auto").strip().lower()
        if profile not in ("auto", "hybrid", "all_gpu", "all_cpu"):
            logger.warning(event="device_profile_unknown", got=profile)
            profile = "auto"
        return profile

    # CHOSEN PROFILE — auto resolves at first call

    @property
    def profile(self) -> str:
        if self._profile == "auto":
            # On CUDA hosts default to all_gpu (L40S 48GB has ample VRAM)
            return "all_gpu" if self._cuda_ok else "all_cpu"
        if self._profile in ("hybrid", "all_gpu") and not self._cuda_ok:
            return "all_cpu"
        return self._profile

    @property
    def cuda_available(self) -> bool:
        return self._cuda_ok

    @property
    def vram_total_gb(self) -> float:
        return self._vram_total_gb

    def free_vram_gb(self) -> float | None:
        """Real free VRAM in GB, as reported by the CUDA driver.

        Returns None when CUDA is unavailable or the query fails, so callers
        can distinguish "no GPU / can't tell" from a genuine low reading and
        decline to act rather than acting on a fabricated 0.0.

        USE THIS, NOT `total_memory - torch.cuda.memory_reserved()`. That
        expression counts only the CALLING process's PyTorch caching
        allocator, which on this deployment is a small minority of what is
        actually resident on the card. It cannot see:
          * the separate llama-server process holding the 14B GGUF
            (app/llm/gguf_model.py's LLM_USE_SERVER path),
          * the eval judge's own llama.cpp worker process
            (app/eval/judges/qwen_judge_worker.py),
          * llama.cpp allocations or non-PyTorch CUDA context overhead.
        `torch.cuda.mem_get_info()` asks the driver and therefore accounts for
        every process on the device. app/eval/judges/qwen_judge.py had a
        private copy of exactly this logic (its CPU-fallback guard was
        silently inert because of the allocator-only version); it now calls
        here instead, so there is one implementation.
        """
        if not self._cuda_ok or self._torch is None:
            return None
        try:
            free_bytes, _total = self._torch.cuda.mem_get_info()
            return free_bytes / (1024**3)
        except Exception:
            return None

    # PER-MODEL DEVICE — per-model override > profile map > fallback

    def device_for(self, name: str) -> str:
        name = name.lower()

        override = self._override_for(name)
        if override:
            return self._normalize(override, name)

        profile_map = self._profile_map()
        device = profile_map.get(name, "cpu")
        device = self._normalize(device, name)

        # ENABLE flags force-disable a model; we keep its device for completeness
        # but the model_loader will short-circuit the actual load.
        return device

    def _override_for(self, name: str) -> str:
        overrides = {
            "llm": settings.LLM_DEVICE_HINT,
            "text_embedder": settings.EMBEDDER_DEVICE,
            "reranker": settings.RERANKER_DEVICE,
            "siglip": settings.SIGLIP_DEVICE,
            "blip": settings.BLIP_DEVICE,
            "qwen2_vl": settings.QWEN2_VL_DEVICE,
            "trocr": settings.TROCR_DEVICE,
            "diarizer": settings.DIARIZER_DEVICE,
            "ner": settings.NER_DEVICE,
            "whisper": settings.WHISPER_DEVICE,
        }
        return (overrides.get(name) or "").strip().lower()

    def _profile_map(self) -> dict[str, str]:
        p = self.profile
        if p == "all_gpu":
            return _ALL_GPU_MAP
        if p == "all_cpu":
            return _ALL_CPU_MAP
        return _HYBRID_MAP

    def _normalize(self, requested: str, name: str) -> str:
        """Demote a request to what the host actually supports."""
        req = (requested or "").strip().lower()
        if req in ("cuda", "gpu"):
            if not self._cuda_ok:
                logger.warning(
                    event="device_demoted_to_cpu",
                    model=name,
                    reason="cuda_unavailable",
                )
                return "cpu"
            return "cuda"
        if req == "mps":
            return "mps" if self._mps_ok else "cpu"
        return "cpu"

    # PER-MODEL DTYPE — fp16 on CUDA for vision/audio; respect EMBEDDER_HALF_PRECISION

    def dtype_for(self, name: str, device: str | None = None) -> str | None:
        name = name.lower()
        dev = device or self.device_for(name)

        if dev == "cpu":
            # Faster-whisper on CPU is fastest with int8 quantization
            if name == "whisper":
                return "int8"
            return None  # let HF default (fp32) on CPU

        if dev == "cuda":
            if name == "whisper":
                ct = (settings.WHISPER_COMPUTE_TYPE or "").strip().lower()
                return ct or "float16"
            if name in ("siglip", "blip", "trocr", "ner") and settings.VISION_HALF_PRECISION:
                return "float16"
            if name == "qwen2_vl":
                # 8-bit load handled by get_qwen2_vl — dtype not used directly
                return "float16"
            if name == "diarizer":
                return "float32"  # pyannote does its own dtype management
            if name == "text_embedder" and settings.EMBEDDER_HALF_PRECISION:
                return "float16"
            if name == "reranker":
                return "float16"
            # LLM dtype is irrelevant — llama.cpp uses GGUF quant internally
            return "float32"

        if dev == "mps":
            return "float32"

        return None

    # FULL DECISION — used by loaders that want device + dtype + notes together

    def decision_for(self, name: str) -> DeviceDecision:
        dev = self.device_for(name)
        dtype = self.dtype_for(name, dev)
        return DeviceDecision(device=dev, dtype=dtype, notes=self.profile)

    # GGUF n_gpu_layers — auto-pick based on actual free VRAM at load time.
    # Mistral Q4_K_M has 32 transformer blocks; each costs ~128 MB of VRAM.
    # We reserve 512 MB headroom so other models (embedder, SigLIP, Whisper) don't OOM.
    _LAYERS_PER_GB = 7.5  # ~128 MB per layer → ~7.5 layers per GB
    _VRAM_HEADROOM_GB = 2.0  # reserve 2 GB for non-LLM models (Qwen3+BGE+SigLIP+BLIP+Whisper)

    def llm_gpu_layers(self) -> int:
        llm_device = self.device_for("llm")
        if llm_device != "cuda":
            return 0
        if not settings.LLM_GPU_LAYERS_AUTO:
            return settings.LLM_GPU_LAYERS

        # Query actual free VRAM right now (other models may already be loaded).
        try:
            import torch

            free_bytes = torch.cuda.get_device_properties(
                0
            ).total_memory - torch.cuda.memory_reserved(0)
            free_gb = free_bytes / (1024**3)
        except Exception:
            # Can't query VRAM — fall back to full offload and let llama_cpp try.
            return settings.LLM_GPU_LAYERS_ALL

        usable_gb = max(0.0, min(free_gb, settings.VRAM_BUDGET_GB) - self._VRAM_HEADROOM_GB)
        layers = int(usable_gb * self._LAYERS_PER_GB)

        if layers >= 32:
            # All layers fit — use -1 so llama_cpp offloads embedding table too.
            return settings.LLM_GPU_LAYERS_ALL  # -1

        import logging as _logging

        _logging.getLogger(__name__).info(
            "llm_gpu_layers_partial",
            extra={
                "free_gb": round(free_gb, 2),
                "usable_gb": round(usable_gb, 2),
                "layers": layers,
            },
        )
        return max(0, layers)

    # SNAPSHOT — for /ready and logging

    def snapshot(self) -> dict[str, dict[str, str | None]]:
        out: dict[str, dict[str, str | None]] = {
            "_summary": {
                "profile": self.profile,
                "cuda": str(self._cuda_ok),
                "mps": str(self._mps_ok),
                "vram_total_gb": str(self._vram_total_gb),
                "vram_budget_gb": str(settings.VRAM_BUDGET_GB),
            },
        }
        for name in MODEL_NAMES:
            d = self.decision_for(name)
            out[name] = {"device": d.device, "dtype": d.dtype}
        return out

    # CACHE PURGE — call this when /reset is requested

    def empty_cuda_cache(self) -> None:
        if self._torch is None or not self._cuda_ok:
            return
        try:
            self._torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning(event="cuda_cache_empty_failed", error=str(exc))


# SINGLETON

device_manager = DeviceManager()
