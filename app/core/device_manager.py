"""
Central device + dtype policy for the model layer.

Tesla T4 deploy budget: 14.6 GB VRAM, CUDA 13.2.

GPU models (all_gpu profile):
  - Mistral 7B Q4_K_M  ~4.1 GB  (llama-cpp n_gpu_layers=-1)
  - CLIP ViT-B/32       ~0.6 GB  (float16)
  - BLIP base           ~1.0 GB  (float16)
  - Whisper large-v3    ~1.5 GB  (float16)
  - CrossEncoder MiniLM ~0.1 GB  (float16)
  - MiniLM text embed   ~0.09 GB (float16)
  Total ≈ 7.4 GB  — fits comfortably in 14.6 GB

CPU services (no VRAM needed):
  - FastAPI / Uvicorn
  - Qdrant client (network I/O only)
  - Redis client (network I/O only)
  - MongoDB client (network I/O only)
  - BM25 (in-memory, CPU-bound)

Profiles
--------
- auto      → all_gpu on CUDA, all_cpu otherwise.
- hybrid    → LLM / CLIP / BLIP / Whisper on CUDA; text embedder + reranker on CPU.
- all_gpu   → every model on CUDA.
- all_cpu   → every model on CPU.

Each per-model env var (EMBEDDER_DEVICE, CLIP_DEVICE, …) takes precedence
over the profile when set.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# Models the loader can place on a device. Names match get_<x> methods in ModelLoader.
MODEL_NAMES = (
    "llm",
    "text_embedder",
    "clip",
    "blip",
    "whisper",
    "reranker",
)

# Hybrid profile: heavy attention models on GPU, tiny models on CPU.
_HYBRID_MAP: Dict[str, str] = {
    "llm":            "cuda",
    "clip":           "cuda",
    "blip":           "cuda",
    "whisper":        "cuda",
    "text_embedder":  "cpu",
    "reranker":       "cpu",
}

_ALL_GPU_MAP: Dict[str, str] = {name: "cuda" for name in MODEL_NAMES}
_ALL_CPU_MAP: Dict[str, str] = {name: "cpu"  for name in MODEL_NAMES}


@dataclass
class DeviceDecision:
    device: str            # "cuda" | "cpu" | "mps"
    dtype:  Optional[str]  # "float16" | "float32" | "int8" | None
    notes:  str = ""


class DeviceManager:

    def __init__(self) -> None:
        self._lock          = threading.Lock()
        self._torch         = None
        self._cuda_ok       = False
        self._mps_ok        = False
        self._vram_total_gb = 0.0
        self._profile       = self._resolve_profile(settings.MODELS_DEVICE_PROFILE)
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
            self._torch  = torch
            self._cuda_ok = bool(torch.cuda.is_available())
            self._mps_ok  = bool(
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            )
            if self._cuda_ok:
                try:
                    props = torch.cuda.get_device_properties(0)
                    self._vram_total_gb = round(props.total_memory / (1024 ** 3), 2)
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
            # On CUDA hosts default to all_gpu (Tesla T4 has plenty of VRAM)
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
            "llm":           settings.LLM_DEVICE_HINT,
            "text_embedder": settings.EMBEDDER_DEVICE,
            "reranker":      settings.RERANKER_DEVICE,
            "clip":          settings.CLIP_DEVICE,
            "blip":          settings.BLIP_DEVICE,
            "whisper":       settings.WHISPER_DEVICE,
        }
        return (overrides.get(name) or "").strip().lower()

    def _profile_map(self) -> Dict[str, str]:
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

    def dtype_for(self, name: str, device: Optional[str] = None) -> Optional[str]:
        name = name.lower()
        dev  = device or self.device_for(name)

        if dev == "cpu":
            # Faster-whisper on CPU is fastest with int8 quantization
            if name == "whisper":
                return "int8"
            return None  # let HF default (fp32) on CPU

        if dev == "cuda":
            if name == "whisper":
                ct = (settings.WHISPER_COMPUTE_TYPE or "").strip().lower()
                return ct or "float16"
            if name in ("clip", "blip") and settings.VISION_HALF_PRECISION:
                return "float16"
            if name == "text_embedder" and settings.EMBEDDER_HALF_PRECISION:
                return "float16"
            if name == "reranker":
                # CrossEncoder benefits from float16 on GPU
                return "float16"
            # LLM dtype is irrelevant — llama.cpp uses GGUF quant internally
            return "float32"

        if dev == "mps":
            return "float32"

        return None

    # FULL DECISION — used by loaders that want device + dtype + notes together

    def decision_for(self, name: str) -> DeviceDecision:
        dev   = self.device_for(name)
        dtype = self.dtype_for(name, dev)
        return DeviceDecision(device=dev, dtype=dtype, notes=self.profile)

    # GGUF n_gpu_layers — auto-pick -1 (all) on CUDA so Mistral fully offloads

    def llm_gpu_layers(self) -> int:
        llm_device = self.device_for("llm")
        if llm_device != "cuda":
            return 0
        if not settings.LLM_GPU_LAYERS_AUTO:
            return settings.LLM_GPU_LAYERS
        return settings.LLM_GPU_LAYERS_ALL  # -1 = offload everything

    # SNAPSHOT — for /ready and logging

    def snapshot(self) -> Dict[str, Dict[str, Optional[str]]]:
        out: Dict[str, Dict[str, Optional[str]]] = {
            "_summary": {
                "profile":        self.profile,
                "cuda":           str(self._cuda_ok),
                "mps":            str(self._mps_ok),
                "vram_total_gb":  str(self._vram_total_gb),
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
