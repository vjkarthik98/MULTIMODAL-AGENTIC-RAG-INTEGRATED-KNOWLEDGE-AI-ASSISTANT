"""
torchaudio_compat.py — compatibility shim for torchaudio ≥ 2.1

torchaudio 2.1 removed the top-level public API that pyannote.audio 3.x
depends on (AudioMetaData, .info(), .load(), .list_audio_backends()).
This shim restores those symbols using soundfile before pyannote is imported.

Usage (call before any pyannote import):
    from app.utils.torchaudio_compat import patch_torchaudio
    patch_torchaudio()
"""

from __future__ import annotations

import dataclasses
import wave

import torch


@dataclasses.dataclass
class _AudioMetaData:
    sample_rate: int
    num_frames: int
    num_channels: int
    bits_per_sample: int = 16
    encoding: str = "PCM_S"


def _info(path: str, backend: str = None) -> _AudioMetaData:
    """Drop-in for torchaudio.info() using soundfile → wave fallback."""
    try:
        import soundfile as sf

        i = sf.info(str(path))
        return _AudioMetaData(
            sample_rate=i.samplerate,
            num_frames=i.frames,
            num_channels=i.channels,
        )
    except Exception:
        pass
    # stdlib wave fallback (WAV only)
    with wave.open(str(path), "rb") as wf:
        return _AudioMetaData(
            sample_rate=wf.getframerate(),
            num_frames=wf.getnframes(),
            num_channels=wf.getnchannels(),
        )


def _load(
    path: str,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    backend: str = None,
) -> tuple[torch.Tensor, int]:
    """Drop-in for torchaudio.load() using soundfile."""
    import soundfile as sf

    data, sr = sf.read(
        str(path),
        frames=num_frames if num_frames != -1 else -1,
        start=frame_offset,
        dtype="float32",
        always_2d=True,
    )
    # soundfile returns (frames, channels); torchaudio returns (channels, frames)
    waveform = torch.tensor(data.T, dtype=torch.float32)
    if not normalize:
        waveform = waveform * 32768.0
    if not channels_first:
        waveform = waveform.T
    return waveform, sr


def _list_audio_backends() -> list[str]:
    """Drop-in for torchaudio.list_audio_backends()."""
    backends = []
    try:
        import soundfile  # noqa: F401

        backends.append("soundfile")
    except ImportError:
        pass
    return backends


def patch_torchaudio() -> None:
    """Patch torchaudio to restore the API surface pyannote.audio 3.x expects.

    Also patches:
    - huggingface_hub.hf_hub_download: accept legacy use_auth_token kwarg
      (renamed to token in huggingface_hub ≥0.17)
    - torch.serialization safe globals: allow TorchVersion in pyannote
      checkpoints (PyTorch 2.6 changed weights_only default to True)

    Safe to call multiple times — idempotent.
    """
    import torchaudio

    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = _AudioMetaData

    if not hasattr(torchaudio, "info"):
        torchaudio.info = _info

    if not hasattr(torchaudio, "load") or _load_is_broken(torchaudio):
        torchaudio.load = _load

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = _list_audio_backends

    _patch_hf_hub_download()
    _patch_torch_safe_globals()


def _patch_hf_hub_download() -> None:
    """Shim huggingface_hub.hf_hub_download to accept legacy use_auth_token kwarg.

    pyannote.audio 3.x calls hf_hub_download(use_auth_token=...) but
    huggingface_hub ≥0.17 renamed that parameter to `token` and later
    removed `use_auth_token` entirely.
    """
    try:
        import huggingface_hub

        _orig = huggingface_hub.hf_hub_download
        if getattr(_orig, "_use_auth_token_patched", False):
            return

        def _patched(*args, use_auth_token=None, **kwargs):
            if use_auth_token is not None and "token" not in kwargs:
                kwargs["token"] = use_auth_token
            return _orig(*args, **kwargs)

        _patched._use_auth_token_patched = True
        huggingface_hub.hf_hub_download = _patched

        # Also patch the symbol re-exported inside pyannote if already imported
        try:
            import pyannote.audio.core.pipeline as _pap

            if hasattr(_pap, "hf_hub_download"):
                _pap.hf_hub_download = _patched
        except Exception:
            pass
    except Exception:
        pass


def _patch_torch_safe_globals() -> None:
    """Register all globals that pyannote checkpoints embed, blocked by PyTorch 2.6.

    PyTorch 2.6 changed weights_only default from False → True.  pyannote
    checkpoints contain Python objects (Specifications, Segment, etc.) beyond
    just tensors.  We allowlist every known class, then fall back to patching
    torch.load to use weights_only=False when called from a pyannote/lightning
    context — the same behaviour as PyTorch ≤ 2.5.
    """
    try:
        import torch
        import torch.torch_version

        safe: list = [torch.torch_version.TorchVersion]

        _PYANNOTE_GLOBALS = [
            ("pyannote.audio.core.task", "Specifications"),
            ("pyannote.audio.core.task", "Problem"),
            ("pyannote.audio.core.task", "Scale"),
            ("pyannote.core", "Segment"),
            ("pyannote.core.segment", "Segment"),
            ("pyannote.core.annotation", "Annotation"),
            ("pyannote.core.timeline", "Timeline"),
            ("pyannote.core.feature", "SlidingWindowFeature"),
            ("pyannote.core", "SlidingWindow"),
        ]
        import importlib

        for mod_path, cls_name in _PYANNOTE_GLOBALS:
            try:
                mod = importlib.import_module(mod_path)
                cls = getattr(mod, cls_name, None)
                if cls is not None and cls not in safe:
                    safe.append(cls)
            except Exception:
                pass

        torch.serialization.add_safe_globals(safe)
    except Exception:
        pass

    # Fallback: patch torch.load to default weights_only=False when the call
    # originates from pyannote or lightning_fabric (same as pre-2.6 behaviour).
    # We control which model files are loaded (HF hub cache on local EBS), so
    # the pickle-safety risk of weights_only=False doesn't apply here.
    _patch_torch_load_for_pyannote()


def _patch_torch_load_for_pyannote() -> None:
    """Wrap torch.load so pyannote/lightning callers get weights_only=False.

    Uses inspect.stack() only when weights_only is not explicitly passed by the
    caller — so normal application code that already passes weights_only=True is
    never affected.  Idempotent.
    """
    try:
        import torch

        _orig = torch.load
        if getattr(_orig, "_pyannote_compat_patched", False):
            return

        import inspect

        def _patched(
            f, map_location=None, pickle_module=None, *, weights_only=None, mmap=None, **kw
        ):
            if weights_only is None:
                # Only pay the stack-inspection cost when the flag wasn't set.
                try:
                    frame_names = " ".join(
                        fr.filename for fr in inspect.stack(context=0) if fr.filename
                    )
                    if "pyannote" in frame_names or "lightning" in frame_names:
                        weights_only = False
                    else:
                        weights_only = True  # keep new PyTorch 2.6 default
                except Exception:
                    weights_only = False  # safe fallback

            call_kw: dict = dict(kw)
            if map_location is not None:
                call_kw["map_location"] = map_location
            if pickle_module is not None:
                call_kw["pickle_module"] = pickle_module
            if mmap is not None:
                call_kw["mmap"] = mmap
            return _orig(f, weights_only=weights_only, **call_kw)

        _patched._pyannote_compat_patched = True
        torch.load = _patched
    except Exception:
        pass


def _load_is_broken(torchaudio_module) -> bool:
    """Detect if torchaudio.load is the broken torchcodec stub."""
    import inspect

    try:
        src = inspect.getsource(torchaudio_module.load)
        return "torchcodec" in src or "TorchCodec" in src
    except Exception:
        return False
