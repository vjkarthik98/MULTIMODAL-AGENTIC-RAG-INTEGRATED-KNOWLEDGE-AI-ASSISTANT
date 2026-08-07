#!/usr/bin/env python3
"""start_server.py — cross-platform, cross-hardware startup for MAGIK.

Single canonical launcher for both targets this project runs on:
  - Local CPU box (Windows/Mac/Linux laptop, no GPU)
  - AWS g6e.xlarge (Linux, NVIDIA L40S 48 GB CUDA)

Device/OS branching happens INSIDE this script (mirrors the pattern already
used by app/core/device_manager.py — detect, don't duplicate config), so
there is exactly one run command regardless of what you're running on:

    python start_server.py
    python start_server.py --port 8080

install_cuda.sh remains the reference for one-time AWS AMI provisioning
(apt packages, CUDA toolkit, building llama-cpp-python from source) — this
script covers everything needed on every subsequent run/boot.
"""
from __future__ import annotations

import atexit
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Windows' default console codepage (cp1252) can't encode every character
# these logs might use — reconfigure to UTF-8 so a stray unicode character
# never crashes the launcher. No-op on Linux/Mac (already UTF-8).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)


def log(msg: str) -> None:
    print(f"[start_server] {msg}", flush=True)


def load_env() -> None:
    from dotenv import load_dotenv
    load_dotenv(SCRIPT_DIR / ".env")


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def is_linux_gpu_box() -> bool:
    """True only on the AWS AMI layout install_cuda.sh/start_server.sh target."""
    return platform.system() == "Linux" and Path("/opt/dlami/nvme").exists()


# ── ENV VARS (universal — same on every target) ──────────────────────────────

def set_common_env() -> None:
    os.environ.setdefault("HF_HOME", str(SCRIPT_DIR / ".hf_cache"))
    os.environ.setdefault("TORCH_HOME", str(SCRIPT_DIR / ".torch_cache"))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(SCRIPT_DIR / ".torch_cache" / "inductor"))

    # transformers/datasets eagerly import TensorFlow if installed, grabbing
    # several GB of host RAM at startup. Stay on the PyTorch backend; YAMNet
    # still imports TF lazily (inside try/except) only when audio needs it.
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_offline_env() -> None:
    # Set AFTER ensure_models() (which needs the network), BEFORE model
    # loading. Every from_pretrained() otherwise HEAD-checks huggingface.co
    # for a newer revision even when fully cached — skip it, this project is
    # 100% local/offline once models are cached. HF_HUB_OFFLINE=0 to force an
    # online check (e.g. after adding a new model not yet in .hf_cache/).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def warn_missing_binaries() -> None:
    for exe, hint in (
        ("tesseract", "OCR fallback in image_ingest.py — winget install UB-Mannheim.TesseractOCR (Windows) / apt install tesseract-ocr (Linux)"),
        ("ffmpeg", "audio/video ingestion — winget install Gyan.FFmpeg (Windows) / apt install ffmpeg (Linux)"),
    ):
        if shutil.which(exe) is None:
            log(f"WARN: '{exe}' not on PATH — {hint}")


def warn_redis_config() -> None:
    if not (os.environ.get("REDIS_URL") and os.environ.get("REDIS_TOKEN")) and os.environ.get("USE_REDIS", "true").lower() == "true":
        log("WARN: REDIS_URL/REDIS_TOKEN not both set — redis_memory.py falls back to "
            "an in-process store (job-status polling won't survive a restart).")


# ── AWS-ONLY SETUP (no-op everywhere else) ────────────────────────────────────

def linux_gpu_setup() -> None:
    """Swapfile + CUDA library path, ported from start_server.sh. Only runs on
    the AWS AMI layout (/opt/dlami/nvme present) — a no-op on a laptop."""
    if not is_linux_gpu_box():
        return

    swapfile = Path("/opt/dlami/nvme/swapfile")
    swap_size_gb = os.environ.get("SWAP_SIZE_GB", "32")
    try:
        active = subprocess.run(["swapon", "--show"], capture_output=True, text=True).stdout
        if str(swapfile) not in active:
            log(f"Creating {swap_size_gb}G swap at {swapfile}...")
            subprocess.run(["sudo", "rm", "-f", str(swapfile)], check=False)
            ok = (
                subprocess.run(["sudo", "fallocate", "-l", f"{swap_size_gb}G", str(swapfile)]).returncode == 0
                and subprocess.run(["sudo", "chmod", "600", str(swapfile)]).returncode == 0
                and subprocess.run(["sudo", "mkswap", str(swapfile)], stdout=subprocess.DEVNULL).returncode == 0
                and subprocess.run(["sudo", "swapon", str(swapfile)]).returncode == 0
            )
            log("Swap active." if ok else "WARN: swap setup failed — startup may OOM.")
        subprocess.run(["sudo", "sysctl", "-w", "vm.swappiness=10"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except FileNotFoundError:
        pass  # swapon/sudo not available — not actually the AMI we think it is

    # torch (cu124) and llama-cpp-python (built from source) share the same
    # nvidia-*-cu12 pip packages' .so files — point LD_LIBRARY_PATH at them so
    # both find libcudart.so.12 / libcublas.so.12 without a dual-runtime conflict.
    import sysconfig
    site_pkgs = Path(sysconfig.get_paths()["purelib"])
    libs = [
        str(site_pkgs / "nvidia" / pkg / "lib")
        for pkg in ("cublas", "cuda_runtime", "cudnn", "cufft", "curand", "cusolver", "cusparse", "nvjitlink")
        if (site_pkgs / "nvidia" / pkg / "lib").is_dir()
    ]
    if Path("/usr/local/cuda/lib64").is_dir():
        libs.append("/usr/local/cuda/lib64")
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(libs + ([existing] if existing else []))

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for exe, pkgs in (("tesseract", "tesseract-ocr tesseract-ocr-eng"), ("ffmpeg", "ffmpeg")):
        if shutil.which(exe) is None:
            log(f"Installing {pkgs} via apt...")
            subprocess.run(["sudo", "apt-get", "install", "-y", *pkgs.split()], check=False)

    # redis-server backs settings.LOCAL_CACHE_HOST/PORT (localhost:6379 by
    # default — job-status polling, embedding cache, rate limits). install_cuda.sh
    # provisions + enables it once; this is a cheap self-heal if the service
    # ever stops (e.g. after an instance reboot with the unit not enabled).
    if shutil.which("redis-server") is not None:
        subprocess.run(
            ["sudo", "systemctl", "start", "redis-server"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


# ── MODELS ─────────────────────────────────────────────────────────────────

def ensure_models() -> None:
    log("Ensuring all models are present (skips anything already cached)...")
    result = subprocess.run([sys.executable, str(SCRIPT_DIR / "app" / "bin" / "models" / "download_all_models.py")])
    if result.returncode != 0:
        log("WARN: model download step exited non-zero — some models may be missing; server may fail to load them on first use.")


# ── PORT CLEANUP (cross-platform via psutil, no lsof needed) ────────────────

def kill_port(port: int) -> None:
    import psutil
    seen: set[int] = set()
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr and conn.laddr.port == port and conn.pid and conn.pid not in seen:
            seen.add(conn.pid)
            try:
                proc = psutil.Process(conn.pid)
                log(f"Stopping existing process on port {port} (PID {conn.pid}, {proc.name()})...")
                proc.terminate()
                proc.wait(timeout=5)
            except psutil.TimeoutExpired:
                proc.kill()
            except Exception as exc:
                log(f"WARN: couldn't stop PID {conn.pid}: {exc}")


# ── LLAMA-SERVER (separate process, own CUDA context on GPU boxes) ──────────
# llama.cpp must NOT run in the uvicorn process: in-process CUDA init corrupts
# PyTorch's CUDA context on worker threads -> embedding SIGSEGVs at ingest
# time (GPU boxes only — irrelevant but harmless on CPU). GGUFModel proxies
# generate()/stream() to this server over HTTP either way.

def launch_llama_server(cuda: bool) -> subprocess.Popen:
    host = os.environ.get("LLM_SERVER_HOST", "127.0.0.1")
    port = int(os.environ.get("LLM_SERVER_PORT", "8081"))
    kill_port(port)

    gguf_path = os.environ.get("LLM_MODEL_PATH", str(SCRIPT_DIR / ".hf_cache" / "gguf" / "Qwen2.5-14B-Instruct-Q4_K_M.gguf"))
    ctx = os.environ.get("CONTEXT_MAX_TOKENS", "16384")
    n_batch = os.environ.get("LLM_N_BATCH", "512")
    cpu_threads = str(os.cpu_count() or 4)
    threads = os.environ.get("LLM_THREADS", cpu_threads if not cuda else "4")
    threads_batch = os.environ.get("LLM_THREADS_BATCH", cpu_threads if not cuda else "4")
    n_gpu_layers = os.environ.get("LLM_GPU_LAYERS", "-1" if cuda else "0")
    flash_attn = os.environ.get("LLM_FLASH_ATTN", "true" if cuda else "false")

    log(f"Launching llama-server on {host}:{port} (device={'cuda' if cuda else 'cpu'}, "
        f"n_gpu_layers={n_gpu_layers}, threads={threads})...")

    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", gguf_path,
        "--host", host,
        "--port", str(port),
        "--n_gpu_layers", n_gpu_layers,
        "--n_ctx", ctx,
        "--n_batch", n_batch,
        "--n_threads", threads,
        "--n_threads_batch", threads_batch,
        "--flash_attn", flash_attn,
        "--cache", os.environ.get("LLM_PROMPT_CACHE", "true"),
        "--use_mlock", "false",
    ]
    logs_dir = SCRIPT_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = open(logs_dir / "llama_server.log", "a", encoding="utf-8")

    # llama_cpp.server's own main() does `os.getenv("HOST"/"PORT", <cli value>)` —
    # env vars win over --host/--port. Our .env sets HOST=0.0.0.0/PORT=8000 for
    # the MAIN app, and that inherits into this subprocess's environment,
    # silently overriding the llama-server-specific host/port CLI flags above
    # (it was binding 0.0.0.0:8000 instead of 127.0.0.1:8081, which then got
    # killed by kill_port(8000) right before the main uvicorn launch). Give
    # this subprocess its own HOST/PORT so the CLI flags actually stick.
    llama_env = {**os.environ, "HOST": host, "PORT": str(port)}
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=llama_env)
    return proc


def wait_for_llama_server(proc: subprocess.Popen, timeout: int = 180) -> bool:
    host = os.environ.get("LLM_SERVER_HOST", "127.0.0.1")
    port = os.environ.get("LLM_SERVER_PORT", "8081")
    url = f"http://{host}:{port}/v1/models"
    log(f"llama-server PID {proc.pid}; waiting for readiness (up to {timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            log("WARN: llama-server died during startup — see logs/llama_server.log")
            return False
        try:
            urllib.request.urlopen(url, timeout=2)
            log("llama-server ready.")
            return True
        except Exception:
            time.sleep(2)
    log("WARN: llama-server did not become ready in time — continuing anyway.")
    return False


def check_llama_cpp_import() -> None:
    try:
        import llama_cpp
        log(f"llama_cpp {llama_cpp.__version__}, GPU offload supported: {llama_cpp.llama_supports_gpu_offload()}")
    except Exception as exc:
        log(f"WARN: llama_cpp not importable: {exc}")


# ── UVICORN ───────────────────────────────────────────────────────────────

def launch_uvicorn(port: int, extra_args: list[str]) -> int:
    kill_port(port)
    # uvloop has no Windows build at all (pip simply won't have installed it
    # there); httptools does ship Windows wheels, so keep that on every OS.
    loop = "uvloop" if os.name == "posix" else "asyncio"
    cmd = [
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--workers", "1",
        "--loop", loop,
        "--http", "httptools",
        "--timeout-keep-alive", "30",
        "--limit-concurrency", "64",
        "--backlog", "256",
        "--log-level", "warning",
        *extra_args,
    ]
    log(f"Starting uvicorn on 0.0.0.0:{port} ({loop} loop)...")
    return subprocess.run(cmd).returncode


def main() -> None:
    load_env()
    set_common_env()
    cuda = cuda_available()
    log(f"Device: {'CUDA' if cuda else 'CPU'}  |  OS: {platform.system()}  |  "
        f"logical cores: {os.cpu_count()}")

    linux_gpu_setup()
    warn_missing_binaries()
    warn_redis_config()

    ensure_models()
    set_offline_env()
    check_llama_cpp_import()

    llama_proc = launch_llama_server(cuda)
    atexit.register(lambda: llama_proc.poll() is None and llama_proc.terminate())
    wait_for_llama_server(llama_proc)

    port = int(os.environ.get("PORT", "8000"))
    extra_args = sys.argv[1:]
    exit_code = launch_uvicorn(port, extra_args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
