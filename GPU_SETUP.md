# GPU Setup — Tesla T4 (14.6 GB VRAM, CUDA 13.2)

## What changed

### Models on GPU (all via `MODELS_DEVICE_PROFILE=all_gpu`)
| Model | VRAM | dtype |
|-------|------|-------|
| Mistral 7B Q4_K_M (llama-cpp) | ~4.1 GB | GGUF quant (n_gpu_layers=-1) |
| CLIP ViT-B/32 | ~0.6 GB | float16 |
| BLIP base (captioning) | ~1.0 GB | float16 |
| Whisper large-v3 | ~1.5 GB | float16 |
| CrossEncoder MiniLM | ~0.1 GB | float16 |
| MiniLM text embedder | ~0.09 GB | float16 |
| **Total** | **~7.4 GB** | leaves 7+ GB free |

### Services on CPU (unchanged)
- FastAPI / Uvicorn
- Qdrant client (cloud, network I/O)
- Redis client (Upstash, network I/O)
- MongoDB client (Atlas, network I/O)
- BM25 index (in-memory)

## Files changed
- `.env` — GPU device profile, batch sizes, timeouts, Whisper large-v3
- `app/core/config.py` — GPU defaults for all settings
- `app/core/device_manager.py` — all_gpu profile default, reranker fp16, auto→all_gpu on CUDA
- `app/core/model_loader.py` — CrossEncoder fp16 cast on GPU
- `app/main.py` — non-blocking lifespan: infra+model warmup as background tasks
- `app/core/startup_optimizer.py` (**new**) — parallel GPU preload + cuBLAS warmup + TF32 flags

## New files
- `scripts/install_cuda_deps.sh` — reproducible CUDA setup
- `scripts/start_server.sh` — GPU-aware Uvicorn launcher with uvloop+httptools

## How to start
```bash
bash scripts/start_server.sh
```
Or directly:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop --http httptools
```

## Uvicorn startup time
**Before**: 7-8 seconds (infra warmup + Qdrant init blocking the ready signal)
**After**: <1 second (Uvicorn ready immediately; infra+GPU models load in background)
First-request latency: models may not be warm yet if hit within ~10s of start.
After that: all GPU, zero cold-start.

## llama-cpp-python CUDA install
Pre-built wheel from `https://abetlen.github.io/llama-cpp-python/whl/cu124`.
CUDA 12 libs provided by `nvidia-cublas-cu12` + `nvidia-cuda-runtime-cu12` (PyPI).
Registered in `/etc/ld.so.conf.d/` — no `LD_LIBRARY_PATH` needed at runtime.
