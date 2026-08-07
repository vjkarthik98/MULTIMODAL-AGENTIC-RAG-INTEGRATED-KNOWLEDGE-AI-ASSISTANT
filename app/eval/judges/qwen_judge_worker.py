"""Out-of-process llama.cpp worker for the Qwen2.5-7B eval judge.

WHY THIS PROCESS EXISTS — read before "simplifying" it back into the parent.

`app/llm/gguf_model.py::_load()` already documents the hard rule this file
enforces for the eval harness:

    # SEPARATE-PROCESS PATH — proxy to the llama-server (its own CUDA
    # context). No in-process llama.cpp, so PyTorch keeps the GPU to
    # itself in this process and ingestion embeds never SIGSEGV.

llama.cpp's CUDA initialization corrupts PyTorch's CUDA context inside the
SAME process. The corruption is latent on the main thread but fatal on
*worker threads* — a subsequent torch CUDA op (an embed/rerank forward pass)
segfaults. That was root-caused once already (2026-06-21) after in-process
mitigations all failed (thread-ordering fence, per-thread warmup, eager
attention); the resolution was to give llama.cpp its own process, which is
why the app runs `llama-server` separately and talks HTTP to it.

`qwen_judge.py` silently reintroduced the banned pattern by calling
`llama_cpp.Llama(..., n_gpu_layers=-1)` in-process, and it broke Tier-2
exactly as predicted — every full-suite run died mid-way, never once
completing:

    retrieval   in-process torch CUDA, main thread          -> OK
    generation  loads the judge => llama.cpp CUDA in-process -> poison planted
    behavioral  grades over HTTP, no local torch CUDA        -> survives
    routing     routing_runner.py calls query_pipeline directly, and
                agent_controller.py:19 dispatches it onto an `agent_ctrl`
                ThreadPoolExecutor worker thread => torch CUDA on a worker
                thread with a corrupted context                -> CRASH

Observed live on the production box: run 30806087765 exit 139 (SIGSEGV) and
run 30745527098 exit 137 (SIGKILL/OOM-killer), both at that exact boundary.
Because a fatal signal is not a Python exception, `runner.py`'s per-suite
`try/except` could not contain it — the whole process died and no report was
ever written.

This worker restores the proven architecture: llama.cpp lives here, alone,
with its own CUDA context, and NEVER imports torch or any `app.*` module that
would pull torch in. The parent keeps PyTorch to itself.

Protocol: newline-delimited JSON on stdin/stdout, one response per request.
    -> {"prompt": str, "max_tokens": int, "temperature": float, "stop": [str]}
    <- {"ok": true, "text": str} | {"ok": false, "error": str}
The first line written is always the handshake:
    <- {"ok": true, "event": "ready"} | {"ok": false, "fatal": true, "error": str}

Configuration arrives via environment variables (QWEN_JUDGE_WORKER_*) rather
than argv so nothing sensitive or path-like has to survive shell quoting.
"""

from __future__ import annotations

import json
import os
import sys


def _main() -> int:
    # ── Protect the protocol channel ─────────────────────────────────────
    # llama.cpp writes its banners/timings from C at the file-descriptor
    # level (printf to fd 1), which `verbose=False` does not fully silence
    # and Python-level redirection cannot intercept. If any of that landed
    # on stdout it would corrupt the JSON-lines stream and desynchronize
    # every subsequent request/response pair.
    #
    # So: duplicate the real stdout to a private fd for the protocol, then
    # point fd 1 at stderr. Anything the C layer prints now goes to stderr,
    # where it stays visible in the CI log for debugging but can never
    # interleave with protocol frames.
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    out = os.fdopen(protocol_fd, "w", buffering=1)

    def _emit(obj: dict) -> None:
        out.write(json.dumps(obj) + "\n")
        out.flush()

    model_path = os.environ.get("QWEN_JUDGE_WORKER_MODEL_PATH", "")
    n_ctx = int(os.environ.get("QWEN_JUDGE_WORKER_N_CTX", "8192"))
    n_gpu_layers = int(os.environ.get("QWEN_JUDGE_WORKER_GPU_LAYERS", "-1"))
    n_threads = int(os.environ.get("QWEN_JUDGE_WORKER_THREADS", "4"))

    if not model_path or not os.path.exists(model_path):
        _emit({"ok": False, "fatal": True, "error": f"model not found at {model_path!r}"})
        return 1

    # BaseException, not Exception: a llama.cpp load failure can surface as a
    # SystemExit/MemoryError rather than a plain Exception, and the parent
    # must always receive a legible handshake instead of a bare EOF.
    try:
        from llama_cpp import Llama

        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            verbose=False,
        )
    except BaseException as exc:  # noqa: BLE001
        _emit({"ok": False, "fatal": True, "error": f"{type(exc).__name__}: {exc}"})
        return 1

    _emit({"ok": True, "event": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"ok": False, "error": f"malformed request: {exc}"})
            continue

        if req.get("cmd") == "shutdown":
            break

        try:
            res = llm(
                req.get("prompt", ""),
                max_tokens=int(req.get("max_tokens", 768)),
                temperature=float(req.get("temperature", 0.0)),
                stop=req.get("stop") or None,
            )
            _emit({"ok": True, "text": res["choices"][0]["text"]})
        except BaseException as exc:  # noqa: BLE001
            # Never let one bad generation take the worker down — the parent
            # degrades that single grading to None and the suite continues.
            _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    return 0


if __name__ == "__main__":
    sys.exit(_main())
