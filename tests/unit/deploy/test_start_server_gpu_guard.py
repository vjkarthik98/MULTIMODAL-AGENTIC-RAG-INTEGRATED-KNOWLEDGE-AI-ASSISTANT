"""start_server must never silently serve the 14B LLM from CPU.

Live 2026-10-02: the box reported `Device: CUDA`, but by the time llama-server
launched the GPU was no longer visible to the container. llama.cpp logged
"failed to initialize CUDA: no CUDA-capable device is detected" and loaded every
layer onto 4 vCPUs; each answer then hit the 630 s generation deadline.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

CUDA_FAIL_LINE = "ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected\n"


@pytest.fixture
def ss(monkeypatch, tmp_path):
    """start_server with SCRIPT_DIR pointed at a temp dir (it chdir()s on import)."""
    cwd = os.getcwd()
    sys.modules.pop("start_server", None)
    mod = importlib.import_module("start_server")
    monkeypatch.setattr(mod, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)
    (tmp_path / "logs").mkdir()
    yield mod
    os.chdir(cwd)
    sys.modules.pop("start_server", None)


class FakeProc:
    def __init__(self):
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.terminated = True


def _wire(ss, tmp_path, monkeypatch, outcomes):
    """Each launch consumes one outcome: 'gpu' (clean log) or 'nogpu' (writes the CUDA failure line)."""
    log_path = tmp_path / "logs" / "llama_server.log"
    launches: list[tuple[bool, FakeProc]] = []
    queue = list(outcomes)

    def fake_launch(cuda):
        outcome = queue.pop(0)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(CUDA_FAIL_LINE if outcome == "nogpu" else "load_tensors: offloaded 49/49 layers to GPU\n")
        proc = FakeProc()
        launches.append((cuda, proc))
        return proc

    monkeypatch.setattr(ss, "launch_llama_server", fake_launch)
    monkeypatch.setattr(ss, "wait_for_llama_server", lambda proc, timeout=180: True)
    return launches


def test_scanner_finds_the_failure_only_after_the_offset(ss, tmp_path):
    log_path = tmp_path / "logs" / "llama_server.log"
    log_path.write_text(CUDA_FAIL_LINE, encoding="utf-8")  # an OLD boot's failure
    offset = log_path.stat().st_size
    assert ss.cuda_init_failure(log_path, offset) is None, "stale failure from a previous boot must not count"

    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(CUDA_FAIL_LINE)
    assert "no CUDA-capable device" in ss.cuda_init_failure(log_path, offset)


def test_scanner_tolerates_a_missing_log(ss, tmp_path):
    assert ss.cuda_init_failure(tmp_path / "nope.log", 0) is None


def test_healthy_gpu_launch_is_accepted_first_try(ss, tmp_path, monkeypatch):
    launches = _wire(ss, tmp_path, monkeypatch, ["gpu"])
    proc = ss.start_llama_server_checked(True)
    assert proc is launches[0][1] and len(launches) == 1 and not proc.terminated


def test_gpu_that_reappears_on_retry_is_accepted(ss, tmp_path, monkeypatch):
    launches = _wire(ss, tmp_path, monkeypatch, ["nogpu", "gpu"])
    proc = ss.start_llama_server_checked(True)
    assert len(launches) == 2
    assert launches[0][1].terminated, "the CPU-resident server must be shut down, not left running"
    assert proc is launches[1][1]


def test_persistent_gpu_loss_exits_loudly_instead_of_serving_from_cpu(ss, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("LLM_ALLOW_CPU_FALLBACK", raising=False)
    launches = _wire(ss, tmp_path, monkeypatch, ["nogpu"] * 3)
    with pytest.raises(SystemExit) as exc:
        ss.start_llama_server_checked(True)
    assert exc.value.code == 3
    assert len(launches) == 3 and all(p.terminated for _, p in launches)
    assert "FATAL" in capsys.readouterr().out


def test_cpu_fallback_is_an_explicit_opt_in(ss, tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_ALLOW_CPU_FALLBACK", "true")
    launches = _wire(ss, tmp_path, monkeypatch, ["nogpu", "nogpu", "nogpu", "gpu"])
    ss.start_llama_server_checked(True)
    assert launches[-1][0] is False, "the opt-in fallback launches the server in CPU mode"


def test_cpu_box_is_never_gated(ss, tmp_path, monkeypatch):
    """A laptop/CI box with no GPU is a legitimate CPU run — only a GPU box that lost its GPU is fatal."""
    launches = _wire(ss, tmp_path, monkeypatch, ["nogpu"])
    proc = ss.start_llama_server_checked(False)
    assert proc is launches[0][1] and len(launches) == 1


def test_attempts_are_configurable(ss, tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_GPU_LAUNCH_ATTEMPTS", "1")
    launches = _wire(ss, tmp_path, monkeypatch, ["nogpu"])
    with pytest.raises(SystemExit):
        ss.start_llama_server_checked(True)
    assert len(launches) == 1
