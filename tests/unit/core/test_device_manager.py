from unittest.mock import MagicMock, patch

import pytest

from app.core.device_manager import DeviceManager


def _make_manager(cuda=False, mps=False, vram=0.0, profile="auto"):
    """Build a DeviceManager with torch probing fully mocked."""
    with patch("app.core.device_manager.settings") as mock_cfg:
        mock_cfg.MODELS_DEVICE_PROFILE = profile
        mock_cfg.VRAM_BUDGET_GB = 8.0
        mock_cfg.LLM_DEVICE_HINT = ""
        mock_cfg.EMBEDDER_DEVICE = ""
        mock_cfg.RERANKER_DEVICE = ""
        mock_cfg.CLIP_DEVICE = ""
        mock_cfg.BLIP_DEVICE = ""
        mock_cfg.WHISPER_DEVICE = ""

        dm = DeviceManager.__new__(DeviceManager)
        import threading
        dm._lock = threading.Lock()
        dm._torch = None
        dm._cuda_ok = cuda
        dm._mps_ok = mps
        dm._vram_total_gb = vram
        dm._profile = dm._resolve_profile(profile)
    return dm


class TestResolveProfile:

    def test_auto_accepted(self):
        dm = _make_manager(profile="auto")
        assert dm._profile == "auto"

    def test_all_gpu_accepted(self):
        dm = _make_manager(profile="all_gpu")
        assert dm._profile == "all_gpu"

    def test_all_cpu_accepted(self):
        dm = _make_manager(profile="all_cpu")
        assert dm._profile == "all_cpu"

    def test_hybrid_accepted(self):
        dm = _make_manager(profile="hybrid")
        assert dm._profile == "hybrid"

    def test_unknown_profile_defaults_to_auto(self):
        dm = _make_manager(profile="garbage_profile")
        assert dm._profile == "auto"


class TestProfileProperty:

    def test_auto_with_cuda_returns_all_gpu(self):
        dm = _make_manager(cuda=True, profile="auto")
        assert dm.profile == "all_gpu"

    def test_auto_without_cuda_returns_all_cpu(self):
        dm = _make_manager(cuda=False, profile="auto")
        assert dm.profile == "all_cpu"

    def test_all_gpu_without_cuda_demotes_to_all_cpu(self):
        dm = _make_manager(cuda=False, profile="all_gpu")
        assert dm.profile == "all_cpu"

    def test_all_cpu_always_returns_all_cpu(self):
        dm = _make_manager(cuda=True, profile="all_cpu")
        assert dm.profile == "all_cpu"

    def test_hybrid_without_cuda_demotes_to_all_cpu(self):
        dm = _make_manager(cuda=False, profile="hybrid")
        assert dm.profile == "all_cpu"


class TestCudaAvailable:

    def test_cuda_false_when_no_torch(self):
        dm = _make_manager(cuda=False)
        assert dm.cuda_available is False

    def test_cuda_true_when_probed(self):
        dm = _make_manager(cuda=True)
        assert dm.cuda_available is True


class TestVramTotalGb:

    def test_zero_without_cuda(self):
        dm = _make_manager(cuda=False, vram=0.0)
        assert dm.vram_total_gb == 0.0

    def test_value_set_when_cuda(self):
        dm = _make_manager(cuda=True, vram=15.5)
        assert dm.vram_total_gb == 15.5


class TestDeviceFor:

    def test_cpu_profile_returns_cpu(self):
        dm = _make_manager(cuda=False, profile="all_cpu")
        result = dm.device_for("text_embedder")
        assert result == "cpu"

    def test_gpu_profile_with_cuda_returns_cuda(self):
        dm = _make_manager(cuda=True, profile="all_gpu")
        result = dm.device_for("text_embedder")
        assert result == "cuda"

    def test_gpu_profile_without_cuda_demotes_to_cpu(self):
        dm = _make_manager(cuda=False, profile="all_gpu")
        result = dm.device_for("text_embedder")
        assert result == "cpu"

    def test_unknown_model_falls_back_to_cpu(self):
        dm = _make_manager(cuda=False, profile="all_cpu")
        result = dm.device_for("nonexistent_model_xyz")
        assert result == "cpu"


class TestDecisionFor:

    def test_returns_device_decision(self):
        from app.core.device_manager import DeviceDecision
        dm = _make_manager(cuda=False, profile="all_cpu")
        decision = dm.decision_for("text_embedder")
        assert isinstance(decision, DeviceDecision)
        assert decision.device in ("cpu", "cuda", "mps")

    def test_decision_has_notes_string(self):
        from app.core.device_manager import DeviceDecision
        dm = _make_manager(cuda=False, profile="all_cpu")
        decision = dm.decision_for("text_embedder")
        assert isinstance(decision.notes, str)


class TestSnapshot:

    def test_snapshot_returns_dict(self):
        dm = _make_manager(cuda=False, profile="all_cpu")
        snap = dm.snapshot()
        assert isinstance(snap, dict)

    def test_snapshot_has_entries(self):
        dm = _make_manager(cuda=False, profile="all_cpu")
        snap = dm.snapshot()
        # Snapshot is a non-empty dict — structure varies by implementation
        assert len(snap) > 0


class TestLlmGpuLayers:

    def test_cpu_profile_returns_zero(self):
        dm = _make_manager(cuda=False, profile="all_cpu")
        layers = dm.llm_gpu_layers()
        assert layers == 0

    def test_return_type_is_int(self):
        dm = _make_manager(cuda=False, profile="all_cpu")
        layers = dm.llm_gpu_layers()
        assert isinstance(layers, int)
