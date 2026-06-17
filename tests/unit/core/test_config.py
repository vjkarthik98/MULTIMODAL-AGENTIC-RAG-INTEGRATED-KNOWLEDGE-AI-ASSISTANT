from pathlib import Path

import pytest

from app.core.config import Settings, get_settings, settings


class TestSettings:

    def test_defaults_are_valid(self):
        s = Settings()
        assert isinstance(s.APP_VERSION, str) and len(s.APP_VERSION) > 0
        assert s.TEXT_EMBEDDING_DIM == 1024
        assert s.VISION_EMBEDDING_DIM == 1152
        assert s.CHUNK_OVERLAP < s.CHUNK_SIZE
        assert s.AGENT_MAX_STEPS > 0
        assert s.RAG_TOP_K > 0
        assert s.MEMORY_TOP_K > 0

    def test_fusion_weights_sum_to_one(self):
        s = Settings()
        total = (
            s.FUSION_SCORE_WEIGHT
            + s.FUSION_QUALITY_WEIGHT
            + s.FUSION_MODALITY_WEIGHT
        )
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

    def test_paths_are_path_objects(self):
        s = Settings()

        assert isinstance(s.DATA_DIR, Path)
        assert isinstance(s.LOG_DIR, Path)
        assert isinstance(s.TEMP_DIR, Path)
        assert isinstance(s.AUDIT_LOG_PATH, Path)

    def test_otel_sampling_ratio_range(self):
        s = Settings()
        assert 0.0 <= s.OTEL_SAMPLING_RATIO <= 1.0

    def test_minhash_threshold_range(self):
        s = Settings()
        assert 0.0 <= s.CHUNK_MINHASH_THRESHOLD <= 1.0

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