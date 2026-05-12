import pytest
from uuid import UUID


class TestUniversalMetadata:

    def test_default_construction(self):
        m = UniversalMetadata()
        assert isinstance(m.file_id, UUID)
        assert m.ingested_at > 0
        assert m.extraction_quality == 1.0
        assert m.status == ProcessingStatus.PENDING
        assert m.malware_clean is True
        assert m.is_duplicate is False

    def test_checksum_validated_length(self):
        with pytest.raises(Exception):
            UniversalMetadata(checksum_sha256="tooshort")

    def test_valid_checksum(self):
        sha = "a" * 64
        m = UniversalMetadata(checksum_sha256=sha)
        assert m.checksum_sha256 == sha

    def test_quality_clamped(self):
        m = UniversalMetadata(extraction_quality=99.0)
        assert m.extraction_quality == 1.0

    def test_add_error(self):
        m = UniversalMetadata()
        m.add_error("OCR_FAILED")
        assert len(m.error_log) == 1
        assert "OCR_FAILED" in m.error_log[0]

    def test_to_dict_has_all_fields(self):
        m = UniversalMetadata()
        d = m.to_dict()
        required = [
            "file_id", "source_path", "modality", "mime_type",
            "file_size_bytes", "checksum_sha256", "ingested_at",
            "language", "encoding", "chunk_count", "embedding_model",
            "extraction_quality", "error_log", "tags", "custom_fields",
            "pii_redacted", "is_duplicate", "malware_scanned", "status",
        ]
        for field in required:
            assert field in d, f"Missing field: {field}"


class TestIngestedDocument:

    def _make(self, text="Hello world this is a test chunk.", modality="text", subtype="paragraph"):
        return IngestedDocument(text=text, modality=modality, subtype=subtype)

    def test_finalize_happy_path(self):
        doc = self._make().finalize()
        assert doc.text
        assert doc.modality == "text"
        assert doc.structure.get("doc_id")
        assert doc.extra_metadata.get("importance_score") <= 1.0

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="EMPTY_TEXT"):
            IngestedDocument(text="   ", modality="text").normalize()

    def test_text_too_short_raises(self):
        with pytest.raises(ValueError, match="TEXT_TOO_SHORT"):
            IngestedDocument(text="Hi", modality="text").validate()

    def test_invalid_modality_raises(self):
        with pytest.raises(ValueError, match="INVALID_MODALITY"):
            IngestedDocument(text="Hello world test chunk here.", modality="unknown").validate()

    def test_invalid_page_raises(self):
        with pytest.raises(ValueError, match="INVALID_PAGE"):
            doc = self._make()
            doc.page = 0
            doc.validate()

    def test_invalid_chunk_id_raises(self):
        with pytest.raises(ValueError, match="INVALID_CHUNK_ID"):
            doc = self._make()
            doc.chunk_id = -1
            doc.validate()

    def test_unknown_subtype_coerced(self):
        doc = IngestedDocument(
            text="Hello world test chunk content here.",
            modality="text",
            subtype="nonsense_subtype",
        ).finalize()
        assert doc.subtype == "unknown"

    def test_null_bytes_stripped(self):
        doc = IngestedDocument(
            text="Hello\x00World\x00test content here.",
            modality="text",
        ).finalize()
        assert "\x00" not in doc.text

    def test_nfc_normalization_applied(self):
        raw = "caf\u0065\u0301"
        doc = IngestedDocument(text=raw + " test content here.", modality="text").finalize()
        assert unicodedata.is_normalized("NFC", doc.text)

    def test_content_hash_sha256(self):
        doc = self._make().finalize()
        h = doc.content_hash()
        assert len(h) == 64
        assert h == hashlib.sha256(doc.text.encode("utf-8")).hexdigest()

    def test_quality_scores_clamped(self):
        doc = self._make()
        doc.extra_metadata = {"importance_score": 99.0, "modality_weight": -1.0, "data_quality_score": 2.0}
        doc.finalize()
        assert doc.extra_metadata["importance_score"] <= 1.0
        assert doc.extra_metadata["modality_weight"] >= 0.0

    def test_is_embeddable_false_without_embedding(self):
        doc = self._make().finalize()
        assert not doc.is_embeddable()

    def test_is_embeddable_true_with_correct_dim(self):
        doc = self._make().finalize()
        doc.embedding = [0.1] * settings.TEXT_EMBEDDING_DIM
        assert doc.is_embeddable()

    def test_invalid_embedding_dim_raises(self):
        with pytest.raises(ValueError, match="INVALID_EMBEDDING_DIM"):
            doc = self._make()
            doc.embedding = [0.1] * 99
            doc.validate()

    def test_clone_produces_independent_copy(self):
        doc = self._make().finalize()
        clone = doc.clone(text="Different text content for clone test.")
        assert clone.text != doc.text
        assert clone.doc_id() != doc.doc_id()

    def test_summary_has_required_keys(self):
        doc = self._make().finalize()
        s = doc.summary()
        for key in ["modality", "subtype", "doc_id", "quality", "has_embedding", "text_length", "content_hash"]:
            assert key in s

    def test_to_dict_roundtrip(self):
        doc = self._make().finalize()
        d = doc.to_dict()
        assert d["modality"] == "text"
        assert d["text"] == doc.text


class TestProcessingResult:

    def test_ok_factory(self):
        r = ok("text", "sess1", 0.5, chunks=10, stored=10)
        assert r.success is True
        assert r.chunks == 10

    def test_err_factory(self):
        r = err("OCR_FAILED", "OCR error", modality="pdf")
        assert r.success is False
        assert len(r.errors) == 1
        assert r.errors[0].code == "OCR_FAILED"

    def test_to_dict(self):
        r = ok("image", "sess2", 1.2, chunks=5)
        d = r.to_dict()
        assert d["success"] is True
        assert d["modality"] == "image"


class TestValidationResult:

    def test_validation_ok_factory(self):
        v = validation_ok("pdf", 1024, "application/pdf")
        assert v.valid is True
        assert not v.has_errors

    def test_validation_err_factory(self):
        v = validation_err("EMPTY_FILE", "File is empty", modality="text")
        assert v.valid is False
        assert v.has_errors

    def test_add_warning(self):
        v = validation_ok("audio", 512, "audio/mpeg")
        v.add_warning("Low SNR detected")
        assert v.has_warnings
        assert "Low SNR" in v.warnings[0]


class TestCustomExceptions:

    def test_empty_file_error(self):
        with pytest.raises(EmptyFileError):
            raise EmptyFileError("File is empty")

    def test_duplicate_file_error(self):
        with pytest.raises(DuplicateFileError):
            raise DuplicateFileError("Duplicate detected")

    def test_password_protected_error(self):
        with pytest.raises(PasswordProtectedError):
            raise PasswordProtectedError("Encrypted file")

    def test_malware_detected_error(self):
        with pytest.raises(MalwareDetectedError):
            raise MalwareDetectedError("Malware found")

    def test_unsupported_mime_error(self):
        with pytest.raises(UnsupportedMimeError):
            raise UnsupportedMimeError("MIME not allowed")


class TestMetadataSchema:

    def test_pii_fields_present(self):
        m = UniversalMetadata()
        assert hasattr(m, "pii_redacted")
        assert hasattr(m, "pii_entities_found")

    def test_security_fields_present(self):
        m = UniversalMetadata()
        assert hasattr(m, "malware_scanned")
        assert hasattr(m, "password_protected")
        assert hasattr(m, "has_javascript")
        assert hasattr(m, "has_macros")
        assert hasattr(m, "has_signatures")

    def test_duplicate_fields_present(self):
        m = UniversalMetadata()
        assert hasattr(m, "is_duplicate")
        assert hasattr(m, "duplicate_of")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])