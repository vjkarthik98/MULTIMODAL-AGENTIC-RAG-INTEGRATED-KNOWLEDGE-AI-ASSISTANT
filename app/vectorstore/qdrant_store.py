import math
import time
import uuid
from typing import Any, Dict, List, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram, Gauge
from tenacity import retry, stop_after_attempt, wait_exponential
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
    PayloadSchemaType,
    UpdateStatus,
)

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

# PROMETHEUS METRICS
_upsert_duration = Histogram(
    "qdrant_upsert_duration_seconds",
    "Qdrant upsert duration",
    ["collection"],
)
_search_duration = Histogram(
    "qdrant_search_duration_seconds",
    "Qdrant search duration",
    ["collection"],
)
_upsert_errors = Counter(
    "qdrant_upsert_errors_total",
    "Qdrant upsert errors",
    ["collection", "error_type"],
)
_search_errors = Counter(
    "qdrant_search_errors_total",
    "Qdrant search errors",
    ["collection", "error_type"],
)
from app.core.metrics import circuit_breaker_state as _circuit_breaker_state
_vectors_stored = Gauge(
    "qdrant_vectors_stored_total",
    "Total vectors stored per collection",
    ["collection"],
)


# CIRCUIT BREAKER STATE

class _CircuitBreaker:

    def __init__(self, name: str, fail_max: int = 5, reset_timeout: int = 60) -> None:
        self.name          = name
        self.fail_max      = fail_max
        self.reset_timeout = reset_timeout
        self._failures     = 0
        self._opened_at    = 0.0
        self._open         = False

    def record_success(self) -> None:
        self._failures  = 0
        self._open      = False
        self._opened_at = 0.0
        _circuit_breaker_state.labels(service=self.name).set(0)

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.fail_max:
            self._open      = True
            self._opened_at = time.time()
            _circuit_breaker_state.labels(service=self.name).set(1)
            logger.warning(
                "circuit_breaker_opened",
                service=self.name,
                failures=self._failures,
            )

    def is_open(self) -> bool:
        if self._open:
            if time.time() - self._opened_at >= self.reset_timeout:
                # HALF-OPEN — ALLOW ONE PROBE
                self._open  = False
                self._failures = 0
                _circuit_breaker_state.labels(service=self.name).set(0)
                logger.info("circuit_breaker_half_open", service=self.name)
                return False
            return True
        return False


_cb = _CircuitBreaker(
    name="qdrant",
    fail_max=getattr(settings, "QDRANT_CB_FAIL_MAX", 5),
    reset_timeout=getattr(settings, "QDRANT_CB_RESET_TIMEOUT", 60),
)


class QdrantVectorStore:

    def __init__(self) -> None:

        self.client = (
            QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
                timeout=settings.QDRANT_TIMEOUT,
            )
            if settings.QDRANT_URL
            else QdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=settings.QDRANT_TIMEOUT,
            )
        )

        self.text_collection   = settings.TEXT_COLLECTION_NAME
        self.vision_collection = settings.VISION_COLLECTION_NAME

        self.batch_size = settings.QDRANT_BATCH_SIZE
        self.max_docs   = settings.QDRANT_MAX_DOCS
        self.text_dim   = settings.TEXT_EMBEDDING_DIM
        self.vision_dim = settings.VISION_EMBEDDING_DIM

        self._collection_cache: set         = set()
        self.modality_filter: Optional[str] = None

        logger.info("qdrant_initialized")
        self._warm_cache()

    def _warm_cache(self) -> None:
        """Populate _collection_cache from live Qdrant collections so searches work
        in fresh processes (e.g. eval scripts) that never called _ensure_collection."""
        try:
            existing = {c.name for c in self.client.get_collections().collections}
            for name in existing:
                self._collection_cache.add(name)
        except Exception:
            pass  # non-fatal — cache stays empty, searches degrade gracefully

    # CIRCUIT BREAKER GUARD

    def _check_circuit(self) -> None:
        if _cb.is_open():
            raise RuntimeError("QDRANT_CIRCUIT_OPEN: too many failures, refusing call")

    # RETRY WRAPPER

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _retry(self, fn, *args, **kwargs):
        self._check_circuit()
        try:
            result = fn(*args, **kwargs)
            _cb.record_success()
            return result
        except Exception as exc:
            _cb.record_failure()
            raise

    # COLLECTION EXISTS CHECK

    def _collection_exists(self, name: str) -> bool:
        try:
            return any(
                c.name == name
                for c in self.client.get_collections().collections
            )
        except Exception:
            return False

    # ENSURE COLLECTION WITH PAYLOAD INDEXES

    # Full list of payload fields that need a Qdrant index for filtered search.
    # Adding a field here is enough — _ensure_collection runs on every startup
    # and is idempotent (skips already-existing indexes).
    _PAYLOAD_INDEXES = [
        ("user_id",              PayloadSchemaType.KEYWORD),
        ("session_id",           PayloadSchemaType.KEYWORD),
        ("modality",             PayloadSchemaType.KEYWORD),
        ("doc_id",               PayloadSchemaType.KEYWORD),
        ("source",               PayloadSchemaType.KEYWORD),
        ("language",             PayloadSchemaType.KEYWORD),
        ("content_type",         PayloadSchemaType.KEYWORD),
        ("embedding_space",      PayloadSchemaType.KEYWORD),
        ("deleted_at",           PayloadSchemaType.KEYWORD),
        ("checksum_sha256",      PayloadSchemaType.KEYWORD),
        ("section_number",       PayloadSchemaType.INTEGER),
        ("is_forward_looking",   PayloadSchemaType.BOOL),
    ]

    def _ensure_payload_indexes(self, name: str) -> None:
        """Create payload indexes on an existing collection. Idempotent."""
        for field, schema in self._PAYLOAD_INDEXES:
            try:
                self.client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception as exc:
                err = str(exc).lower()
                if "already exists" in err or "conflict" in err:
                    pass  # already indexed — fine
                else:
                    logger.warning(
                        "payload_index_failed",
                        collection=name,
                        field=field,
                        error=str(exc),
                    )

    def _ensure_collection(self, name: str, dim: int) -> None:
        if name in self._collection_cache:
            return

        if self._collection_exists(name):
            # Check that the stored vector dim matches the current embedder.
            # A mismatch means the embedding model changed — drop and recreate
            # so stale 384-d (MiniLM) vectors don't corrupt a 1024-d (Qwen3) index.
            try:
                info = self.client.get_collection(name)
                vconf = info.config.params.vectors
                existing_dim = getattr(vconf, "size", None)
                if existing_dim is not None and existing_dim != dim:
                    logger.info(
                        "qdrant_recreate_collection_dim_mismatch",
                        name=name,
                        existing_dim=existing_dim,
                        new_dim=dim,
                    )
                    self.client.delete_collection(name)
                    self.client.create_collection(
                        collection_name=name,
                        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                    )
            except Exception as exc:
                logger.warning(
                    "qdrant_dim_check_failed",
                    collection=name,
                    error=str(exc),
                )
        else:
            logger.info("qdrant_create_collection", collection=name, dim=dim)
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

        # Always (re-)create payload indexes — runs on new AND existing collections
        # so that adding a new field to _PAYLOAD_INDEXES takes effect on restart.
        self._ensure_payload_indexes(name)
        self._collection_cache.add(name)

    # EMBEDDING VALIDATION

    def _valid_vector(self, emb: List[float], expected_dim: int) -> bool:
        if not isinstance(emb, list):
            return False
        if len(emb) != expected_dim:
            return False
        if any(math.isnan(v) or math.isinf(v) for v in emb):
            return False
        return True

    # DETERMINISTIC VECTOR ID — FILE_ID + CHUNK_INDEX FOR IDEMPOTENCY

    def _vector_id(self, doc_id: str, chunk_id: Any) -> str:
        base = f"{doc_id}:{chunk_id}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, base))

    # PAYLOAD BUILDER

    def _payload(self, d: Any, user_id: Optional[str] = None) -> Dict[str, Any]:
        s = dict(getattr(d, "structure", {}) or {})
        modality = getattr(d, "modality", "text")

        payload: Dict[str, Any] = {
            "text":            str(getattr(d, "text", "") or "")[:settings.QDRANT_TEXT_MAX_CHARS],
            "user_id":         user_id or s.get("user_id"),
            "doc_id":          s.get("doc_id"),
            "chunk_id":        getattr(d, "chunk_id", None),
            "modality":        modality,
            "subtype":         getattr(d, "subtype", None),
            "content_type":    s.get("content_type"),
            "session_id":      s.get("session_id"),
            "embedding_space": s.get("embedding_space", "text"),
            "source":          str(getattr(d, "source", "") or "")[:200],
            "source_type":     getattr(d, "source_type", None),
            "page":            getattr(d, "page", None),
            "language":        s.get("language"),
            "tags":            s.get("tags", []),
            "parent_id":       s.get("parent_id"),
            "hierarchy_level": s.get("hierarchy_level"),
            "checksum":        s.get("file_hash"),
            "ingestion_time":  s.get("ingestion_time"),
            "section_id":       s.get("section_id"),
            "section_title":    s.get("section_title"),
            "error_markers":    s.get("error_markers") or [],
            "doc_version":      s.get("doc_version"),
            "title_mismatch":   bool(s.get("title_mismatch", False)),
            "section_number":   s.get("section_number"),
            "is_forward_looking": bool(s.get("is_forward_looking", False)),
            "deleted_at":       None,
        }

        _source_type = getattr(d, "source_type", "") or ""

        # PDF LOCATOR FIELDS — sub-page chunk position for precise citation
        # backward-compat path: modality="text" + source_type="pdf"
        if _source_type == "pdf":
            if s.get("sub_chunk_index") is not None:
                payload["sub_chunk_index"] = int(s["sub_chunk_index"])
            if s.get("total_sub_chunks") is not None:
                payload["total_sub_chunks"] = int(s["total_sub_chunks"])

        # PDF RICH METADATA — from PdfChunker (modality="pdf")
        # Provides full Phase 1.2/2.2 metadata for finance-grade retrieval
        if modality == "pdf":
            for _k in ("page_number", "page_range", "chunk_type", "section_hierarchy",
                       "table_title", "column_headers", "fiscal_years", "row_range", "footnotes",
                       "footnote_markers", "has_figure", "figure_path", "is_ocr",
                       "caption", "ocr_text", "chunk_hash_id", "source_file"):
                _v = s.get(_k)
                if _v is not None:
                    payload[_k] = _v
            if s.get("finance_entities"):
                payload["finance_entities"] = list(s["finance_entities"])[:20]
            if s.get("char_start") is not None:
                payload["char_start"] = int(s["char_start"])
            if s.get("char_end") is not None:
                payload["char_end"] = int(s["char_end"])
            if s.get("token_count") is not None:
                payload["token_count"] = int(s["token_count"])

        # DOCX RICH METADATA — from DocxChunker (modality="docx")
        # Provides full Phase 1.3/2.3 metadata for heading-aware finance retrieval
        if modality == "docx":
            for _k in ("heading", "heading_hierarchy", "heading_level", "chunk_type",
                       "paragraph_index", "table_index", "column_headers", "row_range",
                       "has_bold_terms", "has_italic_terms", "defined_terms",
                       "clause_numbers", "chunk_hash_id", "source_file"):
                _v = s.get(_k)
                if _v is not None:
                    payload[_k] = _v
            if s.get("finance_entities") is not None:
                payload["finance_entities"] = list(s["finance_entities"])[:20]
            if s.get("char_start") is not None:
                payload["char_start"] = int(s["char_start"])
            if s.get("char_end") is not None:
                payload["char_end"] = int(s["char_end"])
            if s.get("token_count") is not None:
                payload["token_count"] = int(s["token_count"])

        # DOCX HEADING LEVEL (backward-compat — backward-compat path sets source_type="word")
        if _source_type == "word" and s.get("heading_level") is not None:
            payload["heading_level"] = int(s["heading_level"])

        # XLSX RICH METADATA — from XlsxChunker (modality="xlsx")
        if modality == "xlsx":
            for _k in ("sheet_name", "sheet_index", "sheet_type", "chunk_type",
                       "chunk_hash_id", "source_file", "table_region", "semantic_group",
                       "unit_scale", "currency", "display_format", "is_hidden",
                       "has_formulas_resolved", "named_ranges_in_chunk",
                       "column_headers", "row_range"):
                _v = s.get(_k)
                if _v is not None:
                    payload[_k] = _v
            if s.get("finance_entities") is not None:
                payload["finance_entities"] = list(s["finance_entities"])[:20]
            if s.get("token_count") is not None:
                payload["token_count"] = int(s["token_count"])

        # EXCEL backward-compat — old path: modality="table", source_type="excel"
        if _source_type == "excel" and modality != "xlsx":
            if s.get("sheet"):
                payload["sheet_name"] = str(s["sheet"])
            if s.get("row_start") is not None:
                payload["row_start"] = int(s["row_start"])
            if s.get("row_end") is not None:
                payload["row_end"] = int(s["row_end"])

        # TXT/TEXT TRANSCRIPT FIELDS — speaker attribution and call section for earnings call retrieval
        if modality in ("text", "txt"):
            if s.get("speaker"):
                payload["speaker"] = str(s["speaker"])
            if s.get("call_section"):
                payload["call_section"] = str(s["call_section"])
            if s.get("is_transcript"):
                payload["is_transcript"] = bool(s["is_transcript"])
            if s.get("chunk_type"):
                payload["chunk_type"] = str(s["chunk_type"])
            if s.get("char_start") is not None:
                payload["char_start"] = int(s["char_start"])
            if s.get("char_end") is not None:
                payload["char_end"] = int(s["char_end"])
            if s.get("token_count") is not None:
                payload["token_count"] = int(s["token_count"])
            if s.get("finance_entities"):
                payload["finance_entities"] = list(s["finance_entities"])[:20]

        # AUDIO RICH METADATA — AudioChunker Phase 1.6/2.6/3.6 fields.
        # Keys: AudioChunker uses start_timestamp/end_timestamp; legacy ingest uses
        # timestamp_start/timestamp_end. Both are normalised here so retrievers
        # can use either key interchangeably.
        if modality in ("audio", "mp3"):
            # Timestamps — accept both naming conventions from old and new ingest paths
            ts_start = s.get("start_timestamp") or s.get("timestamp_start")
            ts_end   = s.get("end_timestamp")   or s.get("timestamp_end")
            if ts_start is not None:
                payload["start_timestamp"] = float(ts_start)
                payload["timestamp_start"] = float(ts_start)    # backward-compat alias
            if ts_end is not None:
                payload["end_timestamp"]   = float(ts_end)
                payload["timestamp_end"]   = float(ts_end)      # backward-compat alias
            if s.get("duration_seconds") is not None:
                payload["duration_seconds"] = float(s["duration_seconds"])

            # Speaker fields
            spk_label = s.get("speaker_label") or s.get("speaker")
            if spk_label:
                payload["speaker_label"] = str(spk_label)
                payload["speaker"]       = str(spk_label)       # backward-compat alias
            if s.get("speaker_name"):
                payload["speaker_name"] = str(s["speaker_name"])
            if s.get("speaker_role"):
                payload["speaker_role"] = str(s["speaker_role"])

            # Earnings-call section and topic
            if s.get("call_section"):
                payload["call_section"]  = str(s["call_section"])
            if s.get("topic_section"):
                payload["topic_section"] = str(s["topic_section"])

            # Chunk statistics
            if s.get("word_count") is not None:
                payload["word_count"]  = int(s["word_count"])
            if s.get("token_count") is not None:
                payload["token_count"] = int(s["token_count"])

            # Q&A and earnings-call flags
            if s.get("is_question") is not None:
                payload["is_question"]     = bool(s["is_question"])
            if s.get("is_answer") is not None:
                payload["is_answer"]       = bool(s["is_answer"])
            if s.get("is_earnings_call") is not None:
                payload["is_earnings_call"] = bool(s["is_earnings_call"])

            # Audio quality signals forwarded from ingestor
            if s.get("snr") is not None:
                payload["snr"]               = float(s["snr"])
            if s.get("snr_degraded") is not None:
                payload["snr_degraded"]      = bool(s["snr_degraded"])
            if s.get("clipping_detected") is not None:
                payload["clipping_detected"] = bool(s["clipping_detected"])

            # Finance entity map — truncated for index efficiency
            fe = s.get("finance_entities")
            if fe and isinstance(fe, dict):
                payload["finance_entities"] = {
                    k: v[:10] if isinstance(v, list) else v
                    for k, v in fe.items()
                }

            # Verbatim transcript stored alongside .text for re-rank display.
            # Matches QDRANT_TEXT_MAX_CHARS (2000) rather than an inconsistent
            # separate 1000-char cap — some chunks run well past 1000 chars
            # and were losing their tail (e.g. the exact rate figure sentence)
            # from the citation/context view even though .text kept it.
            if s.get("transcript"):
                payload["transcript"] = str(s["transcript"])[:settings.QDRANT_TEXT_MAX_CHARS]

            # Deterministic chunk hash for dedup
            if s.get("chunk_hash_id"):
                payload["chunk_hash_id"] = str(s["chunk_hash_id"])

            # FinBERT finance-tone annotation (added post-embedding by
            # AudioEmbedder._annotate_finbert_tone) — was missing from this
            # whitelist entirely, so the model ran but its output never
            # reached Qdrant.
            if s.get("finance_tone"):
                payload["finance_tone"] = str(s["finance_tone"])
            if s.get("finance_tone_score") is not None:
                payload["finance_tone_score"] = float(s["finance_tone_score"])

        # VIDEO RICH METADATA — VideoChunker Phase 1.7/2.7/3.7 fields.
        # Mirrors the audio block above; accepts both timestamp naming conventions.
        if modality in ("video", "mp4"):
            ts_start = s.get("start_timestamp") or s.get("timestamp_start")
            ts_end   = s.get("end_timestamp")   or s.get("timestamp_end")
            if ts_start is not None:
                payload["start_timestamp"] = float(ts_start)
                payload["timestamp_start"] = float(ts_start)
            if ts_end is not None:
                payload["end_timestamp"]   = float(ts_end)
                payload["timestamp_end"]   = float(ts_end)
            if s.get("duration_seconds") is not None:
                payload["duration_seconds"] = float(s["duration_seconds"])

            # Speaker fields — VideoChunker stores speaker_label/speaker_name/speaker_role
            spk_label = s.get("speaker_label") or s.get("speaker")
            if spk_label:
                payload["speaker_label"] = str(spk_label)
                payload["speaker"]       = str(spk_label)
            if s.get("speaker_name"):
                payload["speaker_name"] = str(s["speaker_name"])
            if s.get("speaker_role"):
                payload["speaker_role"] = str(s["speaker_role"])

            # Call section and topic
            if s.get("call_section"):
                payload["call_section"]  = str(s["call_section"])
            if s.get("topic_section"):
                payload["topic_section"] = str(s["topic_section"])

            # Chunk statistics
            if s.get("word_count") is not None:
                payload["word_count"]  = int(s["word_count"])
            if s.get("token_count") is not None:
                payload["token_count"] = int(s["token_count"])

            # Q&A and content flags
            if s.get("is_question") is not None:
                payload["is_question"]     = bool(s["is_question"])
            if s.get("is_answer") is not None:
                payload["is_answer"]       = bool(s["is_answer"])
            if s.get("is_earnings_call") is not None:
                payload["is_earnings_call"] = bool(s["is_earnings_call"])

            # Audio quality signals (from video's audio track)
            if s.get("snr") is not None:
                payload["snr"]               = float(s["snr"])
            if s.get("snr_degraded") is not None:
                payload["snr_degraded"]      = bool(s["snr_degraded"])
            if s.get("clipping_detected") is not None:
                payload["clipping_detected"] = bool(s["clipping_detected"])

            # Visual metadata
            if s.get("has_slide_content") is not None:
                payload["has_slide_content"] = bool(s["has_slide_content"])
            if s.get("has_finance_signal") is not None:
                payload["has_finance_signal"] = bool(s["has_finance_signal"])
            if s.get("slide_numbers_covered"):
                payload["slide_numbers_covered"] = list(s["slide_numbers_covered"])[:20]

            # Frame captions — serialised (truncated) for display and re-rank
            frame_caps = s.get("frame_captions") or []
            if frame_caps:
                # Store compact version: timestamp + caption + slide_number
                compact = []
                for fc in frame_caps[:10]:
                    if isinstance(fc, dict):
                        compact.append({
                            "frame_timestamp": fc.get("frame_timestamp"),
                            "frame_caption":   (fc.get("frame_caption") or "")[:200],
                            "ocr_text":        (fc.get("ocr_text") or "")[:100],
                            "slide_number":    fc.get("slide_number"),
                            "scene_change":    fc.get("scene_change", False),
                            "frame_path":      fc.get("frame_path", ""),
                        })
                if compact:
                    payload["frame_captions"] = compact

            # Finance entities map
            fe = s.get("finance_entities")
            if fe and isinstance(fe, dict):
                payload["finance_entities"] = {
                    k: v[:10] if isinstance(v, list) else v
                    for k, v in fe.items()
                }

            # Verbatim transcript for re-rank display
            if s.get("transcript"):
                payload["transcript"] = str(s["transcript"])[:1000]

        # XLSX DUAL EMBEDDING — store alt vector in payload so Phase 5 retrieval
        # can reconstruct it for structural table search without a named-vector
        # collection migration.
        emb_alt = getattr(d, "embedding_alt", None)
        if emb_alt and isinstance(emb_alt, list) and len(emb_alt) > 0:
            payload["embedding_alt"] = emb_alt

        # VIDEO NAMED VECTORS — audio-only and visual-only BGE embeddings stored
        # in payload until Qdrant multi-vector migration runs.
        emb_audio  = getattr(d, "embedding_audio", None)
        emb_visual = getattr(d, "embedding_visual", None)
        if emb_audio and isinstance(emb_audio, list) and len(emb_audio) > 0:
            payload["embedding_audio"]     = emb_audio
            payload["has_audio_embedding"] = True
        if emb_visual and isinstance(emb_visual, list) and len(emb_visual) > 0:
            payload["embedding_visual"]     = emb_visual
            payload["has_visual_embedding"] = True

        # VISION QUALITY FIELDS — stored for image and video frame chunks so
        # retrieval can filter/rank on content quality rather than blind similarity.
        if modality in ("image", "video"):
            # Sharpness proxy: 0.0 = very blurry, 1.0 = sharp
            if s.get("blur_score") is not None:
                payload["blur_score"] = float(s["blur_score"])

            # Caption richness: word diversity × length, [0, 1]
            if s.get("caption_confidence") is not None:
                payload["caption_confidence"] = float(s["caption_confidence"])

            # Overall ingest quality composite score
            if s.get("quality_score") is not None:
                payload["quality_score"] = float(s["quality_score"])

            # Low-content flag (solid background, test patterns, etc.)
            if s.get("solid_color") is not None:
                payload["solid_color"] = bool(s["solid_color"])

            # Watermark / confidential stamp detected in OCR/caption
            if s.get("watermark_detected") is not None:
                payload["watermark_detected"] = bool(s["watermark_detected"])

            # Face count (anonymised — count only, never identity)
            if s.get("face_count") is not None:
                payload["face_count"] = int(s["face_count"])

            # Perceptual hash for near-duplicate detection at query time
            if s.get("perceptual_hash") is not None:
                payload["perceptual_hash"] = str(s["perceptual_hash"])

            # Dominant palette — list of up to 5 hex colour strings
            if s.get("dominant_colors"):
                payload["dominant_colors"] = list(s["dominant_colors"])[:5]

            # Pixel dimensions after resize
            if s.get("image_width") is not None:
                payload["image_width"]  = int(s["image_width"])
                payload["image_height"] = int(s.get("image_height", 0))

            # Caption text stored for display / hybrid re-rank
            if s.get("caption"):
                payload["caption"] = str(s["caption"])[:500]

            # Image classification and title (Phase 1.5/3.5)
            if s.get("image_type"):
                payload["image_type"] = str(s["image_type"])
            if s.get("image_title"):
                payload["image_title"] = str(s["image_title"])[:200]

            # OCR text — truncated for filtering/re-rank
            if s.get("ocr_text"):
                payload["ocr_text"] = str(s["ocr_text"])[:500]

            # Extracted finance numbers (up to 20 for retrieval filtering)
            if s.get("extracted_numbers"):
                payload["extracted_numbers"] = list(s["extracted_numbers"])[:20]

            # Time period and data series for chart queries
            if s.get("time_period"):
                payload["time_period"] = str(s["time_period"])
            if s.get("data_series"):
                payload["data_series"] = [str(x) for x in s["data_series"]][:8]

            # Asset path so the UI can render the original file / frame
            if s.get("asset_path"):
                payload["asset_path"] = str(s["asset_path"])
            elif s.get("frame_path"):
                payload["asset_path"] = str(s["frame_path"])
            elif s.get("thumbnail_path"):
                payload["asset_path"] = str(s["thumbnail_path"])

            # Video-specific quality signals
            if s.get("timestamp_sec") is not None:
                payload["timestamp_sec"] = float(s["timestamp_sec"])

            # Segment timestamps (video frames and audio chunks share these)
            if s.get("timestamp_start") is not None:
                payload["timestamp_start"] = float(s["timestamp_start"])

            if s.get("timestamp_end") is not None:
                payload["timestamp_end"] = float(s["timestamp_end"])

            if s.get("speaker"):
                payload["speaker"] = str(s["speaker"])

            if s.get("alignment_score") is not None:
                payload["alignment_score"] = float(s["alignment_score"])

            if s.get("is_hdr") is not None:
                payload["is_hdr"] = bool(s["is_hdr"])

            if s.get("conflict_flag") is not None:
                payload["conflict_flag"] = bool(s["conflict_flag"])

            if s.get("video_duration") is not None:
                payload["video_duration"] = float(s["video_duration"])

            if s.get("fps") is not None:
                payload["fps"] = float(s["fps"])

            if s.get("frame_index") is not None:
                payload["frame_index"] = int(s["frame_index"])

        return payload

    # INSERT DOCUMENTS WITH IDEMPOTENT UPSERT

    def insert_documents(self, documents: List[Any], session_id: str = "", user_id: Optional[str] = None) -> None:

        if not documents:
            return

        with tracer.start_as_current_span("qdrant_insert") as span:
            span.set_attribute("docs.input", len(documents))

            start     = time.time()
            documents = documents[:self.max_docs]

            text_points:   List[PointStruct] = []
            vision_points: List[PointStruct] = []
            skipped = 0

            for d in documents:
                emb   = getattr(d, "embedding", None)
                s     = getattr(d, "structure", {}) or {}
                space = s.get("embedding_space", "text")

                doc_id   = s.get("doc_id", str(uuid.uuid4()))
                chunk_id = getattr(d, "chunk_id", 0)
                point_id = self._vector_id(doc_id, chunk_id)

                if space == "vision":
                    if not self._valid_vector(emb, self.vision_dim):
                        skipped += 1
                        continue
                    vision_points.append(
                        PointStruct(
                            id=point_id,
                            vector=emb,
                            payload=self._payload(d, user_id),
                        )
                    )
                else:
                    if not self._valid_vector(emb, self.text_dim):
                        if isinstance(emb, list) and len(emb) > 0 and len(emb) != self.text_dim:
                            raise ValueError(
                                f"EMBEDDING_DIM_MISMATCH: got {len(emb)}, expected "
                                f"{self.text_dim}. Re-ingest after running "
                                f"app/bin/migrate_qdrant_dim.py if you changed "
                                f"TEXT_EMBEDDING_DIM."
                            )
                        skipped += 1
                        continue
                    text_points.append(
                        PointStruct(
                            id=point_id,
                            vector=emb,
                            payload=self._payload(d, user_id),
                        )
                    )

            def _insert(collection_name: str, points: List[PointStruct]) -> None:
                self._ensure_collection(
                    collection_name,
                    self.text_dim if collection_name == self.text_collection else self.vision_dim,
                )
                for i in range(0, len(points), self.batch_size):
                    batch = points[i:i + self.batch_size]
                    result = self._retry(
                        self.client.upsert,
                        collection_name=collection_name,
                        points=batch,
                    )
                    if result and hasattr(result, "status"):
                        if result.status != UpdateStatus.COMPLETED:
                            logger.warning(
                                "qdrant_upsert_not_completed",
                                collection=collection_name,
                                status=str(result.status),
                            )

            try:
                if text_points:
                    _insert(self.text_collection, text_points)
                    _vectors_stored.labels(collection=self.text_collection).inc(len(text_points))

                if vision_points:
                    _insert(self.vision_collection, vision_points)
                    _vectors_stored.labels(collection=self.vision_collection).inc(len(vision_points))

            except Exception as exc:
                _upsert_errors.labels(
                    collection="unknown",
                    error_type=type(exc).__name__,
                ).inc()
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
                raise

            total    = len(text_points) + len(vision_points)
            latency  = round(time.time() - start, 2)
            throughput = round(total / max(latency, 1e-6), 1)

            _upsert_duration.labels(collection=self.text_collection).observe(latency)

            span.set_attribute("docs.stored", total)
            span.set_attribute("docs.skipped", skipped)
            span.set_status(Status(StatusCode.OK))

            logger.info(
                "qdrant_insert_success",
                text=len(text_points),
                vision=len(vision_points),
                skipped=skipped,
                throughput_per_sec=throughput,
                latency=latency,
                session_id=session_id,
            )

    # SOFT DELETE — MARK DELETED_AT WITHOUT REMOVING VECTOR

    def soft_delete(self, doc_id: str, session_id: str = "") -> None:
        if not doc_id:
            return

        with tracer.start_as_current_span("qdrant_soft_delete") as span:
            span.set_attribute("doc.id", doc_id)

            deleted_at = time.time()

            for collection in (self.text_collection, self.vision_collection):
                if collection not in self._collection_cache:
                    continue
                try:
                    self._retry(
                        self.client.set_payload,
                        collection_name=collection,
                        payload={"deleted_at": str(deleted_at)},
                        points=Filter(
                            must=[
                                FieldCondition(
                                    key="doc_id",
                                    match=MatchValue(value=doc_id),
                                )
                            ]
                        ),
                    )
                    logger.info(
                        "qdrant_soft_deleted",
                        doc_id=doc_id,
                        collection=collection,
                        session_id=session_id,
                    )
                except Exception as exc:
                    logger.error(
                        "qdrant_soft_delete_failed",
                        doc_id=doc_id,
                        collection=collection,
                        error=str(exc),
                    )

    # RE-INDEX — OVERWRITE ALL VECTORS FOR A FILE ON RE-INGESTION

    def reindex_by_doc_id(self, doc_id: str, new_documents: List[Any], session_id: str = "") -> None:
        self.delete_by_doc_id(doc_id, session_id=session_id)
        self.insert_documents(new_documents, session_id=session_id)
        logger.info("qdrant_reindex_complete", doc_id=doc_id, session_id=session_id)

    # HARD DELETE BY DOC_ID

    def delete_by_doc_id(self, doc_id: str, session_id: str = "") -> None:
        if not doc_id:
            return

        for collection in (self.text_collection, self.vision_collection):
            if collection not in self._collection_cache:
                continue
            try:
                self._retry(
                    self.client.delete,
                    collection_name=collection,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="doc_id",
                                match=MatchValue(value=doc_id),
                            )
                        ]
                    ),
                )
                logger.info(
                    "qdrant_doc_deleted",
                    doc_id=doc_id,
                    collection=collection,
                    session_id=session_id,
                )
            except Exception as exc:
                logger.error(
                    "qdrant_delete_by_doc_failed",
                    doc_id=doc_id,
                    collection=collection,
                    error=str(exc),
                )

    # GDPR PURGE — DELETE ALL CHUNKS BY USER_ID OR SESSION_ID

    def gdpr_purge(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> None:
        if not user_id and not session_id:
            raise ValueError("GDPR_PURGE_REQUIRES_USER_ID_OR_SESSION_ID")

        conditions = []
        if session_id:
            conditions.append(
                FieldCondition(key="session_id", match=MatchValue(value=session_id))
            )
        if user_id:
            conditions.append(
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            )

        purge_filter = Filter(must=conditions)

        for collection in (self.text_collection, self.vision_collection):
            try:
                self._retry(
                    self.client.delete,
                    collection_name=collection,
                    points_selector=purge_filter,
                )
                logger.info(
                    "qdrant_gdpr_purge_complete",
                    collection=collection,
                    session_id=session_id,
                    user_id=user_id,
                )
            except Exception as exc:
                # Collection may not exist yet — that's fine, nothing to delete
                logger.warning(
                    "qdrant_gdpr_purge_skipped",
                    collection=collection,
                    error=str(exc),
                )

    # FILTER BUILDER — EXCLUDES SOFT-DELETED DOCS, ISOLATES BY USER

    def _build_filter(
        self,
        session_id: Optional[str] = None,
        exclude_deleted: bool = True,
        user_id: Optional[str] = None,
    ) -> Optional[Filter]:
        conditions = []

        # user_id isolation is the primary tenant boundary
        if user_id:
            conditions.append(
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            )
        else:
            logger.warning(
                "qdrant_build_filter_no_user_id",
                warning="_build_filter called without user_id — unscoped filter will match all tenants",
            )

        # session_id is a correlation/tracing field stored in the payload; it is
        # NOT used as a retrieval filter. Documents are ingested with session_id
        # "default" but queries arrive with per-request session IDs — filtering
        # on session_id would return zero results for every knowledge-base query.

        if self.modality_filter:
            conditions.append(
                FieldCondition(key="modality", match=MatchValue(value=self.modality_filter))
            )

        # NOTE: soft-delete filter via IsNullCondition requires a bool/integer
        # payload index which Qdrant Cloud does not support on KEYWORD-indexed fields.
        # Soft-deleted docs are excluded at insert time by not re-upserting them;
        # the deleted_at field is only used by the GDPR purge path, not search.

        return Filter(must=conditions) if conditions else None

    # INTERNAL SEARCH

    def _search(
        self,
        collection: str,
        vector: List[float],
        limit: int,
        session_id: Optional[str],
        score_threshold: float = 0.0,
        exclude_deleted: bool = True,
        user_id: Optional[str] = None,
        extra_filter: Optional["Filter"] = None,
        vector_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:

        if collection not in self._collection_cache:
            logger.warning(
                "qdrant_search_collection_not_ready",
                collection=collection,
                session_id=session_id,
            )
            return []

        self._check_circuit()

        start = time.time()

        with tracer.start_as_current_span("qdrant_search") as span:
            span.set_attribute("collection", collection)
            span.set_attribute("limit", limit)

            try:
                base_filter = self._build_filter(session_id, exclude_deleted, user_id)
                if extra_filter is not None and base_filter is not None:
                    merged_must = list(base_filter.must or []) + list(extra_filter.must or [])
                    from qdrant_client.models import Filter as _Filter
                    query_filter = _Filter(must=merged_must) if merged_must else None
                elif extra_filter is not None:
                    query_filter = extra_filter
                else:
                    query_filter = base_filter
                _qp_kwargs: dict = dict(
                    collection_name=collection,
                    query=vector,
                    limit=limit,
                    query_filter=query_filter,
                    score_threshold=score_threshold if score_threshold > 0 else None,
                )
                if vector_name:
                    _qp_kwargs["using"] = vector_name
                res    = self._retry(self.client.query_points, **_qp_kwargs)
                points = getattr(res, "points", [])

                results = [
                    {
                        "text":     p.payload.get("text"),
                        "score":    float(p.score),
                        "metadata": p.payload,
                    }
                    for p in points
                    if p.payload.get("text")
                ]

                latency = round(time.time() - start, 3)
                _search_duration.labels(collection=collection).observe(latency)

                span.set_attribute("results.count", len(results))
                span.set_status(Status(StatusCode.OK))

                logger.debug(
                    "qdrant_search_success",
                    collection=collection,
                    results=len(results),
                    latency=latency,
                    session_id=session_id,
                )

                return results

            except Exception as exc:
                latency = round(time.time() - start, 3)
                _search_errors.labels(
                    collection=collection,
                    error_type=type(exc).__name__,
                ).inc()
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                span.record_exception(exc)
                # A named vector that doesn't exist on this collection (e.g.
                # "vector_alt" — only some modalities have a secondary
                # table/numeric embedding space) is an EXPECTED, already-
                # handled fallback for callers like
                # hybrid_retriever._vector_search_alt(), not a real failure.
                # Logging it at ERROR every time drowns out genuine Qdrant
                # problems in the logs.
                _is_missing_named_vector = (
                    vector_name is not None
                    and "not existing vector name" in str(exc).lower()
                )
                _log = logger.warning if _is_missing_named_vector else logger.error
                _log(
                    "qdrant_search_missing_named_vector" if _is_missing_named_vector
                    else "qdrant_search_failed",
                    collection=collection,
                    vector_name=vector_name,
                    session_id=session_id,
                    error=str(exc),
                )
                return []

    # PUBLIC SEARCH — TEXT COLLECTION

    def search_text(
        self,
        query_vector: List[float],
        limit: int = None,
        session_id: Optional[str] = None,
        score_threshold: float = 0.0,
        user_id: Optional[str] = None,
        extra_filter=None,
        vector_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self._search(
            self.text_collection,
            query_vector,
            limit or settings.RAG_TOP_K,
            session_id,
            score_threshold,
            user_id=user_id,
            extra_filter=extra_filter,
            vector_name=vector_name,
        )

    # PUBLIC SEARCH — VISION COLLECTION

    def search_vision(
        self,
        query_vector: List[float],
        limit: int = None,
        session_id: Optional[str] = None,
        score_threshold: float = 0.0,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return self._search(
            self.vision_collection,
            query_vector,
            limit or settings.RAG_TOP_K,
            session_id,
            score_threshold,
            user_id=user_id,
        )

    # MODALITY FILTER SETTER

    def set_modality_filter(self, modality: Optional[str]) -> None:
        self.modality_filter = modality

    # DELETE BY SESSION

    def delete_by_session(self, session_id: str) -> None:
        if not session_id:
            return

        for collection in (self.text_collection, self.vision_collection):
            if collection not in self._collection_cache:
                continue
            try:
                self._retry(
                    self.client.delete,
                    collection_name=collection,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="session_id",
                                match=MatchValue(value=session_id),
                            )
                        ]
                    ),
                )
                logger.info(
                    "qdrant_session_deleted",
                    collection=collection,
                    session_id=session_id,
                )
            except Exception as exc:
                logger.error(
                    "qdrant_delete_failed",
                    collection=collection,
                    session_id=session_id,
                    error=str(exc),
                )

    # PAYLOAD FIELD SEARCH — USED BY DEDUP CHECK

    def search_by_payload(
        self,
        field: str,
        value: str,
        user_id: str = "",
        session_id: str = "",
        limit: int = 1,
    ) -> List[Any]:
        if not user_id:
            raise ValueError("search_by_payload requires a non-empty user_id for tenant isolation")
        results: List[Any] = []
        for collection in (self.text_collection, self.vision_collection):
            if collection not in self._collection_cache:
                continue
            try:
                points, _ = self.client.scroll(
                    collection_name=collection,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(key=field, match=MatchValue(value=value)),
                            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                        ]
                    ),
                    limit=limit,
                    with_payload=False,
                    with_vectors=False,
                )
                results.extend(points)
                if results:
                    break
            except Exception as exc:
                logger.warning(
                    "search_by_payload_failed",
                    collection=collection,
                    field=field,
                    error=str(exc),
                )
        return results

    # VISION COLLECTION EMPTY CHECK

    def has_vision_data(self) -> bool:
        """Returns True only if vision_collection has at least one indexed point."""
        try:
            info = self.client.get_collection(self.vision_collection)
            return (info.points_count or 0) > 0
        except Exception:
            return False

    # COLLECTION STATS

    def collection_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}

        for name in (self.text_collection, self.vision_collection):
            try:
                info        = self.client.get_collection(name)
                stats[name] = {
                    "points_count":  info.points_count,
                    "vectors_count": info.vectors_count,
                    "status":        str(info.status),
                }
            except Exception as exc:
                stats[name] = {"error": str(exc)}

        return stats

    # HEALTH CHECK

    def health_check(self) -> Dict[str, Any]:
        return {
            "circuit_open":    _cb.is_open(),
            "circuit_failures": _cb._failures,
            "collections":     list(self._collection_cache),
            "stats":           self.collection_stats(),
        }


def initialize_qdrant() -> None:
    """Explicit startup init — ensures both collections exist with correct dims."""
    from app.core.infra_registry import infra
    store = infra.get_vector_store()
    store._ensure_collection(settings.TEXT_COLLECTION_NAME, settings.TEXT_EMBEDDING_DIM)
    store._ensure_collection(settings.VISION_COLLECTION_NAME, settings.VISION_EMBEDDING_DIM)
    logger.info("qdrant_init_complete")


