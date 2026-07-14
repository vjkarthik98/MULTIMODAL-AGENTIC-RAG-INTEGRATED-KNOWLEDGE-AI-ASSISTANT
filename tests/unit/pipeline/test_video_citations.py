"""Unit tests for the video-modality citation and answer-grounding helpers in
app/pipeline/rag_pipeline.py — the speaker-name resolver, frame-caption
cleaner, query-aspect splitter, and frame stock-price masker added for the
Qwen2.5-14B upgrade. Pure-function tests only: no Qdrant, no LLM, no network.
"""

from app.pipeline.rag_pipeline import (
    _build_p248_sources,
    _clean_frame_label,
    _doc_is_frame_like,
    _mask_frame_stock_price,
    _split_frame_caption,
    _split_query_aspects,
    _video_speaker_name,
)


# ---------------------------------------------------------------------------
# _split_frame_caption
# ---------------------------------------------------------------------------

class TestSplitFrameCaption:

    def test_splits_on_screen_marker(self):
        cap, ocr = _split_frame_caption(
            "AAPL, Apple Inc. at $283.80. [ON-SCREEN]: PANW CDNS TTd Mar")
        assert cap == "AAPL, Apple Inc. at $283.80."
        assert ocr == "PANW CDNS TTd Mar"

    def test_no_marker_returns_full_text_no_ocr(self):
        cap, ocr = _split_frame_caption("just a caption, no ticker")
        assert cap == "just a caption, no ticker"
        assert ocr is None

    def test_empty_text(self):
        cap, ocr = _split_frame_caption("")
        assert cap == ""
        assert ocr is None

    def test_none_text(self):
        cap, ocr = _split_frame_caption(None)
        assert cap == ""
        assert ocr is None


# ---------------------------------------------------------------------------
# _clean_frame_label
# ---------------------------------------------------------------------------

class TestCleanFrameLabel:

    def test_extracts_eps_and_sales_beat(self):
        label = _clean_frame_label(
            "AAPL: Apple, Q4 2025 EPS $1.85 Beats $1.76 Estimate, "
            "Sales $102.466B Beats $102.171B Estimate")
        assert label == "EPS $1.85 beats $1.76 · Sales $102.466B beats $102.171B"

    def test_extracts_eps_only_comma_form(self):
        label = _clean_frame_label(
            "AAPL, Apple Inc. at $280.20, Q4 2025 earnings beat estimate, "
            "$1.85 EPS, $1.76 estimate, $102.466B revenue, $102.171B estimate.")
        assert label is not None
        assert "1.85" in label and "1.76" in label

    def test_price_only_caption_returns_none(self):
        # A bare stock-price/chart caption has no beat-ticker metric to extract.
        label = _clean_frame_label("AAPL, Apple Inc. at $283.80, up 4.96% from the previous day.")
        assert label is None

    def test_slide_number_fallback(self):
        label = _clean_frame_label("This is Slide 4 of the presentation deck.")
        assert label == "Slide 4"

    def test_empty_caption(self):
        assert _clean_frame_label("") is None
        assert _clean_frame_label(None) is None


# ---------------------------------------------------------------------------
# _split_query_aspects
# ---------------------------------------------------------------------------

class TestSplitQueryAspects:

    def test_multipart_question_splits_into_aspects(self):
        q = ("What did Tim Cook say about Apple's December quarter 2025 guidance, "
            "the iPhone Air reception, and Apple's approach to AI foundation "
            "models and M&A?")
        aspects = _split_query_aspects(q)
        assert len(aspects) >= 3

    def test_single_focused_question_yields_few_aspects(self):
        q = "What was Apple's full-year FY2025 annual revenue?"
        aspects = _split_query_aspects(q)
        assert len(aspects) <= 2

    def test_max_aspects_respected(self):
        q = "guidance revenue, iphone air reception, ai models, mergers acquisitions, gross margin, operating expenses"
        aspects = _split_query_aspects(q, max_aspects=3)
        assert len(aspects) <= 3

    def test_empty_query(self):
        assert _split_query_aspects("") == []

    def test_short_stopword_only_fragments_dropped(self):
        # Fragments with < 2 meaningful (non-stopword) content words are dropped.
        q = "what and how"
        assert _split_query_aspects(q) == []


# ---------------------------------------------------------------------------
# _doc_is_frame_like
# ---------------------------------------------------------------------------

class TestDocIsFrameLike:

    def test_embedding_space_vision(self):
        assert _doc_is_frame_like({"text": "x", "metadata": {"embedding_space": "vision"}}) is True

    def test_subtype_frame(self):
        assert _doc_is_frame_like({"text": "x", "metadata": {"subtype": "frame"}}) is True

    def test_asset_path_present(self):
        assert _doc_is_frame_like({"text": "x", "metadata": {"asset_path": "/tmp/f.jpg"}}) is True

    def test_frame_timestamp_present_even_zero(self):
        # 0.0 is a legitimate (falsy) timestamp — must use an explicit None check.
        assert _doc_is_frame_like({"text": "x", "metadata": {"frame_timestamp": 0.0}}) is True

    def test_on_screen_marker_in_text(self):
        assert _doc_is_frame_like({"text": "cap [ON-SCREEN]: XYZ", "metadata": {}}) is True

    def test_plain_transcript_doc_is_not_frame(self):
        assert _doc_is_frame_like(
            {"text": "Good afternoon, welcome to the call.", "metadata": {"modality": "mp4"}}
        ) is False


# ---------------------------------------------------------------------------
# _mask_frame_stock_price
# ---------------------------------------------------------------------------

class TestMaskFrameStockPrice:

    def test_masks_price_on_frame_doc(self):
        doc = {"text": "AAPL, Apple Inc. at $287.50, Q4 2025 earnings beat estimate, $1.85 EPS beats $1.76 estimate.",
              "metadata": {"embedding_space": "vision"}}
        masked = _mask_frame_stock_price(doc)
        assert "$287.50" not in masked["text"]
        assert "$1.85" in masked["text"]  # the real beat figure survives
        assert "$1.76" in masked["text"]

    def test_masks_daily_change_clause(self):
        doc = {"text": "AAPL, Apple Inc. at $283.80, up 4.96% from the previous day.",
              "metadata": {"embedding_space": "vision"}}
        masked = _mask_frame_stock_price(doc)
        assert "4.96%" not in masked["text"]

    def test_transcript_doc_untouched(self):
        doc = {"text": "Today, Apple is proud to report $102.5 billion in revenue, up 8% from a year ago.",
              "metadata": {"modality": "mp4"}}
        masked = _mask_frame_stock_price(doc)
        assert masked["text"] == doc["text"]
        assert masked is doc  # untouched docs are returned as-is, not copied

    def test_yoy_growth_phrase_never_stripped_even_on_a_frame_doc(self):
        # Regression guard: the daily-change regex is anchored on "previous
        # day/close" specifically so it can never eat a transcript's "up 8%
        # from a year ago" YoY figure, even if such text ends up on a
        # frame-classified doc.
        doc = {"text": "revenue, up 8% from a year ago.", "metadata": {"embedding_space": "vision"}}
        masked = _mask_frame_stock_price(doc)
        assert "up 8%" in masked["text"]

    def test_frame_with_no_price_clause_unchanged(self):
        doc = {"text": "EPS $1.85 beats $1.76 estimate", "metadata": {"embedding_space": "vision"}}
        masked = _mask_frame_stock_price(doc)
        assert masked["text"] == doc["text"]


# ---------------------------------------------------------------------------
# _video_speaker_name
# ---------------------------------------------------------------------------

class TestVideoSpeakerName:

    _CAST = {
        "ir": "Suhasini Chandramouli", "ceo": "Tim Cook", "cfo": "Kevan Parekh",
        "cook_start": 43.44, "parekh_start": 894.66, "qa_start": 1525.1,
    }

    def test_ir_intro_before_cook_start(self):
        name, role = _video_speaker_name(self._CAST, 10.0, "operator_intro")
        assert name == "Suhasini Chandramouli"
        assert role == "Investor Relations"

    def test_ceo_prepared_remarks(self):
        name, role = _video_speaker_name(self._CAST, 200.0, "prepared_remarks")
        assert name == "Tim Cook"
        assert role == "CEO"

    def test_cfo_prepared_remarks(self):
        name, role = _video_speaker_name(self._CAST, 1000.0, "prepared_remarks")
        assert name == "Kevan Parekh"
        assert role == "CFO"

    def test_qa_session_returns_none(self):
        name, role = _video_speaker_name(self._CAST, 2000.0, "qa_session")
        assert name is None
        assert role is None

    def test_past_qa_start_timestamp_returns_none_even_if_labeled_prepared_remarks(self):
        name, role = _video_speaker_name(self._CAST, 1600.0, "prepared_remarks")
        assert name is None
        assert role is None

    def test_empty_cast_returns_none(self):
        assert _video_speaker_name({}, 100.0, "prepared_remarks") == (None, None)

    def test_none_timestamp_returns_none(self):
        assert _video_speaker_name(self._CAST, None, "prepared_remarks") == (None, None)


# ---------------------------------------------------------------------------
# _build_p248_sources — video FRAME citation fields
# ---------------------------------------------------------------------------

class TestBuildP248SourcesVideoFrame:

    def _frame_doc(self, **meta_overrides):
        meta = {
            "source": "Q4 2025 Earnings Call.mp4",
            "modality": "mp4",
            "embedding_space": "vision",
            "frame_timestamp": 1473.0,
            "asset_path": "/nonexistent/path/frame_000016.jpg",
        }
        meta.update(meta_overrides)
        text = "AAPL: Apple, Q4 2025 EPS $1.85 Beats $1.76 Estimate, Sales $102.466B Beats $102.171B Estimate"
        return {"text": text, "score": 0.7, "metadata": meta}

    def test_frame_doc_marked_is_frame(self):
        sources = _build_p248_sources([self._frame_doc()])
        assert sources[0]["is_frame"] is True

    def test_frame_label_extracted(self):
        sources = _build_p248_sources([self._frame_doc()])
        assert sources[0]["frame_label"] == "EPS $1.85 beats $1.76 · Sales $102.466B beats $102.171B"

    def test_frame_timestamp_surfaced(self):
        sources = _build_p248_sources([self._frame_doc()])
        assert sources[0]["frame_timestamp"] == 1473.0

    def test_nonexistent_asset_path_nulled(self):
        # asset_path pointing at a file that doesn't exist on disk must not be
        # handed to the client as a broken image URL.
        sources = _build_p248_sources([self._frame_doc()])
        assert sources[0]["asset_path"] is None

    def test_zero_timestamp_frame_not_dropped(self):
        # 0.0 is falsy — must survive via an explicit None check, not `or`.
        sources = _build_p248_sources([self._frame_doc(frame_timestamp=0.0)])
        assert sources[0]["frame_timestamp"] == 0.0
        assert sources[0]["start_time"] == 0.0

    def test_non_frame_doc_has_frame_fields_none(self):
        doc = {"text": "spoken transcript text", "score": 0.5,
              "metadata": {"source": "x.mp4", "modality": "mp4"}}
        sources = _build_p248_sources([doc])
        assert sources[0]["is_frame"] is False
        assert sources[0]["frame_label"] is None
