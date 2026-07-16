"""Unit tests for the video-modality citation and answer-grounding helpers in
app/pipeline/rag_pipeline.py — the speaker-name resolver, frame-caption
cleaner, query-aspect splitter, and frame stock-price masker added for the
Qwen2.5-14B upgrade. Pure-function tests only: no Qdrant, no LLM, no network.
"""

from app.pipeline import rag_pipeline
from app.pipeline.rag_pipeline import (
    _build_p248_sources,
    _clean_frame_label,
    _doc_is_frame_like,
    _doc_is_true_frame,
    _mask_frame_stock_price,
    _split_frame_caption,
    _split_query_aspects,
    _is_degenerate_answer,
    _rank_video_citation_docs,
    _strip_onscreen_ocr,
    _video_completeness_fill,
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

    def test_transcript_with_inline_onscreen_annotation_not_mangled(self):
        # Regression guard: a spoken transcript chunk that merely carries an
        # inline "[ON-SCREEN]" OCR annotation is NOT a frame — the price regex
        # ('at $1.85') must NOT corrupt the real spoken figure in it. Uses the
        # metadata-only frame test, so a doc with no vision/frame metadata is
        # left byte-identical even though its text contains "[ON-SCREEN]".
        doc = {"text": "EPS came in at $1.85, setting a record. [ON-SCREEN]: AAPL 287.50",
              "metadata": {"modality": "mp4", "start_timestamp": 137.8}}
        masked = _mask_frame_stock_price(doc)
        assert masked["text"] == doc["text"]
        assert "$1.85" in masked["text"]
        assert masked is doc


# ---------------------------------------------------------------------------
# _doc_is_true_frame
# ---------------------------------------------------------------------------

class TestDocIsTrueFrame:

    def test_vision_is_true_frame(self):
        assert _doc_is_true_frame({"text": "x", "metadata": {"embedding_space": "vision"}}) is True

    def test_subtype_frame_is_true_frame(self):
        assert _doc_is_true_frame({"text": "x", "metadata": {"subtype": "frame"}}) is True

    def test_frame_timestamp_zero_is_true_frame(self):
        assert _doc_is_true_frame({"text": "x", "metadata": {"frame_timestamp": 0.0}}) is True

    def test_transcript_with_onscreen_text_is_NOT_true_frame(self):
        # The key difference from _doc_is_frame_like: no text-substring fallback,
        # so an inline "[ON-SCREEN]" annotation does not make a transcript a frame.
        d = {"text": "welcome to the call [ON-SCREEN]: AAPL", "metadata": {"modality": "mp4"}}
        assert _doc_is_frame_like(d) is True     # old test matches the substring
        assert _doc_is_true_frame(d) is False    # new test does not


# ---------------------------------------------------------------------------
# _strip_onscreen_ocr
# ---------------------------------------------------------------------------

class TestStripOnscreenOcr:

    def test_strips_trailing_visual_and_onscreen_block(self):
        t = ("Good afternoon, and welcome to the call. "
             "[VISUAL AT 0.0s]: AAPL at $287.50, $1.85 EPS beats $1.76 estimate. "
             "[ON-SCREEN]: PANW CDNS TTD 218.09 Apple Q4 EPS Beats Estimate")
        out = _strip_onscreen_ocr(t)
        assert out == "Good afternoon, and welcome to the call."
        assert "1.85" not in out
        assert "ON-SCREEN" not in out

    def test_strips_when_visual_is_mid_text(self):
        t = "iPhone had a tremendous response [VISUAL AT 163.8s]: AAPL at $283.80, up 4.96%. [ON-SCREEN]: AAPL MSFT"
        out = _strip_onscreen_ocr(t)
        assert out == "iPhone had a tremendous response"

    def test_plain_transcript_unchanged(self):
        t = "Services achieved an all-time revenue record of $28.8 billion, growing 15% from a year ago."
        assert _strip_onscreen_ocr(t) == t

    def test_preserves_spoken_dollar_figures(self):
        t = "Our revenue of $102.5 billion was up 8% year-over-year. [ON-SCREEN]: ticker soup 283.08"
        out = _strip_onscreen_ocr(t)
        assert "$102.5 billion" in out
        assert "up 8%" in out
        assert "ticker soup" not in out

    def test_empty_and_none_safe(self):
        assert _strip_onscreen_ocr("") == ""
        assert _strip_onscreen_ocr(None) == ""


# ---------------------------------------------------------------------------
# _video_completeness_fill
# ---------------------------------------------------------------------------

class TestVideoCompletenessFill:
    """The deterministic fill reads verbatim from the call transcript (mocked
    here via the module sentence cache) and appends a specific asked-for fact
    the generated answer dropped. Tightly gated on query intent. Returns
    (answer, fill_docs) — fill_docs carry the same (timestamp, section) each
    sentence was cached with, for downstream citation attribution."""

    _SENTS = [
        ("This strong momentum drove our total fiscal year services revenue to "
         "surpass $100 billion, up 14 % year-over-year, and our best ever.",
         1439.86, "prepared_remarks"),
        ("Today, Apple is proud to report $102.5 billion in revenue, up 8 % from "
         "a year ago and a September quarter record.",
         43.44, "prepared_remarks"),
        ("We also set a September quarter revenue record in emerging markets and "
         "an all-time revenue record in India.",
         137.86, "prepared_remarks"),
        ("Services achieved an all-time revenue record of $28.8 billion, growing 15 %.",
         137.86, "prepared_remarks"),
    ]

    def setup_method(self):
        rag_pipeline._VIDEO_SENTENCES_CACHE[("u", "s.mp4")] = list(self._SENTS)

    def teardown_method(self):
        rag_pipeline._VIDEO_SENTENCES_CACHE.pop(("u", "s.mp4"), None)

    def test_yoy_fills_total_revenue_not_segment(self):
        # Must pick the TOTAL revenue "+8%" line, never the services "+14%" line.
        out, docs = _video_completeness_fill(
            "revenue, EPS, year-over-year revenue growth, beat estimates",
            "Apple reported EPS $1.85 beating $1.76 and revenue $102.466B.",
            "u", "s.mp4")
        assert "up 8 %" in out
        assert "up 14" not in out
        assert len(docs) == 1
        assert docs[0]["metadata"]["start_timestamp"] == 43.44
        assert docs[0]["metadata"]["source"] == "s.mp4"

    def test_records_fills_named_india_record(self):
        out, docs = _video_completeness_fill(
            "full-year annual revenue, and what all-time records did the company set",
            "We achieved an all-time revenue record of $416 billion for the fiscal year.",
            "u", "s.mp4")
        assert "india" in out.lower()
        assert len(docs) == 1
        assert docs[0]["metadata"]["start_timestamp"] == 137.86

    def test_yoy_not_added_when_answer_already_has_growth_pct(self):
        ans = "Revenue was $102.5 billion, up 8% year-over-year."
        out, docs = _video_completeness_fill(
            "year-over-year revenue growth", ans, "u", "s.mp4")
        assert out == ans  # already covered → untouched
        assert docs == []

    def test_non_matching_query_untouched(self):
        # A Services/antitrust question asks for neither a YoY figure nor a
        # record enumeration → the answer must be returned byte-identical.
        ans = "Services $28.8 billion, organically driven, not the antitrust ruling."
        out, docs = _video_completeness_fill(
            "What did the CFO say about the Google antitrust ruling?", ans, "u", "s.mp4")
        assert out == ans
        assert docs == []

    def test_india_not_duplicated_if_already_present(self):
        ans = "Apple set an all-time revenue record in India during the year."
        out, docs = _video_completeness_fill(
            "what all-time records did the company set", ans, "u", "s.mp4")
        assert out.lower().count("india") == 1
        assert docs == []


class TestVideoCompletenessFillQualitative:
    """Q34-style qualitative aspects (iPhone Air, foundation models, M&A) are
    filled deterministically from the transcript, gated on the query naming
    the topic, and never pick an analyst's question sentence."""

    _SENTS = [
        ("The iPhone Air feels thin and light in your hand, it feels it's going to fly away.",
         1616.23, "qa_session"),
        ("We're obviously creating Apple Foundation models within Apple.",
         3060.1, "qa_session"),
        ("And, we continually surveil the market on M &A and are open to pursuing "
         "M &A if we think that it will advance our roadmap.",
         3060.1, "qa_session"),
        ("Will you continue to use the three-pronged approach with foundation models and M &A?",
         3031.96, "qa_session"),
    ]

    def setup_method(self):
        rag_pipeline._VIDEO_SENTENCES_CACHE[("u", "s.mp4")] = list(self._SENTS)

    def teardown_method(self):
        rag_pipeline._VIDEO_SENTENCES_CACHE.pop(("u", "s.mp4"), None)

    _Q = ("What did Tim Cook say about December quarter guidance, the iPhone Air "
          "reception, and Apple's approach to AI foundation models and M&A?")

    def test_fills_all_three_missing_qualitative_aspects(self):
        out, docs = _video_completeness_fill(
            self._Q, "Apple expects December revenue to grow 10% to 12%, best ever.",
            "u", "s.mp4")
        al = out.lower()
        assert "iphone air" in al
        assert "foundation model" in al
        assert "open to" in al or "m &a" in al
        assert len(docs) == 3
        assert all(d["metadata"]["start_timestamp"] is not None for d in docs)

    def test_never_appends_an_analyst_question(self):
        out, _ = _video_completeness_fill(
            self._Q, "Apple expects December revenue to grow 10% to 12%.", "u", "s.mp4")
        # the "Will you continue...?" question sentence must never be appended
        assert "will you continue" not in out.lower()
        assert not out.rstrip().endswith("?")

    def test_skips_aspect_already_answered(self):
        # If the answer already covers foundation models + M&A, only iPhone Air
        # is appended — no duplication of an already-successful follow-up.
        ans = ("December revenue to grow 10-12%. Apple builds its own foundation "
               "models and is open to M&A.")
        out, docs = _video_completeness_fill(self._Q, ans, "u", "s.mp4")
        assert out.lower().count("foundation model") == 1
        assert "iphone air" in out.lower()
        assert len(docs) == 1  # only the iPhone Air fill

    def test_non_q34_query_untouched(self):
        ans = "Services revenue was $28.8 billion."
        out, docs = _video_completeness_fill(
            "What was Apple Services revenue and why significant?", ans, "u", "s.mp4")
        assert out == ans
        assert docs == []


class TestIsDegenerateAnswer:

    def test_repeated_echo_is_degenerate(self):
        assert _is_degenerate_answer("The. Answer. Answer") is True

    def test_too_short_is_degenerate(self):
        assert _is_degenerate_answer("Yes.") is True
        assert _is_degenerate_answer("") is True

    def test_real_answer_not_degenerate(self):
        assert _is_degenerate_answer(
            "Apple is creating its own foundation models and is open to M&A."
        ) is False

    def test_single_token_domination_is_degenerate(self):
        assert _is_degenerate_answer("revenue revenue revenue revenue growth") is True


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


# ---------------------------------------------------------------------------
# _rank_video_citation_docs
# ---------------------------------------------------------------------------

class TestRankVideoCitationDocs:
    """Attributes each ANSWER SENTENCE to the candidate doc it best overlaps,
    instead of scoring the whole answer against every candidate — a Q&A
    chatter chunk sharing generic words with SOME sentence must not outrank
    the chunk that actually contains the specific fact just stated."""

    def _doc(self, text, ts, sec="prepared_remarks"):
        return {"text": text, "metadata": {"start_timestamp": ts, "call_section": sec}}

    def test_picks_the_fact_bearing_chunk_not_generic_qa_chatter(self):
        fact_doc = self._doc(
            "Services achieved an all-time revenue record of $28.8 billion, "
            "growing 15 % from a year ago.", 137.86)
        chatter_doc = self._doc(
            "And it's clearly, at least from our vantage point, driving some "
            "consumer demand. Okay, got it. And a follow-up on revenue for the "
            "December quarter.", 2433.36, "qa_session")
        answer = "Services achieved an all-time revenue record of $28.8 billion, growing 15% from a year ago."
        picked = _rank_video_citation_docs(answer, [chatter_doc, fact_doc], {}, None)
        assert picked[0] is fact_doc

    def test_multi_sentence_answer_attributes_each_sentence_separately(self):
        doc_a = self._doc("Services achieved an all-time revenue record of $28.8 billion, growing 15 %.", 137.86)
        doc_b = self._doc("We achieved an all-time revenue record of $416 billion for the fiscal year.", 1377.9)
        answer = ("Services achieved an all-time revenue record of $28.8 billion, growing 15%. "
                  "We achieved an all-time revenue record of $416 billion for the fiscal year.")
        picked = _rank_video_citation_docs(answer, [doc_a, doc_b], {}, None)
        assert doc_a in picked and doc_b in picked

    def test_operator_intro_demoted(self):
        intro = self._doc(
            "Good afternoon, and welcome to the Apple Q4 Fiscal Year 2025 Earnings "
            "Conference Call. Revenue guidance will follow.", 0.0, "operator_intro")
        real = self._doc(
            "Revenue guidance for the December quarter is 10 to 12 percent growth.",
            1439.86, "prepared_remarks")
        answer = "Revenue guidance for the December quarter is 10 to 12 percent growth."
        picked = _rank_video_citation_docs(answer, [intro, real], {}, None)
        assert picked[0] is real

    def test_no_candidates_returns_empty(self):
        assert _rank_video_citation_docs("Some answer.", [], {}, None) == []

    def test_empty_answer_falls_back_without_crash(self):
        doc = self._doc("Some transcript text here.", 10.0)
        # No sentences to attribute -> fallback path; must not raise.
        picked = _rank_video_citation_docs("", [doc], {}, None)
        assert isinstance(picked, list)

    def test_max_docs_respected(self):
        docs = [self._doc(f"Revenue figure number {i} was strong this quarter.", float(i))
               for i in range(5)]
        answer = " ".join(d["text"] for d in docs)
        picked = _rank_video_citation_docs(answer, docs, {}, None, max_docs=2)
        assert len(picked) <= 2
