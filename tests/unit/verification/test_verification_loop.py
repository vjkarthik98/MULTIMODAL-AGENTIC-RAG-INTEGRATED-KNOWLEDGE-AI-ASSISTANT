"""Unit tests for app/verification/verification_loop.py — the full
generate -> verify -> decide -> retry orchestration. reasoning_engine/
retriever/llm are all mocked (no GPU, no network); this tests control flow,
not RAG quality.

Two contracts under test that came out of the pre-implementation architect
review (docs/Phase_32_Agentic_Answer_Verification.md §6 amendment):
  1. A baseline (attempt 0) generation failure RE-RAISES, so the caller's own
     exception handling (query_pipeline.py's GGUF fallback chain) still fires
     exactly as it did before this loop existed.
  2. A retry-attempt (1+) generation failure does NOT crash the request — the
     loop already has a working baseline answer.
"""

from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.verification.verification_loop import VerificationLoop


def _doc(text, chunk_id="c1", score=0.8):
    return {"text": text, "metadata": {"chunk_id": chunk_id}, "score": score}


def _source(cite_key, chunk_id="c1", snippet=""):
    return {"cite_key": cite_key, "chunk_id": chunk_id, "snippet": snippet}


def _reasoning_engine(answer_text):
    engine = MagicMock()
    engine.generate_answer.return_value = {"answer": answer_text}
    return engine


class TestVerificationLoopPassthrough:

    def test_disabled_globally_runs_single_shot(self, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_VERIFY_ENABLED", False)
        loop = VerificationLoop()
        engine = _reasoning_engine("A plain answer.")
        answer, report = loop.run(
            "query", "sess1", "user1", retriever=MagicMock(),
            reasoning_engine=engine, initial_docs=[_doc("some text")],
        )
        assert answer == "A plain answer."
        assert report.verified is True
        assert engine.generate_answer.call_count == 1

    def test_disabled_globally_increments_passthrough_counter_and_warns(self, monkeypatch, caplog):
        # Hardening added Phase 2 (2026-08-13, hallucination-reduction
        # initiative): _passthrough() previously force-filled scores to
        # 100/verified=True with no signal it had skipped the real checks.
        import logging

        from app.verification.verification_loop import _passthrough_total

        monkeypatch.setattr(settings, "AGENT_VERIFY_ENABLED", False)
        before = _passthrough_total.labels(reason="disabled_globally")._value.get()
        loop = VerificationLoop()
        engine = _reasoning_engine("A plain answer.")
        with caplog.at_level(logging.WARNING):
            loop.run(
                "query", "sess1", "user1", retriever=MagicMock(),
                reasoning_engine=engine, initial_docs=[_doc("some text")],
            )
        after = _passthrough_total.labels(reason="disabled_globally")._value.get()
        assert after == before + 1
        assert any(
            "verification_passthrough" in r.message or "verification_passthrough" in str(r.msg)
            for r in caplog.records
        )

    def test_modality_opted_out_runs_single_shot(self, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_VERIFY_ENABLED", True)
        monkeypatch.setattr(settings, "AGENT_VERIFY_MODALITIES", ["audio", "video"])
        loop = VerificationLoop()
        engine = _reasoning_engine("A plain answer.")
        answer, report = loop.run(
            "query", "sess1", "user1", retriever=MagicMock(),
            reasoning_engine=engine, initial_docs=[_doc("some text")],
            modality_hint="pdf",
        )
        assert report.verified is True
        assert engine.generate_answer.call_count == 1

    def test_modality_opted_out_increments_passthrough_counter(self, monkeypatch):
        from app.verification.verification_loop import _passthrough_total

        monkeypatch.setattr(settings, "AGENT_VERIFY_ENABLED", True)
        monkeypatch.setattr(settings, "AGENT_VERIFY_MODALITIES", ["audio", "video"])
        before = _passthrough_total.labels(reason="modality_excluded")._value.get()
        loop = VerificationLoop()
        engine = _reasoning_engine("A plain answer.")
        loop.run(
            "query", "sess1", "user1", retriever=MagicMock(),
            reasoning_engine=engine, initial_docs=[_doc("some text")],
            modality_hint="pdf",
        )
        after = _passthrough_total.labels(reason="modality_excluded")._value.get()
        assert after == before + 1

    def test_mp4_alias_normalized_to_video(self, monkeypatch):
        # Regression test (live smoke test, Phase 32): video_chunker.py tags
        # frame/vision chunks modality="mp4" while transcript chunks from
        # the same file are tagged "video" — a live end-to-end test caught
        # verification silently never firing when the top-ranked doc was a
        # frame chunk, because "mp4" wasn't in AGENT_VERIFY_MODALITIES
        # (which only lists the canonical "video"). modality_hint="mp4" must
        # be treated identically to modality_hint="video".
        monkeypatch.setattr(settings, "AGENT_VERIFY_ENABLED", True)
        monkeypatch.setattr(settings, "AGENT_VERIFY_MODALITIES", ["video"])
        loop = VerificationLoop()
        engine = _reasoning_engine("Apple reported net revenue of $94.9 billion. [apple.mp4 t=00:03:49]")
        docs = [_doc("Apple reported net revenue of $94.9 billion.")]
        sources = [_source("[apple.mp4 t=00:03:49]")]

        answer, report = loop.run(
            "What was Q4 revenue?", "sess1", "user1", retriever=MagicMock(),
            reasoning_engine=engine, initial_docs=docs, initial_sources=sources,
            modality_hint="mp4",
        )
        # If the alias weren't normalized, this would silently take the
        # _passthrough() path (no groundedness/citation scoring at all).
        assert len(report.attempts) == 1
        assert report.attempts[0].strategy == "baseline"

    def test_mp3_alias_normalized_to_audio(self, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_VERIFY_ENABLED", True)
        monkeypatch.setattr(settings, "AGENT_VERIFY_MODALITIES", ["audio"])
        loop = VerificationLoop()
        engine = _reasoning_engine("The Fed held rates steady. [call.mp3 t=00:01:00]")
        docs = [_doc("The Fed held rates steady.")]
        sources = [_source("[call.mp3 t=00:01:00]")]

        answer, report = loop.run(
            "What did the Fed decide?", "sess1", "user1", retriever=MagicMock(),
            reasoning_engine=engine, initial_docs=docs, initial_sources=sources,
            modality_hint="mp3",
        )
        assert len(report.attempts) == 1
        assert report.attempts[0].strategy == "baseline"

    def test_text_alias_normalized_to_txt(self, monkeypatch):
        # Regression test (live SSE smoke test, hallucination-reduction
        # initiative Phase 2, 2026-08-13): txt_ingest.py's inline
        # IngestedDocument construction tags modality="text" while
        # txt_chunker.py's dedicated chunker tags "txt". Confirmed against
        # the live eval tenant's BM25 index — every indexed chunk of
        # fomc_dec2024.txt carries modality="text" — so every real txt query
        # silently skipped VerificationLoop before this alias was added.
        # modality_hint="text" must be treated identically to "txt".
        monkeypatch.setattr(settings, "AGENT_VERIFY_ENABLED", True)
        monkeypatch.setattr(settings, "AGENT_VERIFY_MODALITIES", ["txt"])
        loop = VerificationLoop()
        engine = _reasoning_engine("The FOMC cut rates by a quarter point.")
        docs = [_doc("The FOMC cut rates by a quarter point.")]

        answer, report = loop.run(
            "What did the FOMC decide?", "sess1", "user1", retriever=MagicMock(),
            reasoning_engine=engine, initial_docs=docs,
            modality_hint="text",
        )
        # If the alias weren't normalized, this would silently take the
        # _passthrough() path (no groundedness/citation scoring at all).
        assert len(report.attempts) == 1
        assert report.attempts[0].strategy == "baseline"


class TestVerificationLoopPass:

    def test_well_grounded_answer_passes_on_baseline(self, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_VERIFY_ENABLED", True)
        monkeypatch.setattr(settings, "AGENT_VERIFY_MODALITIES",
                             ["txt", "pdf", "docx", "xlsx", "image", "audio", "video"])
        loop = VerificationLoop()
        text = "Apple reported net revenue of $94.9 billion in Q4 2024."
        engine = _reasoning_engine(f"{text} [apple.pdf p.4]")
        docs = [_doc(text)]
        sources = [_source("[apple.pdf p.4]")]

        answer, report = loop.run(
            "What was Q4 revenue?", "sess1", "user1", retriever=MagicMock(),
            reasoning_engine=engine, initial_docs=docs, initial_sources=sources,
            modality_hint="pdf",
        )
        assert report.verified is True
        assert engine.generate_answer.call_count == 1  # no retries needed
        assert len(report.attempts) == 1
        assert report.attempts[0].decision == "PASS"

    def test_cited_sources_reflects_generate_answers_own_filtering_not_raw_pool(self, monkeypatch):
        # Regression test (code review, Phase 32): VerificationLoop must
        # surface generate_answer()'s own citation-filtered `sources` field,
        # not silently drop it in favor of the full initial_sources candidate
        # pool — that would regress citation transparency to "show
        # everything retrieved" instead of "show what was actually cited."
        monkeypatch.setattr(settings, "AGENT_VERIFY_ENABLED", True)
        monkeypatch.setattr(settings, "AGENT_VERIFY_MODALITIES", ["pdf"])
        loop = VerificationLoop()

        engine = MagicMock()
        engine.generate_answer.return_value = {
            "answer": "Apple reported net revenue of $94.9 billion. [apple.pdf p.4]",
            # Only ONE of the two candidate sources was actually cited —
            # generate_answer() itself filters this down.
            "sources": [{"cite_key": "[apple.pdf p.4]", "chunk_id": "c1"}],
        }
        docs = [_doc("Apple reported net revenue of $94.9 billion.")]
        wide_candidate_pool = [
            _source("[apple.pdf p.4]", chunk_id="c1"),
            _source("[apple.pdf p.9]", chunk_id="c9"),  # retrieved but NOT cited
        ]

        answer, report = loop.run(
            "What was Q4 revenue?", "sess1", "user1", retriever=MagicMock(),
            reasoning_engine=engine, initial_docs=docs,
            initial_sources=wide_candidate_pool, modality_hint="pdf",
        )
        assert len(report.cited_sources) == 1
        assert report.cited_sources[0]["cite_key"] == "[apple.pdf p.4]"

    def test_refusal_answer_yields_empty_cited_sources(self, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_VERIFY_ENABLED", True)
        monkeypatch.setattr(settings, "AGENT_VERIFY_MODALITIES", ["pdf"])
        loop = VerificationLoop()

        engine = MagicMock()
        engine.generate_answer.return_value = {
            "answer": "The document does not contain this information.",
            "sources": [],  # generate_answer() empties sources for refusals
        }
        docs = [_doc("Unrelated content.")]
        sources = [_source("[apple.pdf p.4]")]
        retriever = MagicMock()
        retriever.search.return_value = []  # explicit: real HybridRetriever.search() -> List[Dict]

        answer, report = loop.run(
            "What was Q4 revenue?", "sess1", "user1", retriever=retriever,
            reasoning_engine=engine, initial_docs=docs, initial_sources=sources,
            modality_hint="pdf",
        )
        assert report.cited_sources == []


class TestVerificationLoopFailAndRetry:

    def test_exhausts_retries_and_returns_degraded_best_effort(self, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_VERIFY_ENABLED", True)
        monkeypatch.setattr(settings, "AGENT_VERIFY_MODALITIES", ["pdf"])
        monkeypatch.setattr(settings, "AGENT_VERIFY_MAX_RETRIES", 1)
        loop = VerificationLoop()

        # Every generation fabricates a number never in context -> always FAIL.
        engine = _reasoning_engine("Revenue was $999.9 billion. [apple.pdf p.4]")
        docs = [_doc("Apple reported net revenue of $94.9 billion in Q4 2024.")]
        sources = [_source("[apple.pdf p.4]")]
        retriever = MagicMock()
        retriever.search.return_value = docs

        answer, report = loop.run(
            "What was Q4 revenue?", "sess1", "user1", retriever=retriever,
            reasoning_engine=engine, initial_docs=docs, initial_sources=sources,
            modality_hint="pdf",
        )

        assert report.verified is False
        assert report.degraded is True
        assert report.limitation_notice is not None
        assert report.limitation_notice in answer
        # max_retries=1 -> attempt 0 (baseline) + attempt 1 (one retry) = 2 max,
        # but stopping criteria may cut it short on non-improvement too.
        assert len(report.attempts) <= 2
        assert engine.generate_answer.call_count == len(report.attempts)

    def test_baseline_hard_failure_reraises(self):
        loop = VerificationLoop()
        engine = MagicMock()
        engine.generate_answer.side_effect = RuntimeError("LLM unavailable")

        with pytest.raises(RuntimeError, match="LLM unavailable"):
            loop.run(
                "query", "sess1", "user1", retriever=MagicMock(),
                reasoning_engine=engine, initial_docs=[_doc("text")],
                modality_hint="pdf",
            )

    def test_retry_attempt_failure_does_not_crash_request(self, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_VERIFY_ENABLED", True)
        monkeypatch.setattr(settings, "AGENT_VERIFY_MODALITIES", ["pdf"])
        monkeypatch.setattr(settings, "AGENT_VERIFY_MAX_RETRIES", 1)

        loop = VerificationLoop()
        engine = MagicMock()
        # Baseline succeeds with a FAIL-worthy (fabricated) answer, then the
        # retry-attempt generation call raises — must not propagate.
        engine.generate_answer.side_effect = [
            {"answer": "Revenue was $999.9 billion. [apple.pdf p.4]"},
            RuntimeError("transient GPU error"),
        ]
        docs = [_doc("Apple reported net revenue of $94.9 billion in Q4 2024.")]
        sources = [_source("[apple.pdf p.4]")]
        retriever = MagicMock()
        retriever.search.return_value = docs

        # Must not raise.
        answer, report = loop.run(
            "What was Q4 revenue?", "sess1", "user1", retriever=retriever,
            reasoning_engine=engine, initial_docs=docs, initial_sources=sources,
            modality_hint="pdf",
        )
        assert report.degraded is True
        assert answer  # still produced a best-effort answer


class TestSkipRetries:
    """Image chart-value answers and docx table-row lookups are both
    deterministically SYNTHESIZED by the caller after this loop returns,
    discarding whatever the LLM drafted — so retrying against the draft can
    never change the final answer. Live-measured before this fix: 14/14
    image queries retried for 0 eventual successes (query_pipeline.py's
    skip_retries pre-check calls the synth functions speculatively to know
    FOR CERTAIN synthesis will fire, then passes skip_retries=True here).
    """

    def test_skip_retries_stops_after_baseline_even_on_fail(self, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_VERIFY_ENABLED", True)
        monkeypatch.setattr(settings, "AGENT_VERIFY_MODALITIES", ["image"])
        monkeypatch.setattr(settings, "AGENT_VERIFY_MAX_RETRIES", 3)

        loop = VerificationLoop()
        # Every generation would normally trigger a retry (fabricated number
        # never in context -> always FAIL) — with skip_retries=True this must
        # still only be called ONCE.
        engine = _reasoning_engine("The value was approximately $999999 per this chart.")
        docs = [_doc("CHART VALUES - pixel-calibrated reads: 9/28/24: Apple Inc.=~$429")]
        sources = [_source("[img.png chart]")]
        retriever = MagicMock()
        retriever.search.return_value = docs

        answer, report = loop.run(
            "What was the value on 9/28/24?", "sess1", "user1", retriever=retriever,
            reasoning_engine=engine, initial_docs=docs, initial_sources=sources,
            modality_hint="image", skip_retries=True,
        )

        assert len(report.attempts) == 1
        assert engine.generate_answer.call_count == 1
        # Verification itself still ran and still correctly failed — only
        # the RETRY loop was skipped, not the check.
        assert report.verified is False

    def test_default_behavior_unchanged_when_skip_retries_omitted(self, monkeypatch):
        # Same FAIL-worthy setup as test_exhausts_retries_and_returns_
        # degraded_best_effort above — confirms skip_retries defaults to
        # False and normal retry behavior is untouched.
        monkeypatch.setattr(settings, "AGENT_VERIFY_ENABLED", True)
        monkeypatch.setattr(settings, "AGENT_VERIFY_MODALITIES", ["pdf"])
        monkeypatch.setattr(settings, "AGENT_VERIFY_MAX_RETRIES", 1)

        loop = VerificationLoop()
        engine = _reasoning_engine("Revenue was $999.9 billion. [apple.pdf p.4]")
        docs = [_doc("Apple reported net revenue of $94.9 billion in Q4 2024.")]
        sources = [_source("[apple.pdf p.4]")]
        retriever = MagicMock()
        retriever.search.return_value = docs

        answer, report = loop.run(
            "What was Q4 revenue?", "sess1", "user1", retriever=retriever,
            reasoning_engine=engine, initial_docs=docs, initial_sources=sources,
            modality_hint="pdf",
        )
        assert len(report.attempts) > 1
