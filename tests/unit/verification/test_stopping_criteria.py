"""Unit tests for app/verification/stopping_criteria.py — the loop termination
conditions (docs/Phase_32_Agentic_Answer_Verification.md §5). No LLM, no
network; overrides settings via monkeypatch for deterministic thresholds.

Four conditions, not five: the former rules 4 and 5 were merged into a single
conjunction on 2026-08-08 (stop only when retrieval AND overall have both
stalled). See the module docstring under test for the reasoning.
"""

import time

from app.core.config import settings
from app.verification.stopping_criteria import StoppingCriteria
from app.verification.verification_schema import ConfidenceScores, RetryAttempt


def _attempt(n, overall, retrieval=90.0, decision="FAIL"):
    return RetryAttempt(
        attempt_number=n,
        strategy="baseline" if n == 0 else "expand_retrieval",
        scores=ConfidenceScores(
            retrieval=retrieval, grounding=90.0, citation=95.0, overall=overall
        ),
        decision=decision,
        reason="test",
        duration_ms=10.0,
    )


class TestStoppingCriteria:

    def test_no_attempts_never_stops(self):
        sc = StoppingCriteria()
        stop, reason = sc.should_stop([], time.time())
        assert stop is False

    def test_pass_stops_immediately(self):
        sc = StoppingCriteria()
        attempts = [_attempt(0, 95.0, decision="PASS")]
        stop, reason = sc.should_stop(attempts, time.time())
        assert stop is True
        assert reason == "verified"

    def test_max_retries_reached_stops(self, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_VERIFY_MAX_RETRIES", 2)
        sc = StoppingCriteria()
        # attempt_number 0,1,2 => 3 attempts, len-1=2 >= max_retries(2)
        attempts = [_attempt(0, 40.0), _attempt(1, 45.0), _attempt(2, 50.0)]
        stop, reason = sc.should_stop(attempts, time.time())
        assert stop is True
        assert "max_retries_reached" in reason

    def test_timeout_stops(self, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_VERIFY_TIMEOUT_SEC", 0.01)
        sc = StoppingCriteria()
        attempts = [_attempt(0, 40.0)]
        stop, reason = sc.should_stop(attempts, time.time() - 1.0)
        assert stop is True
        assert "timeout" in reason

    # Rules 4+5 are ONE conjunction, not two independent conditions: the loop
    # stops only when retrieval AND overall have BOTH stalled. See the 2026-08-08
    # NOTE in app/verification/stopping_criteria.py for why stalling on retrieval
    # alone must not stop the loop (expand_retrieval always runs first and
    # plateaus as the common case, killing the loop before decomposition — the
    # one strategy built for multi-part finance questions — is ever reached).
    # The three tests below pin all three corners of that conjunction.

    def test_both_stalled_stops(self, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_VERIFY_MIN_IMPROVEMENT_PCT", 10.0)
        sc = StoppingCriteria()
        # retrieval flat (80->75) AND overall improvement 2% < 10% threshold.
        attempts = [_attempt(0, 50.0, retrieval=80.0), _attempt(1, 51.0, retrieval=75.0)]
        stop, reason = sc.should_stop(attempts, time.time())
        assert stop is True
        assert "retrieval_and_overall_confidence_not_improving" in reason

    def test_retrieval_stalled_but_overall_improving_continues(self, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_VERIFY_MIN_IMPROVEMENT_PCT", 10.0)
        monkeypatch.setattr(settings, "AGENT_VERIFY_MAX_RETRIES", 3)
        monkeypatch.setattr(settings, "AGENT_VERIFY_TIMEOUT_SEC", 30.0)
        sc = StoppingCriteria()
        # retrieval flat (80->75) but overall jumps 40->60 (+50%): keep going.
        attempts = [_attempt(0, 40.0, retrieval=80.0), _attempt(1, 60.0, retrieval=75.0)]
        stop, reason = sc.should_stop(attempts, time.time())
        assert stop is False

    def test_low_overall_improvement_alone_continues(self, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_VERIFY_MIN_IMPROVEMENT_PCT", 10.0)
        monkeypatch.setattr(settings, "AGENT_VERIFY_MAX_RETRIES", 3)
        monkeypatch.setattr(settings, "AGENT_VERIFY_TIMEOUT_SEC", 30.0)
        sc = StoppingCriteria()
        # overall barely moves (2% < 10%) but retrieval is still improving
        # (85->90), so the conjunction is not satisfied: keep going.
        attempts = [_attempt(0, 50.0, retrieval=85.0), _attempt(1, 51.0, retrieval=90.0)]
        stop, reason = sc.should_stop(attempts, time.time())
        assert stop is False

    def test_continues_when_genuinely_improving(self, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_VERIFY_MIN_IMPROVEMENT_PCT", 2.0)
        monkeypatch.setattr(settings, "AGENT_VERIFY_MAX_RETRIES", 3)
        monkeypatch.setattr(settings, "AGENT_VERIFY_TIMEOUT_SEC", 30.0)
        sc = StoppingCriteria()
        attempts = [_attempt(0, 40.0, retrieval=60.0), _attempt(1, 70.0, retrieval=80.0)]
        stop, reason = sc.should_stop(attempts, time.time())
        assert stop is False

    def test_best_attempt_picks_highest_overall(self):
        sc = StoppingCriteria()
        attempts = [_attempt(0, 40.0), _attempt(1, 85.0), _attempt(2, 60.0)]
        best = sc.best_attempt(attempts)
        assert best.attempt_number == 1
        assert best.scores.overall == 85.0

    def test_best_attempt_empty_list_returns_none(self):
        sc = StoppingCriteria()
        assert sc.best_attempt([]) is None
