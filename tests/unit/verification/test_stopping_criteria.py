"""Unit tests for app/verification/stopping_criteria.py — the 5 termination
conditions (docs/Phase_32_Agentic_Answer_Verification.md §5). No LLM, no
network; overrides settings via monkeypatch for deterministic thresholds.
"""

import time

import pytest

from app.core.config import settings
from app.verification.stopping_criteria import StoppingCriteria
from app.verification.verification_schema import ConfidenceScores, RetryAttempt


def _attempt(n, overall, retrieval=90.0, decision="FAIL"):
    return RetryAttempt(
        attempt_number=n,
        strategy="baseline" if n == 0 else "expand_retrieval",
        scores=ConfidenceScores(retrieval=retrieval, grounding=90.0, citation=95.0, overall=overall),
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

    def test_retrieval_not_improving_stops(self):
        sc = StoppingCriteria()
        attempts = [_attempt(0, 40.0, retrieval=80.0), _attempt(1, 60.0, retrieval=75.0)]
        stop, reason = sc.should_stop(attempts, time.time())
        assert stop is True
        assert reason == "retrieval_confidence_not_improving"

    def test_low_improvement_stops(self, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_VERIFY_MIN_IMPROVEMENT_PCT", 10.0)
        sc = StoppingCriteria()
        # retrieval improves (85->90) so that gate passes; overall barely moves.
        attempts = [_attempt(0, 50.0, retrieval=85.0), _attempt(1, 51.0, retrieval=90.0)]
        stop, reason = sc.should_stop(attempts, time.time())
        assert stop is True
        assert "overall_improvement_below_threshold" in reason

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
