"""StoppingCriteria — the 5 termination conditions (docs/Phase_32_Agentic_Answer_Verification.md §5).

Terminate the verification loop on ANY of:
1. Verified.
2. Max retries reached (settings.AGENT_VERIFY_MAX_RETRIES).
3. Wall-clock timeout exceeded (settings.AGENT_VERIFY_TIMEOUT_SEC).
4. Retrieval confidence did not improve vs. the previous attempt.
5. Overall-confidence improvement < settings.AGENT_VERIFY_MIN_IMPROVEMENT_PCT.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from app.core.config import settings
from app.verification.verification_schema import RetryAttempt


class StoppingCriteria:
    def __init__(self) -> None:
        self.max_retries = settings.AGENT_VERIFY_MAX_RETRIES
        self.timeout_sec = settings.AGENT_VERIFY_TIMEOUT_SEC
        self.min_improvement_pct = settings.AGENT_VERIFY_MIN_IMPROVEMENT_PCT

    def should_stop(self, attempts: List[RetryAttempt], start_time: float) -> Tuple[bool, str]:
        if not attempts:
            return False, ""

        last = attempts[-1]
        if last.decision == "PASS":
            return True, "verified"

        if len(attempts) - 1 >= self.max_retries:
            return True, f"max_retries_reached ({self.max_retries})"

        elapsed = time.time() - start_time
        if elapsed >= self.timeout_sec:
            return True, f"timeout ({elapsed:.1f}s >= {self.timeout_sec}s)"

        if len(attempts) >= 2:
            prev = attempts[-2]
            if last.scores.retrieval <= prev.scores.retrieval:
                return True, "retrieval_confidence_not_improving"

            improvement_pct = (
                (last.scores.overall - prev.scores.overall) / max(prev.scores.overall, 1.0) * 100.0
            )
            if improvement_pct < self.min_improvement_pct:
                return True, f"overall_improvement_below_threshold ({improvement_pct:.1f}% < {self.min_improvement_pct}%)"

        return False, ""

    def best_attempt(self, attempts: List[RetryAttempt]) -> Optional[RetryAttempt]:
        if not attempts:
            return None
        return max(attempts, key=lambda a: a.scores.overall)
