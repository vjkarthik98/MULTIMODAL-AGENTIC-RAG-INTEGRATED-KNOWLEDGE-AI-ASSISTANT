"""StoppingCriteria — loop termination conditions (docs/Phase_32_Agentic_Answer_Verification.md §5).

Terminate the verification loop on ANY of (4 conditions since 2026-08-08,
when the former rules 4 and 5 were merged into a single conjunction):
1. Verified.
2. Max retries reached (settings.AGENT_VERIFY_MAX_RETRIES).
3. Wall-clock timeout exceeded (settings.AGENT_VERIFY_TIMEOUT_SEC).
4. Retrieval confidence did not improve AND overall-confidence improvement is
   below settings.AGENT_VERIFY_MIN_IMPROVEMENT_PCT vs. the previous attempt.
   This is ONE conjunction (both, not either) — what were once rules 4 and 5
   were merged; neither half stops the loop on its own. See the note below.

NOTE on rule 4 (2026-08-08): originally fired on retrieval-alone stalling,
independent of rule 5. RetryController's 4 strategies are NOT all
retrieval-focused (query_rewrite/decomposition target generation quality and
completeness, not retrieval breadth), and its order is fixed:
expand_retrieval always runs first. On an already-adequately-retrieved corpus
(e.g. a single 20-chunk document), widening top_k frequently returns the same
top docs, so retrieval plateaus after attempt 1 as the COMMON case, not an
edge case — this was killing the loop after just 2 attempts almost every
time, before it ever reached decomposition, which is specifically designed
for the multi-part "missing aspect" completeness failures that dominate
finance-document questions ("revenue AND net income", "X and how did it
compare to Y"). Confirmed live: docx query asking for revenue+net income
retried once (expand_retrieval, identical retrieval score), stopped, and
shipped an answer missing net income with a permanent "could not be fully
verified" hedge — while decomposition (attempt 4, never reached) is the one
strategy built to split exactly this kind of query. Requiring BOTH retrieval
AND overall to be flat before stopping still bounds cost (rules 2+3 are the
hard caps regardless) while letting a later strategy get a real chance when
it's the completeness/citation dimensions moving, not retrieval.
"""

from __future__ import annotations

import time

from app.core.config import settings
from app.verification.verification_schema import RetryAttempt


class StoppingCriteria:
    def __init__(self) -> None:
        self.max_retries = settings.AGENT_VERIFY_MAX_RETRIES
        self.timeout_sec = settings.AGENT_VERIFY_TIMEOUT_SEC
        self.min_improvement_pct = settings.AGENT_VERIFY_MIN_IMPROVEMENT_PCT

    def should_stop(self, attempts: list[RetryAttempt], start_time: float) -> tuple[bool, str]:
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
            improvement_pct = (
                (last.scores.overall - prev.scores.overall) / max(prev.scores.overall, 1.0) * 100.0
            )
            retrieval_stalled = last.scores.retrieval <= prev.scores.retrieval
            overall_stalled = improvement_pct < self.min_improvement_pct
            if retrieval_stalled and overall_stalled:
                return (
                    True,
                    "retrieval_and_overall_confidence_not_improving "
                    f"(retrieval {prev.scores.retrieval:.1f}->{last.scores.retrieval:.1f}, "
                    f"overall {improvement_pct:.1f}% < {self.min_improvement_pct}%)",
                )

        return False, ""

    def best_attempt(self, attempts: list[RetryAttempt]) -> RetryAttempt | None:
        if not attempts:
            return None
        return max(attempts, key=lambda a: a.scores.overall)
