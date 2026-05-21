# Memory Index

- [Project Performance Baselines](project_perf_baselines.md) — startup latency improved 25s→7s via device manager; query responses <60s on CPU; user is satisfied with current perf
- [Corruption Policy](project_corruption_policy.md) — strict preflight gate shipped (Tier 1, HTTP 422); per-chunk quality_score is Phase 26 follow-up (Tier 2). Industry-standard two-tier pattern.
- [Phase 26 Guardrails Follow-ups](phase26_guardrails_followups.md) — three concrete defects from T5 benchmark: literal injection patterns, substring memory matcher, memory/direct paths return router stubs. Acceptance criteria for Phase 26.
