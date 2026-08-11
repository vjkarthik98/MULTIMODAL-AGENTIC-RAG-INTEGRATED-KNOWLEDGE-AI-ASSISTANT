# RAG Eval Report — Phase 25

**Generated:** 2026-08-08T13:34:04Z  
**Git SHA:** 8484a75  

## Suite: `generation`

Duration: 332.9s

| Metric | Value | n | Notes |
|--------|-------|---|-------|
| `answer_correctness` | 0.6607 | 14 | judge=qwen2.5_7b |
| `answer_relevancy` | 0.5893 | 14 | judge=qwen2.5_7b |
| `citation_accuracy` | 1.0000 | 14 | judge=heuristic |
| `context_recall` | 0.6950 | 10 | deterministic (reference facts recoverable from context) |
| `faithfulness` | 0.3036 | 14 | judge=qwen2.5_7b |
| `finance_fidelity` | 0.7202 | 14 | avg over 14 queries (strict 0.5% tol, no scale bridging) |
| `generation_p50_sec` | 16.6737 | 14 | min=10.01s max=62.34s |
| `generation_p95_sec` | 38.5809 | 14 |  |
| `generation_p99_sec` | 57.5875 | 14 |  |
| `hallucination_rate` | 0.4286 | 14 | flagged=6/14 | examples:   'By how much did the FOMC cut rat |
| `template_leak_rate` | 0.0000 | 14 | leaky=0/14 |

