# RAG Eval Report — Phase 25

**Generated:** 2026-07-19T15:21:07Z  
**Git SHA:** 5e28e55  

## Suite: `generation`

Duration: 932.2s

| Metric | Value | n | Notes |
|--------|-------|---|-------|
| `answer_correctness` | 0.3274 | 42 | judge=prometheus_2_7b |
| `answer_relevancy` | 0.3452 | 42 | judge=prometheus_2_7b |
| `citation_accuracy` | 1.0000 | 15 | judge=heuristic |
| `context_recall` | 0.0655 | 42 | judge=prometheus_2_7b |
| `faithfulness` | 0.3988 | 42 | judge=prometheus_2_7b |
| `finance_fidelity` | 0.8794 | 42 | avg over 42 queries (strict 0.5% tol, no scale bridging) |
| `generation_p50_sec` | 11.2928 | 42 | min=0.48s max=23.44s |
| `generation_p95_sec` | 19.4658 | 42 |  |
| `generation_p99_sec` | 21.9195 | 42 |  |
| `hallucination_rate` | 0.7619 | 42 | flagged=32/42 | examples:   'By how much did the FOMC lower  |
| `template_leak_rate` | 0.0000 | 42 | leaky=0/42 |

