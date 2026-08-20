# RAG Eval Report — Phase 25

**Generated:** 2026-08-20T11:43:32Z  
**Git SHA:** d920537  

## Suite: `generation`

Duration: 171.5s

| Metric | Value | n | Notes |
|--------|-------|---|-------|
| `answer_correctness` | 0.7857 | 14 | judge=qwen2.5_7b |
| `answer_relevancy` | 0.8214 | 14 | judge=qwen2.5_7b |
| `avg_retry_count` | 0.3571 | 14 | mean verification retries per query (cost signal) |
| `citation_accuracy` | 1.0000 | 14 | judge=heuristic |
| `citation_accuracy_v2` | 0.9286 | 14 | fraction of answers with zero bad citations (CitationVerifie |
| `context_recall` | 0.8738 | 14 | deterministic (reference facts recoverable from context) |
| `fabrication_rate` | 0.0000 | 14 | flagged=0/14 |
| `faithfulness` | 0.3929 | 14 | judge=qwen2.5_7b |
| `finance_fidelity` | 0.9643 | 14 | avg over 14 queries (strict 0.5% tol, no scale bridging) |
| `generation_p50_sec` | 8.5689 | 14 | min=6.82s max=11.23s |
| `generation_p95_sec` | 11.0306 | 14 |  |
| `generation_p99_sec` | 11.1870 | 14 |  |
| `grounding_success_rate` | 1.0000 | 14 | fraction of answers with zero unsupported claims (Groundedne |
| `hallucination_rate` | 0.2857 | 14 | flagged=4/14 | examples:   'What was Apple's diluted EPS for |
| `omission_rate` | 0.2857 | 14 | flagged=4/14 | examples:   'What was Apple's diluted EPS for |
| `retry_success_rate` | 0.0000 | 5 | retried=5/14 | eventually PASS=0 |
| `template_leak_rate` | 0.0000 | 14 | leaky=0/14 |
| `verification_latency_p50` | 3.9718 | 14 | min=3.27s max=7.20s |
| `verification_latency_p95` | 6.9222 | 14 |  |

