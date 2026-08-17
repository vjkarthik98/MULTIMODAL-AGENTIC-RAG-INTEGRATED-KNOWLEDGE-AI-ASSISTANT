# RAG Eval Report — Phase 25

**Generated:** 2026-08-17T08:10:19Z  
**Git SHA:** a931a85  

## Suite: `hallucination`

Duration: 776.3s

| Metric | Value | n | Notes |
|--------|-------|---|-------|
| `avg_retry_count` | 0.2935 | 92 | mean verification retries per query (cost signal) |
| `citation_accuracy` | 1.0000 | 82 | judge=heuristic |
| `citation_accuracy_v2` | 0.8478 | 92 | fraction of answers with zero bad citations (CitationVerifie |
| `fabrication_rate` | 0.0722 | 97 | flagged=7/97 | examples:   'Did Chair Powell rule out a rate |
| `grounding_success_rate` | 0.9348 | 92 | fraction of answers with zero unsupported claims (Groundedne |
| `hallucination_rate` | 0.2371 | 97 | flagged=23/97 | examples:   'Did Chair Powell rule out a rat |
| `omission_rate` | 0.1856 | 97 | flagged=18/97 | examples:   'Did Chair Powell rule out a rat |
| `retry_success_rate` | 0.2222 | 27 | retried=27/92 | eventually PASS=6 |
| `template_leak_rate` | 0.0000 | 98 | leaky=0/98 |
| `verification_latency_p50` | 2.5463 | 92 | min=1.02s max=12.54s |
| `verification_latency_p95` | 7.5571 | 92 |  |

