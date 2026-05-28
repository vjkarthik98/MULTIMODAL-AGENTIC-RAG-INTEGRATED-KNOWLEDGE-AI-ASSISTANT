# RAG Eval Report — Phase 25

**Generated:** 2026-05-28T10:20:57Z  
**Git SHA:** 4668745  

## Suite: `generation`

Duration: 150.4s

| Metric | Value | n | Notes |
|--------|-------|---|-------|
| `answer_relevancy` | 0.3378 | 21 | judge=lexical_fallback (ragas_error: CUDA out of memory. Tri |
| `citation_accuracy` | 1.0000 | 12 | judge=lexical_fallback (ragas_error: CUDA out of memory. Tri |
| `context_recall` | 0.6288 | 21 | judge=lexical_fallback (ragas_error: CUDA out of memory. Tri |
| `faithfulness` | 0.5623 | 21 | judge=lexical_fallback (ragas_error: CUDA out of memory. Tri |
| `generation_p50_sec` | 0.0022 | 21 | min=0.00s max=66.66s |
| `generation_p95_sec` | 25.3111 | 21 |  |
| `generation_p99_sec` | 58.3886 | 21 |  |
| `hallucination_rate` | 0.7143 | 21 | flagged=15/21 | examples:   'What was Apple's total net sale |
| `template_leak_rate` | 0.0000 | 21 | leaky=0/21 |

