# RAG Eval Report — Phase 25

**Generated:** 2026-05-26T07:12:56Z  
**Git SHA:** aa4da29  

## Suite: `e2e`

Duration: 2876.9s

| Metric | Value | n | Notes |
|--------|-------|---|-------|
| `answer_relevancy` | 0.5625 | 37 | judge=gguf_mistral |
| `citation_accuracy` | 1.0000 | 17 | judge=heuristic |
| `context_precision` | 0.6068 | 37 | judge=gguf_mistral |
| `context_recall` | 0.7613 | 37 | judge=gguf_mistral |
| `faithfulness` | 0.5611 | 37 | judge=gguf_mistral |
| `hallucination_rate` | 0.3784 | 37 | flagged=14/37 | examples:   'What was Apple's total net sale |
| `hit_rate` | 0.6250 | 24 |  |
| `mrr` | 0.5347 | 24 |  |
| `ndcg_at_10` | 0.3725 | 24 |  |
| `p50_sec` | 0.0076 | 37 | min=0.01s max=29.30s |
| `p95_sec` | 22.9488 | 37 |  |
| `p99_sec` | 28.0936 | 37 |  |
| `recall_at_10` | 0.3611 | 24 |  |
| `recall_at_5` | 0.3611 | 24 |  |
| `route_accuracy` | 0.9730 | 37 | correct=36/37 | misroutes:   'What is the current stock pric |
| `template_leak_rate` | 0.0000 | 37 | leaky=0/37 |

