# RAG Eval Report — Phase 25

**Generated:** 2026-05-28T14:01:37Z  
**Git SHA:** 4668745  

## Suite: `generation`

Duration: 890.1s

| Metric | Value | n | Notes |
|--------|-------|---|-------|
| `answer_relevancy` | 0.5454 | 21 | judge=gguf_mistral |
| `citation_accuracy` | 1.0000 | 9 | judge=heuristic |
| `context_precision` | 0.8373 | 21 | judge=gguf_mistral |
| `context_recall` | 0.9667 | 21 | judge=gguf_mistral |
| `faithfulness` | 0.2905 | 21 | judge=gguf_mistral |
| `generation_p50_sec` | 0.0343 | 21 | min=0.03s max=21.10s |
| `generation_p95_sec` | 20.3570 | 21 |  |
| `generation_p99_sec` | 20.9533 | 21 |  |
| `hallucination_rate` | 0.6667 | 21 | flagged=14/21 | examples:   'What was Apple's total net sale |
| `template_leak_rate` | 0.0000 | 21 | leaky=0/21 |

