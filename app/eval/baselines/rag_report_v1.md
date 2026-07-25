# RAG Eval Report — Phase 25

**Generated:** 2026-05-26T06:11:45Z  
**Git SHA:** aa4da29  

## Suite: `full`

Duration: 3391.0s

| Metric | Value | n | Notes |
|--------|-------|---|-------|
| `audio.audio_wer` | nan | 0 | empty: no curated audio gold rows; run download_eval_corpus. |
| `e2e.answer_relevancy` | 0.5622 | 29 | judge=gguf_mistral |
| `e2e.citation_accuracy` | 1.0000 | 12 | judge=heuristic |
| `e2e.context_precision` | 0.6193 | 29 | judge=gguf_mistral |
| `e2e.context_recall` | 0.7500 | 29 | judge=gguf_mistral |
| `e2e.faithfulness` | 0.6227 | 29 | judge=gguf_mistral |
| `e2e.hallucination_rate` | 0.3793 | 29 | flagged=11/29 | examples:   'What was Apple's total net sale |
| `e2e.hit_rate` | 0.0000 | 17 |  |
| `e2e.mrr` | nan | 0 | empty: all queries had TODO relevant_chunk_ids |
| `e2e.ndcg_at_10` | nan | 0 | empty: all queries had TODO relevant_chunk_ids |
| `e2e.p50_sec` | 8.7346 | 29 | min=0.62s max=30.27s |
| `e2e.p95_sec` | 26.3293 | 29 |  |
| `e2e.p99_sec` | 29.4629 | 29 |  |
| `e2e.recall_at_10` | 0.0000 | 17 |  |
| `e2e.recall_at_5` | 0.0000 | 17 |  |
| `e2e.route_accuracy` | 0.0000 | 29 | correct=0/29 | misroutes:   'What was Apple's total net sale |
| `e2e.template_leak_rate` | 0.0000 | 29 | leaky=0/29 |
| `generation.answer_relevancy` | 0.5225 | 14 | judge=gguf_mistral |
| `generation.citation_accuracy` | 1.0000 | 5 | judge=heuristic |
| `generation.context_precision` | 0.9643 | 14 | judge=gguf_mistral |
| `generation.context_recall` | 1.0000 | 14 | judge=gguf_mistral |
| `generation.faithfulness` | 0.5385 | 14 | judge=gguf_mistral |
| `generation.generation_p50_sec` | 10.8337 | 14 | min=5.70s max=19.55s |
| `generation.generation_p95_sec` | 17.3462 | 14 |  |
| `generation.generation_p99_sec` | 19.1062 | 14 |  |
| `generation.hallucination_rate` | 0.5714 | 14 | flagged=8/14 | examples:   'What was Apple's total net sales |
| `generation.template_leak_rate` | 0.0000 | 14 | leaky=0/14 |
| `ocr.ocr_cer` | nan | 0 | empty: no curated image gold rows; run download_eval_corpus. |
| `ocr.ocr_exact_match` | nan | 0 | empty: no curated image gold rows |
| `ocr.ocr_wer` | nan | 0 | empty: no curated image gold rows |
| `regression.baseline_path` | 0.0000 | 0 | /home/ubuntu/multimodal-rag-assistant-1/app/eval/baselines/r |
| `regression.current.context_precision` | 0.1960 | 14 |  |
| `regression.current.hit_rate` | 1.0000 | 14 |  |
| `regression.current.mrr` | 0.8133 | 14 |  |
| `regression.current.ndcg_at_10` | 0.6045 | 14 |  |
| `regression.current.recall_at_10` | 0.6548 | 14 |  |
| `regression.current.recall_at_5` | 0.4405 | 14 |  |
| `regression.current.retrieval_p50_sec` | 1.5257 | 15 | min=1.20s max=2.66s |
| `regression.current.retrieval_p95_sec` | 2.5273 | 15 |  |
| `regression.current.retrieval_p99_sec` | 2.6301 | 15 |  |
| `retrieval.context_precision` | 0.1960 | 14 |  |
| `retrieval.hit_rate` | 1.0000 | 14 |  |
| `retrieval.mrr` | 0.8133 | 14 |  |
| `retrieval.ndcg_at_10` | 0.6045 | 14 |  |
| `retrieval.recall_at_10` | 0.6548 | 14 |  |
| `retrieval.recall_at_5` | 0.4405 | 14 |  |
| `retrieval.retrieval_p50_sec` | 1.5145 | 15 | min=1.18s max=4.21s |
| `retrieval.retrieval_p95_sec` | 2.9388 | 15 |  |
| `retrieval.retrieval_p99_sec` | 3.9553 | 15 |  |
| `routing.hybrid_with_web_rate` | 0.0000 | 4 | hybrid_with_web=0/4 | P1-4 misses: ['What are the latest S&P |
| `routing.route_accuracy` | 1.0000 | 12 | correct=12/12 |
| `routing.route_cm_direct_as_direct` | 2.0000 | 12 | confusion matrix cell |
| `routing.route_cm_hybrid_as_hybrid` | 4.0000 | 12 | confusion matrix cell |
| `routing.route_cm_memory_as_memory` | 1.0000 | 12 | confusion matrix cell |
| `routing.route_cm_rag_as_rag` | 5.0000 | 12 | confusion matrix cell |
| `video.caption_repetition_rate` | nan | 0 | empty: no curated video gold rows |
| `video.frame_caption_recall` | nan | 0 | empty: no curated video gold rows; run download_eval_corpus. |

### Breaches / Errors

- `e2e.query_error_e2e-0004`: HTTPConnectionPool(host='127.0.0.1', port=8000): Read timed out. (read timeout=120)

