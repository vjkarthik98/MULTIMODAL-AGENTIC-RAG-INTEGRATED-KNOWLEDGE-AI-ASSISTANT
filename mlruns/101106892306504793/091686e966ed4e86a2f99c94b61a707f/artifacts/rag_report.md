# RAG Eval Report — Phase 25

**Generated:** 2026-05-26T03:48:05Z  
**Git SHA:** aa4da29  

## Suite: `full`

Duration: 3068.2s

| Metric | Value | n | Notes |
|--------|-------|---|-------|
| `audio.audio_wer` | nan | 0 | empty: no curated audio gold rows; run download_eval_corpus. |
| `e2e.answer_relevancy` | 0.5207 | 30 | judge=gguf_mistral |
| `e2e.citation_accuracy` | 1.0000 | 11 | judge=heuristic |
| `e2e.context_precision` | 0.5714 | 30 | judge=gguf_mistral |
| `e2e.context_recall` | 0.7639 | 30 | judge=gguf_mistral |
| `e2e.faithfulness` | 0.7284 | 30 | judge=gguf_mistral |
| `e2e.hallucination_rate` | 0.3000 | 30 | flagged=9/30 | examples:   'What was Apple's total net sales |
| `e2e.hit_rate` | 0.0000 | 17 |  |
| `e2e.mrr` | nan | 0 | empty: all queries had TODO relevant_chunk_ids |
| `e2e.ndcg_at_10` | nan | 0 | empty: all queries had TODO relevant_chunk_ids |
| `e2e.p50_sec` | 11.0973 | 30 | min=0.64s max=32.49s |
| `e2e.p95_sec` | 23.1694 | 30 |  |
| `e2e.p99_sec` | 30.8900 | 30 |  |
| `e2e.recall_at_10` | 0.0000 | 17 |  |
| `e2e.recall_at_5` | 0.0000 | 17 |  |
| `e2e.route_accuracy` | 0.0000 | 30 | correct=0/30 | misroutes:   'What was Apple's total net sale |
| `e2e.template_leak_rate` | 0.0000 | 30 | leaky=0/30 |
| `generation.answer_relevancy` | 0.5248 | 14 | judge=gguf_mistral |
| `generation.citation_accuracy` | 1.0000 | 6 | judge=heuristic |
| `generation.context_precision` | 0.7798 | 14 | judge=gguf_mistral |
| `generation.context_recall` | 0.9762 | 14 | judge=gguf_mistral |
| `generation.faithfulness` | 0.4643 | 14 | judge=gguf_mistral |
| `generation.generation_p50_sec` | 0.0021 | 14 | min=0.00s max=22.84s |
| `generation.generation_p95_sec` | 20.8549 | 14 |  |
| `generation.generation_p99_sec` | 22.4401 | 14 |  |
| `generation.hallucination_rate` | 0.6429 | 14 | flagged=9/14 | examples:   'What was Apple's total net sales |
| `generation.template_leak_rate` | 0.0000 | 14 | leaky=0/14 |
| `ocr.ocr_cer` | nan | 0 | empty: no curated image gold rows; run download_eval_corpus. |
| `ocr.ocr_exact_match` | nan | 0 | empty: no curated image gold rows |
| `ocr.ocr_wer` | nan | 0 | empty: no curated image gold rows |
| `regression.baseline_path` | 0.0000 | 0 | /home/ubuntu/multimodal-rag-assistant-1/app/eval/baselines/r |
| `regression.current.context_precision` | 0.1962 | 14 |  |
| `regression.current.hit_rate` | 0.8571 | 14 |  |
| `regression.current.mrr` | 0.4714 | 14 |  |
| `regression.current.ndcg_at_10` | 0.4188 | 14 |  |
| `regression.current.recall_at_10` | 0.5714 | 14 |  |
| `regression.current.recall_at_5` | 0.2857 | 14 |  |
| `regression.current.retrieval_p50_sec` | 1.5370 | 15 | min=1.21s max=2.68s |
| `regression.current.retrieval_p95_sec` | 2.5284 | 15 |  |
| `regression.current.retrieval_p99_sec` | 2.6508 | 15 |  |
| `retrieval.context_precision` | 0.1962 | 14 |  |
| `retrieval.hit_rate` | 0.8571 | 14 |  |
| `retrieval.mrr` | 0.4714 | 14 |  |
| `retrieval.ndcg_at_10` | 0.4188 | 14 |  |
| `retrieval.recall_at_10` | 0.5714 | 14 |  |
| `retrieval.recall_at_5` | 0.2857 | 14 |  |
| `retrieval.retrieval_p50_sec` | 1.4968 | 15 | min=1.17s max=4.21s |
| `retrieval.retrieval_p95_sec` | 2.9321 | 15 |  |
| `retrieval.retrieval_p99_sec` | 3.9549 | 15 |  |
| `routing.hybrid_with_web_rate` | 0.0000 | 4 | hybrid_with_web=0/4 | P1-4 misses: ['What are the latest S&P |
| `routing.route_accuracy` | 1.0000 | 12 | correct=12/12 |
| `routing.route_cm_direct_as_direct` | 2.0000 | 12 | confusion matrix cell |
| `routing.route_cm_hybrid_as_hybrid` | 4.0000 | 12 | confusion matrix cell |
| `routing.route_cm_memory_as_memory` | 1.0000 | 12 | confusion matrix cell |
| `routing.route_cm_rag_as_rag` | 5.0000 | 12 | confusion matrix cell |
| `video.caption_repetition_rate` | nan | 0 | empty: no curated video gold rows |
| `video.frame_caption_recall` | nan | 0 | empty: no curated video gold rows; run download_eval_corpus. |

