# RAG Eval Report — Phase 25

**Generated:** 2026-05-26T01:55:18Z  
**Git SHA:** aa4da29  

## Suite: `full`

Duration: 152.0s

| Metric | Value | n | Notes |
|--------|-------|---|-------|
| `audio.audio_wer` | nan | 0 | empty: no curated audio gold rows; run download_eval_corpus. |
| `e2e.answer_relevancy` | 0.0121 | 30 | judge=lexical_fallback (ragas_error: Can't instantiate abstr |
| `e2e.citation_accuracy` | nan | 0 | empty: insufficient data |
| `e2e.context_precision` | nan | 0 | empty: all queries had TODO relevant_chunk_ids |
| `e2e.context_recall` | nan | 0 | empty: insufficient data |
| `e2e.faithfulness` | nan | 0 | empty: insufficient data |
| `e2e.hallucination_rate` | 0.0000 | 30 | flagged=0/30 |
| `e2e.hit_rate` | 0.0000 | 17 |  |
| `e2e.mrr` | nan | 0 | empty: all queries had TODO relevant_chunk_ids |
| `e2e.ndcg_at_10` | nan | 0 | empty: all queries had TODO relevant_chunk_ids |
| `e2e.p50_sec` | 0.8709 | 30 | min=0.26s max=7.04s |
| `e2e.p95_sec` | 2.4190 | 30 |  |
| `e2e.p99_sec` | 5.6997 | 30 |  |
| `e2e.recall_at_10` | 0.0000 | 17 |  |
| `e2e.recall_at_5` | 0.0000 | 17 |  |
| `e2e.route_accuracy` | 0.0000 | 30 | correct=0/30 | misroutes:   'What was Apple's total net sale |
| `e2e.template_leak_rate` | 0.0000 | 30 | leaky=0/30 |
| `generation.answer_relevancy` | 0.4086 | 14 | judge=lexical_fallback (ragas_error: Can't instantiate abstr |
| `generation.citation_accuracy` | 1.0000 | 5 | judge=lexical_fallback (ragas_error: Can't instantiate abstr |
| `generation.context_recall` | 0.4926 | 14 | judge=lexical_fallback (ragas_error: Can't instantiate abstr |
| `generation.faithfulness` | 0.5751 | 14 | judge=lexical_fallback (ragas_error: Can't instantiate abstr |
| `generation.generation_p50_sec` | 0.0021 | 14 | min=0.00s max=15.91s |
| `generation.generation_p95_sec` | 15.4449 | 14 |  |
| `generation.generation_p99_sec` | 15.8172 | 14 |  |
| `generation.hallucination_rate` | 0.7857 | 14 | flagged=11/14 | examples:   'What was Apple's total net sale |
| `generation.template_leak_rate` | 0.0000 | 14 | leaky=0/14 |
| `ocr.ocr_cer` | nan | 0 | empty: no curated image gold rows; run download_eval_corpus. |
| `ocr.ocr_exact_match` | nan | 0 | empty: no curated image gold rows |
| `ocr.ocr_wer` | nan | 0 | empty: no curated image gold rows |
| `regression.baseline_path` | 0.0000 | 0 | /home/ubuntu/multimodal-rag-assistant-1/app/eval/baselines/r |
| `regression.current.context_precision` | 0.1952 | 14 |  |
| `regression.current.hit_rate` | 0.8571 | 14 |  |
| `regression.current.mrr` | 0.4714 | 14 |  |
| `regression.current.ndcg_at_10` | 0.4188 | 14 |  |
| `regression.current.recall_at_10` | 0.5714 | 14 |  |
| `regression.current.recall_at_5` | 0.2857 | 14 |  |
| `regression.current.retrieval_p50_sec` | 1.5262 | 15 | min=1.21s max=2.65s |
| `regression.current.retrieval_p95_sec` | 2.5134 | 15 |  |
| `regression.current.retrieval_p99_sec` | 2.6187 | 15 |  |
| `retrieval.context_precision` | 0.1952 | 14 |  |
| `retrieval.hit_rate` | 0.8571 | 14 |  |
| `retrieval.mrr` | 0.4714 | 14 |  |
| `retrieval.ndcg_at_10` | 0.4188 | 14 |  |
| `retrieval.recall_at_10` | 0.5714 | 14 |  |
| `retrieval.recall_at_5` | 0.2857 | 14 |  |
| `retrieval.retrieval_p50_sec` | 1.4989 | 15 | min=1.17s max=4.22s |
| `retrieval.retrieval_p95_sec` | 2.9412 | 15 |  |
| `retrieval.retrieval_p99_sec` | 3.9666 | 15 |  |
| `routing.hybrid_with_web_rate` | 0.0000 | 4 | hybrid_with_web=0/4 | P1-4 misses: ['What are the latest S&P |
| `routing.route_accuracy` | 1.0000 | 12 | correct=12/12 |
| `routing.route_cm_direct_as_direct` | 2.0000 | 12 | confusion matrix cell |
| `routing.route_cm_hybrid_as_hybrid` | 4.0000 | 12 | confusion matrix cell |
| `routing.route_cm_memory_as_memory` | 1.0000 | 12 | confusion matrix cell |
| `routing.route_cm_rag_as_rag` | 5.0000 | 12 | confusion matrix cell |
| `video.caption_repetition_rate` | nan | 0 | empty: no curated video gold rows |
| `video.frame_caption_recall` | nan | 0 | empty: no curated video gold rows; run download_eval_corpus. |

