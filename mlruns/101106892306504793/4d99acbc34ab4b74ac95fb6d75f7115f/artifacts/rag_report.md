# RAG Eval Report — Phase 25

**Generated:** 2026-05-26T01:48:40Z  
**Git SHA:** aa4da29  

## Suite: `full`

Duration: 145.9s

| Metric | Value | n | Notes |
|--------|-------|---|-------|
| `audio.audio_wer` | nan | 0 | empty: no curated audio gold rows; run download_eval_corpus. |
| `e2e.p50_sec` | nan | 0 | empty: no samples |
| `e2e.p95_sec` | nan | 0 | empty: no samples |
| `e2e.p99_sec` | nan | 0 | empty: no samples |
| `generation.answer_relevancy` | 0.4021 | 14 | judge=lexical_fallback (ragas_error: Can't instantiate abstr |
| `generation.citation_accuracy` | 1.0000 | 6 | judge=lexical_fallback (ragas_error: Can't instantiate abstr |
| `generation.context_recall` | 0.4926 | 14 | judge=lexical_fallback (ragas_error: Can't instantiate abstr |
| `generation.faithfulness` | 0.5790 | 14 | judge=lexical_fallback (ragas_error: Can't instantiate abstr |
| `generation.generation_p50_sec` | 0.0022 | 14 | min=0.00s max=16.09s |
| `generation.generation_p95_sec` | 15.3110 | 14 |  |
| `generation.generation_p99_sec` | 15.9380 | 14 |  |
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
| `regression.current.retrieval_p50_sec` | 1.5356 | 15 | min=1.20s max=2.61s |
| `regression.current.retrieval_p95_sec` | 2.4803 | 15 |  |
| `regression.current.retrieval_p99_sec` | 2.5830 | 15 |  |
| `retrieval.context_precision` | 0.1952 | 14 |  |
| `retrieval.hit_rate` | 0.8571 | 14 |  |
| `retrieval.mrr` | 0.4714 | 14 |  |
| `retrieval.ndcg_at_10` | 0.4188 | 14 |  |
| `retrieval.recall_at_10` | 0.5714 | 14 |  |
| `retrieval.recall_at_5` | 0.2857 | 14 |  |
| `retrieval.retrieval_p50_sec` | 1.5123 | 15 | min=1.18s max=4.25s |
| `retrieval.retrieval_p95_sec` | 2.9631 | 15 |  |
| `retrieval.retrieval_p99_sec` | 3.9899 | 15 |  |
| `routing.hybrid_with_web_rate` | nan | 0 | empty: no hybrid-route queries in set |
| `routing.route_accuracy` | 0.0000 | 12 | correct=0/12 | misroutes:   'What was Apple's revenue in FY2 |
| `routing.route_cm_direct_as_` | 3.0000 | 12 | confusion matrix cell |
| `routing.route_cm_hybrid_as_` | 3.0000 | 12 | confusion matrix cell |
| `routing.route_cm_memory_as_` | 1.0000 | 12 | confusion matrix cell |
| `routing.route_cm_rag_as_` | 3.0000 | 12 | confusion matrix cell |
| `routing.route_cm_search_as_` | 2.0000 | 12 | confusion matrix cell |
| `video.caption_repetition_rate` | nan | 0 | empty: no curated video gold rows |
| `video.frame_caption_recall` | nan | 0 | empty: no curated video gold rows; run download_eval_corpus. |

### Breaches / Errors

- `e2e.query_error_txt-0001`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_txt-0002`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_txt-0003`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_txt-0004`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_txt-0005`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_txt-0006`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_txt-0007`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_txt-0008`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_txt-0009`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_txt-0010`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_txt-0011`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_txt-0012`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_txt-retrieval-miss-001`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_txt-halluc-guard-001`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_route-0001`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_route-0002`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_route-0003`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_route-0004`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_route-0005`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_route-0006`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_route-0007`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_route-0008`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_route-0009`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_route-0010`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_route-0011`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_route-0012`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_e2e-0001`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_e2e-0002`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_e2e-0003`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query
- `e2e.query_error_e2e-0004`: 404 Client Error: Not Found for url: http://127.0.0.1:8000/query

