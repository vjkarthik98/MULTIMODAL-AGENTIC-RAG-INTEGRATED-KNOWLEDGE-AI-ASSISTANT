# RAG Eval Report — Phase 25

**Generated:** 2026-05-26T12:26:14Z  
**Git SHA:** aa4da29  

## Suite: `full`

Duration: 5253.1s

| Metric | Value | n | Notes |
|--------|-------|---|-------|
| `audio.audio_wer` | nan | 0 | empty: no completed transcript/reference pairs |
| `generation.answer_relevancy` | 0.1728 | 21 | judge=gguf_mistral |
| `generation.citation_accuracy` | 1.0000 | 21 | judge=heuristic |
| `generation.context_precision` | 0.8472 | 21 | judge=gguf_mistral |
| `generation.context_recall` | 0.9583 | 21 | judge=gguf_mistral |
| `generation.faithfulness` | nan | 21 | judge=gguf_mistral |
| `generation.generation_p50_sec` | 0.3216 | 21 | min=0.31s max=14.84s |
| `generation.generation_p95_sec` | 0.5983 | 21 |  |
| `generation.generation_p99_sec` | 11.9906 | 21 |  |
| `generation.hallucination_rate` | 0.0000 | 21 | flagged=0/21 |
| `generation.template_leak_rate` | 0.0000 | 21 | leaky=0/21 |
| `ocr.ocr_cer` | 1.9594 | 2 |  |
| `ocr.ocr_exact_match` | 0.0000 | 2 |  |
| `ocr.ocr_wer` | 2.7868 | 2 |  |
| `regression.baseline_path` | 0.0000 | 0 | /home/ubuntu/multimodal-rag-assistant-1/app/eval/baselines/r |
| `regression.current.context_precision` | 0.2159 | 14 |  |
| `regression.current.hit_rate` | 1.0000 | 14 |  |
| `regression.current.mrr` | 0.8150 | 14 |  |
| `regression.current.ndcg_at_10` | 0.6178 | 14 |  |
| `regression.current.recall_at_10` | 0.6786 | 14 |  |
| `regression.current.recall_at_5` | 0.4405 | 14 |  |
| `regression.current.retrieval_p50_sec` | 1.4795 | 16 | min=1.15s max=2.29s |
| `regression.current.retrieval_p95_sec` | 2.2914 | 16 |  |
| `regression.current.retrieval_p99_sec` | 2.2931 | 16 |  |
| `retrieval.context_precision` | 0.2101 | 14 |  |
| `retrieval.hit_rate` | 1.0000 | 14 |  |
| `retrieval.mrr` | 0.8150 | 14 |  |
| `retrieval.ndcg_at_10` | 0.6077 | 14 |  |
| `retrieval.recall_at_10` | 0.6548 | 14 |  |
| `retrieval.recall_at_5` | 0.4405 | 14 |  |
| `retrieval.retrieval_p50_sec` | 0.1466 | 16 | min=0.15s max=34.19s |
| `retrieval.retrieval_p95_sec` | 25.2489 | 16 |  |
| `retrieval.retrieval_p99_sec` | 32.4021 | 16 |  |
| `routing.hybrid_with_web_rate` | 0.0000 | 4 | hybrid_with_web=0/4 | P1-4 misses: ['What are the latest S&P |
| `routing.route_accuracy` | 1.0000 | 12 | correct=12/12 |
| `routing.route_cm_direct_as_direct` | 2.0000 | 12 | confusion matrix cell |
| `routing.route_cm_hybrid_as_hybrid` | 4.0000 | 12 | confusion matrix cell |
| `routing.route_cm_memory_as_memory` | 1.0000 | 12 | confusion matrix cell |
| `routing.route_cm_rag_as_rag` | 5.0000 | 12 | confusion matrix cell |

### Breaches / Errors

- `e2e.server_unreachable`: FastAPI server not reachable at http://127.0.0.1:8000. Start the server with: uvicorn app.main:app before running the e2e suite.
- `audio.ingest_error_audio-0001`: NO_VALID_AUDIO_SEGMENTS
- `audio.ingest_error_audio-0002`: NO_VALID_AUDIO_SEGMENTS
- `video.runner_error`: unsupported operand type(s) for /: 'PosixPath' and 'NoneType'

