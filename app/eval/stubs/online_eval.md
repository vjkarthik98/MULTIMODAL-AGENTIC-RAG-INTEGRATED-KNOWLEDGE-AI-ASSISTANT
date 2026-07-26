# Stub: Online Evaluation (Phase 31)

## What this will be

Shadow Ragas scoring on a sample of live production queries — without affecting the user
experience or response latency.

## Design

### Sampling
- 1% of production queries are selected (configurable via `ONLINE_EVAL_SAMPLE_RATE`)
- Selection is deterministic via hash of session_id (reproducible, not random per call)
- Sampled queries + their retrieved context + generated answer are logged to MongoDB
  audit collection with `eval_shadow: true`

### Async scoring
- Background task (FastAPI BackgroundTasks or Celery) picks up sampled queries
- Runs the same Ragas metrics as offline eval: faithfulness, answer_relevancy
- Uses the GGUF local judge — same as offline, no extra cost
- Writes scores to `MONGO_EVAL_SHADOW_COLLECTION`

### Grafana dashboard
- Panel showing rolling 7-day faithfulness / answer_relevancy means
- Alert if rolling mean drops below threshold for 3 consecutive days

## Integration point
- `app/eval/online/shadow_scorer.py` (Phase 31 creates this)
- Hook: `app/api/api_routes.py:query_rag()` adds to background queue after response

## What exists today (Phase 25)
- Nothing — this file documents the design for Phase 31
- The offline harness (`runner.py`) is the model for this
