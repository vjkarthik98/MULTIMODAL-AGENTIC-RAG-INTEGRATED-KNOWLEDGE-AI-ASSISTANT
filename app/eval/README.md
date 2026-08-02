# Phase 25 — Evaluation Harness & RAG Quality Metrics

## Quick start

```bash
# Install eval deps (once)
pip install ragas==0.1.21 jiwer>=3.0.0 mlflow>=2.10.0

# Bootstrap gold set candidates (human review required before full eval)
python -m app.eval.datasets.build_gold_set --modality all

# Ingest eval corpus into default KB
python -m app.eval.datasets.build_gold_set --ingest

# Run individual suites
python -m app.eval.run --suite retrieval
python -m app.eval.run --suite generation
python -m app.eval.run --suite ocr
python -m app.eval.run --suite audio
python -m app.eval.run --suite video
python -m app.eval.run --suite routing

# Run full suite (CI gate command)
python -m app.eval.run --suite full
echo "exit code: $?"   # 0 = all thresholds pass, 1 = breach, 2 = infra/data error

# Gate proof — weaken pipeline, must exit non-zero
python -m app.eval.run --suite full --weaken top_k=1,no_rerank
echo "exit code: $?"   # must be 1

# Regression vs baseline
python -m app.eval.run --suite regression --baseline app/eval/baselines/rag_report_v1.json
```

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | All thresholds passed |
| `1` | One or more thresholds breached |
| `2` | Infra / data error (Qdrant down, gold file missing, etc.) |

## Suites

| Suite | What it measures | Gold file |
|-------|-----------------|-----------|
| `retrieval` | recall@k, MRR, nDCG, context_precision against real `Retriever.retrieval()` | `text_gold.jsonl` |
| `generation` | faithfulness, answer_relevancy, context_recall, citation_accuracy via the Qwen2.5-7B judge | all gold files |
| `ocr` | CER, WER of image OCR against ground-truth text | `image_gold.jsonl` |
| `audio` | WER of Whisper transcript vs gold transcript | `audio_gold.jsonl` |
| `video` | frame-caption recall, caption repetition rate, transcript WER | `video_gold.jsonl` |
| `routing` | route accuracy, hybrid-with-web rate via `AgentController.handle()` | `routing_gold.jsonl` |
| `e2e` | all metrics, calls FastAPI `/query` end-to-end | `e2e_gold.jsonl` |
| `multimodal` | per-modality cross-modal questions | all gold files |
| `regression` | diffs current run against committed baseline | `baselines/rag_report_v1.json` |
| `full` | runs every suite; this is the Phase 29 CI gate | all |

## Gold set schema

Each `.jsonl` file in `datasets/gold/` has rows shaped like:

```json
{
  "id": "txt-0001",
  "modality": "txt",
  "source_file": "10k_aapl_2023.txt",
  "query": "What was Apple's services revenue in FY2023?",
  "relevant_chunk_ids": ["doc_aapl10k_chunk_142"],
  "reference_answer": "Services revenue was $85.2 billion in FY2023.",
  "expected_route": "rag",
  "tags": ["financial-reasoning", "fact-extraction"],
  "added_by": "human",
  "added_at": "2026-05-25"
}
```

Rows with `"relevant_chunk_ids": "TODO"` or `"reference_answer": "TODO"` are candidates
awaiting human review. **Never run metrics on TODO rows** — the runner skips them.

## Thresholds

`thresholds.yaml` defines min/max for every metric. Initial values are guesses — after
the first baseline run they are rewritten to `baseline * 0.95`. Every threshold has a
`why` comment.

## Adding a new gold triple

1. Add a row to the appropriate `datasets/gold/*.jsonl` file.
2. Set `"added_by": "human"` and fill in `relevant_chunk_ids` + `reference_answer`.
3. Run `python -m app.eval.datasets.build_gold_set --validate` to check schema.
4. Update `datasets/manifest.yaml` SHA-256 entry (done automatically by build_gold_set).

## Stubs for future phases

`stubs/` contains documented designs for:
- `drift_detection.md` — Phase 31 (monitoring)
- `online_eval.md` — Phase 31
- `human_eval.md` — Phase 31
