# Stub: Human Evaluation UI (Phase 31 / Phase 28 extension)

## What this will be

A lightweight Gradio panel (Phase 28 UI extension) where subject-matter experts (SMEs)
can review sampled answers and provide thumbs-up/thumbs-down + free-text feedback.

## Design

### UI
- Gradio tab: "Eval Review"
- Shows: query, retrieved context, generated answer
- Controls: 👍 / 👎 rating + optional 1-line comment
- Pre-populated from MongoDB `eval_shadow` collection (see online_eval.md)

### Data flow
1. Sampled query/answer stored in MongoDB (online eval)
2. Human reviewer opens Gradio panel, sees unevaluated samples
3. Submits rating — stored back to MongoDB as `human_eval_score`
4. Aggregate human scores tracked alongside auto-judge scores in Grafana

### Quality gate extension
- Phase 29 CI gate can optionally check `human_eval_score_avg >= 0.8` before merging
- Gated behind `ENABLE_HUMAN_EVAL_GATE=true` — off by default (requires review queue)

### SME agreement metric
- When multiple reviewers rate same query: inter-annotator agreement (Cohen's kappa)
- Low kappa → ambiguous query class → add to active-learning queue for gold set

## Integration point
- `app/eval/human/gradio_panel.py` (Phase 28/31 creates this)
- `app/eval/human/collector.py` — MongoDB writer for human scores

## What exists today (Phase 25)
- Nothing — this file documents the design
- Gold set has `added_by` and `added_at` fields to track human curation provenance
