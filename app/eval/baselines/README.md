# Eval Baselines

This directory stores committed baseline reports.

## Files

- `rag_report_v1.json` — first baseline (created after Phase 25 first run)
- `rag_report_v1.md` — human-readable version of the first baseline

## How baselines are created

After the first full eval run:

```bash
python -m app.eval.run --suite full
cp app/eval/reports/rag_report.json app/eval/baselines/rag_report_v1.json
cp app/eval/reports/rag_report.md   app/eval/baselines/rag_report_v1.md
git add app/eval/baselines/
git commit -m "feat(eval): commit Phase 25 eval baseline v1"
```

## Regression checking

```bash
python -m app.eval.run --suite regression --baseline app/eval/baselines/rag_report_v1.json
```

The regression runner flags any metric that drops more than 5% from the baseline value.

## Threshold update process

After a baseline is committed, update `thresholds.yaml` to:
  `min_value = baseline_value * 0.95`

This is honest regression-prevention: we flag when the system degrades, not when it
fails to meet an aspirational target we've never actually hit.
