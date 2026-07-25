# Stub: Drift Detection (Phase 31)

## What this will be

Embedding drift + query-distribution drift monitoring integrated with the Prometheus /
Grafana stack that Phase 31 (monitoring) builds.

## Design

### Embedding drift
- Sample production embeddings (1% of queries) via the OTel pipeline
- Compute Population Stability Index (PSI) between baseline embedding distribution and
  current week's distribution
- Alert if PSI > 0.2 (significant drift) on any major cluster

### Query distribution drift
- Track query-length histograms, route distribution, intent class distribution
- KL divergence from baseline distribution
- Alert if KL > 0.3

### Integration point
- `app/eval/metrics/drift.py` (Phase 31 creates this)
- Reads Prometheus counters exposed by `app/core/infra_registry.py`
- Writes drift metrics back to Prometheus for Grafana alerting

### Phase 29 CI gate hook
- `python -m app.eval.run --suite drift` (Phase 31)
- Gate fails if PSI > threshold or KL > threshold

## What exists today (Phase 25)
- Nothing — this file documents the design for Phase 31
- The `app/eval/metrics/` structure is already there to add `drift.py`
