---
name: project-perf-baselines
description: Known latency baselines and user satisfaction thresholds for the Multimodal RAG Assistant
metadata: 
  node_type: memory
  type: project
  originSessionId: d3f592ae-1195-4b56-adfd-3d42d126daef
---

Startup latency improved from ~25s to ~7s after device manager (CPU/CUDA auto-detection) was added to uvicorn/application startup.

Query response latency is currently <60s end-to-end on CPU with local GGUF Mistral-7B-Instruct-v0.2 Q4_K_M. User considers this acceptable.

**Why:** The device manager change eliminated unnecessary CUDA probing at startup, cutting init time by ~72%.

**How to apply:** User is not chasing further latency gains at this time — don't over-engineer inference speed unless they ask. If suggesting perf work, focus on retrieval or memory layers rather than inference.
