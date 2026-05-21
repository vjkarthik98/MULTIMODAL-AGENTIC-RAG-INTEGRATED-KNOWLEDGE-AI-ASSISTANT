---
name: project-corruption-policy
description: "Two-tier corruption policy for the Multimodal RAG Assistant — strict preflight gate at ingestion, soft quality signal at retrieval. Strict layer is shipped; soft layer is a Phase 26 follow-up."
metadata: 
  node_type: memory
  type: project
  originSessionId: 6f50bee0-7e22-4fb5-95bb-d51c8c019589
---

The Multimodal RAG Assistant uses (or will use) a two-tier corruption-handling architecture.

**Tier 1 — Strict preflight gate (SHIPPED, 2026-05-21).**
Implemented in `app/pipeline/ingestion_pipeline.py::_scan_corruption()`. Scans text-like files (`.txt .md .markdown .csv .json .log`) and hard-rejects with `CorruptFileError → HTTP 422 CORRUPTED_FILE` on any of: `null_bytes`, `ansi_escape_sequences`, `invalid_utf8`, `binary_tail` (≥10% non-printable in last 64 bytes), `replacement_char_ratio` ≥0.5%, `control_char_ratio` ≥1%. Wired into both `/rag/ingest` and `/rag/upload` routes. Verified end-to-end against `broken_document.txt` benchmark file (3 reasons fired, HTTP 422 in 34ms, no corpus pollution).

**Tier 2 — Per-chunk quality signal (NOT YET BUILT — Phase 26 follow-up).**
Stamp each chunk with a `quality_score` (alphanumeric ratio, recognizable-sentence heuristic, OCR confidence if applicable) into chunk metadata at chunk time. Retriever gains a knob to down-rank low-quality chunks at query time. Handles the legitimate "soft damage that passed the strict gate" case (mojibake, U+FFFD below threshold, scrambled values inside otherwise-valid text) without weakening Tier 1.

**Why this split:**
Tier 1 prevents corpus poisoning — once a corrupted chunk is embedded, it competes in similarity search forever and there is no clean way to tell at query time which retrieved chunk came from a corrupt source. Tier 2 lets the system still answer "what *is* recoverable?" on lightly-damaged content without inventing what isn't. Industry-standard pattern (Bedrock KB, Azure AI Search, Elastic ingest pipelines all do hard-gate at ingestion; OCR/web-scrape pipelines layer the soft signal on top).

**How to apply:**
- Do NOT weaken `_scan_corruption()` to allow nominally-text files with null bytes / invalid UTF-8 through. The strict gate is load-bearing for the Group A hallucination tests (Q3/Q6/Q7/Q10 in `broken_benchmark_queries.txt`).
- When building Phase 26 ([[project-perf-baselines]] for current state), add `quality_score` to the chunk schema in `app/ingestion/schema.py`, populate in `app/chunking/chunker.py`, expose as a filter/rerank input in `app/retrieval/`.
- Benchmark fixtures live at `data/benchmarks/` — `broken_document.txt` (hard-rejected, tests strict gate) and `broken_document_soft.txt` (passes gate, tests graceful degradation Q2/Q4/Q5/Q8/Q9 + hallucination resistance Q3/Q6/Q7/Q10 at query time).

**Phase 26 disclosure gap found during Q5 benchmark (2026-05-21):**
`app/ingestion/text_repair.py::repair_mojibake()` runs `ftfy.fix_text()` silently when `TEXT_REPAIR_MOJIBAKE=True`. Verified by Q5: `ReykjavÃ­k` was auto-repaired to `Reykjavík` before embedding; `KowaÅski` was preserved (ftfy couldn't confidently fix it). Asymmetric repair is correct behavior, BUT the response payload does not disclose that a repair happened. Industry-grade fix: stamp `repaired_mojibake_count` (and similar) onto chunk metadata, surface as `provenance.repairs_applied` in the query response, optionally set `hallucination_warning: true` with reason "source contained mojibake that was auto-repaired" so the user knows the answer is grounded in *repaired* source text, not raw source text. Bundle with Tier 2 quality_score work.

**Citation-integrity gap found during Q9 benchmark (2026-05-21):**
LLM fabricated a citation `[b62c7383cfee4c0585abbeb5a4b53ed9_valid_document.txt]` in the answer text while the actual retrieved chunk came from `2590c0ec4cfa4d1b83448fbd3ee3de72_broken_document_soft.txt`. The file `valid_document.txt` does not exist anywhere in the project, index, or logs — fully hallucinated. Answer *content* was correct and grounded; only the citation string was invented. Root cause is likely prompt-template leakage (the LLM sees example citation formats during pretraining and fills them in) or insufficient post-generation citation validation. Industry-grade fix: validate every `[filename]` token in the LLM output against the `sources` array before returning; replace mismatched citations with the actual retrieved source or strip them. This is a Phase 26 guardrail — output-side citation integrity check. Critical for compliance use cases (legal, medical, regulated finance) where a fabricated citation is worse than no citation.
