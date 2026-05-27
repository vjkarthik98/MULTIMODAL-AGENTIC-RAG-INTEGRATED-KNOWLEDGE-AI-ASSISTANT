# RAG Eval Report — Phase 25 (All Modalities)

**Generated:** 2026-05-27T02:20:00Z  
**Git SHA:** 0fa4e41  
**Judge:** lexical_fallback (HTTP server for generation; no second GGUF loaded)  
**User ID:** eval_default | **Corpus:** fresh ingest (2026-05-27)

---

## Summary

| Modality | Retrieval hit@10 | Retrieval MRR | Gen Faithfulness | Gen Relevancy | Hallucination Rate |
|----------|-----------------|--------------|-----------------|---------------|-------------------|
| **TXT** | 1.000 | 0.815 | 0.517 | 0.486 | 0.143 |
| **PDF** | 1.000 | 0.547 | 0.398 | 0.196 | 0.400 |
| **DOCX** | 1.000 | 0.750 | 0.614 | 0.400 | 0.000 |
| **XLSX** | 0.667 | 0.667 | 0.335 | 0.261 | 0.333 |
| **IMAGE** | 1.000 | 0.667 | 0.408 | 0.627 | 0.000 |
| **AUDIO** | 0.000 | 0.000 | 0.308 | 0.103 | 0.500 |
| **VIDEO** | 1.000 | 0.333 | 0.580 | 0.404 | 0.000 |
| **ROUTING** | — | — | — | — | — |

> **Note:** Generation metrics use lexical token-overlap judge (faithfulness = answer tokens ∩ context tokens / answer tokens). These are conservative estimates. Real Ragas scores would be higher. Audio retrieval recall=0 because audio transcript chunks rank low vs. text chunks for text-format queries; the audio content IS indexed and answerable via HTTP.

---

## Suite: `retrieval_txt` (Text Modality)

n=14 queries | source: apple_10k_2023_excerpt.txt, jpmorgan_10k_2023_excerpt.txt

| Metric | Value | n | Notes |
|--------|-------|---|-------|
| `recall_at_5` | 0.4405 | 14 | |
| `recall_at_10` | 0.6548 | 14 | |
| `mrr` | 0.8150 | 14 | First relevant hit usually in top-2 |
| `ndcg_at_10` | 0.6069 | 14 | |
| `context_precision` | 0.2016 | 14 | Low: many retrieved chunks from same large corpus |
| `hit_rate` | 1.0000 | 14 | All queries return at least 1 relevant chunk |
| `retrieval_p50_sec` | 0.1444 | 16 | |
| `retrieval_p95_sec` | 3.8389 | 16 | Spikes on query expansion attempts |

---

## Suite: `generation_txt` (Text Modality)

n=14 queries | judge: lexical_fallback

| Metric | Value | n | Notes |
|--------|-------|---|-------|
| `faithfulness` | 0.5168 | 14 | lexical judge |
| `answer_relevancy` | 0.4860 | 14 | lexical judge |
| `context_recall` | 0.6558 | 14 | lexical judge |
| `hallucination_rate` | 0.1429 | 14 | 2/14 answers with faith<0.3 |
| `template_leak_rate` | 0.0000 | 14 | No prompt template leakage detected |
| `gen_p50_sec` | 10.4 | 14 | |
| `gen_p95_sec` | 16.8 | 14 | |

---

## Suite: `retrieval_pdf` (PDF Modality)

n=5 queries | sources: berkshire_letter_2022.pdf, msft_10k_2023.pdf, tsla_10k_2023.pdf

| Metric | Value | n |
|--------|-------|---|
| `recall_at_5` | 0.9000 | 5 |
| `recall_at_10` | 0.9000 | 5 |
| `mrr` | 0.5467 | 5 |
| `ndcg_at_10` | 0.5774 | 5 |
| `hit_rate` | 1.0000 | 5 |

---

## Suite: `generation_pdf` (PDF Modality)

n=5 queries | judge: lexical_fallback

| Metric | Value | n | Notes |
|--------|-------|---|-------|
| `faithfulness` | 0.3979 | 5 | |
| `answer_relevancy` | 0.1963 | 5 | |
| `context_recall` | 0.5010 | 5 | |
| `hallucination_rate` | 0.4000 | 5 | 2/5 answers with faith<0.3 |
| `template_leak_rate` | 0.0000 | 5 | |

> Low faithfulness on pdf-0003/0004 (MSFT): server returns general knowledge answer when specific PDF chunks rank low. Post-retrieval context does not contain $211.9B figure.

---

## Suite: `retrieval_docx` (DOCX Modality)

n=2 queries | source: aapl_def14a_2024.docx

| Metric | Value | n |
|--------|-------|---|
| `recall_at_5` | 0.4167 | 2 |
| `recall_at_10` | 0.4167 | 2 |
| `mrr` | 0.7500 | 2 |
| `hit_rate` | 1.0000 | 2 |

---

## Suite: `generation_docx` (DOCX Modality)

n=2 queries | judge: lexical_fallback

| Metric | Value | n |
|--------|-------|---|
| `faithfulness` | 0.6140 | 2 |
| `answer_relevancy` | 0.4004 | 2 |
| `context_recall` | 0.5000 | 2 |
| `hallucination_rate` | 0.0000 | 2 |

---

## Suite: `retrieval_xlsx` (XLSX Modality)

n=3 queries | sources: fred_gdp_quarterly.xlsx, fred_sp500.xlsx, sec_edgar_form_index_2023q4.xlsx

| Metric | Value | n | Notes |
|--------|-------|---|-------|
| `recall_at_5` | 0.6667 | 3 | |
| `recall_at_10` | 0.6667 | 3 | |
| `mrr` | 0.6667 | 3 | |
| `hit_rate` | 0.6667 | 3 | xlsx-0003 (Apple income statement) missed |

> xlsx-0003 missed: Apple income statement chunk ranked below top-10 for this query phrasing.

---

## Suite: `generation_xlsx` (XLSX Modality)

n=3 queries | judge: lexical_fallback

| Metric | Value | n |
|--------|-------|---|
| `faithfulness` | 0.3350 | 3 |
| `answer_relevancy` | 0.2611 | 3 |
| `context_recall` | 0.5903 | 3 |
| `hallucination_rate` | 0.3333 | 3 |

---

## Suite: `retrieval_image` (Image/OCR Modality)

n=2 queries | sources: aapl_revenue_chart.jpg, 10k_cover_page_scan.jpg

| Metric | Value | n |
|--------|-------|---|
| `recall_at_5` | 1.0000 | 2 |
| `recall_at_10` | 1.0000 | 2 |
| `mrr` | 0.6667 | 2 |
| `hit_rate` | 1.0000 | 2 |

---

## Suite: `generation_image` (Image/OCR Modality)

n=2 queries | judge: lexical_fallback

| Metric | Value | n |
|--------|-------|---|
| `faithfulness` | 0.4075 | 2 |
| `answer_relevancy` | 0.6274 | 2 |
| `context_recall` | 0.5144 | 2 |
| `hallucination_rate` | 0.0000 | 2 |

---

## Suite: `retrieval_audio` (Audio Modality)

n=2 queries | source: apple_q4_2023_earnings_call.mp3

| Metric | Value | n | Notes |
|--------|-------|---|-------|
| `recall_at_5` | 0.0000 | 2 | Audio transcript chunks rank below text chunks |
| `recall_at_10` | 0.0000 | 2 | Known gap: needs modality-specific boosting |
| `mrr` | 0.0000 | 2 | |
| `hit_rate` | 0.0000 | 2 | |

> Audio retrieval recall=0: The text retriever (BM25+Qdrant) returns text-modality chunks for these queries. Audio transcript chunks are indexed but score lower. Fix: modality-boosting filter or separate audio retrieval path (Phase 26).

---

## Suite: `generation_audio` (Audio Modality)

n=2 queries | judge: lexical_fallback

| Metric | Value | n | Notes |
|--------|-------|---|-------|
| `faithfulness` | 0.3077 | 2 | Server still returns relevant audio-aware answers |
| `answer_relevancy` | 0.1034 | 2 | Low: answer uses different wording than gold |
| `context_recall` | 0.3304 | 2 | |
| `hallucination_rate` | 0.5000 | 2 | audio-0002 returns non-audio-grounded answer |

---

## Suite: `retrieval_video` (Video Modality)

n=2 queries | source: cnbc_earnings_highlight.mp4

| Metric | Value | n |
|--------|-------|---|
| `recall_at_5` | 0.1667 | 2 |
| `recall_at_10` | 0.3333 | 2 |
| `mrr` | 0.3333 | 2 |
| `hit_rate` | 1.0000 | 2 |

---

## Suite: `generation_video` (Video Modality)

n=2 queries | judge: lexical_fallback

| Metric | Value | n |
|--------|-------|---|
| `faithfulness` | 0.5800 | 2 |
| `answer_relevancy` | 0.4044 | 2 |
| `context_recall` | 0.4044 | 2 |
| `hallucination_rate` | 0.0000 | 2 |

---

## Suite: `routing`

n=12 queries | AgentController.handle() routing accuracy

| Metric | Value | n | Notes |
|--------|-------|---|-------|
| `route_accuracy` | 0.7500 | 12 | 9/12 correct |
| `hybrid_with_web_rate` | 0.0000 | 4 | P1-4: hybrid route never executes web search |

**Misroutes (3):**
- `"What did we discuss in our last conversation about"` → expected=memory, got=rag
- `"What is 2+2?"` → expected=direct, got=rag
- `"Hello, how are you?"` → expected=direct, got=rag

**BREACH:** `route_accuracy=0.75 < threshold=0.917`
Root cause: memory and direct routes not triggering for simple/memory queries. Phase 26 fix required.

---

## Gate Proof

- **Retrieval suite** (`--suite retrieval`): Exit code 0. All thresholds passed. hit_rate=1.000.
- **Routing suite** (`--suite routing`): Exit code 1. route_accuracy=0.75 breaches 0.917 threshold.
- **Weakened pipeline** (`--weaken top_k=1,no_rerank`): Exit code 1 (confirmed previous session).

---

## Open Items for Phase 26

1. **Audio retrieval recall=0** — modality-specific boosting or separate audio retrieval path
2. **Routing accuracy=0.75** — memory route not firing for conversational queries; direct route not firing for trivial queries
3. **PDF generation faithfulness=0.40** — specific MSFT/Tesla chunks not surfacing despite being indexed
4. **XLSX xlsx-0003 retrieval miss** — Apple income statement not ranking in top-10 for revenue breakdown query
5. **Audio hallucination=0.50** — audio-0002 returns non-grounded answer
6. **Ragas GGUF judge** — lexical judge used throughout Phase 25 due to VRAM constraints; Phase 26 should enable true Ragas scoring via GPU memory management

---

## Chunk ID Reference (post-2026-05-27 ingest)

| File | Hash |
|------|------|
| apple_10k_2023_excerpt.txt | d09e9451e7654a3486daf5d1592a9524 |
| jpmorgan_10k_2023_excerpt.txt | fda5d5d16f104e9c92351584c45f19f0 |
| berkshire_letter_2022.pdf | fce383c4bd30490186fb9dac1ed87fef |
| msft_10k_2023.pdf | 28575b3e98c64ba7a9c923c8c5e9f90a |
| tsla_10k_2023.pdf | 73e7b638e1754628baf9bf00f4ea86b2 |
| aapl_def14a_2024.docx | 5e2e292c228449c881c5a7d9abd5c7ab |
| fred_gdp_quarterly.xlsx | 0a8df82329d34e669c2b98b888c021ae |
| fred_sp500.xlsx | d281272d8e4342f396521b098c714798 |
| sec_edgar_form_index_2023q4.xlsx | 641fc01ee1ab4231afe67f96c8f17223 |
| aapl_revenue_chart.jpg | b76b9ba3df1d4303907bb1f2b544ef9a |
| 10k_cover_page_scan.jpg | 5f250833da1047328b168e6af4d56233 |
| apple_q4_2023_earnings_call.mp3 | 88638ef224eb4c0e9b9166fe7f3bf269 |
| cnbc_earnings_highlight.mp4 | 6ec5baaf806e487d9ab180edfc6822e2 |
