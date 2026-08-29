# Evaluation Report — AUDIO Modality

**Source document:** `FOMC Press Conference September 18, 2024.mp3` (Chair Powell's post-meeting press conference — Whisper-large-v3 transcription + pyannote diarization; the September 2024 meeting that cut the funds rate by 50 bps to 4.75–5.00%)
**Run date:** 2026-07-22 (meeting-scoped retrieval + finer re-chunking + wide cross-encoder — retrieval MRR 0.19→0.40, generation 0.196→0.375) · **Judge:** Prometheus-2-7B (Q8_0) · **Gold set:** v1.0.0 (audio chunk-ids re-mapped to the finer chunks)
**Eval user:** testuser@ragdev.local · **Command:** `python -m app.eval.run --suite {retrieval,generation,behavioral} --modality audio`

> **Audio is the weakest modality, and this run only partially improved it.** The report is deliberately honest about what was fixed, what remains, and why.

---

## The defining problem — two FOMC press conferences in one KB

The KB contains **both** FOMC transcripts used by the eval: this September 2024 **audio** (the audio-gold source) and `fomc_dec2024.txt` (the December 2024 **text**, the txt-gold source). They are near-identical in topic (rate cut, PCE, unemployment, SEP projections) but differ in every figure. With 34 December text chunks vs. 28 September audio chunks, **December out-competes September in retrieval**, so the model answered September questions with December's numbers:

| Query (September) | Reference | Before fix (December leak) |
|---|---|---|
| Rate cut + new range | 50 bps → 4.75–5.00% | "1/4 point → 4.25–4.50%" (December) |
| Headline/core PCE | 2.2% / 2.7% (ending Aug) | "2.5% … ending November" (December) |
| Median SEP funds rate | 4.4% (2024) / 3.4% (2025) | "3.4% (2025) / 3.4% (2026)" (December) |

finance_fidelity was 1.0 because the answers were internally consistent with the *retrieved* (wrong) transcript.

**The fix — meeting-scoped retrieval** (`_query_meeting_month_tokens` + scoped retrieval in query_pipeline). When the query is about a **dated meeting** (a full month + year **and** FOMC/Powell/meeting/press-conference/rate-cut context), the **primary** retrieval is scoped to the source whose filename encodes that month (the retriever's `sources` filter is a substring match, so "september" hits the mp3, "december" hits `fomc_dec2024.txt`). This **eliminates** the competing meeting from the pool entirely — not just demotes it — and, because the pool is now a single 28-chunk transcript, the correct chunk reranks up (a deep top-K of 24 is retrieved so a buried fact still has a chance). A `_meeting_source_disambiguation` demote pass backstops it after reranking.

Guards that make it safe:
- **Meeting-context gate.** Requires an event word (meeting / press conference / FOMC / Powell / committee / rate cut …), so `apple_10k.pdf`'s "Ireland State Aid Decision **as of September 28, 2024**" and the docx balance-sheet "**as of September 28, 2024**" queries — which merely *cite* a date — do NOT scope to the September FOMC audio. Verified: pdf-0013 still answers "$10.2 billion State Aid" from the 10-K; 0 pdf/docx queries scope.
- **Full month names only**, so "SEP participants" (Summary of Economic Projections) never reads as September.
- **Auto-fallback to unscoped** when the scope matches < 3 chunks (e.g. "ending August 2024" — no August source; "November 2024 FOMC meeting" in a refusal row — no November source → correctly abstains).
- **Bidirectional**: a December (txt) query scopes to `fomc_dec2024.txt`; only month-dated filenames are ever scoped.

---

## 1. Retrieval (n = 14) — fusion component — IMPROVED by finer chunking

| Metric | Coarse (75/225, 72 chunks) | **Finer (45/130, ~102 chunks)** |
|---|---|---|
| recall@5 / recall@10 | 0.429 / 0.500 | 0.429 / **0.643** |
| **hit_rate** | 0.786 | **0.857** |
| **MRR** | 0.189 | **0.403** |
| nDCG@10 | 0.253 | **0.448** |

**Read:** the finer re-chunking (below) made retrieval markedly more precise — **MRR more than doubled** (0.19 → 0.40) and nDCG nearly doubled — because a specific fact now lands in a focused ~90-word chunk that ranks near the top, instead of being diluted inside a ~1.5-minute block. hit_rate 0.79 → 0.86. This is the clearest single-number win of the re-chunking effort.

---

## Finer re-chunking + wide cross-encoder (the requested "make it strongest" pass)

Two audio-only changes on top of meeting-scoped retrieval:
1. **Finer chunking (audio only).** `_assemble_chunks` (in `av_shared.py`, shared with video) was parameterized with `min_words`/`max_words` — **defaults unchanged (75/225), so VIDEO is byte-for-byte identical**. The audio chunker passes `AUDIO_CHUNK_MIN/MAX_WORDS = 45/130`, re-transcribed + re-ingested the mp3 (72 → ~102 chunks). A specific fact ("inflation eased from a peak of 7 percent to 2.2 percent", the balance-sheet-runoff answer, "116,000" payrolls) now lands in a focused chunk instead of a generic-heavy block. *Too fine (20/50, 201 chunks) over-fragmented and regressed generation to 0.30 — 45/130 was the sweet spot.*
2. **Wide cross-encoder for scoped queries.** The dense embedding of a fact chunk that opens with an unrelated sentence ("The labor market has cooled…") ranks below the fusion cap, so the reranker never saw it. Added a per-call `max_inputs` to `Reranker.rerank` (default unchanged for every other caller); a single-meeting scoped query now hands the cross-encoder the FULL scoped candidate set (~50), and the cross-encoder — which reads each chunk's text — pins the right one. This is what turned the inflation / balance-sheet / payroll queries from abstain → correct.

The BM25 index (149 xlsx docs from other work) was preserved through the re-ingest (the fresh-process add_documents overwrite bug was avoided by pre-loading the shared index); audio was newly **added** to BM25 (149 → 350 docs).

## 2. Generation (n = 14)

| Metric | Baseline | Demote-only | Meeting-scoped | **+ finer chunk + wide XC (shipped)** |
|---|---|---|---|---|
| **answer_correctness** | 0.196 | 0.232 | 0.375 | **0.375** |
| **answer_relevancy** | 0.179 | 0.375 | 0.375 | **0.375** |
| **faithfulness** | 0.482* | 0.339 | 0.268* | **0.429** |
| context_recall | 0.770 | 0.770 | 0.770 | 0.708 |
| citation_accuracy | 1.000 | 1.000 | 1.000 | 1.000 |
| finance_fidelity | 1.000* | 0.982 | 0.798* | 0.982 |
| hallucination_rate | 0.500* | 0.571 | 0.714* | 0.429 |

**Generation correctness held at 0.375** while **faithfulness rose (0.27 → 0.43)** and hallucination fell (0.71 → 0.43). The re-chunking's gain shows up in **retrieval** (§1: MRR 0.19 → 0.40) and in specific buried-fact queries now answering correctly (inflation 7%→2.2%, balance-sheet runoff, payrolls), but the *aggregate* correctness is now capped **downstream in generation**, not retrieval — see below.

*The baseline's high faithfulness/fidelity are **misleading**: they were "faithful" to the *wrong* (December) context and, because more queries abstained, had fewer numbers to flag. As more queries now attempt real answers, the numeric heuristics see more (sometimes computed/derived) figures — cross-check **answer_correctness 0.375**, which is the trustworthy signal. Correct-meeting queries — rate cut (50 bps → 4.75–5.00%), median SEP funds rate (4.4%/3.4%), SEP PCE (2.3%/2.1%), "not behind" Q&A — now answer correctly.

**Scoping (source-filtered primary retrieval) beat demote-only** because demote left the December chunks in the pool to backstop, so they still leaked; scoping removes them entirely and lets the September pool's right chunk rerank up. answer_correctness **0.196 → 0.375** (+91%), answer_relevancy **0.179 → 0.375** (+109%).

**Remaining bottlenecks (now DOWNSTREAM of retrieval — a generation-model ceiling):**
1. **Wrong-aspect extraction.** The retrieved context now contains the right facts (retrieval MRR 0.40), but the 14B still picks a nearby-but-wrong figure — the unemployment query returns the *projected* 4.4% instead of the *current* 4.2%; the SEP-vs-current distinction and multi-part tallies ("17 of 19 wrote 3+ cuts") are error-prone. This is a generation/prompting problem, not retrieval.
2. **"August" / undated queries.** `audio-0002` names "August 2024" (the data month, not the meeting) so it can't scope to the September source and falls back to the December text; 3 other rows ("19 SEP participants…", "GDP first half of 2024", "vacancies ratio") name no meeting date at all.
3. **Prometheus variance.** At temperature > 0 the per-run correctness swings ±~0.05; 0.375 is the stable centre across runs.

**Approaches tried and reverted** (measured *worse*): drop-conflicting-entirely (→ 0.214); additive supplemental retrieval + pre-rerank disambiguation (→ 0.196); a coarse-chunk widened context (regressed 0003); and **over-fine 20/50 chunking (201 chunks) → 0.304** — fragmentation cut context_recall and hurt the multi-sentence facts. 45/130 was the sweet spot.

---

## 3. Behavioral (n = 5)

| Metric | Value | n |
|---|---|---|
| refusal_accuracy | 0.083 | 3 |
| adversarial_pass | 0.000 | 2 |

Noisy at n=2–3, consistent with every modality (swings 0.08–0.17 run to run). The two adversarial rows (both name "September 18, 2024") answer from the correct meeting, but Prometheus still scores the adversarial rubric 0 (judge miscalibration on adversarial, noted across modalities).

---

## Verdict — AUDIO

**Substantially improved across the pipeline.** Three audio-only changes, layered:
1. **Meeting-scoped retrieval** — fixed the September-vs-December contamination (**generation 0.196 → 0.375**), guarded so it never mis-scopes the Apple 10-K/docx date-citing queries and protects the TXT/December modality bidirectionally.
2. **Finer re-chunking (45/130, audio-only — video untouched)** — **retrieval MRR 0.19 → 0.40, nDCG 0.25 → 0.45, hit_rate 0.79 → 0.86, recall@10 0.50 → 0.64**. Facts now sit in focused, precisely-retrievable chunks.
3. **Wide cross-encoder for scoped queries** — surfaced the buried opening-remarks facts (inflation 7%→2.2%, balance-sheet runoff, payrolls) that dense embeddings alone ranked below the cap; lifted **faithfulness 0.27 → 0.43** and cut hallucination 0.71 → 0.43.

**Retrieval quality is now solid** (hit_rate 0.86, MRR 0.40). Generation **answer_correctness holds at 0.375** — the ceiling has moved *downstream* into the 14B's **wrong-aspect extraction** (it now has the right context but sometimes states the projected rather than the current figure), plus the "August"/undated queries that carry no meeting date to scope on. Those are generation/prompting and gold-phrasing limits, not retrieval — a good place for the ceiling to sit.

**Wins banked this pass (global):** meeting-scoped retrieval + meeting-date disambiguation (helps AUDIO, protects TXT/December, and is gated off for every non-meeting modality).

**No regression (verified).** The meeting-scope change is shared, so it was re-checked against the other modalities: **TXT** (December FOMC, whose 9 dated queries now scope to `fomc_dec2024.txt`) held at answer_correctness **0.482** / finance_fidelity **1.0** (vs 0.446 before — a slight gain, not a regression); **PDF/DOCX/XLSX/IMAGE** don't trigger scoping at all (the meeting-context gate + full-month rule exclude them — e.g. pdf-0013's "as of September 28, 2024" still answers "$10.2B State Aid" from the 10-K, not the FOMC audio).
