# Master Prompt — Golden Evaluation Dataset Generation (MAGIK)

Companion to [README.md](README.md) and `build_gold_set.py`. This file is the prompt you
hand to an LLM to turn a *real* source document + its *real* ingested chunk manifest into
**candidate** rows for `datasets/gold/{modality}_gold.jsonl`. It does not replace human
review — MAGIK's gold sets are real-world-only (`README.md:3`), so every row this prompt
produces is a draft that a human must verify against the source document before the `TODO`
marker is removed (`build_gold_set.py` docstring, step 2-4 of the README workflow).

Use it to fill `build_gold_set.py`'s scaffolded `"TODO_ingest_then_fill"` /
`"TODO"` rows faster, and to generate additional candidates beyond the static
`QUESTION_TEMPLATES` — never to invent facts about a document the model hasn't been shown.

### Target composition per file

Each of the 7 modality gold files (`text`, `pdf`, `docx`, `xlsx`, `image`, `audio`,
`video`) targets **15 rows: 14 `expected_route: "rag"` queries + 1
`expected_route: "search"` (`websearch-probe`) query**. 7 files × 15 rows = **105 queries
total** for this pass. This supersedes the "40 triples" figure in `README.md`'s file-naming
table for the current generation pass — treat 15/file as the working target until told
otherwise. `routing_gold.jsonl` and `e2e_gold.jsonl` are separate files, not counted in the
105, and are not in scope for this composition rule.

---

## 1. How to use this file

1. Ingest the real source document through the normal pipeline (or `--ingest` in
   `build_gold_set.py`) so it has a real `doc_id` and real `chunk_N` ids in Qdrant/BM25.
2. Pull the ingested chunk manifest for that document: `chunk_id`, `text`, and whatever
   locator metadata that modality writes (see §3 table) — the same fields
   `_build_sources_array` (`app/pipeline/query_pipeline.py:455-580`) reads at query time.
3. Paste the **Prompt block** (§2) into an LLM session, followed by:
   - the modality,
   - the source document (or the excerpt/table/frame set you want questions from),
   - the chunk manifest from step 2.
4. The LLM returns JSONL rows in the exact schema of §4. Every `relevant_chunk_ids` value
   MUST be a `chunk_id` that actually exists in the manifest you supplied — the LLM must
   never fabricate one.
5. A human opens the source document, verifies each row (§7 checklist), fixes anything
   wrong, and removes any remaining `TODO`. Never fabricate ground truth
   (`build_gold_set.py` docstring).
6. Run `python -m app.eval.datasets.build_gold_set --validate` to check schema + refresh
   `manifest.yaml`'s sha256/`triples_curated` counts.

---

## 2. Prompt block (paste this to the LLM)

```
You are generating CANDIDATE rows for a production RAG evaluation gold set. You will be
given: (a) a modality, (b) a source document/excerpt, and (c) the REAL list of ingested
chunks for that document (chunk_id + text + locator metadata). Every fact in every
reference_answer must be traceable, verbatim or near-verbatim, to the chunk text you were
given. You are not allowed to use outside knowledge, round numbers, or "reasonable"
guesses. If the document does not contain enough information to answer a question you
propose, do not include that question — propose a different one, or mark it as a negative
test case per the rules below.

Hard rules:
1. GROUNDING ONLY. Never cite a chunk_id that is not in the supplied manifest. Never state
   a number, date, or name that does not appear verbatim in the cited chunk(s) (tolerance:
   the number may differ from the source by scale-formatting only, e.g. "$1.3 billion" vs
   "1,300" in a table cell — never by value).
2. NO SYNTHETIC FACTS. If you are not certain a claim is grounded, omit it rather than
   guess. It is correct behavior to produce fewer, fully-grounded rows.
3. CITATIONS MUST MATCH THE MODALITY'S LOCATOR FIELDS (see the per-modality table below).
   A pdf row without a defensible page number, an xlsx row without a sheet/row locator, or
   an audio/video row without a timestamp is incomplete — flag it TODO rather than omit
   the locator silently.
4. EVERY ROW IS A CANDIDATE. Prefix reference_answer with nothing extra — write it as a
   clean, directly-usable answer — but leave "added_by": "llm_candidate" (a human reviewer
   will change this to "human" after verification, matching the project's provenance
   convention: added_by is always "human" once a row is trusted).
5. Cover a MIX of question types across the RAG rows, not just fact lookup: fact-extraction,
   comparative-analysis, multi-hop (only if you were given chunks from >1 document),
   table-extraction (xlsx), chart-understanding + ocr (image), audio-transcription /
   financial-reasoning (audio), video-understanding / frame-extraction (video).
6. PRODUCE EXACTLY 15 ROWS PER MODALITY FILE: 14 rows with expected_route: "rag" (fully
   grounded per rules 1-4 above) + exactly 1 row with expected_route: "search" (a
   websearch-probe negative case per §6.1 — real-time/out-of-KB question,
   relevant_chunk_ids: [], reference_answer prefixed "SEARCH_REQUIRED: ..."). Do not
   produce more or fewer than this split.
7. Output STRICT JSONL, one JSON object per line, schema exactly as specified in §4. No
   markdown fences, no commentary, no trailing text.
```

---

## 3. Per-modality citation rules

Every `relevant_chunk_ids` entry must resolve to a chunk whose Qdrant payload carries the
locator fields below — these are exactly what `_build_sources_array` (
`app/pipeline/query_pipeline.py:528-580`) surfaces to the UI's source chips at query time,
so a gold row's citation is only valid if it would produce the same chip a real user sees.

| Modality | Chunk id format | Required locator(s) in metadata | Gold-row extra field | UI chip fields it feeds |
|---|---|---|---|---|
| `txt`   | `{source}::chunk_{chunk_id}` | none beyond chunk text | — | `source` |
| `pdf`   | `{source}::chunk_{chunk_id}` | `page_number` | — | `source`, `page_number` |
| `docx`  | `{source}::chunk_{chunk_id}` | `heading` / `section_title` | — | `source`, `heading` |
| `xlsx`  | `{source}::chunk_{chunk_id}` | `sheet_name` + `row_range` (or `row_start`/`row_end`) | — | `source`, `sheet_name`, `row_range` |
| `image` | `{source}::chunk_{chunk_id}` | `image_title` | `gold_ocr_text` (verbatim OCR string the answer relies on) | `source`, `image_title` |
| `audio` | `{source}::chunk_{chunk_id}` | `start_time`/`timestamp_start`, `end_time`, `speaker_name`, `speaker_role` | `gold_transcript_excerpt` (verbatim quote spanning the cited time range) | `source`, `timestamp_start`, `speaker_name`, `speaker_role` |
| `video` | `{source}::chunk_{chunk_id}` | same as audio, plus frame-level captions | `gold_transcript_excerpt` + `gold_frame_captions` (list, verbatim on-screen text) | same as audio, plus caption overlay |

`{source}` is the bare ingested filename exactly as stored in the Qdrant payload's
`source` field (e.g. `"apple_10k.pdf"` — no hash prefix, no path), and `{chunk_id}` is the
payload's integer `chunk_id`. This is the composite ID the live harness actually builds and
matches against — see `app/eval/runners/retrieval_runner.py:87-93` (comment: *"format:
"{source}::chunk_{chunk_id}" to match the gold set format"*) and the identical construction
in `app/eval/runners/e2e_runner.py:45-83`. **Do not use a `{doc_id}_{filename}` hash-prefixed
form** — that pattern appears in a few older rows already in the gold files but does not
match what `retrieval_runner.py`/`e2e_runner.py` build today, so it silently scores those
rows as zero recall. Always pull `chunk_id` from the real ingested manifest (Qdrant payload
or the `IngestedDocument.chunk_id` attribute in a BM25 index) — never invent a sequence
number.

Rules that follow directly from this table:

- **Never invent a `chunk_id`.** It must be the literal integer from the real ingested
  manifest for that `source_file` (e.g. `"apple_10k.pdf::chunk_859"`).
- **pdf** — if the manifest chunk has no `page_number` (extraction gap), do not claim one;
  mark the row `TODO` instead of guessing a page.
- **xlsx** — `reference_answer` must match the cell values under the cited `sheet_name`
  and `row_range` exactly (numbers, units, and time period), not a re-derived figure.
- **image** — `gold_ocr_text` is the OCR ground truth checked by `ocr_cer`/`ocr_wer`
  (`thresholds.yaml:81-87`); it must be copied from the chunk's actual OCR text, not
  paraphrased.
- **audio/video** — `gold_transcript_excerpt` is checked against Whisper WER
  (`thresholds.yaml:89-92`, `audio_wer` max 0.25); it must be an exact substring of the
  supplied transcript, not a summary. `gold_frame_captions` (video only) feeds
  `frame_caption_recall`/`caption_repetition_rate` (`thresholds.yaml:94-100`) and must be
  the literal on-screen text (e.g. `"APPLE EPS BEAT $1.64 ADJ. VS. $1.60 EST."`), not a
  description of the frame.
- **speaker_role**, when present, must be one of the roles the UI colors —
  ceo/cfo/analyst/operator (`ui/src/components/MediaTimestampChip.jsx`) — never invented
  free text.

---

## 4. Output schema (exact — matches `datasets/gold/*.jsonl` today)

Common fields, every row:

```json
{
  "id": "{modality}-NNNN",
  "modality": "txt|pdf|docx|xlsx|image|audio|video|routing|e2e",
  "source_file": "exact_ingested_filename.ext",
  "query": "natural-language question a user would actually ask",
  "relevant_chunk_ids": ["{source}::chunk_{chunk_id}", "..."],
  "reference_answer": "grounded, verbatim-traceable answer",
  "expected_route": "rag|search|memory|direct|hybrid",
  "tags": ["question-type-tag", "modality-tag", "..."],
  "added_by": "llm_candidate",
  "added_at": "YYYY-MM-DD"
}
```

Modality-specific additions (append to the common object, per §3):

```json
// image
"gold_ocr_text": "verbatim OCR string"

// audio
"gold_transcript_excerpt": "verbatim transcript quote"

// video
"gold_transcript_excerpt": "verbatim transcript quote",
"gold_frame_captions": ["VERBATIM ON-SCREEN TEXT 1", "VERBATIM ON-SCREEN TEXT 2"]
```

`routing` and `e2e` rows use the common schema only (`source_file: null`,
`relevant_chunk_ids: []` for pure routing probes; `e2e` rows may cite chunks across
multiple `source_file`s for multi-hop questions — see the worked example in §8).

---

## 5. Golden question-type taxonomy (use these as `tags`)

| Tag | Meaning | Typical modality |
|---|---|---|
| `fact-extraction` | single-number/single-fact lookup | all |
| `comparative-analysis` | compares two entities/periods | txt, e2e |
| `multi-hop` / `cross-document` | needs chunks from ≥2 source files | e2e |
| `table-extraction` | reads a specific cell/row/column | xlsx |
| `chart-understanding` | reads a chart's plotted values | image |
| `ocr` | answer depends on OCR'd text/numbers | image |
| `audio-transcription` | quotes or paraphrases spoken content | audio |
| `financial-reasoning` | requires combining 2+ disclosed figures | audio, txt |
| `video-understanding` | depends on visual + audio track together | video |
| `frame-extraction` | answer is a specific on-screen caption | video |
| `websearch-probe` | intentionally NOT answerable from the KB (§6) | any |
| `routing-eval` | tests router classification only, no retrieval | routing |

---

## 6. Negative / broken-document / unanswerable rules

These protect `route_accuracy` (`thresholds.yaml:105-111`) and the refusal path from
silent regressions. **For the current 105-query pass (§ Target composition), rule 1 below
is the only mandatory negative case** — exactly 1 per modality file, per rule 6 of the
prompt block (§2). Rules 2-5 are additional techniques available for future/larger
batches; do not use them to replace the 14 RAG rows in this pass.

1. **Real-time / out-of-KB fact** → `expected_route: "search"`, `relevant_chunk_ids: []`,
   `reference_answer` prefixed `"SEARCH_REQUIRED: ..."` — mirrors the existing convention
   in `docx_gold.jsonl` (`docx-websearch-001`). Tag `websearch-probe`.
2. **Unanswerable-from-corpus** (document doesn't contain the fact at all) →
   `expected_route: "rag"`, `relevant_chunk_ids: []`, `reference_answer` states the
   expected refusal, e.g. `"REFUSAL_REQUIRED: the document does not disclose {X}; a
   correct answer must say so rather than guess."` This is what the hallucination/
   groundedness suite regression-tests against (`GroundednessChecker`,
   `verification_schema.py:103-109`).
3. **Corrupted/broken document** → do not synthesize corruption yourself; if the source
   corpus has a known-broken file (see project memory: `apple_10k.pdf` was previously a
   corrupt data file), the expected answer documents graceful degradation, not a crash —
   `reference_answer`: `"INGESTION_ERROR_EXPECTED: pipeline must surface a readable error,
   not fabricate content."`
4. **Ambiguous query** (plausible under >1 interpretation) → still require a real
   `reference_answer`, but tag `ambiguous`; used to check the model asks for
   clarification or answers the dominant interpretation rather than hallucinating a
   merged answer.
5. **Routing-only probes** (no retrieval expected) → `modality: "routing"`,
   `relevant_chunk_ids: []`, `reference_answer` equals the expected route string itself
   (matches `routing_gold.jsonl`'s `route-0001` pattern). Include at least one `hybrid`
   case to exercise `hybrid_with_web_rate` (`thresholds.yaml:109-111`).

---

## 7. Grounding & hallucination-detection checklist (apply before removing `TODO`)

Mirrors what `GroundednessChecker` / `CitationVerifier` check automatically at query time
(`verification_schema.py:103-118`) — a gold row that wouldn't pass its own citation check
is not a valid gold row.

- [ ] Every `relevant_chunk_ids` entry exists in the real ingested manifest for
      `source_file` (not guessed, not from a different document).
- [ ] Every number, date, and name in `reference_answer` appears verbatim (or
      scale-equivalent, per `finance.numeric_fidelity` 0.5% tolerance,
      `thresholds.yaml:76-79`) inside the cited chunk text — this is the
      `unsupported_numbers` check.
- [ ] No claim in `reference_answer` requires information from a chunk that is NOT listed
      in `relevant_chunk_ids` — this is the `unsupported_claims` check.
- [ ] The modality-specific locator (`page_number`/`heading`/`sheet_name`+`row_range`/
      `image_title`/`timestamp_start`+`speaker_name`) is present and matches the cited
      chunk's real metadata, not inferred.
- [ ] `gold_ocr_text` / `gold_transcript_excerpt` / `gold_frame_captions` (where
      applicable) are copied verbatim from the source, not paraphrased.
- [ ] `expected_route` matches how MAGIK would actually classify this query today
      (`rag` for KB-answerable, `search` for real-time, `direct` for chit-chat/greetings,
      `hybrid` only for genuinely mixed queries).
- [ ] `added_by` changed from `"llm_candidate"` to `"human"` only after a human has opened
      the source document and independently confirmed every fact above.

---

## 8. Worked examples (one per modality, template style)

```json
{"id": "pdf-00NN", "modality": "pdf", "source_file": "{real_filename}.pdf", "query": "{question grounded in the supplied chunks}", "relevant_chunk_ids": ["{real_filename}.pdf::chunk_{real_chunk_id}"], "reference_answer": "{answer with every number/name traceable to the cited chunk, includes the page it came from implicitly via page_number}", "expected_route": "rag", "tags": ["fact-extraction", "pdf-extraction"], "added_by": "llm_candidate", "added_at": "2026-07-16"}
{"id": "xlsx-00NN", "modality": "xlsx", "source_file": "{real_filename}.xlsx", "query": "{question about a specific cell/row}", "relevant_chunk_ids": ["{real_filename}.xlsx::chunk_{real_chunk_id}"], "reference_answer": "{value(s) exactly as they appear under the cited sheet_name/row_range}", "expected_route": "rag", "tags": ["table-extraction", "xlsx-extraction"], "added_by": "llm_candidate", "added_at": "2026-07-16"}
{"id": "img-00NN", "modality": "image", "source_file": "{real_filename}.jpg", "query": "{question about the chart/scan}", "relevant_chunk_ids": ["{real_filename}.jpg::chunk_{real_chunk_id}"], "reference_answer": "{figures exactly as plotted/printed}", "expected_route": "rag", "tags": ["chart-understanding", "ocr"], "added_by": "llm_candidate", "added_at": "2026-07-16", "gold_ocr_text": "{verbatim OCR string}"}
{"id": "audio-00NN", "modality": "audio", "source_file": "{real_filename}.mp3", "query": "{question about what was said}", "relevant_chunk_ids": ["{real_filename}.mp3::chunk_{real_chunk_id}"], "reference_answer": "{claim traceable to the transcript excerpt}", "expected_route": "rag", "tags": ["audio-transcription", "financial-reasoning"], "added_by": "llm_candidate", "added_at": "2026-07-16", "gold_transcript_excerpt": "{verbatim quote}"}
{"id": "video-00NN", "modality": "video", "source_file": "{real_filename}.mp4", "query": "{question about on-screen figures}", "relevant_chunk_ids": ["{real_filename}.mp4::chunk_{real_chunk_id}"], "reference_answer": "{figures traceable to caption/transcript}", "expected_route": "rag", "tags": ["video-understanding", "frame-extraction"], "added_by": "llm_candidate", "added_at": "2026-07-16", "gold_transcript_excerpt": "{verbatim quote}", "gold_frame_captions": ["{verbatim on-screen text}"]}
{"id": "docx-websearch-00N", "modality": "docx", "source_file": null, "query": "{real-time question outside the static KB}", "relevant_chunk_ids": [], "reference_answer": "SEARCH_REQUIRED: {what a correct answer must contain}", "expected_route": "search", "tags": ["websearch-probe", "real-time"], "added_by": "llm_candidate", "added_at": "2026-07-16"}
{"id": "route-00NN", "modality": "routing", "source_file": null, "query": "{query whose classification is being tested}", "relevant_chunk_ids": [], "reference_answer": "{rag|search|memory|direct|hybrid}", "expected_route": "{same value}", "tags": ["routing-eval"], "added_by": "llm_candidate", "added_at": "2026-07-16"}
```

---

## 9. RAGAS / DeepEval / judge compatibility

These gold fields are consumed directly by the metrics already wired in `thresholds.yaml`
— do not invent new field names:

- `relevant_chunk_ids` → `recall@5`, `recall@10`, `mrr`, `ndcg@10`, `context_precision`,
  `hit_rate` (`metrics/retrieval.py`, `thresholds.yaml:19-37`; Ragas-equivalent: context
  precision/recall).
- `reference_answer` vs. the system's `{query, answer, contexts, retrieved_docs}` →
  `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`,
  `citation_accuracy`, `template_leak_rate` (`metrics/generation.py`,
  `thresholds.yaml:51-69`).
- Judged by whichever of `judges/{gguf_judge, crossencoder_judge, lexical_judge,
  phi3_judge}.py` the suite selects — write `reference_answer` as a clean prose answer a
  judge model can NLI-compare against the generated answer, not as bullet fragments.
- `gold_ocr_text` / `gold_transcript_excerpt` / `gold_frame_captions` →
  `ocr.ocr_cer/ocr_wer`, `audio.audio_wer`, `video.frame_caption_recall` /
  `caption_repetition_rate` (`thresholds.yaml:81-100`).
- `expected_route` → `routing.route_accuracy`, `routing.hybrid_with_web_rate`
  (`thresholds.yaml:105-111`).
- Citation locator fields (§3) → `verification.citation_accuracy_v2` /
  `grounding_success_rate` once the Phase 32 verification loop gets its first baseline
  (`thresholds.yaml:149-161`, `verification_schema.py:112-118`).

---

## 10. LLM-as-judge instructions (for whoever runs the judge, not the generator)

When a judge model scores a generated answer against one of these gold rows:

1. Compare the generated answer's claims to `reference_answer`, not to the raw source
   document — `reference_answer` is the grounded contract.
2. A generated answer is **faithful** only if every number/name it states also appears in
   `reference_answer` or in the chunk text under `relevant_chunk_ids`.
3. A generated answer's **citations** are correct only if the locator it displays (page /
   sheet+row / timestamp+speaker / image title) matches the one recorded for that
   `relevant_chunk_ids` entry in §3 — a citation to the right file but wrong page/sheet/
   timestamp must be scored as a bad citation, not a partial credit.
4. For `SEARCH_REQUIRED` / `REFUSAL_REQUIRED` rows, the ONLY correct generated answer is
   one that takes that action (routes to search / refuses) — any fabricated on-topic
   answer is an automatic fail regardless of surface plausibility.

---

## 11. Web search evaluation

- Use the `websearch-probe` tag (§5/§6.1) for any query whose answer is time-sensitive
  and therefore cannot live in a static gold set (stock price today, latest news,
  post-cutoff events).
- `reference_answer` for these rows never states the real-time fact (it would go stale);
  it states the requirement: `"SEARCH_REQUIRED: ..."` plus what a correct answer must
  cover, exactly like `docx_gold.jsonl`'s `docx-websearch-001`.
- Pair every `websearch-probe` with a `hybrid` variant where useful — a query that needs
  both a KB fact and a live fact — to keep `routing.hybrid_with_web_rate` exercised
  (`thresholds.yaml:109-111`, currently gated at `min: 0.0` pending the P1-4 fix per
  project memory).

---

## 12. Scoring rubric for candidate rows (human reviewer, before removing `TODO`)

| Score | Meaning |
|---|---|
| **Accept** | Every §7 checklist item passes; row is ready, set `added_by: "human"`. |
| **Fix locator** | Facts are correct but the citation locator is wrong/missing — correct it, don't discard. |
| **Reject — ungrounded** | A number/claim has no source in the cited chunk(s); either fix `reference_answer` to only state what's grounded, or discard the row. |
| **Reject — bad chunk_id** | `relevant_chunk_ids` references a chunk that doesn't exist for that `source_file`; re-derive from the real manifest. |
| **Keep as negative** | Row is intentionally unanswerable/out-of-KB — verify it's tagged and phrased per §6, don't "fix" it into a fabricated answer. |
