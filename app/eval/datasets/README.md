# Eval Datasets — Real-World Finance Corpus

All datasets use **public, real-world financial documents only**. No synthetic data.

## Modality sources

| Modality | Suggested sources | License |
|----------|------------------|---------|
| TXT | SEC EDGAR 10-K/10-Q full-text (edgar.sec.gov), FRED notes | Public domain |
| PDF | SEC 10-K/10-Q PDFs (AAPL, MSFT, JPM, TSLA), annual investor letters | Public domain |
| DOCX | Investor presentations rebuilt to DOCX from public IR PDFs | Public domain |
| XLSX | Macrotrends / SEC financial-statement datasets, public XLSX exports | CC0 / public |
| JPG | Revenue/EPS chart screenshots, scanned 10-K cover pages from public IR | Public domain |
| MP3 | Public earnings-call audio from IR sites (quarterly call segments) | Check per source |
| MP4 | CNBC public clips, IR earnings highlight videos | Check per source |

## Download

```bash
bash scripts/download_eval_corpus.sh
```

Small samples (< 1 MB each) are committed directly to `datasets/gold/` for smoke tests.
Large binaries are downloaded on first run via the script above.

## Dataset manifest

`manifest.yaml` pins the SHA-256 of each gold JSONL file. Any change to a gold file
requires updating the manifest — protects against silent edits to ground truth.

## Gold set construction workflow

1. Run `python -m app.eval.datasets.build_gold_set --modality txt` to scaffold candidates.
2. Review the output JSONL — rows with `TODO` values need human annotation.
3. Fill in `relevant_chunk_ids` (from ingest output) and `reference_answer` (from source doc).
4. Remove the `TODO` value. Never fabricate ground truth.
5. Run `python -m app.eval.datasets.build_gold_set --validate` to check schema + update manifest.

## File naming

```
datasets/gold/
  text_gold.jsonl       40 triples — SEC filings, earnings transcripts
  pdf_gold.jsonl        40 triples — SEC 10-K PDFs
  docx_gold.jsonl       40 triples — investor presentations
  xlsx_gold.jsonl       40 triples — financial statement tables
  image_gold.jsonl      40 triples — revenue charts, scanned 10-K pages
  audio_gold.jsonl      40 triples — earnings-call MP3 segments
  video_gold.jsonl      40 triples — CNBC clips, IR videos
  routing_gold.jsonl    40 triples — {query, expected_route} pairs
  e2e_gold.jsonl        40 triples — cross-modal multi-hop queries
```
