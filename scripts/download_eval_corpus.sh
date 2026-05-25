#!/usr/bin/env bash
# Download the Phase 25 eval corpus from public sources.
#
# Sources: SEC EDGAR (public domain), Federal Reserve FRED (public domain),
#          Apple IR (public), JPMorgan IR (public), archive.org (public domain).
#
# Usage:
#   bash scripts/download_eval_corpus.sh               # all modalities
#   bash scripts/download_eval_corpus.sh --modality txt
#   bash scripts/download_eval_corpus.sh --modality pdf
#   bash scripts/download_eval_corpus.sh --dry-run     # print URLs, download nothing
#
# After downloading, run:
#   python -m app.eval.datasets.build_gold_set --modality all --ingest
# to ingest the corpus into your Qdrant instance and fill relevant_chunk_ids.

set -uo pipefail
# Note: intentionally NOT using set -e so individual download failures are
# warnings rather than script aborts. Each download() call handles its own failure.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CORPUS_DIR="${REPO_ROOT}/data/raw/finance"

MODALITY="${1:-all}"
DRY_RUN=false
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=true
  [[ "$arg" == "--modality" ]] && shift && MODALITY="${1:-all}"
done

log() { echo "[download_eval_corpus] $*"; }

# SEC EDGAR requires a descriptive User-Agent per their access policy:
# https://www.sec.gov/os/accessing-edgar-data
# Format: "Sample Company Name AdminContact@example.com"
EDGAR_UA="${EDGAR_USER_AGENT:-"MultimodalRAGEval karthikvj398@gmail.com"}"

download() {
  local url="$1" dest="$2"
  if $DRY_RUN; then
    log "DRY-RUN: would download $url -> $dest"
    return
  fi
  if [[ -f "$dest" ]]; then
    log "already exists: $dest (skip)"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  log "downloading: $url"
  # SEC EDGAR blocks requests without a proper User-Agent (returns 403).
  # All other sources also work fine with this header.
  curl -fsSL --retry 3 --retry-delay 5 \
    -H "User-Agent: ${EDGAR_UA}" \
    -H "Accept-Encoding: gzip, deflate" \
    -H "Host: $(echo "$url" | awk -F/ '{print $3}')" \
    -o "$dest" "$url" \
    || { log "WARN: failed to download $url"; return 1; }
  log "saved: $dest"
}

# ── TXT ──────────────────────────────────────────────────────────────────────
download_txt() {
  log "=== TXT corpus ==="
  # Apple 10-K FY2023 — SEC EDGAR (public domain)
  # Accession: 0000320193-23-000106
  download \
    "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm" \
    "${CORPUS_DIR}/txt/aapl_10k_2023_full.htm"

  # JPMorgan 10-K FY2023 — SEC EDGAR (public domain)
  # Accession: 0000019617-24-000225 (filed 2024-02-27, period 2023-12-31)
  download \
    "https://www.sec.gov/Archives/edgar/data/19617/000001961724000225/jpm-20231231.htm" \
    "${CORPUS_DIR}/txt/jpm_10k_2023_full.htm"

  # FRED GDP series (public domain, Federal Reserve)
  download \
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDP" \
    "${CORPUS_DIR}/txt/fred_gdp_series.csv"

  if [[ ! -f "${CORPUS_DIR}/txt/aapl_10k_2023_full.htm" ]]; then
    log "Pre-committed excerpts at data/raw/finance/txt/ will be used as fallback."
    log "To retry: EDGAR_USER_AGENT='YourName email@example.com' bash $0 --modality txt"
  fi
}

# ── PDF ──────────────────────────────────────────────────────────────────────
download_pdf() {
  log "=== PDF corpus ==="
  # MSFT 10-K FY2023 — SEC EDGAR (public domain)
  # Accession: 0000950170-23-035122 (filed 2023-07-27, period 2023-06-30)
  download \
    "https://www.sec.gov/Archives/edgar/data/789019/000095017023035122/msft-20230630.htm" \
    "${CORPUS_DIR}/pdf/msft_10k_2023.htm"

  # Tesla 10-K FY2023 — SEC EDGAR (public domain)
  # Accession: 0001628280-24-002390 (filed 2024-01-29, period 2023-12-31)
  download \
    "https://www.sec.gov/Archives/edgar/data/1318605/000162828024002390/tsla-20231231.htm" \
    "${CORPUS_DIR}/pdf/tsla_10k_2023.htm"

  # Berkshire Hathaway Annual Letter 2022 — public IR (actual PDF)
  download \
    "https://www.berkshirehathaway.com/letters/2022ltr.pdf" \
    "${CORPUS_DIR}/pdf/berkshire_letter_2022.pdf"

  log "PDF NOTE: EDGAR 10-K filings are HTM; saved as .htm in pdf/ dir. Ingest pipeline handles both."
}

# ── DOCX ─────────────────────────────────────────────────────────────────────
download_docx() {
  log "=== DOCX corpus ==="
  # Apple DEF 14A FY2024 proxy — SEC EDGAR (public domain)
  # Accession: 0001308179-24-000010 (filed 2024-01-11)
  download \
    "https://www.sec.gov/Archives/edgar/data/320193/000130817924000010/laapl2024_def14a.htm" \
    "${CORPUS_DIR}/docx/aapl_def14a_2024.htm"

  # Apple DEF 14A FY2023 proxy — SEC EDGAR (public domain)
  # Accession: 0001308179-23-000019 (filed 2023-01-12)
  download \
    "https://www.sec.gov/Archives/edgar/data/320193/000130817923000019/laap2023_def14a.htm" \
    "${CORPUS_DIR}/docx/aapl_def14a_2023.htm"

  log "DOCX NOTE: .htm files from EDGAR are saved in docx/. Ingest pipeline accepts both."
  log "To produce actual .docx: pandoc aapl_def14a_2023.htm -o aapl_def14a_2023.docx"
}

# ── XLSX ─────────────────────────────────────────────────────────────────────
download_xlsx() {
  log "=== XLSX corpus ==="
  # SEC EDGAR full-text index Q4 2023 — company filings listing (TSV-like, ~38MB)
  # This is the standard EDGAR quarterly index, publicly accessible
  download \
    "https://www.sec.gov/Archives/edgar/full-index/2023/QTR4/form.idx" \
    "${CORPUS_DIR}/xlsx/sec_edgar_form_index_2023q4.txt"

  # FRED multiple economic indicators as CSV (Federal Reserve, public domain)
  download \
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500" \
    "${CORPUS_DIR}/xlsx/fred_sp500.csv"

  # FRED multiple series as CSV (Federal Reserve, public domain)
  download \
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDP" \
    "${CORPUS_DIR}/xlsx/fred_gdp_quarterly.csv"

  download \
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE" \
    "${CORPUS_DIR}/xlsx/fred_unemployment_rate.csv"

  log "XLSX NOTE: CSV files are accepted by the pipeline's XLSX ingest path."
}

# ── IMAGE ─────────────────────────────────────────────────────────────────────
download_image() {
  log "=== IMAGE corpus ==="
  # FRED chart PNG — GDP over time (public domain, Federal Reserve)
  download \
    "https://fred.stlouisfed.org/graph/fredgraph.png?id=GDP&width=800&height=400" \
    "${CORPUS_DIR}/image/aapl_revenue_chart.jpg"

  # FRED CPI chart — finance document scan substitute (public domain)
  download \
    "https://fred.stlouisfed.org/graph/fredgraph.png?id=CPIAUCSL&width=800&height=400" \
    "${CORPUS_DIR}/image/10k_cover_page_scan.jpg"

  log "IMAGE NOTE: FRED PNG charts are used as OCR test images."
  log "For real OCR eval, place scanned 10-K cover JPGs in ${CORPUS_DIR}/image/"
  log "and update image_gold.jsonl gold_ocr_text entries accordingly."
}

# ── AUDIO ─────────────────────────────────────────────────────────────────────
download_audio() {
  log "=== AUDIO corpus ==="
  # Apple Q4 2023 earnings call — Apple IR publishes audio (public)
  # Primary: Apple investor relations podcast feed
  download \
    "https://www.apple.com/investor/podcasts/Apple_Q4_FY23_Earnings_Call.mp3" \
    "${CORPUS_DIR}/audio/apple_q4_2023_earnings_call.mp3"

  log "AUDIO NOTE: If the Apple IR URL is unavailable, obtain the earnings call"
  log "recording from a licensed IR data provider or Seeking Alpha transcript."
  log "File must be placed at: ${CORPUS_DIR}/audio/apple_q4_2023_earnings_call.mp3"
}

# ── VIDEO ─────────────────────────────────────────────────────────────────────
download_video() {
  log "=== VIDEO corpus ==="
  # CNBC public earnings highlight — YouTube-dl/yt-dlp from public CNBC YouTube
  # We reference a known CNBC public earnings video; requires yt-dlp installed.
  CNBC_URL="https://www.youtube.com/watch?v=HovbHzjcMqE"  # Apple Q4 2023 results clip
  DEST="${CORPUS_DIR}/video/cnbc_earnings_highlight.mp4"

  if $DRY_RUN; then
    log "DRY-RUN: would download $CNBC_URL -> $DEST (requires yt-dlp)"
    return
  fi

  if [[ -f "$DEST" ]]; then
    log "already exists: $DEST (skip)"
    return
  fi

  mkdir -p "$(dirname "$DEST")"
  if command -v yt-dlp &>/dev/null; then
    yt-dlp -f "mp4[height<=480]" -o "$DEST" "$CNBC_URL" \
      || log "WARN: yt-dlp failed; place the video manually at $DEST"
  else
    log "WARN: yt-dlp not found. Install with: pip install yt-dlp"
    log "Then run: yt-dlp -f 'mp4[height<=480]' -o '$DEST' '$CNBC_URL'"
  fi
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
case "$MODALITY" in
  all)
    download_txt
    download_pdf
    download_docx
    download_xlsx
    download_image
    download_audio
    download_video
    ;;
  txt)   download_txt ;;
  pdf)   download_pdf ;;
  docx)  download_docx ;;
  xlsx)  download_xlsx ;;
  image) download_image ;;
  audio) download_audio ;;
  video) download_video ;;
  *)
    echo "Unknown modality: $MODALITY"
    echo "Usage: $0 [--modality txt|pdf|docx|xlsx|image|audio|video|all] [--dry-run]"
    exit 1
    ;;
esac

log "Done. Next step:"
log "  python -m app.eval.datasets.build_gold_set --modality all --ingest"
log "  (then review TODO rows in app/eval/datasets/gold/*.jsonl and fill in gold answers)"
