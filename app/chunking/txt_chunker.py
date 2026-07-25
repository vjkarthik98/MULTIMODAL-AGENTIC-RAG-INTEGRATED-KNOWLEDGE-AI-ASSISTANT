from __future__ import annotations

import re
import time
from pathlib import Path

from prometheus_client import Counter

from app.chunking.base_chunker import BaseChunker
from app.chunking.finance_numbers import (
    approx_tokens,
    deterministic_chunk_id,
    extract_finance_entities,
)
from app.core.config import settings
from app.ingestion.schema import IngestedDocument, RawExtract, UniversalMetadata
from app.utils.logger import get_logger, modality_var

logger = get_logger(__name__)

_CHUNKS_TOTAL = Counter(
    "magik_txt_chunks_total",
    "Total chunks produced by txt chunker",
)
_CHUNK_ERRORS = Counter(
    "magik_txt_chunk_errors_total",
    "Total errors in txt chunker",
)

_SPEAKER_TURN_RE = re.compile(r"^\s*([A-Z][A-Z &\-]{2,40})[:.]\s", re.MULTILINE)
_ALL_CAPS_HEADING = re.compile(r"^[A-Z][A-Z\s&/,()]{4,}$")
_NUMBERED_HEADING = re.compile(r"^(\d+\.)+\s+\S")
_LIST_ITEM = re.compile(r"^[-*•]\s+|^\d+\.\s+")

_CALL_SECTION_KEYWORDS = {
    "operator intro": ["welcome", "good morning", "good afternoon", "operator", "conference call"],
    "prepared_remarks": ["prepared remarks", "opening remarks", "good morning everyone", "ceo"],
    "qa_session": [
        "question and answer",
        "q&a session",
        "thank you",
        "next question",
        "please go ahead",
    ],
    "closing_remarks": ["thank you for joining", "this concludes", "goodbye"],
}


def _detect_chunk_type(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "paragraph"
    first = lines[0]
    if _SPEAKER_TURN_RE.match(first):
        return "speaker_turn"
    if _ALL_CAPS_HEADING.match(first) or _NUMBERED_HEADING.match(first):
        return "section"
    if all(_LIST_ITEM.match(ln) for ln in lines[:3] if ln):
        return "list"
    return "paragraph"


def _extract_speaker(text: str) -> str | None:
    m = _SPEAKER_TURN_RE.match(text)
    return m.group(1) if m else None


def _detect_section_title(text: str) -> str | None:
    first_line = text.splitlines()[0].strip() if text.strip() else ""
    if _ALL_CAPS_HEADING.match(first_line) or _NUMBERED_HEADING.match(first_line):
        return first_line
    return None


def _is_transcript(extracts: list[RawExtract]) -> bool:
    # Check if ingestor already tagged extracts as speaker turns
    speaker_turns = sum(1 for e in extracts if e.extract_type == "speaker_turn")
    if speaker_turns >= 3 or speaker_turns > len(extracts) * 0.3:
        return True
    # Fall back: scan raw text for speaker-name patterns (handles prose extracts)
    sample = "\n".join(e.text for e in extracts[:3])[:8000]
    return len(_SPEAKER_TURN_RE.findall(sample)) >= 3


def _split_on_speaker_turns(text: str) -> list[tuple]:
    """Split a prose block into (speaker, text) segments at speaker turn boundaries.

    Returns list of (speaker_label_or_None, segment_text) pairs in document order.
    """
    positions: list[tuple] = []
    for m in _SPEAKER_TURN_RE.finditer(text):
        # Already anchored to start-of-line by ^\s* — record speaker name from group(1)
        positions.append((m.start(), m.end(), m.group(1)))

    if not positions:
        return [(None, text)]

    segments: list[tuple] = []
    # Text before first speaker turn
    if positions[0][0] > 0:
        prefix_text = text[: positions[0][0]].strip()
        if prefix_text:
            segments.append((None, prefix_text))

    for i, (start, _end, speaker) in enumerate(positions):
        seg_end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        seg_text = text[start:seg_end].strip()
        if seg_text:
            segments.append((speaker, seg_text))

    return segments


class TxtChunker(BaseChunker):
    """Finance-grade chunker for plain-text files (earnings calls, filings, news)."""

    def chunk(
        self,
        extracts: list[RawExtract],
        meta: UniversalMetadata,
    ) -> list[IngestedDocument]:
        source = Path(meta.source_path).name or "unknown.txt"
        surface = "txt_chunker"
        modality_var.set("txt")
        _t0 = time.time()
        logger.info(event="chunking_start", modality="txt", source=source, extracts=len(extracts))
        if not extracts:
            logger.warning(event="no_extracts_received", modality="txt", source=source)
            return []
        try:
            is_transcript_file = _is_transcript(extracts)

            docs: list[IngestedDocument] = []
            chunk_idx = 0
            current_section: str | None = None
            seen_hashes: set = set()
            char_offset = 0  # running character offset across all extract text
            paragraph_number = 0  # counter reset per section (MD Phase 1.1)
            last_section_for_para_counter: str | None = None

            for extract in extracts:
                text = (extract.text or "").strip()
                if not text:
                    continue

                extract_char_start = char_offset

                # For prose extracts in transcript files, split on speaker boundaries first
                # so each chunk carries the correct speaker label and subtype.
                if extract.extract_type == "speaker_turn":
                    sub_segments: list[tuple] = [(extract.speaker_label, text)]
                elif is_transcript_file and extract.extract_type == "prose":
                    sub_segments = _split_on_speaker_turns(text)
                else:
                    title = _detect_section_title(text)
                    if title:
                        current_section = title
                    sub_segments = [(None, text)]

                seg_offset = extract_char_start
                for seg_speaker, seg_text in sub_segments:
                    seg_text = seg_text.strip()
                    if not seg_text:
                        seg_offset += len(seg_text) + 1
                        continue

                    is_speaker_seg = bool(seg_speaker) or extract.extract_type == "speaker_turn"
                    chunk_type = "speaker_turn" if is_speaker_seg else _detect_chunk_type(seg_text)

                    if approx_tokens(seg_text) <= settings.CHUNK_TARGET_TOKENS:
                        pieces = [seg_text]
                    else:
                        pieces = self._split_text(seg_text)

                    piece_offset = seg_offset
                    for piece in pieces:
                        if not piece.strip():
                            piece_offset += len(piece) + 1
                            continue
                        h = hash(piece)
                        if h in seen_hashes:
                            piece_offset += len(piece) + 1
                            continue
                        seen_hashes.add(h)

                        char_start = piece_offset
                        char_end = piece_offset + len(piece)

                        # Reset paragraph counter when section changes
                        if current_section != last_section_for_para_counter:
                            paragraph_number = 0
                            last_section_for_para_counter = current_section
                        if chunk_type == "paragraph":
                            paragraph_number += 1

                        speaker = seg_speaker or _extract_speaker(piece)
                        fin_entities = extract_finance_entities(piece)
                        chunk_hash = deterministic_chunk_id(
                            source, extract.raw_source_ref or "txt", chunk_idx
                        )

                        structure = {
                            "chunk_hash_id": chunk_hash,
                            "source_file": source,
                            "chunk_index": chunk_idx,
                            "paragraph_number": paragraph_number,
                            "char_start": char_start,
                            "char_end": char_end,
                            "section_title": current_section,
                            "speaker": speaker,
                            "is_transcript": is_transcript_file,
                            "chunk_type": chunk_type,
                            "finance_entities": fin_entities,
                        }

                        piece_offset = char_end + 1

                        subtype = (
                            "speaker_turn"
                            if chunk_type == "speaker_turn"
                            else ("heading" if chunk_type == "section" else "paragraph")
                        )

                        doc = self._make_doc(
                            text=piece,
                            modality="txt",
                            subtype=subtype,
                            source=source,
                            page=None,
                            chunk_idx=chunk_idx,
                            structure=structure,
                            meta=meta,
                            surface=surface,
                        )
                        if doc:
                            docs.append(doc)
                            chunk_idx += 1

                    seg_offset += len(seg_text) + 1

                char_offset += len(text) + 1  # +1 for separator between extracts

            logger.info(event="txt_chunking_done", source=source, chunks=len(docs))
            _CHUNKS_TOTAL.inc(len(docs))
            return docs
        except Exception as _exc:
            _CHUNK_ERRORS.inc()
            logger.error(event="chunking_failed", modality="txt", source=source, error=str(_exc))
            raise

    def health_check(self) -> dict:
        return {
            "modality": "txt",
            "status": "ok",
            "class": self.__class__.__name__,
        }
