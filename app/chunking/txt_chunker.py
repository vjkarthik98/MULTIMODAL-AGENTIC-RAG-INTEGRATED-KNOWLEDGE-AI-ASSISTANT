from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from app.chunking.base_chunker import BaseChunker
from app.chunking.finance_numbers import (
    deterministic_chunk_id,
    extract_finance_entities,
)
from app.core.config import settings
from app.ingestion.schema import IngestedDocument, RawExtract, UniversalMetadata
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SPEAKER_TURN_RE = re.compile(r"^[A-Z][A-Z &\-]{2,40}:\s", re.MULTILINE)
_ALL_CAPS_HEADING = re.compile(r"^[A-Z][A-Z\s&/,()]{4,}$")
_NUMBERED_HEADING = re.compile(r"^(\d+\.)+\s+\S")
_LIST_ITEM = re.compile(r"^[-*•]\s+|^\d+\.\s+")

_CALL_SECTION_KEYWORDS = {
    "operator intro": ["welcome", "good morning", "good afternoon", "operator", "conference call"],
    "prepared_remarks": ["prepared remarks", "opening remarks", "good morning everyone", "ceo"],
    "qa_session": ["question and answer", "q&a session", "thank you", "next question", "please go ahead"],
    "closing_remarks": ["thank you for joining", "this concludes", "goodbye"],
}


def _detect_chunk_type(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return "paragraph"
    first = lines[0]
    if _SPEAKER_TURN_RE.match(first):
        return "speaker_turn"
    if _ALL_CAPS_HEADING.match(first) or _NUMBERED_HEADING.match(first):
        return "section"
    if all(_LIST_ITEM.match(l) for l in lines[:3] if l):
        return "list"
    return "paragraph"


def _extract_speaker(text: str) -> Optional[str]:
    m = _SPEAKER_TURN_RE.match(text)
    return m.group().rstrip(": ").strip() if m else None


def _detect_section_title(text: str) -> Optional[str]:
    first_line = text.splitlines()[0].strip() if text.strip() else ""
    if _ALL_CAPS_HEADING.match(first_line) or _NUMBERED_HEADING.match(first_line):
        return first_line
    return None


def _is_transcript(extracts: List[RawExtract]) -> bool:
    speaker_turns = sum(1 for e in extracts if e.extract_type == "speaker_turn")
    return speaker_turns >= 3 or speaker_turns > len(extracts) * 0.3


class TxtChunker(BaseChunker):
    """Finance-grade chunker for plain-text files (earnings calls, filings, news)."""

    def chunk(
        self,
        extracts: List[RawExtract],
        meta: UniversalMetadata,
    ) -> List[IngestedDocument]:
        source = Path(meta.source_path).name or "unknown.txt"
        surface = "txt_chunker"
        is_transcript_file = _is_transcript(extracts)

        docs: List[IngestedDocument] = []
        chunk_idx = 0
        current_section: Optional[str] = None
        seen_hashes: set = set()

        for extract in extracts:
            text = (extract.text or "").strip()
            if not text:
                continue

            if extract.extract_type == "speaker_turn":
                pieces = [text]
                chunk_type = "speaker_turn"
            else:
                title = _detect_section_title(text)
                if title:
                    current_section = title

                chunk_type = _detect_chunk_type(text)
                pieces = self._split_text(text) if chunk_type != "speaker_turn" else [text]

            for piece in pieces:
                if not piece.strip():
                    continue
                h = hash(piece)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                speaker = extract.speaker_label or _extract_speaker(piece)
                fin_entities = extract_finance_entities(piece)
                chunk_hash = deterministic_chunk_id(source, extract.raw_source_ref or "txt", chunk_idx)

                structure = {
                    "chunk_hash_id":    chunk_hash,
                    "source_file":      source,
                    "chunk_index":      chunk_idx,
                    "section_title":    current_section,
                    "speaker":          speaker,
                    "is_transcript":    is_transcript_file,
                    "chunk_type":       chunk_type,
                    "finance_entities": fin_entities,
                }

                subtype = "speaker_turn" if chunk_type == "speaker_turn" else (
                    "heading" if chunk_type == "section" else "paragraph"
                )

                doc = self._make_doc(
                    text=piece,
                    modality="text",
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

        logger.info(event="txt_chunking_done", source=source, chunks=len(docs))
        return docs
