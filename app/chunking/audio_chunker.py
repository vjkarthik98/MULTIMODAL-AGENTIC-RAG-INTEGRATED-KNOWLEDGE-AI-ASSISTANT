from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.chunking.base_chunker import BaseChunker
from app.chunking.finance_numbers import deterministic_chunk_id, extract_finance_entities
from app.ingestion.schema import IngestedDocument, RawExtract, UniversalMetadata
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL WRAPPERS  (merged from app/models/diarizer.py and ner_extractor.py)
# ══════════════════════════════════════════════════════════════════════════════

def diarize(audio_path: str) -> List[Tuple[float, float, str]]:
    """Run pyannote speaker diarization. Returns (start, end, speaker) tuples."""
    try:
        from app.core.model_loader import loader
        pipeline = loader.get_diarizer()
    except Exception as exc:
        logger.warning(event="diarizer_unavailable", error=str(exc))
        return []
    try:
        diarization = pipeline(audio_path)
        segments: List[Tuple[float, float, str]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
        return sorted(segments, key=lambda x: x[0])
    except Exception as exc:
        logger.warning(event="diarization_failed", error=str(exc))
        return []


_ENTITY_KEYS = {"ORG": "companies", "PER": "persons", "LOC": "locations", "MISC": "misc"}


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Run dslim/bert-base-NER. Returns {companies, persons, locations, misc}."""
    result: Dict[str, List[str]] = {v: [] for v in _ENTITY_KEYS.values()}
    if not text.strip():
        return result
    try:
        from app.core.model_loader import loader
        ner_pipeline = loader.get_ner()
    except Exception as exc:
        logger.warning(event="ner_unavailable", error=str(exc))
        return result
    try:
        seen: set = set()
        for ent in ner_pipeline(text[:2000]):
            label = ent.get("entity_group", "")
            word  = ent.get("word", "").strip()
            if not word or word in seen:
                continue
            seen.add(word)
            result[_ENTITY_KEYS.get(label, "misc")].append(word)
        return result
    except Exception as exc:
        logger.warning(event="ner_failed", error=str(exc))
        return result


# Spoken word rate: ~150 wpm → 90s ≈ 225 words ≈ upper bound per chunk.
_MIN_WORDS = 75
_MAX_WORDS = 225
_MERGE_SHORT_S = 30.0    # merge speaker segments shorter than this

_TOPIC_TRANSITIONS = re.compile(
    r"\b(moving to|turning to|let me now|switching to|on the balance sheet|"
    r"cash flow|guidance|outlook|next question|question and answer|q&a|"
    r"basis points|year over year|quarter over quarter)\b",
    re.IGNORECASE,
)

_CALL_SECTIONS = [
    ("operator_intro",   re.compile(r"\b(welcome|thank you for joining|good (?:morning|afternoon))\b", re.I)),
    ("qa_session",       re.compile(r"\b(next question|please go ahead|analyst|q&a|question and answer)\b", re.I)),
    ("closing_remarks",  re.compile(r"\b(this concludes|thank you for joining|goodbye)\b", re.I)),
    ("prepared_remarks", re.compile(r".*", re.I)),   # fallback
]

_ROLE_KEYWORDS = {
    "CEO":      re.compile(r"\bceo\b|\bchief executive\b", re.I),
    "CFO":      re.compile(r"\bcfo\b|\bchief financial\b", re.I),
    "Analyst":  re.compile(r"\banalyst\b|\bgoldman\b|\bmorgan\b|\bjp morgan\b|\bciti\b", re.I),
    "Operator": re.compile(r"\boperator\b", re.I),
}

_FILLER = re.compile(r"\b(um|uh|er|ah|you know|i mean|like|so|basically|essentially)\b", re.I)


def _detect_role(text: str) -> Optional[str]:
    for role, pat in _ROLE_KEYWORDS.items():
        if pat.search(text):
            return role
    return None


def _detect_call_section(text: str) -> str:
    for section, pat in _CALL_SECTIONS:
        if pat.search(text):
            return section
    return "prepared_remarks"


def _remove_fillers(text: str) -> str:
    return re.sub(r"\s+", " ", _FILLER.sub("", text)).strip()


def _map_speaker_roles(
    diarization: List[Tuple[float, float, str]],
    transcript_text: str,
) -> Dict[str, str]:
    """Map SPEAKER_XX labels to roles using the first 60s of transcript."""
    role_map: Dict[str, str] = {}
    early = [d for d in diarization if d[0] < 60]
    for start, end, label in early:
        if label in role_map:
            continue
        role = _detect_role(transcript_text[:500])
        if role:
            role_map[label] = role
    return role_map


def _run_whisper(wav_path: str) -> List[Dict]:
    """Transcribe with faster_whisper; returns list of word dicts."""
    try:
        from app.core.model_loader import loader
        model = loader.get_whisper()
        segments, _ = model.transcribe(wav_path, word_timestamps=True)
        words = []
        for seg in segments:
            if hasattr(seg, "words") and seg.words:
                for w in seg.words:
                    words.append({"word": w.word, "start": w.start, "end": w.end})
            else:
                # Segment-level fallback when word_timestamps not available.
                words.append({
                    "word": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                })
        return words
    except Exception as exc:
        logger.warning(event="whisper_failed", error=str(exc))
        return []


def _assemble_chunks(
    words: List[Dict],
    diarization: List[Tuple[float, float, str]],
    role_map: Dict[str, str],
) -> List[Dict]:
    """Group words into speaker-aware, topic-bounded chunks."""
    if not words:
        return []

    def speaker_at(t: float) -> str:
        for start, end, label in diarization:
            if start <= t <= end:
                return label
        return "SPEAKER_00"

    chunks: List[Dict] = []
    buf_words: List[str] = []
    buf_start = words[0]["start"]
    buf_end = words[0]["end"]
    buf_speaker = speaker_at(buf_start)

    for w in words:
        word_text = w["word"]
        word_start = w["start"]
        word_end = w["end"]
        current_speaker = speaker_at(word_start)

        over_max = len(buf_words) >= _MAX_WORDS
        speaker_changed = current_speaker != buf_speaker and len(buf_words) >= _MIN_WORDS
        topic_hit = bool(_TOPIC_TRANSITIONS.search(word_text)) and len(buf_words) >= _MIN_WORDS

        if over_max or speaker_changed or topic_hit:
            if buf_words:
                transcript = " ".join(buf_words)
                chunks.append({
                    "transcript":   _remove_fillers(transcript),
                    "start":        buf_start,
                    "end":          buf_end,
                    "speaker":      buf_speaker,
                    "role":         role_map.get(buf_speaker),
                    "call_section": _detect_call_section(transcript),
                })
            buf_words = []
            buf_start = word_start
            buf_speaker = current_speaker

        buf_words.append(word_text)
        buf_end = word_end

    if buf_words:
        transcript = " ".join(buf_words)
        chunks.append({
            "transcript":   _remove_fillers(transcript),
            "start":        buf_start,
            "end":          buf_end,
            "speaker":      buf_speaker,
            "role":         role_map.get(buf_speaker),
            "call_section": _detect_call_section(transcript),
        })

    return chunks


class AudioChunker(BaseChunker):
    """Finance-grade chunker for audio files (earnings calls, press conferences, podcasts).

    Pipeline: Whisper transcription → pyannote diarization → speaker-role mapping
              → topic-aware chunking → NER entity extraction.
    """

    def chunk(
        self,
        extracts: List[RawExtract],
        meta: UniversalMetadata,
    ) -> List[IngestedDocument]:
        source = Path(meta.source_path).name or "unknown.mp3"
        surface = "audio_chunker"
        logger.info(event="chunking_start", modality="audio", source=source, extracts=len(extracts))
        if not extracts:
            logger.warning(event="no_extracts_received", modality="audio", source=source)
            return []
        docs: List[IngestedDocument] = []

        for ext in extracts:
            if ext.extract_type != "audio_raw":
                continue

            # Write WAV bytes to a temp file for Whisper and pyannote.
            raw = ext.raw_bytes or b""
            if not raw:
                logger.warning(event="audio_chunker_empty_bytes", source=source)
                continue

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(raw)
                wav_path = f.name

            try:
                words = _run_whisper(wav_path)

                # Diarization (optional — skipped if model unavailable).
                diarization: List[Tuple[float, float, str]] = []
                try:
                    diarization = diarize(wav_path)
                except Exception:
                    pass

                full_transcript = " ".join(w["word"] for w in words)
                role_map = _map_speaker_roles(diarization, full_transcript)
                raw_chunks = _assemble_chunks(words, diarization, role_map)

                for chunk_idx, ch in enumerate(raw_chunks):
                    transcript = ch["transcript"]
                    if not transcript.strip():
                        continue

                    # NER entity extraction.
                    ner_entities: Dict = {}
                    try:
                        ner_entities = extract_entities(transcript)
                    except Exception:
                        pass

                    fin_entities = extract_finance_entities(transcript)
                    duration = ch["end"] - ch["start"]
                    word_count = len(transcript.split())
                    call_section = ch.get("call_section", "prepared_remarks")

                    chunk_hash = deterministic_chunk_id(
                        source, f"audio_{ch['start']:.1f}", chunk_idx
                    )
                    structure = {
                        "chunk_hash_id":    chunk_hash,
                        "source_file":      source,
                        "chunk_index":      chunk_idx,
                        "start_timestamp":  round(ch["start"], 3),
                        "end_timestamp":    round(ch["end"], 3),
                        "duration_seconds": round(duration, 3),
                        "speaker_label":    ch["speaker"],
                        "speaker_name":     ch.get("name"),
                        "speaker_role":     ch.get("role"),
                        "topic_section":    ch.get("topic_section"),
                        "call_section":     call_section,
                        "transcript":       transcript,
                        "finance_entities": {
                            "regex": fin_entities,
                            **ner_entities,
                        },
                        "word_count":       word_count,
                        "is_question":      transcript.rstrip().endswith("?"),
                        "is_answer":        call_section == "qa_session" and not transcript.rstrip().endswith("?"),
                    }

                    doc = self._make_doc(
                        text=transcript,
                        modality="mp3",
                        subtype="speech",
                        source=source,
                        page=None,
                        chunk_idx=chunk_idx,
                        structure=structure,
                        meta=meta,
                        surface=surface,
                    )
                    if doc:
                        docs.append(doc)

            finally:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

        logger.info(event="audio_chunking_done", source=source, chunks=len(docs))
        return docs
