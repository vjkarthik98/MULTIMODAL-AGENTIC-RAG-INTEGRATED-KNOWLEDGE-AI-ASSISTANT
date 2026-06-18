from __future__ import annotations

import bisect
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.chunking.base_chunker import BaseChunker
from app.chunking.finance_numbers import (
    approx_tokens,
    deterministic_chunk_id,
    extract_finance_entities,
)
from app.ingestion.schema import IngestedDocument, RawExtract, UniversalMetadata
from app.utils.logger import get_logger, modality_var

import time
from prometheus_client import Counter, Histogram

logger = get_logger(__name__)

_CHUNKS_TOTAL = Counter(
    "magik_audio_chunks_total",
    "Total chunks produced by audio chunker",
)
_CHUNK_ERRORS = Counter(
    "magik_audio_chunk_errors_total",
    "Total errors in audio chunker",
)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL WRAPPERS  (merged from app/models/diarizer.py and ner_extractor.py)
# ══════════════════════════════════════════════════════════════════════════════

def diarize(audio_path: str) -> List[Tuple[float, float, str]]:
    """Run pyannote speaker diarization. Returns (start, end, speaker) tuples."""
    try:
        from app.core.model_loader import model_loader as loader
        pipeline = loader.get_diarizer()
    except Exception as exc:
        logger.warning(event="diarizer_unavailable", error=str(exc))
        return []
    try:
        raw = pipeline(audio_path)
        # pyannote ≥3.3 wraps result in DiarizeOutput; unwrap to Annotation.
        annotation = getattr(raw, "speaker_diarization", raw)
        segments: List[Tuple[float, float, str]] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
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
        from app.core.model_loader import model_loader as loader
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
    # FOMC/press-conference Q&A: "QUESTION.", "QUESTIONER", reporter intros, "from [outlet]"
    ("qa_session",       re.compile(
        r"\b(next question|please go ahead|analyst|q&a|question and answer"
        r"|QUESTION\b|questioner|from (?:the |)(?:wall street|new york times?|reuters|bloomberg|wsj|cnbc|fox|ap |associated press)"
        r"|reporter|press conference question)\b",
        re.I,
    )),
    ("closing_remarks",  re.compile(r"\b(this concludes|thank you for joining|goodbye)\b", re.I)),
    ("prepared_remarks", re.compile(r".*", re.I)),   # fallback
]

_ROLE_KEYWORDS = {
    "Fed Chair":  re.compile(r"\b(chairman|chairwoman|chair powell|chair bernanke|chair yellen|chair jerome|federal reserve chair)\b", re.I),
    "CEO":        re.compile(r"\bceo\b|\bchief executive\b", re.I),
    "CFO":        re.compile(r"\bcfo\b|\bchief financial\b", re.I),
    "President":  re.compile(r"\bpresident\b(?!\s+of the\s+united)", re.I),
    "Analyst":    re.compile(r"\banalyst\b|\bgoldman\b|\bmorgan\b|\bjp morgan\b|\bciti\b|\bbarclays\b|\bubs\b|\bdb\b|\bdeutsche bank\b", re.I),
    "Reporter":   re.compile(r"\b(reporter|journalist|correspondent|from (?:the |)(?:wall street|new york times?|reuters|bloomberg|wsj|cnbc|fox|ap |associated press|financial times|ft\b))\b", re.I),
    "Operator":   re.compile(r"\boperator\b", re.I),
    "COO":        re.compile(r"\bcoo\b|\bchief operating\b", re.I),
}

_FILLER = re.compile(r"\b(um|uh|er|ah|you know|i mean|like|so|basically|essentially)\b", re.I)

# Spoken name patterns (three forms):
#   "MR. POWELL:" / "MS. BRAINARD:" — title prefix, all-caps surname
#   "CHAIRMAN BERNANKE." / "VICE CHAIR WILLIAMS." — FOMC all-caps role+name
#   "Luca Maestri:" / "Tim Cook," — earnings call title-case name
_SPOKEN_NAME_RE = re.compile(
    r"(?:^|\n)\s*(?:MR\.|MS\.|MRS\.|DR\.)\s+([A-Z][A-Z\s\-']+?)[:.]\s"
    r"|(?:^|\n)\s*(?:CHAIR(?:MAN|WOMAN)?|VICE\s+CHAIR(?:MAN|WOMAN)?|PRESIDENT|GOVERNOR|SECRETARY)\s+([A-Z][A-Z\s\-']{1,30})[:.]\s"
    r"|([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*[:,]",
    re.MULTILINE,
)


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
    full_transcript: str,
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """Map SPEAKER_XX labels to (role, name) using per-speaker transcript context.

    Strategy:
    1. Parse FOMC-style "MR. POWELL:" and earnings-call "Luca Maestri:" name markers from
       the surrounding window of each spoken-name occurrence to bind name → speaker.
    2. Detect role from each speaker's own transcribed text (not the global transcript),
       so the dominant speaker (CEO/Chair) doesn't pollute everyone else's role.
    Returns {speaker_label: (role, name)} — both may be None.
    """
    role_name_map: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    if not diarization:
        return role_name_map

    # --- Build approximate word → time index so we can map name-marker positions ---
    # diarization is sorted; build start-time list for bisect lookups.
    import bisect as _bisect
    _diar_starts = [seg[0] for seg in diarization]

    def _speaker_at_approx(char_pos: int, total_chars: int) -> str:
        """Estimate speaker label at character position via linear time mapping."""
        if not diarization:
            return "SPEAKER_00"
        total_dur = diarization[-1][1] if diarization else 1.0
        t = (char_pos / max(total_chars, 1)) * total_dur
        idx = _bisect.bisect_right(_diar_starts, t) - 1
        if idx >= 0:
            _, end, label = diarization[idx]
            if t <= end:
                return label
        return diarization[0][2]

    total_chars = len(full_transcript)

    # --- Map name markers to speaker labels ---
    label_to_name: Dict[str, str] = {}
    for m in _SPOKEN_NAME_RE.finditer(full_transcript):
        # group(1): MR./MS. prefix form  group(2): FOMC CHAIRMAN/PRESIDENT form  group(3): title-case form
        raw = m.group(1) or m.group(2) or m.group(3) or ""
        name = raw.strip().title()
        if not name:
            continue
        spk = _speaker_at_approx(m.start(), total_chars)
        if spk not in label_to_name:   # keep first (most reliable)
            label_to_name[spk] = name

    # --- Per-speaker word buckets for independent role detection ---
    # Split transcript words by time-proportional position.
    words = full_transcript.split()
    n = max(len(words), 1)
    total_dur = diarization[-1][1] if diarization else 1.0
    speaker_word_buckets: Dict[str, List[str]] = {}
    for wi, word in enumerate(words):
        t = (wi / n) * total_dur
        idx = _bisect.bisect_right(_diar_starts, t) - 1
        if idx >= 0:
            _, end, label = diarization[idx]
            if t <= end:
                speaker_word_buckets.setdefault(label, []).append(word)

    # --- Assign role and name to each unique speaker label ---
    ordered_labels = list(dict.fromkeys(seg[2] for seg in diarization))
    for label in ordered_labels:
        name = label_to_name.get(label)
        # Detect role from this speaker's own words (first 500), then fall back to name.
        own_text = " ".join(speaker_word_buckets.get(label, [])[:500])
        role = _detect_role(own_text) or (name and _detect_role(name)) or None
        role_name_map[label] = (role, name)

    return role_name_map


def _run_whisper(wav_path: str) -> List[Dict]:
    """Transcribe with faster_whisper; returns list of word dicts."""
    try:
        from app.core.model_loader import model_loader as loader
        model = loader.get_whisper()
        segments, _ = model.transcribe(
            wav_path,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
        )
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


def _last_sentence(text: str) -> str:
    """Return the last complete sentence from text for overlap seeding."""
    # Split on sentence-ending punctuation followed by space or end-of-string.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return sentences[-1] if sentences else ""


def _assemble_chunks(
    words: List[Dict],
    diarization: List[Tuple[float, float, str]],
    role_map: Dict[str, Tuple[Optional[str], Optional[str]]],
) -> List[Dict]:
    """Group words into speaker-aware, topic-bounded chunks with 1-sentence overlap.

    Hard boundaries: speaker change (≥ MIN_WORDS buffered), topic transition phrase,
                     exceeding MAX_WORDS.
    Soft boundary: nearest sentence end to the target word count.
    Overlap: last sentence of the previous chunk is prepended to each new chunk so
             a financial statement split across a boundary remains retrievable.
    """
    if not words:
        return []

    # Pre-extract sorted diarization start times for O(log s) binary search.
    _diar_starts: List[float] = [seg[0] for seg in diarization]

    def speaker_at(t: float) -> str:
        idx = bisect.bisect_right(_diar_starts, t) - 1
        if idx >= 0:
            _start, _end, _label = diarization[idx]
            if t <= _end:
                return _label
        return "SPEAKER_00"

    def _flush(buf: List[str], start: float, end: float, spk: str, overlap_seed: str) -> Dict:
        raw = " ".join(buf)
        text = _remove_fillers(raw)
        role, name = role_map.get(spk, (None, None))
        topic = _TOPIC_TRANSITIONS.search(raw)
        # Prepend overlap sentence (context bridge, not counted in word_count)
        display = f"{overlap_seed} {text}".strip() if overlap_seed else text
        return {
            "transcript":    display,
            "start":         start,
            "end":           end,
            "speaker":       spk,
            "role":          role,
            "name":          name,
            "call_section":  _detect_call_section(raw),
            "topic_section": topic.group(0).replace(" ", "_") if topic else None,
        }

    chunks: List[Dict] = []
    buf_words: List[str] = []
    buf_start = words[0]["start"]
    buf_end = words[0]["end"]
    buf_speaker = speaker_at(buf_start)
    overlap_seed = ""  # last sentence of previous chunk

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
                chunk = _flush(buf_words, buf_start, buf_end, buf_speaker, overlap_seed)
                overlap_seed = _last_sentence(chunk["transcript"])
                chunks.append(chunk)
            buf_words = []
            buf_start = word_start
            buf_speaker = current_speaker

        buf_words.append(word_text)
        buf_end = word_end

    if buf_words:
        chunks.append(_flush(buf_words, buf_start, buf_end, buf_speaker, overlap_seed))

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
        modality_var.set("audio")
        _t0 = time.time()
        logger.info(event="chunking_start", modality="audio", source=source, extracts=len(extracts))
        if not extracts:
            logger.warning(event="no_extracts_received", modality="audio", source=source)
            return []
        try:
            docs: List[IngestedDocument] = []

            for ext in extracts:
                if ext.extract_type != "audio_raw":
                    continue

                # Write WAV bytes to a temp file for Whisper and pyannote.
                raw = ext.raw_bytes or b""
                if not raw:
                    logger.warning(event="audio_chunker_empty_bytes", source=source)
                    continue

                # Pull quality signals forwarded from AudioIngestor.extract().
                ext_extra = ext.extra or {}
                snr = ext_extra.get("snr")
                snr_degraded = ext_extra.get("snr_degraded", False)
                clipping_detected = ext_extra.get("clipping_detected", False)

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

                    # Document-level earnings-call detection — checked once per extract.
                    _ft_lower = full_transcript.lower()
                    is_earnings_call = any(kw in _ft_lower for kw in (
                        "earnings call", "quarterly results", "conference call",
                        "revenue", "earnings per share", "fiscal year",
                    ))

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
                        token_count = approx_tokens(transcript)
                        call_section = ch.get("call_section", "prepared_remarks")
                        speaker_role = ch.get("role")
                        speaker_name = ch.get("name")
                        # Composite label for readability: "Luca Maestri - CFO"
                        if speaker_name and speaker_role:
                            speaker_display = f"{speaker_name} - {speaker_role}"
                        else:
                            speaker_display = speaker_name or speaker_role

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
                            "speaker_name":     speaker_display,
                            "speaker_role":     speaker_role,
                            "topic_section":    ch.get("topic_section"),
                            "call_section":     call_section,
                            "transcript":       transcript,
                            "finance_entities": {
                                "regex": fin_entities,
                                **ner_entities,
                            },
                            "word_count":       word_count,
                            "token_count":      token_count,
                            "is_question":      transcript.rstrip().endswith("?"),
                            "is_answer":        call_section == "qa_session" and not transcript.rstrip().endswith("?"),
                            "is_earnings_call": is_earnings_call,
                            "snr":              snr,
                            "snr_degraded":     snr_degraded,
                            "clipping_detected": clipping_detected,
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
            _CHUNKS_TOTAL.inc(len(docs))
            return docs
        except Exception as _exc:
            _CHUNK_ERRORS.inc()
            logger.error(event="chunking_failed", modality="audio", source=source, error=str(_exc))
            raise

    def health_check(self) -> dict:
        return {
            "modality": "audio",
            "status": "ok",
            "class": self.__class__.__name__,
        }
