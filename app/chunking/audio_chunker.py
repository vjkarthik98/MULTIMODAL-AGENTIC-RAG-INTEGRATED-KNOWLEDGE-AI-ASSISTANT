from __future__ import annotations

import math
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from prometheus_client import Counter

from app.chunking.av_shared import (
    _assemble_chunks,
    _map_speaker_roles,
    _merge_fragmented_hosts,
)
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


def diarize(audio_path: str) -> list[tuple[float, float, str]]:
    """Run pyannote speaker diarization. Returns (start, end, speaker) tuples.

    The pyannote/speaker-diarization-3.1 pipeline internally loads and runs
    three models in sequence (segmentation-3.0 for voice-activity/change
    detection, a speaker-embedding model for clustering, and the diarization
    pipeline itself which agglomeratively clusters those embeddings) —
    Pipeline.from_pretrained() pulls in all three, so a single get_diarizer()
    call already exercises the full pyannote stack.
    """
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
        segments: list[tuple[float, float, str]] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append((turn.start, turn.end, speaker))
        return sorted(segments, key=lambda x: x[0])
    except Exception as exc:
        logger.warning(event="diarization_failed", error=str(exc))
        return []


_ENTITY_KEYS = {"ORG": "companies", "PER": "persons", "LOC": "locations", "MISC": "misc"}


def extract_entities(text: str) -> dict[str, list[str]]:
    """Run dslim/bert-base-NER. Returns {companies, persons, locations, misc}."""
    result: dict[str, list[str]] = {v: [] for v in _ENTITY_KEYS.values()}
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
            word = ent.get("word", "").strip()
            if not word or word in seen:
                continue
            seen.add(word)
            result[_ENTITY_KEYS.get(label, "misc")].append(word)
        return result
    except Exception as exc:
        logger.warning(event="ner_failed", error=str(exc))
        return result


# Priming prompt — biases faster-whisper toward proper capitalization,
# punctuation, and the domain's proper nouns (Fed officials, news outlets,
# reporter self-introductions). Without it the model drifts to all-lowercase,
# unpunctuated text in the busier Q&A half of a press conference, which both
# breaks reporter self-intro detection (the name regex needs capitalized names)
# and makes cited answers read as a raw transcript dump.
_WHISPER_INITIAL_PROMPT = (
    "The following is a Federal Reserve press conference and financial earnings "
    "call, transcribed with correct capitalization and punctuation. Chair "
    "Jerome Powell delivers prepared remarks and answers questions. Reporters "
    "introduce themselves before asking, for example: \"Hi Chair Powell, Greg "
    "Robb from MarketWatch\" or \"Gina Smialek with the New York Times.\" Other "
    "outlets include the Wall Street Journal, Reuters, Bloomberg, CNBC, the "
    "Associated Press, Politico, and the Washington Post."
)


def _run_whisper(wav_path: str) -> list[dict]:
    """Transcribe with faster_whisper; returns list of word dicts."""
    try:
        from app.core.model_loader import model_loader as loader

        model = loader.get_whisper()
        segments, _ = model.transcribe(
            wav_path,
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=_WHISPER_INITIAL_PROMPT,
            beam_size=5,
        )
        words = []
        for seg in segments:
            if hasattr(seg, "words") and seg.words:
                for w in seg.words:
                    words.append({"word": w.word, "start": w.start, "end": w.end})
            else:
                # Segment-level fallback when word_timestamps not available.
                words.append(
                    {
                        "word": seg.text,
                        "start": seg.start,
                        "end": seg.end,
                    }
                )
        return words
    except Exception as exc:
        logger.warning(event="whisper_failed", error=str(exc))
        return []


def _transcribe_long_audio(wav_path: str, duration_sec: float) -> list[dict]:
    """Transcribe audio, splitting into AUDIO_CHUNK_DURATION_SEC segments first
    for anything longer than that.

    A single faster-whisper call over an hour-scale recording measurably
    degrades quality in the later portion of the file (dropped capitalization,
    garbled proper nouns) compared to transcribing the same audio region in
    isolation — confirmed by direct comparison on this pipeline's FOMC test
    file. Splitting into bounded segments and transcribing each independently
    (as the legacy ingest() path already did) avoids that drift; segments run
    concurrently across AUDIO_TRANSCRIPTION_WORKERS since CTranslate2 releases
    the GIL during CUDA ops.
    """
    # Cap transcription-segment length at 10 min. faster-whisper's output
    # quality (capitalization, punctuation, proper nouns) is most consistent on
    # ~10-min windows; the earlier 30-min windows drifted to unpunctuated
    # lowercase in their later portion (the FOMC Q&A section). Each segment is
    # re-primed with _WHISPER_INITIAL_PROMPT, so more/shorter segments means the
    # casing prompt takes effect more often, not less.
    _TRANSCRIBE_SEGMENT_SEC = 600
    if duration_sec <= 0 or duration_sec <= _TRANSCRIBE_SEGMENT_SEC:
        return _run_whisper(wav_path)

    from pydub import AudioSegment

    audio = AudioSegment.from_wav(wav_path)
    chunk_sec = _TRANSCRIBE_SEGMENT_SEC
    n_segments = math.ceil(duration_sec / chunk_sec)

    segment_paths: list[tuple[str, float]] = []
    for i in range(n_segments):
        start_ms = int(i * chunk_sec * 1000)
        end_ms = int(min((i + 1) * chunk_sec, duration_sec) * 1000)
        seg = audio[start_ms:end_ms]
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        seg.export(tmp.name, format="wav")
        tmp.close()
        segment_paths.append((tmp.name, i * chunk_sec))

    words: list[dict] = []
    try:
        with ThreadPoolExecutor(max_workers=settings.AUDIO_TRANSCRIPTION_WORKERS) as pool:
            futures = {pool.submit(_run_whisper, p): off for p, off in segment_paths}
            results: list[tuple[float, list[dict]]] = []
            for fut, off in futures.items():
                try:
                    seg_words = fut.result()
                    for w in seg_words:
                        w["start"] += off
                        w["end"] += off
                    results.append((off, seg_words))
                except Exception as exc:
                    logger.warning(
                        event="audio_segment_transcribe_failed", offset=off, error=str(exc)
                    )
        results.sort(key=lambda r: r[0])
        for _, seg_words in results:
            words.extend(seg_words)
    finally:
        for p, _off in segment_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    return words


class AudioChunker(BaseChunker):
    """Finance-grade chunker for audio files (earnings calls, press conferences, podcasts).

    Pipeline: Whisper transcription → pyannote diarization → speaker-role mapping
              → topic-aware chunking → NER entity extraction.
    """

    def chunk(
        self,
        extracts: list[RawExtract],
        meta: UniversalMetadata,
    ) -> list[IngestedDocument]:
        source = Path(meta.source_path).name or "unknown.mp3"
        surface = "audio_chunker"
        modality_var.set("audio")
        _t0 = time.time()
        logger.info(event="chunking_start", modality="audio", source=source, extracts=len(extracts))
        if not extracts:
            logger.warning(event="no_extracts_received", modality="audio", source=source)
            return []
        try:
            docs: list[IngestedDocument] = []

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
                    duration_sec = ext_extra.get("duration_seconds") or 0.0
                    words = _transcribe_long_audio(wav_path, duration_sec)

                    # Diarization (optional — skipped if model unavailable).
                    diarization: list[tuple[float, float, str]] = []
                    try:
                        diarization = diarize(wav_path)
                        diarization = _merge_fragmented_hosts(diarization)
                    except Exception:
                        pass

                    full_transcript = " ".join(w["word"] for w in words)
                    role_map = _map_speaker_roles(diarization, words)
                    # Audio-only finer chunking (video keeps av_shared defaults).
                    raw_chunks = _assemble_chunks(
                        words,
                        diarization,
                        role_map,
                        min_words=settings.AUDIO_CHUNK_MIN_WORDS,
                        max_words=settings.AUDIO_CHUNK_MAX_WORDS,
                    )

                    # Document-level earnings-call detection — checked once per extract.
                    _ft_lower = full_transcript.lower()
                    is_earnings_call = any(
                        kw in _ft_lower
                        for kw in (
                            "earnings call",
                            "quarterly results",
                            "conference call",
                            "revenue",
                            "earnings per share",
                            "fiscal year",
                        )
                    )

                    for chunk_idx, ch in enumerate(raw_chunks):
                        transcript = ch["transcript"]
                        if not transcript.strip():
                            continue

                        # NER entity extraction.
                        ner_entities: dict = {}
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
                            "chunk_hash_id": chunk_hash,
                            "source_file": source,
                            "chunk_index": chunk_idx,
                            "start_timestamp": round(ch["start"], 3),
                            "end_timestamp": round(ch["end"], 3),
                            "duration_seconds": round(duration, 3),
                            "speaker_label": ch["speaker"],
                            "speaker_name": speaker_display,
                            "speaker_role": speaker_role,
                            "topic_section": ch.get("topic_section"),
                            "call_section": call_section,
                            "transcript": transcript,
                            "finance_entities": {
                                "regex": fin_entities,
                                **ner_entities,
                            },
                            "word_count": word_count,
                            "token_count": token_count,
                            "is_question": transcript.rstrip().endswith("?"),
                            "is_answer": call_section == "qa_session"
                            and not transcript.rstrip().endswith("?"),
                            "is_earnings_call": is_earnings_call,
                            "snr": snr,
                            "snr_degraded": snr_degraded,
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
