"""av_shared.py

Diarization/speaker-role/chunk-assembly helpers shared by audio_chunker.py and
video_chunker.py (video re-uses the audio pipeline for its own transcript).

Moved out of audio_chunker.py during the 2026-07 security/architecture audit:
video_chunker.py was importing these directly from audio_chunker.py, which
violates the per-modality isolation contract (a change to audio_chunker.py
could silently break video — this exact coupling previously caused a hard
crash in video ingestion via _map_speaker_roles). This module is the shared,
modality-neutral home for the pure text/diarization-tuple logic; anything
that touches audio-specific I/O or model loading (diarize(), extract_entities(),
_run_whisper(), AudioChunker itself) stays in audio_chunker.py.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from app.utils.logger import get_logger

logger = get_logger(__name__)


def _merge_fragmented_hosts(
    diarization: List[Tuple[float, float, str]],
) -> List[Tuple[float, float, str]]:
    """Collapse diarization labels that are almost certainly the same speaker
    but got split into multiple pseudo-identities by pyannote's clustering.

    This is a known failure mode on hour-scale, single-dominant-speaker audio
    (earnings calls, press conferences): a continuous host/chair voice
    periodically drifts in the embedding space over 30-45+ minutes and gets
    re-clustered under a new label, while each one-off questioner still gets
    a single short, correctly-isolated label. Any label whose total talk time
    is a large share of the recording (or a large multiple of the median
    speaker's talk time) is treated as a "host candidate"; when 2+ such
    candidates exist they are merged into the single label with the most
    total duration. Real multi-host recordings with genuinely comparable
    talk time are unaffected as long as there's no evidence of fragmentation
    (only 2+ candidates plus this share test trigger a merge).
    """
    if not diarization:
        return diarization

    total_dur: Dict[str, float] = defaultdict(float)
    for start, end, label in diarization:
        total_dur[label] += (end - start)

    file_duration = max((e for _, e, _ in diarization), default=0.0)
    if file_duration <= 0 or len(total_dur) < 2:
        return diarization

    durations = sorted(total_dur.values())
    median_dur = durations[len(durations) // 2]

    host_candidates = [
        label for label, dur in total_dur.items()
        if dur >= 0.10 * file_duration or (median_dur > 0 and dur >= 3 * median_dur)
    ]
    logger.info(
        event="audio_speaker_fragment_check",
        n_labels=len(total_dur),
        file_duration_sec=round(file_duration, 1),
        median_dur_sec=round(median_dur, 1),
        host_candidates=host_candidates,
        top5=sorted(({k: round(v, 1) for k, v in total_dur.items()}).items(), key=lambda kv: -kv[1])[:5],
    )
    if len(host_candidates) < 2:
        return diarization

    canonical = max(host_candidates, key=lambda l: total_dur[l])
    remap = {label: canonical for label in host_candidates if label != canonical}
    if not remap:
        return diarization

    logger.info(
        event="audio_speaker_fragments_merged",
        canonical=canonical,
        merged=list(remap.keys()),
        canonical_duration_sec=round(total_dur[canonical], 1),
    )
    return [(s, e, remap.get(lbl, lbl)) for s, e, lbl in diarization]


def _label_at_time(
    t: float,
    diar_starts: List[float],
    diarization: List[Tuple[float, float, str]],
) -> str:
    """Speaker label at time t, snapping to the NEAREST turn across gaps.

    pyannote turns rarely cover 100% of the timeline (breaths, micro-pauses,
    detection gaps between turns of the SAME speaker) — any word whose
    timestamp falls in such a gap needs the label of whichever turn is
    temporally closest, not a hardcoded fallback. A fixed fallback (e.g.
    always "SPEAKER_00", or always the first turn) systematically
    misattributes every gap-word for the whole recording to one label,
    fabricating a large phantom "extra speaker" out of a single continuous
    speaker's own natural pauses.
    """
    import bisect

    if not diarization:
        return "SPEAKER_00"
    idx = bisect.bisect_right(diar_starts, t) - 1
    if idx < 0:
        return diarization[0][2]
    start, end, label = diarization[idx]
    if t <= end:
        return label
    if idx + 1 < len(diarization):
        next_start, _next_end, next_label = diarization[idx + 1]
        if (next_start - t) < (t - end):
            return next_label
    return label


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

# News outlets a reporter names when self-introducing at a press conference.
_NEWS_OUTLETS = (
    r"(?:MarketWatch|New\s+York\s+Times|NYT|Wall\s+Street\s+Journal|WSJ|Reuters|"
    r"Bloomberg|CNBC|Fox(?:\s+Business)?|Associated\s+Press|\bAP\b|Financial\s+Times|"
    r"\bFT\b|Politico|Axios|Yahoo|Barron'?s|Washington\s+Post|Economist|NPR|CBS|"
    r"ABC|NBC|The\s+Hill|American\s+Banker|Nikkei|Semafor)"
)

_ROLE_KEYWORDS = {
    "Fed Chair":  re.compile(r"\b(chairman|chairwoman|chair powell|chair bernanke|chair yellen|chair jerome|federal reserve chair)\b", re.I),
    "CEO":        re.compile(r"\bceo\b|\bchief executive\b", re.I),
    "CFO":        re.compile(r"\bcfo\b|\bchief financial\b", re.I),
    # NOTE: "President" is intentionally NOT a role keyword. In FOMC transcripts
    # "president" almost always refers to a regional-Fed president MENTIONED in
    # speech ("New York Fed President John Williams"), not the speaker's own
    # role — matching it mislabelled reporters as "President". Company/earnings-
    # call presidents are named via the self-intro/role signals instead.
    "Analyst":    re.compile(r"\banalyst\b|\bgoldman\b|\bmorgan\b|\bjp morgan\b|\bciti\b|\bbarclays\b|\bubs\b|\bdeutsche bank\b", re.I),
    "Reporter":   re.compile(r"\b(reporter|journalist|correspondent|from (?:the |)(?:wall street|new york times?|reuters|bloomberg|wsj|cnbc|fox|associated press|financial times|marketwatch|politico|washington post))\b", re.I),
    "Operator":   re.compile(r"\boperator\b", re.I),
    "COO":        re.compile(r"\bcoo\b|\bchief operating\b", re.I),
}

_FILLER = re.compile(r"\b(um|uh|er|ah|you know|i mean|like|so|basically|essentially)\b", re.I)

# Reporter self-introduction: "<Name> from/with <known news outlet>", e.g.
# "Greg Robb from MarketWatch.com", "Gina Smialek with the New York Times".
# Anchored on a KNOWN outlet (not a generic capitalized word) so it stays
# high-precision, and CASE-INSENSITIVE so it still fires on the all-lowercase
# ASR that faster-whisper produces in the busy Q&A section. Group 1 is the
# 1-3 word reporter name immediately preceding "from/with <outlet>".
_SELF_INTRO_RE = re.compile(
    r"\b([A-Za-z][A-Za-z'’]+(?:\s+[A-Za-z][A-Za-z'’]+){0,2})\s+(?:from|with)\s+"
    r"(?:the\s+)?" + _NEWS_OUTLETS,
    re.IGNORECASE,
)
# Words that are never part of a reporter's name — guards against capturing a
# trailing sentence fragment ("...took all of that data with Reuters") as the
# name. Any match whose name tokens are all stopwords is rejected.
_INTRO_NAME_STOPWORDS = {
    "the", "that", "this", "data", "it", "them", "us", "questions", "question",
    "you", "chair", "here", "and", "our", "your", "is", "was", "hi", "hello",
    "thanks", "thank", "yeah", "so", "just", "now", "again", "well", "of", "all",
    "into", "account", "took", "take", "taken", "look", "looked", "go", "going",
    "get", "got", "sort", "kind", "lot", "some", "any", "over", "out", "up",
    "down", "back", "then", "than", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "on", "in", "at", "to",
    "for", "with", "from", "we", "i", "he", "she", "they", "my", "his", "her",
    "along", "shared", "spoke", "talked", "along", "put",
}


def _clean_intro_name(raw: str) -> Optional[str]:
    """Return a plausible 1-2 token reporter name from a self-intro capture,
    or None if it looks like a sentence fragment rather than a name."""
    toks = [w.strip(".,'’") for w in raw.split()
            if w.strip(".,'’").lower() not in _INTRO_NAME_STOPWORDS]
    toks = [w for w in toks if w.isalpha()]
    if not toks:
        return None
    name = " ".join(toks[-2:])  # keep the last 1-2 tokens (first + surname)
    surname = name.split()[-1]
    if len(surname) < 3:        # a real surname is ≥3 letters
        return None
    return name.title()

# Vocative address opening a turn — "Chair Powell, ..." / "Mr. Chairman, ...".
# Case-insensitive and tolerant of lowercase ASR output (unlike a self-intro,
# this is a short fixed phrase so a loose match stays low-risk).
_CHAIR_ADDRESS_RE = re.compile(
    r"\b(?:chair|chairman|chairwoman)\s+([a-z]+)\b|\bmr\.?\s+chairman\b",
    re.IGNORECASE,
)
_CHAIR_ADDRESS_STOPWORDS = {
    "of", "the", "and", "is", "was", "said", "noted", "board", "person", "here",
}
_TURN_LEAD_WORDS = 20  # leading words of a turn counted as "near the start"


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


_DECIMAL_SPACING_RE = re.compile(r"(\d)\s+\.\s*(\d)")
_HYPHEN_SPACING_RE = re.compile(r"(\w)\s+-(\w)")


def _fix_asr_spacing(text: str) -> str:
    """Collapse faster-whisper's spurious space before decimal points/hyphens.

    This model consistently emits "2 .2 percent" instead of "2.2 percent" and
    "dual -mandate" instead of "dual-mandate" for this audio. Left unfixed,
    "2.2" (correctly formatted by the LLM in its answer) never string-matches
    "2 .2" (as stored in the retrieved context), so the numeric-faithfulness
    guard in reasoning_engine.py flags a real, correctly-cited number as
    "unsupported" and discards an otherwise-faithful answer.
    """
    text = _DECIMAL_SPACING_RE.sub(r"\1.\2", text)
    text = _HYPHEN_SPACING_RE.sub(r"\1-\2", text)
    return text


def _remove_fillers(text: str) -> str:
    text = _fix_asr_spacing(text)
    return re.sub(r"\s+", " ", _FILLER.sub("", text)).strip()


def _map_speaker_roles(
    diarization: List[Tuple[float, float, str]],
    words: List[Dict],
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """Map SPEAKER_XX labels to (role, name) using turn-anchored context.

    Binds names using exact word timestamps rather than proportional
    character-position estimation over the joined transcript — on a long
    recording that estimate drifts far enough from real time that a name
    mentioned in one turn gets bound to whichever unrelated turn the linear
    estimate happens to land on. Two independent, turn-anchored signals:
      1. Self-introduction — "<Name> from/with <Outlet>" near the start of a
         speaker's own turn (reporters/analysts stating their affiliation).
      2. Vocative address — a turn opens by addressing "Chair <Surname>" /
         "Mr. Chairman"; the addressee is whoever speaks in the immediately
         following (different-label) turn.
    Falls back to per-speaker role-keyword detection when no name is bound.
    """
    role_name_map: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
    if not diarization or not words:
        return role_name_map

    _diar_starts = [seg[0] for seg in diarization]

    def _label_at(t: float) -> str:
        return _label_at_time(t, _diar_starts, diarization)

    # Group words into contiguous per-speaker turns using the real diarization
    # boundaries (ground truth), independent of chunk-assembly granularity.
    turns: List[Dict] = []
    for w in words:
        lbl = _label_at(w["start"])
        if not turns or turns[-1]["label"] != lbl:
            turns.append({"label": lbl, "words": [w["word"]]})
        else:
            turns[-1]["words"].append(w["word"])

    label_to_name: Dict[str, str] = {}
    label_to_role: Dict[str, str] = {}

    for i, turn in enumerate(turns):
        # Scan the first ~40 words of the turn (not the whole turn): a reporter's
        # self-intro sits at the very start, and diarization sometimes prepends
        # a few words of the previous (host) speaker — but scanning deep into a
        # long chair turn risks matching an incidental "...that data with
        # Reuters". 40 words comfortably covers a prepended tail + the intro.
        lead_text = " ".join(turn["words"][:40])

        # Signal 1: reporter self-introduction ("<Name> from/with <outlet>").
        # Only a reporter says "<name> from <outlet>", so binding it to this
        # turn's label can't mis-name the chair/host.
        if turn["label"] not in label_to_name:
            m = _SELF_INTRO_RE.search(lead_text)
            if m:
                name = _clean_intro_name(m.group(1))
                if name:
                    label_to_name[turn["label"]] = name
                    label_to_role[turn["label"]] = "Reporter"

        lead_text = " ".join(turn["words"][:_TURN_LEAD_WORDS])
        # Signal 2: vocative address — bind the addressee to the NEXT turn.
        # A specific surname ("Chair Powell") is allowed to upgrade a prior
        # generic binding ("Chair", from a bare "Mr. Chairman" address earlier
        # in the recording) — first-match-wins would otherwise let whichever
        # phrasing happens to occur first permanently block the better one.
        m = _CHAIR_ADDRESS_RE.search(lead_text)
        if m and i + 1 < len(turns):
            next_label = turns[i + 1]["label"]
            surname_raw = (m.group(1) or "").strip().lower()
            already_specific = label_to_name.get(next_label, "") not in ("", "Chair")
            if next_label != turn["label"] and not already_specific:
                if surname_raw and surname_raw not in _CHAIR_ADDRESS_STOPWORDS:
                    label_to_name[next_label] = f"Chair {surname_raw.title()}"
                    label_to_role[next_label] = "Federal Reserve Chair"
                elif not surname_raw and next_label not in label_to_name:
                    label_to_name[next_label] = "Chair"
                    label_to_role[next_label] = "Federal Reserve Chair"

    # --- Assign role and name to each unique speaker label ---
    ordered_labels = list(dict.fromkeys(t["label"] for t in turns))
    speaker_text: Dict[str, str] = defaultdict(str)
    for t in turns:
        speaker_text[t["label"]] += " " + " ".join(t["words"][:200])

    for label in ordered_labels:
        name = label_to_name.get(label)
        role = label_to_role.get(label) or _detect_role(speaker_text.get(label, "")[:1500])
        role_name_map[label] = (role, name)

    return role_name_map


_OVERLAP_MAX_WORDS = 30


def _last_sentence(text: str) -> str:
    """Return the last complete sentence from text for overlap seeding.

    Bounded fallback: when a chunk has no sentence-ending punctuation at all
    (spoken audio frequently lacks terminal punctuation on disfluent stretches),
    the naive split returns the WHOLE chunk as "one sentence". Feeding that
    back in as the next chunk's overlap_seed compounds every subsequent chunk
    — each one swallows all of its predecessors' text — producing chunks that
    balloon past 1000 words and read as near-duplicates of each other. Cap the
    fallback to the last _OVERLAP_MAX_WORDS words so overlap size stays
    bounded regardless of punctuation.
    """
    text = text.strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    last = sentences[-1] if sentences else ""
    words = last.split()
    if len(words) > _OVERLAP_MAX_WORDS:
        last = " ".join(words[-_OVERLAP_MAX_WORDS:])
    return last


def _assemble_chunks(
    words: List[Dict],
    diarization: List[Tuple[float, float, str]],
    role_map: Dict[str, Tuple[Optional[str], Optional[str]]],
    min_words: int = _MIN_WORDS,
    max_words: int = _MAX_WORDS,
) -> List[Dict]:
    """Group words into speaker-aware, topic-bounded chunks with 1-sentence overlap.

    Hard boundaries: speaker change (≥ min_words buffered), topic transition phrase,
                     exceeding max_words.
    Soft boundary: nearest sentence end to the target word count.
    Overlap: last sentence of the previous chunk is prepended to each new chunk so
             a financial statement split across a boundary remains retrievable.

    `min_words`/`max_words` default to the module constants (75/225) so existing
    callers — notably the VIDEO chunker — are unchanged. The AUDIO chunker passes
    finer sizes so a specific spoken fact lands in a focused, retrievable chunk
    instead of being buried in a ~1.5-minute block of generic prose.
    """
    if not words:
        return []

    # Pre-extract sorted diarization start times for O(log s) binary search.
    _diar_starts: List[float] = [seg[0] for seg in diarization]

    def speaker_at(t: float) -> str:
        return _label_at_time(t, _diar_starts, diarization)

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

        over_max = len(buf_words) >= max_words
        speaker_changed = current_speaker != buf_speaker and len(buf_words) >= min_words
        topic_hit = bool(_TOPIC_TRANSITIONS.search(word_text)) and len(buf_words) >= min_words

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
