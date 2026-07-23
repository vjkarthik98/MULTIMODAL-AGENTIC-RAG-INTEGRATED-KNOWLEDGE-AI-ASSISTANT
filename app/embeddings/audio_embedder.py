"""audio_embedder.py — Finance-grade embedder for audio/transcript chunks."""

from __future__ import annotations

import re
from typing import Any

from prometheus_client import Counter

from app.core.config import settings
from app.embeddings.base_embedder import BaseEmbedder
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Spoken Fed-style rate-change phrasings carry NO digits ("half percentage
# point"), so a query asking "by how much did the Fed cut / 50 basis points"
# lexically and semantically matches the Q&A DISCUSSION of the cut (reporters
# who literally say "50 basis points") far better than the actual rate-cut
# ANNOUNCEMENT — which is the chunk that should rank first. Amplify each spoken
# form into its numeric equivalents at embed time so the announcement chunk is
# retrievable by number too. Amplification only affects the embedding vector;
# the stored transcript and the answer text are untouched.
_RATE_WORD_MAP = [
    (
        re.compile(r"\bthree(?:\s+|-)quarters?\s+of\s+a\s+percentage\s+point", re.I),
        "75 basis points 0.75 percentage point",
    ),
    (
        re.compile(r"\b(?:a\s+)?half(?:\s+of)?\s+a?\s*percentage\s+point", re.I),
        "50 basis points 0.50 percentage point half point rate cut",
    ),
    (re.compile(r"\bhalf\s+point\b", re.I), "50 basis points 0.50 percentage point"),
    (
        re.compile(r"\b(?:a\s+)?quarter(?:\s+of\s+a)?\s+percentage\s+point", re.I),
        "25 basis points 0.25 percentage point quarter point",
    ),
    (re.compile(r"\bquarter\s+point\b", re.I), "25 basis points 0.25 percentage point"),
    (
        re.compile(r"\b(?:a\s+)?full\s+percentage\s+point", re.I),
        "100 basis points 1.00 percentage point",
    ),
]


def _amplify_rate_expressions(text: str) -> list[str]:
    """Return numeric amplification tokens for any spoken rate expression found."""
    out: list[str] = []
    for pat, expansion in _RATE_WORD_MAP:
        if pat.search(text):
            out.append(expansion)
    return out


_EMBED_BUILT = Counter(
    "magik_audio_embed_text_built_total",
    "Embed texts successfully built for audio",
)
_EMBED_ERRORS = Counter(
    "magik_audio_embed_text_errors_total",
    "Errors building embed text for audio",
)


class AudioEmbedder(BaseEmbedder):
    """Embedder for audio chunks (earnings calls, analyst days, podcasts).

    Enrichment:
      [Speaker: {name}] [{role}] [{ts_start}s-{ts_end}s] [{call_section}] {text}
      + finance entity amplification at end

    Entity amplification appends tickers, dollar amounts, and company names
    so queries like "AAPL Q3 revenue" hit the right speaker segment.
    """

    def embed_documents(self, documents: list[Any], session_id: str = "default") -> list[Any]:
        results = super().embed_documents(documents, session_id=session_id)
        # FinBERT tone annotation is globally opt-in (extra GPU pass per batch,
        # settings.FINBERT_ENABLED defaults off since most modalities get no
        # retrieval benefit from it). Fed press conferences / earnings calls
        # are exactly the case FinBERT-tone was built for, so audio always
        # gets tone-tagged regardless of the global default.
        if not settings.FINBERT_ENABLED and results:
            self._annotate_finbert_tone(results)
        return results

    def _build_embed_text(self, doc: Any, cleaned_text: str) -> str:
        try:
            s = getattr(doc, "structure", {}) or {}

            header_parts: list[str] = []

            # Section prefix — bakes call context into the vector.
            # "[Q&A SESSION]" / "[PREPARED REMARKS]" from spec Phase 2.6.
            call_section = (s.get("call_section") or "").strip()
            if call_section:
                section_label = call_section.upper().replace("_", " ")
                header_parts.append(f"[{section_label}]")

            # Speaker identity — avoid "Name - Role - Role" when name already embeds role.
            name = (s.get("speaker_name") or s.get("speaker") or "").strip()
            role = (s.get("speaker_role") or "").strip()
            role_in_name = role and role.lower() in name.lower()
            if name and role and not role_in_name:
                header_parts.append(f"[Speaker: {name} - {role}]")
            elif name:
                header_parts.append(f"[Speaker: {name}]")
            elif role:
                header_parts.append(f"[Speaker: {role}]")

            # Timestamps at millisecond precision — spec Phase 2.6 critical field.
            ts_s = s.get("start_timestamp") or s.get("timestamp_start")
            ts_e = s.get("end_timestamp") or s.get("timestamp_end")
            if ts_s is not None and ts_e is not None:
                header_parts.append(f"[{float(ts_s):.3f}s-{float(ts_e):.3f}s]")
            elif ts_s is not None:
                header_parts.append(f"[{float(ts_s):.3f}s]")

            header = (" ".join(header_parts) + " ") if header_parts else ""

            # Finance entity amplification — append tickers/amounts/companies.
            fin_entities: dict = s.get("finance_entities") or {}
            entity_tokens: list[str] = []
            if isinstance(fin_entities, dict):
                for v in fin_entities.values():
                    if isinstance(v, list):
                        entity_tokens.extend(str(x) for x in v[:4])

            suffix = (f" [ENTITIES: {', '.join(entity_tokens)}]") if entity_tokens else ""

            # Numeric amplification for spoken rate-change phrasings — makes the
            # rate-cut ANNOUNCEMENT chunk ("half percentage point") retrievable
            # by queries phrased with digits ("50 basis points", "how much").
            rate_tokens = _amplify_rate_expressions(cleaned_text)
            rate_suffix = (f" [RATE: {', '.join(rate_tokens)}]") if rate_tokens else ""

            # Q&A section token for section-aware retrieval.
            qa_token = ""
            if s.get("is_question"):
                qa_token = " [analyst question]"
            elif s.get("is_answer"):
                qa_token = " [management answer]"

            result = f"{header}{cleaned_text}{qa_token}{rate_suffix}{suffix}"
            result = result[: settings.MAX_PROMPT_CHARS]
            logger.debug(event="embed_text_built", modality="audio", chars=len(result))
            _EMBED_BUILT.inc()
            return result
        except Exception as _exc:
            _EMBED_ERRORS.inc()
            logger.error(event="embed_text_build_failed", modality="audio", error=str(_exc))
            return cleaned_text  # safe fallback to unenriched text

    def health_check(self) -> dict:
        return {
            "modality": "audio",
            "status": "ok",
            "class": self.__class__.__name__,
        }
