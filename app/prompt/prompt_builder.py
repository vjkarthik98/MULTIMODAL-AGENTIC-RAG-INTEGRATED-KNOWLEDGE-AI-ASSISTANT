import time
import unicodedata
from typing import Optional, Tuple

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# BUDGET RATIOS
_MEM_RATIO  = 0.20
_CTX_RATIO  = 0.55
_QUERY_MAX  = 0.15

# PROMPT INJECTION PATTERNS
_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "disregard the above",
    "forget everything",
    "you are now",
    "act as",
    "jailbreak",
]

# STRUCTURED KEYWORDS
_STRUCTURED_KEYWORDS = [
    "table", "row", "column", "page number",
    "which page", "toc", "section", "cell",
    "extract", "list all", "enumerate",
]

# MULTIMODAL KEYWORDS
_IMAGE_KEYWORDS  = {"image", "photo", "diagram", "figure", "chart", "screenshot", "picture"}
_AUDIO_KEYWORDS  = {"audio", "sound", "speech", "transcript", "recording", "voice"}
_VIDEO_KEYWORDS  = {"video", "clip", "footage", "scene", "frame"}

# CODE KEYWORDS
_CODE_KEYWORDS = {"code", "function", "class", "implement", "script", "snippet", "syntax"}


class PromptBuilder:

    def __init__(self) -> None:
        self.max_chars = settings.MAX_PROMPT_CHARS

    # CLEAN

    def _clean(self, text: str) -> str:
        text = unicodedata.normalize("NFC", str(text or ""))
        return " ".join(text.strip().split())

    # TRUNCATE

    def _truncate(self, text: str, limit: int) -> str:
        if not text:
            return ""
        return text[:max(limit, 0)]

    # INJECTION GUARD

    def _sanitize_query(self, query: str) -> str:
        lower = query.lower()
        for pattern in _INJECTION_PATTERNS:
            if pattern in lower:
                logger.warning(
                    event="prompt_injection_detected",
                    pattern=pattern,
                )
                query = query[:query.lower().find(pattern)].strip()
                break
        return query

    # DEDUP

    def _deduplicate(self, memory: str, context: str) -> Tuple[str, str]:
        if not memory or not context:
            return memory, context

        key = memory[:200].strip()
        if key and key in context:
            context = context.replace(key, "").strip()

        return memory, context

    # QUERY MODE DETECTION

    def _is_structured(self, query: str) -> bool:
        q = query.lower()
        return any(k in q for k in _STRUCTURED_KEYWORDS)

    def _is_code(self, query: str) -> bool:
        tokens = set(query.lower().split())
        return bool(tokens & _CODE_KEYWORDS)

    def _detect_modality(self, query: str, context: str) -> Optional[str]:
        combined = (query + " " + context).lower()
        tokens   = set(combined.split())

        if tokens & _IMAGE_KEYWORDS:
            return "image"
        if tokens & _AUDIO_KEYWORDS:
            return "audio"
        if tokens & _VIDEO_KEYWORDS:
            return "video"
        return None

    # SYSTEM PROMPT

    def _system(
        self,
        structured: bool,
        is_code: bool,
        modality: Optional[str],
    ) -> str:

        if structured:
            return (
                "You are a strict extraction system.\n"
                "- Use ONLY context\n"
                "- Return exact value\n"
                "- No explanation\n"
                "- If missing → I don't know\n\n"
            )

        if is_code:
            return (
                "You are a precise code assistant.\n"
                "- Use ONLY provided context\n"
                "- Return clean, working code\n"
                "- No hallucination\n"
                "- If unsure → say so\n\n"
            )

        if modality == "image":
            return (
                "You are an expert in visual reasoning.\n"
                "- Use ONLY provided context\n"
                "- Describe visual content accurately\n"
                "- No hallucination\n"
                "- If unsure → I don't know\n\n"
            )

        if modality == "audio":
            return (
                "You are an expert in audio and speech understanding.\n"
                "- Use ONLY provided transcripts and context\n"
                "- No hallucination\n"
                "- If unsure → I don't know\n\n"
            )

        if modality == "video":
            return (
                "You are an expert in video understanding.\n"
                "- Use ONLY provided frames and transcripts\n"
                "- No hallucination\n"
                "- If unsure → I don't know\n\n"
            )

        return (
            "You are a grounded assistant.\n"
            "- Use ONLY context\n"
            "- No hallucination\n"
            "- If unsure → I don't know\n"
            "- Be precise\n\n"
        )

    # OUTPUT FORMAT

    def _output_format(self, structured: bool, is_code: bool) -> str:
        if structured:
            return "OUTPUT:\n<exact answer>"

        if is_code:
            return "OUTPUT:\n```\n<code here>\n```"

        return (
            "FORMAT:\n"
            "Answer:\n<text>\n"
            "Confidence:\n<0-1>"
        )

    # MAIN

    def build_prompt(
        self,
        query: str,
        context: str,
        memory: str = "",
        session_id: str = "default",
    ) -> str:

        start = time.time()

        try:
            query   = self._clean(self._sanitize_query(query))
            context = self._clean(context)
            memory  = self._clean(memory)

            if not query:
                raise ValueError("EMPTY_QUERY")

            structured = self._is_structured(query)
            is_code    = self._is_code(query)
            modality   = self._detect_modality(query, context)

            memory, context = self._deduplicate(memory, context)

            # BUDGET ALLOCATION
            mem_budget   = int(self.max_chars * _MEM_RATIO)
            ctx_budget   = int(self.max_chars * _CTX_RATIO)
            query_budget = int(self.max_chars * _QUERY_MAX)

            memory  = self._truncate(memory,  mem_budget)
            context = self._truncate(context, ctx_budget)
            query   = self._truncate(query,   query_budget)

            system       = self._system(structured, is_code, modality)
            output_fmt   = self._output_format(structured, is_code)

            mem_block   = f"MEMORY:\n{memory}\n\n"   if memory   else ""
            ctx_block   = f"CONTEXT:\n{context}\n\n" if context  else ""
            query_block = (
                f"TASK:\n{query}\n\n"  if structured
                else f"QUERY:\n{query}\n\n"
            )

            prompt = system + mem_block + ctx_block + query_block + output_fmt

            # FINAL OVERFLOW GUARD
            if len(prompt) > self.max_chars:
                fixed   = system + query_block + output_fmt
                allowed = self.max_chars - len(fixed) - 20
                middle  = self._truncate(mem_block + ctx_block, allowed)
                prompt  = system + middle + query_block + output_fmt

                logger.warning(
                    event="prompt_truncated",
                    original_size=len(system + mem_block + ctx_block + query_block + output_fmt),
                    final_size=len(prompt),
                    session_id=session_id,
                )

            logger.debug(
                event="prompt_built",
                size=len(prompt),
                mem_chars=len(memory),
                ctx_chars=len(context),
                query_chars=len(query),
                structured=structured,
                is_code=is_code,
                modality=modality,
                latency=round(time.time() - start, 3),
                session_id=session_id,
            )

            return prompt

        except Exception as e:
            logger.error(
                event="prompt_build_failed",
                error=str(e),
                session_id=session_id,
            )
            raise