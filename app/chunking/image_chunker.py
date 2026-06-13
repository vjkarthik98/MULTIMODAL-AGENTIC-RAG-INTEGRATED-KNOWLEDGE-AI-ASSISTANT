"""image_chunker.py

Finance-grade Image chunker + image captioning utilities.

All ML inference (BLIP2, TrOCR, SigLIP, BLIP1 fallback) lives here because
captioning is a chunking concern — it transforms raw image bytes from a
RawExtract into semantic text for IngestedDocuments.

Ingestors (image_ingest.py, video_ingest.py) call classify_and_caption /
generate_caption from here via a lazy import.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps

from app.chunking.base_chunker import BaseChunker
from app.chunking.finance_numbers import deterministic_chunk_id, extract_finance_entities
from app.core.config import settings
from app.ingestion.schema import EmptyContentError, IngestedDocument, RawExtract, UniversalMetadata
from app.utils.logger import get_logger

try:
    import torch
except ImportError:
    torch = None

logger = get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# OCR-CAPTION MISMATCH
# ══════════════════════════════════════════════════════════════════════════════

_NUMBER_RE = re.compile(
    r"[$€£]\d[\d,.]*[BMKbmk]?|\d[\d,.]*\s?%|\d[\d,.]*[xX]|\d[\d,.]*\s?bps",
    re.IGNORECASE,
)
_MISMATCH_THRESHOLD = 0.20


def _numbers_in(text: str) -> set:
    return set(re.sub(r"[,\s]", "", n.lower()) for n in _NUMBER_RE.findall(text))


def _check_mismatch(ocr_text: str, caption: str) -> bool:
    ocr_nums = _numbers_in(ocr_text)
    if not ocr_nums:
        return False
    cap_nums = _numbers_in(caption)
    missing = ocr_nums - cap_nums
    return len(missing) / len(ocr_nums) > _MISMATCH_THRESHOLD


# ══════════════════════════════════════════════════════════════════════════════
# FINANCE IMAGE TYPE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

_IMAGE_TYPES = (
    "bar_chart", "line_chart", "pie_chart", "candlestick",
    "table_image", "org_chart", "flow_diagram", "infographic",
)

_FIN_NUMBER_RE = re.compile(
    r'(?:'
    r'\$[\d,]+\.?\d*\s*[BMKTbmkt]?'
    r'|[\d,]+\.?\d*\s*%'
    r'|[\d,]+\.?\d*\s*[xX]'
    r'|[\d,]+\.?\d*\s*bps'
    r')',
    re.IGNORECASE,
)


def _classify_image_type(image: "Image.Image", ocr_text: str = "") -> str:
    """Heuristic image type classification for finance images."""
    import numpy as np

    if ocr_text:
        nums = _FIN_NUMBER_RE.findall(ocr_text)
        lines_with_nums = [
            line for line in ocr_text.split("\n")
            if len(re.findall(r'\d', line)) >= 3
        ]
        if len(nums) >= 5 and len(lines_with_nums) >= 3:
            return "table_image"

    try:
        w, h = image.size
        arr = np.array(image.convert("RGB"))
        r_ch = arr[:, :, 0].astype(float)
        g_ch = arr[:, :, 1].astype(float)
        b_ch = arr[:, :, 2].astype(float)
        dark_pixels = float((b_ch < 30).mean())
        red_green_contrast = float(abs(r_ch.mean() - g_ch.mean()))
        if dark_pixels > 0.3 and red_green_contrast > 20:
            return "candlestick"
        if 0.8 <= w / max(h, 1) <= 1.2:
            white_pixels = float(((arr > 230).all(axis=2)).mean())
            if white_pixels < 0.4:
                return "pie_chart"
        col_means = arr.mean(axis=0).mean(axis=1)
        row_means = arr.mean(axis=1).mean(axis=1)
        col_var = float(col_means.var())
        row_var = float(row_means.var())
        if col_var > 800:
            return "bar_chart"
        if row_var > 800 and h > w:
            return "bar_chart"
        white_pct = float(((arr > 220).all(axis=2)).mean())
        if white_pct > 0.6 and col_var > 200:
            return "line_chart"
        if white_pct > 0.75:
            return "org_chart" if h > w * 1.2 else "flow_diagram"
    except Exception:
        pass

    return "infographic"


def _finance_caption_prompt(image_type: str) -> str:
    _PROMPTS = {
        "bar_chart":    "a financial bar chart showing",
        "line_chart":   "a financial line chart showing",
        "pie_chart":    "a pie chart showing portfolio or segment allocation",
        "candlestick":  "a candlestick stock price chart showing",
        "table_image":  "a financial table with exact numeric values showing",
        "org_chart":    "a corporate structure or organizational chart showing",
        "flow_diagram": "a business process or financial flow diagram showing",
        "infographic":  "a financial infographic or dashboard showing",
    }
    return _PROMPTS.get(image_type, "a financial document showing")


# ══════════════════════════════════════════════════════════════════════════════
# CAPTION QUALITY + VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

_WEAK_PREFIXES = [
    "a blurry image of", "a close up of", "an image of", "a picture of",
    "a photo of", "photo of", "image of", "this is a", "this is an",
    "a view of", "a shot of",
]

_CAPTION_MAX_WORDS = 50
_CAPTION_MIN_WORDS = 5
_CAPTION_MIN_CHARS = 10
_SOLID_COLOR_VARIANCE_THRESHOLD = 5.0


def _cache_key(image_path: str) -> str:
    return hashlib.sha256(image_path.encode("utf-8")).hexdigest()


def _caption_cache_get(key: str) -> Optional[str]:
    try:
        from app.core.infra_registry import infra
        mem = infra.get_memory()
        if mem:
            val = mem.cache_get(f"caption:{key}")
            return val if isinstance(val, str) else None
    except Exception:
        pass
    return None


def _caption_cache_set(key: str, caption: str) -> None:
    try:
        from app.core.infra_registry import infra
        mem = infra.get_memory()
        if mem:
            mem.cache_set(f"caption:{key}", caption, ttl=settings.REDIS_EMBEDDING_CACHE_TTL)
    except Exception:
        pass


def _remove_repetition(text: str) -> str:
    words = text.split()
    if len(words) < 6:
        return text
    half = len(words) // 2
    first = " ".join(words[:half])
    second = " ".join(words[half:])
    if first.strip().lower() == second.strip().lower():
        return first.strip()
    for n in (1, 2, 3, 4):
        if len(words) < n * 3:
            continue
        ngram = tuple(words[:n])
        run = 1
        i = n
        while i + n <= len(words):
            if tuple(words[i:i + n]) == ngram:
                run += 1
                i += n
            else:
                break
        if run >= 3:
            return " ".join(words[:n]).strip()
    return text


def _sanitize_caption(text: str) -> str:
    try:
        from app.guardrails.input_guard import sanitize as _guard_sanitize
        return _guard_sanitize(text, surface="frame_captioner")
    except Exception:
        return text


def _clean_caption(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    text = unicodedata.normalize("NFC", text)
    lower = text.lower()
    for pattern in _WEAK_PREFIXES:
        if lower.startswith(pattern):
            text = text[len(pattern):].strip()
            lower = text.lower()
            break
    if "\x00" in text:
        text = text.replace("\x00", "")
    text = _sanitize_caption(text)
    text = _remove_repetition(text)
    words = text.split()
    if len(words) < _CAPTION_MIN_WORDS:
        return None
    text = " ".join(words[:_CAPTION_MAX_WORDS])
    text = text[0].upper() + text[1:] if text else text
    if len(text) < _CAPTION_MIN_CHARS:
        return None
    return text


def _caption_confidence(caption: str) -> float:
    words = caption.split()
    if not words:
        return 0.0
    unique = set(w.lower() for w in words)
    diversity = len(unique) / len(words)
    length_score = min(len(caption) / 100.0, 1.0)
    return round((diversity + length_score) / 2.0, 3)


def _is_solid_color(image: "Image.Image") -> bool:
    try:
        import numpy as np
        arr = np.array(image.convert("RGB"), dtype=float)
        return float(arr.var()) < _SOLID_COLOR_VARIANCE_THRESHOLD
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE LOADING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _load_image(image_path: str) -> "Image.Image":
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"IMAGE_NOT_FOUND: {image_path}")
    if path.stat().st_size == 0:
        raise EmptyContentError(f"EMPTY_IMAGE_FILE: {image_path}")
    if path.stat().st_size > settings.MAX_FILE_SIZE_IMAGE:
        raise ValueError(
            f"IMAGE_TOO_LARGE: {path.stat().st_size} bytes exceeds {settings.MAX_FILE_SIZE_IMAGE}"
        )
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            if getattr(img, "is_animated", False):
                img.seek(0)
            if img.mode == "RGBA":
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            if w == 0 or h == 0:
                raise ValueError("INVALID_IMAGE_DIMENSIONS: zero dimension")
            if w < 32 or h < 32:
                raise ValueError(f"IMAGE_TOO_SMALL: {w}x{h}")
            total_mp = (w * h) / 1_000_000
            if total_mp > settings.MAX_IMAGE_SIZE_MP:
                scale = (settings.MAX_IMAGE_SIZE_MP * 1_000_000 / (w * h)) ** 0.5
                new_w = max(int(w * scale), 32)
                new_h = max(int(h * scale), 32)
                img = img.resize((new_w, new_h), Image.LANCZOS)
            elif max(w, h) > settings.MAX_IMAGE_DIM:
                img.thumbnail((settings.MAX_IMAGE_DIM, settings.MAX_IMAGE_DIM), Image.LANCZOS)
            return img.copy()
    except (FileNotFoundError, EmptyContentError, ValueError):
        raise
    except Exception as exc:
        raise ValueError(f"IMAGE_LOAD_FAILED: {exc}") from exc


def _generate_thumbnail(image: "Image.Image", output_path: Path) -> Optional[str]:
    try:
        thumb = image.copy()
        thumb.thumbnail((settings.THUMBNAIL_WIDTH, settings.THUMBNAIL_HEIGHT), Image.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        thumb.save(str(output_path), "JPEG", quality=85)
        return str(output_path)
    except Exception as exc:
        logger.warning(event="thumbnail_generation_failed", error=str(exc))
        return None


def _extract_tiff_frames(image_path: str) -> List["Image.Image"]:
    frames: List["Image.Image"] = []
    try:
        with Image.open(image_path) as img:
            if not hasattr(img, "n_frames") or img.n_frames <= 1:
                return []
            for i in range(img.n_frames):
                img.seek(i)
                frames.append(img.copy().convert("RGB"))
    except Exception as exc:
        logger.warning(event="tiff_frame_extraction_failed", error=str(exc))
    return frames


def _rasterize_svg(svg_path: str) -> Optional["Image.Image"]:
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(url=svg_path)
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")
    except ImportError:
        logger.warning(event="cairosvg_not_installed", hint="pip install cairosvg")
        return None
    except Exception as exc:
        logger.warning(event="svg_rasterize_failed", error=str(exc))
        return None


def _build_quality_metadata(
    caption: str, confidence: float, is_solid: bool,
    infer_ms: float, source: str,
) -> Dict[str, Any]:
    return {
        "caption_confidence": confidence,
        "caption_word_count": len(caption.split()),
        "caption_char_count": len(caption),
        "is_solid_color":     is_solid,
        "inference_ms":       infer_ms,
        "caption_source":     source,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

_BLIP2_FINANCE_PROMPT = (
    "You are a financial analyst describing an image for a retrieval system. "
    "Describe: 1) Image type (bar chart, line chart, table, infographic, etc.) "
    "2) Exact title if visible "
    "3) All data series with EXACT numbers read from labels "
    "4) Time period covered "
    "5) Axis labels and units "
    "6) Key trends or anomalies. "
    "Be precise about numbers — do not round or paraphrase."
)

_SIGLIP_IMAGE_TYPES = [
    "bar_chart", "line_chart", "pie_chart", "candlestick_chart",
    "table_image", "org_chart", "flow_diagram", "infographic",
    "scanned_document", "dashboard_screenshot",
]


def blip2_caption(image: "Image.Image", prompt: Optional[str] = None) -> str:
    """Generate a finance-aware caption using BLIP2. Returns '' on failure."""
    try:
        from app.core.model_loader import model_loader
        processor, model, device = model_loader.get_blip2()
    except Exception as exc:
        logger.warning(event="blip2_unavailable", error=str(exc))
        return ""
    try:
        import torch as _torch
        text_prompt = prompt or _BLIP2_FINANCE_PROMPT
        inputs = processor(image, text=text_prompt, return_tensors="pt").to(device)
        with _torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=settings.BLIP2_MAX_TOKENS)
        result = processor.decode(out[0], skip_special_tokens=True).strip()
        if result.startswith(text_prompt):
            result = result[len(text_prompt):].strip()
        return result
    except Exception as exc:
        logger.warning(event="blip2_caption_failed", error=str(exc))
        return ""


def blip2_classify_image_type(image: "Image.Image", ocr_text: str = "") -> str:
    """Zero-shot image type classification via SigLIP; falls back to heuristic."""
    try:
        from app.core.model_loader import model_loader
        import torch as _torch
        processor, clip_model, device = model_loader.get_siglip()
        inputs = processor(
            text=_SIGLIP_IMAGE_TYPES, images=image,
            return_tensors="pt", padding=True,
        ).to(device)
        with _torch.no_grad():
            outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1)[0]
        return _SIGLIP_IMAGE_TYPES[probs.argmax().item()]
    except Exception:
        pass
    return _classify_image_type(image, ocr_text)


def ocr(image: "Image.Image") -> str:
    """Run TrOCR large-printed on a PIL Image. Returns '' on failure."""
    try:
        from app.core.model_loader import model_loader
        processor, model, device = model_loader.get_trocr()
    except Exception as exc:
        logger.warning(event="trocr_unavailable", error=str(exc))
        return ""
    try:
        import torch as _torch
        img = image.convert("RGB")
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
        with _torch.no_grad():
            generated_ids = model.generate(pixel_values)
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    except Exception as exc:
        logger.warning(event="trocr_ocr_failed", error=str(exc))
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# BLIP CAPTION  (BLIP2 → BLIP1 fallback; calls blip2_caption above — no cross-layer import)
# ══════════════════════════════════════════════════════════════════════════════

def _blip_caption(
    image: "Image.Image",
    session_id: str,
    text_prompt: Optional[str] = None,
) -> Optional[str]:
    if torch is None:
        logger.warning(event="torch_unavailable_blip_skipped", session_id=session_id)
        return None

    # BLIP2 path (preferred)
    try:
        result = blip2_caption(image, prompt=text_prompt)
        if result:
            logger.debug(event="blip2_caption_used", session_id=session_id)
            return result
    except Exception as exc:
        logger.debug(event="blip2_caption_skipped", error=str(exc))

    # BLIP1 fallback
    try:
        from app.core.model_loader import model_loader
        processor, model, device = model_loader.get_blip()
        if text_prompt:
            inputs = processor(image, text=text_prompt, return_tensors="pt").to(device)
        else:
            inputs = processor(image, return_tensors="pt").to(device)
        t_infer = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=settings.BLIP_MAX_TOKENS,
                num_beams=settings.BLIP_NUM_BEAMS,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
            )
        infer_latency_ms = round((time.time() - t_infer) * 1000, 1)
        if infer_latency_ms > settings.LATENCY_TARGET_IMAGE_MS:
            logger.warning(
                event="blip_latency_exceeded",
                latency_ms=infer_latency_ms,
                target_ms=settings.LATENCY_TARGET_IMAGE_MS,
                session_id=session_id,
            )
        raw = processor.decode(output[0], skip_special_tokens=True)
        return raw.strip() if raw else None
    except Exception as exc:
        logger.warning(event="blip_caption_failed", error=str(exc), session_id=session_id)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC CAPTION API  (called by image_ingest.py and video_ingest.py via lazy import)
# ══════════════════════════════════════════════════════════════════════════════

def generate_caption(
    image_path: str,
    session_id: str,
    use_cache: bool = True,
) -> Optional[str]:
    """Generate a plain BLIP caption for an image file."""
    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    start = time.time()
    path  = Path(image_path)

    try:
        logger.debug(event="caption_start", image=path.name, session_id=session_id)

        if path.suffix.lower() == ".svg":
            image = _rasterize_svg(image_path)
            if image is None:
                logger.warning(event="svg_rasterize_failed_no_caption", image=path.name)
                return None
        else:
            image = _load_image(image_path)

        is_solid = _is_solid_color(image)
        if is_solid:
            return "Solid color or blank image frame."

        if path.suffix.lower() in (".tiff", ".tif"):
            tiff_frames = _extract_tiff_frames(image_path)
            if tiff_frames:
                image = tiff_frames[0]

        try:
            from app.utils.paths import resolved_temp_dir
            thumb_path = resolved_temp_dir() / "thumbs" / f"{_cache_key(image_path)}.jpg"
            _generate_thumbnail(image, thumb_path)
        except Exception:
            pass

        image_hash   = _cache_key(image_path)
        raw_caption: Optional[str] = None
        caption_source = "blip"
        infer_ms = 0.0

        if use_cache:
            raw_caption = _caption_cache_get(image_hash)
            if raw_caption:
                caption_source = "cache"

        if raw_caption is None:
            t_infer     = time.time()
            raw_caption = _blip_caption(image, session_id)
            infer_ms    = round((time.time() - t_infer) * 1000, 1)
            if use_cache and raw_caption:
                _caption_cache_set(image_hash, raw_caption)

        caption = _clean_caption(raw_caption) if raw_caption else None

        if not caption:
            logger.warning(
                event="caption_rejected",
                raw=str(raw_caption)[:80] if raw_caption else "",
                session_id=session_id,
            )
            return None

        confidence = _caption_confidence(caption)
        _build_quality_metadata(
            caption=caption, confidence=confidence, is_solid=is_solid,
            infer_ms=infer_ms, source=caption_source,
        )

        logger.debug(
            event="caption_success",
            length=len(caption), words=len(caption.split()),
            confidence=confidence, source=caption_source,
            latency_ms=round((time.time() - start) * 1000, 1),
            session_id=session_id,
        )
        return caption

    except (FileNotFoundError, EmptyContentError, ValueError) as exc:
        logger.warning(event="caption_image_error", image=path.name, error=str(exc), session_id=session_id)
        return None
    except Exception as exc:
        logger.error(event="caption_failed", image=path.name, session_id=session_id, error=str(exc),
                     latency=round(time.time() - start, 3))
        return None


def classify_and_caption(
    image_path: str,
    session_id: str,
    ocr_text: str = "",
    use_cache: bool = True,
) -> Tuple[Optional[str], str, List[str]]:
    """Generate a finance-specific caption with image type classification.

    Returns:
        (caption, image_type, extracted_numbers)
    """
    start = time.time()
    path  = Path(image_path)
    image_type = "infographic"
    extracted_numbers: List[str] = []

    try:
        if path.suffix.lower() == ".svg":
            image = _rasterize_svg(image_path)
        else:
            image = _load_image(image_path)

        if image is None:
            return None, image_type, extracted_numbers

        if ocr_text:
            extracted_numbers = _FIN_NUMBER_RE.findall(ocr_text)[:20]

        image_type  = _classify_image_type(image, ocr_text)
        text_prompt = _finance_caption_prompt(image_type)

        image_hash = hashlib.sha256((image_path + "|" + text_prompt).encode()).hexdigest()
        raw_caption: Optional[str] = None

        if use_cache:
            raw_caption = _caption_cache_get(image_hash)

        if raw_caption is None:
            raw_caption = _blip_caption(image, session_id, text_prompt=text_prompt)
            if use_cache and raw_caption:
                _caption_cache_set(image_hash, raw_caption)

        caption = _clean_caption(raw_caption) if raw_caption else None

        if caption and extracted_numbers:
            caption_lower = caption.lower()
            missing = [
                n for n in extracted_numbers
                if re.sub(r"[\s,]", "", n.lower()) not in re.sub(r"[\s,]", "", caption_lower)
            ]
            if missing:
                caption = caption + " [OCR values: " + " ".join(missing[:8]) + "]"

        logger.debug(
            event="classify_and_caption_done",
            image=path.name, image_type=image_type,
            numbers=len(extracted_numbers),
            latency_ms=round((time.time() - start) * 1000, 1),
            session_id=session_id,
        )
        return caption, image_type, extracted_numbers

    except Exception as exc:
        logger.warning(
            event="classify_and_caption_failed", image=path.name,
            error=str(exc), session_id=session_id,
        )
        return None, image_type, extracted_numbers


async def classify_and_caption_async(
    image_path: str,
    session_id: str,
    ocr_text: str = "",
    use_cache: bool = True,
) -> Tuple[Optional[str], str, List[str]]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: classify_and_caption(image_path, session_id, ocr_text, use_cache),
    )


async def generate_caption_async(
    image_path: str,
    session_id: str,
    use_cache: bool = True,
) -> Optional[str]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: generate_caption(image_path, session_id, use_cache),
    )


async def generate_captions_batch(
    image_paths: List[str],
    session_id: str,
) -> List[Optional[str]]:
    sem = asyncio.Semaphore(settings.ASYNC_SEMAPHORE_WORKERS)

    async def _cap(path: str) -> Optional[str]:
        async with sem:
            return await generate_caption_async(path, session_id)

    return await asyncio.gather(*[_cap(p) for p in image_paths], return_exceptions=False)


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE CHUNKER
# ══════════════════════════════════════════════════════════════════════════════

class ImageChunker(BaseChunker):
    """Finance-grade chunker for image files (charts, infographics, scanned docs).

    Each image = exactly 1 chunk (images are atomic).
    Pipeline: TrOCR → BLIP2 caption → SigLIP image type → consistency check.
    """

    def chunk(
        self,
        extracts: List[RawExtract],
        meta: UniversalMetadata,
    ) -> List[IngestedDocument]:
        source  = Path(meta.source_path).name or "unknown.jpg"
        surface = "image_chunker"
        docs: List[IngestedDocument] = []

        for chunk_idx, ext in enumerate(extracts):
            if ext.extract_type != "image_raw":
                continue
            raw = ext.raw_bytes or b""
            if not raw:
                continue

            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as exc:
                logger.warning(event="image_chunker_open_failed", error=str(exc), source=source)
                continue

            img_width, img_height = img.size

            # Step 1: OCR (ground-truth numbers).
            ocr_text = ""
            try:
                ocr_text = ocr(img)
            except Exception as exc:
                logger.warning(event="image_trocr_failed", error=str(exc))

            # Step 2: BLIP2 caption.
            caption_text = ""
            try:
                caption_text = blip2_caption(img)
            except Exception as exc:
                logger.warning(event="image_blip2_failed", error=str(exc))

            # Step 3: SigLIP image type classification (falls back to heuristic).
            image_type = "infographic"
            try:
                image_type = blip2_classify_image_type(img, ocr_text)
            except Exception:
                pass

            # Step 4: OCR-caption consistency check.
            mismatch = _check_mismatch(ocr_text, caption_text)
            if mismatch:
                logger.warning(event="ocr_caption_mismatch", source=source)

            # Step 5: Combine (prefer OCR numbers as ground truth).
            combined = f"{image_type}: {caption_text}"
            if ocr_text:
                combined += f"\n{ocr_text}"
            combined = combined.strip()
            if not combined:
                continue

            fin_entities      = extract_finance_entities(combined)
            extracted_numbers = list(_NUMBER_RE.findall(ocr_text or caption_text))
            chunk_hash        = deterministic_chunk_id(source, "image_raw_0", chunk_idx)

            # Extract image_title: first sentence/line of caption (MD Phase 1.5)
            raw_title = (caption_text or "").split(".")[0].split("\n")[0].strip()
            image_title = raw_title[:120] if raw_title else source

            structure = {
                "chunk_hash_id":        chunk_hash,
                "source_file":          source,
                "chunk_index":          chunk_idx,
                "image_type":           image_type,
                "image_title":          image_title,
                "caption":              caption_text,
                "ocr_text":             ocr_text,
                "extracted_numbers":    extracted_numbers,
                "time_period":          ext.extra.get("time_period"),
                "data_series":          ext.extra.get("data_series", []),
                "ocr_caption_mismatch": mismatch,
                "parent_document":      ext.extra.get("parent_document"),
                "parent_page":          ext.extra.get("parent_page"),
                "finance_entities":     fin_entities,
                "image_width":          img_width,
                "image_height":         img_height,
            }

            doc = self._make_doc(
                text=combined,
                modality="jpg",
                subtype="caption",
                source=source,
                page=None,
                chunk_idx=chunk_idx,
                structure=structure,
                meta=meta,
                surface=surface,
            )
            if doc:
                docs.append(doc)

        logger.info(event="image_chunking_done", source=source, chunks=len(docs))
        return docs
