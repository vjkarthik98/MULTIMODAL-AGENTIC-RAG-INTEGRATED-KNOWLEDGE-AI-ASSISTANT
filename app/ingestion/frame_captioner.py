from __future__ import annotations

import asyncio
import hashlib
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps

from app.core.config import settings
from app.ingestion.schema import EmptyContentError
from app.utils.logger import get_logger

try:
    import torch
except ImportError:
    torch = None

logger = get_logger(__name__)


# WEAK CAPTION PREFIXES TO STRIP

_WEAK_PREFIXES = [
    "a blurry image of",
    "a close up of",
    "an image of",
    "a picture of",
    "a photo of",
    "photo of",
    "image of",
    "this is a",
    "this is an",
    "a view of",
    "a shot of",
]

# CAPTION QUALITY THRESHOLDS
_CAPTION_MAX_WORDS  = 50
_CAPTION_MIN_WORDS  = 5
_CAPTION_MIN_CHARS  = 10

# PROMPT INJECTION — delegated to unified guardrail (Phase 26)

# SOLID COLOR DETECTION
_SOLID_COLOR_VARIANCE_THRESHOLD = 5.0


# SHA-256 CACHE KEY

def _cache_key(image_path: str) -> str:
    return hashlib.sha256(image_path.encode("utf-8")).hexdigest()


# CAPTION REDIS CACHE HELPERS

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
            mem.cache_set(
                f"caption:{key}",
                caption,
                ttl=settings.REDIS_EMBEDDING_CACHE_TTL,
            )
    except Exception:
        pass


# REPETITION CLEAN

def _remove_repetition(text: str) -> str:
    words = text.split()
    if len(words) < 6:
        return text

    # Exact half-duplication check (original logic)
    half  = len(words) // 2
    first = " ".join(words[:half])
    second = " ".join(words[half:])
    if first.strip().lower() == second.strip().lower():
        return first.strip()

    # N-gram repetition loop detection — catches "invoice invoice invoice..."
    # and longer phrase loops like "the dashboard the dashboard the dashboard"
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
        # If the same n-gram repeats 3+ times consecutively, truncate after first occurrence
        if run >= 3:
            return " ".join(words[:n]).strip()

    return text


# PROMPT INJECTION SANITIZATION — delegates to unified guardrail (Phase 26)

def _sanitize_caption(text: str) -> str:
    from app.guardrails.input_guard import sanitize as _guard_sanitize
    return _guard_sanitize(text, surface="frame_captioner")


# CAPTION CLEAN

def _clean_caption(text: str) -> Optional[str]:
    if not text:
        return None

    text  = text.strip()
    text  = unicodedata.normalize("NFC", text)
    lower = text.lower()

    # STRIP WEAK PREFIXES
    for pattern in _WEAK_PREFIXES:
        if lower.startswith(pattern):
            text  = text[len(pattern):].strip()
            lower = text.lower()
            break

    # STRIP NULL BYTES — SECTION 2.3
    if "\x00" in text:
        text = text.replace("\x00", "")

    # PROMPT INJECTION SANITIZATION
    text = _sanitize_caption(text)

    text  = _remove_repetition(text)
    words = text.split()

    if len(words) < _CAPTION_MIN_WORDS:
        return None

    text = " ".join(words[:_CAPTION_MAX_WORDS])
    text = text[0].upper() + text[1:] if text else text

    if len(text) < _CAPTION_MIN_CHARS:
        return None

    return text


# CAPTION CONFIDENCE PROXY

def _caption_confidence(caption: str) -> float:
    words      = caption.split()
    unique     = set(w.lower() for w in words)
    word_count = len(words)
    if word_count == 0:
        return 0.0
    diversity    = len(unique) / word_count
    length_score = min(len(caption) / 100.0, 1.0)
    return round((diversity + length_score) / 2.0, 3)


# SOLID COLOR DETECTION — SECTION 4.1 (flag low-content images)

def _is_solid_color(image: Image.Image) -> bool:
    try:
        import numpy as np
        arr = np.array(image.convert("RGB"), dtype=float)
        variance = float(arr.var())
        return variance < _SOLID_COLOR_VARIANCE_THRESHOLD
    except Exception:
        return False


# IMAGE LOADING WITH FULL EDGE CASE HANDLING — SECTION 4.1

def _load_image(image_path: str) -> Image.Image:
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"IMAGE_NOT_FOUND: {image_path}")

    if path.stat().st_size == 0:
        raise EmptyContentError(f"EMPTY_IMAGE_FILE: {image_path}")

    if path.stat().st_size > settings.MAX_FILE_SIZE_IMAGE:
        raise ValueError(
            f"IMAGE_TOO_LARGE: {path.stat().st_size} bytes "
            f"exceeds {settings.MAX_FILE_SIZE_IMAGE}"
        )

    try:
        with Image.open(path) as img:
            # AUTO-ROTATION FROM EXIF — SECTION 4.1
            img = ImageOps.exif_transpose(img)

            # ANIMATED GIF — EXTRACT FIRST FRAME — SECTION 4.1
            if getattr(img, "is_animated", False):
                img.seek(0)
                logger.debug(event="animated_gif_first_frame_used", path=path.name)

            # PNG WITH ALPHA — FLATTEN TO WHITE — SECTION 4.1
            if img.mode == "RGBA":
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            w, h = img.size

            # ZERO-DIMENSION CHECK — SECTION 4.1
            if w == 0 or h == 0:
                raise ValueError("INVALID_IMAGE_DIMENSIONS: zero dimension")

            # TOO SMALL CHECK — SECTION 4.1
            if w < 32 or h < 32:
                raise ValueError(f"IMAGE_TOO_SMALL: {w}x{h}")

            # OVERSIZED RESIZE — SECTION 4.1 (> 50 MP or max dim)
            total_mp = (w * h) / 1_000_000
            if total_mp > settings.MAX_IMAGE_SIZE_MP:
                scale = (settings.MAX_IMAGE_SIZE_MP * 1_000_000 / (w * h)) ** 0.5
                new_w = max(int(w * scale), 32)
                new_h = max(int(h * scale), 32)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                logger.debug(event="image_resized_mp", original_mp=round(total_mp, 1))
            elif max(w, h) > settings.MAX_IMAGE_DIM:
                img.thumbnail(
                    (settings.MAX_IMAGE_DIM, settings.MAX_IMAGE_DIM),
                    Image.LANCZOS,
                )

            return img.copy()

    except (FileNotFoundError, EmptyContentError, ValueError):
        raise
    except Exception as e:
        raise ValueError(f"IMAGE_LOAD_FAILED: {e}") from e


# BLIP LOCAL CAPTION

def _blip_caption(
    image: Image.Image,
    session_id: str,
) -> Optional[str]:
    if torch is None:
        logger.warning(event="torch_unavailable_blip_skipped", session_id=session_id)
        return None

    try:
        from app.core.model_loader import model_loader
        processor, model, device = model_loader.get_blip()

        inputs = processor(image, return_tensors="pt").to(device)

        t_infer = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=settings.BLIP_MAX_TOKENS,
                num_beams=settings.BLIP_NUM_BEAMS,
                repetition_penalty=1.5,   # penalise repeated tokens/phrases
                no_repeat_ngram_size=3,   # forbid any 3-gram from appearing twice
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

    except Exception as e:
        logger.warning(
            event="blip_caption_failed",
            error=str(e),
            session_id=session_id,
        )
        return None


# THUMBNAIL GENERATION — SECTION 4.1

def _generate_thumbnail(
    image: Image.Image,
    output_path: Path,
) -> Optional[str]:
    try:
        thumb = image.copy()
        thumb.thumbnail(
            (settings.THUMBNAIL_WIDTH, settings.THUMBNAIL_HEIGHT),
            Image.LANCZOS,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        thumb.save(str(output_path), "JPEG", quality=85)
        return str(output_path)
    except Exception as e:
        logger.warning(event="thumbnail_generation_failed", error=str(e))
        return None


# MULTI-PAGE TIFF — SECTION 4.1

def _extract_tiff_frames(image_path: str) -> List[Image.Image]:
    frames: List[Image.Image] = []
    try:
        with Image.open(image_path) as img:
            if not hasattr(img, "n_frames") or img.n_frames <= 1:
                return []
            for i in range(img.n_frames):
                img.seek(i)
                frame = img.copy().convert("RGB")
                frames.append(frame)
    except Exception as e:
        logger.warning(event="tiff_frame_extraction_failed", error=str(e))
    return frames


# SVG RASTERIZATION — SECTION 4.1

def _rasterize_svg(svg_path: str) -> Optional[Image.Image]:
    try:
        import cairosvg
        from io import BytesIO
        png_bytes = cairosvg.svg2png(url=svg_path)
        return Image.open(BytesIO(png_bytes)).convert("RGB")
    except ImportError:
        logger.warning(event="cairosvg_not_installed", hint="pip install cairosvg")
        return None
    except Exception as e:
        logger.warning(event="svg_rasterize_failed", error=str(e))
        return None


# CAPTION QUALITY METADATA

def _build_quality_metadata(
    caption: str,
    confidence: float,
    is_solid: bool,
    infer_ms: float,
    source: str,
) -> Dict[str, Any]:
    return {
        "caption_confidence":  confidence,
        "caption_word_count":  len(caption.split()),
        "caption_char_count":  len(caption),
        "is_solid_color":      is_solid,
        "inference_ms":        infer_ms,
        "caption_source":      source,
    }


# MAIN CAPTION GENERATOR

def generate_caption(
    image_path: str,
    session_id: str,
    use_cache: bool = True,
) -> Optional[str]:

    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    start = time.time()
    path  = Path(image_path)

    try:
        logger.debug(
            event="caption_start",
            image=path.name,
            session_id=session_id,
        )

        # SVG SPECIAL HANDLING — SECTION 4.1
        if path.suffix.lower() == ".svg":
            image = _rasterize_svg(image_path)
            if image is None:
                logger.warning(event="svg_rasterize_failed_no_caption", image=path.name)
                return None
        else:
            image = _load_image(image_path)

        # SOLID COLOR DETECTION — SECTION 4.1
        is_solid = _is_solid_color(image)
        if is_solid:
            logger.info(
                event="solid_color_image_detected",
                image=path.name,
                session_id=session_id,
            )
            return "Solid color or blank image frame."

        # MULTI-PAGE TIFF — CAPTION FIRST FRAME — SECTION 4.1
        if path.suffix.lower() in (".tiff", ".tif"):
            tiff_frames = _extract_tiff_frames(image_path)
            if tiff_frames:
                image = tiff_frames[0]
                logger.debug(
                    event="tiff_first_frame_used",
                    total_frames=len(tiff_frames),
                    image=path.name,
                )

        # THUMBNAIL GENERATION — SECTION 4.1
        from app.utils.paths import resolved_temp_dir
        thumb_path = resolved_temp_dir() / "thumbs" / f"{_cache_key(image_path)}.jpg"
        _generate_thumbnail(image, thumb_path)

        # BLIP LOCAL CAPTION — WITH REDIS CACHE
        image_hash = _cache_key(image_path)
        raw_caption: Optional[str] = None
        caption_source = "blip"
        infer_ms = 0.0

        if use_cache:
            raw_caption = _caption_cache_get(image_hash)
            if raw_caption:
                caption_source = "cache"

        if raw_caption is None:
            t_infer = time.time()
            raw_caption = _blip_caption(image, session_id)
            infer_ms = round((time.time() - t_infer) * 1000, 1)
            if use_cache and raw_caption:
                _caption_cache_set(image_hash, raw_caption)

        # CLEAN AND VALIDATE
        caption = _clean_caption(raw_caption) if raw_caption else None

        if not caption:
            logger.warning(
                event="caption_rejected",
                raw=str(raw_caption)[:80] if raw_caption else "",
                session_id=session_id,
            )
            return None

        confidence = _caption_confidence(caption)

        quality_meta = _build_quality_metadata(
            caption=caption,
            confidence=confidence,
            is_solid=is_solid,
            infer_ms=infer_ms,
            source=caption_source,
        )

        total_latency = round(time.time() - start, 3)

        logger.debug(
            event="caption_success",
            length=len(caption),
            words=len(caption.split()),
            confidence=confidence,
            source=caption_source,
            latency_ms=round(total_latency * 1000, 1),
            session_id=session_id,
        )

        return caption

    except (FileNotFoundError, EmptyContentError, ValueError) as e:
        logger.warning(
            event="caption_image_error",
            image=path.name,
            error=str(e),
            session_id=session_id,
        )
        return None

    except Exception as e:
        logger.error(
            event="caption_failed",
            image=path.name,
            session_id=session_id,
            error=str(e),
            latency=round(time.time() - start, 3),
        )
        return None


# ASYNC WRAPPER

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


# BATCH CAPTION — SECTION 4.1

async def generate_captions_batch(
    image_paths: List[str],
    session_id: str,
) -> List[Optional[str]]:
    sem = asyncio.Semaphore(settings.ASYNC_SEMAPHORE_WORKERS)

    async def _cap(path: str) -> Optional[str]:
        async with sem:
            return await generate_caption_async(path, session_id)

    tasks = [_cap(p) for p in image_paths]
    return await asyncio.gather(*tasks, return_exceptions=False)


