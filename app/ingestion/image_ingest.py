from __future__ import annotations

import asyncio
import hashlib
import io
import math
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import settings
from app.ingestion.schema import (
    EmptyFileError,
    FileTooLargeError,
    IngestedDocument,
    UnsupportedMimeError,
    build_universal_metadata,
    Modality,
    normalize_text,
    redact_pii,
    sanitize_prompt_injection,
    sha256_file,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# SUPPORTED FORMATS
SUPPORTED_IMAGE_FORMATS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".tiff", ".tif", ".gif", ".heic", ".heif",
    ".svg", ".cr2", ".nef", ".arw",
}

# RAW FORMATS REQUIRING RAWPY
RAW_FORMATS = {".cr2", ".nef", ".arw"}

# HEIC FORMATS REQUIRING PILLOW-HEIF
HEIC_FORMATS = {".heic", ".heif"}

# WATERMARK KEYWORDS
_WATERMARK_KEYWORDS = {
    "CONFIDENTIAL", "DRAFT", "WATERMARK", "SAMPLE",
    "RESTRICTED", "CLASSIFIED", "DO NOT COPY",
}

# WEAK CAPTION PREFIXES TO STRIP
_WEAK_PREFIXES = [
    "a blurry image of", "a close up of", "an image of",
    "a picture of", "a photo of", "photo of", "image of",
    "this is a", "this is an",
]


# SHA256 FILE HASH

def _file_hash(file_path: str) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# MAGIC BYTE MIME DETECTION

def _detect_mime_from_bytes(header: bytes, suffix: str) -> str:
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    if header.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    if header.startswith(b"BM"):
        return "image/bmp"
    if header.lstrip()[:4] in (b"<svg", b"<SVG"):
        return "image/svg+xml"
    if suffix in HEIC_FORMATS:
        return "image/heic"
    if suffix in RAW_FORMATS:
        return "image/x-raw"
    return "application/octet-stream"


# DISK SPACE GUARD

def _check_disk_space(path: Path) -> None:
    try:
        usage = shutil.disk_usage(path.parent)
        min_bytes = settings.MIN_FREE_DISK_MB * 1024 * 1024
        if usage.free < min_bytes:
            logger.warning(
                event="low_disk_space",
                free_mb=round(usage.free / 1024 / 1024, 1),
                required_mb=settings.MIN_FREE_DISK_MB,
            )
    except OSError as exc:
        logger.warning(event="disk_space_check_failed", error=str(exc))


# PATH TRAVERSAL GUARD

def _safe_resolve(file_path: str) -> Path:
    path = Path(file_path).expanduser().resolve()
    chroot = settings.CHROOT_BASE.resolve()
    try:
        path.relative_to(chroot)
    except ValueError:
        raise ValueError(f"PATH_TRAVERSAL_BLOCKED: {path} is outside chroot {chroot}")
    return path


# VALIDATION

def _validate_image_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"FILE_NOT_FOUND: {path}")

    if path.stat().st_size == 0:
        raise EmptyFileError(str(path))

    if path.stat().st_size > settings.MAX_FILE_SIZE_IMAGE:
        raise FileTooLargeError(
            f"FILE_TOO_LARGE: {path.stat().st_size} bytes exceeds "
            f"{settings.MAX_FILE_SIZE_IMAGE} bytes"
        )

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_FORMATS:
        raise UnsupportedMimeError(f"UNSUPPORTED_IMAGE_FORMAT: {suffix}")

    # MAGIC BYTE CHECK
    if settings.MAGIC_BYTE_MIME_DETECTION:
        with open(path, "rb") as f:
            header = f.read(16)
        mime = _detect_mime_from_bytes(header, suffix)
        if mime == "application/octet-stream" and suffix not in HEIC_FORMATS | RAW_FORMATS | {".svg"}:
            raise UnsupportedMimeError(f"MAGIC_BYTE_MIME_MISMATCH: detected {mime} for {suffix}")


# IMAGE LOADING — WITH FORMAT BRANCHING

def _load_image(path: Path, session_id: str) -> Image.Image:
    suffix = path.suffix.lower()

    # RAW FORMAT
    if suffix in RAW_FORMATS:
        try:
            import rawpy
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
            import numpy as np
            return Image.fromarray(rgb, "RGB")
        except ImportError:
            raise ImportError("RAWPY_REQUIRED for RAW image formats (.cr2, .nef, .arw)")
        except Exception as exc:
            raise ValueError(f"RAW_DECODE_FAILED: {exc}")

    # HEIC FORMAT
    if suffix in HEIC_FORMATS:
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except ImportError:
            raise ImportError("PILLOW_HEIF_REQUIRED for HEIC/HEIF images")

    # SVG FORMAT
    if suffix == ".svg":
        try:
            import cairosvg
            png_data = cairosvg.svg2png(url=str(path))
            img = Image.open(io.BytesIO(png_data)).convert("RGB")
            logger.debug(event="svg_rasterized", file=path.name, session_id=session_id)
            return img
        except ImportError:
            # FALLBACK: extract text from SVG XML
            logger.warning(
                event="cairosvg_not_installed_svg_text_only",
                file=path.name,
                session_id=session_id,
            )
            raise UnsupportedMimeError("SVG_RASTERIZATION_FAILED_CAIROSVG_MISSING")

    # ANIMATED GIF — USE FIRST FRAME
    try:
        with Image.open(path) as raw:
            frame_count = getattr(raw, "n_frames", 1)
            if frame_count > 1:
                logger.debug(
                    event="animated_gif_first_frame",
                    frames=frame_count,
                    file=path.name,
                    session_id=session_id,
                )
                raw.seek(0)

            img = ImageOps.exif_transpose(raw)

            # ICC COLOR PROFILE — CONVERT TO SRGB
            if img.info.get("icc_profile"):
                try:
                    from PIL import ImageCms
                    icc_data = img.info["icc_profile"]
                    src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_data))
                    dst_profile = ImageCms.createProfile("sRGB")
                    img = ImageCms.profileToProfile(img, src_profile, dst_profile)
                    logger.debug(event="icc_profile_converted", file=path.name)
                except Exception:
                    pass

            # PNG WITH ALPHA — FLATTEN TO WHITE
            if img.mode in ("RGBA", "LA", "PA"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "PA":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            return img.copy()

    except UnidentifiedImageError as exc:
        raise ValueError(f"CORRUPT_IMAGE: {exc}")
    except Exception as exc:
        raise ValueError(f"IMAGE_LOAD_FAILED: {exc}")


# DIMENSION VALIDATION

def _validate_dimensions(img: Image.Image) -> None:
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("INVALID_IMAGE_DIMENSIONS: zero dimension")
    if w < 32 or h < 32:
        raise ValueError(f"IMAGE_TOO_SMALL: {w}x{h} (minimum 32x32)")


# ASPECT RATIO CHECK

def _check_aspect_ratio(w: int, h: int) -> Optional[str]:
    if h == 0:
        return None
    ratio = w / h
    if ratio > 20 or ratio < 0.05:
        return f"EXTREME_ASPECT_RATIO_{ratio:.2f}"
    return None


# RESIZE IF NEEDED

def _resize_if_needed(img: Image.Image) -> Image.Image:
    w, h = img.size
    max_mp = settings.MAX_IMAGE_SIZE_MP * 1_000_000
    max_dim = settings.MAX_IMAGE_DIM

    # 50 MP GUARD
    if w * h > max_mp:
        scale = math.sqrt(w * h / max_mp)
        new_w = int(w / scale)
        new_h = int(h / scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.debug(event="image_resized_mp_guard", original_mp=round(w * h / 1_000_000, 1))
        w, h = img.size

    if max(w, h) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)

    return img


# EXIF EXTRACTION

def _extract_exif(path: Path) -> Dict[str, Any]:
    exif_data: Dict[str, Any] = {}
    try:
        from PIL.ExifTags import TAGS, GPSTAGS
        with Image.open(path) as img:
            raw_exif = img._getexif()  # type: ignore[attr-defined]
            if not raw_exif:
                return exif_data

            for tag_id, value in raw_exif.items():
                tag = TAGS.get(tag_id, str(tag_id))

                if tag == "GPSInfo":
                    if settings.IMAGE_PRIVACY_MODE:
                        # ANONYMIZE GPS IN PRIVACY MODE
                        exif_data["gps_stripped"] = True
                        continue
                    gps: Dict[str, Any] = {}
                    for gps_tag_id, gps_value in value.items():
                        gps_tag = GPSTAGS.get(gps_tag_id, str(gps_tag_id))
                        gps[gps_tag] = str(gps_value)
                    exif_data["gps"] = gps
                    continue

                if isinstance(value, bytes):
                    continue
                if isinstance(value, (int, float, str)):
                    exif_data[tag] = value

    except Exception as exc:
        logger.debug(event="exif_extraction_failed", error=str(exc))

    return exif_data


# IPTC / XMP METADATA

def _extract_iptc_xmp(path: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    try:
        from PIL import IptcImagePlugin
        with Image.open(path) as img:
            iptc = IptcImagePlugin.getiptcinfo(img)
            if iptc:
                for key, val in iptc.items():
                    try:
                        meta[str(key)] = val.decode("utf-8", errors="replace") if isinstance(val, bytes) else str(val)
                    except Exception:
                        pass
    except Exception:
        pass
    return meta


# OCR PREPROCESSING

def _preprocess_for_ocr(img: Image.Image):
    import numpy as np
    import cv2
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2,
    )
    return thresh


# OCR EXTRACTION — TESSERACT + EASYOCR ENSEMBLE

def _extract_ocr(img: Image.Image, session_id: str) -> str:
    text_parts: List[str] = []

    # TESSERACT
    try:
        import pytesseract
        processed = _preprocess_for_ocr(img)
        tesseract_text = (pytesseract.image_to_string(processed) or "").strip()
        if len(tesseract_text) >= settings.CHUNK_MIN_SIZE:
            text_parts.append(tesseract_text)
    except Exception as exc:
        logger.debug(event="tesseract_ocr_failed", error=str(exc), session_id=session_id)

    # EASYOCR ENSEMBLE
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        import numpy as np
        results = reader.readtext(np.array(img))
        easy_text = " ".join([r[1] for r in results if r[2] > 0.3]).strip()
        if len(easy_text) >= settings.CHUNK_MIN_SIZE:
            text_parts.append(easy_text)
    except ImportError:
        pass
    except Exception as exc:
        logger.debug(event="easyocr_failed", error=str(exc), session_id=session_id)

    combined = " ".join(text_parts)
    if len(combined) < settings.CHUNK_MIN_SIZE:
        return ""
    return combined[:settings.MAX_PROMPT_CHARS]


# BLUR SCORE

def _blur_score(img: Image.Image) -> float:
    try:
        import numpy as np
        import cv2
        gray = np.array(img.convert("L"))
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(min(laplacian / 100.0, 1.0))
    except Exception:
        return 1.0


# PERCEPTUAL HASH

def _phash(img: Image.Image) -> Optional[str]:
    try:
        import imagehash
        return str(imagehash.phash(img))
    except ImportError:
        return None
    except Exception:
        return None


# COLOR ANALYSIS — DOMINANT PALETTE

def _dominant_colors(img: Image.Image, k: int = 5) -> List[str]:
    try:
        import numpy as np
        from sklearn.cluster import KMeans
        small = img.resize((100, 100))
        arr = np.array(small).reshape(-1, 3).astype(float)
        km = KMeans(n_clusters=k, n_init=3, random_state=0)
        km.fit(arr)
        centers = km.cluster_centers_.astype(int)
        return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in centers]
    except Exception:
        return []


# FACE DETECTION — COUNT ONLY, NO IDENTIFICATION

def _face_count(img: Image.Image) -> int:
    try:
        import mediapipe as mp
        import numpy as np
        mp_face = mp.solutions.face_detection
        with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
            arr = np.array(img)
            results = detector.process(arr)
            if results.detections:
                return len(results.detections)
        return 0
    except ImportError:
        return 0
    except Exception:
        return 0


# SOLID COLOR CHECK — LOW CONTENT FLAG

def _is_solid_color(img: Image.Image) -> bool:
    try:
        import numpy as np
        arr = np.array(img.resize((50, 50)))
        std = float(arr.std())
        return std < 5.0
    except Exception:
        return False


# WATERMARK DETECTION

def _detect_watermark(text: str) -> bool:
    upper = text.upper()
    return any(kw in upper for kw in _WATERMARK_KEYWORDS)


# THUMBNAIL GENERATION

def _generate_thumbnail(img: Image.Image, path: Path, session_id: str) -> Optional[str]:
    try:
        thumb = img.copy()
        thumb.thumbnail(
            (settings.THUMBNAIL_WIDTH, settings.THUMBNAIL_HEIGHT),
            Image.LANCZOS,
        )
        thumb_path = settings.TEMP_DIR / f"thumb_{uuid.uuid4().hex}.jpg"
        thumb.save(str(thumb_path), "JPEG", quality=85)
        logger.debug(event="thumbnail_generated", path=str(thumb_path), session_id=session_id)
        return str(thumb_path)
    except Exception as exc:
        logger.debug(event="thumbnail_failed", error=str(exc), session_id=session_id)
        return None


# CAPTION CLEANING

def _clean_caption(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    lower = text.lower()
    for prefix in _WEAK_PREFIXES:
        if lower.startswith(prefix):
            text = text[len(prefix):].strip()
            lower = text.lower()
            break
    words = text.split()
    if len(words) < 5:
        return None
    text = " ".join(words[:30])
    return text[0].upper() + text[1:] if text else None


# CAPTION GENERATION — VISION PROVIDER

def _generate_caption(file_path: str, img: Image.Image, session_id: str) -> str:
    try:
        from app.ingestion.frame_captioner import generate_caption
        raw = generate_caption(file_path, session_id=session_id)
        caption = _clean_caption(raw) if raw else None
        if caption and len(caption.split()) >= 5:
            return caption.strip()
    except Exception as exc:
        logger.warning(event="caption_failed", error=str(exc), session_id=session_id)

    return "Generic image content"


# QUALITY SCORE

def _image_quality_score(
    blur: float,
    has_ocr: bool,
    watermark: bool,
    w: int,
    h: int,
    solid: bool,
) -> float:
    score = blur
    if w * h > 500_000:
        score = min(score + 0.1, 1.0)
    if watermark:
        score = max(score - 0.2, 0.0)
    if not has_ocr:
        score = max(score - 0.05, 0.0)
    if solid:
        score = max(score - 0.3, 0.0)
    return round(score, 3)


# SHA256 DEDUP CHECK

def _sha256_check(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# MULTI-PAGE TIFF HANDLER

def _load_tiff_pages(path: Path) -> List[Image.Image]:
    pages: List[Image.Image] = []
    try:
        with Image.open(path) as img:
            for i in range(getattr(img, "n_frames", 1)):
                img.seek(i)
                frame = ImageOps.exif_transpose(img).convert("RGB")
                pages.append(frame.copy())
    except Exception as exc:
        logger.warning(event="tiff_multipage_failed", error=str(exc))
    return pages


# MAIN INGEST

def ingest(file_path: str, session_id: str) -> List[IngestedDocument]:
    if not session_id:
        raise ValueError("SESSION_ID_REQUIRED")

    try:
        path = _safe_resolve(file_path)
    except ValueError as exc:
        raise ValueError(str(exc))

    _validate_image_file(path)
    _check_disk_space(path)

    ext = path.suffix.lower()
    file_size = path.stat().st_size
    start = time.time()
    source_name = path.name
    source_path = str(path)
    doc_id = str(uuid.uuid4())
    file_hash = _sha256_check(path)

    logger.info(
        event="image_ingest_start",
        file=source_name,
        ext=ext,
        size=file_size,
        session_id=session_id,
    )

    # MULTI-PAGE TIFF
    is_multi_tiff = ext in {".tiff", ".tif"}
    tiff_pages: List[Image.Image] = []

    try:
        if is_multi_tiff:
            tiff_pages = _load_tiff_pages(path)
            if len(tiff_pages) > 1:
                logger.debug(
                    event="multi_page_tiff",
                    pages=len(tiff_pages),
                    file=source_name,
                    session_id=session_id,
                )

        # LOAD PRIMARY IMAGE
        img = _load_image(path, session_id)
        _validate_dimensions(img)

        original_width, original_height = img.size

        aspect_warning = _check_aspect_ratio(original_width, original_height)
        if aspect_warning:
            logger.warning(
                event="aspect_ratio_warning",
                warning=aspect_warning,
                file=source_name,
                session_id=session_id,
            )

        img = _resize_if_needed(img)
        width, height = img.size

        # ANALYSIS
        ocr_text = _extract_ocr(img, session_id)
        caption = _generate_caption(file_path, img, session_id)
        blur = _blur_score(img)
        ph = _phash(img)
        solid = _is_solid_color(img)
        exif_data = _extract_exif(path)
        iptc_data = _extract_iptc_xmp(path)
        colors = _dominant_colors(img)
        watermark = _detect_watermark((ocr_text or "") + " " + (caption or ""))
        quality = _image_quality_score(blur, bool(ocr_text), watermark, width, height, solid)

        face_count = 0
        if not settings.IMAGE_PRIVACY_MODE:
            face_count = _face_count(img)

        thumb_path = _generate_thumbnail(img, path, session_id)

        if solid:
            logger.warning(
                event="solid_color_image_low_content",
                file=source_name,
                session_id=session_id,
            )

        # UNIVERSAL METADATA
        metadata = build_universal_metadata(
            path,
            Modality.IMAGE,
            "image/jpeg",
            language="und",
            chunk_count=0,
            tags=[],
            custom_fields={
                "original_width": original_width,
                "original_height": original_height,
                "image_width": width,
                "image_height": height,
                "image_format": ext.lstrip(".").upper(),
                "blur_score": blur,
                "perceptual_hash": ph,
                "watermark_detected": watermark,
                "solid_color": solid,
                "face_count": face_count,
                "dominant_colors": colors,
                "exif": exif_data,
                "iptc": iptc_data,
                "thumbnail_path": thumb_path,
                "privacy_mode": settings.IMAGE_PRIVACY_MODE,
                "checksum_sha256": file_hash,
            },
        )

        base_structure: Dict[str, Any] = {
            "doc_id": doc_id,
            "session_id": session_id,
            "file_hash": file_hash,
            "checksum_sha256": file_hash,
            "source_path": source_path,
            "asset_path": source_path,
            "original_width": original_width,
            "original_height": original_height,
            "image_width": width,
            "image_height": height,
            "image_format": ext.lstrip(".").upper(),
            "watermark_detected": watermark,
            "blur_score": blur,
            "perceptual_hash": ph,
            "solid_color": solid,
            "face_count": face_count,
            "dominant_colors": colors,
            "thumbnail_path": thumb_path,
            "ingestion_time": time.time(),
            "privacy_mode": settings.IMAGE_PRIVACY_MODE,
        }

        documents: List[IngestedDocument] = []

        # CAPTION DOCUMENT
        caption_clean = sanitize_prompt_injection(redact_pii(normalize_text(caption)))
        documents.append(
            IngestedDocument(
                text=caption_clean,
                modality="image",
                subtype="caption",
                source_type="image",
                source=source_name,
                metadata=metadata,
                structure={**base_structure, "content_type": "image_caption", "embedding_space": "text"},
                extra_metadata={
                    "modality_weight": 1.0,
                    "importance_score": quality,
                    "data_quality_score": quality,
                    "tags": [],
                    "pii_redacted": settings.PII_DETECTION_ENABLED,
                },
            ).finalize()
        )

        # OCR DOCUMENT
        if ocr_text:
            ocr_clean = sanitize_prompt_injection(redact_pii(normalize_text(ocr_text)))
            documents.append(
                IngestedDocument(
                    text=ocr_clean,
                    modality="image",
                    subtype="ocr",
                    source_type="image",
                    source=source_name,
                    metadata=metadata,
                    structure={**base_structure, "content_type": "image_ocr", "embedding_space": "text"},
                    extra_metadata={
                        "modality_weight": 0.9,
                        "importance_score": min(quality + 0.1, 1.0),
                        "data_quality_score": 0.8,
                        "pii_redacted": settings.PII_DETECTION_ENABLED,
                    },
                ).finalize()
            )

        # VISION EMBEDDING DOCUMENT (CLIP)
        documents.append(
            IngestedDocument(
                text=caption_clean,
                modality="image",
                subtype="caption",
                source_type="image",
                source=source_name,
                metadata=metadata,
                structure={
                    **base_structure,
                    "content_type": "image_semantic",
                    "embedding_space": "vision",
                    "frame_path": source_path,
                },
                extra_metadata={
                    "modality_weight": 1.0,
                    "importance_score": quality,
                    "data_quality_score": quality,
                },
            ).finalize()
        )

        # MULTI-PAGE TIFF — ADDITIONAL PAGES
        for page_idx, page_img in enumerate(tiff_pages[1:], start=1):
            try:
                page_caption = _generate_caption(file_path, page_img, session_id)
                page_ocr = _extract_ocr(page_img, session_id)
                page_caption_clean = sanitize_prompt_injection(redact_pii(normalize_text(page_caption)))

                documents.append(
                    IngestedDocument(
                        text=page_caption_clean,
                        modality="image",
                        subtype="caption",
                        source_type="image",
                        source=source_name,
                        metadata=metadata,
                        structure={
                            **base_structure,
                            "content_type": "tiff_page_caption",
                            "tiff_page": page_idx,
                            "embedding_space": "text",
                        },
                        extra_metadata={
                            "modality_weight": 1.0,
                            "importance_score": quality,
                            "data_quality_score": quality,
                        },
                    ).finalize()
                )

                if page_ocr:
                    page_ocr_clean = sanitize_prompt_injection(redact_pii(normalize_text(page_ocr)))
                    documents.append(
                        IngestedDocument(
                            text=page_ocr_clean,
                            modality="image",
                            subtype="ocr",
                            source_type="image",
                            source=source_name,
                            metadata=metadata,
                            structure={
                                **base_structure,
                                "content_type": "tiff_page_ocr",
                                "tiff_page": page_idx,
                                "embedding_space": "text",
                            },
                            extra_metadata={
                                "modality_weight": 0.9,
                                "importance_score": quality,
                                "data_quality_score": 0.8,
                            },
                        ).finalize()
                    )
            except Exception as exc:
                logger.warning(
                    event="tiff_page_failed",
                    page=page_idx,
                    error=str(exc),
                    session_id=session_id,
                )

        if not documents:
            raise ValueError("NO_VALID_IMAGE_DOCS")

        metadata.chunk_count = len(documents)
        latency = round(time.time() - start, 2)

        if latency * 1000 > settings.SLO_IMAGE_P95_MS:
            logger.warning(
                event="image_slo_exceeded",
                latency_ms=round(latency * 1000, 1),
                target_ms=settings.SLO_IMAGE_P95_MS,
                file=source_name,
                session_id=session_id,
            )

        logger.info(
            event="image_ingest_success",
            file=source_name,
            docs=len(documents),
            blur=blur,
            quality=quality,
            watermark=watermark,
            solid_color=solid,
            face_count=face_count,
            has_ocr=bool(ocr_text),
            has_phash=bool(ph),
            has_exif=bool(exif_data),
            latency=latency,
            session_id=session_id,
        )

        return documents

    except (EmptyFileError, FileTooLargeError, UnsupportedMimeError):
        raise
    except Exception as exc:
        logger.error(
            event="image_ingest_failed",
            file=path.name,
            session_id=session_id,
            error=str(exc),
            latency=round(time.time() - start, 2),
        )
        raise


# ASYNC WRAPPER

async def async_ingest(file_path: str, session_id: str) -> List[IngestedDocument]:
    return await asyncio.to_thread(ingest, file_path, session_id)

