"""OCR engine wrappers for odometer_obb_ocr.

PaddleOCR is the primary engine, EasyOCR an optional fallback. Both are
imported lazily so importing this module never requires either package to
be installed; only calling ``recognize()`` (or the reader constructors)
does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

ALLOWED_CHARSET = "0123456789kmKM"

_DIGIT_RE = re.compile(r"[^0-9]")


@dataclass
class OcrResult:
    raw_text: str
    confidence: float


class OcrEngineError(RuntimeError):
    """Raised when the requested OCR engine package is not installed."""


def _require_paddleocr():
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise OcrEngineError(
            "PaddleOCR is not installed. Install it with:\n"
            "    pip install paddleocr paddlepaddle\n"
            "See odometer_obb_ocr/requirements-optional.txt."
        ) from exc
    return PaddleOCR


def _require_easyocr():
    try:
        import easyocr
    except ImportError as exc:
        raise OcrEngineError(
            "EasyOCR is not installed. Install it with:\n"
            "    pip install easyocr\n"
            "See odometer_obb_ocr/requirements-optional.txt."
        ) from exc
    return easyocr


def _require_pytesseract():
    try:
        import pytesseract
    except ImportError as exc:
        raise OcrEngineError(
            "pytesseract is not installed. Install it with:\n"
            "    pip install pytesseract\n"
            "And ensure Tesseract 5.x is installed:\n"
            "    Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
            "    CentOS/RHEL: sudo yum install tesseract\n"
            "    macOS: brew install tesseract\n"
            "See odometer_obb_ocr/requirements-optional.txt."
        ) from exc
    return pytesseract


_PADDLE_SINGLETON = None
_EASY_SINGLETON = None
_TESSERACT_AVAILABLE = None


def get_paddle_reader():
    """Lazily construct and cache a PaddleOCR reader instance.

    PaddleOCR's constructor kwargs vary across versions (2.x used
    ``use_angle_cls``; 3.x's PP-OCRv6 pipeline uses ``use_textline_orientation``
    plus ``use_doc_orientation_classify``/``use_doc_unwarping`` for its extra
    document-preprocessing models). We try a few known-good spellings and fall
    back to a bare ``PaddleOCR(lang="en")`` if none are accepted.

    Since our input crops are already perspective-rectified, the 3.x pipeline's
    document-orientation-classify and document-unwarping (UVDoc) stages are
    unnecessary and are disabled where supported. ``enable_mkldnn=False`` works
    around a known PaddlePaddle/oneDNN crash
    (``ConvertPirAttribute2RuntimeAttribute not support``) on some CPU builds.

    **Angle classification is ENABLED** (``use_angle_cls=True`` /
    ``use_textline_orientation=True``) to robustly handle vertical, rotated, or
    upside-down text without manual rotation heuristics. This allows the OCR
    engine to natively recognize text at any orientation.

    There is no reliable cross-version kwarg for restricting recognition to a
    charset; that restriction is instead applied as post-processing via
    ``extract_digits``.
    """
    global _PADDLE_SINGLETON
    if _PADDLE_SINGLETON is None:
        paddle_ocr_cls = _require_paddleocr()
        for kwargs in (
            {
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": True,
                "enable_mkldnn": False,
                "lang": "en",
            },
            {
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": True,
                "lang": "en",
            },
            {"use_angle_cls": True, "enable_mkldnn": False, "lang": "en"},
            {"use_angle_cls": True, "lang": "en"},
            {"use_textline_orientation": True, "lang": "en"},
            {"lang": "en"},
        ):
            try:
                _PADDLE_SINGLETON = paddle_ocr_cls(**kwargs)
                break
            except Exception:  # noqa: BLE001 - constructor kwargs vary by version
                continue
        if _PADDLE_SINGLETON is None:
            raise OcrEngineError(
                "Could not construct a PaddleOCR reader with any known constructor "
                "kwargs. Your installed PaddleOCR version may use a different API; "
                "check `python -c \"from paddleocr import PaddleOCR; help(PaddleOCR)\"`."
            )
    return _PADDLE_SINGLETON


def get_easy_reader():
    """Lazily construct and cache an EasyOCR reader instance."""
    global _EASY_SINGLETON
    if _EASY_SINGLETON is None:
        easyocr = _require_easyocr()
        _EASY_SINGLETON = easyocr.Reader(["en"], gpu=False)
    return _EASY_SINGLETON


def check_tesseract_available():
    """Check if Tesseract is properly installed and accessible."""
    global _TESSERACT_AVAILABLE
    if _TESSERACT_AVAILABLE is None:
        pytesseract = _require_pytesseract()
        try:
            # Try to get Tesseract version
            version = pytesseract.get_tesseract_version()
            _TESSERACT_AVAILABLE = True
        except Exception:
            raise OcrEngineError(
                "Tesseract is installed via pip but the tesseract binary is not found.\n"
                "Install Tesseract OCR:\n"
                "    Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
                "    CentOS/RHEL: sudo yum install tesseract\n"
                "    macOS: brew install tesseract"
            )
    return _TESSERACT_AVAILABLE


def _parse_paddle_legacy_lines(lines) -> OcrResult:
    """Parse PaddleOCR 2.x-style output: a list of [box, (text, score)]."""
    texts = []
    scores = []
    for _box, (text, score) in lines:
        texts.append(text)
        scores.append(float(score))

    if not texts:
        return OcrResult(raw_text="", confidence=0.0)
    return OcrResult(raw_text="".join(texts), confidence=min(scores))


def _parse_paddle_v3_result(item) -> OcrResult:
    """Parse PaddleOCR 3.x-style output: a dict-like Result with
    ``rec_texts``/``rec_scores`` keys (from the PP-OCRv6 pipeline)."""
    texts = item.get("rec_texts") or []
    scores = item.get("rec_scores") or []

    if not texts:
        return OcrResult(raw_text="", confidence=0.0)

    scores = [float(s) for s in scores] if scores else [0.0] * len(texts)
    return OcrResult(raw_text="".join(texts), confidence=min(scores))


def run_paddle_ocr(reader, crop_bgr: np.ndarray) -> OcrResult:
    """Run a PaddleOCR reader over a crop and parse its raw output.

    PaddleOCR 2.x's ``.ocr()`` returns ``[[ [box, (text, score)], ... ]]`` per
    image. PaddleOCR 3.x's PP-OCRv6 pipeline instead returns a list of
    dict-like ``Result`` objects with ``rec_texts``/``rec_scores`` keys. Both
    shapes are handled here; in either case we concatenate all recognized
    text fragments and take the minimum score across them as a conservative
    confidence estimate.

    ``cls=True`` explicitly enables angle classification (text direction
    detection) for versions that support this parameter. This allows the OCR
    to handle vertical, rotated, or upside-down text without manual
    preprocessing.
    """
    try:
        raw = reader.ocr(crop_bgr, cls=True)
    except TypeError:
        # Fallback for versions that don't accept cls parameter
        raw = reader.ocr(crop_bgr)
    except Exception:  # noqa: BLE001 - .ocr() may not exist on some 3.x builds
        raw = reader.predict(crop_bgr)

    if not raw or raw[0] is None:
        return OcrResult(raw_text="", confidence=0.0)

    first = raw[0]

    if hasattr(first, "get") or isinstance(first, dict):
        return _parse_paddle_v3_result(first)

    if isinstance(first, list):
        return _parse_paddle_legacy_lines(first)

    if hasattr(first, "rec_texts"):
        texts = list(getattr(first, "rec_texts") or [])
        scores = [float(s) for s in (getattr(first, "rec_scores", None) or [])]
        if not texts:
            return OcrResult(raw_text="", confidence=0.0)
        scores = scores or [0.0] * len(texts)
        return OcrResult(raw_text="".join(texts), confidence=min(scores))

    return OcrResult(raw_text="", confidence=0.0)


def run_easy_ocr(reader, crop_bgr: np.ndarray) -> OcrResult:
    """Run an EasyOCR reader over a crop and parse its raw output.

    EasyOCR's ``.readtext()`` returns a list of (box, text, confidence)
    tuples. We concatenate text and take the minimum confidence across
    fragments as a conservative estimate.
    """
    results = reader.readtext(
        crop_bgr, allowlist=ALLOWED_CHARSET, detail=1, paragraph=False
    )

    if not results:
        return OcrResult(raw_text="", confidence=0.0)

    texts = [item[1] for item in results]
    scores = [float(item[2]) for item in results]

    return OcrResult(raw_text="".join(texts), confidence=min(scores))


def preprocess_for_low_res_ocr(crop_bgr: np.ndarray) -> np.ndarray:
    """Preprocess low-resolution crops for better OCR accuracy.

    Applies a series of image enhancement techniques:
    1. Upscale by 4x using INTER_CUBIC
    2. Convert to grayscale
    3. Denoise with bilateral filter
    4. Adaptive histogram equalization (CLAHE) for contrast
    5. Adaptive threshold (binarization)

    This pipeline is optimized for small, low-quality odometer digit crops
    common in industrial mobile phone captures.
    """
    h, w = crop_bgr.shape[:2]

    # 1. Aggressive upscaling (4x)
    scale = 4
    new_w, new_h = w * scale, h * scale
    upscaled = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 2. Convert to grayscale
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)

    # 3. Denoise while preserving edges
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # 4. Enhance contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 5. Adaptive threshold for binarization
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Convert back to BGR for Tesseract
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def run_tesseract_ocr(crop_bgr: np.ndarray) -> OcrResult:
    """Run Tesseract OCR over a crop with optimized settings for vertical text.

    Uses PSM 12 (sparse text with OSD) which works best for difficult crops.
    The LSTM engine (OEM 1) handles rotation and text direction better than
    the legacy engine.

    For industrial robustness with low-resolution images, applies aggressive
    preprocessing: 4x upscaling, denoising, contrast enhancement, and
    binarization before OCR.

    Returns concatenated text from all detected regions with the minimum
    confidence as a conservative estimate.
    """
    pytesseract = _require_pytesseract()
    check_tesseract_available()

    # Preprocess for low-resolution images
    processed = preprocess_for_low_res_ocr(crop_bgr)

    # Custom config optimized for processed images
    # PSM 12: Sparse text with OSD (best for difficult crops)
    # --oem 1: Use LSTM neural network engine
    custom_config = r'--oem 1 --psm 12'

    try:
        # Get detailed output with confidence scores
        data = pytesseract.image_to_data(
            processed,
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )

        # Extract text and confidence from detected words
        texts = []
        confidences = []

        for i, text in enumerate(data['text']):
            conf = int(data['conf'][i])
            if conf > 0 and text.strip():  # Filter out empty/low-confidence
                texts.append(text.strip())
                confidences.append(conf / 100.0)  # Convert to 0-1 range

        if not texts:
            return OcrResult(raw_text="", confidence=0.0)

        # Concatenate all text and use minimum confidence
        return OcrResult(raw_text="".join(texts), confidence=min(confidences))

    except Exception as e:
        # Return error details for debugging
        import sys
        print(f"[Tesseract Error] {e}", file=sys.stderr)
        print(f"[Tesseract] Original crop shape: {crop_bgr.shape}", file=sys.stderr)
        print(f"[Tesseract] Processed shape: {processed.shape}", file=sys.stderr)
        return OcrResult(raw_text="", confidence=0.0)


def recognize(engine: str, crop_bgr: np.ndarray) -> OcrResult:
    """Run the requested OCR engine over a rectified crop.

    ``engine`` must be ``"paddle"``, ``"easy"``, or ``"tesseract"``. Lazily
    constructs and caches the reader for the selected engine.

    - PaddleOCR and EasyOCR are configured with angle classification enabled
      to handle text at any orientation natively.
    - Tesseract uses PSM 6 (single uniform vertical block) which is optimized
      for vertically-stacked odometer digits common in industrial dashboards.
    """
    if engine == "paddle":
        reader = get_paddle_reader()
        return run_paddle_ocr(reader, crop_bgr)
    if engine == "easy":
        reader = get_easy_reader()
        return run_easy_ocr(reader, crop_bgr)
    if engine == "tesseract":
        return run_tesseract_ocr(crop_bgr)
    raise ValueError(f"unknown OCR engine '{engine}', expected 'paddle', 'easy', or 'tesseract'")


def extract_digits(raw_text: str) -> str:
    """Strip everything except '0'-'9' from raw_text. No character
    substitution (e.g. does not map 'O' to '0')."""
    return _DIGIT_RE.sub("", raw_text)
