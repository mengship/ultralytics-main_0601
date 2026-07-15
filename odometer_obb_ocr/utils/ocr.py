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


_PADDLE_SINGLETON = None
_EASY_SINGLETON = None


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


def recognize(engine: str, crop_bgr: np.ndarray) -> OcrResult:
    """Run the requested OCR engine over a rectified crop.

    ``engine`` must be ``"paddle"`` or ``"easy"``. Lazily constructs and
    caches the reader for the selected engine.

    Both engines are configured to handle text at any orientation (vertical,
    rotated, upside-down) natively through their angle classification models,
    so no manual rotation is required.
    """
    if engine == "paddle":
        reader = get_paddle_reader()
        return run_paddle_ocr(reader, crop_bgr)
    if engine == "easy":
        reader = get_easy_reader()
        return run_easy_ocr(reader, crop_bgr)
    raise ValueError(f"unknown OCR engine '{engine}', expected 'paddle' or 'easy'")


def extract_digits(raw_text: str) -> str:
    """Strip everything except '0'-'9' from raw_text. No character
    substitution (e.g. does not map 'O' to '0')."""
    return _DIGIT_RE.sub("", raw_text)
