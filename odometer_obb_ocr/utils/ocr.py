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

    PaddleOCR's constructor kwargs (e.g. ``use_angle_cls`` vs ``use_textline_orientation``,
    charset-restriction options) vary across versions, so we try a couple of
    known-good spellings and fall back to a bare ``PaddleOCR(lang="en")`` if
    none of them are accepted. There is no reliable cross-version kwarg for
    restricting recognition to a charset; that restriction is instead applied
    as post-processing via ``extract_digits``.
    """
    global _PADDLE_SINGLETON
    if _PADDLE_SINGLETON is None:
        paddle_ocr_cls = _require_paddleocr()
        for kwargs in (
            {"use_angle_cls": False, "lang": "en"},
            {"use_textline_orientation": False, "lang": "en"},
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


def run_paddle_ocr(reader, crop_bgr: np.ndarray) -> OcrResult:
    """Run a PaddleOCR reader over a crop and parse its raw output.

    PaddleOCR's ``.ocr()`` return shape is ``[[ [box, (text, score)], ... ]]``
    per image. We concatenate all recognized text fragments and take the
    minimum score across them as a conservative confidence estimate.
    """
    try:
        raw = reader.ocr(crop_bgr, cls=False)
    except TypeError:
        raw = reader.ocr(crop_bgr)

    if not raw or raw[0] is None:
        return OcrResult(raw_text="", confidence=0.0)

    lines = raw[0]
    texts = []
    scores = []
    for _box, (text, score) in lines:
        texts.append(text)
        scores.append(float(score))

    if not texts:
        return OcrResult(raw_text="", confidence=0.0)

    return OcrResult(raw_text="".join(texts), confidence=min(scores))


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
