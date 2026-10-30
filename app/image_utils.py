"""Image optimization: the biggest lever for hitting the 1-2 second target.

Why it matters:
  * A 12 MP phone photo is ~4-6 MB. Tesseract on that takes many seconds.
  * Shrinking to ~1600px long side keeps text readable but is 10x faster.
  * Sending a small JPEG (not the original) to a vision LLM cuts upload time.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from . import config


@dataclass
class OptimizedImage:
    ocr_image: np.ndarray    # grayscale + cleaned, best for Tesseract
    jpeg_bytes: bytes        # small colour JPEG, for storage / vision-LLM fallback


def decode_image(raw: bytes) -> np.ndarray:
    """bytes -> OpenCV BGR image (also applies phone EXIF rotation)."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("File is not a readable image.")
    return img


def _resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:                       # already small enough
        return img
    # INTER_AREA is the best interpolation for shrinking
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def optimize_image(img: np.ndarray) -> OptimizedImage:
    # 1) Downscale big photos
    small = _resize_max_side(img, config.IMAGE_MAX_SIDE)

    # 2) Compressed colour copy (quality 75 is visually fine, ~150-300 KB)
    ok, buf = cv2.imencode(".jpg", _resize_max_side(small, 1280), [cv2.IMWRITE_JPEG_QUALITY, 75])
    jpeg_bytes = buf.tobytes() if ok else b""

    # 3) Grayscale: Tesseract doesn't need colour
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 4) Very small images: upscale so characters are ~20+ px tall (helps OCR accuracy)
    if max(gray.shape) < 1000:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # 5) Light blur removes camera grain (much faster than fastNlMeansDenoising)
    gray = cv2.medianBlur(gray, 3)

    # 6) Adaptive threshold: fixes uneven lighting / shadows on phone photos.
    #    It computes a threshold per small neighbourhood instead of one global value.
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 15)
    return OptimizedImage(ocr_image=binary, jpeg_bytes=jpeg_bytes)
