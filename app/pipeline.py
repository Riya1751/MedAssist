"""Turns an uploaded file (photo or PDF) into clean text. Synchronous + CPU-bound,
so main.py runs it in a worker thread via asyncio.to_thread()."""
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
try:
    import pymupdf as fitz   # new name
except ImportError:
    import fitz              # older PyMuPDF

from . import config
from .image_utils import decode_image, optimize_image
from .ocr_service import ocr_image
from .text_utils import clean_ocr_text


@dataclass
class Extracted:
    text: str
    confidence: float
    engine: str
    jpeg_bytes: Optional[bytes]   # None for digital PDFs (no image needed)


def extract_text(raw: bytes, filename: str, content_type: str) -> Extracted:
    is_pdf = content_type == "application/pdf" or filename.lower().endswith(".pdf")
    return _from_pdf(raw) if is_pdf else _from_image(raw)


def _from_image(raw: bytes) -> Extracted:
    opt = optimize_image(decode_image(raw))
    text, conf = ocr_image(opt.ocr_image)
    return Extracted(clean_ocr_text(text), conf, "tesseract", opt.jpeg_bytes)


def _from_pdf(raw: bytes) -> Extracted:
    doc = fitz.open(stream=raw, filetype="pdf")
    pages = min(len(doc), config.MAX_PDF_PAGES)

    # FAST PATH: digital lab PDFs already contain real text -> no OCR needed (~50 ms)
    embedded = "\n".join(doc[i].get_text() for i in range(pages))
    if len(embedded.strip()) > 100:
        return Extracted(clean_ocr_text(embedded), 100.0, "pdf-text", None)

    # SLOW PATH: scanned PDF -> render each page to an image, then OCR it
    texts, confs, first_jpeg = [], [], None
    for i in range(pages):
        pix = doc[i].get_pixmap(dpi=200, colorspace=fitz.csRGB, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        opt = optimize_image(img)
        t, c = ocr_image(opt.ocr_image)
        texts.append(t); confs.append(c)
        first_jpeg = first_jpeg or opt.jpeg_bytes
    avg = sum(confs) / len(confs) if confs else 0.0
    return Extracted(clean_ocr_text("\n".join(texts)), round(avg, 1), "tesseract-pdf", first_jpeg)