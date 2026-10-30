"""OCR with Tesseract (fast, free, runs locally, no API cost)."""
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# --oem 3 = default engine, --psm 6 = treat page as one uniform block of text
TESS_CONFIG = "--oem 3 --psm 6"


def _downscale_if_large(img, max_dim: int = 1600):
    """Downscales images to ~1600px max dimension.
    Prevents Tesseract from hanging on 4K/8MB+ high-resolution uploads.
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def ocr_image(img) -> tuple[str, float]:
    """Returns (text, average_confidence_0_to_100).

    We use image_to_data (not image_to_string) so we also get a per-word
    confidence score. Low confidence (e.g. doctor's handwriting or timeout) triggers
    the vision-LLM fallback in main.py.
    """
    # Downscale high-resolution images first to speed up OCR by 4-5x
    processed_img = _downscale_if_large(img)

    try:
        data = pytesseract.image_to_data(
            processed_img,
            config=TESS_CONFIG,
            lang="eng",
            output_type=Output.DICT,
            timeout=40,  # Generous safety window for Render shared vCPUs
        )
    except (RuntimeError, pytesseract.pytesseract.TesseractTimeoutError, Exception) as e:
        # If Tesseract times out or fails, return empty text + 0 confidence
        # to trigger the vision LLM fallback in main.py cleanly
        print(f"[OCR warning] Tesseract skipped or timed out: {e}")
        return "", 0.0

    lines: dict[tuple, list[str]] = {}
    confs: list[float] = []
    for i, word in enumerate(data.get("text", [])):
        word = word.strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:  # conf == -1 means "not a word"
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confs.append(conf)

    text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return text, round(avg_conf, 1)
