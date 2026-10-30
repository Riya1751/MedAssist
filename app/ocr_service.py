"""OCR with Tesseract (fast, free, runs locally, no API cost)."""
import cv2
import numpy as np
import pytesseract
from pytesseract import Output

TESS_CONFIG = "--oem 3 --psm 6"


def _downscale_if_large(img, max_dim: int = 1400):
    """Downscales images to ~1400px max dimension for fast processing."""
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def ocr_image(img) -> tuple[str, float]:
    """Returns (text, average_confidence_0_to_100)."""
    # 1. Image ko 1400px par resize karo taaki Tesseract turant run ho
    processed_img = _downscale_if_large(img)

    try:
        # Timeout 25 second rakha hai
        data = pytesseract.image_to_data(
            processed_img,
            config=TESS_CONFIG,
            lang="eng",
            output_type=Output.DICT,
            timeout=25,
        )
    except Exception as e:
        # Kisi bhi error ya timeout par crash mat karo, seedha Vision LLM fallback trigger hoga
        print(f"[OCR warning] Tesseract failed or timed out: {e}")
        return "", 0.0

    lines: dict[tuple, list[str]] = {}
    confs: list[float] = []
    for i, word in enumerate(data.get("text", [])):
        word = word.strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confs.append(conf)

    text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return text, round(avg_conf, 1)





