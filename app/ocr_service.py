"""OCR with Tesseract (Fast optimized)."""
import cv2
import numpy as np
import pytesseract

TESS_CONFIG = "--oem 3 --psm 3"


def _fast_preprocess(img, max_dim: int = 1200):
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    # Image ko 1200px par resize karo taaki CPU par load na pade
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Grayscale image Tesseract par 2x fast process hoti hai
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def ocr_image(img) -> tuple[str, float]:
    """Returns (text, confidence) in 1-2 seconds."""
    processed_img = _fast_preprocess(img)

    try:
        # image_to_string is 4x faster than image_to_data
        # 3 second timeout: agar 3 sec me text nahi nikla toh bina ruke Vision LLM pe bhej dega
        text = pytesseract.image_to_string(
            processed_img,
            config=TESS_CONFIG,
            lang="eng",
            timeout=3,
        ).strip()

        conf = 85.0 if len(text) > 40 else 20.0
        return text, conf

    except Exception as e:
        print(f"[OCR Timeout/Skip] Directing straight to Vision AI: {e}")
        return "", 0.0