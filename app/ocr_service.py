"""OCR with Tesseract (fast, free, runs locally, no API cost)."""
import pytesseract
from pytesseract import Output

# --oem 3 = default engine, --psm 6 = treat page as one uniform block of text
TESS_CONFIG = "--oem 3 --psm 6"


def ocr_image(img) -> tuple[str, float]:
    """Returns (text, average_confidence_0_to_100).

    We use image_to_data (not image_to_string) so we also get a per-word
    confidence score. Low confidence (e.g. doctor's handwriting) triggers the
    vision-LLM fallback in main.py.
    """
    data = pytesseract.image_to_data(img, config=TESS_CONFIG, lang="eng",
                                     output_type=Output.DICT, timeout=90)
    lines: dict[tuple, list[str]] = {}
    confs: list[float] = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:           # conf == -1 means "not a word"
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confs.append(conf)

    text = "\n".join(" ".join(words) for _, words in sorted(lines.items()))
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return text, round(avg_conf, 1)
