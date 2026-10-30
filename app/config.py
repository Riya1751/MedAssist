"""Central place for all settings. Everything comes from the .env file."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # reads .env into environment variables

def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes")

# --- LLM settings (vendor-neutral: works with any OpenAI-compatible API) ---
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_BASE_URL = os.getenv("AI_BASE_URL", "").strip() or None   # None -> SDK default
AI_USE_VISION = _bool("AI_USE_VISION", True)
# Backup models, tried in order when the main one is rate-limited (429) or missing (404)
# AI_MODEL_NAME may (by mistake) contain "a,b,c" -> first = main model, rest = backups
_main = [m.strip() for m in os.getenv("AI_MODEL_NAME", "").split(",") if m.strip()]
_fb = [m.strip() for m in os.getenv("AI_FALLBACK_MODELS", "").split(",") if m.strip()]
AI_MODEL_NAME = _main[0] if _main else ""
AI_FALLBACK_MODELS = list(dict.fromkeys(_main[1:] + _fb))   # merge, remove duplicates
AI_TIMEOUT_SECONDS = float(os.getenv("AI_TIMEOUT_SECONDS", "30"))

# --- Database ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "medassist")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "medassist")

# --- App tuning ---
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
IMAGE_MAX_SIDE = int(os.getenv("IMAGE_MAX_SIDE", "1600"))
OCR_LOW_CONFIDENCE = float(os.getenv("OCR_LOW_CONFIDENCE", "60"))
MAX_PDF_PAGES = 3                      # keep latency low: only read the first pages
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

def ai_configured() -> bool:
    """True only if the user actually filled in the API settings."""
    return bool(AI_API_KEY and AI_MODEL_NAME) and "your_" not in AI_API_KEY