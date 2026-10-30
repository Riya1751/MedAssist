"""Finds which model names work with YOUR API key.   Run:  python check_ai.py"""
from openai import OpenAI
from app import config

if not config.ai_configured():
    raise SystemExit("Set AI_API_KEY and AI_MODEL_NAME in .env first.")

client = OpenAI(api_key=config.AI_API_KEY, base_url=config.AI_BASE_URL, timeout=30)

print("Asking your provider which models it offers...\n")
try:
    names = [m.id.replace("models/", "") for m in client.models.list()]
except Exception as e:
    print("Could not list models:", type(e).__name__, str(e)[:200])
    names = []

# Only text-capable "flash" models are cheap + fast enough for this project
flash = [n for n in names if "flash" in n and not any(x in n for x in ("image", "tts", "live", "audio"))]
others = [n for n in names if not any(x in n.lower() for x in ("whisper", "guard", "tts", "embed", "image", "audio", "live"))]
candidates = list(dict.fromkeys([config.AI_MODEL_NAME] + (flash[:12] or others[:10])))
print("Testing:", ", ".join(candidates), "\n")

working = []
for name in candidates:
    try:
        client.chat.completions.create(model=name, max_tokens=30,
                                       messages=[{"role": "user", "content": "Say OK"}])
        print(f"  WORKS   {name}")
        working.append(name)
    except Exception as e:
        print(f"  FAILS   {name}   ({type(e).__name__})")

print()
if working:
    print(f"Put this in .env ->  AI_MODEL_NAME={working[0]}")
    if len(working) > 1:
        print(f"Backups (for rate limits) ->  AI_FALLBACK_MODELS={','.join(working[1:3])}")
else:
    print("No model worked. Check AI_API_KEY and AI_BASE_URL in .env.")