"""LLM calls via the OpenAI-compatible SDK -> works with many providers
(just change AI_BASE_URL / AI_MODEL_NAME / AI_API_KEY in .env).
Everything here is async, so the server stays responsive while waiting for the API."""
import asyncio
import base64
import json
import re
from typing import Optional
from openai import AsyncOpenAI, RateLimitError, NotFoundError, BadRequestError

from . import config
from .prompts import PRESCRIPTION_PROMPT, LAB_PROMPT, GUARDRAILS, CHAT_SYSTEM, CHAT_DOCS

# One shared client (connection pooling = lower latency on repeat calls)
_client: Optional[AsyncOpenAI] = None

def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=config.AI_API_KEY, base_url=config.AI_BASE_URL,
                              timeout=config.AI_TIMEOUT_SECONDS, max_retries=0)
    return _client


def describe_error(e: Exception) -> str:
    """Readable error text. Google puts the useful message in the raw response body."""
    detail = ""
    try:
        detail = e.response.text[:300]
    except Exception:
        pass
    return f"{type(e).__name__}: {detail or str(e)[:250]}"


def _retry_delay(e: Exception) -> float:
    """Google's 429 says 'retry in 23s'. Wait that long (max 12 s), default 4 s."""
    m = re.search(r"retry\w*\D{0,12}([\d.]+)\s*s", describe_error(e), re.I)
    return min(float(m.group(1)), 12.0) if m else 4.0


async def _complete(messages: list, max_tokens: int = 1500, temperature: float = 0.2) -> str:
    """Try the main model; on 429 wait + retry once, then move to the next backup model.
    On 404 (model retired) move on immediately. Free-tier quotas are per model."""
    last: Exception = RuntimeError("No model configured")
    for model in [config.AI_MODEL_NAME, *config.AI_FALLBACK_MODELS]:
        for attempt in range(2):
            try:
                resp = await _get_client().chat.completions.create(
                    model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
                return resp.choices[0].message.content or ""
            except RateLimitError as e:
                last = e
                if attempt == 0:
                    await asyncio.sleep(_retry_delay(e))
                    continue
                break                      # still limited -> next model
            except (NotFoundError, BadRequestError) as e:
                last = e
                break                      # model missing / rejects request -> next model
    raise last


def _parse_json(raw: str) -> dict:
    """LLMs sometimes wrap JSON in ```fences``` or add chatter. Extract the {...} part."""
    raw = re.sub(r"```(?:json)?", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in model reply")
    return json.loads(raw[start:end + 1])


def _as_list(v) -> list:
    return [str(x) for x in v][:6] if isinstance(v, list) else []


def _user_content(prompt: str, image_jpeg: Optional[bytes]):
    """Plain text, or text + image (vision) when OCR was weak (e.g. handwriting)."""
    if not image_jpeg:
        return prompt
    b64 = base64.b64encode(image_jpeg).decode()
    return [{"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]


async def analyze_prescription(text: str, hints: dict, image_jpeg: Optional[bytes] = None) -> dict:
    prompt = PRESCRIPTION_PROMPT.format(text=text[:4000] or "(OCR found no text - read the image)",
                                        hints=json.dumps(hints.get("medicines", []))[:800])
    raw = await _complete([{"role": "system", "content": GUARDRAILS},
                           {"role": "user", "content": _user_content(prompt, image_jpeg)}])
    d = _parse_json(raw)
    meds = d.get("medicines") if isinstance(d.get("medicines"), list) else []
    return {"condition": str(d.get("condition", "Unclear"))[:250],
            "medicines": [m for m in meds if isinstance(m, dict)][:15],
            "dos": _as_list(d.get("dos")), "donts": _as_list(d.get("donts")),
            "diet_lifestyle": _as_list(d.get("diet_lifestyle")),
            "prevention": _as_list(d.get("prevention")),
            "summary": str(d.get("summary", ""))}


async def analyze_lab_report(text: str, hints: dict, image_jpeg: Optional[bytes] = None) -> dict:
    prompt = LAB_PROMPT.format(text=text[:5000] or "(OCR found no text - read the image)",
                               hints=json.dumps(hints.get("parameters", []))[:1500])
    raw = await _complete([{"role": "system", "content": GUARDRAILS},
                           {"role": "user", "content": _user_content(prompt, image_jpeg)}],
                          max_tokens=2000)
    d = _parse_json(raw)
    params = d.get("parameters") if isinstance(d.get("parameters"), list) else []
    return {"parameters": [p for p in params if isinstance(p, dict)][:40],
            "dos": _as_list(d.get("dos")), "donts": _as_list(d.get("donts")),
            "diet_lifestyle": _as_list(d.get("diet_lifestyle")),
            "follow_up": _as_list(d.get("follow_up")),
            "summary": str(d.get("summary", ""))}


async def chat_reply(message: str, history: list[dict], doc_ctxs: list[dict], profile: Optional[dict] = None) -> str:
    """Natural chat. doc_ctxs = every uploaded document to use as context (prescription and/or lab report).
    `history` is the visible conversation, so replies stay consistent and do not repeat themselves."""
    system = CHAT_SYSTEM
    if profile:
        system += "\n\nKnown about the user (use only when relevant, never recite it): " + "; ".join(f"{k}: {v}" for k, v in profile.items())
    if doc_ctxs:
        blocks = [f"### {d['doc_type'].replace('_', ' ').title()}\n--- scanned text ---\n{d['text'][:2500]}\n"
                  f"--- AI summary ---\n{json.dumps(d['analysis'], default=str)[:1800]}" for d in doc_ctxs]
        system += CHAT_DOCS.format(docs="\n\n".join(blocks))
    messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": message}]
    return (await _complete(messages, max_tokens=1000, temperature=0.6)).strip()