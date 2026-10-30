"""FastAPI entry point.

Async strategy (why the server never "freezes"):
  * OCR/OpenCV are CPU-bound   -> asyncio.to_thread()    (worker thread)
  * MySQL calls are blocking   -> run_in_threadpool()    (worker thread)
  * LLM HTTP calls             -> AsyncOpenAI (true async I/O)
So while one user waits for the LLM, other requests are still served.
"""
import asyncio
from contextlib import asynccontextmanager
import logging
import time
import uuid
from typing import Literal, Optional

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from . import config, database as db, llm_service
from .pipeline import extract_text
from .prompts import DISCLAIMER
from .text_utils import rule_based_prescription, rule_based_lab, is_emergency, is_crisis
from .profile import profile_for_ai

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("medassist")

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    yield

app = FastAPI(title="MedAssist API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "application/pdf"}


# ------------------------------------------------------------------ auth
async def current_user(x_session_token: str = Header(...)) -> dict:
    """Every protected endpoint needs the header  X-Session-Token: <token>."""
    user = await run_in_threadpool(db.get_user_by_token, x_session_token)
    if not user:
        raise HTTPException(401, "Invalid session. Call POST /api/session first.")
    return user


@app.get("/api/health")
async def health():
    return {"status": "ok", "ai_configured": config.ai_configured()}


@app.post("/api/session")
async def create_session():
    """Creates a guest user and returns a token for the frontend to store."""
    user = await run_in_threadpool(db.create_guest_user)
    return {"session_token": user["session_token"]}


# --------------------------------------------------------- upload pipeline
async def process_upload(doc_type: str, file: UploadFile, user: dict) -> dict:
    t0 = time.perf_counter()
    raw = await file.read()
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Upload a JPG, PNG, WEBP image or a PDF.")
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (max {config.MAX_UPLOAD_BYTES // 1024 // 1024} MB).")
    if doc_type == "prescription" and file.content_type == "application/pdf":
        pass  # allowed: some clinics email PDFs

    # 1) Optimize + OCR in a worker thread
    try:
        ex = await asyncio.to_thread(extract_text, raw, file.filename or "upload", file.content_type)
    except Exception as e:
        log.exception("Extraction failed")
        raise HTTPException(422, f"Could not read this file: {e}")
    t_ocr = time.perf_counter()

    # 2) Cheap rule-based pass (always runs; also acts as hints + fallback)
    rule_fn = rule_based_prescription if doc_type == "prescription" else rule_based_lab
    rule_result = rule_fn(ex.text)

    # 3) LLM pass. Send the image too only when OCR is weak (handwriting, blurry photo)
    use_vision = (config.AI_USE_VISION and ex.jpeg_bytes is not None and
                  (ex.confidence < config.OCR_LOW_CONFIDENCE or len(ex.text) < 40))
    if len(ex.text) < 15 and not use_vision:
        raise HTTPException(422, "Couldn't read any text. Try a clearer, well-lit, straight photo.")

    result, source, ai_error = rule_result, "rule_based", None
    if not config.ai_configured():
        ai_error = "AI_API_KEY / AI_MODEL_NAME are not set in .env"
    else:
        try:
            fn = llm_service.analyze_prescription if doc_type == "prescription" \
                else llm_service.analyze_lab_report
            result = await fn(ex.text, rule_result, ex.jpeg_bytes if use_vision else None)
            source = "llm"
        except Exception as e:
            log.exception("LLM failed - using rule-based fallback")
            ai_error = llm_service.describe_error(e)
    t_ai = time.perf_counter()

    # 4) Save the (small, optimized) image + DB rows
    stored_path = None
    if ex.jpeg_bytes:
        stored_path = str(config.UPLOAD_DIR / f"{uuid.uuid4().hex}.jpg")
        await asyncio.to_thread(lambda: open(stored_path, "wb").write(ex.jpeg_bytes))
    doc_id = await run_in_threadpool(db.save_document, user["id"], doc_type,
                                     file.filename or "upload", stored_path,
                                     ex.text, ex.engine, ex.confidence)
    await run_in_threadpool(db.save_analysis, doc_id, result, source)

    return {
        "document_id": doc_id, "doc_type": doc_type, "analysis_source": source,
        "ocr": {"engine": ex.engine, "confidence": ex.confidence, "characters": len(ex.text),
                "used_vision_fallback": use_vision},
        "analysis": result, "ai_error": ai_error, "disclaimer": DISCLAIMER,
        "timings_ms": {"ocr": round((t_ocr - t0) * 1000), "ai": round((t_ai - t_ocr) * 1000),
                       "total": round((time.perf_counter() - t0) * 1000)},
    }


@app.post("/api/upload-prescription")
async def upload_prescription(file: UploadFile = File(...), user: dict = Depends(current_user)):
    return await process_upload("prescription", file, user)


@app.post("/api/upload-lab-report")
async def upload_lab_report(file: UploadFile = File(...), user: dict = Depends(current_user)):
    return await process_upload("lab_report", file, user)


# -------------------------------------------------------------------- chat
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2500)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    document_id: Optional[int] = None                                      # old single-document field (still accepted)
    document_ids: list[int] = Field(default_factory=list, max_length=4)    # prescription and/or lab report
    history: list[ChatTurn] = Field(default_factory=list, max_length=20)   # the visible conversation so far


@app.post("/api/chat")
async def chat(req: ChatRequest, user: dict = Depends(current_user)):
    # 1) Collect every document the user wants the assistant to use (0 / duplicates ignored)
    ids = list(dict.fromkeys(i for i in [*req.document_ids, req.document_id] if i))
    doc_ctxs, valid_ids = [], []
    for i in ids[:4]:
        c = await run_in_threadpool(db.get_document_context, user["id"], i)
        if c:
            doc_ctxs.append(c); valid_ids.append(i)
    if ids and not doc_ctxs:
        raise HTTPException(404, "Document not found.")
    log_doc = valid_ids[0] if valid_ids else None

    await run_in_threadpool(db.save_chat, user["id"], log_doc, "user", req.message)

    # 2) Safety nets run BEFORE the LLM (instant + reliable)
    if is_crisis(req.message):
        reply = ("I'm really sorry you're going through this, and I'm glad you said it out loud. You deserve support right now. "
                 "If you might act on these thoughts or you're in danger, please call your local emergency number (112 in India) "
                 "or go to the nearest hospital. You can also talk to someone for free, any time, on Tele-MANAS at 14416. "
                 "If someone is nearby, please tell them how you feel too. I'm here to keep talking if you'd like.")
    elif is_emergency(req.message):
        reply = "This could be serious. Please call your local emergency number (112 in India) or get to the nearest hospital right away."
    elif not config.ai_configured():
        reply = "The AI service isn't configured yet. Add AI_API_KEY and AI_MODEL_NAME to .env."
    else:
        profile = await run_in_threadpool(profile_for_ai, user["id"])
        history = [{"role": h.role, "content": h.content} for h in req.history[-8:]]
        try:
            reply = await llm_service.chat_reply(req.message, history, doc_ctxs, profile)
        except Exception as e:
            log.exception("Chat LLM failed")
            raise HTTPException(502, f"AI request failed - {llm_service.describe_error(e)}. "
                                     "Run `python check_ai.py` to find a working model name.")

    await run_in_threadpool(db.save_chat, user["id"], log_doc, "assistant", reply)
    return {"reply": reply, "mode": "document" if doc_ctxs else "general", "documents": [c["doc_type"] for c in doc_ctxs]}


# ---- profile endpoints (separate file) ----
from .profile import router as profile_router
app.include_router(profile_router)

# ------------------------------------------------------------ frontend
# Must be LAST so the /api routes above take priority. Open http://127.0.0.1:8000
from fastapi.staticfiles import StaticFiles
_FRONTEND = config.UPLOAD_DIR.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/", StaticFiles(directory=_FRONTEND, html=True), name="frontend")