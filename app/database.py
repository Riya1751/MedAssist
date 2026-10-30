"""SQLAlchemy models + small CRUD helpers.

All functions here are *synchronous*. In main.py we call them through
`run_in_threadpool`, so they never block the async event loop.
Each helper returns plain dicts/ints (not ORM objects) to avoid
"detached instance" errors for beginners.
"""
from datetime import datetime
from typing import Optional
import secrets

from sqlalchemy import (create_engine, ForeignKey, Integer, String, Text, Float,
                        DateTime, JSON, select, func)
from sqlalchemy.engine import URL
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.dialects.mysql import MEDIUMTEXT

from . import config

# Build the connection URL safely (handles special characters in the password)
db_url = URL.create("mysql+pymysql", username=config.DB_USER, password=config.DB_PASSWORD,
                    host=config.DB_HOST, port=config.DB_PORT, database=config.DB_NAME,
                    query={"charset": "utf8mb4"})
engine = create_engine(db_url, pool_pre_ping=True, pool_recycle=3600, pool_size=5)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
def init_db():
    Base.metadata.create_all(bind=engine)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_token: Mapped[str] = mapped_column(String(64), unique=True)
    username: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    doc_type: Mapped[str] = mapped_column(String(20))
    original_name: Mapped[str] = mapped_column(String(255))
    stored_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    raw_ocr_text: Mapped[Optional[str]] = mapped_column(MEDIUMTEXT, nullable=True)
    ocr_engine: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    ocr_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), unique=True)
    condition_text: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    medicines_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    dos_donts_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    lab_flags_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    summary_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(20), default="llm")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ChatMessage(Base):
    __tablename__ = "chat_history"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    document_id: Mapped[Optional[int]] = mapped_column(ForeignKey("documents.id"), nullable=True)
    role: Mapped[str] = mapped_column(String(20))
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


# ----------------------------- CRUD helpers -----------------------------

def create_guest_user() -> dict:
    token = secrets.token_hex(32)           # 64 hex chars, unguessable
    with SessionLocal() as s:
        u = User(session_token=token, username="guest")
        s.add(u); s.commit()
        return {"id": u.id, "session_token": token}


def get_user_by_token(token: str) -> Optional[dict]:
    with SessionLocal() as s:
        u = s.scalar(select(User).where(User.session_token == token))
        return {"id": u.id} if u else None


def save_document(user_id, doc_type, name, path, text, engine_name, conf) -> int:
    with SessionLocal() as s:
        d = Document(user_id=user_id, doc_type=doc_type, original_name=name[:255],
                     stored_path=path, raw_ocr_text=text, ocr_engine=engine_name,
                     ocr_confidence=conf)
        s.add(d); s.commit()
        return d.id


def save_analysis(document_id: int, result: dict, source: str) -> None:
    """`result` is the normalized dict produced by llm_service / rule-based fallback."""
    with SessionLocal() as s:
        s.add(AnalysisResult(
            document_id=document_id,
            condition_text=(result.get("condition") or "")[:255] or None,
            medicines_json=result.get("medicines"),
            dos_donts_json={k: result.get(k, []) for k in
                            ("dos", "donts", "diet_lifestyle", "prevention", "follow_up")},
            lab_flags_json=result.get("parameters"),
            summary_text=result.get("summary"),
            source=source))
        s.commit()


def get_document_context(user_id: int, document_id: int) -> Optional[dict]:
    """Fetch OCR text + analysis for the chatbot. Returns None if not the user's document."""
    with SessionLocal() as s:
        d = s.scalar(select(Document).where(Document.id == document_id,
                                            Document.user_id == user_id))
        if not d:
            return None
        a = s.scalar(select(AnalysisResult).where(AnalysisResult.document_id == d.id))
        analysis = None
        if a:
            analysis = {"condition": a.condition_text, "medicines": a.medicines_json,
                        "advice": a.dos_donts_json, "lab_parameters": a.lab_flags_json,
                        "summary": a.summary_text}
        return {"doc_type": d.doc_type, "text": d.raw_ocr_text or "", "analysis": analysis}


def save_chat(user_id: int, document_id: Optional[int], role: str, message: str) -> None:
    with SessionLocal() as s:
        s.add(ChatMessage(user_id=user_id, document_id=document_id, role=role, message=message))
        s.commit()


def get_recent_chat(user_id: int, document_id: Optional[int], limit: int = 8) -> list[dict]:
    """Last few turns (oldest first) so the LLM remembers the conversation."""
    with SessionLocal() as s:
        q = select(ChatMessage).where(ChatMessage.user_id == user_id)
        q = q.where(ChatMessage.document_id == document_id) if document_id \
            else q.where(ChatMessage.document_id.is_(None))
        rows = s.scalars(q.order_by(ChatMessage.id.desc()).limit(limit)).all()
        return [{"role": r.role, "content": r.message} for r in reversed(rows)]
