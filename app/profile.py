"""User profile endpoints (additive - nothing else in the backend changes).
The `user_profiles` table is created automatically the first time it is used,
so you do NOT need to run any extra SQL."""
import re
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Integer, String, Text, ForeignKey, DateTime, select, func
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base, engine, SessionLocal, get_user_by_token

router = APIRouter()
FIELDS = ["full_name", "age", "gender", "blood_group", "phone", "email", "allergies", "conditions"]


class UserProfile(Base):
    __tablename__ = "user_profiles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    full_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    gender: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    blood_group: Mapped[Optional[str]] = mapped_column(String(5), nullable=True)
    phone: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    email: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    allergies: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    conditions: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    updated_at: Mapped[Optional[object]] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


_table_ready = False
def _ensure_table():
    global _table_ready
    if not _table_ready:
        UserProfile.__table__.create(bind=engine, checkfirst=True)   # no-op if it already exists
        _table_ready = True


class ProfileIn(BaseModel):
    full_name: Optional[str] = Field(None, max_length=100)
    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[str] = Field(None, max_length=20)
    blood_group: Optional[str] = Field(None, max_length=5)
    phone: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=120)
    allergies: Optional[str] = Field(None, max_length=500)
    conditions: Optional[str] = Field(None, max_length=500)

    @field_validator("email")
    @classmethod
    def _email(cls, v):
        if v and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v):
            raise ValueError("Enter a valid email address")
        return v

    @field_validator("phone")
    @classmethod
    def _phone(cls, v):
        if v and not re.fullmatch(r"\+?[\d\s-]{7,18}", v):
            raise ValueError("Enter a valid contact number")
        return v


async def _user(x_session_token: str = Header(...)) -> dict:
    user = await run_in_threadpool(get_user_by_token, x_session_token)
    if not user:
        raise HTTPException(401, "Invalid session.")
    return user


def _load(user_id: int) -> dict:
    _ensure_table()
    with SessionLocal() as s:
        p = s.scalar(select(UserProfile).where(UserProfile.user_id == user_id))
        return {k: (getattr(p, k) if p else None) for k in FIELDS}


def _save(user_id: int, data: dict) -> None:
    _ensure_table()
    with SessionLocal() as s:
        p = s.scalar(select(UserProfile).where(UserProfile.user_id == user_id))
        if not p:
            p = UserProfile(user_id=user_id)
            s.add(p)
        for k in FIELDS:
            setattr(p, k, data.get(k))
        s.commit()


@router.get("/api/profile")
async def get_profile(user: dict = Depends(_user)):
    return await run_in_threadpool(_load, user["id"])


@router.put("/api/profile")
async def save_profile(body: ProfileIn, user: dict = Depends(_user)):
    await run_in_threadpool(_save, user["id"], body.model_dump())
    return {"ok": True}


def profile_for_ai(user_id: int) -> dict:
    """Passes user profile details including name to the AI."""
    p = _load(user_id)
    names = {
        "full_name": "name",
        "age": "age",
        "gender": "gender",
        "blood_group": "blood group",
        "allergies": "allergies",
        "conditions": "existing conditions"
    }
    return {names[k]: p[k] for k in names if p.get(k)}