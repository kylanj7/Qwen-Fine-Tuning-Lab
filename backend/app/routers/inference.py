"""Inference/Chat endpoints."""
import os
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..db.models import ChatSession

router = APIRouter()


class ChatMessage(BaseModel):
    """Chat message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[datetime] = None
    tokens: Optional[int] = None
    tokens_per_sec: Optional[float] = None


class ChatRequest(BaseModel):
    """Chat request."""
    model_path: str
    message: str
    session_id: Optional[int] = None
    system_prompt: Optional[str] = ""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048


class ChatSessionResponse(BaseModel):
    """Chat session response."""
    id: int
    model_path: str
    model_name: str
    system_prompt: str
    messages: List[ChatMessage]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class GenerationSettings(BaseModel):
    """Generation settings."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    system_prompt: str = ""


@router.post("/session/create", response_model=ChatSessionResponse)
async def create_session(
    model_path: str,
    model_name: str,
    settings: Optional[GenerationSettings] = None,
    db: Session = Depends(get_db)
):
    """Create a new chat session."""
    if settings is None:
        settings = GenerationSettings()

    session = ChatSession(
        model_path=model_path,
        model_name=model_name,
        system_prompt=settings.system_prompt,
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        max_tokens=settings.max_tokens,
        messages=[]
    )

    db.add(session)
    db.commit()
    db.refresh(session)

    return ChatSessionResponse(
        id=session.id,
        model_path=session.model_path,
        model_name=session.model_name,
        system_prompt=session.system_prompt,
        messages=[],
        created_at=session.created_at,
        updated_at=session.updated_at
    )


@router.get("/session/{session_id}", response_model=ChatSessionResponse)
async def get_session(session_id: int, db: Session = Depends(get_db)):
    """Get a chat session."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = [ChatMessage(**m) for m in (session.messages or [])]

    return ChatSessionResponse(
        id=session.id,
        model_path=session.model_path,
        model_name=session.model_name,
        system_prompt=session.system_prompt,
        messages=messages,
        created_at=session.created_at,
        updated_at=session.updated_at
    )


@router.get("/sessions", response_model=List[ChatSessionResponse])
async def list_sessions(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """List chat sessions."""
    sessions = db.query(ChatSession).order_by(
        ChatSession.updated_at.desc()
    ).offset(skip).limit(limit).all()

    result = []
    for session in sessions:
        messages = [ChatMessage(**m) for m in (session.messages or [])]
        result.append(ChatSessionResponse(
            id=session.id,
            model_path=session.model_path,
            model_name=session.model_name,
            system_prompt=session.system_prompt,
            messages=messages,
            created_at=session.created_at,
            updated_at=session.updated_at
        ))

    return result


@router.delete("/session/{session_id}")
async def delete_session(session_id: int, db: Session = Depends(get_db)):
    """Delete a chat session."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    db.delete(session)
    db.commit()

    return {"status": "deleted", "session_id": session_id}


@router.post("/session/{session_id}/clear")
async def clear_session(session_id: int, db: Session = Depends(get_db)):
    """Clear messages from a chat session."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.messages = []
    db.commit()

    return {"status": "cleared", "session_id": session_id}


@router.put("/session/{session_id}/settings")
async def update_session_settings(
    session_id: int,
    settings: GenerationSettings,
    db: Session = Depends(get_db)
):
    """Update session settings."""
    session = db.query(ChatSession).filter(ChatSession.id == session_id).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.system_prompt = settings.system_prompt
    session.temperature = settings.temperature
    session.top_p = settings.top_p
    session.top_k = settings.top_k
    session.max_tokens = settings.max_tokens
    db.commit()

    return {"status": "updated", "session_id": session_id}
