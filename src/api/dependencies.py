"""
Dependency injection for the Capstone RAG API.

Provides reusable FastAPI dependencies for settings, RAG agent, session
management, and metrics tracking.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from threading import Lock
from typing import AsyncGenerator

from fastapi import Request

from src.api.schemas import (
    CitationResponse,
    FeedbackRequest,
    MetricsResponse,
    SessionMessage,
    SessionResponse,
)
from src.config import Settings, get_settings
from src.graph import RAGAgent, get_rag_agent
from src.utils import get_logger

logger = get_logger("api.dependencies")

# ── Agent Singleton ─────────────────────────────────────────────────────────

_agent: RAGAgent | None = None
_agent_lock = Lock()


def get_agent(settings: Settings | None = None) -> RAGAgent:
    """Get or create the shared RAGAgent singleton (thread-safe)."""
    global _agent
    if _agent is None:
        with _agent_lock:
            if _agent is None:
                _agent = get_rag_agent(settings or get_settings())
                logger.info("RAGAgent singleton created")
    return _agent


def close_agent() -> None:
    """Close the agent singleton and release resources."""
    global _agent
    if _agent is not None:
        _agent.close()
        _agent = None
        logger.info("RAGAgent singleton closed")


# ── Session Store (in-memory) ───────────────────────────────────────────────

_sessions: dict[str, SessionResponse] = {}
_sessions_lock = Lock()


def get_or_create_session(session_id: str) -> SessionResponse:
    """Retrieve an existing session or create a new one."""
    with _sessions_lock:
        if session_id not in _sessions:
            now = datetime.now(timezone.utc).isoformat()
            _sessions[session_id] = SessionResponse(
                session_id=session_id,
                created_at=now,
                updated_at=now,
            )
        return _sessions[session_id]


def add_message_to_session(
    session_id: str,
    role: str,
    content: str,
    citations: list[CitationResponse] | None = None,
    refused: bool = False,
) -> None:
    """Append a message to a session's history."""
    session = get_or_create_session(session_id)
    with _sessions_lock:
        session.messages.append(
            SessionMessage(
                role=role,
                content=content,
                citations=citations or [],
                refused=refused,
            )
        )
        session.updated_at = datetime.now(timezone.utc).isoformat()


# ── Metrics Store ────────────────────────────────────────────────────────────

_start_time = time.time()
_metrics_lock = Lock()
_metrics = {
    "total_queries": 0,
    "total_refusals": 0,
    "total_response_time_ms": 0.0,
    "total_citations_given": 0,
}


def record_query(response_time_ms: float, refused: bool, citations_count: int) -> None:
    """Record metrics for a single query."""
    with _metrics_lock:
        _metrics["total_queries"] += 1
        _metrics["total_response_time_ms"] += response_time_ms
        if refused:
            _metrics["total_refusals"] += 1
        _metrics["total_citations_given"] += citations_count


def get_metrics() -> MetricsResponse:
    """Return current metrics snapshot."""
    with _metrics_lock:
        total = _metrics["total_queries"]
        avg_time = (
            round(_metrics["total_response_time_ms"] / total, 2)
            if total > 0
            else 0.0
        )
        return MetricsResponse(
            total_queries=total,
            total_refusals=_metrics["total_refusals"],
            avg_response_time_ms=avg_time,
            total_citations_given=_metrics["total_citations_given"],
            uptime_seconds=round(time.time() - _start_time, 2),
        )


# ── Feedback Store ───────────────────────────────────────────────────────────

_feedback_store: list[dict] = []
_feedback_lock = Lock()


def store_feedback(request: FeedbackRequest) -> str:
    """Store feedback and return a feedback ID."""
    feedback_id = str(uuid.uuid4())
    with _feedback_lock:
        _feedback_store.append(
            {
                "feedback_id": feedback_id,
                "session_id": request.session_id,
                "message_index": request.message_index,
                "rating": request.rating,
                "comment": request.comment,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
    return feedback_id


# ── FastAPI Dependencies ────────────────────────────────────────────────────


async def get_settings_dependency() -> Settings:
    """Dependency that provides application settings."""
    return get_settings()


async def get_agent_dependency() -> RAGAgent:
    """Dependency that provides the shared RAGAgent."""
    return get_agent()