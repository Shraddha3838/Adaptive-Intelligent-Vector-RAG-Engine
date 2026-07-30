"""
Pydantic request/response schemas for the Capstone RAG API.

These models define the public contract for all API endpoints.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Health ──────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Response for the GET /health endpoint."""

    status: str = "ok"


# ── Chat ────────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    session_id: str = Field(
        default="default",
        description="Unique session identifier for conversation tracking",
        max_length=128,
    )
    message: str = Field(
        ...,
        description="User's natural-language question or message",
        min_length=1,
        max_length=2000,
    )

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message must not be empty or whitespace-only")
        return stripped


class CitationResponse(BaseModel):
    """A single citation attached to an agent answer."""

    listing_id: str = Field(..., description="MongoDB _id of the Airbnb listing")
    listing_name: str = Field(..., description="Human-readable listing name")
    chunk_text: str = Field(..., description="The actual text chunk used for the answer")
    score: float = Field(default=0.0, description="Vector search similarity score")
    location: dict[str, Any] = Field(
        default_factory=dict,
        description="Geographic metadata (city, country, etc.)",
    )


class ChatResponse(BaseModel):
    """Response body for POST /chat."""

    answer: str = Field(..., description="Generated answer from the RAG agent")
    citations: list[CitationResponse] = Field(
        default_factory=list,
        description="Source citations supporting the answer",
    )
    refused: bool = Field(
        default=False,
        description="True when the agent refused to answer due to insufficient context",
    )
    retrieved_docs: int = Field(
        default=0,
        description="Number of documents retrieved from vector search",
    )
    trace_id: str = Field(
        default="",
        description="LangSmith trace ID for observability (empty if tracing is disabled)",
    )


# ── Sessions ────────────────────────────────────────────────────────────────


class SessionMessage(BaseModel):
    """A single message in a session history."""

    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    citations: list[CitationResponse] = Field(default_factory=list)
    refused: bool = False
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class SessionResponse(BaseModel):
    """Response for GET /sessions/{session_id}."""

    session_id: str
    messages: list[SessionMessage] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


# ── Feedback ────────────────────────────────────────────────────────────────


class FeedbackRequest(BaseModel):
    """Request body for POST /feedback."""

    session_id: str = Field(..., max_length=128)
    message_index: int = Field(..., ge=0, description="Index of the assistant message in the session")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 (worst) to 5 (best)")
    comment: str = Field(default="", max_length=1000, description="Optional free-text feedback")


class FeedbackResponse(BaseModel):
    """Response for POST /feedback."""

    status: str = "ok"
    feedback_id: str = ""


# ── Metrics ─────────────────────────────────────────────────────────────────


class MetricsResponse(BaseModel):
    """Response for GET /metrics."""

    total_queries: int = 0
    total_refusals: int = 0
    avg_response_time_ms: float = 0.0
    total_citations_given: int = 0
    uptime_seconds: float = 0.0


# ── Error ────────────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Standard error response body."""

    detail: str = Field(..., description="Human-readable error message")
    status_code: int = 500