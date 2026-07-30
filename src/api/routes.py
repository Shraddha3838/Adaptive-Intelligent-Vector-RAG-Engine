"""
FastAPI routes for the Capstone RAG API.

Provides endpoints for health checks, chat, session management, feedback,
and metrics — all backed by the real LangGraph RAG agent.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from langsmith import traceable

from src.api.dependencies import (
    add_message_to_session,
    get_agent,
    get_metrics,
    get_or_create_session,
    record_query,
    store_feedback,
)
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    CitationResponse,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MetricsResponse,
    SessionResponse,
)
from src.utils import get_logger

logger = get_logger("api.routes")

router = APIRouter(tags=["capstone-rag"])


# ── Health ───────────────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns a simple status to confirm the API is running.",
)
async def health() -> HealthResponse:
    """Health check endpoint. Returns {'status': 'ok'}."""
    return HealthResponse(status="ok")


# ── Chat ─────────────────────────────────────────────────────────────────────


def _build_citations_from_agent_response(
    agent_response: Any,
) -> list[CitationResponse]:
    """Convert agent response citations to API CitationResponse objects."""
    citations: list[CitationResponse] = []

    # Try to extract from the agent's citations first
    for c in getattr(agent_response, "citations", []) or []:
        location_dict = {}
        if hasattr(c, "location") and c.location is not None:
            loc = c.location
            if hasattr(loc, "model_dump"):
                location_dict = loc.model_dump()
            elif isinstance(loc, dict):
                location_dict = loc

        citations.append(
            CitationResponse(
                listing_id=c.listing_id,
                listing_name=c.listing_name,
                chunk_text="",
                score=c.score,
                location=location_dict,
            )
        )

    # If no citations from the agent, try to build from documents
    if not citations:
        for doc in getattr(agent_response, "documents", []) or []:
            location_dict = {}
            if hasattr(doc, "location") and doc.location is not None:
                loc = doc.location
                if hasattr(loc, "model_dump"):
                    location_dict = loc.model_dump()
                elif isinstance(loc, dict):
                    location_dict = loc

            citations.append(
                CitationResponse(
                    listing_id=doc.listing_id,
                    listing_name=doc.listing_name,
                    chunk_text=doc.text,
                    score=doc.score,
                    location=location_dict,
                )
            )

    # Deduplicate by listing_id, keep highest score
    best_by_id: dict[str, CitationResponse] = {}
    for c in citations:
        existing = best_by_id.get(c.listing_id)
        if existing is None or c.score > existing.score:
            best_by_id[c.listing_id] = c

    return sorted(best_by_id.values(), key=lambda c: c.score, reverse=True)


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question to the RAG agent",
    description=(
        "Sends a user message to the self-correcting LangGraph RAG agent. "
        "The agent retrieves relevant chunks via MongoDB Atlas Vector Search, "
        "grades relevance, rewrites queries if needed, generates a grounded "
        "answer, and attaches citations. Returns the answer, citations, "
        "refusal status, retrieved document count, and a trace ID."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Bad request (e.g., empty message)"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    """Process a chat message through the real LangGraph RAG agent."""
    # Record user message in session
    add_message_to_session(
        session_id=request.session_id,
        role="user",
        content=request.message,
    )

    start_time = time.perf_counter()

    try:
        agent = get_agent()
        # The agent.run() method is synchronous; LangGraph's invoke is sync
        agent_response = agent.run(request.message)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Build citations
        citations = _build_citations_from_agent_response(agent_response)

        # Count retrieved docs
        retrieved_docs = len(getattr(agent_response, "documents", []) or [])

        # Generate a trace_id (LangSmith trace ID if available)
        trace_id = _get_trace_id()

        # Record metrics
        record_query(
            response_time_ms=elapsed_ms,
            refused=agent_response.refused,
            citations_count=len(citations),
        )

        # Record assistant message in session
        add_message_to_session(
            session_id=request.session_id,
            role="assistant",
            content=agent_response.answer,
            citations=citations,
            refused=agent_response.refused,
        )

        logger.info(
            "Chat: session=%s refused=%s docs=%d citations=%d time=%.0fms",
            request.session_id,
            agent_response.refused,
            retrieved_docs,
            len(citations),
            elapsed_ms,
        )

        return ChatResponse(
            answer=agent_response.answer or "",
            citations=citations,
            refused=agent_response.refused,
            retrieved_docs=retrieved_docs,
            trace_id=trace_id,
        )

    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.exception(
            "Chat failed: session=%s message=%.50s time=%.0fms error=%s",
            request.session_id,
            request.message,
            elapsed_ms,
            exc,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Agent processing failed: {exc}",
        )


def _get_trace_id() -> str:
    """Attempt to get the current LangSmith trace ID."""
    try:
        from langsmith.run_helpers import get_current_run_tree

        run_tree = get_current_run_tree()
        if run_tree is not None:
            return str(run_tree.trace_id)
    except Exception:
        pass
    return ""


# ── Sessions ─────────────────────────────────────────────────────────────────


@router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    summary="Get session history",
    description="Retrieve the full message history for a given session.",
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
    },
)
async def get_session(session_id: str) -> SessionResponse:
    """Return the message history for a session."""
    session = get_or_create_session(session_id)
    if not session.messages and session.created_at == session.updated_at:
        # Session was just created, no messages yet
        pass
    return session


# ── Feedback ─────────────────────────────────────────────────────────────────


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit feedback for a chat response",
    description="Submit a rating and optional comment for a specific assistant message in a session.",
)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Store user feedback for a chat interaction."""
    feedback_id = store_feedback(request)
    logger.info(
        "Feedback: session=%s message_index=%d rating=%d id=%s",
        request.session_id,
        request.message_index,
        request.rating,
        feedback_id,
    )
    return FeedbackResponse(status="ok", feedback_id=feedback_id)


# ── Metrics ──────────────────────────────────────────────────────────────────


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get API metrics",
    description="Return aggregate metrics: total queries, refusals, average response time, and uptime.",
)
async def metrics() -> MetricsResponse:
    """Return current API metrics."""
    return get_metrics()