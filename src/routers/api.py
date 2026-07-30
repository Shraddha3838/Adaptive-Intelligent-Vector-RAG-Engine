"""
FastAPI router for the RAG agent API.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.config import get_settings
from src.graph import RAGAgent
from src.ingest import IngestionPipeline
from src.utils import get_logger

logger = get_logger("api")
router = APIRouter(prefix="/api", tags=["rag"])

# Global agent instance (lazy initialized)
_agent: RAGAgent | None = None


def get_agent() -> RAGAgent:
    """Get or create the RAG agent singleton."""
    global _agent
    if _agent is None:
        settings = get_settings()
        _agent = RAGAgent(settings)
    return _agent


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Natural language question")


class CitationResponse(BaseModel):
    listing_id: str
    listing_name: str
    chunk_id: str
    score: float
    location: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationResponse] = Field(default_factory=list)
    cited_listing_ids: list[str] = Field(default_factory=list)
    retrieval_attempts: int = 1
    rewritten_queries: list[str] = Field(default_factory=list)
    refused: bool = False
    duration_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    timestamp: float = 0.0


class StatsResponse(BaseModel):
    source_count: int = 0
    vector_count: int = 0
    source_collection: str = ""
    vector_collection: str = ""


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        timestamp=time.time(),
    )


@router.get("/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    """Get collection statistics."""
    try:
        settings = get_settings()
        pipeline = IngestionPipeline(settings)
        try:
            coll_stats = pipeline.get_collection_stats()
            return StatsResponse(
                source_count=coll_stats["source_count"],
                vector_count=coll_stats["vector_count"],
                source_collection=coll_stats["source_collection"],
                vector_collection=coll_stats["vector_collection"],
            )
        finally:
            pipeline.close()
    except Exception as exc:
        logger.error("Failed to get stats: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Ask a question to the RAG agent."""
    start = time.perf_counter()
    try:
        agent = get_agent()
        response = agent.run(request.question)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        return QueryResponse(
            question=response.question,
            answer=response.answer,
            citations=[
                CitationResponse(
                    listing_id=c.listing_id,
                    listing_name=c.listing_name,
                    chunk_id=c.chunk_id,
                    score=c.score,
                    location=c.location.model_dump() if c.location else {},
                )
                for c in response.citations
            ],
            cited_listing_ids=response.cited_listing_ids,
            retrieval_attempts=response.retrieval_attempts,
            rewritten_queries=response.rewritten_queries,
            refused=response.refused,
            duration_ms=duration_ms,
        )
    except Exception as exc:
        logger.exception("Query failed: %s", request.question)
        raise HTTPException(status_code=500, detail=str(exc))