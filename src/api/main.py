"""
FastAPI application factory for the Capstone RAG API.

Creates and configures the FastAPI application with CORS middleware,
exception handlers, structured logging, and route inclusion.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import close_agent
from src.api.routes import router as api_router
from src.config import get_settings
from src.utils import get_logger, setup_logging

logger = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: configure logging on startup, clean up on shutdown."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info(
        "Capstone RAG API starting — env=%s log_level=%s",
        settings.app_env,
        settings.log_level,
    )
    yield
    close_agent()
    logger.info("Capstone RAG API shut down")


def create_app() -> FastAPI:
    """Create and return a fully configured FastAPI application."""
    app = FastAPI(
        title="Capstone RAG API",
        description=(
            "Self-correcting RAG agent over MongoDB Atlas Vector Search. "
            "Built with LangGraph, Groq (free chat), and HuggingFace (free embeddings)."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS — allow Streamlit frontend and any origin during development ──
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Include API routes ──
    app.include_router(api_router, prefix="/api")

    # ── Exception handlers ──
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch unhandled exceptions and return a structured error response."""
        logger.exception("Unhandled exception: %s on %s %s", exc, request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {exc}", "status_code": 500},
        )

    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
        """Return a safe error message without exposing internals."""
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "status_code": 500},
        )

    return app


app = create_app()