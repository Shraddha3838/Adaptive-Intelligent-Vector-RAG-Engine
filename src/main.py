"""
FastAPI application entry point for the Capstone RAG system.

Usage:
    uvicorn src.main:app --reload --port 8000

The application is created in src.api.main; this module re-exports it
for convenience.
"""

from __future__ import annotations

from src.api.main import app

__all__ = ["app"]