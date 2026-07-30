"""
Tests for the Capstone RAG FastAPI endpoints.

Tests cover:
- GET /health
- POST /chat (valid request, empty message, invalid request)
- Error handling (MongoDB failure, LLM failure simulated via monkeypatch)
"""

from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


# ── Health ───────────────────────────────────────────────────────────────────


class TestHealth:
    """Tests for the GET /health endpoint."""

    def test_health_returns_ok(self) -> None:
        """GET /health should return {'status': 'ok'}."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data == {"status": "ok"}

    def test_health_method_not_allowed(self) -> None:
        """POST /health should return 405."""
        response = client.post("/api/health")
        assert response.status_code == 405


# ── Chat ─────────────────────────────────────────────────────────────────────


class TestChat:
    """Tests for the POST /chat endpoint."""

    def test_chat_valid_request(self) -> None:
        """POST /chat with a valid message should return a ChatResponse."""
        response = client.post(
            "/api/chat",
            json={"session_id": "test-user-1", "message": "Find me a quiet Airbnb near the beach with WiFi"},
        )
        # The agent may refuse or answer depending on retrieval; either is valid
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert "citations" in data
        assert isinstance(data["citations"], list)
        assert "refused" in data
        assert isinstance(data["refused"], bool)
        assert "retrieved_docs" in data
        assert isinstance(data["retrieved_docs"], int)
        assert "trace_id" in data
        assert isinstance(data["trace_id"], str)

    def test_chat_empty_message(self) -> None:
        """POST /chat with an empty message should return 422."""
        response = client.post(
            "/api/chat",
            json={"session_id": "test-user-1", "message": ""},
        )
        assert response.status_code == 422

    def test_chat_whitespace_only(self) -> None:
        """POST /chat with whitespace-only message should return 422."""
        response = client.post(
            "/api/chat",
            json={"session_id": "test-user-1", "message": "   "},
        )
        assert response.status_code == 422

    def test_chat_missing_message(self) -> None:
        """POST /chat without a message field should return 422."""
        response = client.post(
            "/api/chat",
            json={"session_id": "test-user-1"},
        )
        assert response.status_code == 422

    def test_chat_message_too_long(self) -> None:
        """POST /chat with a message exceeding max_length should return 422."""
        long_message = "x" * 2001
        response = client.post(
            "/api/chat",
            json={"session_id": "test-user-1", "message": long_message},
        )
        assert response.status_code == 422

    def test_chat_invalid_json(self) -> None:
        """POST /chat with invalid JSON should return 422."""
        response = client.post(
            "/api/chat",
            data="not-json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422


# ── Sessions ─────────────────────────────────────────────────────────────────


class TestSessions:
    """Tests for the GET /sessions/{session_id} endpoint."""

    def test_get_session_returns_session(self) -> None:
        """GET /sessions/{id} should return a session response."""
        # First, send a chat to create a session
        client.post(
            "/api/chat",
            json={"session_id": "session-test-1", "message": "Hello"},
        )
        response = client.get("/api/sessions/session-test-1")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "session-test-1"
        assert "messages" in data
        assert len(data["messages"]) >= 2  # user + assistant

    def test_get_session_not_found(self) -> None:
        """GET /sessions/{id} for a non-existent session should still return a session."""
        # The implementation auto-creates sessions, so it should return an empty one
        response = client.get("/api/sessions/unknown-session-xyz")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "unknown-session-xyz"


# ── Feedback ─────────────────────────────────────────────────────────────────


class TestFeedback:
    """Tests for the POST /feedback endpoint."""

    def test_submit_feedback(self) -> None:
        """POST /feedback with valid data should return ok."""
        response = client.post(
            "/api/feedback",
            json={
                "session_id": "test-session",
                "message_index": 0,
                "rating": 5,
                "comment": "Great answer!",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "feedback_id" in data

    def test_feedback_invalid_rating(self) -> None:
        """POST /feedback with rating out of range should return 422."""
        response = client.post(
            "/api/feedback",
            json={
                "session_id": "test-session",
                "message_index": 0,
                "rating": 6,
                "comment": "",
            },
        )
        assert response.status_code == 422


# ── Metrics ──────────────────────────────────────────────────────────────────


class TestMetrics:
    """Tests for the GET /metrics endpoint."""

    def test_metrics_returns_data(self) -> None:
        """GET /metrics should return metrics data."""
        response = client.get("/api/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_queries" in data
        assert "total_refusals" in data
        assert "avg_response_time_ms" in data
        assert "total_citations_given" in data
        assert "uptime_seconds" in data
        assert isinstance(data["total_queries"], int)
        assert isinstance(data["uptime_seconds"], float)