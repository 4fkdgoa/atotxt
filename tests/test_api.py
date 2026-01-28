"""Tests for the FastAPI endpoints."""

import io
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_system_info():
    response = client.get("/system")
    assert response.status_code == 200
    data = response.json()
    assert "device" in data
    assert "whisper_model" in data
    assert "compute_type" in data
    assert "ollama_model" in data


def test_upload_invalid_extension():
    file = io.BytesIO(b"not an audio file")
    response = client.post(
        "/upload",
        files={"file": ("test.txt", file, "text/plain")},
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_upload_valid_extension():
    """Test that upload accepts valid audio extensions and returns task ID."""
    file = io.BytesIO(b"fake audio data")

    with patch("app.api.routes.process_audio_task", new_callable=AsyncMock):
        response = client.post(
            "/upload",
            files={"file": ("test.wav", file, "audio/wav")},
        )

    assert response.status_code == 200
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "pending"


def test_get_task_not_found():
    response = client.get("/task/nonexistent")
    assert response.status_code == 404
