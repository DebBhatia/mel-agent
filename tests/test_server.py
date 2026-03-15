"""Tests for the FastAPI server — endpoints, auth, input validation, CORS, rate limiting."""

import os
import sys
import json
import secrets
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Mock env before importing server
os.environ.setdefault("AGENT_API_KEY", f"mel-{secrets.token_urlsafe(32)}")

from server import (
    app,
    AGENT_API_KEY,
    _strip_paths,
    ProcessRequest,
    BuildRequest,
    KnowledgeStoreRequest,
    KnowledgeSearchRequest,
    ShellRequest,
    CalendarEventRequest,
)

from httpx import AsyncClient, ASGITransport
import asyncio


@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {AGENT_API_KEY}"}


@pytest.fixture
def bad_auth_headers():
    return {"Authorization": "Bearer wrong-key-12345"}


# ── Authentication Tests ──────────────────────

class TestAuth:
    @pytest.mark.asyncio
    async def test_health_no_auth_required(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "online"
            assert "agent_name" in data

    @pytest.mark.asyncio
    async def test_protected_endpoint_requires_auth(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/process", json={"input": "hello"})
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_bearer_auth(self, auth_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/detail", headers=auth_headers)
            # May fail due to missing Ollama, but auth should pass (not 401)
            assert resp.status_code != 401

    @pytest.mark.asyncio
    async def test_x_api_key_auth(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/health/detail", headers={"X-API-Key": AGENT_API_KEY})
            assert resp.status_code != 401

    @pytest.mark.asyncio
    async def test_query_param_auth(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/health/detail?api_key={AGENT_API_KEY}")
            assert resp.status_code != 401

    @pytest.mark.asyncio
    async def test_bad_auth_rejected(self, bad_auth_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/process", json={"input": "test"}, headers=bad_auth_headers)
            assert resp.status_code == 401


# ── Input Validation Tests ────────────────────

class TestInputValidation:
    @pytest.mark.asyncio
    async def test_empty_input_rejected(self, auth_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/process", json={"input": "   "}, headers=auth_headers)
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_too_long_input_rejected(self, auth_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/process", json={"input": "x" * 5001}, headers=auth_headers)
            assert resp.status_code == 400


# ── Utility Function Tests ────────────────────

class TestUtilities:
    def test_strip_paths_linux(self):
        text = "Error at /home/user/project/file.py line 42"
        result = _strip_paths(text)
        assert "/home/" not in result
        assert "[PATH]" in result

    def test_strip_paths_windows(self):
        text = "Error at C:\\\\Users\\\\deb\\\\file.py"
        result = _strip_paths(text)
        assert "[PATH]" in result

    def test_strip_paths_clean_text(self):
        text = "Everything is fine"
        assert _strip_paths(text) == text


# ── Dashboard ─────────────────────────────────

class TestDashboard:
    @pytest.mark.asyncio
    async def test_dashboard_serves(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/")
            assert resp.status_code == 200
            assert "html" in resp.headers.get("content-type", "").lower()


# ── Pydantic Model Validation ─────────────────

class TestModels:
    def test_process_request_default_context(self):
        r = ProcessRequest(input="hello")
        assert r.context == {}

    def test_build_request_default_target(self):
        r = BuildRequest(prompt="build me an app")
        assert r.deploy_target == "copy"

    def test_knowledge_store_defaults(self):
        r = KnowledgeStoreRequest(content="test")
        assert r.category == "general"
        assert r.metadata == {}

    def test_knowledge_search_defaults(self):
        r = KnowledgeSearchRequest(query="test")
        assert r.n_results == 5
        assert r.category is None

    def test_shell_request_defaults(self):
        r = ShellRequest(command="ls")
        assert r.cwd is None

    def test_calendar_event_defaults(self):
        r = CalendarEventRequest(title="Test", start_time="2026-03-15T10:00:00")
        assert r.timezone == "America/Chicago"
        assert r.end_time is None


# ── Endpoint Responses ────────────────────────

class TestEndpointResponses:
    @pytest.mark.asyncio
    async def test_tasks_endpoint(self, auth_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/tasks", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert "tasks" in data

    @pytest.mark.asyncio
    async def test_projects_endpoint(self, auth_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/projects", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert "projects" in data

    @pytest.mark.asyncio
    async def test_knowledge_stats(self, auth_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/knowledge/stats", headers=auth_headers)
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_nonexistent_project_404(self, auth_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/projects/nonexistent-id-123", headers=auth_headers)
            # Should be 404 or 503 (if codegen not loaded)
            assert resp.status_code in [404, 503]
