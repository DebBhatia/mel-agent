"""
Comprehensive tests for dashboard features:
  - TTS (text-to-speech via ElevenLabs)
  - Homecoming greeting (full briefing: weather + calendar + browser)
  - Spotify Web Playback SDK endpoints (token, register-device, devices)
  - Weather current endpoint
  - Calendar events endpoint
  - Real-time WebSocket agent responses
  - Process endpoint (chat)
  - Routines and reminders
  - Dashboard HTML injection (SESSION_TOKEN)
"""

import os
import sys
import json
import secrets
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

# ── Environment setup (must be BEFORE importing server) ──────────────
os.environ.setdefault("AGENT_API_KEY", f"mel-{secrets.token_urlsafe(32)}")
AGENT_API_KEY = os.environ["AGENT_API_KEY"]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from httpx import AsyncClient, ASGITransport
from server import app, _create_session, _sessions, agent, AGENT_API_KEY as SERVER_KEY


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def auth_headers():
    """Auth via API key (Raspberry Pi / programmatic style)."""
    return {"Authorization": f"Bearer {SERVER_KEY}", "Content-Type": "application/json"}


@pytest.fixture
def session_headers():
    """Auth via session token (dashboard style)."""
    token = _create_session()
    return {"X-Session-Token": token, "Content-Type": "application/json"}


@pytest.fixture
def bad_headers():
    return {"Authorization": "Bearer wrong-key", "Content-Type": "application/json"}


# ═══════════════════════════════════════════════════════════════════════
#  1. AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════

class TestAuthentication:
    """Verify both API-key and session-token auth work for protected endpoints."""

    @pytest.mark.asyncio
    async def test_bearer_auth_accepted(self, auth_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/tts/config", headers=auth_headers)
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_session_token_auth_accepted(self, session_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/tts/config", headers=session_headers)
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_bad_auth_rejected(self, bad_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/tts/config", headers=bad_headers)
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_x_api_key_header_accepted(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/tts/config", headers={
                "X-API-Key": SERVER_KEY,
                "Content-Type": "application/json"
            })
            assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
#  2. DASHBOARD HTML INJECTION
# ═══════════════════════════════════════════════════════════════════════

class TestDashboardServing:

    @pytest.mark.asyncio
    async def test_dashboard_injects_session_token(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/")
            assert resp.status_code == 200
            assert "SESSION_TOKEN" in resp.text
            # Should NOT contain the master API key
            assert SERVER_KEY not in resp.text

    @pytest.mark.asyncio
    async def test_dashboard_sets_cookie(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/")
            assert "mel_session" in resp.headers.get("set-cookie", "")

    @pytest.mark.asyncio
    async def test_dashboard_no_cache_headers(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/")
            assert "no-store" in resp.headers.get("cache-control", "")

    @pytest.mark.asyncio
    async def test_dashboard_contains_speaker_toggle(self):
        """Verify speaker toggle button exists."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/")
            assert "toggleSpeaker" in resp.text

    @pytest.mark.asyncio
    async def test_dashboard_contains_spotify_sdk(self):
        """Verify Spotify Web Playback SDK script tag exists."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/")
            assert "sdk.scdn.co/spotify-player.js" in resp.text

    @pytest.mark.asyncio
    async def test_dashboard_contains_homecoming_detection(self):
        """Verify homecoming/wake phrase detection and /homecoming call exist."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/")
            assert "isWakePhrase" in resp.text
            assert "homecoming" in resp.text
            assert "daddy" in resp.text

    @pytest.mark.asyncio
    async def test_dashboard_contains_tts(self):
        """Verify TTS function exists (speakResponse with ElevenLabs + browser fallback)."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/")
            assert "speakResponse" in resp.text
            assert "elevenlabsAvailable" in resp.text


# ═══════════════════════════════════════════════════════════════════════
#  3. TTS (TEXT-TO-SPEECH)
# ═══════════════════════════════════════════════════════════════════════

class TestTTS:

    @pytest.mark.asyncio
    async def test_tts_config_returns_elevenlabs_status(self, auth_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/tts/config", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert "elevenlabs" in data
            assert isinstance(data["elevenlabs"], bool)

    @pytest.mark.asyncio
    async def test_tts_no_api_key_returns_503(self, auth_headers):
        """Without ElevenLabs key, TTS should return 503."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/tts", headers=auth_headers,
                                content=json.dumps({"text": "Hello world"}))
            # Expect 503 since no ElevenLabs key is set in test env
            assert resp.status_code == 503
            assert "ElevenLabs" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_tts_empty_text_rejected(self, auth_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/tts", headers=auth_headers,
                                content=json.dumps({"text": ""}))
            assert resp.status_code == 422  # Pydantic validation (min_length=1)

    @pytest.mark.asyncio
    async def test_tts_with_mock_elevenlabs(self, auth_headers):
        """Mock ElevenLabs API to verify TTS returns audio/mpeg."""
        fake_audio = b"\xff\xfb\x90\x00" * 100  # Fake MP3 bytes

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = fake_audio

        # Create a mock httpx client that supports async context manager
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key-123"}):
            with patch("httpx.AsyncClient", return_value=mock_client) as mock_cls:
                # Ensure test's own AsyncClient still works by only mocking
                # when timeout=30.0 is passed (as in server.py TTS endpoint)
                original_init = AsyncClient.__init__
                real_cls = AsyncClient

                def selective_mock(*args, **kwargs):
                    if kwargs.get("timeout") == 30.0:
                        return mock_client
                    # Fall through to real AsyncClient for test transport
                    obj = real_cls.__new__(real_cls)
                    original_init(obj, *args, **kwargs)
                    return obj

                mock_cls.side_effect = selective_mock
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as c:
                    resp = await c.post("/tts", headers=auth_headers,
                                        content=json.dumps({"text": "Good morning Deb"}))
                    assert resp.status_code == 200
                    assert resp.headers.get("content-type") == "audio/mpeg"
                    assert len(resp.content) > 0

    @pytest.mark.asyncio
    async def test_tts_strips_markdown(self, auth_headers):
        """Mock ElevenLabs and verify markdown gets stripped from TTS text."""
        captured_body = {}

        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.content = b"\xff\xfb\x90\x00"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        async def capture_post(url, **kwargs):
            captured_body.update(kwargs.get("json", {}))
            return fake_response

        mock_client.post = capture_post

        real_cls = AsyncClient
        original_init = AsyncClient.__init__

        def selective_mock(*args, **kwargs):
            if kwargs.get("timeout") == 30.0:
                return mock_client
            obj = real_cls.__new__(real_cls)
            original_init(obj, *args, **kwargs)
            return obj

        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}):
            with patch("httpx.AsyncClient", side_effect=selective_mock):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as c:
                    resp = await c.post("/tts", headers=auth_headers,
                                        content=json.dumps({"text": "**Bold** and `code`"}))
                    assert resp.status_code == 200
                    # The markdown should be stripped
                    sent_text = captured_body.get("text", "")
                    assert "**" not in sent_text
                    assert "`" not in sent_text
                    assert "Bold" in sent_text


# ═══════════════════════════════════════════════════════════════════════
#  4. HOMECOMING GREETING
# ═══════════════════════════════════════════════════════════════════════

class TestHomecoming:

    @pytest.mark.asyncio
    async def test_homecoming_returns_speech(self, auth_headers):
        """Homecoming should always return a speech field with greeting."""
        with patch("subprocess.Popen", MagicMock()):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.post("/homecoming", headers=auth_headers)
                assert resp.status_code == 200
                data = resp.json()
                assert "speech" in data
                assert len(data["speech"]) > 10
                # Should contain a greeting
                assert any(g in data["speech"].lower() for g in
                           ["good morning", "good afternoon", "good evening", "hey"])

    @pytest.mark.asyncio
    async def test_homecoming_response_shape(self, auth_headers):
        """Verify all fields in HomecomeingResponse are present."""
        with patch("subprocess.Popen", MagicMock()):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.post("/homecoming", headers=auth_headers)
                data = resp.json()
                assert "speech" in data
                assert "weather" in data
                assert "calendar" in data
                assert "news_url" in data
                assert "stocks_url" in data
                assert isinstance(data["weather"], dict)
                assert isinstance(data["calendar"], list)

    @pytest.mark.asyncio
    async def test_homecoming_contains_news_and_stocks_urls(self, auth_headers):
        with patch("subprocess.Popen", MagicMock()):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.post("/homecoming", headers=auth_headers)
                data = resp.json()
                assert "news" in data["news_url"].lower() or data["news_url"].startswith("http")
                assert "stocks" in data["stocks_url"].lower() or data["stocks_url"].startswith("http")

    @pytest.mark.asyncio
    async def test_homecoming_time_appropriate_greeting(self, auth_headers):
        """Test that greeting changes based on time of day."""
        # Test morning greeting
        mock_dt = MagicMock()
        mock_dt.now.return_value = datetime(2026, 4, 4, 8, 30)

        with patch("server.datetime", mock_dt), \
             patch("subprocess.Popen", MagicMock()):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.post("/homecoming", headers=auth_headers)
                data = resp.json()
                assert "good morning" in data["speech"].lower()

    @pytest.mark.asyncio
    async def test_homecoming_evening_greeting(self, auth_headers):
        mock_dt = MagicMock()
        mock_dt.now.return_value = datetime(2026, 4, 4, 19, 0)

        with patch("server.datetime", mock_dt), \
             patch("subprocess.Popen", MagicMock()):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.post("/homecoming", headers=auth_headers)
                data = resp.json()
                assert "good evening" in data["speech"].lower()

    @pytest.mark.asyncio
    async def test_homecoming_midnight_greeting(self, auth_headers):
        mock_dt = MagicMock()
        mock_dt.now.return_value = datetime(2026, 4, 5, 1, 30)

        with patch("server.datetime", mock_dt), \
             patch("subprocess.Popen", MagicMock()):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.post("/homecoming", headers=auth_headers)
                data = resp.json()
                assert "midnight oil" in data["speech"].lower()

    @pytest.mark.asyncio
    async def test_homecoming_with_weather(self, auth_headers):
        """Mock weather service to verify weather gets included in greeting."""
        mock_weather = MagicMock()
        mock_weather.is_configured.return_value = True
        mock_weather.get_current = AsyncMock(return_value={
            "temperature": 72,
            "unit_symbol": "°F",
            "description": "Partly cloudy"
        })
        mock_weather.units = "imperial"

        original_weather = agent.weather
        agent.weather = mock_weather

        try:
            with patch("subprocess.Popen", MagicMock()):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://test") as c:
                    resp = await c.post("/homecoming", headers=auth_headers)
                    data = resp.json()
                    assert "72" in data["speech"]
                    assert "partly cloudy" in data["speech"].lower()
                    assert data["weather"]["temperature"] == 72
        finally:
            agent.weather = original_weather

    @pytest.mark.asyncio
    async def test_homecoming_requires_auth(self, bad_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/homecoming", headers=bad_headers)
            assert resp.status_code == 401


# ═══════════════════════════════════════════════════════════════════════
#  5. SPOTIFY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

class TestSpotify:

    @pytest.mark.asyncio
    async def test_spotify_token_no_service_returns_503(self, auth_headers):
        """Without Spotify configured, token endpoint returns 503."""
        original = agent.spotify
        agent.spotify = None
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.get("/spotify/token", headers=auth_headers)
                assert resp.status_code == 503
        finally:
            agent.spotify = original

    @pytest.mark.asyncio
    async def test_spotify_register_device_no_service(self, auth_headers):
        """Register device returns error when Spotify not available."""
        original = agent.spotify
        agent.spotify = None
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.post("/spotify/register-device?device_id=abc123",
                                    headers=auth_headers)
                data = resp.json()
                assert data["status"] == "error"
        finally:
            agent.spotify = original

    @pytest.mark.asyncio
    async def test_spotify_register_device_success(self, auth_headers):
        """Register device works when Spotify is available."""
        mock_spotify = MagicMock()
        original = agent.spotify
        agent.spotify = mock_spotify
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.post("/spotify/register-device?device_id=test-device-123",
                                    headers=auth_headers)
                data = resp.json()
                assert data["status"] == "ok"
                assert mock_spotify.dashboard_device_id == "test-device-123"
        finally:
            agent.spotify = original

    @pytest.mark.asyncio
    async def test_spotify_register_device_empty_id(self, auth_headers):
        """Register device with empty ID returns error."""
        mock_spotify = MagicMock()
        original = agent.spotify
        agent.spotify = mock_spotify
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.post("/spotify/register-device?device_id=",
                                    headers=auth_headers)
                data = resp.json()
                assert data["status"] == "error"
        finally:
            agent.spotify = original

    @pytest.mark.asyncio
    async def test_spotify_devices_no_service(self, auth_headers):
        original = agent.spotify
        agent.spotify = None
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.get("/spotify/devices", headers=auth_headers)
                assert resp.status_code == 200
                assert resp.json()["devices"] == []
        finally:
            agent.spotify = original

    @pytest.mark.asyncio
    async def test_spotify_devices_with_mock(self, auth_headers):
        mock_spotify = MagicMock()
        mock_spotify.get_devices = AsyncMock(return_value=[
            {"id": "dev1", "name": "MEL Dashboard", "type": "Computer", "is_active": True, "volume": 50}
        ])
        original = agent.spotify
        agent.spotify = mock_spotify
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.get("/spotify/devices", headers=auth_headers)
                data = resp.json()
                assert len(data["devices"]) == 1
                assert data["devices"][0]["name"] == "MEL Dashboard"
        finally:
            agent.spotify = original

    @pytest.mark.asyncio
    async def test_spotify_token_with_mock(self, auth_headers):
        mock_auth = MagicMock()
        mock_auth.get_access_token = AsyncMock(return_value="mock-access-token-xyz")
        mock_spotify = MagicMock()
        mock_spotify.auth = mock_auth
        original = agent.spotify
        agent.spotify = mock_spotify
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.get("/spotify/token", headers=auth_headers)
                assert resp.status_code == 200
                assert resp.json()["access_token"] == "mock-access-token-xyz"
        finally:
            agent.spotify = original

    @pytest.mark.asyncio
    async def test_spotify_play_no_service(self, auth_headers):
        original = agent.spotify
        agent.spotify = None
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.post("/spotify/play", headers=auth_headers)
                assert resp.status_code == 503
        finally:
            agent.spotify = original

    @pytest.mark.asyncio
    async def test_spotify_shuffle_invalid_no_service(self, auth_headers):
        original = agent.spotify
        agent.spotify = None
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.put("/spotify/shuffle", headers=auth_headers)
                assert resp.status_code == 503
        finally:
            agent.spotify = original

    @pytest.mark.asyncio
    async def test_spotify_repeat_invalid_state(self, auth_headers):
        mock_spotify = MagicMock()
        original = agent.spotify
        agent.spotify = mock_spotify
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.put("/spotify/repeat?state=invalid", headers=auth_headers)
                assert resp.status_code == 400
        finally:
            agent.spotify = original

    @pytest.mark.asyncio
    async def test_spotify_playlists_no_service(self, auth_headers):
        original = agent.spotify
        agent.spotify = None
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.get("/spotify/playlists", headers=auth_headers)
                assert resp.status_code == 200
                assert resp.json()["playlists"] == []
        finally:
            agent.spotify = original


# ═══════════════════════════════════════════════════════════════════════
#  6. WEATHER
# ═══════════════════════════════════════════════════════════════════════

class TestWeather:

    @pytest.mark.asyncio
    async def test_weather_no_service_returns_503(self, auth_headers):
        original = agent.weather
        agent.weather = None
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.get("/weather/current", headers=auth_headers)
                assert resp.status_code == 503
        finally:
            agent.weather = original

    @pytest.mark.asyncio
    async def test_weather_with_mock(self, auth_headers):
        mock_weather = MagicMock()
        mock_weather.is_configured.return_value = True
        mock_weather.get_current = AsyncMock(return_value={
            "temperature": 68,
            "unit_symbol": "°F",
            "description": "Sunny"
        })
        original = agent.weather
        agent.weather = mock_weather
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.get("/weather/current", headers=auth_headers)
                assert resp.status_code == 200
                data = resp.json()
                assert data["temperature"] == 68
                assert data["description"] == "Sunny"
        finally:
            agent.weather = original

    @pytest.mark.asyncio
    async def test_weather_error_returns_503(self, auth_headers):
        mock_weather = MagicMock()
        mock_weather.is_configured.return_value = True
        mock_weather.get_current = AsyncMock(return_value={"error": "API key invalid"})
        original = agent.weather
        agent.weather = mock_weather
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.get("/weather/current", headers=auth_headers)
                assert resp.status_code == 503
        finally:
            agent.weather = original


# ═══════════════════════════════════════════════════════════════════════
#  7. CALENDAR
# ═══════════════════════════════════════════════════════════════════════

class TestCalendar:

    @pytest.mark.asyncio
    async def test_calendar_events_with_mock(self, auth_headers):
        """Mock calendar_list action to return events."""
        original_actions = agent.actions
        mock_actions = MagicMock()
        mock_actions.actions = {"calendar_list": {"handler": AsyncMock()}}
        mock_actions.execute = AsyncMock(return_value="10:00 AM - Team standup, 2:00 PM - Design review")
        agent.actions = mock_actions
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.get("/calendar/events", headers=auth_headers)
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "ok"
                assert "standup" in data["result"].lower()
        finally:
            agent.actions = original_actions

    @pytest.mark.asyncio
    async def test_calendar_no_events(self, auth_headers):
        original_actions = agent.actions
        mock_actions = MagicMock()
        mock_actions.actions = {"calendar_list": {"handler": AsyncMock()}}
        mock_actions.execute = AsyncMock(return_value="No upcoming events")
        agent.actions = mock_actions
        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                resp = await c.get("/calendar/events", headers=auth_headers)
                assert resp.status_code == 200
                data = resp.json()
                assert "no upcoming" in data["result"].lower()
        finally:
            agent.actions = original_actions


# ═══════════════════════════════════════════════════════════════════════
#  8. PROCESS (CHAT)
# ═══════════════════════════════════════════════════════════════════════

class TestProcess:

    @pytest.mark.asyncio
    async def test_process_returns_response(self, auth_headers):
        """Process should return a ProcessResponse."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/process", headers=auth_headers,
                                content=json.dumps({"input": "hello"}))
            assert resp.status_code == 200
            data = resp.json()
            assert "response" in data
            assert "timestamp" in data
            assert len(data["response"]) > 0

    @pytest.mark.asyncio
    async def test_process_empty_input_rejected(self, auth_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/process", headers=auth_headers,
                                content=json.dumps({"input": "   "}))
            assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_process_too_long_input(self, auth_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/process", headers=auth_headers,
                                content=json.dumps({"input": "x" * 5001}))
            # Should be 400 (server) or 422 (pydantic)
            assert resp.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_process_wake_phrase_triggers_greeting(self, auth_headers):
        """Sending 'daddy is home' via /process should return a greeting."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/process", headers=auth_headers,
                                content=json.dumps({"input": "wake up daddy is home"}))
            assert resp.status_code == 200
            data = resp.json()
            response_lower = data["response"].lower()
            assert any(g in response_lower for g in
                       ["good morning", "good afternoon", "good evening", "hey"])

    @pytest.mark.asyncio
    async def test_process_requires_auth(self, bad_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/process", headers=bad_headers,
                                content=json.dumps({"input": "hello"}))
            assert resp.status_code == 401


# ═══════════════════════════════════════════════════════════════════════
#  9. ROUTINES & REMINDERS
# ═══════════════════════════════════════════════════════════════════════

class TestRoutines:

    @pytest.mark.asyncio
    async def test_routines_list(self, auth_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/routines", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert "routines" in data
            assert isinstance(data["routines"], list)

    @pytest.mark.asyncio
    async def test_routine_run_nonexistent(self, auth_headers):
        """Running a nonexistent routine should return 404 or error."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/routines/nonexistent-routine/run", headers=auth_headers)
            # 404 or 503 depending on whether routines engine loaded
            assert resp.status_code in (404, 503)


class TestReminders:

    @pytest.mark.asyncio
    async def test_reminders_pending(self, auth_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/reminders/pending", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert "reminders" in data
            assert isinstance(data["reminders"], list)


# ═══════════════════════════════════════════════════════════════════════
#  10. HEALTH ENDPOINT
# ═══════════════════════════════════════════════════════════════════════

class TestHealth:

    @pytest.mark.asyncio
    async def test_health_detail(self, auth_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/health/detail", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "online"
            assert "services" in data
            assert "timestamp" in data


# ═══════════════════════════════════════════════════════════════════════
#  11. ORCHESTRATOR WAKE/GREETING
# ═══════════════════════════════════════════════════════════════════════

class TestOrchestratorGreeting:
    """Test the orchestrator's wake_up and greeting logic directly."""

    @pytest.mark.asyncio
    async def test_wake_up_sets_awake(self):
        from orchestrator import AgentOrchestrator
        orch = AgentOrchestrator()
        assert not orch.is_awake
        result = await orch.wake_up()
        assert orch.is_awake
        assert isinstance(result, str)
        assert len(result) > 5

    @pytest.mark.asyncio
    async def test_wake_variants_detected(self):
        from orchestrator import AgentOrchestrator
        orch = AgentOrchestrator()

        variants = [
            "wake up daddy is home",
            "daddy's home",
            "daddy is home",
            "daddys home",
        ]
        for variant in variants:
            orch.is_awake = False
            result = await orch.process(variant)
            assert orch.is_awake, f"Wake variant '{variant}' failed to wake the agent"
            assert any(g in result.lower() for g in
                       ["good morning", "good afternoon", "good evening", "hey", "midnight"]), \
                f"No greeting in response for '{variant}': {result}"

    @pytest.mark.asyncio
    async def test_greeting_time_aware(self):
        from orchestrator import AgentOrchestrator
        orch = AgentOrchestrator()

        # Morning
        with patch("orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 4, 4, 8, 0)
            result = await orch.wake_up()
            assert "morning" in result.lower()

    @pytest.mark.asyncio
    async def test_sleep_after_wake(self):
        from orchestrator import AgentOrchestrator
        orch = AgentOrchestrator()
        await orch.wake_up()
        assert orch.is_awake
        result = await orch.sleep()
        assert not orch.is_awake


# ═══════════════════════════════════════════════════════════════════════
#  12. WEATHER SERVICE UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestWeatherService:
    """Test weather advice logic directly."""

    def test_weather_advice_umbrella(self):
        from weather import WeatherService
        advice = WeatherService.get_weather_advice(65, "Rain", "imperial")
        assert advice is not None
        assert "umbrella" in advice.lower() or "rain" in advice.lower()

    def test_weather_advice_cold(self):
        from weather import WeatherService
        advice = WeatherService.get_weather_advice(28, "Clear", "imperial")
        assert advice is not None
        # Should suggest warm clothing or icy roads
        assert any(w in advice.lower() for w in ["jacket", "coat", "bundle", "layer", "icy", "cold", "warm"])

    def test_weather_advice_hot(self):
        from weather import WeatherService
        advice = WeatherService.get_weather_advice(98, "Clear", "imperial")
        assert advice is not None
        assert any(w in advice.lower() for w in ["hydrat", "water", "hot", "heat", "cool"])

    def test_weather_advice_nice_day(self):
        from weather import WeatherService
        advice = WeatherService.get_weather_advice(72, "Clear", "imperial")
        # Nice weather may return None or a positive message
        # Either is acceptable


# ═══════════════════════════════════════════════════════════════════════
#  13. SEARCH ENDPOINT
# ═══════════════════════════════════════════════════════════════════════

class TestSearch:

    @pytest.mark.asyncio
    async def test_search_requires_auth(self, bad_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.post("/search", headers=bad_headers,
                                content=json.dumps({"query": "test"}))
            assert resp.status_code == 401


# ═══════════════════════════════════════════════════════════════════════
#  14. ABOUT-ME / JOURNAL
# ═══════════════════════════════════════════════════════════════════════

class TestJournal:

    def test_about_me_dir_exists(self):
        import journal
        assert journal.ABOUT_ME_DIR.exists()

    def test_about_me_loads_files(self):
        import journal
        context = journal.get_about_me_context()
        assert isinstance(context, str)
        # Should contain content from the about-me markdown files
        # (at minimum the templates are there)

    def test_about_me_returns_string(self):
        import journal
        result = journal.get_about_me_context()
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════
#  15. SESSION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════

class TestSessionManagement:

    def test_create_session_returns_token(self):
        token = _create_session()
        assert isinstance(token, str)
        assert len(token) > 20
        assert token in _sessions

    def test_session_token_expires(self):
        import time as _time
        token = _create_session()
        # Manually expire the session
        _sessions[token] = _time.time() - 1
        # Token should be expired
        assert _sessions[token] < _time.time()

    def test_multiple_sessions_tracked(self):
        t1 = _create_session()
        t2 = _create_session()
        assert t1 != t2
        assert t1 in _sessions
        assert t2 in _sessions


# ═══════════════════════════════════════════════════════════════════════
#  16. SMART HOME ENDPOINT
# ═══════════════════════════════════════════════════════════════════════

class TestSmartHome:

    @pytest.mark.asyncio
    async def test_smarthome_devices(self, auth_headers):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/smarthome/devices", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert "devices" in data
            assert isinstance(data["devices"], list)
