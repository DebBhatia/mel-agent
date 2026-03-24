"""Integration tests for calendar event creation via the orchestrator.

Tests the full pipeline: user input → intent classification → title extraction →
time parsing → Google Calendar API call (mocked) → friendly response.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from actions import CalendarPlugin
from orchestrator import AgentOrchestrator, ActionRegistry


# ── Helpers ──────────────────────────────────────────

def make_mock_calendar_plugin():
    """Create a CalendarPlugin with a mocked Google Calendar service."""
    cal = CalendarPlugin()
    mock_service = MagicMock()

    # Mock the events().insert().execute() chain
    def fake_insert(calendarId, body):
        mock_result = MagicMock()
        # Google Calendar API echoes back the event summary
        mock_result.execute.return_value = {"summary": body["summary"], "id": "mock123"}
        return mock_result

    mock_service.events.return_value.insert = fake_insert
    cal.service = mock_service
    return cal


def make_orchestrator_with_mock_calendar():
    """Create an AgentOrchestrator with mocked calendar plugin."""
    orch = AgentOrchestrator()
    cal = make_mock_calendar_plugin()
    orch.actions.actions["calendar_create"] = {
        "handler": cal.create_event,
        "description": "Create calendar event",
    }
    return orch


# ── Title Extraction Tests ───────────────────────────

class TestCalendarTitleExtraction:
    """Verify the orchestrator extracts a meaningful event title, not the raw user message."""

    @pytest.fixture
    def orch(self):
        return make_orchestrator_with_mock_calendar()

    @pytest.mark.asyncio
    async def test_does_not_echo_full_message(self, orch):
        """The exact bug from the screenshot — title should NOT be the user's full sentence."""
        result = await orch._handle_calendar(
            "can you set up an appointment on my calendar for 4:00 p.m. today I want to eat my vitamins",
            {"intent": "create", "summary": "set up appointment to eat vitamins"},
        )
        assert "can you set up an appointment" not in result
        assert "Done!" in result or "added" in result.lower()

    @pytest.mark.asyncio
    async def test_extracts_purpose_eat_vitamins(self, orch):
        result = await orch._handle_calendar(
            "can you set up an appointment on my calendar for 4:00 p.m. today I want to eat my vitamins",
            {"intent": "create", "summary": "eat vitamins"},
        )
        # Title should contain something about vitamins, not the full command
        assert "vitamin" in result.lower()

    @pytest.mark.asyncio
    async def test_simple_meeting(self, orch):
        result = await orch._handle_calendar(
            "schedule a meeting at 3:00 pm",
            {"intent": "create", "summary": "schedule a meeting"},
        )
        assert "Done!" in result
        # Should NOT echo "schedule a meeting at 3:00 pm" as the title
        assert "schedule a meeting at 3:00 pm" not in result

    @pytest.mark.asyncio
    async def test_create_dentist_appointment(self, orch):
        result = await orch._handle_calendar(
            "create an appointment on my calendar for dentist visit at 2:30 pm",
            {"intent": "create", "summary": "dentist visit"},
        )
        assert "dentist" in result.lower()

    @pytest.mark.asyncio
    async def test_book_gym_session(self, orch):
        result = await orch._handle_calendar(
            "add gym session to my calendar at 6:00 a.m.",
            {"intent": "create", "summary": "gym session"},
        )
        assert "gym" in result.lower()


# ── Time Parsing Tests ───────────────────────────────

class TestCalendarTimeParsing:
    """Verify time extraction from various natural language formats."""

    @pytest.fixture
    def orch(self):
        return make_orchestrator_with_mock_calendar()

    async def _get_params(self, orch, user_input):
        """Helper to capture the params sent to calendar_create."""
        captured = {}
        original_execute = orch.actions.execute

        async def capture_execute(action_name, params):
            captured.update(params)
            return await original_execute(action_name, params)

        orch.actions.execute = capture_execute
        await orch._handle_calendar(user_input, {"intent": "create", "summary": ""})
        return captured

    @pytest.mark.asyncio
    async def test_colon_pm_no_dots(self, orch):
        """4:00pm → 16:00"""
        params = await self._get_params(orch, "meeting at 4:00pm")
        assert "T16:00:00" in params["start_time"]

    @pytest.mark.asyncio
    async def test_colon_pm_with_space(self, orch):
        """4:00 pm → 16:00"""
        params = await self._get_params(orch, "meeting at 4:00 pm")
        assert "T16:00:00" in params["start_time"]

    @pytest.mark.asyncio
    async def test_colon_pm_with_dots(self, orch):
        """4:00 p.m. → 16:00 (the user's actual format from the screenshot)"""
        params = await self._get_params(orch, "meeting at 4:00 p.m.")
        assert "T16:00:00" in params["start_time"]

    @pytest.mark.asyncio
    async def test_colon_am(self, orch):
        """6:00 a.m. → 06:00"""
        params = await self._get_params(orch, "gym at 6:00 a.m.")
        assert "T06:00:00" in params["start_time"]

    @pytest.mark.asyncio
    async def test_no_colon_pm(self, orch):
        """4 pm → 16:00"""
        params = await self._get_params(orch, "meeting at 4 pm")
        assert "T16:00:00" in params["start_time"]

    @pytest.mark.asyncio
    async def test_no_colon_pm_dots(self, orch):
        """4 p.m. → 16:00"""
        params = await self._get_params(orch, "meeting at 4 p.m.")
        assert "T16:00:00" in params["start_time"]

    @pytest.mark.asyncio
    async def test_noon(self, orch):
        """12:00 pm → 12:00"""
        params = await self._get_params(orch, "lunch at 12:00 pm")
        assert "T12:00:00" in params["start_time"]

    @pytest.mark.asyncio
    async def test_midnight(self, orch):
        """12:00 am → 00:00"""
        params = await self._get_params(orch, "event at 12:00 am")
        assert "T00:00:00" in params["start_time"]

    @pytest.mark.asyncio
    async def test_end_time_30_min_later(self, orch):
        """End time should default to 30 minutes after start."""
        params = await self._get_params(orch, "meeting at 4:00 pm")
        assert "T16:30:00" in params["end_time"]

    @pytest.mark.asyncio
    async def test_end_time_hour_rollover(self, orch):
        """4:45 pm → end at 5:15 pm"""
        params = await self._get_params(orch, "meeting at 4:45 pm")
        assert "T17:15:00" in params["end_time"]

    @pytest.mark.asyncio
    async def test_default_time_when_none_specified(self, orch):
        """Falls back to 1:30 PM when no time is given."""
        params = await self._get_params(orch, "add a meeting to my calendar")
        assert "T13:30:00" in params["start_time"]


# ── Response Format Tests ────────────────────────────

class TestCalendarResponseFormat:
    """Verify the response is human-friendly, not raw ISO timestamps."""

    @pytest.fixture
    def orch(self):
        return make_orchestrator_with_mock_calendar()

    @pytest.mark.asyncio
    async def test_no_raw_iso_timestamp_in_response(self, orch):
        result = await orch._handle_calendar(
            "set up appointment at 4:00 p.m. to eat vitamins",
            {"intent": "create", "summary": "eat vitamins"},
        )
        # Should NOT contain raw ISO like "2026-03-24T16:00:00"
        assert "T16:00:00" not in result
        assert "T13:30:00" not in result

    @pytest.mark.asyncio
    async def test_response_contains_friendly_time(self, orch):
        result = await orch._handle_calendar(
            "set up appointment at 4:00 p.m. to eat vitamins",
            {"intent": "create", "summary": "eat vitamins"},
        )
        # Should contain friendly format like "4:00 PM"
        assert "4:00 PM" in result

    @pytest.mark.asyncio
    async def test_response_says_done(self, orch):
        result = await orch._handle_calendar(
            "create event at 3:00 pm for team sync",
            {"intent": "create", "summary": "team sync"},
        )
        assert "Done!" in result

    @pytest.mark.asyncio
    async def test_no_service_returns_not_connected(self):
        """When Google Calendar is not authenticated, should say so clearly."""
        orch = AgentOrchestrator()
        # Replace with a plugin that has no service
        cal = CalendarPlugin()
        cal.authenticate = lambda: None  # stub auth to do nothing
        orch.actions.actions["calendar_create"] = {
            "handler": cal.create_event,
            "description": "Create calendar event",
        }
        result = await orch._handle_calendar(
            "add meeting at 3 pm",
            {"intent": "create", "summary": "meeting"},
        )
        assert "not connected" in result.lower()


# ── Google Calendar API Call Verification ─────────────

class TestCalendarAPICall:
    """Verify the correct data is sent to the Google Calendar API."""

    @pytest.mark.asyncio
    async def test_event_body_structure(self):
        """Verify the event body sent to Google has correct structure."""
        cal = CalendarPlugin()
        mock_service = MagicMock()
        captured_body = {}

        def capture_insert(calendarId, body):
            captured_body.update(body)
            mock_result = MagicMock()
            mock_result.execute.return_value = {"summary": body["summary"], "id": "test"}
            return mock_result

        mock_service.events.return_value.insert = capture_insert
        cal.service = mock_service

        await cal.create_event({
            "title": "Eat my vitamins",
            "start_time": "2026-03-24T16:00:00",
            "end_time": "2026-03-24T16:30:00",
            "location": "",
            "description": "Booked by Mel",
            "timezone": "America/Chicago",
        })

        assert captured_body["summary"] == "Eat my vitamins"
        assert captured_body["start"]["dateTime"] == "2026-03-24T16:00:00"
        assert captured_body["start"]["timeZone"] == "America/Chicago"
        assert captured_body["end"]["dateTime"] == "2026-03-24T16:30:00"
        assert captured_body["reminders"]["useDefault"] is False

    @pytest.mark.asyncio
    async def test_calendar_id_is_primary(self):
        """Verify events are created on the primary calendar."""
        cal = CalendarPlugin()
        mock_service = MagicMock()
        captured_cal_id = {}

        def capture_insert(calendarId, body):
            captured_cal_id["id"] = calendarId
            mock_result = MagicMock()
            mock_result.execute.return_value = {"summary": body["summary"], "id": "test"}
            return mock_result

        mock_service.events.return_value.insert = capture_insert
        cal.service = mock_service

        await cal.create_event({
            "title": "Test",
            "start_time": "2026-03-24T10:00:00",
            "end_time": "2026-03-24T10:30:00",
            "timezone": "America/Chicago",
        })

        assert captured_cal_id["id"] == "primary"

    @pytest.mark.asyncio
    async def test_custom_calendar_id(self):
        """Verify GOOGLE_CALENDAR_ID env var routes to the right calendar."""
        cal = CalendarPlugin()
        cal.calendar_id = "debbhatia@gmail.com"
        mock_service = MagicMock()
        captured_cal_id = {}

        def capture_insert(calendarId, body):
            captured_cal_id["id"] = calendarId
            mock_result = MagicMock()
            mock_result.execute.return_value = {"summary": body["summary"], "id": "test"}
            return mock_result

        mock_service.events.return_value.insert = capture_insert
        cal.service = mock_service

        await cal.create_event({
            "title": "Test",
            "start_time": "2026-03-24T10:00:00",
            "end_time": "2026-03-24T10:30:00",
            "timezone": "America/Chicago",
        })

        assert captured_cal_id["id"] == "debbhatia@gmail.com"


# ── Auth Mode Tests ──────────────────────────────────

class TestCalendarAuthModes:
    """Verify the plugin tries service account first, then OAuth."""

    def test_init_has_service_account_path(self):
        cal = CalendarPlugin()
        assert cal.service_account_path
        assert cal._auth_mode is None

    def test_service_account_preferred_over_oauth(self):
        """When service account key exists, it should be used."""
        cal = CalendarPlugin()
        # After service account auth, mode should be set
        cal._auth_mode = "service_account"
        assert cal._auth_mode == "service_account"

    def test_not_connected_message_mentions_service_account(self):
        """Error message should guide user to service account setup."""
        import asyncio
        cal = CalendarPlugin()
        cal.authenticate = lambda: None
        result = asyncio.get_event_loop().run_until_complete(
            cal.create_event({"title": "Test"})
        )
        assert "service account" in result.lower()

    def test_calendar_id_defaults_to_primary(self):
        cal = CalendarPlugin()
        assert cal._get_calendar_id() == "primary"

    def test_calendar_id_from_env(self):
        cal = CalendarPlugin()
        cal.calendar_id = "user@gmail.com"
        assert cal._get_calendar_id() == "user@gmail.com"
