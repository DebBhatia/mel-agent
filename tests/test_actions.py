"""Tests for action plugins — CalendarPlugin, ReservationPlugin, CommunicationPlugin, registration."""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from actions import (
    CalendarPlugin,
    ReservationPlugin,
    CommunicationPlugin,
    register_all_plugins,
)
from orchestrator import ActionRegistry


# ── CalendarPlugin ────────────────────────────

class TestCalendarPlugin:
    def test_init_defaults(self):
        cal = CalendarPlugin()
        assert cal.service is None
        assert cal.credentials_path
        assert cal.token_path

    @pytest.mark.asyncio
    async def test_create_event_no_service(self):
        cal = CalendarPlugin()
        cal.authenticate = lambda: None  # Mock auth to do nothing
        result = await cal.create_event({"title": "Test"})
        assert "not connected" in result.lower()

    @pytest.mark.asyncio
    async def test_get_events_no_service(self):
        cal = CalendarPlugin()
        cal.authenticate = lambda: None
        result = await cal.get_events({"days": 7})
        assert "not connected" in result.lower()

    @pytest.mark.asyncio
    async def test_check_availability_no_service(self):
        cal = CalendarPlugin()
        cal.authenticate = lambda: None
        result = await cal.check_availability({})
        assert "not connected" in result.lower()


# ── ReservationPlugin ─────────────────────────

class TestReservationPlugin:
    def test_init_twilio_config(self):
        rp = ReservationPlugin()
        # Should read from env (likely empty in test)
        assert isinstance(rp.twilio_sid, str)
        assert isinstance(rp.twilio_token, str)

    @pytest.mark.asyncio
    async def test_search_restaurants_returns_json(self):
        rp = ReservationPlugin()
        result = await rp.search_restaurants({
            "cuisine": "Italian",
            "location": "downtown",
            "party_size": 4,
        })
        data = json.loads(result)
        assert data["cuisine"] == "Italian"
        assert data["party_size"] == 4

    @pytest.mark.asyncio
    async def test_reservation_call_no_twilio(self):
        rp = ReservationPlugin()
        rp.twilio_sid = ""
        result = await rp.make_reservation_call({"phone_number": "555-1234"})
        assert "not configured" in result.lower()

    @pytest.mark.asyncio
    async def test_reservation_call_no_phone(self):
        rp = ReservationPlugin()
        rp.twilio_sid = "test"
        rp.twilio_token = "test"
        rp.twilio_phone = "+1234567890"
        result = await rp.make_reservation_call({})
        assert "phone number" in result.lower()


# ── CommunicationPlugin ──────────────────────

class TestCommunicationPlugin:
    @pytest.mark.asyncio
    async def test_send_sms_no_twilio(self):
        comm = CommunicationPlugin()
        comm.twilio_sid = ""
        result = await comm.send_sms({"to": "+1234567890", "message": "Hi"})
        assert "not configured" in result.lower()

    @pytest.mark.asyncio
    async def test_send_email_no_smtp(self):
        comm = CommunicationPlugin()
        # SMTP_USER is likely not set in test env
        os.environ.pop("SMTP_USER", None)
        result = await comm.send_email({"to": "x@y.com", "body": "Hi"})
        assert "not configured" in result.lower()


# ── Plugin Registration ───────────────────────

class TestPluginRegistration:
    def test_register_all(self):
        registry = ActionRegistry()
        register_all_plugins(registry)
        assert "calendar_create" in registry.actions
        assert "calendar_list" in registry.actions
        assert "calendar_check" in registry.actions
        assert "restaurant_search" in registry.actions
        assert "restaurant_call" in registry.actions
        assert "send_sms" in registry.actions
        assert "send_email" in registry.actions
        # Action count grows as new plugins are added (calendar, gmail, etc.)
        assert len(registry.actions) >= 7
