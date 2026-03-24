"""Tests for wake_listener module — WakeMode, VoiceAgent helpers, calendar formatting."""

import os
import sys
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wake_listener import (
    WakeMode,
    WakeWordDetector,
    VoiceAgent,
    TextToSpeech,
    SAMPLE_RATE,
    CHUNK_SIZE,
    SILENCE_THRESHOLD,
)


# ── WakeMode ──────────────────────────────────

class TestWakeMode:
    def test_homecoming_value(self):
        assert WakeMode.HOMECOMING.value == "homecoming"

    def test_command_value(self):
        assert WakeMode.COMMAND.value == "command"


# ── WakeWordDetector ──────────────────────────

class TestWakeWordDetector:
    def test_no_engine_returns_none(self):
        detector = WakeWordDetector()
        # No porcupine loaded, not in keyboard fallback
        detector.porcupine = None
        detector.use_keyboard_fallback = False
        chunk = np.zeros(CHUNK_SIZE, dtype=np.int16)
        assert detector.detect(chunk) is None

    def test_keyboard_fallback_returns_none_when_not_triggered(self):
        detector = WakeWordDetector()
        detector.use_keyboard_fallback = True
        detector._keyboard_triggered = None
        chunk = np.zeros(CHUNK_SIZE, dtype=np.int16)
        assert detector.detect(chunk) is None

    def test_keyboard_fallback_returns_mode_when_triggered(self):
        detector = WakeWordDetector()
        detector.use_keyboard_fallback = True
        detector._keyboard_triggered = WakeMode.COMMAND
        chunk = np.zeros(CHUNK_SIZE, dtype=np.int16)
        assert detector.detect(chunk) == WakeMode.COMMAND
        # Should be consumed after detection
        assert detector._keyboard_triggered is None


# ── VoiceAgent._get_time_greeting ─────────────

class TestTimeGreeting:
    def test_morning(self):
        agent = VoiceAgent.__new__(VoiceAgent)
        with patch("wake_listener.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 15, 8, 0)
            mock_dt.fromisoformat = datetime.fromisoformat
            greeting, followup = agent._get_time_greeting()
            assert "morning" in greeting.lower()

    def test_afternoon(self):
        agent = VoiceAgent.__new__(VoiceAgent)
        with patch("wake_listener.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 15, 14, 0)
            greeting, followup = agent._get_time_greeting()
            assert "afternoon" in greeting.lower()

    def test_evening(self):
        agent = VoiceAgent.__new__(VoiceAgent)
        with patch("wake_listener.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 15, 19, 0)
            greeting, followup = agent._get_time_greeting()
            assert "evening" in greeting.lower()

    def test_night(self):
        agent = VoiceAgent.__new__(VoiceAgent)
        with patch("wake_listener.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 3, 15, 23, 0)
            greeting, followup = agent._get_time_greeting()
            assert "hey" in greeting.lower() or "there" in greeting.lower()


# ── VoiceAgent._format_calendar_events ────────

class TestFormatCalendarEvents:
    def test_string_passthrough(self):
        result = VoiceAgent._format_calendar_events("Already formatted")
        assert result == "Already formatted"

    def test_empty_list(self):
        assert VoiceAgent._format_calendar_events([]) == ""

    def test_single_event(self):
        events = [{"summary": "Team standup", "start": {"dateTime": "2026-03-15T09:00:00"}}]
        result = VoiceAgent._format_calendar_events(events)
        assert "one thing" in result.lower()
        assert "Team standup" in result

    def test_multiple_events(self):
        events = [
            {"summary": "Standup", "start": {"dateTime": "2026-03-15T09:00:00"}},
            {"summary": "Lunch", "start": {"dateTime": "2026-03-15T12:00:00"}},
            {"summary": "Review", "start": {"dateTime": "2026-03-15T15:00:00"}},
        ]
        result = VoiceAgent._format_calendar_events(events)
        assert "3 things" in result
        assert "Standup" in result
        assert "Lunch" in result
        assert "Review" in result

    def test_string_events(self):
        events = ["Meeting at 9am", "Lunch at noon"]
        result = VoiceAgent._format_calendar_events(events)
        assert "2 things" in result

    def test_date_only_event(self):
        events = [{"summary": "All day event", "start": {"date": "2026-03-15"}}]
        result = VoiceAgent._format_calendar_events(events)
        assert "All day event" in result

    def test_non_list_returns_empty(self):
        assert VoiceAgent._format_calendar_events(42) == ""
        assert VoiceAgent._format_calendar_events({}) == ""


# ── TextToSpeech ──────────────────────────────

class TestTextToSpeech:
    def test_default_engine(self):
        tts = TextToSpeech()
        assert tts.engine in ["piper", "elevenlabs"]


# ── Constants ─────────────────────────────────

class TestConstants:
    def test_sample_rate(self):
        assert SAMPLE_RATE == 16000

    def test_chunk_size(self):
        assert CHUNK_SIZE == 1280

    def test_silence_threshold(self):
        assert SILENCE_THRESHOLD == 500
