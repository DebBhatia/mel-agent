"""Tests for wake_listener module — WakeMode, VoiceAgent helpers, calendar formatting."""

import os
import sys
import asyncio
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
    AudioCapture,
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


# ── Barge-in ──────────────────────────────────

class TestWatchForSpeech:
    @pytest.mark.asyncio
    async def test_detects_sustained_loud_audio(self):
        audio = AudioCapture()
        quiet = np.zeros(CHUNK_SIZE, dtype=np.int16)
        loud = np.full(CHUNK_SIZE, 20000, dtype=np.int16)
        chunks = iter([quiet, loud, loud, loud, loud])
        audio.read_chunk = MagicMock(side_effect=lambda: next(chunks))

        triggered, heard = await audio.watch_for_speech(
            threshold=1000, min_consecutive=3, stop_event=asyncio.Event()
        )

        assert triggered is True
        # The leading quiet chunk resets the streak, so only the 3
        # consecutive loud chunks that actually triggered it are returned.
        assert len(heard) == 3

    @pytest.mark.asyncio
    async def test_single_loud_blip_does_not_trigger(self):
        """One loud chunk surrounded by quiet ones shouldn't count as speech --
        this is the debounce that protects against a click/pop/bump."""
        audio = AudioCapture()
        quiet = np.zeros(CHUNK_SIZE, dtype=np.int16)
        loud = np.full(CHUNK_SIZE, 20000, dtype=np.int16)
        stop_event = asyncio.Event()
        chunks = iter([quiet, loud, quiet, quiet])

        def _read():
            try:
                return next(chunks)
            except StopIteration:
                stop_event.set()
                return quiet

        audio.read_chunk = MagicMock(side_effect=_read)
        triggered, _ = await audio.watch_for_speech(
            threshold=1000, min_consecutive=3, stop_event=stop_event
        )
        assert triggered is False

    @pytest.mark.asyncio
    async def test_stops_when_event_set_before_threshold(self):
        audio = AudioCapture()
        quiet = np.zeros(CHUNK_SIZE, dtype=np.int16)
        stop_event = asyncio.Event()
        calls = {"n": 0}

        def _read():
            calls["n"] += 1
            if calls["n"] >= 3:
                stop_event.set()
            return quiet

        audio.read_chunk = MagicMock(side_effect=_read)
        triggered, _ = await audio.watch_for_speech(
            threshold=1000, min_consecutive=3, stop_event=stop_event
        )
        assert triggered is False


class TestCaptureCommandTextPreroll:
    @pytest.mark.asyncio
    async def test_preroll_is_prepended_to_captured_audio(self):
        agent = VoiceAgent()
        quiet = np.zeros(CHUNK_SIZE, dtype=np.int16)
        agent.audio.read_chunk = MagicMock(return_value=quiet)
        agent.audio.drain = MagicMock()
        agent.stt.transcribe = MagicMock(return_value="test command")
        preroll_chunk = np.full(CHUNK_SIZE, 5000, dtype=np.int16)

        with patch("wake_listener._notify_server", new=AsyncMock()):
            result = await agent._capture_command_text(drain_first=False, preroll=[preroll_chunk])

        assert result == "test command"
        transcribed_audio = agent.stt.transcribe.call_args[0][0]
        # Preroll chunk plus at least the silence chunks read during capture.
        assert len(transcribed_audio) > CHUNK_SIZE


class TestPlayWithBargeIn:
    @pytest.mark.asyncio
    async def test_interrupted_when_watcher_detects_speech(self):
        agent = VoiceAgent()

        async def slow_play(path):
            await asyncio.sleep(10)  # would hang the test if not cancelled

        agent.tts._play_file = slow_play
        preroll_chunk = np.full(CHUNK_SIZE, 5000, dtype=np.int16)
        agent.audio.watch_for_speech = AsyncMock(return_value=(True, [preroll_chunk]))
        agent._capture_command_text = AsyncMock(return_value="wait actually")

        interrupted, heard = await asyncio.wait_for(
            agent._play_with_barge_in("/tmp/fake.mp3"), timeout=5
        )

        assert interrupted is True
        assert heard == "wait actually"
        agent._capture_command_text.assert_awaited_once_with(drain_first=False, preroll=[preroll_chunk])

    @pytest.mark.asyncio
    async def test_not_interrupted_when_playback_finishes_first(self):
        agent = VoiceAgent()

        async def fast_play(path):
            await asyncio.sleep(0.01)

        async def never_triggers(threshold, min_consecutive, stop_event):
            await stop_event.wait()
            return False, []

        agent.tts._play_file = fast_play
        agent.audio.watch_for_speech = never_triggers

        interrupted, heard = await asyncio.wait_for(
            agent._play_with_barge_in("/tmp/fake.mp3"), timeout=5
        )

        assert interrupted is False
        assert heard == ""
