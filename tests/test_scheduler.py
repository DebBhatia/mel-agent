"""Tests for the scheduler/reminders module."""

import os
import sys
import json
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scheduler import (
    Reminder,
    ReminderStore,
    TaskScheduler,
    parse_reminder_from_text,
)


class TestReminder:
    def test_defaults(self):
        r = Reminder(id="test", title="Test", trigger_time="2026-03-15T10:00:00")
        assert r.status == "pending"
        assert r.recurrence is None
        assert r.action is None
        assert r.created_at  # Should have timestamp

    def test_with_recurrence(self):
        r = Reminder(id="test", title="Weekly", trigger_time="2026-03-15T10:00:00", recurrence="weekly")
        assert r.recurrence == "weekly"


class TestReminderStore:
    def test_add_and_get_pending(self, tmp_path):
        store = ReminderStore(path=str(tmp_path / "reminders.json"))
        r = Reminder(id="r1", title="Test", trigger_time="2099-01-01T00:00:00")
        store.add(r)
        pending = store.get_pending()
        assert len(pending) == 1
        assert pending[0].id == "r1"

    def test_cancel(self, tmp_path):
        store = ReminderStore(path=str(tmp_path / "reminders.json"))
        store.add(Reminder(id="r2", title="Test", trigger_time="2099-01-01T00:00:00"))
        assert store.cancel("r2")
        assert len(store.get_pending()) == 0

    def test_cancel_nonexistent(self, tmp_path):
        store = ReminderStore(path=str(tmp_path / "reminders.json"))
        assert not store.cancel("nonexistent")

    def test_remove(self, tmp_path):
        store = ReminderStore(path=str(tmp_path / "reminders.json"))
        store.add(Reminder(id="r3", title="Test", trigger_time="2099-01-01T00:00:00"))
        assert store.remove("r3")
        assert len(store.reminders) == 0

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "reminders.json")
        store1 = ReminderStore(path=path)
        store1.add(Reminder(id="rp", title="Persist", trigger_time="2099-01-01T00:00:00"))
        store2 = ReminderStore(path=path)
        assert len(store2.reminders) == 1
        assert store2.reminders[0].title == "Persist"

    def test_get_due(self, tmp_path):
        store = ReminderStore(path=str(tmp_path / "reminders.json"))
        # Past time
        store.add(Reminder(id="due1", title="Due", trigger_time="2020-01-01T00:00:00"))
        # Future time
        store.add(Reminder(id="future1", title="Future", trigger_time="2099-01-01T00:00:00"))
        due = store.get_due()
        assert len(due) == 1
        assert due[0].id == "due1"

    def test_mark_triggered_onetime(self, tmp_path):
        store = ReminderStore(path=str(tmp_path / "reminders.json"))
        store.add(Reminder(id="t1", title="Once", trigger_time="2020-01-01T00:00:00"))
        store.mark_triggered("t1")
        assert store.reminders[0].status == "triggered"

    def test_mark_triggered_recurring(self, tmp_path):
        store = ReminderStore(path=str(tmp_path / "reminders.json"))
        store.add(Reminder(id="t2", title="Daily", trigger_time="2020-01-01T00:00:00", recurrence="daily"))
        store.mark_triggered("t2")
        assert store.reminders[0].status == "pending"  # Still pending
        assert store.reminders[0].trigger_time > "2020-01-01T00:00:00"  # Rescheduled


class TestTaskScheduler:
    def test_create_reminder(self, tmp_path):
        sched = TaskScheduler()
        sched.store = ReminderStore(path=str(tmp_path / "reminders.json"))
        reminder = sched.create_reminder("Test", "2099-01-01T00:00:00")
        assert reminder.title == "Test"
        assert reminder.id
        assert len(sched.get_pending()) == 1

    def test_create_reminder_relative(self, tmp_path):
        sched = TaskScheduler()
        sched.store = ReminderStore(path=str(tmp_path / "reminders.json"))
        reminder = sched.create_reminder_relative("In 30 min", minutes=30)
        trigger = datetime.fromisoformat(reminder.trigger_time)
        now = datetime.now()
        assert trigger > now
        diff = (trigger - now).total_seconds()
        assert 1700 < diff < 1900  # ~30 minutes

    def test_cancel_reminder(self, tmp_path):
        sched = TaskScheduler()
        sched.store = ReminderStore(path=str(tmp_path / "reminders.json"))
        r = sched.create_reminder("Cancel me", "2099-01-01T00:00:00")
        assert sched.cancel_reminder(r.id)
        assert len(sched.get_pending()) == 0

    @pytest.mark.asyncio
    async def test_check_and_trigger(self, tmp_path):
        sched = TaskScheduler()
        sched.store = ReminderStore(path=str(tmp_path / "reminders.json"))
        sched.create_reminder("Past", "2020-01-01T00:00:00")
        sched.create_reminder("Future", "2099-01-01T00:00:00")
        triggered = await sched.check_and_trigger()
        assert len(triggered) == 1
        assert triggered[0].title == "Past"


class TestParseReminderFromText:
    def test_in_minutes(self):
        result = parse_reminder_from_text("in 30 minutes call mom")
        assert result["trigger_time"] is not None
        dt = datetime.fromisoformat(result["trigger_time"])
        assert dt > datetime.now()

    def test_in_hours(self):
        result = parse_reminder_from_text("in 2 hours check email")
        dt = datetime.fromisoformat(result["trigger_time"])
        diff = (dt - datetime.now()).total_seconds()
        assert 7000 < diff < 7400

    def test_at_time(self):
        result = parse_reminder_from_text("at 3:00 pm do something")
        assert result["trigger_time"] is not None

    def test_every_day(self):
        result = parse_reminder_from_text("every day check calendar")
        assert result["recurrence"] == "daily"

    def test_every_week(self):
        result = parse_reminder_from_text("every week review tasks")
        assert result["recurrence"] == "weekly"

    def test_default_trigger(self):
        result = parse_reminder_from_text("just a reminder about something")
        assert result["trigger_time"] is not None  # Defaults to 1 hour
