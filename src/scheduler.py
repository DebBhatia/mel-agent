"""
SCHEDULED TASKS & REMINDERS
=============================
Background scheduler for recurring tasks, reminders, and timed actions.
Uses APScheduler for cron-like scheduling with local persistence.

Supports:
- One-time reminders ("remind me at 3pm to call Mom")
- Recurring tasks ("every Monday check my calendar")
- Delayed actions ("in 30 minutes pause the music")

All scheduling runs locally. No cloud dependency.
"""

import os
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("scheduler")

REMINDERS_PATH = os.path.expanduser("~/.config/agent/reminders.json")


@dataclass
class Reminder:
    id: str
    title: str
    trigger_time: str  # ISO format
    recurrence: Optional[str] = None  # "daily", "weekly", "monthly", or None for one-time
    status: str = "pending"  # pending, triggered, cancelled
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    action: Optional[str] = None  # Optional action to execute when triggered
    action_params: dict = field(default_factory=dict)


class ReminderStore:
    """Persistent store for reminders and scheduled tasks."""

    def __init__(self, path: str = None):
        self.path = path or REMINDERS_PATH
        self.reminders: list[Reminder] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self.reminders = [Reminder(**r) for r in data]
            except Exception as e:
                logger.warning(f"Failed to load reminders: {e}")
                self.reminders = []

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump([asdict(r) for r in self.reminders], f, indent=2)

    def add(self, reminder: Reminder) -> str:
        self.reminders.append(reminder)
        self._save()
        logger.info(f"Reminder added: {reminder.id} - {reminder.title}")
        return reminder.id

    def remove(self, reminder_id: str) -> bool:
        before = len(self.reminders)
        self.reminders = [r for r in self.reminders if r.id != reminder_id]
        if len(self.reminders) < before:
            self._save()
            return True
        return False

    def cancel(self, reminder_id: str) -> bool:
        for r in self.reminders:
            if r.id == reminder_id:
                r.status = "cancelled"
                self._save()
                return True
        return False

    def get_pending(self) -> list[Reminder]:
        return [r for r in self.reminders if r.status == "pending"]

    def get_due(self) -> list[Reminder]:
        """Get reminders that are due (trigger_time <= now)."""
        now = datetime.now().isoformat()
        return [r for r in self.reminders if r.status == "pending" and r.trigger_time <= now]

    def mark_triggered(self, reminder_id: str):
        for r in self.reminders:
            if r.id == reminder_id:
                if r.recurrence:
                    # Reschedule recurring reminder
                    r.trigger_time = self._next_occurrence(r.trigger_time, r.recurrence)
                else:
                    r.status = "triggered"
                self._save()
                return

    @staticmethod
    def _next_occurrence(current_time: str, recurrence: str) -> str:
        dt = datetime.fromisoformat(current_time)
        if recurrence == "daily":
            dt += timedelta(days=1)
        elif recurrence == "weekly":
            dt += timedelta(weeks=1)
        elif recurrence == "monthly":
            # Approximate month increment
            month = dt.month + 1
            year = dt.year
            if month > 12:
                month = 1
                year += 1
            try:
                dt = dt.replace(year=year, month=month)
            except ValueError:
                dt = dt.replace(year=year, month=month, day=28)
        elif recurrence == "hourly":
            dt += timedelta(hours=1)
        return dt.isoformat()

    def get_all(self) -> list[dict]:
        return [asdict(r) for r in self.reminders]


class TaskScheduler:
    """
    Lightweight scheduler that checks for due reminders periodically.
    Designed to be polled from the server's health-check loop.
    No separate process needed — integrates with the existing FastAPI event loop.
    """

    def __init__(self, callback: Callable = None):
        self.store = ReminderStore()
        self.callback = callback  # Called with (reminder) when a reminder triggers
        self._triggered_ids: set = set()  # Prevent double-triggers within same poll cycle

    def create_reminder(self, title: str, trigger_time: str,
                        recurrence: str = None, action: str = None,
                        action_params: dict = None) -> Reminder:
        reminder = Reminder(
            id=uuid.uuid4().hex[:12],
            title=title,
            trigger_time=trigger_time,
            recurrence=recurrence,
            action=action,
            action_params=action_params or {},
        )
        self.store.add(reminder)
        return reminder

    def create_reminder_relative(self, title: str, minutes: int = 0,
                                  hours: int = 0, days: int = 0,
                                  action: str = None, action_params: dict = None) -> Reminder:
        """Create a reminder relative to now (e.g., 'in 30 minutes')."""
        trigger = datetime.now() + timedelta(minutes=minutes, hours=hours, days=days)
        return self.create_reminder(title, trigger.isoformat(), action=action, action_params=action_params)

    def cancel_reminder(self, reminder_id: str) -> bool:
        return self.store.cancel(reminder_id)

    def get_pending(self) -> list[dict]:
        return [asdict(r) for r in self.store.get_pending()]

    def get_all(self) -> list[dict]:
        return self.store.get_all()

    async def check_and_trigger(self) -> list[Reminder]:
        """Check for due reminders and trigger them. Call this periodically."""
        due = self.store.get_due()
        triggered = []
        for reminder in due:
            if reminder.id in self._triggered_ids:
                continue
            self._triggered_ids.add(reminder.id)
            self.store.mark_triggered(reminder.id)
            triggered.append(reminder)
            logger.info(f"Reminder triggered: {reminder.title}")
            if self.callback:
                try:
                    await self.callback(reminder)
                except Exception as e:
                    logger.error(f"Reminder callback error: {e}")
        # Clean up triggered IDs periodically
        if len(self._triggered_ids) > 100:
            self._triggered_ids.clear()
        return triggered


def parse_reminder_from_text(text: str) -> dict:
    """
    Parse natural language into reminder parameters.
    Returns dict with title, trigger_time, recurrence.
    """
    import re
    now = datetime.now()
    result = {"title": text, "trigger_time": None, "recurrence": None}

    # "in X minutes/hours"
    m = re.search(r'in\s+(\d+)\s+(minute|min|hour|hr|day)s?', text, re.IGNORECASE)
    if m:
        amount = int(m.group(1))
        unit = m.group(2).lower()
        if unit in ("minute", "min"):
            trigger = now + timedelta(minutes=amount)
        elif unit in ("hour", "hr"):
            trigger = now + timedelta(hours=amount)
        elif unit == "day":
            trigger = now + timedelta(days=amount)
        else:
            trigger = now + timedelta(minutes=amount)
        result["trigger_time"] = trigger.isoformat()
        # Clean title
        result["title"] = re.sub(r'in\s+\d+\s+\w+s?\s*', '', text).strip()
        return result

    # "at HH:MM am/pm"
    m = re.search(r'at\s+(\d{1,2}):?(\d{2})?\s*(am|pm)?', text, re.IGNORECASE)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm = (m.group(3) or '').lower()
        if ampm == 'pm' and hour < 12:
            hour += 12
        elif ampm == 'am' and hour == 12:
            hour = 0
        trigger = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if trigger <= now:
            trigger += timedelta(days=1)
        result["trigger_time"] = trigger.isoformat()
        result["title"] = re.sub(r'at\s+\d{1,2}:?\d{0,2}\s*(?:am|pm)?\s*', '', text, flags=re.IGNORECASE).strip()
        return result

    # "tomorrow at..."
    m = re.search(r'tomorrow\s+(?:at\s+)?(\d{1,2}):?(\d{2})?\s*(am|pm)?', text, re.IGNORECASE)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm = (m.group(3) or '').lower()
        if ampm == 'pm' and hour < 12:
            hour += 12
        elif ampm == 'am' and hour == 12:
            hour = 0
        trigger = (now + timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0)
        result["trigger_time"] = trigger.isoformat()
        result["title"] = re.sub(r'tomorrow\s+(?:at\s+)?\d{1,2}:?\d{0,2}\s*(?:am|pm)?\s*', '', text, flags=re.IGNORECASE).strip()
        return result

    # "every day/week/month"
    m = re.search(r'every\s+(day|daily|week|weekly|month|monthly|hour|hourly)', text, re.IGNORECASE)
    if m:
        freq = m.group(1).lower()
        recurrence_map = {
            "day": "daily", "daily": "daily",
            "week": "weekly", "weekly": "weekly",
            "month": "monthly", "monthly": "monthly",
            "hour": "hourly", "hourly": "hourly",
        }
        result["recurrence"] = recurrence_map.get(freq, "daily")
        result["trigger_time"] = (now + timedelta(hours=1)).isoformat()
        result["title"] = re.sub(r'every\s+\w+\s*', '', text).strip()
        return result

    # Default: 1 hour from now
    if not result["trigger_time"]:
        result["trigger_time"] = (now + timedelta(hours=1)).isoformat()

    return result
