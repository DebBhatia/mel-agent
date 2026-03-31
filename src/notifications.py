"""
NOTIFICATION MODULE
====================
Push notifications to your phone/desktop.
Supports multiple backends: Ntfy (free, self-hostable), Pushover, and Telegram.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict

import httpx

logger = logging.getLogger("notifications")

NOTIFICATION_LOG = os.path.join(os.path.dirname(__file__), "..", "data", "notification_log.json")


def _load_secret(key: str, default: str = "") -> str:
    """Load a secret from the encrypted vault; fall back to env var."""
    try:
        from security import SecretVault
        val = SecretVault().get(key, "")
        if val:
            return val
    except Exception:
        pass
    return os.getenv(key, default)


@dataclass
class Notification:
    id: str
    title: str
    message: str
    priority: str = "normal"  # low, normal, high, urgent
    channel: str = ""         # Which backend sent it
    sent_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "sent"


class NtfyBackend:
    """
    Ntfy.sh — free, open-source push notifications.
    No account needed. Just pick a topic name.
    Install ntfy app on your phone and subscribe to the same topic.
    """

    def __init__(self):
        self.server = os.getenv("NTFY_SERVER", "https://ntfy.sh")
        self.topic = _load_secret("NTFY_TOPIC")
        self.token = _load_secret("NTFY_TOKEN")

    def is_configured(self) -> bool:
        return bool(self.topic)

    async def send(self, title: str, message: str, priority: str = "default", tags: str = "") -> bool:
        """Send notification via ntfy."""
        if not self.is_configured():
            return False

        headers = {"Title": title, "Priority": priority}
        if tags:
            headers["Tags"] = tags
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{self.server}/{self.topic}",
                    content=message,
                    headers=headers,
                )
                return resp.status_code == 200
        except Exception as e:
            logger.error(f"Ntfy send failed: {e}")
            return False


class PushoverBackend:
    """Pushover — reliable push notifications ($5 one-time purchase)."""

    def __init__(self):
        self.user_key = _load_secret("PUSHOVER_USER_KEY")
        self.api_token = _load_secret("PUSHOVER_API_TOKEN")

    def is_configured(self) -> bool:
        return bool(self.user_key and self.api_token)

    async def send(self, title: str, message: str, priority: str = "0", **kwargs) -> bool:
        """Send notification via Pushover."""
        if not self.is_configured():
            return False

        priority_map = {"low": "-1", "normal": "0", "high": "1", "urgent": "2"}
        pushover_priority = priority_map.get(priority, priority)

        payload = {
            "token": self.api_token,
            "user": self.user_key,
            "title": title,
            "message": message,
            "priority": pushover_priority,
        }

        if pushover_priority == "2":
            payload["retry"] = 60
            payload["expire"] = 3600

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    "https://api.pushover.net/1/messages.json",
                    data=payload,
                )
                return resp.status_code == 200
        except Exception as e:
            logger.error(f"Pushover send failed: {e}")
            return False


class TelegramBackend:
    """Telegram Bot API — free, feature-rich notifications."""

    def __init__(self):
        self.bot_token = _load_secret("TELEGRAM_BOT_TOKEN")
        self.chat_id = _load_secret("TELEGRAM_CHAT_ID")

    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    async def send(self, title: str, message: str, **kwargs) -> bool:
        """Send notification via Telegram."""
        if not self.is_configured():
            return False

        text = f"*{title}*\n{message}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": "Markdown",
                    },
                )
                return resp.status_code == 200
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False


class NotificationService:
    """Unified notification service — sends via all configured backends."""

    def __init__(self):
        self.ntfy = NtfyBackend()
        self.pushover = PushoverBackend()
        self.telegram = TelegramBackend()
        self._log: list[dict] = []
        self._load_log()

    def _ensure_data_dir(self):
        os.makedirs(os.path.dirname(NOTIFICATION_LOG), exist_ok=True)

    def _load_log(self):
        try:
            if os.path.exists(NOTIFICATION_LOG):
                with open(NOTIFICATION_LOG) as f:
                    self._log = json.load(f)
        except Exception:
            self._log = []

    def _save_log(self, notification: Notification):
        self._ensure_data_dir()
        self._log.append(asdict(notification))
        # Keep last 100 entries
        self._log = self._log[-100:]
        try:
            with open(NOTIFICATION_LOG, "w") as f:
                json.dump(self._log, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save notification log: {e}")

    def get_configured_backends(self) -> list:
        """List which notification backends are configured."""
        backends = []
        if self.ntfy.is_configured():
            backends.append("ntfy")
        if self.pushover.is_configured():
            backends.append("pushover")
        if self.telegram.is_configured():
            backends.append("telegram")
        return backends

    def is_configured(self) -> bool:
        return len(self.get_configured_backends()) > 0

    async def send(self, title: str, message: str, priority: str = "normal", backend: Optional[str] = None) -> dict:
        """
        Send notification via configured backend(s).
        If backend is specified, use only that one.
        Otherwise, send via all configured backends.
        """
        import uuid
        results = {}

        backends_to_use = []
        if backend:
            backends_to_use = [backend]
        else:
            backends_to_use = self.get_configured_backends()

        if not backends_to_use:
            return {"status": "error", "message": "No notification backends configured. Add NTFY_TOPIC, PUSHOVER credentials, or TELEGRAM_BOT_TOKEN to .env"}

        for b in backends_to_use:
            if b == "ntfy":
                success = await self.ntfy.send(title, message, priority)
                results["ntfy"] = "sent" if success else "failed"
            elif b == "pushover":
                success = await self.pushover.send(title, message, priority)
                results["pushover"] = "sent" if success else "failed"
            elif b == "telegram":
                success = await self.telegram.send(title, message)
                results["telegram"] = "sent" if success else "failed"

        notification = Notification(
            id=uuid.uuid4().hex[:12],
            title=title,
            message=message,
            priority=priority,
            channel=",".join(backends_to_use),
            status="sent" if any(v == "sent" for v in results.values()) else "failed",
        )
        self._save_log(notification)

        sent_count = sum(1 for v in results.values() if v == "sent")
        return {
            "status": "sent" if sent_count > 0 else "failed",
            "backends": results,
            "notification_id": notification.id,
        }

    async def send_reminder_notification(self, title: str) -> dict:
        """Send a reminder notification with high priority."""
        return await self.send(
            title=f"Reminder: {title}",
            message=title,
            priority="high",
        )

    def get_recent(self, count: int = 20) -> list:
        """Get recent notification history."""
        return self._log[-count:]


def register_notification_plugins(action_registry):
    """Register notification actions with the action registry."""
    service = NotificationService()

    async def notify_send(params: dict) -> str:
        result = await service.send(
            title=params.get("title", "Mel Agent"),
            message=params.get("message", ""),
            priority=params.get("priority", "normal"),
            backend=params.get("backend"),
        )
        if result["status"] == "sent":
            return f"Notification sent via {', '.join(result['backends'].keys())}."
        return f"Notification failed: {result.get('message', 'Unknown error')}"

    action_registry.register("notify_send", notify_send, "Send push notification")

    return service
