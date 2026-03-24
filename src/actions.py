"""
ACTION PLUGINS
===============
Pluggable actions your agent can perform.
Each plugin handles a specific domain (calendar, reservations, etc.)
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

import httpx

logger = logging.getLogger("actions")


# ─────────────────────────────────────────────
# Google Calendar Plugin
# ─────────────────────────────────────────────
class CalendarPlugin:
    """
    Manages Google Calendar events via Google Calendar API.

    Supports two auth modes (in priority order):
    1. **Service Account** (recommended) — uses a JSON key file that never expires.
       Set GOOGLE_SERVICE_ACCOUNT_PATH or place file at ~/.config/agent/google_service_account.json
       Then share your calendar with the service account email (grant "Make changes to events").
       Set GOOGLE_CALENDAR_ID to your email (e.g. debbhatia@gmail.com).
    2. **OAuth2** (legacy) — uses browser-based login with refresh token.
       Requires GOOGLE_CREDENTIALS_PATH and manual first-run browser auth.
    """

    SCOPES = ["https://www.googleapis.com/auth/calendar"]

    def __init__(self):
        # Service account (preferred — never expires)
        self.service_account_path = os.getenv(
            "GOOGLE_SERVICE_ACCOUNT_PATH",
            os.path.expanduser("~/.config/agent/google_service_account.json")
        )
        self.calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")

        # OAuth2 (legacy fallback)
        self.credentials_path = os.getenv(
            "GOOGLE_CREDENTIALS_PATH",
            os.path.expanduser("~/.config/agent/google_credentials.json")
        )
        self.token_path = os.getenv(
            "GOOGLE_TOKEN_PATH",
            os.path.expanduser("~/.config/agent/google_token.json")
        )
        self.service = None
        self._auth_mode = None

    def authenticate(self):
        """Authenticate with Google Calendar API.

        Tries service account first (zero-maintenance), then falls back to OAuth2.
        """
        # ── Attempt 1: Service Account (recommended) ──
        if os.path.exists(self.service_account_path):
            try:
                from google.oauth2 import service_account
                from googleapiclient.discovery import build

                creds = service_account.Credentials.from_service_account_file(
                    self.service_account_path, scopes=self.SCOPES
                )
                self.service = build("calendar", "v3", credentials=creds)
                self._auth_mode = "service_account"
                logger.info("Google Calendar authenticated via service account")
                return
            except Exception as e:
                logger.warning(f"Service account auth failed: {e}")

        # ── Attempt 2: OAuth2 (legacy) ──
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
            from googleapiclient.discovery import build

            creds = None
            if os.path.exists(self.token_path):
                creds = Credentials.from_authorized_user_file(self.token_path, self.SCOPES)

            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                else:
                    if not os.path.exists(self.credentials_path):
                        logger.error(
                            "No Google credentials found. "
                            "Place a service account key at %s (recommended) "
                            "or OAuth credentials at %s",
                            self.service_account_path,
                            self.credentials_path,
                        )
                        return
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, self.SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                os.makedirs(os.path.dirname(self.token_path), exist_ok=True)
                with open(self.token_path, "w") as f:
                    f.write(creds.to_json())

            self.service = build("calendar", "v3", credentials=creds)
            self._auth_mode = "oauth2"
            logger.info("Google Calendar authenticated via OAuth2")
        except Exception as e:
            logger.error(f"Calendar auth failed: {e}")

    def _get_calendar_id(self) -> str:
        """Return the calendar ID to use for API calls."""
        return self.calendar_id

    async def create_event(self, params: dict) -> str:
        """Create a calendar event."""
        if not self.service:
            self.authenticate()
            if not self.service:
                return (
                    "Calendar not connected. "
                    "Place a Google service account key at "
                    f"{self.service_account_path} and set "
                    "GOOGLE_CALENDAR_ID in your .env file."
                )

        event = {
            "summary": params.get("title", "New Event"),
            "location": params.get("location", ""),
            "description": params.get("description", "Created by AI Agent"),
            "start": {
                "dateTime": params.get("start_time"),
                "timeZone": params.get("timezone", "America/Chicago"),
            },
            "end": {
                "dateTime": params.get("end_time"),
                "timeZone": params.get("timezone", "America/Chicago"),
            },
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "popup", "minutes": 30},
                ],
            },
        }

        try:
            result = self.service.events().insert(
                calendarId=self._get_calendar_id(), body=event
            ).execute()
            # Format a friendly response
            from datetime import datetime
            try:
                start_dt = datetime.fromisoformat(params.get("start_time", ""))
                friendly_time = start_dt.strftime("%-I:%M %p on %B %-d, %Y")
            except (ValueError, TypeError):
                friendly_time = params.get("start_time", "")
            return f"Done! I've added \"{result.get('summary')}\" to your calendar at {friendly_time}."
        except Exception as e:
            return f"Failed to create event: {e}"

    async def get_events(self, params: dict) -> str:
        """Get upcoming events."""
        if not self.service:
            self.authenticate()
            if not self.service:
                return "Calendar not connected."

        now = datetime.utcnow().isoformat() + "Z"
        days_ahead = params.get("days", 7)
        time_max = (datetime.utcnow() + timedelta(days=days_ahead)).isoformat() + "Z"

        try:
            result = self.service.events().list(
                calendarId=self._get_calendar_id(),
                timeMin=now,
                timeMax=time_max,
                maxResults=10,
                singleEvents=True,
                orderBy="startTime",
            ).execute()

            events = result.get("items", [])
            if not events:
                return f"No events in the next {days_ahead} days."

            summary = []
            for event in events:
                start = event["start"].get("dateTime", event["start"].get("date"))
                summary.append(f"- {event['summary']} at {start}")

            return f"Upcoming events ({days_ahead} days):\n" + "\n".join(summary)
        except Exception as e:
            return f"Failed to get events: {e}"

    async def check_availability(self, params: dict) -> str:
        """Check if a time slot is free."""
        if not self.service:
            self.authenticate()
            if not self.service:
                return "Calendar not connected."

        start = params.get("start_time")
        end = params.get("end_time")

        try:
            result = self.service.events().list(
                calendarId=self._get_calendar_id(),
                timeMin=start,
                timeMax=end,
                singleEvents=True,
            ).execute()

            events = result.get("items", [])
            if not events:
                return f"You're free from {start} to {end}!"
            else:
                conflicts = [e["summary"] for e in events]
                return f"Conflict found: {', '.join(conflicts)}"
        except Exception as e:
            return f"Failed to check availability: {e}"


# ─────────────────────────────────────────────
# Restaurant Reservation Plugin
# ─────────────────────────────────────────────
class ReservationPlugin:
    """
    Handles restaurant reservations.
    Can use OpenTable API, Resy API, or Twilio for phone bookings.
    """

    def __init__(self):
        self.twilio_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.twilio_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.twilio_phone = os.getenv("TWILIO_PHONE_NUMBER", "")

    async def search_restaurants(self, params: dict) -> str:
        """Search for restaurants (uses Claude for recommendations)."""
        # This would typically call an API like Yelp or Google Places
        # For now, returns a structured response for the orchestrator
        return json.dumps({
            "action": "search_restaurants",
            "cuisine": params.get("cuisine", ""),
            "location": params.get("location", ""),
            "party_size": params.get("party_size", 2),
            "date": params.get("date", ""),
            "time": params.get("time", ""),
        })

    async def make_reservation_call(self, params: dict) -> str:
        """
        Use Twilio to call a restaurant and make a reservation.
        This is the Mel-style approach - your AI calls for you!
        """
        if not all([self.twilio_sid, self.twilio_token, self.twilio_phone]):
            return "Twilio not configured. Add credentials to .env file."

        from twilio.rest import Client
        client = Client(self.twilio_sid, self.twilio_token)

        restaurant_phone = params.get("phone_number")
        if not restaurant_phone:
            return "Need the restaurant's phone number to make a reservation."

        # TwiML for the AI to speak to the restaurant
        twiml_url = params.get("twiml_url", f"http://your-server.com/reservation-script")

        try:
            call = client.calls.create(
                to=restaurant_phone,
                from_=self.twilio_phone,
                url=twiml_url,
                status_callback=f"http://your-server.com/call-status",
            )
            return "Calling the restaurant to make your reservation. You'll be notified when it's confirmed."
        except Exception as e:
            return f"Failed to make call: {e}"


# ─────────────────────────────────────────────
# Communication Plugin
# ─────────────────────────────────────────────
class CommunicationPlugin:
    """Send messages via SMS, email, or voice."""

    def __init__(self):
        self.twilio_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.twilio_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.twilio_phone = os.getenv("TWILIO_PHONE_NUMBER", "")

    async def send_sms(self, params: dict) -> str:
        """Send an SMS message."""
        if not all([self.twilio_sid, self.twilio_token, self.twilio_phone]):
            return "Twilio not configured."

        from twilio.rest import Client
        client = Client(self.twilio_sid, self.twilio_token)

        try:
            message = client.messages.create(
                body=params.get("message", ""),
                from_=self.twilio_phone,
                to=params.get("to", ""),
            )
            return "SMS sent successfully."
        except Exception as e:
            return f"SMS failed: {e}"

    async def send_email(self, params: dict) -> str:
        """Send email via SMTP."""
        import smtplib
        from email.mime.text import MIMEText

        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER", "")
        smtp_pass = os.getenv("SMTP_PASS", "")  # Use app password!

        if not smtp_user:
            return "Email not configured. Add SMTP credentials to .env"

        msg = MIMEText(params.get("body", ""))
        msg["Subject"] = params.get("subject", "From your AI Agent")
        msg["From"] = smtp_user
        msg["To"] = params.get("to", "")

        try:
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
            return "Email sent successfully."
        except Exception as e:
            return f"Email failed: {e}"


# ─────────────────────────────────────────────
# Plugin Registry Helper
# ─────────────────────────────────────────────
def register_all_plugins(action_registry):
    """Register all available plugins with the action registry."""
    calendar = CalendarPlugin()
    reservation = ReservationPlugin()
    communication = CommunicationPlugin()

    action_registry.register("calendar_create", calendar.create_event, "Create calendar event")
    action_registry.register("calendar_list", calendar.get_events, "List upcoming events")
    action_registry.register("calendar_check", calendar.check_availability, "Check availability")
    action_registry.register("restaurant_search", reservation.search_restaurants, "Search restaurants")
    action_registry.register("restaurant_call", reservation.make_reservation_call, "Call to book")
    action_registry.register("send_sms", communication.send_sms, "Send SMS")
    action_registry.register("send_email", communication.send_email, "Send email")

    logger.info(f"Registered {len(action_registry.actions)} action plugins")
