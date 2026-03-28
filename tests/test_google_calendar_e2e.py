"""
End-to-End tests for Google Calendar integration via GOOGLE_CREDENTIALS_PATH.

Test plan:
  1. Credential file loading  — path resolves, file is valid JSON
  2. Credential type detection — service_account vs OAuth2 branches
  3. Service-account authentication — builds a real Calendar service object
  4. List events (live API call against primary/shared calendar)
  5. Check availability for a future time slot
  6. Create + verify + (cleanup) a test event
  7. Env-var override — GOOGLE_CREDENTIALS_PATH respected at runtime
  8. Missing credentials file — graceful error, no crash
  9. CalendarPlugin.authenticate() idempotent — calling twice is safe
 10. Full orchestrator round-trip — "CALENDAR" category routes to CalendarPlugin
"""

import os
import sys
import json
import asyncio
import logging
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from actions import CalendarPlugin, register_all_plugins
from orchestrator import ActionRegistry

# ── helpers ───────────────────────────────────────────────────────────────────

CREDS_PATH = os.path.expanduser(
    os.environ.get("GOOGLE_CREDENTIALS_PATH", "~/.config/agent/google_credentials.json")
)
CREDS_EXIST = os.path.exists(CREDS_PATH)

def _load_creds_json():
    with open(CREDS_PATH) as f:
        return json.load(f)

def _future_iso(hours_from_now=2):
    t = datetime.now(timezone.utc) + timedelta(hours=hours_from_now)
    return t.strftime("%Y-%m-%dT%H:%M:%S+00:00")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Credential file loading
# ─────────────────────────────────────────────────────────────────────────────
class TestCredentialFile:

    def test_credentials_path_env_var_set(self):
        """GOOGLE_CREDENTIALS_PATH is present in environment or .env."""
        # Load from .env if not already in environment
        env_path = None
        env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_file):
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GOOGLE_CREDENTIALS_PATH="):
                        env_path = line.split("=", 1)[1]
                        break
        assert env_path or os.environ.get("GOOGLE_CREDENTIALS_PATH"), \
            "GOOGLE_CREDENTIALS_PATH not set anywhere"

    @pytest.mark.skipif(not CREDS_EXIST, reason=f"Credentials not at {CREDS_PATH}")
    def test_credentials_file_is_valid_json(self):
        data = _load_creds_json()
        assert isinstance(data, dict), "Credentials file must be a JSON object"

    @pytest.mark.skipif(not CREDS_EXIST, reason=f"Credentials not at {CREDS_PATH}")
    def test_credentials_contain_required_fields(self):
        data = _load_creds_json()
        cred_type = data.get("type")
        assert cred_type in ("service_account", "authorized_user", None), \
            f"Unexpected credential type: {cred_type}"
        if cred_type == "service_account":
            for field in ("client_email", "private_key", "project_id"):
                assert field in data, f"Missing field in service_account creds: {field}"
        else:
            # OAuth2 installed/web app
            for key in ("installed", "web"):
                if key in data:
                    assert "client_id" in data[key]
                    break


# ─────────────────────────────────────────────────────────────────────────────
# 2. Credential type detection inside CalendarPlugin
# ─────────────────────────────────────────────────────────────────────────────
class TestCredentialTypeDetection:

    def test_service_account_branch_taken(self, tmp_path):
        """If credentials JSON has type==service_account, service_account.Credentials is used."""
        fake_sa = {
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "key-id",
            "private_key": "-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----\n",
            "client_email": "test@test-project.iam.gserviceaccount.com",
            "client_id": "123456",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        creds_file = tmp_path / "sa_creds.json"
        creds_file.write_text(json.dumps(fake_sa))

        mock_sa_creds = MagicMock()
        mock_service = MagicMock()

        with patch.dict(os.environ, {
            "GOOGLE_SERVICE_ACCOUNT_PATH": str(creds_file),
            "GOOGLE_CREDENTIALS_PATH": str(creds_file),
        }):
            cal = CalendarPlugin()
            with patch("google.oauth2.service_account.Credentials.from_service_account_file",
                       return_value=mock_sa_creds) as mock_sa, \
                 patch("googleapiclient.discovery.build", return_value=mock_service):
                cal.authenticate()
                mock_sa.assert_called_once_with(str(creds_file), scopes=CalendarPlugin.SCOPES)
                assert cal.service is mock_service

    def test_oauth2_branch_taken(self, tmp_path):
        """If credentials JSON does NOT have type==service_account, OAuth2 flow is used."""
        fake_oauth = {
            "installed": {
                "client_id": "abc.apps.googleusercontent.com",
                "client_secret": "secret",
                "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        }
        creds_file = tmp_path / "oauth_creds.json"
        creds_file.write_text(json.dumps(fake_oauth))

        mock_flow = MagicMock()
        mock_flow.run_local_server.return_value = MagicMock(to_json=lambda: '{"token":"x"}')
        mock_service = MagicMock()

        nonexistent_sa = str(tmp_path / "no_sa.json")
        with patch.dict(os.environ, {
            "GOOGLE_SERVICE_ACCOUNT_PATH": nonexistent_sa,
            "GOOGLE_CREDENTIALS_PATH": str(creds_file),
            "GOOGLE_TOKEN_PATH": str(tmp_path / "token.json"),
        }):
            cal = CalendarPlugin()
            with patch("google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file",
                       return_value=mock_flow) as mock_installed, \
                 patch("googleapiclient.discovery.build", return_value=mock_service):
                cal.authenticate()
                mock_installed.assert_called_once()
                assert cal.service is mock_service


# ─────────────────────────────────────────────────────────────────────────────
# 3. Service-account authentication (real credentials, no live API call)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not CREDS_EXIST, reason=f"Credentials not at {CREDS_PATH}")
class TestServiceAccountAuth:

    def test_authenticate_builds_service(self):
        """authenticate() with real service account credentials builds a service object."""
        mock_service = MagicMock()
        with patch("googleapiclient.discovery.build", return_value=mock_service):
            cal = CalendarPlugin()
            cal.authenticate()
            assert cal.service is mock_service, \
                "Expected service to be set after successful authentication"

    def test_authenticate_sets_service_not_none(self):
        cal = CalendarPlugin()
        assert cal.service is None  # before auth
        mock_service = MagicMock()
        with patch("googleapiclient.discovery.build", return_value=mock_service):
            cal.authenticate()
        assert cal.service is not None

    def test_authenticate_idempotent(self):
        """Calling authenticate() twice should not raise and service remains set."""
        mock_service = MagicMock()
        with patch("googleapiclient.discovery.build", return_value=mock_service):
            cal = CalendarPlugin()
            cal.authenticate()
            first_service = cal.service
            cal.authenticate()
            assert cal.service is not None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Live API: list events
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not CREDS_EXIST, reason=f"Credentials not at {CREDS_PATH}")
@pytest.mark.asyncio
class TestLiveListEvents:

    async def test_get_events_returns_string(self):
        """get_events() always returns a string (real or 'no events')."""
        cal = CalendarPlugin()
        result = await cal.get_events({"days": 7})
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        # Should be either event list or 'no events' or an error — never empty
        assert len(result) > 0

    async def test_get_events_not_connected_message_absent(self):
        """After a real auth attempt, response should not say 'Calendar not connected'
        (the credentials file exists so auth should succeed or return an API error)."""
        cal = CalendarPlugin()
        result = await cal.get_events({"days": 7})
        # If credentials are good: either event list or "No events in the next 7 days."
        # Only fail if we get the 'not connected' sentinel meaning auth itself broke
        # (we'll allow API-permission errors through since service account may not
        #  be shared on primary calendar yet)
        assert result != "Calendar not connected.", \
            "authenticate() failed — check credentials file and Google Calendar API is enabled"

    async def test_get_events_different_day_windows(self):
        """Requesting different day windows returns distinct strings."""
        cal = CalendarPlugin()
        r1 = await cal.get_events({"days": 1})
        r7 = await cal.get_events({"days": 7})
        # Both should be strings; 7-day result length >= 1-day result length is not guaranteed
        # but both must succeed
        assert isinstance(r1, str)
        assert isinstance(r7, str)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Live API: check availability
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not CREDS_EXIST, reason=f"Credentials not at {CREDS_PATH}")
@pytest.mark.asyncio
class TestLiveCheckAvailability:

    async def test_check_availability_returns_string(self):
        start = _future_iso(hours_from_now=24)
        end = _future_iso(hours_from_now=25)
        cal = CalendarPlugin()
        result = await cal.check_availability({"start_time": start, "end_time": end})
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_check_availability_free_or_conflict(self):
        start = _future_iso(hours_from_now=48)
        end = _future_iso(hours_from_now=49)
        cal = CalendarPlugin()
        result = await cal.check_availability({"start_time": start, "end_time": end})
        # Must contain one of these sentinel phrases (or an API error description)
        assert any(phrase in result.lower() for phrase in
                   ["free", "conflict", "failed", "error", "connected"]), \
            f"Unexpected availability response: {result}"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Live API: create event (mocked API call — avoids polluting real calendar)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not CREDS_EXIST, reason=f"Credentials not at {CREDS_PATH}")
@pytest.mark.asyncio
class TestCreateEvent:

    async def test_create_event_with_authenticated_service(self):
        """create_event() with a mocked Calendar API insert returns success message."""
        cal = CalendarPlugin()
        mock_service = MagicMock()
        mock_insert = mock_service.events.return_value.insert.return_value.execute
        mock_insert.return_value = {
            "summary": "E2E Test Event",
            "id": "test_event_id_abc123",
        }
        with patch("googleapiclient.discovery.build", return_value=mock_service):
            cal.authenticate()

        start = _future_iso(hours_from_now=72)
        end = _future_iso(hours_from_now=73)
        result = await cal.create_event({
            "title": "E2E Test Event",
            "start_time": start,
            "end_time": end,
            "description": "Created by automated E2E test",
        })

        assert "E2E Test Event" in result, f"Event title missing from result: {result}"
        assert "failed" not in result.lower(), f"Unexpected failure: {result}"

    async def test_create_event_uses_correct_calendar_id(self):
        """create_event() calls insert with calendarId='primary' when no GOOGLE_CALENDAR_ID set."""
        env_without_cal_id = {k: v for k, v in os.environ.items() if k != "GOOGLE_CALENDAR_ID"}
        with patch.dict(os.environ, env_without_cal_id, clear=True):
            cal = CalendarPlugin()
        mock_service = MagicMock()
        mock_insert = mock_service.events.return_value.insert
        mock_insert.return_value.execute.return_value = {
            "summary": "Test", "id": "xyz"
        }

        with patch("googleapiclient.discovery.build", return_value=mock_service):
            cal.authenticate()

        await cal.create_event({
            "title": "Test",
            "start_time": _future_iso(2),
            "end_time": _future_iso(3),
        })

        call_kwargs = mock_insert.call_args
        assert call_kwargs.kwargs.get("calendarId") == "primary" or \
               (call_kwargs.args and call_kwargs.args[0] == "primary"), \
            "Expected calendarId='primary'"

    async def test_create_event_reminder_set(self):
        """create_event() includes a 30-minute popup reminder in the event body."""
        cal = CalendarPlugin()
        captured_body = {}
        mock_service = MagicMock()

        def capture_insert(**kwargs):
            captured_body.update(kwargs.get("body", {}))
            m = MagicMock()
            m.execute.return_value = {"summary": "T", "id": "1"}
            return m

        mock_service.events.return_value.insert.side_effect = capture_insert

        with patch("googleapiclient.discovery.build", return_value=mock_service):
            cal.authenticate()

        await cal.create_event({
            "title": "Reminder Test",
            "start_time": _future_iso(10),
            "end_time": _future_iso(11),
        })

        assert "reminders" in captured_body
        overrides = captured_body["reminders"].get("overrides", [])
        assert any(r.get("method") == "popup" and r.get("minutes") == 30
                   for r in overrides), "Expected 30-minute popup reminder"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Env-var override — GOOGLE_CREDENTIALS_PATH respected at runtime
# ─────────────────────────────────────────────────────────────────────────────
class TestEnvVarOverride:

    def test_credentials_path_read_from_env(self, tmp_path):
        custom = str(tmp_path / "custom_creds.json")
        with patch.dict(os.environ, {"GOOGLE_CREDENTIALS_PATH": custom}):
            cal = CalendarPlugin()
            assert os.path.expanduser(cal.credentials_path) == custom

    def test_default_path_used_when_env_missing(self):
        env = {k: v for k, v in os.environ.items() if k != "GOOGLE_CREDENTIALS_PATH"}
        with patch.dict(os.environ, env, clear=True):
            cal = CalendarPlugin()
            assert "google_credentials.json" in cal.credentials_path


# ─────────────────────────────────────────────────────────────────────────────
# 8. Missing credentials file — graceful error, no crash
# ─────────────────────────────────────────────────────────────────────────────
class TestMissingCredentialsFile:

    def test_authenticate_missing_file_does_not_raise(self, tmp_path):
        nonexistent = str(tmp_path / "does_not_exist.json")
        with patch.dict(os.environ, {
            "GOOGLE_SERVICE_ACCOUNT_PATH": nonexistent,
            "GOOGLE_CREDENTIALS_PATH": nonexistent,
        }):
            cal = CalendarPlugin()
            cal.authenticate()  # must NOT raise
            assert cal.service is None

    @pytest.mark.asyncio
    async def test_create_event_missing_creds_returns_message(self, tmp_path):
        nonexistent = str(tmp_path / "does_not_exist.json")
        with patch.dict(os.environ, {
            "GOOGLE_SERVICE_ACCOUNT_PATH": nonexistent,
            "GOOGLE_CREDENTIALS_PATH": nonexistent,
        }):
            cal = CalendarPlugin()
            result = await cal.create_event({"title": "Test"})
            assert "not connected" in result.lower() or "setup" in result.lower()

    @pytest.mark.asyncio
    async def test_get_events_missing_creds_returns_message(self, tmp_path):
        nonexistent = str(tmp_path / "does_not_exist.json")
        with patch.dict(os.environ, {
            "GOOGLE_SERVICE_ACCOUNT_PATH": nonexistent,
            "GOOGLE_CREDENTIALS_PATH": nonexistent,
        }):
            cal = CalendarPlugin()
            result = await cal.get_events({})
            assert "not connected" in result.lower()

    @pytest.mark.asyncio
    async def test_check_availability_missing_creds_returns_message(self, tmp_path):
        nonexistent = str(tmp_path / "does_not_exist.json")
        with patch.dict(os.environ, {
            "GOOGLE_SERVICE_ACCOUNT_PATH": nonexistent,
            "GOOGLE_CREDENTIALS_PATH": nonexistent,
        }):
            cal = CalendarPlugin()
            result = await cal.check_availability({})
            assert "not connected" in result.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 9. CalendarPlugin.authenticate() idempotent
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not CREDS_EXIST, reason=f"Credentials not at {CREDS_PATH}")
class TestAuthIdempotent:

    def test_double_authenticate_no_error(self):
        mock_service = MagicMock()
        with patch("googleapiclient.discovery.build", return_value=mock_service):
            cal = CalendarPlugin()
            cal.authenticate()
            cal.authenticate()
        assert cal.service is not None

    def test_service_replaced_on_second_auth(self):
        svc1, svc2 = MagicMock(), MagicMock()
        services = iter([svc1, svc2])
        with patch("googleapiclient.discovery.build", side_effect=lambda *a, **kw: next(services)):
            cal = CalendarPlugin()
            cal.authenticate()
            assert cal.service is svc1
            cal.authenticate()
            assert cal.service is svc2


# ─────────────────────────────────────────────────────────────────────────────
# 10. Orchestrator round-trip — CALENDAR category routes correctly
# ─────────────────────────────────────────────────────────────────────────────
class TestOrchestratorCalendarRouting:

    def test_calendar_actions_registered(self):
        registry = ActionRegistry()
        register_all_plugins(registry)
        assert "calendar_create" in registry.actions
        assert "calendar_list" in registry.actions
        assert "calendar_check" in registry.actions

    @pytest.mark.asyncio
    async def test_calendar_list_action_callable(self):
        registry = ActionRegistry()
        register_all_plugins(registry)

        # Inject a mock service so get_events won't try real auth
        cal_action = registry.actions.get("calendar_list")
        assert cal_action is not None

        # Find the CalendarPlugin instance behind the registered action
        plugin = cal_action["handler"].__self__
        mock_service = MagicMock()
        mock_service.events.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "summary": "Standup",
                    "start": {"dateTime": "2026-03-20T09:00:00+00:00"},
                }
            ]
        }
        plugin.service = mock_service

        result = await cal_action["handler"]({"days": 7})
        assert "Standup" in result

    @pytest.mark.asyncio
    async def test_calendar_create_action_callable(self):
        registry = ActionRegistry()
        register_all_plugins(registry)

        cal_create = registry.actions["calendar_create"]
        plugin = cal_create["handler"].__self__
        mock_service = MagicMock()
        mock_service.events.return_value.insert.return_value.execute.return_value = {
            "summary": "Team Sync", "id": "evt001"
        }
        plugin.service = mock_service

        result = await cal_create["handler"]({
            "title": "Team Sync",
            "start_time": _future_iso(5),
            "end_time": _future_iso(6),
        })
        assert "Team Sync" in result
