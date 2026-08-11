import os
import sys
import shutil
import importlib

# Auto-clear Python bytecode cache on startup to prevent stale code loading
_src_dir = os.path.dirname(os.path.abspath(__file__))
for _root, _dirs, _files in os.walk(_src_dir):
    if "__pycache__" in _dirs:
        shutil.rmtree(os.path.join(_root, "__pycache__"), ignore_errors=True)
        _dirs.remove("__pycache__")

# Force fresh imports (no cached .pyc)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

print("=" * 60)
print(f">>> MEL SERVER — loading from {_src_dir}")
print("=" * 60)
"""
AGENT API SERVER v2.0
======================
FastAPI server exposing the full agent capabilities:
- Voice command processing (from Raspberry Pi)
- Code generation & deployment
- Knowledge base (memory)
- Project management
- DevOps sandboxed commands
- System health monitoring

Security: All endpoints (except / and /health) require API key authentication.
Set AGENT_API_KEY in your .env file. A random key is generated on first run if not set.
"""

import os
import re
import secrets
import logging
import uvicorn
from dotenv import load_dotenv
load_dotenv(override=True)
from contextlib import asynccontextmanager
import json as _json
import asyncio as _asyncio
from fastapi import FastAPI, HTTPException, Depends, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional

from orchestrator import AgentOrchestrator, Config

logger = logging.getLogger("server")

# ── WebSocket Connection Manager ──────────────
class ConnectionManager:
    """Tracks active WebSocket connections for real-time push updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, event_type: str, data: dict):
        """Send an event to all connected clients."""
        message = _json.dumps({"event": event_type, **data})
        for conn in self.active_connections[:]:
            try:
                await conn.send_text(message)
            except Exception:
                self.active_connections.remove(conn)

ws_manager = ConnectionManager()


# ── Lifespan (startup / shutdown) ──────────────
# Defined before app so it can be passed to FastAPI constructor.
# References `agent` by name — resolved at runtime after module load.
@asynccontextmanager
async def lifespan(app: FastAPI):
    await agent.wake_up()
    logger.info("✅ Mel auto-woke on startup")
    # Launch background tasks
    _asyncio.create_task(_scheduler_loop())
    _asyncio.create_task(_health_broadcast_loop())
    _asyncio.create_task(_calendar_alert_loop())
    logger.info("✅ Background tasks started (scheduler, health broadcast, calendar alerts)")
    yield


async def _scheduler_loop():
    """Check reminders every 15 seconds and push via WebSocket + phone notifications."""
    while True:
        try:
            if agent.scheduler:
                triggered = agent.scheduler.check_and_trigger()
                if _asyncio.iscoroutine(triggered):
                    triggered = await triggered
                if triggered:
                    for reminder in triggered:
                        title = reminder.title if hasattr(reminder, 'title') else str(reminder)
                        rid = reminder.id if hasattr(reminder, 'id') else ''
                        await ws_manager.broadcast("reminder", {"title": title, "id": rid})
                        # Send push notification to phone
                        if agent.notifications:
                            try:
                                await agent.notifications.send_reminder_notification(title)
                            except Exception as e:
                                logger.error(f"Reminder notification failed: {e}")
        except Exception as e:
            logger.error(f"Scheduler loop error: {e}")
        await _asyncio.sleep(15)


async def _health_broadcast_loop():
    """Broadcast health status to WebSocket clients every 30 seconds."""
    while True:
        try:
            if ws_manager.active_connections:
                ollama_ok = await agent.ollama.is_available()
                await ws_manager.broadcast("health", {
                    "is_awake": agent.is_awake,
                    "ollama": "online" if ollama_ok else "offline",
                })
        except Exception:
            pass
        await _asyncio.sleep(30)


# Track which events we've already alerted for (prevent duplicate alerts)
_alerted_events: set = set()


async def _calendar_alert_loop():
    """Check upcoming calendar events every 5 minutes. Alert 30 min before."""
    await _asyncio.sleep(30)  # Wait for services to initialize
    while True:
        try:
            # Only run if calendar actions are registered and agent has actions
            if agent.actions and "calendar_list" in agent.actions.actions:
                from datetime import timedelta
                result = await agent.actions.execute("calendar_list", {"count": 10})
                if result and not result.startswith(("No upcoming", "Calendar not", "Failed")):
                    # Parse events from result text
                    import re
                    now = datetime.now()
                    alert_window_start = now + timedelta(minutes=25)
                    alert_window_end = now + timedelta(minutes=35)

                    # Try to extract event times and titles
                    for line in result.split("\n"):
                        # Look for time patterns like "10:30 AM" or ISO timestamps
                        time_match = re.search(r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))', line)
                        if time_match:
                            try:
                                event_time_str = time_match.group(1).strip().upper()
                                event_time = datetime.strptime(
                                    f"{now.strftime('%Y-%m-%d')} {event_time_str}",
                                    "%Y-%m-%d %I:%M %p"
                                )
                                # Check if event is within 25-35 min from now
                                if alert_window_start <= event_time <= alert_window_end:
                                    # Extract title (text before or after the time)
                                    title = re.sub(r'\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)', '', line).strip()
                                    title = re.sub(r'^[\s\-•|:]+|[\s\-•|:]+$', '', title)
                                    event_key = f"{now.strftime('%Y-%m-%d')}_{event_time_str}_{title[:30]}"
                                    if event_key not in _alerted_events and title:
                                        _alerted_events.add(event_key)
                                        mins_away = int((event_time - now).total_seconds() / 60)
                                        alert_msg = f"Upcoming in ~{mins_away} min: {title}"
                                        # Push via WebSocket
                                        await ws_manager.broadcast("reminder", {
                                            "title": alert_msg,
                                            "id": f"cal_alert_{event_key}",
                                        })
                                        # Push via phone notification
                                        if agent.notifications and agent.notifications.is_configured():
                                            try:
                                                await agent.notifications.send(
                                                    title="Calendar Alert",
                                                    message=alert_msg,
                                                    priority="high",
                                                )
                                            except Exception as e:
                                                logger.error(f"Calendar alert notification failed: {e}")
                                        logger.info(f"Calendar alert sent: {alert_msg}")
                            except (ValueError, TypeError):
                                continue
            # Clean up old alerted events daily
            if len(_alerted_events) > 200:
                _alerted_events.clear()
        except Exception as e:
            logger.error(f"Calendar alert loop error: {e}")
        await _asyncio.sleep(300)  # Check every 5 minutes


app = FastAPI(
    title=f"{Config.AGENT_NAME} - Personal AI Agent",
    description="Private, secure AI agent with code generation, memory, and real-world actions",
    version="2.1.0",
    docs_url=None,    # Disable Swagger UI in production
    redoc_url=None,   # Disable ReDoc in production
    lifespan=lifespan,
)

# ── API Key Authentication ─────────────────────
def _load_or_generate_api_key() -> str:
    """Load API key from encrypted vault, then env, or generate on first run."""
    # 1. Try encrypted vault first
    try:
        from security import SecretVault
        vault = SecretVault()
        key = vault.get("AGENT_API_KEY", "")
        if key:
            logger.info("API key loaded from encrypted vault")
            return key
    except Exception as e:
        logger.warning(f"Could not read from vault: {e}")

    # 2. Fall back to environment variable (legacy / unencrypted)
    key = os.getenv("AGENT_API_KEY", "")
    if key:
        return key

    # 3. Generate a new key, save to vault
    key = f"mel-{secrets.token_urlsafe(32)}"
    try:
        from security import SecretVault
        vault = SecretVault()
        vault.set("AGENT_API_KEY", key)
        logger.warning(f"Generated new API key and saved to encrypted vault. Key: {key}")
    except Exception:
        # Last resort: write to .env only (vault unavailable)
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        try:
            with open(env_path, "a", encoding="utf-8") as f:
                f.write(f"\n# Auto-generated API key\nAGENT_API_KEY={key}\n")
            logger.warning(f"Generated new API key. Saved to .env. Key: {key}")
        except OSError:
            logger.warning(f"Generated API key (could not persist): {key}")
    return key

AGENT_API_KEY = _load_or_generate_api_key()

# ── Session management (HTTP-only cookies for dashboard) ───────────────────
import time as _time
_sessions: dict[str, float] = {}   # token → expiry epoch
_SESSION_TTL = 8 * 3600            # 8 hours

def _create_session() -> str:
    """Generate a one-time-per-page-load session token for the dashboard."""
    # Prune expired sessions to prevent unbounded growth
    now = _time.time()
    expired = [t for t, exp in list(_sessions.items()) if exp < now]
    for t in expired:
        del _sessions[t]
    token = secrets.token_urlsafe(32)
    _sessions[token] = now + _SESSION_TTL
    return token

async def require_api_key(request: Request):
    """
    Enforce authentication on protected endpoints.
    Accepts:
      • Authorization: Bearer <key>   (programmatic / Raspberry Pi clients)
      • X-API-Key: <key>              (programmatic / Raspberry Pi clients)
      • X-Session-Token: <token>      (dashboard — short-lived, injected per page load)
      • mel_session cookie            (dashboard — HTTP-only, defence-in-depth)
    Query-parameter auth is intentionally NOT supported (logs keys in plaintext).
    """
    auth = request.headers.get("Authorization", "")
    header_key = request.headers.get("X-API-Key", "")

    # API key via header
    provided_key = ""
    if auth.startswith("Bearer "):
        provided_key = auth[7:]
    elif header_key:
        provided_key = header_key

    if provided_key and secrets.compare_digest(provided_key, AGENT_API_KEY):
        return  # ✓ valid API key

    # Session token via header (dashboard JS — explicit, browser-compatible)
    session_header = request.headers.get("X-Session-Token", "")
    if session_header and session_header in _sessions:
        if _sessions[session_header] > _time.time():
            return  # ✓ valid session token (header)
        del _sessions[session_header]

    # Session token via cookie (defence-in-depth fallback)
    cookie_token = request.cookies.get("mel_session", "")
    if cookie_token and cookie_token in _sessions:
        if _sessions[cookie_token] > _time.time():
            return  # ✓ valid session token (cookie)
        del _sessions[cookie_token]

    raise HTTPException(
        status_code=401,
        detail="Authentication required. Use Authorization: Bearer <key> or X-API-Key header.",
    )


# ── CORS — restricted to configured origins ────
# Subnet wildcards (e.g. 192.168.1.0/24) are NOT valid CORS origins and are
# silently dropped. List individual IPs if cross-LAN access is needed.
_SAFE_ORIGIN_RE = re.compile(r'^https?://(localhost|127\.0\.0\.1|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(:\d+)?$')

def _parse_origins(raw: str) -> list[str]:
    result = []
    for o in raw.split(","):
        o = o.strip()
        if _SAFE_ORIGIN_RE.match(o):
            result.append(o)
        elif o:
            logger.warning(f"CORS: ignoring unsafe/invalid origin '{o}' (subnet wildcards not allowed)")
    return result

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").strip()
if ALLOWED_ORIGINS and ALLOWED_ORIGINS != "*":
    _origins = _parse_origins(ALLOWED_ORIGINS) or ["http://localhost:8000", "http://127.0.0.1:8000"]
else:
    _origins = [
        "http://localhost:8000",
        "http://localhost:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "X-API-Key", "Content-Type", "X-Session-Token"],
)

agent = AgentOrchestrator()

# ── Rate Limiting ──────────────────────────────
try:
    from security import RateLimiter
    _rate_limiter = RateLimiter()
except ImportError:
    _rate_limiter = None

# Map endpoints to rate limit services
_RATE_LIMIT_MAP = {
    "/process": "claude_api",
    "/build": "code_gen",
    "/build/iterate": "code_gen",
    "/shell": "shell_exec",
    "/health": "public_health",
    "/tts": "tts",
}

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if _rate_limiter:
        service = _RATE_LIMIT_MAP.get(request.url.path)
        if service and not _rate_limiter.check(service):
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )
    return await call_next(request)


# ── Dashboard served from server (no CORS issues) ──
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the dashboard with a short-lived session token.

    Security model:
      • A per-page-load session token (NOT the master API key) is injected
        into the page as SESSION_TOKEN.  It expires in 8 hours.
      • The same token is also set as an HTTP-only cookie (belt + suspenders).
      • The master AGENT_API_KEY never appears in the HTML/JavaScript.
    """
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            content = f.read()
        session_token = _create_session()
        # Inject session token as JS variable — short-lived, NOT the master key
        content = content.replace(
            "const API = window.location.origin;",
            f"const API = window.location.origin;\nconst SESSION_TOKEN = '{session_token}';"
        )
        html_response = HTMLResponse(content=content, headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        })
        # Also set as HTTP-only cookie for defence-in-depth
        html_response.set_cookie(
            key="mel_session",
            value=session_token,
            httponly=True,
            samesite="strict",
            max_age=_SESSION_TTL,
            path="/",
        )
        return html_response
    return HTMLResponse(content="<h1>Dashboard not found. Place dashboard.html in the src/ folder.</h1>")


# ── Request / Response Models ────────────────
class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

class ProcessRequest(BaseModel):
    input: str = Field(..., min_length=1, max_length=5000)
    context: dict = Field(default_factory=dict)

class ProcessResponse(BaseModel):
    response: str
    timestamp: str = ""

class BuildRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    deploy_target: str = Field(default="copy", max_length=50)

class IterateRequest(BaseModel):
    project_id: str = Field(..., min_length=1, max_length=100)
    feedback: str = Field(..., min_length=1, max_length=4000)
    deploy_target: str = Field(default="copy", max_length=50)

class KnowledgeStoreRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)
    category: str = Field(default="general", max_length=100)
    metadata: dict = Field(default_factory=dict)

class KnowledgeSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    n_results: int = Field(default=5, ge=1, le=20)
    category: Optional[str] = Field(default=None, max_length=100)

class ShellRequest(BaseModel):
    command: str = Field(..., min_length=1, max_length=500)
    cwd: Optional[str] = Field(default=None, max_length=260)

    @field_validator("cwd")
    @classmethod
    def no_path_traversal(cls, v):
        if v and ".." in v:
            raise ValueError("Path traversal not allowed in cwd")
        return v

class CalendarEventRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    start_time: str = Field(..., max_length=50)
    end_time: Optional[str] = Field(default=None, max_length=50)
    location: Optional[str] = Field(default="", max_length=300)
    description: Optional[str] = Field(default="", max_length=2000)
    timezone: Optional[str] = Field(default="America/Chicago", max_length=50)


class ReminderRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    trigger_time: Optional[str] = Field(default=None, max_length=50)
    minutes: Optional[int] = Field(default=None, ge=0, le=10080)   # max 1 week
    hours: Optional[int] = Field(default=None, ge=0, le=168)
    days: Optional[int] = Field(default=None, ge=0, le=365)
    recurrence: Optional[str] = Field(default=None, max_length=50)


class SmartHomeCommandRequest(BaseModel):
    action: str = Field(..., min_length=1, max_length=50)
    device: Optional[str] = Field(default="", max_length=100)
    value: Optional[str] = Field(default="", max_length=100)


class NotificationRequest(BaseModel):
    title: str = Field(default="Mel Agent", max_length=100)
    message: str = Field(..., min_length=1, max_length=1000)
    priority: str = Field(default="normal", max_length=20)
    backend: Optional[str] = Field(default=None, max_length=50)


class RoutineCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="", max_length=500)
    steps: list = Field(default_factory=list)


class UIUXSearchRequest(BaseModel):
    query: str
    domain: Optional[str] = None
    max_results: int = 3

    @field_validator("max_results")
    @classmethod
    def cap_max_results(cls, v):
        return min(v, 10)


class UIUXDesignSystemRequest(BaseModel):
    query: str
    project_name: Optional[str] = None


# ── Core ─────────────────────────────────────

@app.get("/health/detail", dependencies=[Depends(require_api_key)])
async def health_detail():
    """Detailed health — requires authentication."""
    ollama_ok = await agent.ollama.is_available()
    kb_entries = agent.knowledge.get_stats().get("entries", 0) if agent.knowledge else 0
    projects_count = len(agent.build_pipeline.list_projects()) if agent.build_pipeline else 0
    return {
        "status": "online",
        "agent_name": Config.AGENT_NAME,
        "is_awake": agent.is_awake,
        "services": {
            "ollama": "online" if ollama_ok else "offline",
            "claude": "configured" if Config.ANTHROPIC_API_KEY else "not configured",
            "codegen": "ready" if agent.build_pipeline else "not loaded",
            "knowledge_base_entries": kb_entries,
        },
        "stats": {"projects_built": projects_count, "pending_tasks": len(agent.tasks.get_pending())},
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/process", response_model=ProcessResponse, dependencies=[Depends(require_api_key)])
async def process_input(request: ProcessRequest):
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Empty input")
    if len(request.input) > 5000:
        raise HTTPException(status_code=400, detail="Input too long (max 5000 chars)")
    await ws_manager.broadcast("status", {"status": "thinking"})
    try:
        response = await agent.process(request.input)
    finally:
        await ws_manager.broadcast("status", {"status": "idle"})
    return ProcessResponse(response=response, timestamp=datetime.now().isoformat())


@app.post("/process/stream", dependencies=[Depends(require_api_key)])
async def process_input_stream(request: ProcessRequest):
    """Stream agent response token-by-token via Server-Sent Events."""
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Empty input")
    if len(request.input) > 5000:
        raise HTTPException(status_code=400, detail="Input too long (max 5000 chars)")

    async def event_generator():
        yield f"event: status\ndata: {_json.dumps({'status': 'thinking'})}\n\n"
        await ws_manager.broadcast("status", {"status": "thinking"})
        try:
            async for token in agent.process_stream(request.input):
                yield f"event: token\ndata: {_json.dumps({'token': token})}\n\n"
            yield f"event: done\ndata: {_json.dumps({'status': 'complete'})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"event: error\ndata: {_json.dumps({'error': str(e)})}\n\n"
        finally:
            await ws_manager.broadcast("status", {"status": "idle"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Persistent WebSocket for real-time agent events (status, reminders, health)."""
    # Authenticate via short-lived session token: ws://host/ws?token=<session-token>
    # The master API key is intentionally NOT accepted here (or via any query
    # parameter) — query strings are commonly captured in server/proxy access
    # logs and browser history, which is exactly the leak vector
    # require_api_key's query-param policy is designed to avoid.
    token = websocket.query_params.get("token", "")
    expiry = _sessions.get(token)
    valid_session = expiry is not None and expiry > _time.time()
    if not valid_session:
        await websocket.close(code=4001, reason="Invalid token")
        return

    await ws_manager.connect(websocket)
    logger.info(f"WebSocket connected ({len(ws_manager.active_connections)} clients)")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = _json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_text(_json.dumps({"event": "pong"}))
            except _json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info(f"WebSocket disconnected ({len(ws_manager.active_connections)} clients)")


@app.post("/wake", dependencies=[Depends(require_api_key)])
async def wake_agent():
    return {"status": "awake", "message": await agent.wake_up()}

@app.post("/sleep", dependencies=[Depends(require_api_key)])
async def sleep_agent():
    return {"status": "sleeping", "message": await agent.sleep()}

@app.post("/tts", dependencies=[Depends(require_api_key)])
async def text_to_speech(request: TTSRequest):
    """Convert text to speech via ElevenLabs. Returns audio/mpeg."""
    import httpx
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")
    if len(text) > 2000:
        text = text[:2000]

    try:
        from security import SecretVault
        _vault = SecretVault()
        api_key = _vault.get("ELEVENLABS_API_KEY") or os.getenv("ELEVENLABS_API_KEY", "")
        voice_id = _vault.get("ELEVENLABS_VOICE_ID") or os.getenv("ELEVENLABS_VOICE_ID", "eXpIbVcVbLo8ZJQDlDnl")
    except Exception:
        api_key = os.getenv("ELEVENLABS_API_KEY", "")
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "eXpIbVcVbLo8ZJQDlDnl")

    if not api_key:
        raise HTTPException(status_code=503, detail="ElevenLabs not configured")

    # Strip markdown so it doesn't get read aloud ("asterisk asterisk bold asterisk asterisk")
    import re as _re
    text = _re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)   # bold/italic
    text = _re.sub(r'`{1,3}[^`]*`{1,3}', '', text)         # inline code / code blocks
    text = _re.sub(r'^#{1,6}\s+', '', text, flags=_re.MULTILINE)  # headers
    text = _re.sub(r'^\s*[-*•]\s+', '', text, flags=_re.MULTILINE)  # bullet points
    text = _re.sub(r'\n{3,}', '\n\n', text).strip()

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={"xi-api-key": api_key, "Content-Type": "application/json"},
                json={
                    "text": text,
                    "model_id": "eleven_flash_v2_5",  # newest model — most natural, lowest latency
                    "voice_settings": {
                        "stability": 0.40,          # balanced: not flat, not erratic
                        "similarity_boost": 0.85,   # strong character presence
                        "style": 0.65,              # expressive & warm — sounds happy/engaged
                        "use_speaker_boost": True   # sharper, more present sound
                    }
                },
            )
            if resp.status_code == 200:
                return Response(content=resp.content, media_type="audio/mpeg")
            raise HTTPException(status_code=resp.status_code, detail=f"ElevenLabs error: {resp.text[:200]}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="TTS request timed out")

@app.get("/tts/config", dependencies=[Depends(require_api_key)])
async def tts_config():
    """Tell the dashboard whether ElevenLabs is configured."""
    try:
        from security import SecretVault
        key = SecretVault().get("ELEVENLABS_API_KEY") or os.getenv("ELEVENLABS_API_KEY", "")
    except Exception:
        key = os.getenv("ELEVENLABS_API_KEY", "")
    return {"elevenlabs": bool(key)}


class VoiceEventRequest(BaseModel):
    event: str = Field(..., max_length=30)   # wake | listening_start | listening_stop | speaking_start | speaking_stop
    mode: Optional[str] = Field(default=None, max_length=20)  # command | homecoming

_VOICE_EVENTS = {"wake", "listening_start", "listening_stop", "speaking_start", "speaking_stop"}

@app.post("/voice/event", dependencies=[Depends(require_api_key)])
async def voice_event(request: VoiceEventRequest):
    """Receive real-time voice-pipeline state from wake_listener.py (Pi/Mac) and
    rebroadcast to all connected dashboard WebSocket clients."""
    if request.event not in _VOICE_EVENTS:
        raise HTTPException(status_code=400, detail="Unknown voice event")
    await ws_manager.broadcast("voice", {"state": request.event, "mode": request.mode or ""})
    return {"status": "ok"}


# ── Code Generation ──────────────────────────
@app.post("/build", dependencies=[Depends(require_api_key)])
async def build_project(request: BuildRequest):
    if not agent.build_pipeline:
        raise HTTPException(status_code=503, detail="Code generation engine not loaded")
    return await agent.build_pipeline.build(request.prompt, request.deploy_target)

@app.post("/build/iterate", dependencies=[Depends(require_api_key)])
async def iterate_project(request: IterateRequest):
    if not agent.build_pipeline:
        raise HTTPException(status_code=503, detail="Code generation engine not loaded")
    return await agent.build_pipeline.iterate(request.project_id, request.feedback, request.deploy_target)

@app.get("/projects", dependencies=[Depends(require_api_key)])
async def list_projects():
    if not agent.build_pipeline:
        return {"projects": []}
    return {"projects": agent.build_pipeline.list_projects()}

@app.get("/projects/{project_id}", dependencies=[Depends(require_api_key)])
async def get_project(project_id: str):
    if not agent.build_pipeline:
        raise HTTPException(status_code=503, detail="Code generation engine not loaded")
    project = agent.build_pipeline.manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return {
        "id": project.id, "name": project.name, "type": project.project_type.value,
        "description": project.description, "status": project.status,
        "files": [{"path": f.path, "language": f.language, "description": f.description} for f in project.files],
        "iteration": project.iteration, "created_at": project.created_at,
    }


# ── Knowledge Base ───────────────────────────
@app.post("/knowledge/store", dependencies=[Depends(require_api_key)])
async def store_knowledge(request: KnowledgeStoreRequest):
    if not agent.knowledge:
        raise HTTPException(status_code=503, detail="Knowledge base not available")
    doc_id = agent.knowledge.store(request.content, request.metadata, request.category)
    return {"doc_id": doc_id, "status": "stored"}

@app.post("/knowledge/search", dependencies=[Depends(require_api_key)])
async def search_knowledge(request: KnowledgeSearchRequest):
    if not agent.knowledge:
        raise HTTPException(status_code=503, detail="Knowledge base not available")
    if request.n_results > 20:
        request.n_results = 20  # Cap search results
    results = agent.knowledge.search(request.query, request.n_results, request.category)
    return {"results": results, "count": len(results)}

@app.get("/knowledge/stats", dependencies=[Depends(require_api_key)])
async def knowledge_stats():
    if not agent.knowledge:
        return {"status": "disabled"}
    stats = agent.knowledge.get_stats()
    stats.pop("persist_dir", None)  # Don't expose filesystem paths
    return stats


# ── DevOps ───────────────────────────────────
@app.post("/shell", dependencies=[Depends(require_api_key)])
async def execute_shell(request: ShellRequest):
    try:
        from codegen import ShellExecutor
        result = await ShellExecutor.execute(request.command, request.cwd)
        # Strip filesystem paths from error output
        if result.get("stderr"):
            result["stderr"] = _strip_paths(result["stderr"])
        return result
    except ImportError:
        raise HTTPException(status_code=503, detail="Shell executor not available")


# ── Calendar ─────────────────────────────────
@app.post("/calendar/book", dependencies=[Depends(require_api_key)])
async def book_calendar_event(request: CalendarEventRequest):
    """Book a calendar event via Google Calendar API."""
    from datetime import timedelta

    end_time = request.end_time
    if not end_time:
        # Default to 30 min event
        start_dt = datetime.fromisoformat(request.start_time)
        end_time = (start_dt + timedelta(minutes=30)).isoformat()

    params = {
        "title": request.title,
        "start_time": request.start_time,
        "end_time": end_time,
        "location": request.location,
        "description": request.description,
        "timezone": request.timezone,
    }

    result = await agent.actions.execute("calendar_create", params)
    return {"status": "ok", "result": result}

@app.get("/calendar/events", dependencies=[Depends(require_api_key)])
async def list_calendar_events():
    """List upcoming calendar events."""
    result = await agent.actions.execute("calendar_list", {"days": 7})
    return {"status": "ok", "result": result}


# ── Tasks ────────────────────────────────────
@app.get("/tasks", dependencies=[Depends(require_api_key)])
async def get_tasks():
    pending = agent.tasks.get_pending()
    # Strip raw_input from task responses to prevent data leakage
    safe_tasks = []
    for t in pending:
        td = t.__dict__.copy()
        td.pop("raw_input", None)
        safe_tasks.append(td)
    return {"tasks": safe_tasks}


# ── Spotify / Music ──────────────────────────
@app.get("/spotify/status")
async def spotify_status():
    """Check Spotify configuration and auth status."""
    try:
        from music import SpotifyPlayer
        player = SpotifyPlayer()
        return {
            "configured": player.auth.is_configured(),
            "authenticated": player.auth.is_authenticated,
        }
    except ImportError:
        return {"configured": False, "authenticated": False}

@app.get("/spotify/auth")
async def spotify_auth():
    """Start Spotify OAuth flow — redirect user to Spotify login."""
    try:
        from music import SpotifyPlayer
        player = SpotifyPlayer()
        if not player.auth.is_configured():
            return HTMLResponse("<h2>Spotify not configured. Add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to .env</h2>")
        auth_url = player.auth.get_auth_url()
        return HTMLResponse(f'<html><body style="background:#050508;color:#e0e0f0;font-family:sans-serif;display:flex;justify-content:center;align-items:center;height:100vh;">'
                          f'<a href="{auth_url}" style="padding:16px 32px;background:linear-gradient(135deg,#1DB954,#1ed760);color:#000;text-decoration:none;border-radius:40px;font-size:1.2rem;font-weight:700;">Connect Spotify</a>'
                          f'</body></html>')
    except ImportError:
        return HTMLResponse("<h2>Music module not available</h2>")

@app.get("/spotify/callback")
async def spotify_callback(code: str = "", state: str = ""):
    """Spotify OAuth callback — exchanges code for token."""
    if not code:
        return HTMLResponse("<h2>No authorization code received</h2>")
    try:
        from music import SpotifyPlayer
        player = SpotifyPlayer()
        if not player.auth.verify_state(state):
            return HTMLResponse('<html><body style="background:#050508;color:#e04060;font-family:sans-serif;text-align:center;padding:60px;">'
                              '<h2>Invalid or expired login request</h2><p>Please restart the Spotify connection from /spotify/auth.</p></body></html>')
        success = await player.auth.exchange_code(code)
        if success:
            return HTMLResponse('<html><body style="background:#050508;color:#40e080;font-family:sans-serif;display:flex;justify-content:center;align-items:center;height:100vh;font-size:1.5rem;">'
                              'Spotify connected! You can close this tab.</body></html>')
        return HTMLResponse('<html><body style="background:#050508;color:#e04060;font-family:sans-serif;text-align:center;padding:60px;">'
                          '<h2>Authentication failed</h2><p>Check your Spotify credentials in .env</p></body></html>')
    except ImportError:
        return HTMLResponse("<h2>Music module not available</h2>")

@app.get("/spotify/now-playing", dependencies=[Depends(require_api_key)])
async def spotify_now_playing():
    """Get currently playing track."""
    if agent.spotify:
        return await agent.spotify.now_playing()
    return {"is_playing": False, "track": None}

@app.post("/spotify/play", dependencies=[Depends(require_api_key)])
async def spotify_play(query: Optional[str] = None):
    """Play music or search and play a track."""
    if not agent.spotify:
        raise HTTPException(status_code=503, detail="Spotify not available")
    if query:
        results = await agent.spotify.search(query, "track", 1)
        if results:
            return {"status": "ok", "message": await agent.spotify.play(uri=results[0]["uri"])}
        return {"status": "error", "message": f"No results for '{query}'"}
    return {"status": "ok", "message": await agent.spotify.play()}

@app.post("/spotify/pause", dependencies=[Depends(require_api_key)])
async def spotify_pause():
    if not agent.spotify:
        raise HTTPException(status_code=503, detail="Spotify not available")
    return {"status": "ok", "message": await agent.spotify.pause()}

@app.post("/spotify/next", dependencies=[Depends(require_api_key)])
async def spotify_next():
    if not agent.spotify:
        raise HTTPException(status_code=503, detail="Spotify not available")
    return {"status": "ok", "message": await agent.spotify.next_track()}

@app.post("/spotify/previous", dependencies=[Depends(require_api_key)])
async def spotify_previous():
    if not agent.spotify:
        raise HTTPException(status_code=503, detail="Spotify not available")
    return {"status": "ok", "message": await agent.spotify.previous_track()}

@app.get("/spotify/playlists", dependencies=[Depends(require_api_key)])
async def spotify_playlists():
    if not agent.spotify:
        return {"playlists": []}
    return {"playlists": await agent.spotify.get_playlists()}

@app.get("/spotify/token", dependencies=[Depends(require_api_key)])
async def spotify_token():
    """Return the current access token for the Web Playback SDK."""
    if not agent.spotify:
        raise HTTPException(status_code=503, detail="Spotify not available")
    token = await agent.spotify.auth.get_access_token()
    if not token:
        raise HTTPException(status_code=401, detail="Spotify not authenticated")
    return {"access_token": token}

@app.get("/spotify/devices", dependencies=[Depends(require_api_key)])
async def spotify_devices():
    """List available Spotify devices."""
    if not agent.spotify:
        return {"devices": []}
    return {"devices": await agent.spotify.get_devices()}

@app.put("/spotify/transfer", dependencies=[Depends(require_api_key)])
async def spotify_transfer(device_id: str = ""):
    """Transfer playback to a specific device."""
    if not agent.spotify:
        raise HTTPException(status_code=503, detail="Spotify not available")
    result = await agent.spotify._api("PUT", "/me/player", json_body={"device_ids": [device_id], "play": True})
    if result is not None:
        return {"status": "ok", "message": "Playback transferred"}
    return {"status": "error", "message": "Could not transfer playback"}

@app.put("/spotify/shuffle", dependencies=[Depends(require_api_key)])
async def spotify_shuffle(state: bool = True):
    if not agent.spotify:
        raise HTTPException(status_code=503, detail="Spotify not available")
    return {"status": "ok", "message": await agent.spotify.shuffle(state)}

@app.put("/spotify/repeat", dependencies=[Depends(require_api_key)])
async def spotify_repeat(state: str = "context"):
    if not agent.spotify:
        raise HTTPException(status_code=503, detail="Spotify not available")
    if state not in ("track", "context", "off"):
        raise HTTPException(status_code=400, detail="state must be track, context, or off")
    return {"status": "ok", "message": await agent.spotify.repeat(state)}

@app.put("/spotify/volume", dependencies=[Depends(require_api_key)])
async def spotify_volume(volume: int = 50):
    if not agent.spotify:
        raise HTTPException(status_code=503, detail="Spotify not available")
    return {"status": "ok", "message": await agent.spotify.set_volume(volume)}

@app.put("/spotify/seek", dependencies=[Depends(require_api_key)])
async def spotify_seek(position_ms: int = 0):
    if not agent.spotify:
        raise HTTPException(status_code=503, detail="Spotify not available")
    return {"status": "ok", "message": await agent.spotify.seek(position_ms)}

@app.post("/spotify/register-device", dependencies=[Depends(require_api_key)])
async def spotify_register_device(device_id: str = ""):
    """Register the dashboard web player device ID so the backend prefers it."""
    if agent.spotify and device_id:
        agent.spotify.dashboard_device_id = device_id
        logger.info(f"Dashboard Spotify device registered: {device_id}")
        return {"status": "ok"}
    return {"status": "error"}


# ── Reminders / Scheduler ────────────────────
@app.post("/reminders", dependencies=[Depends(require_api_key)])
async def create_reminder(request: ReminderRequest):
    """Create a reminder."""
    if not agent.scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    if request.trigger_time:
        reminder = agent.scheduler.create_reminder(
            title=request.title,
            trigger_time=request.trigger_time,
            recurrence=request.recurrence,
        )
    else:
        reminder = agent.scheduler.create_reminder_relative(
            title=request.title,
            minutes=request.minutes or 0,
            hours=request.hours or 0,
            days=request.days or 0,
        )
    from dataclasses import asdict
    return {"status": "ok", "reminder": asdict(reminder)}

@app.get("/reminders", dependencies=[Depends(require_api_key)])
async def list_reminders():
    """List all reminders."""
    if not agent.scheduler:
        return {"reminders": []}
    return {"reminders": agent.scheduler.get_all()}

@app.get("/reminders/pending", dependencies=[Depends(require_api_key)])
async def pending_reminders():
    """List pending reminders."""
    if not agent.scheduler:
        return {"reminders": []}
    return {"reminders": agent.scheduler.get_pending()}

@app.delete("/reminders/{reminder_id}", dependencies=[Depends(require_api_key)])
async def cancel_reminder(reminder_id: str):
    if not agent.scheduler:
        raise HTTPException(status_code=503, detail="Scheduler not available")
    success = agent.scheduler.cancel_reminder(reminder_id)
    if not success:
        raise HTTPException(status_code=404, detail="Reminder not found")
    return {"status": "cancelled"}

@app.get("/reminders/check", dependencies=[Depends(require_api_key)])
async def check_reminders():
    """Check for due reminders and trigger them."""
    if not agent.scheduler:
        return {"triggered": []}
    triggered = await agent.scheduler.check_and_trigger()
    from dataclasses import asdict
    return {"triggered": [asdict(r) for r in triggered]}


# ── Smart Home ───────────────────────────────
@app.get("/smarthome/devices", dependencies=[Depends(require_api_key)])
async def smarthome_devices():
    """Get all smart home devices."""
    if not agent.smarthome:
        raise HTTPException(status_code=503, detail="Smart home not available")
    devices = await agent.smarthome.get_all_devices()
    return {"devices": devices}

@app.get("/smarthome/status", dependencies=[Depends(require_api_key)])
async def smarthome_status():
    """Get smart home status summary."""
    if not agent.smarthome:
        return {"status": "not configured", "summary": "Smart home module not connected."}
    summary = await agent.smarthome.get_status_summary()
    configured = agent.smarthome.ha.is_configured()
    return {"status": "connected" if configured else "simulated", "summary": summary}

@app.post("/smarthome/command", dependencies=[Depends(require_api_key)])
async def smarthome_command(request: SmartHomeCommandRequest):
    """Execute a smart home command."""
    if not agent.smarthome:
        raise HTTPException(status_code=503, detail="Smart home not available")

    action = request.action
    device = request.device
    value = request.value

    if action == "turn_on":
        result = await agent.smarthome.turn_on(device)
    elif action == "turn_off":
        result = await agent.smarthome.turn_off(device)
    elif action == "set_temperature":
        result = await agent.smarthome.set_temperature(float(value))
    elif action == "lock":
        result = await agent.smarthome.lock_door(device or "front_door")
    elif action == "unlock":
        result = await agent.smarthome.unlock_door(device or "front_door")
    elif action == "brightness":
        result = await agent.smarthome.set_brightness(device, int(value or 100))
    elif action == "scene":
        result = await agent.smarthome.activate_scene(value or device)
    elif action == "status":
        result = await agent.smarthome.get_status_summary()
    else:
        result = f"Unknown action: {action}"

    return {"status": "ok", "result": result}


# ── Weather ──────────────────────────────────
@app.get("/weather/current", dependencies=[Depends(require_api_key)])
async def weather_current(city: Optional[str] = None):
    """Get current weather."""
    if not agent.weather:
        raise HTTPException(status_code=503, detail="Weather service not configured")
    result = await agent.weather.get_current(city)
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result

@app.get("/weather/forecast", dependencies=[Depends(require_api_key)])
async def weather_forecast(city: Optional[str] = None, days: int = 3):
    """Get weather forecast."""
    if not agent.weather:
        raise HTTPException(status_code=503, detail="Weather service not configured")
    if days > 5:
        days = 5
    result = await agent.weather.get_forecast(city, days)
    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])
    return result

@app.get("/weather/summary", dependencies=[Depends(require_api_key)])
async def weather_summary(city: Optional[str] = None):
    """Get natural language weather summary."""
    if not agent.weather:
        raise HTTPException(status_code=503, detail="Weather service not configured")
    return {"summary": await agent.weather.get_summary(city)}


# ── Routines ────────────────────────────────
@app.get("/routines", dependencies=[Depends(require_api_key)])
async def list_routines():
    """List all available routines."""
    if not agent.routines:
        return {"routines": []}
    return {"routines": agent.routines.list_routines()}

@app.post("/routines/{routine_id}/run", dependencies=[Depends(require_api_key)])
async def run_routine(routine_id: str):
    """Execute a routine."""
    if not agent.routines:
        raise HTTPException(status_code=503, detail="Routines engine not available")
    result = await agent.routines.execute(routine_id, agent.actions)
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    return result

@app.post("/routines/create", dependencies=[Depends(require_api_key)])
async def create_routine(request: RoutineCreateRequest):
    """Create a custom routine."""
    if not agent.routines:
        raise HTTPException(status_code=503, detail="Routines engine not available")
    routine = agent.routines.create_custom(request.name, request.description, request.steps)
    from dataclasses import asdict
    return {"status": "created", "routine": asdict(routine)}

@app.delete("/routines/{routine_id}", dependencies=[Depends(require_api_key)])
async def delete_routine(routine_id: str):
    """Delete a custom routine."""
    if not agent.routines:
        raise HTTPException(status_code=503, detail="Routines engine not available")
    success = agent.routines.delete_custom(routine_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot delete built-in routines or routine not found")
    return {"status": "deleted"}


# ── Notifications ───────────────────────────
@app.post("/notifications/send", dependencies=[Depends(require_api_key)])
async def send_notification(request: NotificationRequest):
    """Send a push notification."""
    if not agent.notifications:
        raise HTTPException(status_code=503, detail="Notification service not configured")
    result = await agent.notifications.send(
        title=request.title,
        message=request.message,
        priority=request.priority,
        backend=request.backend,
    )
    return result

@app.get("/notifications/status", dependencies=[Depends(require_api_key)])
async def notification_status():
    """Get notification service status."""
    if not agent.notifications:
        return {"configured": False, "backends": []}
    return {
        "configured": agent.notifications.is_configured(),
        "backends": agent.notifications.get_configured_backends(),
    }

@app.get("/notifications/history", dependencies=[Depends(require_api_key)])
async def notification_history():
    """Get recent notification history."""
    if not agent.notifications:
        return {"notifications": []}
    return {"notifications": agent.notifications.get_recent()}


# ── UI/UX Design Intelligence ─────────────────
@app.post("/design/search", dependencies=[Depends(require_api_key)])
async def design_search(request: UIUXSearchRequest):
    """Search UI/UX design knowledge base (styles, colors, charts, typography, etc.)."""
    result = await agent.actions.execute("ui_ux_search", {
        "query": request.query,
        "domain": request.domain,
        "max_results": request.max_results,
    })
    return {"status": "ok", "result": result}


@app.post("/design/system", dependencies=[Depends(require_api_key)])
async def design_system(request: UIUXDesignSystemRequest):
    """Generate a complete design system recommendation."""
    result = await agent.actions.execute("ui_ux_design_system", {
        "query": request.query,
        "project_name": request.project_name,
    })
    return {"status": "ok", "result": result}


@app.get("/design/domains", dependencies=[Depends(require_api_key)])
async def design_domains():
    """List available UI/UX search domains and stacks."""
    result = await agent.actions.execute("ui_ux_domains", {})
    return {"status": "ok", "result": result}


# ── Enhanced health check with new services ──
@app.get("/health", dependencies=[Depends(require_api_key)])
async def health_check_v2():
    """Health check — includes all service statuses for dashboard."""
    ollama_ok = await agent.ollama.is_available()
    kb_entries = agent.knowledge.get_stats().get("entries", 0) if agent.knowledge else 0
    projects_count = len(agent.build_pipeline.list_projects()) if agent.build_pipeline else 0

    # Check for due reminders while we're at it
    if agent.scheduler:
        try:
            await agent.scheduler.check_and_trigger()
        except Exception:
            pass

    spotify_status = "not configured"
    if agent.spotify:
        if agent.spotify.auth.is_authenticated:
            spotify_status = "connected"
        elif agent.spotify.auth.is_configured():
            spotify_status = "not authenticated"

    smarthome_status = "not configured"
    if agent.smarthome:
        smarthome_status = "connected" if agent.smarthome.ha.is_configured() else "simulated"

    weather_status = "not configured"
    if agent.weather and agent.weather.is_configured():
        weather_status = "configured"

    routines_count = len(agent.routines.list_routines()) if agent.routines else 0

    notification_backends = agent.notifications.get_configured_backends() if agent.notifications else []

    pending_reminders = len(agent.scheduler.get_pending()) if agent.scheduler else 0

    return {
        "status": "online",
        "agent_name": Config.AGENT_NAME,
        "is_awake": agent.is_awake,
        "services": {
            "ollama": "online" if ollama_ok else "offline",
            "claude": "configured" if Config.ANTHROPIC_API_KEY else "not configured",
            "codegen": "ready" if agent.build_pipeline else "not loaded",
            "knowledge_base": {"entries": kb_entries},
            "spotify": spotify_status,
            "smarthome": smarthome_status,
            "weather": weather_status,
            "notifications": notification_backends if notification_backends else "not configured",
            "routines": routines_count,
            "ui_ux": "ready" if agent.ui_ux else "not loaded",
        },
        "stats": {
            "projects_built": projects_count,
            "pending_tasks": len(agent.tasks.get_pending()),
            "pending_reminders": pending_reminders,
        },
        "timestamp": datetime.now().isoformat(),
    }


# ── Homecoming ───────────────────────────────
class HomecomeingResponse(BaseModel):
    speech: str
    weather: dict = {}
    calendar: list = []
    news_url: str = ""
    stocks_url: str = ""


@app.post("/homecoming", dependencies=[Depends(require_api_key)])
async def homecoming():
    """Full homecoming sequence: greet, weather with advice, calendar, open browser tabs.
    Called by the Raspberry Pi (or keyboard mode) when 'wake up daddy's home' is detected.
    Browser opens happen server-side (Windows machine).
    """
    import subprocess
    import platform
    from weather import WeatherService

    user_name = Config.USER_NAME
    now = datetime.now()
    hour = now.hour

    # 1. Time-appropriate greeting
    if 5 <= hour < 12:
        greeting = f"Good morning {user_name},"
    elif 12 <= hour < 17:
        greeting = f"Good afternoon {user_name},"
    elif 17 <= hour < 21:
        greeting = f"Good evening {user_name}, welcome home."
    else:
        greeting = f"Hey {user_name}, burning the midnight oil I see."

    speech_parts = [greeting]
    weather_data = {}

    # 2. Weather with smart advice
    if agent.weather and agent.weather.is_configured():
        try:
            current = await agent.weather.get_current()
            if "error" not in current:
                weather_data = current
                unit = current["unit_symbol"]
                temp = current["temperature"]
                desc = current["description"]
                speech_parts.append(
                    f"Outside it's {temp}{unit} and {desc.lower()}."
                )
                # Smart contextual advice
                advice = WeatherService.get_weather_advice(temp, desc, agent.weather.units)
                if advice:
                    speech_parts.append(advice)
        except Exception as e:
            logger.warning(f"Homecoming weather fetch failed: {e}")

    # 3. Calendar briefing
    calendar_events = []
    if agent.actions and "calendar_list" in agent.actions.actions:
        try:
            result = await agent.actions.execute("calendar_list", {"days": 1})
            if result and not result.startswith(("No upcoming", "Calendar not", "Failed")):
                speech_parts.append(f"Here's your schedule for today. {result}")
                # Try to parse events for the response payload
                calendar_events = [{"summary": result}]
            else:
                speech_parts.append("You have a clear schedule today.")
        except Exception as e:
            logger.warning(f"Homecoming calendar fetch failed: {e}")

    # 4. Personal context note from about-me/priorities.md
    try:
        import journal as _journal_mod
        priorities_path = _journal_mod.ABOUT_ME_DIR / "priorities.md"
        if priorities_path.exists():
            content = priorities_path.read_text(encoding="utf-8").strip()
            # Only include if user has filled it in (placeholder lines have "[")
            if content and content.count("[") < 3:
                speech_parts.append("Based on your notes, here's what's on your plate this week.")
    except Exception:
        pass

    # 5. Open browser tabs (server-side, Windows)
    news_url = os.getenv("NEWS_URL", "https://news.google.com")
    stocks_url = os.getenv("STOCKS_URL", "https://finance.yahoo.com")

    def _open_url(url: str):
        """Open URL in default browser (cross-platform)."""
        try:
            system = platform.system()
            if system == "Windows":
                subprocess.Popen(["cmd", "/c", "start", "", url], shell=False)
            elif system == "Darwin":
                subprocess.Popen(["open", url])
            else:
                subprocess.Popen(["xdg-open", url])
        except Exception as ex:
            logger.warning(f"Could not open browser for {url}: {ex}")

    _open_url(news_url)
    _open_url(stocks_url)

    speech_parts.append("I've opened today's news and markets for you.")
    speech_parts.append("What can I do for you?")

    full_speech = " ".join(speech_parts)

    return {
        "speech": full_speech,
        "weather": weather_data,
        "calendar": calendar_events,
        "news_url": news_url,
        "stocks_url": stocks_url,
    }


# ── Web Search ───────────────────────────────
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=20)
    news: bool = Field(default=False)


@app.post("/search", dependencies=[Depends(require_api_key)])
async def web_search(request: SearchRequest):
    """Search the web via DuckDuckGo. Free, no API key required."""
    try:
        from search import WebSearch
    except ImportError:
        raise HTTPException(status_code=503, detail="Search module not available")

    if request.news:
        results = await WebSearch.news_search(request.query, max_results=request.max_results)
    else:
        results = await WebSearch.search(request.query, max_results=request.max_results)

    return {"query": request.query, "results": results, "count": len(results)}


# ── About-Me Folder ───────────────────────────
class NoteRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    file: str = Field(default="notes.md", max_length=50)


@app.get("/about-me", dependencies=[Depends(require_api_key)])
async def get_about_me():
    """Return all about-me context files as JSON."""
    try:
        from journal import AboutMe
        return {
            "context": AboutMe.load_all(),
            "files": AboutMe.list_files(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/about-me/note", dependencies=[Depends(require_api_key)])
async def add_about_me_note(request: NoteRequest):
    """Append a timestamped note to a file in data/about-me/."""
    try:
        from journal import AboutMe
        ok = AboutMe.append_note(request.text, request.file)
        if ok:
            return {"status": "ok", "message": f"Note saved to {request.file}"}
        raise HTTPException(status_code=500, detail="Could not save note")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Pi Diagnostics ───────────────────────────
@app.get("/health/pi")
async def health_pi(request: Request, api_key: Optional[str] = None):
    """Raspberry Pi connectivity diagnostic. Tests API key, server reachability, and services.
    Can be called without auth to check if server is reachable; auth check is explicit.
    Usage from Pi: curl http://SERVER_IP:8000/health/pi -H 'X-API-Key: your-key'
    """
    # Validate API key if provided
    provided_key = api_key or request.headers.get("x-api-key") or request.headers.get("authorization", "").replace("Bearer ", "")
    auth_ok = bool(provided_key) and secrets.compare_digest(provided_key, AGENT_API_KEY)

    server_ip = request.headers.get("host", "unknown")

    return {
        "status": "reachable",
        "server": server_ip,
        "api_key_valid": auth_ok,
        "api_key_hint": "Provide X-API-Key or Authorization: Bearer <key>" if not auth_ok else "✓ Valid",
        "agent_name": Config.AGENT_NAME,
        "services": {
            "weather": "configured" if (agent.weather and agent.weather.is_configured()) else "not configured",
            "calendar": "ready" if (agent.actions and "calendar_list" in agent.actions.actions) else "not configured",
            "spotify": "connected" if (agent.spotify and hasattr(agent.spotify, 'auth') and agent.spotify.auth.is_authenticated) else "not connected",
        },
        "timestamp": datetime.now().isoformat(),
        "next_steps": [] if auth_ok else [
            "Set ORCHESTRATOR_URL=http://" + server_ip + " in Pi .env",
            "Set AGENT_API_KEY to match server .env",
            "Set PICOVOICE_ACCESS_KEY (free at console.picovoice.ai)",
        ],
    }


# ── Global error handler — never leak internals ──
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


def _strip_paths(text: str) -> str:
    """Remove filesystem paths from error output."""
    return re.sub(r'(/home/\S+|/Users/\S+|C:\\\\Users\\\\\S+|[A-Z]:\\\\\S+)', '[PATH]', text)


if __name__ == "__main__":
    host = os.getenv("AGENT_HOST", "127.0.0.1")  # Localhost only by default — safe on Windows
    port = int(os.getenv("AGENT_PORT", "8000"))
    uvicorn.run("server:app", host=host, port=port, reload=True, log_level="info")