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
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, field_validator
from datetime import datetime
from typing import Optional

from orchestrator import AgentOrchestrator, Config

logger = logging.getLogger("server")

app = FastAPI(
    title=f"{Config.AGENT_NAME} - Personal AI Agent",
    description="Private, secure AI agent with code generation, memory, and real-world actions",
    version="2.1.0",
    docs_url=None,   # Disable Swagger UI in production
    redoc_url=None,   # Disable ReDoc in production
)

# ── API Key Authentication ─────────────────────
def _load_or_generate_api_key() -> str:
    """Load API key from env, or generate and save one on first run."""
    key = os.getenv("AGENT_API_KEY", "")
    if key:
        return key
    # Auto-generate a secure key and write it to .env so user can find it
    key = f"mel-{secrets.token_urlsafe(32)}"
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    try:
        with open(env_path, "a") as f:
            f.write(f"\n# Auto-generated API key for agent server\nAGENT_API_KEY={key}\n")
        logger.warning(f"Generated new API key. Saved to .env file. Key: {key}")
    except OSError:
        logger.warning(f"Generated API key (could not save to .env): {key}")
    return key

AGENT_API_KEY = _load_or_generate_api_key()

async def require_api_key(request: Request):
    """Dependency that enforces API key on protected endpoints."""
    auth = request.headers.get("Authorization", "")
    api_key = request.headers.get("X-API-Key", "")
    query_key = request.query_params.get("api_key", "")

    provided_key = ""
    if auth.startswith("Bearer "):
        provided_key = auth[7:]
    elif api_key:
        provided_key = api_key
    elif query_key:
        provided_key = query_key

    if not provided_key or not secrets.compare_digest(provided_key, AGENT_API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Use Authorization: Bearer <key>, X-API-Key header, or ?api_key= param.",
        )


# ── CORS — restricted to configured origins ────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").strip()
if ALLOWED_ORIGINS and ALLOWED_ORIGINS != "*":
    _origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
else:
    # Default: only localhost variants (safe for local Windows usage)
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
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "X-API-Key", "Content-Type"],
)

agent = AgentOrchestrator()

# Auto-wake on startup so Mel is always ready
@app.on_event("startup")
async def startup_wake():
    await agent.wake_up()
    logger.info("✅ Mel auto-woke on startup")

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
    """Serve the dashboard directly — open http://localhost:8000 in your browser."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Inject API key so dashboard can authenticate without user input
        content = content.replace(
            "const API = window.location.origin;",
            f"const API = window.location.origin;\nconst API_KEY = '{AGENT_API_KEY}';"
        )
        return HTMLResponse(content=content, headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        })
    return HTMLResponse(content="<h1>Dashboard not found. Place dashboard.html in the src/ folder.</h1>")


# ── Request / Response Models ────────────────
class TTSRequest(BaseModel):
    text: str

class ProcessRequest(BaseModel):
    input: str
    context: dict = {}

class ProcessResponse(BaseModel):
    response: str
    timestamp: str = ""

class BuildRequest(BaseModel):
    prompt: str
    deploy_target: str = "copy"

class IterateRequest(BaseModel):
    project_id: str
    feedback: str
    deploy_target: str = "copy"

class KnowledgeStoreRequest(BaseModel):
    content: str
    category: str = "general"
    metadata: dict = {}

class KnowledgeSearchRequest(BaseModel):
    query: str
    n_results: int = 5
    category: Optional[str] = None

class ShellRequest(BaseModel):
    command: str
    cwd: Optional[str] = None

class CalendarEventRequest(BaseModel):
    title: str
    start_time: str
    end_time: Optional[str] = None
    location: Optional[str] = ""
    description: Optional[str] = ""
    timezone: Optional[str] = "America/Chicago"


class ReminderRequest(BaseModel):
    title: str
    trigger_time: Optional[str] = None
    minutes: Optional[int] = None
    hours: Optional[int] = None
    days: Optional[int] = None
    recurrence: Optional[str] = None


class SmartHomeCommandRequest(BaseModel):
    action: str  # turn_on, turn_off, set_temperature, lock, unlock, brightness, scene, status
    device: Optional[str] = ""
    value: Optional[str] = ""


class NotificationRequest(BaseModel):
    title: str = "Mel Agent"
    message: str
    priority: str = "normal"
    backend: Optional[str] = None


class RoutineCreateRequest(BaseModel):
    name: str
    description: str = ""
    steps: list = []


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
    response = await agent.process(request.input)
    return ProcessResponse(response=response, timestamp=datetime.now().isoformat())

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

    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel

    if not api_key:
        raise HTTPException(status_code=503, detail="ElevenLabs not configured")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={"xi-api-key": api_key, "Content-Type": "application/json"},
                json={
                    "text": text,
                    "model_id": "eleven_turbo_v2_5",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.85, "style": 0.2, "use_speaker_boost": True}
                },
            )
            if resp.status_code == 200:
                return Response(content=resp.content, media_type="audio/mpeg")
            raise HTTPException(status_code=resp.status_code, detail=f"ElevenLabs error: {resp.text[:200]}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="TTS request timed out")

@app.get("/tts/config")
async def tts_config():
    """Tell the dashboard whether ElevenLabs is configured."""
    return {"elevenlabs": bool(os.getenv("ELEVENLABS_API_KEY", ""))}


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
async def spotify_callback(code: str = ""):
    """Spotify OAuth callback — exchanges code for token."""
    if not code:
        return HTMLResponse("<h2>No authorization code received</h2>")
    try:
        from music import SpotifyPlayer
        player = SpotifyPlayer()
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

@app.get("/spotify/playlists", dependencies=[Depends(require_api_key)])
async def spotify_playlists():
    if not agent.spotify:
        return {"playlists": []}
    return {"playlists": await agent.spotify.get_playlists()}


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


# ── Enhanced health check with new services ──
@app.get("/health")
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
        },
        "stats": {
            "projects_built": projects_count,
            "pending_tasks": len(agent.tasks.get_pending()),
            "pending_reminders": pending_reminders,
        },
        "timestamp": datetime.now().isoformat(),
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