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
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
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
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "X-API-Key", "Content-Type"],
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
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Dashboard not found. Place dashboard.html in the src/ folder.</h1>")


# ── Request / Response Models ────────────────
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


# ── Core ─────────────────────────────────────
@app.get("/health")
async def health_check():
    """Minimal health check — no sensitive details unless authenticated."""
    return {
        "status": "online",
        "agent_name": Config.AGENT_NAME,
        "is_awake": agent.is_awake,
        "timestamp": datetime.now().isoformat(),
    }

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