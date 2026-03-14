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
"""

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

from orchestrator import AgentOrchestrator, Config

app = FastAPI(
    title=f"{Config.AGENT_NAME} - Personal AI Agent",
    description="Private, secure AI agent with code generation, memory, and real-world actions",
    version="2.0.0",
)

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "*"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = AgentOrchestrator()


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
    timezone: Optional[str] = "Europe/Dublin"


# ── Core ─────────────────────────────────────
@app.get("/health")
async def health_check():
    ollama_ok = await agent.ollama.is_available()
    kb_stats = agent.knowledge.get_stats() if agent.knowledge else {"status": "disabled"}
    projects_count = len(agent.build_pipeline.list_projects()) if agent.build_pipeline else 0
    return {
        "status": "online",
        "agent_name": Config.AGENT_NAME,
        "is_awake": agent.is_awake,
        "services": {
            "ollama": "online" if ollama_ok else "offline",
            "claude": "configured" if Config.ANTHROPIC_API_KEY else "not configured",
            "codegen": "ready" if agent.build_pipeline else "not loaded",
            "knowledge_base": kb_stats,
        },
        "stats": {"projects_built": projects_count, "pending_tasks": len(agent.tasks.get_pending())},
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/process", response_model=ProcessResponse)
async def process_input(request: ProcessRequest):
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Empty input")
    response = await agent.process(request.input)
    return ProcessResponse(response=response, timestamp=datetime.now().isoformat())

@app.post("/wake")
async def wake_agent():
    return {"status": "awake", "message": await agent.wake_up()}

@app.post("/sleep")
async def sleep_agent():
    return {"status": "sleeping", "message": await agent.sleep()}


# ── Code Generation ──────────────────────────
@app.post("/build")
async def build_project(request: BuildRequest):
    if not agent.build_pipeline:
        raise HTTPException(status_code=503, detail="Code generation engine not loaded")
    return await agent.build_pipeline.build(request.prompt, request.deploy_target)

@app.post("/build/iterate")
async def iterate_project(request: IterateRequest):
    if not agent.build_pipeline:
        raise HTTPException(status_code=503, detail="Code generation engine not loaded")
    return await agent.build_pipeline.iterate(request.project_id, request.feedback, request.deploy_target)

@app.get("/projects")
async def list_projects():
    if not agent.build_pipeline:
        return {"projects": []}
    return {"projects": agent.build_pipeline.list_projects()}

@app.get("/projects/{project_id}")
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
        "directory": project.directory, "deployed_url": project.deployed_url,
        "build_log": project.build_log, "iteration": project.iteration, "created_at": project.created_at,
    }


# ── Knowledge Base ───────────────────────────
@app.post("/knowledge/store")
async def store_knowledge(request: KnowledgeStoreRequest):
    if not agent.knowledge:
        raise HTTPException(status_code=503, detail="Knowledge base not available")
    doc_id = agent.knowledge.store(request.content, request.metadata, request.category)
    return {"doc_id": doc_id, "status": "stored"}

@app.post("/knowledge/search")
async def search_knowledge(request: KnowledgeSearchRequest):
    if not agent.knowledge:
        raise HTTPException(status_code=503, detail="Knowledge base not available")
    results = agent.knowledge.search(request.query, request.n_results, request.category)
    return {"results": results, "count": len(results)}

@app.get("/knowledge/stats")
async def knowledge_stats():
    if not agent.knowledge:
        return {"status": "disabled"}
    return agent.knowledge.get_stats()


# ── DevOps ───────────────────────────────────
@app.post("/shell")
async def execute_shell(request: ShellRequest):
    try:
        from codegen import ShellExecutor
        return await ShellExecutor.execute(request.command, request.cwd)
    except ImportError:
        raise HTTPException(status_code=503, detail="Shell executor not available")


# ── Calendar ─────────────────────────────────
@app.post("/calendar/book")
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

@app.get("/calendar/events")
async def list_calendar_events():
    """List upcoming calendar events."""
    result = await agent.actions.execute("calendar_list", {"days": 7})
    return {"status": "ok", "result": result}


# ── Tasks ────────────────────────────────────
@app.get("/tasks")
async def get_tasks():
    pending = agent.tasks.get_pending()
    return {"tasks": [t.__dict__ for t in pending]}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True, log_level="info")