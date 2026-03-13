"""
MEL AI AGENT - Core Orchestrator
==================================
Routes tasks between local Ollama (private) and Claude API (heavy reasoning).
All PII stays local. Only sanitized requests go to Claude.
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("orchestrator")


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
class Config:
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5-20250514")
    WAKE_PHRASE = os.getenv("WAKE_PHRASE", "wake up daddy is home")
    AGENT_NAME = os.getenv("AGENT_NAME", "Mel")


# ─────────────────────────────────────────────
# Task Classification
# ─────────────────────────────────────────────
class TaskType(Enum):
    LOCAL = "local"       # Handled entirely by Ollama (private data)
    CLOUD = "cloud"       # Requires Claude API (complex reasoning)
    ACTION = "action"     # Requires external tool execution
    HYBRID = "hybrid"     # Local processing + Claude for planning


@dataclass
class Task:
    id: str
    intent: str
    raw_input: str
    task_type: TaskType
    status: str = "pending"
    result: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# PII Sanitizer - Keeps your data safe
# ─────────────────────────────────────────────
class PIISanitizer:
    """Strips personal info before sending to Claude API."""

    def __init__(self):
        # Load your personal identifiers from local config
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "pii_mappings.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                self.mappings = json.load(f)
        else:
            self.mappings = {}

    def sanitize(self, text: str) -> str:
        """Replace PII with placeholders before sending to cloud."""
        sanitized = text
        for real_value, placeholder in self.mappings.items():
            sanitized = sanitized.replace(real_value, placeholder)
        return sanitized

    def desanitize(self, text: str) -> str:
        """Restore PII from placeholders in cloud response."""
        restored = text
        for real_value, placeholder in self.mappings.items():
            restored = restored.replace(placeholder, real_value)
        return restored


# ─────────────────────────────────────────────
# Intent Classifier (runs locally via Ollama)
# ─────────────────────────────────────────────
class IntentClassifier:
    """Uses local Ollama to classify intents - no data leaves your machine."""

    CLASSIFICATION_PROMPT_TEMPLATE = """You are an intent classifier for a personal AI assistant.
Classify the user's request into exactly ONE of these categories:

- CALENDAR: Anything about scheduling, events, meetings, reminders
- RESERVATION: Booking restaurants, hotels, appointments
- INFORMATION: General questions, research, recommendations
- COMMUNICATION: Sending messages, emails, making calls
- HOME: Smart home controls, device management
- PERSONAL: Questions about personal data, preferences, history
- SYSTEM: Agent management, settings, status checks
- CODE: Build/create/generate apps, websites, scripts, interfaces, dashboards, components, landing pages, any coding task
- DEVOPS: Server management, deployments, git operations, system monitoring, docker, process management
- KNOWLEDGE: Remember something, recall past conversations, "what did I say about...", "do you remember..."

Respond with ONLY a JSON object with these keys: category, intent, requires_cloud (boolean), summary.
Example: {"category": "CALENDAR", "intent": "book_event", "requires_cloud": false, "summary": "User wants to schedule a meeting"}

User request: """

    async def classify(self, user_input: str) -> dict:
        """Classify intent. Keywords first (instant), Ollama for ambiguous."""
        # Step 1: Try keyword match first — instant, no API call
        kw_result = self._keyword_classify(user_input)
        if kw_result["category"] != "INFORMATION":
            # Keywords matched a specific category — use it immediately
            logger.info(f"Keyword classified: {kw_result['category']}")
            return kw_result

        # Step 2: Ambiguous request — ask Ollama for deeper classification
        prompt = self.CLASSIFICATION_PROMPT_TEMPLATE + user_input
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{Config.OLLAMA_URL}/api/generate",
                    json={
                        "model": Config.OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                    },
                )
                result = response.json()
                raw = result.get("response", "")
                logger.info(f"Ollama raw: {raw[:200]}")

                parsed = None
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    import re
                    match = re.search(r'\{[^{}]+\}', raw)
                    if match:
                        try:
                            parsed = json.loads(match.group())
                        except json.JSONDecodeError:
                            pass

                valid_categories = ["CALENDAR", "RESERVATION", "INFORMATION", "COMMUNICATION",
                                    "HOME", "PERSONAL", "SYSTEM", "CODE", "DEVOPS", "KNOWLEDGE"]
                if parsed and isinstance(parsed, dict) and parsed.get("category", "").upper() in valid_categories:
                    parsed["category"] = parsed["category"].upper()
                    return parsed

                return kw_result

        except Exception as e:
            logger.error(f"Ollama classification failed: {e}")
            return kw_result

    def _keyword_classify(self, user_input: str) -> dict:
        """Fallback keyword-based classifier when Ollama fails."""
        text = user_input.lower()

        if any(kw in text for kw in ["remember", "save this", "note that", "keep in mind", "don't forget"]):
            return {"category": "KNOWLEDGE", "intent": "store", "requires_cloud": False, "summary": user_input}
        elif any(kw in text for kw in ["recall", "what did i", "do you remember", "what is my", "what's my"]):
            return {"category": "KNOWLEDGE", "intent": "recall", "requires_cloud": False, "summary": user_input}
        elif any(kw in text for kw in ["build", "create", "generate", "make me a", "code", "website", "app", "script", "dashboard", "landing page"]):
            return {"category": "CODE", "intent": "generate", "requires_cloud": True, "summary": user_input}
        elif any(kw in text for kw in ["schedule", "calendar", "meeting", "event", "remind"]):
            return {"category": "CALENDAR", "intent": "manage", "requires_cloud": False, "summary": user_input}
        elif any(kw in text for kw in ["book", "reservation", "table", "hotel"]):
            return {"category": "RESERVATION", "intent": "book", "requires_cloud": True, "summary": user_input}
        elif any(kw in text for kw in ["send", "text", "email", "message", "call"]):
            return {"category": "COMMUNICATION", "intent": "send", "requires_cloud": False, "summary": user_input}
        elif any(kw in text for kw in ["light", "lock", "thermostat", "temperature", "turn on", "turn off"]):
            return {"category": "HOME", "intent": "control", "requires_cloud": False, "summary": user_input}
        elif any(kw in text for kw in ["status", "system", "setting", "shut down", "sleep", "what can you"]):
            return {"category": "SYSTEM", "intent": "status", "requires_cloud": False, "summary": user_input}
        elif any(kw in text for kw in ["deploy", "git", "server", "docker", "process", "restart"]):
            return {"category": "DEVOPS", "intent": "manage", "requires_cloud": False, "summary": user_input}
        elif any(kw in text for kw in ["my name", "my address", "my phone", "about me", "personal"]):
            return {"category": "PERSONAL", "intent": "query", "requires_cloud": False, "summary": user_input}
        else:
            return {"category": "INFORMATION", "intent": "unknown", "requires_cloud": True, "summary": user_input}


# ─────────────────────────────────────────────
# Ollama Client (Local - Private)
# ─────────────────────────────────────────────
class OllamaClient:
    """Handles all local inference. Your data never leaves this machine."""

    def __init__(self):
        self.base_url = Config.OLLAMA_URL
        self.model = Config.OLLAMA_MODEL
        self.conversation_history = []

    async def chat(self, message: str, system_prompt: str = None) -> str:
        """Send a message to local Ollama model."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.conversation_history[-10:])  # Keep last 10 turns
        messages.append({"role": "user", "content": message})

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={"model": self.model, "messages": messages, "stream": False},
                )
                result = response.json()
                assistant_msg = result["message"]["content"]

                # Update local history
                self.conversation_history.append({"role": "user", "content": message})
                self.conversation_history.append({"role": "assistant", "content": assistant_msg})

                return assistant_msg
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return f"Local model error: {e}"

    async def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False


# ─────────────────────────────────────────────
# Claude Client (Cloud - Heavy Reasoning)
# ─────────────────────────────────────────────
class ClaudeClient:
    """Handles complex reasoning via Claude API. Only sanitized data sent."""

    def __init__(self):
        self.api_key = Config.ANTHROPIC_API_KEY
        self.model = Config.CLAUDE_MODEL
        self.sanitizer = PIISanitizer()
        # Initialize audit logger
        try:
            from security import AuditLogger, EnhancedPIISanitizer
            self.audit = AuditLogger()
            self.enhanced_sanitizer = EnhancedPIISanitizer()
        except ImportError:
            self.audit = None
            self.enhanced_sanitizer = None

    async def reason(self, task_description: str, context: str = "") -> str:
        """Send sanitized request to Claude for complex reasoning."""
        if not self.api_key:
            return "Claude API key not configured. Using local model only."

        # SECURITY: Sanitize before sending (both layers)
        safe_description = self.sanitizer.sanitize(task_description)
        safe_context = self.sanitizer.sanitize(context)
        if self.enhanced_sanitizer:
            safe_description = self.enhanced_sanitizer.sanitize(safe_description)
            safe_context = self.enhanced_sanitizer.sanitize(safe_context)

        # AUDIT: Log what's being sent to the cloud
        if self.audit:
            self.audit.log_api_call(
                service="claude",
                endpoint="https://api.anthropic.com/v1/messages",
                data_sent_preview=f"Task: {safe_description[:200]}"
            )

        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": self.model,
            "max_tokens": 2048,
            "system": f"You are a task planner for a personal AI agent named {Config.AGENT_NAME}. "
                      "Generate actionable step-by-step plans. Be specific and concise. "
                      "Never ask for personal information - work with what's provided.",
            "messages": [
                {"role": "user", "content": f"Task: {safe_description}\nContext: {safe_context}"}
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                )
                result = response.json()
                raw_response = result["content"][0]["text"]
                # SECURITY: Restore PII only in local response
                return self.sanitizer.desanitize(raw_response)
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return f"Cloud reasoning error: {e}"


# ─────────────────────────────────────────────
# Action Registry - Tools your agent can use
# ─────────────────────────────────────────────
class ActionRegistry:
    """Registry of actions the agent can perform."""

    def __init__(self):
        self.actions = {}

    def register(self, name: str, handler, description: str = ""):
        """Register an action handler."""
        self.actions[name] = {"handler": handler, "description": description}
        logger.info(f"Registered action: {name}")

    async def execute(self, action_name: str, params: dict) -> str:
        """Execute a registered action."""
        if action_name not in self.actions:
            return f"Unknown action: {action_name}"
        try:
            result = await self.actions[action_name]["handler"](params)
            return result
        except Exception as e:
            logger.error(f"Action {action_name} failed: {e}")
            return f"Action failed: {e}"


# ─────────────────────────────────────────────
# Task Manager - Tracks all pending work
# ─────────────────────────────────────────────
class TaskManager:
    """Manages task queue with persistence."""

    def __init__(self, db_path: str = "tasks.json"):
        self.db_path = db_path
        self.tasks: list[Task] = []
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path) as f:
                data = json.load(f)
                self.tasks = [Task(**t) for t in data]

    def _save(self):
        with open(self.db_path, "w") as f:
            json.dump([t.__dict__ for t in self.tasks], f, indent=2, default=str)

    def add(self, task: Task):
        self.tasks.append(task)
        self._save()

    def update(self, task_id: str, **kwargs):
        for t in self.tasks:
            if t.id == task_id:
                for k, v in kwargs.items():
                    setattr(t, k, v)
                self._save()
                return

    def get_pending(self) -> list[Task]:
        return [t for t in self.tasks if t.status == "pending"]


# ─────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────
class AgentOrchestrator:
    """The brain of your personal AI agent."""

    def __init__(self):
        self.ollama = OllamaClient()
        self.claude = ClaudeClient()
        self.classifier = IntentClassifier()
        self.actions = ActionRegistry()
        self.tasks = TaskManager()
        self.is_awake = False

        # New capabilities
        self.build_pipeline = None
        self.knowledge = None
        self._init_extensions()

    def _init_extensions(self):
        """Initialize optional extensions (codegen, knowledge base)."""
        try:
            from codegen import BuildPipeline
            self.build_pipeline = BuildPipeline()
            logger.info("✅ Code generation engine loaded")
        except ImportError:
            logger.warning("Code generation engine not available")

        try:
            from knowledge import KnowledgeBase
            self.knowledge = KnowledgeBase()
            logger.info(f"✅ Knowledge base loaded ({self.knowledge.get_stats().get('entries', 0)} entries)")
        except ImportError:
            logger.warning("Knowledge base not available (install chromadb)")

    async def wake_up(self):
        """Activate the agent."""
        self.is_awake = True
        logger.info(f"🟢 {Config.AGENT_NAME} is awake and ready!")

        # Check system status
        ollama_ok = await self.ollama.is_available()
        claude_ok = bool(Config.ANTHROPIC_API_KEY)

        status = []
        status.append(f"Local AI (Ollama): {'✅ Online' if ollama_ok else '❌ Offline'}")
        status.append(f"Cloud AI (Claude): {'✅ Ready' if claude_ok else '⚠️ No API key'}")
        status.append(f"Code Engine: {'✅ Ready' if self.build_pipeline else '❌ Not loaded'}")
        status.append(f"Knowledge Base: {'✅ ' + str(self.knowledge.get_stats().get('entries', 0)) + ' memories' if self.knowledge else '❌ Not loaded'}")
        status.append(f"Pending tasks: {len(self.tasks.get_pending())}")

        return f"Good to see you! {Config.AGENT_NAME} is online.\n" + "\n".join(status)

    async def sleep(self):
        """Deactivate the agent."""
        self.is_awake = False
        logger.info(f"🔴 {Config.AGENT_NAME} going to sleep.")
        return f"{Config.AGENT_NAME} is going to sleep. Say '{Config.WAKE_PHRASE}' to wake me up."

    async def process(self, user_input: str) -> str:
        """Main processing pipeline."""

        # Check for wake/sleep commands
        if Config.WAKE_PHRASE.lower() in user_input.lower():
            return await self.wake_up()

        if any(cmd in user_input.lower() for cmd in ["go to sleep", "sleep mode", "shut down"]):
            return await self.sleep()

        if not self.is_awake:
            return ""  # Silent when sleeping

        # Step 1: Classify intent locally (no data leaves machine)
        logger.info(f"Classifying: {user_input[:50]}...")
        classification = await self.classifier.classify(user_input)
        logger.info(f"Intent: {classification}")

        category = classification.get("category", "INFORMATION")
        requires_cloud = classification.get("requires_cloud", False)
        intent = classification.get("intent", "unknown")

        # Step 2: Route to appropriate handler
        if category == "CODE":
            # ── Code Generation ─────────────────────
            response = await self._handle_code(user_input, classification)

        elif category == "DEVOPS":
            # ── DevOps / Shell Commands ─────────────
            response = await self._handle_devops(user_input, classification)

        elif category == "KNOWLEDGE":
            # ── Knowledge Base (Memory) ─────────────
            response = await self._handle_knowledge(user_input, classification)

        elif category == "PERSONAL" or category == "SYSTEM":
            # ALWAYS local - never send personal data to cloud
            response = await self.ollama.chat(
                user_input,
                system_prompt=f"You are {Config.AGENT_NAME}, a personal AI assistant. "
                              "Answer based on the user's personal context. Be helpful and concise."
            )

        elif category in ["CALENDAR", "RESERVATION", "COMMUNICATION"]:
            # Hybrid: plan with Claude, execute locally
            if requires_cloud:
                plan = await self.claude.reason(
                    classification.get("summary", user_input),
                    context=f"Category: {category}, Intent: {intent}"
                )
                response = f"Here's my plan:\n{plan}\n\nShall I execute this?"
            else:
                response = await self.ollama.chat(
                    user_input,
                    system_prompt=f"You are {Config.AGENT_NAME}. Help with: {category.lower()}"
                )

        elif category == "INFORMATION" and requires_cloud:
            # Complex research - use Claude
            response = await self.claude.reason(user_input)

        else:
            # Default to local
            response = await self.ollama.chat(user_input)

        # Store conversation in knowledge base
        if self.knowledge and response:
            try:
                self.knowledge.store_conversation(user_input, response[:500])
            except Exception:
                pass  # Don't let KB errors break the main flow

        return response

    # ─────────────────────────────────────────────
    # CODE Handler - Build apps, sites, scripts
    # ─────────────────────────────────────────────
    async def _handle_code(self, user_input: str, classification: dict) -> str:
        """Handle code generation requests."""
        if not self.build_pipeline:
            return ("Code generation engine not loaded. "
                    "Make sure codegen.py is in the src/ directory and dependencies are installed.")

        intent = classification.get("intent", "")
        summary = classification.get("summary", user_input)

        # Check if this is an iteration on existing project
        if any(kw in user_input.lower() for kw in ["update", "change", "modify", "fix", "improve", "iterate"]):
            # Try to find the most recent project to iterate on
            projects = self.build_pipeline.list_projects()
            if projects:
                latest = projects[-1]
                result = await self.build_pipeline.iterate(
                    latest["id"], user_input, deploy_target="copy"
                )
                if result["status"] == "complete":
                    return (f"✅ Updated project '{latest['name']}' (iteration {result['iteration']}).\n"
                            f"Files updated: {', '.join(result.get('files', []))}\n"
                            f"Location: {result.get('deploy_url', 'workspace')}")
                else:
                    return f"❌ Update failed: {result.get('error', 'Unknown error')}"

        # New project generation
        logger.info(f"🏗️ Building: {summary}")
        result = await self.build_pipeline.build(user_input, deploy_target="copy")

        if result["status"] == "complete":
            files_list = ", ".join(result.get("files", [])[:5])
            extra = f" (+{len(result['files']) - 5} more)" if len(result.get("files", [])) > 5 else ""

            response = (
                f"✅ Project '{result['project_name']}' is ready!\n\n"
                f"Files created: {files_list}{extra}\n"
                f"Location: {result.get('deploy_url', 'workspace')}\n"
            )

            # Add validation warnings if any
            validation = result.get("validation", {})
            if validation and validation.get("warnings"):
                response += f"\n⚠️ {len(validation['warnings'])} security warnings — review recommended."

            # Store in knowledge base
            if self.knowledge:
                self.knowledge.store_project(
                    result["project_name"],
                    user_input,
                    result.get("files", []),
                )

            return response

        elif result["status"] == "validation_failed":
            issues = result.get("validation", {}).get("blocked_issues", [])
            return (f"🛡️ Security validation blocked the build.\n"
                    f"Issues found: {len(issues)}\n"
                    + "\n".join(f"  - {i['file']}: {i['reason']}" for i in issues[:5]))

        else:
            return f"❌ Build failed: {result.get('error', 'Unknown error')}"

    # ─────────────────────────────────────────────
    # DEVOPS Handler - Shell commands, monitoring
    # ─────────────────────────────────────────────
    async def _handle_devops(self, user_input: str, classification: dict) -> str:
        """Handle DevOps and system management requests."""
        try:
            from codegen import ShellExecutor
        except ImportError:
            return "Shell executor not available."

        summary = classification.get("summary", user_input)

        # Use Ollama locally to determine what command to run
        cmd_prompt = f"""Based on this request, determine the exact shell command to run.
Only respond with a JSON object: {{"command": "the shell command", "explanation": "what it does"}}
Allowed commands: npm, pip, python3, node, git, ls, cat, mkdir, cp, vercel, netlify.
NEVER suggest: rm -rf, sudo, chmod 777, curl|sh, or any destructive command.

Request: {summary}"""

        cmd_response = await self.ollama.chat(cmd_prompt)

        try:
            import re
            match = re.search(r'\{[^}]+\}', cmd_response)
            if match:
                cmd_data = json.loads(match.group())
                command = cmd_data.get("command", "")
                explanation = cmd_data.get("explanation", "")

                result = await ShellExecutor.execute(command)

                if result["success"]:
                    output = result["stdout"][:1000] if result["stdout"] else "Command completed successfully."
                    return f"✅ Executed: `{command}`\n{explanation}\n\nOutput:\n{output}"
                else:
                    return f"❌ Command failed: {result['stderr']}"
            else:
                return f"I understood: {summary}, but couldn't determine the right command. Can you be more specific?"
        except Exception as e:
            return f"DevOps processing error: {e}"

    # ─────────────────────────────────────────────
    # KNOWLEDGE Handler - Memory & recall
    # ─────────────────────────────────────────────
    async def _handle_knowledge(self, user_input: str, classification: dict) -> str:
        """Handle knowledge storage and retrieval."""
        if not self.knowledge:
            return "Knowledge base not available. Install chromadb: pip install chromadb"

        intent = classification.get("intent", "")

        # Determine if storing or retrieving
        store_keywords = ["remember", "save", "store", "note", "keep in mind"]
        recall_keywords = ["recall", "remember", "what did", "do you know", "search", "find"]

        if any(kw in user_input.lower() for kw in store_keywords):
            # Store new knowledge
            doc_id = self.knowledge.store(user_input, category="user_note")
            return f"✅ Got it, I'll remember that. (memory ID: {doc_id})"

        elif any(kw in user_input.lower() for kw in recall_keywords):
            # Recall from knowledge base
            results = self.knowledge.search(user_input, n_results=5)
            if results:
                response = "Here's what I found in my memory:\n\n"
                for i, r in enumerate(results, 1):
                    content = r["content"][:300]
                    when = r["metadata"].get("timestamp", "unknown")[:10]
                    response += f"{i}. {content}\n   (from {when})\n\n"
                return response
            else:
                return "I don't have any memories matching that. Try being more specific?"

        else:
            # Default: search
            return self.knowledge.recall(user_input)


# ─────────────────────────────────────────────
# CLI Interface (for testing)
# ─────────────────────────────────────────────
async def main():
    agent = AgentOrchestrator()
    print(f"\n{'='*50}")
    print(f"  {Config.AGENT_NAME} - Personal AI Agent")
    print(f"  Say '{Config.WAKE_PHRASE}' to start")
    print(f"{'='*50}\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit"]:
                print(f"{Config.AGENT_NAME}: Goodbye!")
                break

            response = await agent.process(user_input)
            if response:
                print(f"{Config.AGENT_NAME}: {response}\n")
        except KeyboardInterrupt:
            print(f"\n{Config.AGENT_NAME}: Goodbye!")
            break


if __name__ == "__main__":
    asyncio.run(main())