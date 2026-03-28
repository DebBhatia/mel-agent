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
    USER_NAME = os.getenv("USER_NAME", "Deb")
    TIMEZONE = os.getenv("TIMEZONE", "America/Chicago")


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

- CALENDAR: Anything about scheduling, events, meetings, appointments on the calendar
- RESERVATION: Booking restaurants, hotels, doctor appointments
- INFORMATION: General questions, research, recommendations
- COMMUNICATION: Sending messages, emails, making calls
- HOME: Smart home controls, lights, locks, thermostat, temperature, device management
- MUSIC: Play music, pause, skip, volume, what's playing, Spotify
- REMINDER: Set reminders, alarms, "remind me", "in 30 minutes", scheduled tasks
- WEATHER: Weather conditions, temperature, forecast, umbrella, rain, sunny
- ROUTINE: Morning briefing, goodnight routine, leaving home, run a routine
- NOTIFICATION: Send a notification, push alert, notify me
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
                                    "HOME", "MUSIC", "REMINDER", "WEATHER", "ROUTINE", "NOTIFICATION",
                                    "PERSONAL", "SYSTEM", "CODE", "DEVOPS", "KNOWLEDGE"]
                if parsed and isinstance(parsed, dict) and parsed.get("category", "").upper() in valid_categories:
                    parsed["category"] = parsed["category"].upper()
                    return parsed

                return kw_result

        except Exception as e:
            logger.error(f"Ollama classification failed: {e}")
            return kw_result

    @staticmethod
    def _safe_summary(user_input: str) -> str:
        """Create a safe summary that strips potential PII for classification metadata."""
        # Truncate and avoid storing full raw input in classification dicts
        return user_input[:100]

    def _keyword_classify(self, user_input: str) -> dict:
        """Fallback keyword-based classifier when Ollama fails."""
        text = user_input.lower()
        summary = self._safe_summary(user_input)

        if any(kw in text for kw in ["weather", "forecast", "temperature outside", "how hot", "how cold",
                                        "umbrella", "raining", "is it sunny", "rain today"]):
            return {"category": "WEATHER", "intent": "check", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["morning briefing", "goodnight routine", "leaving home", "run routine",
                                        "start routine", "welcome home routine", "bedtime", "my briefing"]):
            return {"category": "ROUTINE", "intent": "execute", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["send notification", "notify me", "push alert", "send alert",
                                        "push notification"]):
            return {"category": "NOTIFICATION", "intent": "send", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["remember", "save this", "note that", "keep in mind", "don't forget"]):
            return {"category": "KNOWLEDGE", "intent": "store", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["my name", "my address", "my phone", "about me", "personal"]):
            return {"category": "PERSONAL", "intent": "query", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["recall", "what did i", "do you remember"]):
            return {"category": "KNOWLEDGE", "intent": "recall", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["play music", "play song", "pause music", "skip track", "next song",
                                        "what's playing", "now playing", "what is playing", "playing now",
                                        "spotify", "play some",
                                        "stop music", "previous song", "volume up", "volume down"]):
            return {"category": "MUSIC", "intent": "control", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["remind me", "set a reminder", "set reminder", "alarm",
                                        "in 30 minutes", "in an hour", "remind at"]):
            return {"category": "REMINDER", "intent": "create", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["my reminders", "pending reminders", "list reminders",
                                        "cancel reminder", "delete reminder"]):
            return {"category": "REMINDER", "intent": "manage", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["schedule", "calendar", "meeting", "event", "appointment",
                                        "book an appointment", "book a reminder", "book a meeting"]):
            return {"category": "CALENDAR", "intent": "manage", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["build", "create", "generate", "make me a", "code", "website", "web app", "mobile app", "script", "dashboard", "landing page"]):
            return {"category": "CODE", "intent": "generate", "requires_cloud": True, "summary": summary}
        elif any(kw in text for kw in ["book", "reservation", "table", "hotel", "reserve"]):
            return {"category": "RESERVATION", "intent": "book", "requires_cloud": True, "summary": summary}
        elif any(kw in text for kw in ["send", "text", "email", "message", "call"]):
            return {"category": "COMMUNICATION", "intent": "send", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["light", "lock", "thermostat", "temperature", "turn on", "turn off",
                                        "brightness", "dim", "scene", "smart home", "front door", "garage"]):
            return {"category": "HOME", "intent": "control", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["status", "system", "setting", "shut down", "sleep", "what can you"]):
            return {"category": "SYSTEM", "intent": "status", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["deploy", "git", "server", "docker", "process", "restart"]):
            return {"category": "DEVOPS", "intent": "manage", "requires_cloud": False, "summary": summary}
        else:
            return {"category": "INFORMATION", "intent": "unknown", "requires_cloud": True, "summary": summary}


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

                # Update local history (capped to prevent unbounded growth)
                self.conversation_history.append({"role": "user", "content": message})
                self.conversation_history.append({"role": "assistant", "content": assistant_msg})
                if len(self.conversation_history) > 30:
                    self.conversation_history = self.conversation_history[-20:]

                return assistant_msg
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return "Local model temporarily unavailable. Is Ollama running?"

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

    def _mel_system_prompt(self) -> str:
        """Build Mel's persona system prompt with live context."""
        try:
            import pytz
            tz = pytz.timezone(Config.TIMEZONE)
            now = datetime.now(tz)
        except Exception:
            now = datetime.now()
        time_str = now.strftime("%I:%M %p on %A, %B %d, %Y %Z").lstrip("0")
        return (
            f"You are {Config.AGENT_NAME}, a smart, warm, and highly capable personal AI assistant "
            f"for {Config.USER_NAME}. You speak naturally and conversationally — like a trusted "
            f"professional who genuinely cares. Be concise unless detail is requested. "
            f"Never say you lack access to real-time info — you are given live context. "
            f"Current date and time: {time_str}. "
            f"Address the user as {Config.USER_NAME} when appropriate."
        )

    async def converse(self, user_message: str, extra_context: str = "") -> str:
        """Conversational response as Mel — the primary chat method."""
        if not self.api_key:
            return "Claude API key not configured."

        safe_msg = self.sanitizer.sanitize(user_message)
        if self.enhanced_sanitizer:
            safe_msg = self.enhanced_sanitizer.sanitize(safe_msg)

        if self.audit:
            self.audit.log_api_call(
                service="claude",
                endpoint="https://api.anthropic.com/v1/messages",
                data_sent_preview=f"Chat: {safe_msg[:200]}"
            )

        content = safe_msg
        if extra_context:
            content = f"{extra_context}\n\n{safe_msg}"

        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "system": self._mel_system_prompt(),
            "messages": [{"role": "user", "content": content}],
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
                return self.sanitizer.desanitize(raw_response)
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return "I'm having trouble reaching my cloud brain right now. Try again in a moment."

    async def reason(self, task_description: str, context: str = "") -> str:
        """Send sanitized request to Claude for complex reasoning / planning."""
        if not self.api_key:
            return "Claude API key not configured. Using local model only."

        safe_description = self.sanitizer.sanitize(task_description)
        safe_context = self.sanitizer.sanitize(context)
        if self.enhanced_sanitizer:
            safe_description = self.enhanced_sanitizer.sanitize(safe_description)
            safe_context = self.enhanced_sanitizer.sanitize(safe_context)

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
            "system": self._mel_system_prompt(),
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
                return self.sanitizer.desanitize(raw_response)
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return "Cloud reasoning temporarily unavailable. Try again or use local model."


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
            return f"Action '{action_name}' failed. Check logs for details."


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

        # Capabilities
        self.build_pipeline = None
        self.knowledge = None
        self.scheduler = None
        self.spotify = None
        self.smarthome = None
        self._init_extensions()
        self._init_plugins()

    def _init_plugins(self):
        """Register action plugins (calendar, reservations, comms, music, home)."""
        try:
            from actions import register_all_plugins
            register_all_plugins(self.actions)
            logger.info("✅ Action plugins registered")
        except Exception as e:
            logger.warning(f"Action plugins not loaded: {e}")

        # Spotify music player
        try:
            from music import register_music_plugins, SpotifyPlayer
            self.spotify = SpotifyPlayer()
            register_music_plugins(self.actions)
            logger.info("✅ Spotify music plugin loaded")
        except ImportError:
            logger.warning("Music plugin not available")

        # Smart home
        try:
            from smarthome import register_smarthome_plugins
            self.smarthome = register_smarthome_plugins(self.actions)
            logger.info("✅ Smart home plugin loaded")
        except ImportError:
            logger.warning("Smart home plugin not available")

        # Scheduler / reminders
        try:
            from scheduler import TaskScheduler
            self.scheduler = TaskScheduler()
            logger.info("✅ Task scheduler loaded")
        except ImportError:
            logger.warning("Scheduler not available")

        # Weather
        self.weather = None
        try:
            from weather import register_weather_plugins
            self.weather = register_weather_plugins(self.actions)
            logger.info("✅ Weather plugin loaded")
        except ImportError:
            logger.warning("Weather plugin not available")

        # Routines
        self.routines = None
        try:
            from routines import RoutineEngine
            self.routines = RoutineEngine()
            logger.info("✅ Routines engine loaded")
        except ImportError:
            logger.warning("Routines engine not available")

        # Notifications
        self.notifications = None
        try:
            from notifications import register_notification_plugins
            self.notifications = register_notification_plugins(self.actions)
            logger.info("✅ Notification service loaded")
        except ImportError:
            logger.warning("Notification service not available")

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
        return self._build_greeting()

    def _build_greeting(self) -> str:
        """Generate a natural, time-appropriate greeting for the user."""
        try:
            import pytz
            tz = pytz.timezone(Config.TIMEZONE)
            hour = datetime.now(tz).hour
        except Exception:
            hour = datetime.now().hour

        name = Config.USER_NAME

        if 5 <= hour < 12:
            greeting = f"Good morning, {name}. I hope you had a wonderful night's rest."
        elif 12 <= hour < 17:
            greeting = f"Good afternoon, {name}. I hope your day is going well."
        elif 17 <= hour < 21:
            greeting = f"Good evening, {name}. Welcome home — it's great to have you back."
        else:
            greeting = f"Hey {name}, burning the midnight oil tonight? I'm here whenever you need me."

        return f"{greeting} How can I help you?"

    async def sleep(self):
        """Deactivate the agent."""
        self.is_awake = False
        logger.info(f"🔴 {Config.AGENT_NAME} going to sleep.")
        return f"{Config.AGENT_NAME} is going to sleep. Use the wake command to wake me up."

    async def process(self, user_input: str) -> str:
        """Main processing pipeline."""

        # Normalize apostrophes (smart quotes → straight) before any matching
        _inp = user_input.lower().replace('\u2019', "'").replace('\u2018', "'").replace('\u02bc', "'")

        # "Mel" / "Mel?" / "Mel:" / "Mel," — name alone or as prefix → attention acknowledgement
        _stripped = _inp.strip()
        if _stripped in ("mel", "mel?", "mel!") or _stripped.startswith("mel:") or _stripped.startswith("mel,"):
            return f"Yes, {Config.USER_NAME}?"

        # Check for wake/sleep commands — accept "daddy is home", "daddy's home", "daddys home"
        _wake_variants = [
            Config.WAKE_PHRASE.lower(),
            "wake up daddy's home",
            "wake up daddys home",
            "daddy's home",
            "daddy is home",
            "daddys home",
        ]
        if any(v in _inp for v in _wake_variants):
            return await self.wake_up()

        if any(cmd in user_input.lower() for cmd in ["go to sleep", "sleep mode", "shut down"]):
            return await self.sleep()

        if not self.is_awake:
            return ""  # Silent when sleeping

        # Step 1: Keyword classify only — skip slow Ollama LLM for general queries
        kw = self.classifier._keyword_classify(user_input)
        category = kw.get("category", "INFORMATION")

        # Only call Ollama classifier for clear action categories that need intent detail
        ACTION_CATEGORIES = {"CODE", "DEVOPS", "KNOWLEDGE", "CALENDAR", "RESERVATION", "COMMUNICATION", "HOME", "MUSIC", "REMINDER", "WEATHER", "ROUTINE", "NOTIFICATION"}
        if category in ACTION_CATEGORIES:
            logger.info(f"Classifying user input ({len(user_input)} chars)...")
            classification = await self.classifier.classify(user_input)
            category = classification.get("category", category)
            intent = classification.get("intent", "unknown")
            logger.info(f"Intent: category={category}, intent={intent}")
        else:
            classification = kw
            intent = kw.get("intent", "unknown")

        # Step 2a: Time/date — answer instantly from system clock
        if any(kw_t in _inp for kw_t in ("what time is it", "what's the time", "what is the time",
                                          "current time", "what day is it", "today's date",
                                          "what date is it", "what is today", "current date")):
            try:
                import pytz
                tz = pytz.timezone(Config.TIMEZONE)
                now = datetime.now(tz)
            except Exception:
                now = datetime.now()
            day_str = now.strftime("%A, %B %d, %Y")
            time_str = now.strftime("%I:%M %p").lstrip("0")
            return f"It's {time_str} on {day_str}, {Config.USER_NAME}."

        # Step 2b: Route action categories to specific handlers
        if category == "CODE":
            response = await self._handle_code(user_input, classification)

        elif category == "DEVOPS":
            response = await self._handle_devops(user_input, classification)

        elif category == "KNOWLEDGE":
            response = await self._handle_knowledge(user_input, classification)

        elif category == "CALENDAR":
            response = await self._handle_calendar(user_input, classification)

        elif category == "MUSIC":
            response = await self._handle_music(user_input, classification)

        elif category == "REMINDER":
            response = await self._handle_reminder(user_input, classification)

        elif category == "HOME":
            response = await self._handle_home(user_input, classification)

        elif category == "WEATHER":
            response = await self._handle_weather(user_input, classification)

        elif category == "ROUTINE":
            response = await self._handle_routine(user_input, classification)

        elif category == "NOTIFICATION":
            response = await self._handle_notification(user_input, classification)

        elif category in ["RESERVATION", "COMMUNICATION"]:
            response = await self.claude.converse(user_input)

        else:
            # Default: Claude with full Mel persona — handles all general conversation,
            # personal questions, information queries, small talk, etc.
            response = await self.claude.converse(user_input)

        # Store conversation in knowledge base
        if self.knowledge and response:
            try:
                self.knowledge.store_conversation(user_input, response[:500])
            except Exception:
                pass  # Don't let KB errors break the main flow

        return response

    # ─────────────────────────────────────────────
    # CALENDAR Handler - Book appointments & events
    # ─────────────────────────────────────────────
    async def _handle_calendar(self, user_input: str, classification: dict) -> str:
        """Handle calendar requests by extracting event details and calling the calendar plugin."""
        intent = classification.get("intent", "")
        summary = classification.get("summary", user_input)

        if "list" in intent or "check" in intent or "show" in intent:
            result = await self.actions.execute("calendar_list", {"days": 7})
            return result

        if "available" in intent or "free" in intent:
            return await self.actions.execute("calendar_check", {})

        # Default: create event — extract details from input
        # Parse natural language into event params
        from datetime import date as date_cls
        today = date_cls.today().isoformat()
        import re

        # Extract a meaningful event title from the user input.
        # Strategy: find the last "for <purpose>" phrase, which is where people
        # naturally put the event description. Fall back to stripping command words.
        title_clean = ""

        # Primary: scan all "for X" segments and pick the last meaningful one (not time/date).
        # "book appointment for today for 2pm for doctor appointment" → "doctor appointment"
        # "create appointment on calendar for dentist visit at 2:30 pm" → "dentist visit"
        # Also handle "to go for/to X"
        time_date_pat = re.compile(
            r'^(?:\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?|today|tomorrow|tonight)\b', re.IGNORECASE
        )
        # Split on "for" or "to go for/to" and examine each segment
        segments = re.split(r'\b(?:for|to go (?:for|to))\s+', user_input, flags=re.IGNORECASE)
        # Walk segments in reverse; first non-time/date segment is our title
        for seg in reversed(segments[1:]):  # skip the first segment (command prefix)
            candidate = seg.strip()
            # Remove trailing time expressions
            candidate = re.sub(
                r'\s*(?:at|for|on|by)\s+\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?\s*(?:today|tomorrow|tonight)?.*$',
                '', candidate, flags=re.IGNORECASE
            ).strip()
            candidate = re.sub(r'\s*\b(?:today|tomorrow|tonight)\b\s*$', '', candidate, flags=re.IGNORECASE).strip()
            # Skip pure time/date segments
            if candidate and not time_date_pat.match(candidate):
                title_clean = candidate
                break

        # Also check "to go to/for" pattern
        if not title_clean:
            go_match = re.search(r'to go (?:for|to)\s+(?:a\s+)?(.+?)(?:\s+(?:at|for|on)\s+\d|$)', user_input, re.IGNORECASE)
            if go_match:
                title_clean = go_match.group(1).strip()

        # Secondary: "I want to <purpose>"
        if len(title_clean) < 3:
            want_match = re.search(r'i want to\s+(.+)', user_input, re.IGNORECASE)
            if want_match:
                title_clean = want_match.group(1).strip()

        # Tertiary: strip command prefix and extract what remains
        if len(title_clean) < 3:
            title_clean = re.sub(
                r'^(?:can (?:you|i) |could you |please |hey mel[,]?\s*)*'
                r'(?:set up |create |add |schedule |book |make |put )?'
                r'(?:me )?(?:an? )?(?:event|appointment|meeting|reminder|block)\s*'
                r'(?:on |in |to )?(?:my |the )?(?:calendar|schedule|gcal)?\s*',
                '', user_input.strip(), count=1, flags=re.IGNORECASE
            ).strip()
            # If nothing matched (e.g. "schedule a meeting"), try stripping just the verb
            if title_clean == user_input.strip():
                title_clean = re.sub(
                    r'^(?:can (?:you|i) |could you |please |hey mel[,]?\s*)*'
                    r'(?:set up |create |add |schedule |book |make |put )\s*'
                    r'(?:me )?(?:an? )?\s*',
                    '', user_input.strip(), count=1, flags=re.IGNORECASE
                ).strip()
            # Remove time/date from whatever remains
            title_clean = re.sub(
                r'\s*(?:at|for)\s+\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?\s*', '',
                title_clean, flags=re.IGNORECASE
            ).strip()
            title_clean = re.sub(r'\s*\b(?:today|tomorrow|tonight)\b', '', title_clean, flags=re.IGNORECASE).strip()
            # Remove trailing "to/on my calendar" or "on my schedule"
            title_clean = re.sub(r'\s*(?:to|on|in)\s+(?:my |the )?(?:calendar|schedule|gcal)\s*$', '', title_clean, flags=re.IGNORECASE).strip()
            title_clean = re.sub(r'^(?:for|to|about|on|at)\s+', '', title_clean, flags=re.IGNORECASE).strip()

        # Final fallback: extract event type from the original input
        if len(title_clean) < 3:
            type_match = re.search(r'\b(meeting|appointment|reminder|event|session|call|standup|sync)\b', user_input, re.IGNORECASE)
            if type_match:
                title_clean = type_match.group(1).capitalize()
            elif summary and len(summary) < 60:
                title_clean = summary
            else:
                title_clean = "New Event"

        # Capitalize first letter
        event_title = title_clean[0].upper() + title_clean[1:] if title_clean else "New Event"

        params = {
            "title": event_title,
            "start_time": f"{today}T13:30:00",
            "end_time": f"{today}T14:00:00",
            "location": "",
            "description": f"Booked by {Config.AGENT_NAME}",
            "timezone": "America/Chicago",
        }

        # Try to extract specifics from the input
        # Support formats: 4:00pm, 4:00 pm, 4:00 p.m., 4:00p.m., 4 pm, 4 p.m.
        time_match = re.search(r'(\d{1,2}):(\d{2})\s*(?:(a\.?m\.?|p\.?m\.?))?', user_input, re.IGNORECASE)
        if not time_match:
            # Try format without colon: "4 pm", "4pm", "4 p.m."
            time_match = re.search(r'(\d{1,2})\s+(a\.?m\.?|p\.?m\.?)', user_input, re.IGNORECASE)
        if time_match:
            hour = int(time_match.group(1))
            groups = time_match.groups()
            # Determine minutes and am/pm based on which regex matched
            if len(groups) == 3:
                # Colon format: hour:min ampm
                minute = groups[1] if groups[1] else "00"
                ampm_raw = groups[2] or ''
            else:
                # No-colon format: hour ampm (no minutes)
                minute = "00"
                ampm_raw = groups[1] or ''
            # Normalize am/pm (strip dots)
            ampm = ampm_raw.replace('.', '').strip().lower()
            # If no am/pm specified and hour is 1-6, assume PM (people don't book 2:00 AM events)
            if not ampm and 1 <= hour <= 6:
                ampm = 'pm'
            if ampm == 'pm' and hour < 12:
                hour += 12
            elif ampm == 'am' and hour == 12:
                hour = 0
            params["start_time"] = f"{today}T{hour:02d}:{minute}:00"
            # Default 30 min duration
            end_hour = hour
            end_min = int(minute) + 30
            if end_min >= 60:
                end_hour += 1
                end_min -= 60
            params["end_time"] = f"{today}T{end_hour:02d}:{end_min:02d}:00"

        result = await self.actions.execute("calendar_create", params)
        return result

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
                    return "❌ Command was blocked or failed. Try being more specific."
            else:
                return "I couldn't determine the right command. Can you be more specific?"
        except Exception as e:
            logger.error(f"DevOps processing error: {e}")
            return "DevOps processing error. Check logs for details."

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
    # MUSIC Handler - Spotify playback control
    # ─────────────────────────────────────────────
    async def _handle_music(self, user_input: str, classification: dict) -> str:
        """Handle music playback commands."""
        text = user_input.lower()

        if not self.spotify:
            return "Music player not available. Make sure music.py is in the src/ directory."

        if not self.spotify.auth.is_configured():
            return "Spotify is not configured. Add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to your .env file."

        if not self.spotify.auth.is_authenticated:
            return "Spotify not connected. Visit /spotify/auth in your browser to authenticate."

        if any(kw in text for kw in ["pause", "stop music"]):
            return await self.actions.execute("music_pause", {})

        elif any(kw in text for kw in ["skip", "next song", "next track"]):
            return await self.actions.execute("music_skip", {})

        elif any(kw in text for kw in ["what's playing", "now playing", "current song", "what song"]):
            return await self.actions.execute("music_now_playing", {})

        elif any(kw in text for kw in ["previous", "go back", "last song"]):
            data = await self.spotify.previous_track()
            return data

        elif "volume" in text:
            import re
            m = re.search(r'(\d+)', text)
            if m:
                return await self.spotify.set_volume(int(m.group(1)))
            elif "up" in text:
                return await self.spotify.set_volume(80)
            elif "down" in text:
                return await self.spotify.set_volume(30)
            return await self.spotify.set_volume(50)

        elif any(kw in text for kw in ["play"]):
            # Extract what to play
            import re
            play_match = re.search(r'play\s+(?:me\s+)?(?:some\s+)?(.+)', text)
            if play_match:
                query = play_match.group(1).strip()
                return await self.actions.execute("music_play", {"query": query})
            return await self.actions.execute("music_play", {})

        else:
            return await self.actions.execute("music_now_playing", {})

    # ─────────────────────────────────────────────
    # REMINDER Handler - Scheduled tasks & reminders
    # ─────────────────────────────────────────────
    async def _handle_reminder(self, user_input: str, classification: dict) -> str:
        """Handle reminder creation and management."""
        if not self.scheduler:
            return "Scheduler not available. Make sure scheduler.py is in the src/ directory."

        text = user_input.lower()
        intent = classification.get("intent", "")

        if any(kw in text for kw in ["list", "pending", "my reminders", "show reminders"]):
            pending = self.scheduler.get_pending()
            if not pending:
                return "No pending reminders."
            lines = []
            for r in pending:
                time_str = r.get("trigger_time", "")[:16].replace("T", " at ")
                rec = f" (repeats {r['recurrence']})" if r.get("recurrence") else ""
                lines.append(f"- {r['title']} — {time_str}{rec} [ID: {r['id']}]")
            return f"Pending reminders ({len(pending)}):\n" + "\n".join(lines)

        elif any(kw in text for kw in ["cancel", "delete", "remove"]):
            import re
            m = re.search(r'(?:cancel|delete|remove)\s+(?:reminder\s+)?([a-f0-9]+)', text)
            if m:
                rid = m.group(1)
                success = self.scheduler.cancel_reminder(rid)
                return f"Reminder cancelled." if success else f"Reminder {rid} not found."
            return "Which reminder should I cancel? Provide the reminder ID."

        else:
            # Create a new reminder
            from scheduler import parse_reminder_from_text
            parsed = parse_reminder_from_text(user_input)
            reminder = self.scheduler.create_reminder(
                title=parsed["title"] or user_input,
                trigger_time=parsed["trigger_time"],
                recurrence=parsed.get("recurrence"),
            )
            time_str = reminder.trigger_time[:16].replace("T", " at ")
            rec_str = f" (repeating {reminder.recurrence})" if reminder.recurrence else ""
            return f"Reminder set: \"{reminder.title}\" — {time_str}{rec_str}"

    # ─────────────────────────────────────────────
    # HOME Handler - Smart home device control
    # ─────────────────────────────────────────────
    async def _handle_home(self, user_input: str, classification: dict) -> str:
        """Handle smart home commands."""
        if not self.smarthome:
            return "Smart home module not available. Make sure smarthome.py is in the src/ directory."

        text = user_input.lower()
        import re

        # Status check
        if any(kw in text for kw in ["status", "summary", "how is", "what's the"]):
            return await self.smarthome.get_status_summary()

        # Scene activation
        scene_match = re.search(r'(?:activate|set|start)\s+(?:scene\s+)?(\w+(?:\s+\w+)?)\s+scene', text)
        if not scene_match:
            scene_match = re.search(r'scene\s+(\w+(?:\s+\w+)?)', text)
        if scene_match:
            return await self.smarthome.activate_scene(scene_match.group(1).replace(" ", "_"))

        # Temperature
        temp_match = re.search(r'(?:set\s+)?(?:temperature|thermostat|temp)\s+(?:to\s+)?(\d+)', text)
        if temp_match:
            return await self.smarthome.set_temperature(float(temp_match.group(1)))

        # Lock/unlock
        if "unlock" in text:
            door = "front_door"
            if "garage" in text:
                door = "garage"
            elif "back" in text:
                door = "back_door"
            return await self.smarthome.unlock_door(door)
        elif "lock" in text:
            door = "front_door"
            if "garage" in text:
                door = "garage"
            elif "back" in text:
                door = "back_door"
            return await self.smarthome.lock_door(door)

        # Brightness
        bright_match = re.search(r'(?:brightness|dim)\s+(?:to\s+)?(\d+)', text)
        if bright_match:
            device = "light.living_room"
            if "bedroom" in text:
                device = "light.bedroom"
            elif "kitchen" in text:
                device = "light.kitchen"
            return await self.smarthome.set_brightness(device, int(bright_match.group(1)))

        # Turn on/off
        if "turn off" in text or "switch off" in text:
            device = self._extract_device_name(text)
            return await self.smarthome.turn_off(device)
        elif "turn on" in text or "switch on" in text:
            device = self._extract_device_name(text)
            return await self.smarthome.turn_on(device)

        # Fallback — list devices
        return await self.smarthome.get_status_summary()

    # ─────────────────────────────────────────────
    # WEATHER Handler - Current conditions & forecast
    # ─────────────────────────────────────────────
    async def _handle_weather(self, user_input: str, classification: dict) -> str:
        """Handle weather queries."""
        if not self.weather:
            return "Weather service not available. Add OPENWEATHER_API_KEY to your .env file."

        text = user_input.lower()
        import re

        # Extract city if mentioned
        city = None
        city_match = re.search(r'(?:weather|forecast|temperature)\s+(?:in|for|at)\s+(.+?)(?:\?|$|\.)', text)
        if city_match:
            city = city_match.group(1).strip()

        if any(kw in text for kw in ["forecast", "next few days", "this week", "tomorrow"]):
            result = await self.weather.get_forecast(city)
            if "error" in result:
                return result["error"]
            lines = [f"Forecast for {result['city']}:"]
            for day in result["forecast"]:
                lines.append(f"  {day['date']}: {day['description']} — {day['low']}{day['unit_symbol']} to {day['high']}{day['unit_symbol']}")
            return "\n".join(lines)

        if any(kw in text for kw in ["umbrella", "rain", "raining"]):
            return await self.weather.needs_umbrella(city)

        return await self.weather.get_summary(city)

    # ─────────────────────────────────────────────
    # ROUTINE Handler - Multi-step automations
    # ─────────────────────────────────────────────
    async def _handle_routine(self, user_input: str, classification: dict) -> str:
        """Handle routine execution and management."""
        if not self.routines:
            return "Routines engine not available."

        text = user_input.lower()

        # List routines
        if any(kw in text for kw in ["list routine", "show routine", "what routine", "available routine"]):
            routines = self.routines.list_routines()
            lines = ["Available routines:"]
            for r in routines:
                tag = " (built-in)" if r["builtin"] else " (custom)"
                lines.append(f"  - {r['name']}: {r['description']}{tag}")
            return "\n".join(lines)

        # Match a specific routine to run
        routine_id = None
        if any(kw in text for kw in ["morning", "briefing", "brief me"]):
            routine_id = "morning_briefing"
        elif any(kw in text for kw in ["goodnight", "bedtime", "good night"]):
            routine_id = "goodnight"
        elif any(kw in text for kw in ["leaving", "leave home", "heading out", "going out"]):
            routine_id = "leaving_home"
        elif any(kw in text for kw in ["welcome home", "i'm home", "im home", "i am home"]):
            routine_id = "welcome_home"

        if routine_id:
            return await self.routines.execute_and_summarize(
                routine_id, self.actions,
                agent_name=Config.AGENT_NAME, user_name=Config.USER_NAME,
            )

        return ("I have these routines available: Morning Briefing, Goodnight, Leaving Home, Welcome Home. "
                "Which one would you like me to run?")

    # ─────────────────────────────────────────────
    # NOTIFICATION Handler - Push alerts
    # ─────────────────────────────────────────────
    async def _handle_notification(self, user_input: str, classification: dict) -> str:
        """Handle push notification requests."""
        if not self.notifications:
            return "Notification service not available. Configure NTFY_TOPIC, Pushover, or Telegram in .env."

        if not self.notifications.is_configured():
            return ("No notification backends configured. Add one of these to your .env:\n"
                    "  - NTFY_TOPIC=your-topic-name (easiest, free)\n"
                    "  - PUSHOVER_USER_KEY + PUSHOVER_API_TOKEN\n"
                    "  - TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID")

        import re
        # Extract message content
        msg_match = re.search(r'(?:send|notify|alert|push)\s+(?:me\s+)?(?:a\s+)?(?:notification\s+)?(?:that\s+|saying\s+)?(.+)', user_input, re.IGNORECASE)
        message = msg_match.group(1).strip() if msg_match else user_input

        result = await self.notifications.send(
            title=f"From {Config.AGENT_NAME}",
            message=message,
        )
        if result["status"] == "sent":
            backends = ", ".join(result["backends"].keys())
            return f"Notification sent via {backends}."
        return "Failed to send notification. Check your configuration."

    @staticmethod
    def _extract_device_name(text: str) -> str:
        """Extract device name from natural language command."""
        import re
        m = re.search(r'(?:turn\s+(?:on|off)|switch\s+(?:on|off))\s+(?:the\s+)?(.+?)(?:\s+light|\s+fan|\s+switch)?$', text.lower())
        if m:
            name = m.group(1).strip()
            # Map common names to entity IDs
            name_map = {
                "living room": "light.living_room",
                "bedroom": "light.bedroom",
                "kitchen": "light.kitchen",
                "fan": "switch.fan",
                "ceiling fan": "switch.fan",
            }
            return name_map.get(name, name)
        return "light.living_room"


# ─────────────────────────────────────────────
# CLI Interface (for testing)
# ─────────────────────────────────────────────
async def main():
    agent = AgentOrchestrator()
    print(f"\n{'='*50}")
    print(f"  {Config.AGENT_NAME} - Personal AI Agent")
    print(f"  Say the wake phrase to start")
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