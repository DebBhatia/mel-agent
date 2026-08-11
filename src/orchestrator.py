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
# Secret loader — vault first, env fallback
# ─────────────────────────────────────────────
def _load_secret(key: str, default: str = "") -> str:
    """Load a secret from the encrypted vault; fall back to env var."""
    try:
        from security import SecretVault
        _vault = SecretVault()
        val = _vault.get(key, "")
        if val:
            return val
    except Exception:
        pass
    return os.getenv(key, default)


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
class Config:
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
    # Small, fast model used only for intent classification (a simple
    # single-label task) -- keeps voice latency down without giving up
    # OLLAMA_MODEL's quality for actual chat/command-generation use.
    OLLAMA_CLASSIFY_MODEL = os.getenv("OLLAMA_CLASSIFY_MODEL", "llama3.2:1b")
    ANTHROPIC_API_KEY = _load_secret("ANTHROPIC_API_KEY")
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
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
                raw = json.load(f)
            # Filter out non-string values (e.g. encrypted stub keys like _encrypted: true)
            self.mappings = {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}
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
- DESIGN: UI/UX design, style guides, color palettes, typography, design system, font pairing, chart type, landing page design

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
                        "model": Config.OLLAMA_CLASSIFY_MODEL,
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

                valid_categories = ["CALENDAR", "RESERVATION", "INFORMATION", "COMMUNICATION", "EMAIL",
                                    "HOME", "MUSIC", "REMINDER", "WEATHER", "ROUTINE", "NOTIFICATION",
                                    "PERSONAL", "SYSTEM", "CODE", "DEVOPS", "KNOWLEDGE", "DESIGN", "SEARCH"]
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
        elif any(kw in text for kw in ["remember", "save this", "note that", "keep in mind", "don't forget",
                                        "add a note", "make a note", "write this down", "jot this down"]):
            return {"category": "KNOWLEDGE", "intent": "store", "requires_cloud": False, "summary": summary}
        elif (any(kw in text for kw in ["search for", "search the web", "look up", "google that", "google this",
                                        "find on the internet", "find online", "web search", "browse for",
                                        "search online", "look it up", "look that up",
                                        "search the internet", "look on the internet", "find on the web",
                                        "search on the internet", "search on internet",
                                        "search the net", "search on the web"])
              or __import__('re').search(r'\b(search|research|look)\b.{0,25}\b(internet|online|web|for me)\b', text)
              or __import__('re').search(r'\b(check|find out|look into|find)\b.{0,20}\b(online|internet|web)\b', text)
              or __import__('re').search(r"\b(what'?s|what\s+is|how\s+much\s+is)\b.{0,30}\b(price|cost|worth)\b", text)
              or __import__('re').search(r'\b(price|stock price|exchange rate|score of)\b.*\bright now\b', text)):
            return {"category": "SEARCH", "intent": "web_search", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["open news", "show me the news", "show news", "open the news",
                                        "open stocks", "show stocks", "open the market", "show the market",
                                        "open stock market", "open my stocks", "check my stocks",
                                        "show me stocks", "open finance", "open yahoo finance"]):
            return {"category": "SEARCH", "intent": "open_browser", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["my name", "my address", "my phone", "about me", "personal"]):
            return {"category": "PERSONAL", "intent": "query", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["recall", "what did i", "do you remember"]):
            return {"category": "KNOWLEDGE", "intent": "recall", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in [
                    # Play / Resume / Continue
                    "play music", "play a music", "play me music", "play some music",
                    "play song", "play a song", "play me a song", "play some",
                    "resume music", "resume playback", "resume the music",
                    "continue playing", "continue the music", "continue music",
                    "keep playing", "unpause",
                    # Pause / Stop
                    "pause music", "pause the music", "stop music", "stop the music",
                    "stop playing", "turn off the music", "turn off music", "mute the music", "mute music",
                    # Skip / Next
                    "skip track", "skip this song", "skip song", "next song", "next track", "play next",
                    # Previous
                    "previous song", "previous track", "go back", "last song", "play previous",
                    # Now playing
                    "what's playing", "now playing", "what is playing", "playing now",
                    "what song is this", "what song", "current song", "current track",
                    # Volume
                    "volume up", "volume down", "set volume", "turn up", "turn down",
                    # General
                    "spotify",
                ]) \
                or (text.strip() in ("pause", "skip", "resume", "next", "previous")) \
                or __import__('re').search(r'\b(pause|skip|resume)\b', text) and any(w in text for w in ["music", "song", "track", "spotify", "playing"]) \
                or __import__('re').search(r'\b(stop|pause|skip|mute)\b.*\b(music|song|track|playing|spotify)\b', text) \
                or __import__('re').search(r'\bplay\b.{0,15}\b(music|song|track|playlist|album|genre|country|rock|pop|jazz|hip.?hop|r&b|classical|lo.?fi|chill|rap|indie|latin|edm|metal|blues|folk|punk|soul|reggae|electronic|beats|mix|radio|station|vibe|mood)\b', text) \
                or (__import__('re').search(r'\bplay\b\s+(?:me\s+|us\s+)?(?:some\s+|the\s+|a\s+|today.?s?\s+)?\w', text)
                    and not any(w in text for w in ["play game", "play video", "play movie", "play a role", "display"])):
            return {"category": "MUSIC", "intent": "control", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["remind me", "set a reminder", "set reminder", "alarm",
                                        "in 30 minutes", "in an hour", "remind at"]):
            # If the reminder mentions "calendar" or a specific date, route to CALENDAR (Google Calendar)
            # Only send to REMINDER handler for short relative timers ("in X minutes/hours")
            import re as _re_kw
            has_calendar_ref = "calendar" in text or "gcal" in text
            has_specific_date = bool(_re_kw.search(
                r'\b(today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday'
                r'|\d{1,2}[:/]\d{2}|\d{1,2}\s*(?:am|pm))\b', text
            ))
            is_relative_timer = bool(_re_kw.search(r'\bin\s+\d+\s+(minute|hour|second)', text))
            if has_calendar_ref or (has_specific_date and not is_relative_timer):
                return {"category": "CALENDAR", "intent": "create", "requires_cloud": False, "summary": summary}
            return {"category": "REMINDER", "intent": "create", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["my reminders", "pending reminders", "list reminders",
                                        "cancel reminder", "delete reminder"]):
            return {"category": "REMINDER", "intent": "manage", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["design system", "color palette", "font pairing", "ui style",
                                        "ux guideline", "chart type", "typography", "design for",
                                        "style guide", "icon set", "landing pattern"]):
            return {"category": "DESIGN", "intent": "search", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["schedule", "calendar", "meeting", "event", "appointment",
                                        "book an appointment", "book a reminder", "book a meeting",
                                        "add to my calendar", "add to the calendar", "add to calendar",
                                        "put on my calendar", "put on the calendar",
                                        "block time", "block off", "free time", "my schedule",
                                        "open schedule", "what do i have", "am i free", "am i busy",
                                        "show my", "my appointments", "my events", "my meetings",
                                        "delete event", "remove event", "cancel event",
                                        "reschedule", "move my", "move the meeting",
                                        "set up a meeting", "set up a call"]) \
                or __import__('re').search(
                    r'\b(delete|remove|cancel|reschedule|move|update|edit)\b\s+(?:the\s+|my\s+|a\s+|an\s+)?'
                    r'(?:\w+\s+){0,3}(?:meeting|appointment|event|reminder|checkup|checkups|'
                    r'lunch|dinner|breakfast|call|session|sync|standup|visit|consultation)',
                    text
                ):
            return {"category": "CALENDAR", "intent": "manage", "requires_cloud": False, "summary": summary}
        elif __import__('re').search(
            r'\b(add|create|book|schedule|set up|put|plan)\b.{1,40}\b(on|for|at)\b.{1,30}\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow)\b'
            r'|\b(delete|remove|cancel)\b.{1,40}\b(on|for|at)\b.{1,30}\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow)\b'
            r'|\b(move|reschedule|change)\b.{1,40}\b(to|from)\b.{1,30}\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}(am|pm))\b',
            text
        ):
            return {"category": "CALENDAR", "intent": "manage", "requires_cloud": False, "summary": summary}
        elif any(kw in text for kw in ["build", "create", "generate", "make me a", "code", "website", "web app", "mobile app", "script", "dashboard", "landing page"]):
            return {"category": "CODE", "intent": "generate", "requires_cloud": True, "summary": summary}
        elif any(kw in text for kw in ["book", "reservation", "table", "hotel", "reserve"]):
            return {"category": "RESERVATION", "intent": "book", "requires_cloud": True, "summary": summary}
        elif any(kw in text for kw in ["check my email", "read my email", "check email", "read email",
                                        "my inbox", "check inbox", "new emails", "unread email",
                                        "search email", "search my email", "find email",
                                        "draft email", "draft an email", "compose email",
                                        "check my gmail", "read my gmail", "gmail"]):
            return {"category": "EMAIL", "intent": "read", "requires_cloud": False, "summary": summary}
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

    async def chat_stream(self, message: str, system_prompt: str = None):
        """Stream tokens from Ollama. Yields text chunks as they arrive."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.conversation_history[-10:])
        messages.append({"role": "user", "content": message})

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json={"model": self.model, "messages": messages, "stream": True},
                ) as response:
                    full_response = ""
                    async for line in response.aiter_lines():
                        if line:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                full_response += token
                                yield token
                    # Update history after stream completes
                    self.conversation_history.append({"role": "user", "content": message})
                    self.conversation_history.append({"role": "assistant", "content": full_response})
                    if len(self.conversation_history) > 30:
                        self.conversation_history = self.conversation_history[-20:]
        except Exception as e:
            logger.error(f"Ollama stream error: {e}")
            yield "Local model temporarily unavailable. Is Ollama running?"

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
        self.conversation_history = []  # Multi-turn memory for Claude
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

        # Load personal context from about-me folder (silent fail if not available)
        about_me_context = ""
        try:
            from journal import AboutMe
            about_me_text = AboutMe.get_summary()
            if about_me_text:
                about_me_context = f"\n\n## What you know about {Config.USER_NAME}:\n{about_me_text}"
        except Exception:
            pass

        return (
            f"You are {Config.AGENT_NAME}, {Config.USER_NAME}'s personal assistant. "
            f"{Config.USER_NAME} is male — always use he/him pronouns when referring to him. "
            f"You talk like a close friend who happens to know everything — direct, casual, no fluff. "
            f"Current date and time: {time_str}. "
            f"\n\nTone rules (never break these):"
            f"\n- Give the actual answer immediately. No 'great question!' or 'I'd be happy to help' ever."
            f"\n- Never ask clarifying questions unless the request is genuinely impossible to answer without them."
            f"\n- Never use bullet points or headers for simple conversational questions — just talk."
            f"\n- Keep it short. 1-3 sentences for casual questions. Expand only if they ask for more."
            f"\n- Sound like a person texting, not a corporate help desk."
            f"\n- Use contractions. Be opinionated. Say 'yeah', 'nah', 'honestly', 'totally fine', etc."
            f"\n- If something is fine/safe/okay, just say so directly. Don't hedge with 5 disclaimers."
            f"\n- NEVER invent personal details, names, events, plans, or facts about the user's life that you don't actually know. If you don't know, just don't mention it."
            f"\n- Only reference calendar events, reminders, or personal info if it was explicitly provided to you in this conversation."
            f"\n- Never list possible concerns unless asked. Just answer the actual question."
            f"\n- Humor: NEVER tell the same joke twice in a session. Vary your style every single time — try puns, dry wit, observational humor, self-deprecating AI jokes, dark comedy, one-liners, wordplay. Never default to 'Why don't scientists trust atoms?' — that joke is banned."
            f"\n- Variety: never start two consecutive responses the same way. Mix it up naturally — sometimes dive straight in, sometimes 'Honestly,', 'Nah,', 'Yeah,', 'So,', 'Look,' or nothing at all. Sound like a real person, not a bot."
            f"\n\nExample of BAD response: 'I'd be happy to help! Could you tell me your dietary restrictions? Are you concerned about sodium, allergens, or medication interactions?'"
            f"\nExample of GOOD response: 'Yeah totally fine — cocktail peanuts and beer are a classic combo. Just watch the sodium if you're having a bunch, but one serving won't hurt you.'"
            f"{about_me_context}"
        )

    def _prepare_converse(self, user_message: str, extra_context: str = ""):
        """Shared prep for converse/converse_stream: sanitize input, build payload."""
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

        # Build messages with conversation history for multi-turn context
        messages = list(self.conversation_history[-10:])
        messages.append({"role": "user", "content": content})

        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "system": self._mel_system_prompt(),
            "messages": messages,
        }
        return content, headers, payload

    def _update_converse_history(self, content: str, response_text: str):
        """Append a turn to conversation history (stores sanitized text only)."""
        self.conversation_history.append({"role": "user", "content": content})
        self.conversation_history.append({"role": "assistant", "content": response_text})
        if len(self.conversation_history) > 30:
            self.conversation_history = self.conversation_history[-20:]

    async def converse(self, user_message: str, extra_context: str = "") -> str:
        """Conversational response as Mel — the primary chat method."""
        if not self.api_key:
            return "Claude API key not configured."

        content, headers, payload = self._prepare_converse(user_message, extra_context)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                )
                result = response.json()
                raw_response = result["content"][0]["text"]
                self._update_converse_history(content, raw_response)
                return self.sanitizer.desanitize(raw_response)
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return "I'm having trouble reaching my cloud brain right now. Try again in a moment."

    async def converse_stream(self, user_message: str, extra_context: str = ""):
        """Stream tokens from Claude API. Yields text chunks as they arrive."""
        if not self.api_key:
            yield "Claude API key not configured."
            return

        content, headers, payload = self._prepare_converse(user_message, extra_context)
        payload["stream"] = True

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                ) as response:
                    # Check for HTTP errors before trying to read the stream
                    if response.status_code != 200:
                        body = ""
                        async for chunk in response.aiter_bytes():
                            body += chunk.decode("utf-8", errors="replace")
                            if len(body) > 500:
                                break
                        logger.error(f"Claude stream HTTP {response.status_code}: {body[:300]}")
                        yield f"Hmm, something went sideways on my end ({response.status_code}). Try again in a sec?"
                        return

                    full_response = ""
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data.get("type") == "content_block_delta":
                                    text = data.get("delta", {}).get("text", "")
                                    if text:
                                        full_response += text
                                        yield text
                                elif data.get("type") == "error":
                                    logger.error(f"Claude stream error event: {data}")
                            except json.JSONDecodeError:
                                continue
                    # Update history after stream completes (sanitized text only)
                    if full_response:
                        self._update_converse_history(content, full_response)
        except Exception as e:
            logger.error(f"Claude stream error: {e}")
            # Don't yield an error message — let the orchestrator fallback handle it

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
        # Pending calendar action awaiting user confirmation
        # Structure: {"type": "create"|"delete"|"update", "params": {}, "description": str}
        self.pending_calendar_action = None
        self._last_search_results = None

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

        # UI/UX design intelligence
        self.ui_ux = None
        try:
            from ui_ux import register_ui_ux_plugins
            self.ui_ux = register_ui_ux_plugins(self.actions)
            logger.info("✅ UI/UX Pro Max plugin loaded")
        except ImportError:
            logger.warning("UI/UX plugin not available")

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
        """Activate the agent with a full homecoming briefing (weather + calendar)."""
        self.is_awake = True
        logger.info(f"🟢 {Config.AGENT_NAME} is awake and ready!")

        parts = [self._build_greeting()]

        # Add weather briefing
        weather_added = False
        try:
            if self.weather and self.weather.is_configured():
                current = await self.weather.get_current()
                if current and "error" not in current:
                    temp = current.get("temperature", "")
                    unit = current.get("unit_symbol", "°")
                    desc = current.get("description", "")
                    if temp and desc:
                        parts.append(f"It's currently {temp}{unit} and {desc.lower()} outside.")
                        weather_added = True
                        # Smart advice
                        try:
                            from weather import WeatherService
                            advice = WeatherService.get_weather_advice(temp, desc, self.weather.units)
                            if advice:
                                parts.append(advice)
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"Wake-up weather fetch failed: {e}")
        if not weather_added:
            logger.info("Weather not available for wake-up briefing")

        # Add calendar briefing
        calendar_added = False
        try:
            if self.actions and "calendar_list" in self.actions.actions:
                result = await self.actions.execute("calendar_list", {"days": 1})
                if result and not result.startswith(("No upcoming", "Calendar not", "Failed")):
                    parts.append(f"Here's your schedule: {result}")
                    calendar_added = True
                else:
                    parts.append("Your calendar is clear today.")
                    calendar_added = True
        except Exception as e:
            logger.warning(f"Wake-up calendar fetch failed: {e}")
        if not calendar_added:
            logger.info("Calendar not available for wake-up briefing")

        # Always end with availability
        if len(parts) > 1:
            parts.append("What can I do for you?")

        return " ".join(parts)

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

        # Check for wake/sleep commands — primary: "wake up daddy is home"
        _wake_variants = [
            Config.WAKE_PHRASE.lower(),
            "wake up daddy is home",
            "wake up daddy's home",
            "wake up daddys home",
            "daddy's home",
            "daddy is home",
            "daddys home",
            "dayy is home",      # common speech-to-text typo
            "day is home",       # another STT variant
        ]
        if any(v in _inp for v in _wake_variants):
            return await self.wake_up()

        if any(cmd in user_input.lower() for cmd in ["go to sleep", "sleep mode", "shut down"]):
            return await self.sleep()

        if not self.is_awake:
            return f"I'm sleeping right now, {Config.USER_NAME}. Say 'wake up daddy is home' or press the WAKE MEL button to wake me up."

        # Step 0: If there's a pending confirmation action, route to calendar handler first
        if self.pending_calendar_action:
            return await self._handle_calendar(user_input, {})

        # Quick check: user asking for search sources/links from last search
        if hasattr(self, '_last_search_results') and self._last_search_results:
            _links_ask = any(kw in _inp for kw in ["show me the links", "show the links", "share the links",
                                                     "give me the sources", "show sources", "share sources",
                                                     "send me the links", "what are the sources", "the urls",
                                                     "show me the resources", "share the resources"])
            if _links_ask:
                lines = ["Here are the sources:\n"]
                for i, r in enumerate(self._last_search_results, 1):
                    lines.append(f"{i}. **{r['title']}**\n   {r.get('url', '')}")
                self._last_search_results = None
                return "\n".join(lines)

        # Step 1: Keyword classify only — skip slow Ollama LLM for general queries
        kw = self.classifier._keyword_classify(user_input)
        category = kw.get("category", "INFORMATION")



        # Only call Ollama classifier for clear action categories that need intent detail
        ACTION_CATEGORIES = {"CODE", "DEVOPS", "KNOWLEDGE", "CALENDAR", "RESERVATION", "COMMUNICATION", "EMAIL", "HOME", "MUSIC", "REMINDER", "WEATHER", "ROUTINE", "NOTIFICATION", "DESIGN"}
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

        elif category == "EMAIL":
            response = await self._handle_email(user_input, classification)

        elif category == "DESIGN":
            response = await self._handle_design(user_input, classification)

        elif category == "SEARCH":
            response = await self._handle_search(user_input, classification)

        elif category in ["RESERVATION", "COMMUNICATION"]:
            response = await self.claude.converse(user_input)

        else:
            # Default: Claude with full Mel persona — handles all general conversation,
            # personal questions, information queries, small talk, etc.
            response = await self.claude.converse(user_input)
            # If Claude returned empty or errored, use fallback
            if (not response or not response.strip()
                    or "trouble reaching" in response.lower()
                    or "not configured" in response.lower()):
                response = await self._conversational_fallback(user_input)

        # Store conversation in knowledge base
        if self.knowledge and response:
            try:
                self.knowledge.store_conversation(user_input, response[:500])
            except Exception:
                pass  # Don't let KB errors break the main flow

        return response

    async def process_stream(self, user_input: str):
        """Streaming variant of process(). Yields tokens for SSE delivery.
        For action categories (instant results), yields the full response at once.
        For Claude/Ollama-backed categories, streams token-by-token.
        """
        # Normalize apostrophes
        _inp = user_input.lower().replace('\u2019', "'").replace('\u2018', "'").replace('\u02bc', "'")
        _stripped = _inp.strip()

        # Quick responses — yield full result at once
        if _stripped in ("mel", "mel?", "mel!") or _stripped.startswith("mel:") or _stripped.startswith("mel,"):
            yield f"Yes, {Config.USER_NAME}?"
            return

        _wake_variants = [
            Config.WAKE_PHRASE.lower(), "wake up daddy's home", "wake up daddys home",
            "daddy's home", "daddy is home", "daddys home",
        ]
        if any(v in _inp for v in _wake_variants):
            yield await self.wake_up()
            return

        if any(cmd in _inp for cmd in ["go to sleep", "sleep mode", "shut down"]):
            yield await self.sleep()
            return

        if not self.is_awake:
            yield f"I'm sleeping right now, {Config.USER_NAME}. Say 'wake up daddy is home' or press the WAKE MEL button to wake me up."
            return

        if self.pending_calendar_action:
            yield await self._handle_calendar(user_input, {})
            return

        # Classify intent
        kw = self.classifier._keyword_classify(user_input)
        category = kw.get("category", "INFORMATION")

        ACTION_CATEGORIES = {"CODE", "DEVOPS", "KNOWLEDGE", "CALENDAR", "RESERVATION", "COMMUNICATION", "EMAIL", "HOME", "MUSIC", "REMINDER", "WEATHER", "ROUTINE", "NOTIFICATION"}
        if category in ACTION_CATEGORIES:
            classification = await self.classifier.classify(user_input)
            category = classification.get("category", category)
        else:
            classification = kw

        # Time/date — instant answer
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
            yield f"It's {time_str} on {day_str}, {Config.USER_NAME}."
            return

        # Action categories — instant results, yield full response
        INSTANT_CATEGORIES = {"CODE", "DEVOPS", "KNOWLEDGE", "CALENDAR", "EMAIL", "MUSIC", "REMINDER", "HOME", "WEATHER", "ROUTINE", "NOTIFICATION", "SEARCH"}
        if category in INSTANT_CATEGORIES:
            response = await self.process(user_input)
            yield response
            return

        # Claude-backed categories — stream token-by-token
        full_response = ""
        try:
            async for token in self.claude.converse_stream(user_input):
                full_response += token
                yield token
        except Exception as e:
            logger.error(f"Stream failed: {e}")

        # If Claude returned nothing, generate a local fallback response
        if not full_response.strip():
            logger.warning("Claude stream returned empty — using fallback")
            fallback = await self._conversational_fallback(user_input)
            full_response = fallback
            yield fallback

        # Store conversation in knowledge base
        if self.knowledge and full_response:
            try:
                self.knowledge.store_conversation(user_input, full_response[:500])
            except Exception:
                pass

    # ─────────────────────────────────────────────
    # Conversational Fallback
    # ─────────────────────────────────────────────
    async def _conversational_fallback(self, user_input: str) -> str:
        """Generate a response when Claude streaming fails or returns empty.
        Tries non-streaming Claude first, then Ollama, then a hardcoded response."""
        # Try non-streaming Claude
        try:
            response = await self.claude.converse(user_input)
            if (response and response.strip()
                    and "trouble reaching" not in response.lower()
                    and "not configured" not in response.lower()):
                return response
        except Exception as e:
            logger.warning(f"Claude non-stream fallback failed: {e}")

        # Try local Ollama
        try:
            if await self.ollama.is_available():
                response = await self.ollama.chat(
                    user_input,
                    system_prompt=(
                        f"You are {Config.AGENT_NAME}, {Config.USER_NAME}'s personal assistant. "
                        "Be casual, friendly, direct. Keep responses to 1-3 sentences. "
                        "Sound like a person texting, not a help desk."
                    )
                )
                if response and "unavailable" not in response.lower():
                    return response
        except Exception as e:
            logger.warning(f"Ollama fallback failed: {e}")

        # Last resort: pattern-matched casual responses
        _inp = user_input.lower().strip()
        greetings = ["how are you", "how you doing", "how's it going", "what's up",
                     "how do you do", "how have you been", "how're you", "sup",
                     "what's good", "how goes it", "how are things"]
        if any(g in _inp for g in greetings):
            return f"I'm doing great, {Config.USER_NAME}! What can I help you with?"

        compliments = ["you're awesome", "you're great", "good job", "nice work",
                       "thank you", "thanks", "appreciate", "you rock"]
        if any(c in _inp for c in compliments):
            return f"Thanks, {Config.USER_NAME}! Always happy to help."

        farewells = ["bye", "goodbye", "see you", "good night", "later", "peace out"]
        if any(f in _inp for f in farewells):
            return f"Later, {Config.USER_NAME}! I'll be here whenever you need me."

        opinions = ["what do you think", "your opinion", "thoughts on",
                    "are you", "do you like", "do you think"]
        if any(o in _inp for o in opinions):
            return "Honestly, that's a great question. Let me think on that — ask me again in a sec?"

        import random
        last_resorts = [
            f"Caught that, {Config.USER_NAME} — but I'm drawing a blank on this one. Try rewording it?",
            f"Hmm, I'm not sure what to do with that. Give me a different angle?",
            f"That one slipped past me. Want to rephrase and try again?",
            f"Nah, I got nothing on that. Hit me with it a different way.",
            f"I hear you, {Config.USER_NAME}, but I need a bit more to work with — try again?",
            f"Honestly, that one stumped me. Different wording might help.",
        ]
        return random.choice(last_resorts)

    # ─────────────────────────────────────────────
    # CALENDAR Handler - Book appointments & events
    # ─────────────────────────────────────────────

    def _resolve_date(self, text: str) -> str:
        """Resolve natural language date references to YYYY-MM-DD strings."""
        import re
        from datetime import date as _date, timedelta as _td
        today = _date.today()
        text_lower = text.lower()

        if re.search(r'\btomorrow\b', text_lower):
            return (today + _td(days=1)).isoformat()
        if re.search(r'\btoday\b', text_lower) or re.search(r'\btonight\b', text_lower):
            return today.isoformat()
        if re.search(r'\bnext week\b', text_lower):
            return (today + _td(days=7)).isoformat()

        day_names = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
        for i, day in enumerate(day_names):
            if re.search(rf'\b{day}\b', text_lower):
                days_ahead = (i - today.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                return (today + _td(days=days_ahead)).isoformat()

        return today.isoformat()

    def _extract_time(self, text: str, base_date: str) -> tuple:
        """Extract start/end times from text. Returns (start_iso, end_iso)."""
        import re
        hour, minute = 13, 30  # default 1:30 PM

        # Check for word-based times first
        if re.search(r'\bnoon\b', text, re.IGNORECASE):
            hour, minute = 12, 0
        elif re.search(r'\bmidnight\b', text, re.IGNORECASE):
            hour, minute = 0, 0
        elif re.search(r'\bmorning\b', text, re.IGNORECASE) and not re.search(r'\d', text):
            hour, minute = 9, 0
        elif re.search(r'\bevening\b', text, re.IGNORECASE) and not re.search(r'\d', text):
            hour, minute = 18, 0
        else:
            time_match = re.search(r'(\d{1,2}):(\d{2})\s*(a\.?m\.?|p\.?m\.?)?', text, re.IGNORECASE)
            if not time_match:
                time_match = re.search(r'(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)', text, re.IGNORECASE)
                if time_match:
                    hour = int(time_match.group(1))
                    minute = 0
                    ampm_raw = time_match.group(2) or ''
                else:
                    ampm_raw = ''
            else:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2))
                ampm_raw = time_match.group(3) or ''

            ampm = ampm_raw.replace('.', '').strip().lower()
            if not ampm and 1 <= hour <= 6:
                ampm = 'pm'
            if ampm == 'pm' and hour < 12:
                hour += 12
            elif ampm == 'am' and hour == 12:
                hour = 0

        end_minute = minute + 30
        end_hour = hour + (1 if end_minute >= 60 else 0)
        end_minute = end_minute % 60

        return (
            f"{base_date}T{hour:02d}:{minute:02d}:00",
            f"{base_date}T{end_hour:02d}:{end_minute:02d}:00"
        )

    def _extract_title(self, user_input: str, summary: str = "") -> str:
        """Extract a clean event title from user input."""
        import re
        title_clean = ""
        time_date_pat = re.compile(
            r'^(?:\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?|today|tomorrow|tonight'
            r'|monday|tuesday|wednesday|thursday|friday|saturday|sunday'
            r'|next\s+\w+|this\s+\w+|last\s+\w+)\b', re.IGNORECASE
        )
        segments = re.split(r'\b(?:for|to go (?:for|to))\s+', user_input, flags=re.IGNORECASE)
        for seg in reversed(segments[1:]):
            candidate = seg.strip()
            candidate = re.sub(
                r'\s*(?:at|for|on|by)\s+\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?\s*(?:today|tomorrow|tonight)?.*$',
                '', candidate, flags=re.IGNORECASE
            ).strip()
            candidate = re.sub(r'\s*\b(?:today|tomorrow|tonight)\b\s*$', '', candidate, flags=re.IGNORECASE).strip()
            if candidate and not time_date_pat.match(candidate):
                title_clean = candidate
                break

        if not title_clean:
            go_match = re.search(r'to go (?:for|to)\s+(?:a\s+)?(.+?)(?:\s+(?:at|for|on)\s+\d|$)', user_input, re.IGNORECASE)
            if go_match:
                title_clean = go_match.group(1).strip()

        # "remind me [at time] to [activity]" → extract activity as title
        if not title_clean or len(title_clean) < 3:
            remind_match = re.search(
                r'\bremind\s+me\b.{0,30}?\bto\s+(.+?)(?:\s+at\s+\d|\s+on\s+|\s+for\s+\d|$)',
                user_input, re.IGNORECASE
            )
            if remind_match:
                title_clean = remind_match.group(1).strip()

        if len(title_clean) < 3:
            want_match = re.search(r'i want to\s+(.+)', user_input, re.IGNORECASE)
            if want_match:
                title_clean = want_match.group(1).strip()

        if len(title_clean) < 3:
            title_clean = re.sub(
                r'^(?:can (?:you|i) |could you |please |hey mel[,]?\s*)*'
                r'(?:set up |create |add |schedule |book |make |put |move |reschedule |change |update |delete |remove |cancel )?'
                r'(?:me )?(?:an? )?(?:event|appointment|meeting|reminder|block)\s*'
                r'(?:on |in |to )?(?:my |the )?(?:calendar|schedule|gcal)?\s*',
                '', user_input.strip(), count=1, flags=re.IGNORECASE
            ).strip()
            if title_clean == user_input.strip():
                title_clean = re.sub(
                    r'^(?:can (?:you|i) |could you |please |hey mel[,]?\s*)*'
                    r'(?:set up |create |add |schedule |book |make |put )\s*'
                    r'(?:me )?(?:an? )?\s*',
                    '', user_input.strip(), count=1, flags=re.IGNORECASE
                ).strip()
            title_clean = re.sub(r'\s*(?:at|for)\s+\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?\s*', '', title_clean, flags=re.IGNORECASE).strip()
            title_clean = re.sub(r'\s*\bat\s+(?:noon|midnight|morning|evening|night)\b', '', title_clean, flags=re.IGNORECASE).strip()
            title_clean = re.sub(r'\s*\b(?:next|this|last)\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|week|month)\b', '', title_clean, flags=re.IGNORECASE).strip()
            title_clean = re.sub(r'\s*\b(?:today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', '', title_clean, flags=re.IGNORECASE).strip()
            title_clean = re.sub(r'\s*(?:to|on|in)\s+(?:my |the )?(?:calendar|schedule|gcal)\s*$', '', title_clean, flags=re.IGNORECASE).strip()
            # Clean up orphaned prepositions left after stripping times/dates
            title_clean = re.sub(r'\s+\b(?:on|at|for|in)\s*$', '', title_clean, flags=re.IGNORECASE).strip()
            title_clean = re.sub(r'^(?:for|to|about|on|at)\s+', '', title_clean, flags=re.IGNORECASE).strip()
            # Collapse multiple spaces
            title_clean = re.sub(r'\s{2,}', ' ', title_clean).strip()

        if len(title_clean) < 3:
            type_match = re.search(r'\b(meeting|appointment|reminder|event|session|call|standup|sync|lunch|dinner|breakfast)\b', user_input, re.IGNORECASE)
            if type_match:
                title_clean = type_match.group(1).capitalize()
            elif summary and len(summary) < 60:
                title_clean = summary
            else:
                title_clean = "New Event"

        title_clean = re.sub(r'\s+\b(?:for|at|to|on|in|a|an|the|by)\b\s*$', '', title_clean, flags=re.IGNORECASE).strip()

        # Normalize vague destination phrases to proper event titles
        _normalizations = [
            (r'^(?:the\s+)?doctors?\s*(?:office|appointment)?$', "Doctor's Appointment"),
            (r'^(?:the\s+)?dentist\s*(?:office|appointment)?$', "Dentist Appointment"),
            (r'^(?:the\s+)?gym\s*$', "Gym"),
            (r'^(?:the\s+)?hospital\s*$', "Hospital Visit"),
            (r'^(?:the\s+)?pharmacy\s*$', "Pharmacy"),
            (r'^(?:i\s+)?have\s+to\s+go\s+to\s+(?:the\s+)?doctors?\s*$', "Doctor's Appointment"),
        ]
        for pattern, replacement in _normalizations:
            if re.match(pattern, title_clean, re.IGNORECASE):
                title_clean = replacement
                break

        return title_clean[0].upper() + title_clean[1:] if title_clean else "New Event"

    async def _handle_calendar(self, user_input: str, classification: dict) -> str:
        """Full-featured calendar handler: list, check, create, delete, update with confirmation."""
        import re
        text_lower = user_input.lower()
        summary = classification.get("summary", user_input)

        logger.info(f"[CALENDAR] input={user_input!r} pending={self.pending_calendar_action is not None}")

        # ── 1. Handle pending confirmation ──────────────────────────────────
        if self.pending_calendar_action:
            pending = self.pending_calendar_action
            self.pending_calendar_action = None

            yes_words = ["yes", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "do it", "go ahead",
                         "correct", "right", "sounds good", "let's do it", "go for it", "definitely",
                         "absolutely", "please", "of course", "add it", "delete it", "remove it",
                         "that's right", "that's correct", "affirmative", "proceed", "continue"]
            no_words = ["no", "nope", "cancel", "nevermind", "never mind", "don't", "stop", "abort", "forget it", "skip it"]

            action = pending["action"]
            params = pending["params"]

            # Handle numbered choice for multi-match delete
            if action == "choose_delete":
                num_match = re.search(r'\b([1-5])\b', text_lower)
                if num_match:
                    idx = int(num_match.group(1)) - 1
                    evs = params.get("events", [])
                    if 0 <= idx < len(evs):
                        ev = evs[idx]
                        result = await self.actions.execute("calendar_delete", {
                            "event_id": ev["id"],
                            "title": ev["title"]
                        })
                        return result
                    return "That number doesn't match any of the events I listed."
                elif any(w in text_lower for w in no_words):
                    return "Got it, cancelled."
                else:
                    # Re-show choices
                    self.pending_calendar_action = pending
                    lines = ["Which one? Say the number:"]
                    for i, ev in enumerate(params.get("events", []), 1):
                        lines.append(f"{i}. {ev['title']} ({ev['start'][:10] if ev['start'] else '?'})")
                    return "\n".join(lines)

            if any(w in text_lower for w in yes_words):
                if action == "create":
                    result = await self.actions.execute("calendar_create", params)
                    return result

                elif action == "delete":
                    result = await self.actions.execute("calendar_delete", {
                        "event_id": params["event_id"],
                        "title": params["title"]
                    })
                    return result

                elif action == "update":
                    result = await self.actions.execute("calendar_update", params)
                    return result

            elif any(w in text_lower for w in no_words):
                return "Got it, cancelled. Let me know if you want to make any changes."
            else:
                # Not a clear yes/no — re-prompt
                self.pending_calendar_action = pending
                title = params.get("title", "that event")
                return f"Just to confirm — do you want me to {action} \"{title}\"? Say yes or no."

        # ── 2. Intent detection ──────────────────────────────────────────────
        _ev_types = (r'event|appointment|meeting|reminder|checkup|checkups|consultation|'
                     r'lunch|dinner|breakfast|brunch|call|session|sync|standup|visit|class|workout')
        delete_signals = re.search(
            rf'\b(delete|remove|cancel|get rid of|erase|drop)\b.{{0,60}}\b({_ev_types}|it|that)\b'
            rf'|\b(delete|remove|cancel|get rid of|erase|drop)\b\s+(?:my\s+)?(?:the\s+)?(?:\w+\s+){{0,5}}(?:{_ev_types})',
            text_lower
        )
        update_signals = re.search(
            rf'\b(move|reschedule|change|update|edit|modify|shift)\b.{{0,60}}\b(to|from|at|by|earlier|later)\b'
            rf'|\b(move|reschedule|change|update|edit|modify)\b\s+(?:my\s+)?(?:the\s+)?(?:\w+\s+){{0,6}}(?:{_ev_types}|to\s+\d)',
            text_lower
        )
        check_signals = re.search(
            r'\b(free|available|open|busy|slot)\b'
            r'|\bdo i have\b'
            r'|\bam i (?:free|available|busy)\b'
            r'|\bany (?:free|open|available) (?:time|slot|window)\b',
            text_lower
        )
        list_phrases = [
            "show me", "what are my", "list my", "list all", "what's on",
            "what is on", "see my", "view my", "check my", "show my",
            "get my", "display my", "upcoming", "what do i have",
            "open schedule", "my schedule", "my appointments", "my events",
            "my calendar", "what's happening", "what is happening"
        ]
        day_names_pattern = r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
        create_verbs = re.search(
            r'\b(add|create|book|schedule|set up|put|place|plan|make|arrange|remind me|set a reminder|set reminder)\b', text_lower
        )
        is_list_request = (
            (any(phrase in text_lower for phrase in list_phrases) and not create_verbs)
            or (re.search(day_names_pattern, text_lower) and not create_verbs)
        )

        # ── 3. LIST / SHOW events ────────────────────────────────────────────
        if is_list_request and not delete_signals and not update_signals:
            day_match = re.search(r'next\s+(\d+)\s+days?', text_lower)
            specific_day = re.search(day_names_pattern, text_lower)
            if day_match:
                days = int(day_match.group(1))
                result = await self.actions.execute("calendar_list", {"days": days})
            elif specific_day:
                target_date = self._resolve_date(user_input)
                result = await self.actions.execute("calendar_day", {"date": target_date})
            elif "next week" in text_lower or "this week" in text_lower:
                result = await self.actions.execute("calendar_list", {"days": 7})
            elif "today" in text_lower or "tonight" in text_lower:
                result = await self.actions.execute("calendar_day", {"date": self._resolve_date(user_input)})
            elif "tomorrow" in text_lower:
                result = await self.actions.execute("calendar_day", {"date": self._resolve_date(user_input)})
            else:
                result = await self.actions.execute("calendar_list", {"days": 7})
            return result

        # ── 4. CHECK availability ────────────────────────────────────────────
        if check_signals and not delete_signals and not update_signals:
            target_date = self._resolve_date(user_input)
            result = await self.actions.execute("calendar_day", {"date": target_date})
            # Wrap with availability context
            if "no events" in result.lower() or "nothing" in result.lower() or "empty" in result.lower():
                day_ref = "that day"
                for day in ["monday","tuesday","wednesday","thursday","friday","saturday","sunday","today","tomorrow"]:
                    if day in text_lower:
                        day_ref = day.capitalize()
                        break
                return f"Looks like you're wide open on {day_ref} — no events scheduled."
            return result

        # ── 5. DELETE event ──────────────────────────────────────────────────
        if delete_signals:
            # Strip delete verbs before extracting title
            clean_for_title = re.sub(
                r'^(?:can you |please |hey mel[,]?\s*)?'
                r'(?:delete|remove|cancel|get rid of|erase|drop)\s+(?:my\s+)?(?:the\s+)?',
                '', user_input, flags=re.IGNORECASE
            ).strip()
            search_query = self._extract_title(clean_for_title, summary)
            # Only narrow by date if user explicitly mentioned one
            has_date_ref = __import__('re').search(
                r'\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next week|this week)\b',
                user_input, __import__('re').IGNORECASE
            )
            find_params = {"query": search_query, "days": 30}
            if has_date_ref:
                find_params["date"] = self._resolve_date(user_input)
            events = await self.actions.execute("calendar_find", find_params)

            if isinstance(events, str):
                # calendar_find returned a string error/not-found message
                return f"I couldn't find an event matching \"{search_query}\" to delete. Can you be more specific?"

            if not events:
                return f"I couldn't find any event matching \"{search_query}\" in the next 2 weeks."

            if len(events) == 1:
                ev = events[0]
                ev_title = ev.get("summary", "that event")
                ev_id = ev.get("id", "")
                ev_start = ev.get("start", {}).get("dateTime", ev.get("start", {}).get("date", ""))
                self.pending_calendar_action = {
                    "action": "delete",
                    "params": {"event_id": ev_id, "title": ev_title}
                }
                return f"Just to confirm — you want me to delete \"{ev_title}\" ({ev_start[:10] if ev_start else 'date unknown'})? Say yes to confirm or no to cancel."
            else:
                # Multiple matches — filter out any that start with action words (accidental events)
                clean_events = [e for e in events if not re.match(
                    r'^(?:delete|cancel|remove|create|add|schedule)\b',
                    e.get("summary",""), re.IGNORECASE
                )]
                if len(clean_events) == 1:
                    ev = clean_events[0]
                    ev_title = ev.get("summary", "that event")
                    ev_id = ev.get("id", "")
                    ev_start = ev.get("start", {}).get("dateTime", ev.get("start", {}).get("date", ""))
                    self.pending_calendar_action = {
                        "action": "delete",
                        "params": {"event_id": ev_id, "title": ev_title}
                    }
                    return f"Just to confirm — delete \"{ev_title}\" ({ev_start[:10] if ev_start else '?'})? Say yes or no."
                shown = (clean_events or events)[:5]
                self.pending_calendar_action = {
                    "action": "choose_delete",
                    "params": {"events": [{"id": e.get("id"), "title": e.get("summary","Untitled"),
                                           "start": e.get("start",{}).get("dateTime", e.get("start",{}).get("date",""))}
                                          for e in shown]}
                }
                lines = ["Found a few events matching that. Which one do you want to delete? Say the number."]
                for i, ev in enumerate(shown, 1):
                    ev_title = ev.get("summary", "Untitled")
                    ev_start = ev.get("start", {}).get("dateTime", ev.get("start", {}).get("date", ""))
                    lines.append(f"{i}. {ev_title} ({ev_start[:10] if ev_start else '?'})")
                return "\n".join(lines)

        # ── 6. UPDATE / RESCHEDULE event ────────────────────────────────────
        if update_signals:
            search_query = self._extract_title(user_input, summary)
            # Strip update verbs from the query so we don't search for "move dentist"
            search_query = re.sub(
                r'^(?:move|reschedule|change|update|edit|modify|shift)\s+(?:my\s+)?(?:the\s+)?',
                '', search_query, flags=re.IGNORECASE
            ).strip()
            # Strip trailing "to [date/time/anything]" rescheduling target from search query
            search_query = re.sub(
                r'\s+to\s+(?:next|this|last)\b.*$'
                r'|\s+to\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|today|\d{1,2}(?::\d{2})?\s*(?:am|pm)|noon|midnight).*$'
                r'|\s+(?:next|this|last)\s*(?:\w+\s*)?$'
                r'|\s+at\s+(?:noon|midnight|\d{1,2}(?::\d{2})?\s*(?:am|pm))\s*$',
                '', search_query, flags=re.IGNORECASE
            ).strip()

            events = await self.actions.execute("calendar_find", {
                "query": search_query,
                "days": 30
            })

            if isinstance(events, str) or not events:
                return f"I couldn't find an event matching \"{search_query}\". Can you be more specific?"

            ev = events[0]
            ev_id = ev.get("id", "")
            ev_title = ev.get("summary", "that event")

            # Extract new time/date from input
            new_date = self._resolve_date(user_input)
            new_start, new_end = self._extract_time(user_input, new_date)

            self.pending_calendar_action = {
                "action": "update",
                "params": {
                    "event_id": ev_id,
                    "title": ev_title,
                    "start_time": new_start,
                    "end_time": new_end,
                    "timezone": "America/Indiana/Indianapolis"
                }
            }
            # Format for human display
            from datetime import datetime as _dt
            try:
                display_time = _dt.fromisoformat(new_start).strftime("%B %d at %I:%M %p").lstrip("0")
            except Exception:
                display_time = new_start
            return f"Got it — move \"{ev_title}\" to {display_time}? Say yes to confirm or no to cancel."

        # ── 7. CREATE event ──────────────────────────────────────────────────
        event_title = self._extract_title(user_input, summary)
        event_date = self._resolve_date(user_input)
        start_time, end_time = self._extract_time(user_input, event_date)

        params = {
            "title": event_title,
            "start_time": start_time,
            "end_time": end_time,
            "location": "",
            "description": f"Booked by {Config.AGENT_NAME}",
            "timezone": "America/Indiana/Indianapolis",
        }

        logger.info(f"[CALENDAR CREATE] title={event_title!r} start={start_time} end={end_time}")

        # Execute immediately — no confirmation gate (voice assistants just do it)
        result = await self.actions.execute("calendar_create", params)
        logger.info(f"[CALENDAR CREATE] result={result!r}")
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
    # DESIGN Handler - UI/UX design intelligence
    # ─────────────────────────────────────────────
    async def _handle_design(self, user_input: str, classification: dict) -> str:
        """Handle UI/UX design queries using the design intelligence plugin."""
        if not self.ui_ux:
            return "UI/UX design plugin not loaded. Ensure ui_ux.py is in the src/ directory."

        intent = classification.get("intent", "")
        text_lower = user_input.lower()

        # Check if requesting a full design system
        if any(kw in text_lower for kw in ["design system", "full design", "complete design", "style guide for"]):
            return await self.actions.execute("ui_ux_design_system", {
                "query": user_input,
                "project_name": None,
            })

        # Check for stack-specific queries
        from ui_ux import AVAILABLE_STACKS
        for stack in AVAILABLE_STACKS:
            if stack.replace("-", " ") in text_lower or stack in text_lower:
                return await self.actions.execute("ui_ux_stack", {
                    "query": user_input,
                    "stack": stack,
                })

        # Default: domain search (auto-detect domain)
        return await self.actions.execute("ui_ux_search", {
            "query": user_input,
        })

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
    # SEARCH Handler - Web search via DuckDuckGo
    # ─────────────────────────────────────────────
    async def _handle_search(self, user_input: str, classification: dict) -> str:
        """Handle web search requests and browser-open commands."""
        intent = classification.get("intent", "")

        # Browser-open commands (open news, stocks, etc.)
        if intent == "open_browser":
            import subprocess
            import platform
            text = user_input.lower()
            opened = []

            def _open(url: str):
                try:
                    system = platform.system()
                    if system == "Windows":
                        subprocess.Popen(["cmd", "/c", "start", "", url], shell=False)
                    elif system == "Darwin":
                        subprocess.Popen(["open", url])
                    else:
                        subprocess.Popen(["xdg-open", url])
                except Exception as ex:
                    logger.warning(f"Could not open {url}: {ex}")

            news_url = os.getenv("NEWS_URL", "https://news.google.com")
            stocks_url = os.getenv("STOCKS_URL", "https://finance.yahoo.com")

            if any(w in text for w in ["news", "headlines"]):
                _open(news_url)
                opened.append("news")
            if any(w in text for w in ["stock", "market", "finance"]):
                _open(stocks_url)
                opened.append("stocks")
            if not opened:
                _open(news_url)
                opened.append("news")

            return f"Opened {' and '.join(opened)} for you."

        try:
            from search import WebSearch
        except ImportError:
            # Fall back to Claude for search-like questions
            return await self.claude.converse(user_input)

        # Extract the search query — strip conversational preamble to get the core question
        query = user_input
        import re
        # Iteratively strip common preamble patterns
        preamble_patterns = [
            r"^(hey\s+mel[,\s]*|mel[,\s]+)",
            r"^(can|could|would)\s+you\s+(please\s+)?",
            r"^(please\s+)?",
            r"^(research|search|look\s+up|google|find|browse|check)\s+"
            r"(on\s+the\s+(internet|web)\s+|on\s+the\s+|on\s+|the\s+(internet|web)\s+(for\s+)?|for\s+|online\s+)?"
            r"(for\s+me\s+|for\s+us\s+)?"
            r"(about\s+|for\s+|on\s+)?",
            r"^(tell\s+me|i\s+want\s+to\s+know|i\s+need\s+to\s+know|find\s+out)\s+(about\s+)?",
            r"^(what'?s|what\s+is)\s+",
            # Don't strip "how much" — it's useful context for the search query
        ]
        changed = True
        while changed:
            changed = False
            for p in preamble_patterns:
                cleaned = re.sub(p, "", query.strip(), flags=re.IGNORECASE).strip()
                if cleaned and len(cleaned) < len(query.strip()):
                    query = cleaned
                    changed = True
                    break
        # Clean up trailing noise words
        query = re.sub(r'\s+(for|about|on|the|a|an|is|are|of)\s*$', '', query, flags=re.IGNORECASE).strip()
        # If stripping removed too much, use original input
        if len(query) < 3:
            query = user_input

        # Add "price" context if the original question was about cost
        if re.search(r'\b(how much|price|cost|worth)\b', user_input, re.IGNORECASE) and not re.search(r'\b(price|cost)\b', query, re.IGNORECASE):
            query += " price"

        results = await WebSearch.search(query, max_results=5)
        if not results:
            return f"Couldn't find anything for '{query}'. Try a different search term?"

        # Build context from search results for Claude to summarize
        search_context = "\n".join(
            f"- {r['title']}: {r.get('snippet', '')} (source: {r.get('url', '')})"
            for r in results
        )

        # Use Claude to produce a smart, concise summary
        try:
            summary_prompt = (
                f"The user asked: \"{user_input}\"\n\n"
                f"Here are web search results:\n{search_context}\n\n"
                f"Give a concise, direct answer based on these results. "
                f"Speak naturally as Mel (personal AI assistant). "
                f"Include key facts, numbers, and prices if relevant. "
                f"At the end, mention you can share the source links if they want more detail. "
                f"Keep it to 2-4 sentences max."
            )
            summary = await self.claude.converse(summary_prompt, extra_context="You are summarizing web search results. Be factual and concise.")
            # Store the URLs so user can ask for them
            self._last_search_results = results
            return summary
        except Exception as e:
            logger.warning(f"Claude summarization failed, returning raw results: {e}")
            # Fallback to raw results
            lines = [f"Here's what I found for **{query}**:\n"]
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. **{r['title']}**")
                if r.get("snippet"):
                    lines.append(f"   {r['snippet']}")
            return "\n".join(lines)

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
            # Store in ChromaDB
            doc_id = self.knowledge.store(user_input, category="user_note")
            # Also persist to about-me/notes.md so it survives restarts
            try:
                from journal import AboutMe
                # Extract the note content (strip the "remember that" prefix)
                note_text = user_input
                for prefix in ["remember that ", "remember ", "save this: ", "note that ", "add a note: ",
                                "make a note: ", "keep in mind ", "don't forget ", "jot this down: "]:
                    if note_text.lower().startswith(prefix):
                        note_text = note_text[len(prefix):]
                        break
                AboutMe.append_note(note_text)
            except Exception:
                pass
            return f"Got it, {Config.USER_NAME}. I'll remember that."

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

        import re as _re

        # ── PAUSE / STOP ──
        if any(kw in text for kw in ["pause", "stop music", "stop the music", "stop playing",
                                      "turn off the music", "turn off music", "mute the music", "mute music"]) \
                or _re.search(r'\b(stop|pause|mute)\b.*\b(music|song|track|playing)\b', text):
            return await self.actions.execute("music_pause", {})

        # ── SKIP / NEXT ──
        elif any(kw in text for kw in ["skip", "next song", "next track", "skip track",
                                        "skip this song", "skip song", "play next"]):
            return await self.actions.execute("music_skip", {})

        # ── PREVIOUS ──
        elif any(kw in text for kw in ["previous", "go back", "last song", "play previous",
                                        "previous song", "previous track"]):
            return await self.spotify.previous_track()

        # ── NOW PLAYING ──
        elif any(kw in text for kw in ["what's playing", "now playing", "what is playing",
                                        "current song", "current track", "what song"]):
            return await self.actions.execute("music_now_playing", {})

        # ── VOLUME ──
        elif any(kw in text for kw in ["volume", "turn up", "turn down", "louder", "quieter"]):
            m = _re.search(r'(\d+)', text)
            if m:
                return await self.spotify.set_volume(int(m.group(1)))
            elif any(w in text for w in ["up", "louder", "higher"]):
                return await self.spotify.set_volume(80)
            elif any(w in text for w in ["down", "quieter", "lower"]):
                return await self.spotify.set_volume(30)
            return await self.spotify.set_volume(50)

        # ── PLAY (must be last — broad match) ──
        elif "play" in text or "resume" in text or "continue" in text or "unpause" in text or "keep playing" in text:
            play_match = _re.search(r'play\s+(?:me\s+|us\s+)?(?:some\s+|the\s+|a\s+|today.?s?\s+)?(.+)', text)
            if play_match:
                query = play_match.group(1).strip()
                # Clean trailing filler words
                query = _re.sub(r'\s+(for me|for us|please|right now|now)$', '', query)
                if query and query not in ("music", "a music", "the music", "some music", "song", "a song", "songs", "me a song"):
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

    async def _handle_email(self, user_input: str, classification: dict) -> str:
        """Handle Gmail read/search/draft requests."""
        import re
        _inp = user_input.lower()

        # Draft / compose
        if any(kw in _inp for kw in ["draft", "compose", "write an email", "write email"]):
            # Try to extract to/subject/body from input via Claude
            context = (
                "The user wants to draft an email. Extract: to (email address), subject, body. "
                "If the user didn't specify a recipient, set to as empty. "
                "Return ONLY a JSON object: {\"to\": \"\", \"subject\": \"\", \"body\": \"\"}"
            )
            try:
                result = await self.claude.reason(user_input, context)
                import json
                params = json.loads(result)
                return await self.actions.execute("gmail_draft", params)
            except Exception:
                return await self.actions.execute("gmail_draft", {
                    "to": "", "subject": user_input, "body": ""
                })

        # Search
        if any(kw in _inp for kw in ["search", "find email", "look for email", "find my email"]):
            query_match = re.search(r'(?:search|find|look for)\s+(?:emails?\s+)?(?:about\s+|from\s+|for\s+)?(.+)', _inp)
            query = query_match.group(1).strip() if query_match else user_input
            return await self.actions.execute("gmail_search", {"query": query, "count": 5})

        # Default: read inbox
        count_match = re.search(r'(\d+)\s+(?:email|message)', _inp)
        count = int(count_match.group(1)) if count_match else 5
        return await self.actions.execute("gmail_inbox", {"count": count})

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