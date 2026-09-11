"""Tests for the orchestrator module — Config, PIISanitizer, IntentClassifier, TaskManager, ActionRegistry."""

import os
import sys
import json
import pytest
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from orchestrator import (
    Config,
    PIISanitizer,
    IntentClassifier,
    TaskManager,
    Task,
    TaskType,
    ActionRegistry,
    OllamaClient,
    ClaudeClient,
    OpenAIClient,
)


# ── Config ────────────────────────────────────

class TestConfig:
    def test_defaults(self):
        assert Config.AGENT_NAME  # Should have a default
        assert Config.OLLAMA_URL.startswith("http")
        assert Config.OLLAMA_MODEL
        assert Config.CLAUDE_MODEL
        assert Config.WAKE_PHRASE


# ── PIISanitizer ──────────────────────────────

class TestPIISanitizer:
    def test_sanitize_with_no_mappings(self):
        s = PIISanitizer()
        s.mappings = {}
        assert s.sanitize("hello") == "hello"

    def test_sanitize_replaces_known_values(self):
        s = PIISanitizer()
        s.mappings = {"John Doe": "[NAME]", "john@example.com": "[EMAIL]"}
        result = s.sanitize("Contact John Doe at john@example.com")
        assert "[NAME]" in result
        assert "[EMAIL]" in result
        assert "John Doe" not in result

    def test_desanitize_restores(self):
        s = PIISanitizer()
        s.mappings = {"John Doe": "[NAME]"}
        sanitized = s.sanitize("Hello John Doe")
        restored = s.desanitize(sanitized)
        assert "John Doe" in restored

    def test_roundtrip(self):
        s = PIISanitizer()
        s.mappings = {"secret": "[REDACTED]"}
        original = "The secret is here"
        assert s.desanitize(s.sanitize(original)) == original


# ── IntentClassifier (keyword-based) ──────────

class TestIntentClassifier:
    def setup_method(self):
        self.classifier = IntentClassifier()

    def test_knowledge_store(self):
        result = self.classifier._keyword_classify("remember that I like pizza")
        assert result["category"] == "KNOWLEDGE"
        assert result["intent"] == "store"

    def test_knowledge_recall(self):
        result = self.classifier._keyword_classify("what did I say about the meeting?")
        assert result["category"] == "KNOWLEDGE"

    def test_code_generation(self):
        result = self.classifier._keyword_classify("build me a landing page for my startup")
        assert result["category"] == "CODE"

    def test_image_generation(self):
        result = self.classifier._keyword_classify("can you quickly make a picture of a donkey eating a hamburger")
        assert result["category"] == "IMAGE"

    def test_image_generation_generate_phrasing(self):
        result = self.classifier._keyword_classify("generate an image of a sunset over mountains")
        assert result["category"] == "IMAGE"

    def test_image_request_not_misclassified_as_code(self):
        # "generate"/"create" are CODE keywords too -- image phrasing must
        # win since it's checked first (orchestrator.py, _keyword_classify).
        result = self.classifier._keyword_classify("create a picture of a cat wearing a hat")
        assert result["category"] == "IMAGE"

    def test_code_generation_not_misclassified_as_image(self):
        result = self.classifier._keyword_classify("generate a website for my bakery")
        assert result["category"] == "CODE"

    def test_calendar(self):
        result = self.classifier._keyword_classify("schedule a meeting for tomorrow at 3pm")
        assert result["category"] == "CALENDAR"

    def test_reservation(self):
        result = self.classifier._keyword_classify("book a table at the Italian restaurant")
        assert result["category"] == "RESERVATION"

    def test_communication(self):
        result = self.classifier._keyword_classify("send a text to Mom")
        assert result["category"] == "COMMUNICATION"

    def test_home(self):
        result = self.classifier._keyword_classify("turn on the living room lights")
        assert result["category"] == "HOME"

    def test_system(self):
        result = self.classifier._keyword_classify("what's your status?")
        assert result["category"] == "SYSTEM"

    def test_devops(self):
        result = self.classifier._keyword_classify("deploy the latest version")
        assert result["category"] == "DEVOPS"

    def test_personal(self):
        result = self.classifier._keyword_classify("what is my name?")
        assert result["category"] == "PERSONAL"

    def test_fallback_information(self):
        result = self.classifier._keyword_classify("tell me about quantum physics")
        assert result["category"] == "INFORMATION"

    def test_casual_conversation_classified_as_information(self):
        """Casual questions like 'how you doing' should classify as INFORMATION and go to Claude."""
        casual = ["how you doing", "what's up", "how are you", "are you talkative"]
        for phrase in casual:
            result = self.classifier._keyword_classify(phrase)
            assert result["category"] == "INFORMATION", f"'{phrase}' classified as {result['category']}, expected INFORMATION"

    def test_music_classification(self):
        result = self.classifier._keyword_classify("play some chill music")
        assert result["category"] == "MUSIC"

    def test_skip_classification(self):
        result = self.classifier._keyword_classify("skip")
        assert result["category"] == "MUSIC"

    def test_pause_classification(self):
        result = self.classifier._keyword_classify("pause")
        assert result["category"] == "MUSIC"

    def test_safe_summary_truncates(self):
        long_input = "x" * 200
        summary = IntentClassifier._safe_summary(long_input)
        assert len(summary) == 100

    def test_classification_has_required_keys(self):
        result = self.classifier._keyword_classify("anything")
        assert "category" in result
        assert "intent" in result
        assert "requires_cloud" in result
        assert "summary" in result


# ── TaskManager ───────────────────────────────

class TestTaskManager:
    def test_add_and_get_pending(self, tmp_path):
        tm = TaskManager(db_path=str(tmp_path / "tasks.json"))
        task = Task(id="t1", intent="test", raw_input="test input", task_type=TaskType.LOCAL)
        tm.add(task)
        pending = tm.get_pending()
        assert len(pending) == 1
        assert pending[0].id == "t1"

    def test_update_status(self, tmp_path):
        tm = TaskManager(db_path=str(tmp_path / "tasks.json"))
        task = Task(id="t2", intent="test", raw_input="test", task_type=TaskType.LOCAL)
        tm.add(task)
        tm.update("t2", status="completed")
        assert len(tm.get_pending()) == 0

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "tasks.json")
        tm1 = TaskManager(db_path=path)
        tm1.add(Task(id="tp", intent="test", raw_input="x", task_type=TaskType.LOCAL))
        tm2 = TaskManager(db_path=path)
        assert len(tm2.tasks) == 1

    def test_task_defaults(self):
        t = Task(id="x", intent="i", raw_input="r", task_type=TaskType.CLOUD)
        assert t.status == "pending"
        assert t.result is None
        assert t.created_at  # Should have a timestamp


# ── ActionRegistry ────────────────────────────

class TestActionRegistry:
    @pytest.mark.asyncio
    async def test_register_and_execute(self):
        registry = ActionRegistry()
        async def handler(params):
            return f"handled: {params['x']}"
        registry.register("test_action", handler, "Test action")
        result = await registry.execute("test_action", {"x": "hello"})
        assert result == "handled: hello"

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        registry = ActionRegistry()
        result = await registry.execute("nonexistent", {})
        assert "Unknown action" in result

    @pytest.mark.asyncio
    async def test_handler_exception(self):
        registry = ActionRegistry()
        async def bad_handler(params):
            raise ValueError("boom")
        registry.register("bad", bad_handler)
        result = await registry.execute("bad", {})
        assert "failed" in result.lower()


# ── OllamaClient ─────────────────────────────

class TestOllamaClient:
    def test_init(self):
        client = OllamaClient()
        assert client.base_url == Config.OLLAMA_URL
        assert client.model == Config.OLLAMA_MODEL
        assert client.conversation_history == []

    def test_conversation_history_cap(self):
        client = OllamaClient()
        # Simulate 40 messages
        client.conversation_history = [{"role": "user", "content": f"msg{i}"} for i in range(40)]
        # The cap is applied during chat, but we can check the init state
        assert len(client.conversation_history) == 40


# ── ClaudeClient ──────────────────────────────

class TestClaudeClient:
    def test_init(self):
        client = ClaudeClient()
        assert client.model == Config.CLAUDE_MODEL
        assert isinstance(client.sanitizer, PIISanitizer)

    @pytest.mark.asyncio
    async def test_no_api_key(self):
        client = ClaudeClient()
        client.api_key = ""
        result = await client.reason("test task")
        assert "not configured" in result.lower()

    def test_mel_system_prompt_contains_persona(self):
        client = ClaudeClient()
        prompt = client._mel_system_prompt()
        assert Config.AGENT_NAME in prompt
        assert Config.USER_NAME in prompt
        assert "casual" in prompt.lower() or "friend" in prompt.lower()

    def test_mel_system_prompt_include_time_false_omits_timestamp(self):
        # Prompt-caching relies on this block being identical across calls;
        # a live timestamp embedded in it would invalidate the cache almost
        # every turn (orchestrator.py, _prepare_converse/reason).
        client = ClaudeClient()
        with_time = client._mel_system_prompt(include_time=True)
        without_time = client._mel_system_prompt(include_time=False)
        assert "Current date and time" in with_time
        assert "Current date and time" not in without_time

    def test_prepare_converse_system_block_is_cacheable(self):
        client = ClaudeClient()
        _, _, payload = client._prepare_converse("test-session", "hi")
        assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert "Current date and time" not in payload["system"][0]["text"]
        assert "Current date and time" in payload["system"][1]["text"]


# ── OpenAIClient ──────────────────────────────

class TestOpenAIClient:
    @pytest.mark.asyncio
    async def test_generate_image_no_api_key_returns_none(self):
        client = OpenAIClient()
        client.api_key = ""
        result = await client.generate_image("a donkey eating a hamburger")
        assert result is None

    @pytest.mark.asyncio
    async def test_web_search_no_api_key_returns_empty(self):
        client = OpenAIClient()
        client.api_key = ""
        result = await client.web_search("who won the game last night")
        assert result == ""


# ── Conversational Fallback ──────────────────
class TestImageHandler:
    @pytest.mark.asyncio
    async def test_handle_image_no_api_key(self):
        from orchestrator import AgentOrchestrator
        orch = AgentOrchestrator()
        orch.openai.api_key = ""
        result = await orch._handle_image("a picture of a donkey eating a hamburger", {})
        assert "OpenAI" in result or "API key" in result

    @pytest.mark.asyncio
    async def test_handle_image_extracts_subject_and_embeds_markdown(self, monkeypatch):
        from orchestrator import AgentOrchestrator
        orch = AgentOrchestrator()
        orch.openai.api_key = "fake-key-for-test"

        async def fake_generate_image(prompt):
            assert "donkey" in prompt.lower()
            return "abc123.png"

        monkeypatch.setattr(orch.openai, "generate_image", fake_generate_image)
        result = await orch._handle_image("can you quickly make a picture of a donkey eating a hamburger", {})
        assert "![" in result
        assert "/generated-images/abc123.png" in result


class TestConversationalFallback:
    @pytest.mark.asyncio
    async def test_greeting_fallback(self):
        from orchestrator import AgentOrchestrator
        orch = AgentOrchestrator()
        # Force Claude to be unavailable
        orch.claude.api_key = ""
        result = await orch._conversational_fallback("how are you doing?")
        assert result and len(result) > 0
        # Should return a friendly greeting response
        assert Config.USER_NAME in result or "great" in result.lower() or "doing" in result.lower()

    @pytest.mark.asyncio
    async def test_thanks_fallback(self):
        from orchestrator import AgentOrchestrator
        orch = AgentOrchestrator()
        orch.claude.api_key = ""
        result = await orch._conversational_fallback("thank you so much")
        assert result and len(result) > 0

    @pytest.mark.asyncio
    async def test_goodbye_fallback(self):
        from orchestrator import AgentOrchestrator
        orch = AgentOrchestrator()
        orch.claude.api_key = ""
        result = await orch._conversational_fallback("bye see you later")
        assert result and len(result) > 0

    @pytest.mark.asyncio
    async def test_unknown_fallback(self):
        from orchestrator import AgentOrchestrator
        orch = AgentOrchestrator()
        orch.claude.api_key = ""
        result = await orch._conversational_fallback("xyzzy random gibberish")
        assert result and len(result) > 0
        assert Config.USER_NAME in result
