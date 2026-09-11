"""Tests for multi_agent.py — registry, selection, execution flow, non-escalation."""
import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from router import Backend, ModelRouter
from multi_agent import (
    AgentDefinition,
    AgentTaskContext,
    AgentRegistry,
    build_default_registry,
    select_agent,
    MultiAgentManager,
    UnknownSpecialistError,
)


EXPECTED_SPECIALISTS = {"general", "email", "calendar", "research", "coding", "home", "memory"}


# ── Registry ──────────────────────────────────

class TestAgentRegistry:
    def test_default_registry_has_seven_specialists(self):
        registry = build_default_registry()
        assert set(registry.names()) == EXPECTED_SPECIALISTS

    def test_all_specialists_default_to_local_backend(self):
        registry = build_default_registry()
        for agent in registry.all():
            assert agent.default_backend == Backend.LOCAL, f"{agent.name} should default to LOCAL"

    def test_coding_specialist_defaults_local_not_claude(self):
        # Explicit check per spec: coding included in "all local by default".
        registry = build_default_registry()
        coding = registry.get("coding")
        assert coding.default_backend == Backend.LOCAL

    def test_each_specialist_has_required_fields(self):
        registry = build_default_registry()
        for agent in registry.all():
            assert agent.name
            assert agent.description
            assert agent.system_prompt
            assert isinstance(agent.capabilities, list)
            assert isinstance(agent.allowed_tools, list)

    def test_get_unknown_agent_returns_none(self):
        registry = build_default_registry()
        assert registry.get("nonexistent") is None

    def test_register_and_get_custom_agent(self):
        registry = AgentRegistry()
        defn = AgentDefinition(name="custom", description="d", system_prompt="p")
        registry.register(defn)
        assert registry.get("custom") is defn


# ── Selection ─────────────────────────────────

class TestSelectAgent:
    @pytest.mark.parametrize("category,expected", [
        ("CALENDAR", "calendar"),
        ("EMAIL", "email"),
        ("HOME", "home"),
        ("CODE", "coding"),
        ("DEVOPS", "coding"),
        ("KNOWLEDGE", "memory"),
        ("SEARCH", "research"),
    ])
    def test_known_category_maps_to_expected_specialist(self, category, expected):
        assert select_agent(category, {}) == expected

    @pytest.mark.parametrize("category", [
        "INFORMATION", "PERSONAL", "SYSTEM", "RESERVATION", "COMMUNICATION",
        "MUSIC", "WEATHER", "REMINDER", "ROUTINE", "NOTIFICATION", "DESIGN", "IMAGE", "UNKNOWN_CATEGORY",
    ])
    def test_unmapped_category_defaults_to_general(self, category):
        assert select_agent(category, {}) == "general"

    def test_selection_ignores_requires_cloud_flag(self):
        # Selection must not be swayed by requires_cloud -- it's a name lookup only.
        assert select_agent("INFORMATION", {"requires_cloud": True}) == "general"
        assert select_agent("CALENDAR", {"requires_cloud": True}) == "calendar"

    def test_selection_is_deterministic(self):
        results = {select_agent("CODE", {}) for _ in range(20)}
        assert results == {"coding"}

    def test_selection_returns_string_not_backend(self):
        result = select_agent("CODE", {})
        assert isinstance(result, str)
        assert not isinstance(result, Backend)

    def test_none_classification_does_not_error(self):
        assert select_agent("HOME", None) == "home"


# ── Task context isolation ────────────────────

class TestAgentTaskContext:
    def test_scoped_session_id_is_namespaced(self):
        task = AgentTaskContext(agent_name="coding", session_id="session:abc", user_input="hi")
        assert task.scoped_session_id == "agent:coding:session:abc"

    def test_different_agents_get_different_scoped_ids_for_same_session(self):
        t1 = AgentTaskContext(agent_name="coding", session_id="s1", user_input="hi")
        t2 = AgentTaskContext(agent_name="email", session_id="s1", user_input="hi")
        assert t1.scoped_session_id != t2.scoped_session_id

    def test_extra_context_defaults_empty(self):
        task = AgentTaskContext(agent_name="general", session_id="s1", user_input="hi")
        assert task.extra_context == ""


# ── Execution flow ─────────────────────────────

def make_manager():
    registry = build_default_registry()
    router = ModelRouter()
    gateway = AsyncMock()
    gateway.generate = AsyncMock(return_value="specialist reply")

    async def _stream(backend, session_id, message, system_prompt=None):
        for tok in ["spec", "ialist"]:
            yield tok
    gateway.generate_stream = _stream

    manager = MultiAgentManager(registry, router, gateway)
    return manager, registry, router, gateway


class TestMultiAgentManagerExecute:
    @pytest.mark.asyncio
    async def test_execute_calls_gateway_with_specialist_system_prompt(self):
        manager, registry, router, gateway = make_manager()
        result = await manager.execute("email", "EMAIL", {"requires_cloud": False}, "draft a note", "sess-1")
        assert result == "specialist reply"
        gateway.generate.assert_awaited_once()
        args, kwargs = gateway.generate.call_args
        assert kwargs["system_prompt"] == registry.get("email").system_prompt

    @pytest.mark.asyncio
    async def test_execute_uses_scoped_session_id_not_raw_session_id(self):
        manager, registry, router, gateway = make_manager()
        await manager.execute("coding", "CODE", {}, "write a function", "raw-session")
        args, kwargs = gateway.generate.call_args
        scoped_session_id_arg = args[1]
        assert scoped_session_id_arg == "agent:coding:raw-session"
        assert scoped_session_id_arg != "raw-session"

    @pytest.mark.asyncio
    async def test_execute_calls_gateway_exactly_once(self):
        manager, registry, router, gateway = make_manager()
        await manager.execute("general", "INFORMATION", {}, "hi", "sess-1")
        assert gateway.generate.await_count == 1

    @pytest.mark.asyncio
    async def test_unknown_agent_name_raises_explicit_error(self):
        # An invalid agent_name passed explicitly is a caller bug -- it must
        # fail loudly, not silently fall back to 'general'.
        manager, registry, router, gateway = make_manager()
        with pytest.raises(UnknownSpecialistError):
            await manager.execute("does-not-exist", "INFORMATION", {}, "hi", "sess-1")
        gateway.generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_agent_name_stream_raises_explicit_error(self):
        manager, registry, router, gateway = make_manager()
        with pytest.raises(UnknownSpecialistError):
            async for _ in manager.execute_stream("does-not-exist", "INFORMATION", {}, "hi", "sess-1"):
                pass

    @pytest.mark.asyncio
    async def test_execute_stream_yields_tokens_from_gateway(self):
        manager, registry, router, gateway = make_manager()
        chunks = [c async for c in manager.execute_stream("research", "SEARCH", {}, "look this up", "sess-1")]
        assert chunks == ["spec", "ialist"]

    def test_manager_does_not_construct_its_own_clients(self):
        # Manager must be a pure pass-through onto the router/gateway it's
        # given -- no OllamaClient/ClaudeClient import or construction here.
        import multi_agent
        import inspect
        source = inspect.getsource(multi_agent)
        assert "OllamaClient(" not in source
        assert "ClaudeClient(" not in source
        assert "httpx.AsyncClient(" not in source


# ── Non-escalation ─────────────────────────────

class TestNonEscalation:
    """Backend choice must come solely from the real ModelRouter, and only
    CODE/DEVOPS escalate -- agent identity itself must never imply Claude."""

    @pytest.mark.asyncio
    async def test_non_coding_specialists_stay_local_even_with_requires_cloud_true(self):
        manager, registry, router, gateway = make_manager()
        for agent_name, category in [
            ("email", "EMAIL"), ("calendar", "CALENDAR"), ("home", "HOME"),
            ("memory", "KNOWLEDGE"), ("research", "SEARCH"), ("general", "INFORMATION"),
        ]:
            gateway.generate.reset_mock()
            await manager.execute(agent_name, category, {"requires_cloud": True}, "input", "sess")
            backend_arg = gateway.generate.call_args.args[0]
            assert backend_arg == Backend.LOCAL, f"{agent_name} ({category}) should stay LOCAL"

    @pytest.mark.asyncio
    async def test_coding_specialist_still_escalates_via_category_not_identity(self):
        # CODE category escalates because ModelRouter.choose("CODE", ...) says
        # so -- not because the "coding" AgentDefinition carries any special
        # backend. Confirm by checking a real ModelRouter instance is used.
        manager, registry, router, gateway = make_manager()
        assert registry.get("coding").default_backend == Backend.LOCAL
        await manager.execute("coding", "CODE", {}, "refactor this function", "sess")
        backend_arg = gateway.generate.call_args.args[0]
        assert backend_arg == Backend.CLAUDE

    @pytest.mark.asyncio
    async def test_selection_call_alone_never_touches_backend(self):
        # select_agent() has no Backend-typed return path at all.
        for category in ["CODE", "DEVOPS", "CALENDAR", "EMAIL", "HOME", "KNOWLEDGE", "SEARCH", "INFORMATION"]:
            result = select_agent(category, {"requires_cloud": True})
            assert isinstance(result, str)
