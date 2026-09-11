"""
MEL AI AGENT - Multi-Agent Runtime
====================================
Lightweight local multi-agent layer. Specialists are role/context wrappers
around the same local Ollama model, not separate processes or model
servers -- there is no new daemon and no new provider client here.
Execution always goes through the existing ModelRouter/ModelGateway from
router.py/model_gateway.py, so backend policy (local-by-default, Claude
only for explicit complex-coding/deep-reasoning categories) is inherited
unchanged, not reimplemented.

Selection is intentionally independent of backend choice: select_agent()
picks a specialist *identity* (persona, description, capabilities) via a
pure category lookup, never a Backend. The only thing that ever escalates
a request to Claude is ModelRouter.choose(), called inside execute() --
exactly the same call site phase one already tests.
"""

from dataclasses import dataclass, field

from router import Backend


# ─────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    system_prompt: str
    capabilities: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    # Metadata/fallback hint only -- never consulted by execute() when a
    # category is available. ModelRouter.choose() is the sole source of
    # truth for backend selection. Always LOCAL: multi-agent does not
    # mean cloud.
    default_backend: Backend = Backend.LOCAL


# ─────────────────────────────────────────────
# Isolated per-task context
# ─────────────────────────────────────────────
@dataclass
class AgentTaskContext:
    agent_name: str
    session_id: str
    user_input: str
    extra_context: str = ""

    @property
    def scoped_session_id(self) -> str:
        """Namespaced session id so a specialist's turn history (kept inside
        ClaudeClient.conversation_history, keyed by session id) never mixes
        with Mel's main conversation or another specialist's -- this is
        isolation via the existing per-session dict, not a new mechanism.
        Full private history is not passed by default: a specialist only
        ever sees its own prior turns under this namespaced key, starting
        empty on first use."""
        return f"agent:{self.agent_name}:{self.session_id}"


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────
class AgentRegistry:
    """Holds registered AgentDefinitions by name."""

    def __init__(self):
        self._agents: dict[str, AgentDefinition] = {}

    def register(self, definition: AgentDefinition):
        self._agents[definition.name] = definition

    def get(self, name: str) -> AgentDefinition | None:
        return self._agents.get(name)

    def all(self) -> list[AgentDefinition]:
        return list(self._agents.values())

    def names(self) -> list[str]:
        return list(self._agents.keys())


def build_default_registry() -> AgentRegistry:
    """Registers the seven initial specialists. All default_backend=LOCAL --
    coding included. The router (not agent identity) decides when a CODE/
    DEVOPS request actually needs Claude."""
    registry = AgentRegistry()

    registry.register(AgentDefinition(
        name="general",
        description="Default conversational specialist for everyday chat, small talk, and information questions.",
        system_prompt=(
            "You're Mel's general specialist -- casual, direct, friendly conversation. "
            "Handle small talk, opinions, and general questions naturally."
        ),
        capabilities=["conversation", "small_talk", "general_information"],
        allowed_tools=[],
    ))

    registry.register(AgentDefinition(
        name="email",
        description="Email-domain specialist -- drafting, summarizing, and reasoning about email content.",
        system_prompt=(
            "You're Mel's email specialist. Help compose, summarize, and reason about email "
            "content clearly and concisely."
        ),
        capabilities=["email_drafting", "email_summarization"],
        allowed_tools=["gmail_draft", "gmail_search", "gmail_inbox"],
    ))

    registry.register(AgentDefinition(
        name="calendar",
        description="Calendar-domain specialist -- scheduling reasoning and event-related conversation.",
        system_prompt=(
            "You're Mel's calendar specialist. Help reason about scheduling, availability, "
            "and event details clearly."
        ),
        capabilities=["scheduling_reasoning", "event_conversation"],
        allowed_tools=["calendar_list", "calendar_create", "calendar_delete", "calendar_update"],
    ))

    registry.register(AgentDefinition(
        name="research",
        description="Research specialist -- answering open-ended questions that need synthesis, not a stored action.",
        system_prompt=(
            "You're Mel's research specialist. Answer open-ended questions clearly and factually, "
            "synthesizing what you know rather than guessing."
        ),
        capabilities=["research", "synthesis"],
        allowed_tools=[],
    ))

    registry.register(AgentDefinition(
        name="coding",
        description="Coding specialist -- complex coding and deep technical reasoning.",
        system_prompt=(
            "You're Mel's coding specialist. Reason carefully about code, architecture, and "
            "technical tradeoffs. Be precise and correct over fast."
        ),
        capabilities=["coding", "technical_reasoning"],
        allowed_tools=[],
        default_backend=Backend.LOCAL,
    ))

    registry.register(AgentDefinition(
        name="home",
        description="Smart-home specialist -- device and routine conversation.",
        system_prompt=(
            "You're Mel's smart-home specialist. Help reason about devices, scenes, and "
            "home routines clearly."
        ),
        capabilities=["smart_home_reasoning"],
        allowed_tools=["home_control"],
    ))

    registry.register(AgentDefinition(
        name="memory",
        description="Memory specialist -- recall and reasoning over previously stored knowledge.",
        system_prompt=(
            "You're Mel's memory specialist. Help recall and reason about things previously "
            "stored or discussed, without inventing details you don't actually have."
        ),
        capabilities=["recall", "knowledge_reasoning"],
        allowed_tools=["knowledge_store", "knowledge_recall"],
    ))

    return registry


# ─────────────────────────────────────────────
# Deterministic selection (no LLM, no Claude, no backend implication)
# ─────────────────────────────────────────────
_CATEGORY_TO_AGENT = {
    "CALENDAR": "calendar",
    "EMAIL": "email",
    "HOME": "home",
    "CODE": "coding",
    "DEVOPS": "coding",
    "KNOWLEDGE": "memory",
    "SEARCH": "research",
}


def select_agent(category: str, classification: dict | None = None) -> str:
    """Pure category lookup -- returns a specialist name only. Never
    returns or implies a Backend; that's ModelRouter's job, decided
    separately inside execute()."""
    return _CATEGORY_TO_AGENT.get(category, "general")


class UnknownSpecialistError(ValueError):
    """Raised by execute()/execute_stream() when given an agent_name that
    isn't registered. This is a caller bug (something passed a bad name
    explicitly), not an unresolved user intent -- select_agent() already
    handles the "don't know what the user meant" case by returning
    'general', so execute()/execute_stream() must not paper over a bad
    explicit name with the same fallback."""


# ─────────────────────────────────────────────
# Manager: Mel delegates execution here for the general/reasoning path
# ─────────────────────────────────────────────
class MultiAgentManager:
    """Ties the registry to the *existing* router/gateway. Does not
    construct any client itself -- reuses the ModelRouter/ModelGateway
    instances the orchestrator already owns."""

    def __init__(self, registry: AgentRegistry, router, gateway):
        self.registry = registry
        self.router = router
        self.gateway = gateway

    async def execute(self, agent_name: str, category: str, classification: dict,
                       user_input: str, session_id: str) -> str:
        agent = self.registry.get(agent_name)
        if agent is None:
            raise UnknownSpecialistError(f"No specialist registered as '{agent_name}'")

        task = AgentTaskContext(agent_name=agent.name, session_id=session_id, user_input=user_input)

        # Backend choice is entirely the router's call -- unchanged from
        # phase one, not re-decided here based on agent identity.
        backend = self.router.choose(category, classification, user_input)

        return await self.gateway.generate(
            backend, task.scoped_session_id, task.user_input,
            system_prompt=agent.system_prompt,
            extra_context=task.extra_context,
        )

    async def execute_stream(self, agent_name: str, category: str, classification: dict,
                              user_input: str, session_id: str):
        agent = self.registry.get(agent_name)
        if agent is None:
            raise UnknownSpecialistError(f"No specialist registered as '{agent_name}'")

        task = AgentTaskContext(agent_name=agent.name, session_id=session_id, user_input=user_input)
        backend = self.router.choose(category, classification, user_input)

        async for token in self.gateway.generate_stream(
            backend, task.scoped_session_id, task.user_input,
            system_prompt=agent.system_prompt,
        ):
            yield token
