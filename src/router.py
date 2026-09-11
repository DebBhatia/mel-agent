"""
MEL AI AGENT - Model Router
============================
Decides which model backend should handle a conversational/reasoning
request. This is routing policy only -- it does not talk to any
provider itself (see model_gateway.py for that).

Policy: local (Ollama) by default -- keeps everyday chat private and
free. Escalate to Claude only for complex coding or reasoning work,
where local model quality isn't good enough. OpenAI-search routing is
a placeholder for a future phase (web-search-grounded answers already
have their own dedicated path in orchestrator.py's SEARCH handler --
this stub is for when that gets folded into general routing).
"""

from enum import Enum


class Backend(Enum):
    LOCAL = "local"
    CLAUDE = "claude"
    OPENAI_SEARCH = "openai_search"


# Categories the classifier can hand back that represent genuinely complex
# coding or reasoning work -- everything else defaults to local.
COMPLEX_CATEGORIES = {"CODE", "DEVOPS"}


class ModelRouter:
    """Chooses a Backend for a conversational/reasoning request.

    Feature-handler categories (CALENDAR, MUSIC, WEATHER, etc.) already
    dispatch to their own action handlers in orchestrator.py and never
    reach this router -- it only decides the backend for the general
    chat/reasoning path (today: process()'s default fallthrough and
    similar free-form requests).
    """

    def choose(self, category: str = "", classification: dict | None = None,
               user_input: str = "") -> Backend:
        classification = classification or {}

        if category in COMPLEX_CATEGORIES:
            return Backend.CLAUDE

        if category == "OPENAI_SEARCH":
            return Backend.OPENAI_SEARCH

        # classification.get("requires_cloud") is intentionally NOT consulted
        # here. IntentClassifier sets it as a blanket True for any
        # keyword-unmatched request (i.e. plain INFORMATION fallthrough) and
        # RESERVATION, and IntentClassifier.classify()'s Ollama path can echo
        # whatever value the local model guesses for it -- neither is a
        # reliable signal of actual complexity, and honoring it defeated the
        # local-first policy for ordinary conversation. Only an explicit
        # complex-coding/deep-reasoning category escalates to Claude; default
        # to local when in doubt.
        return Backend.LOCAL
