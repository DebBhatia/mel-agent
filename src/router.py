"""
MEL AI AGENT - Model Router
============================
Decides which model backend should handle a conversational/reasoning
request. This is routing policy only -- it does not talk to any
provider itself (see model_gateway.py for that).

Policy: local (Ollama) by default -- keeps everyday chat private and
free. Escalate to Claude only when a deterministic complexity heuristic
fires on genuinely hard coding/software-engineering/reasoning work --
not just because the category happens to be CODE/DEVOPS. Escalate to
OpenAI only for SEARCH (live web / current-information lookups already
implemented in orchestrator.py's _handle_search -- this mapping keeps
the router's own policy correct and testable independent of that path).

No LLM call is used to decide any of this -- the heuristic below is a
plain keyword/regex check, deliberately conservative: when uncertain,
stay local.
"""

import re
from enum import Enum


class Backend(Enum):
    LOCAL = "local"
    CLAUDE = "claude"
    OPENAI_SEARCH = "openai_search"


# Categories where a complexity check (not blanket escalation) decides
# whether Claude is warranted. Being CODE/DEVOPS is necessary but no
# longer sufficient on its own.
CODING_CATEGORIES = {"CODE", "DEVOPS"}

# Deterministic complexity signals -- genuinely hard software-engineering
# work, not "build me a website" or "fix this typo". Conservative on
# purpose: a false negative just means an extra local attempt, a false
# positive means an unnecessary cloud call -- staying local is the safer
# default either way.
_COMPLEXITY_SIGNALS = re.compile(
    r"\b("
    r"refactor(?:ing)?|architect(?:ure)?|debug(?:ging)?\s+across|multi-?file\s+debug"
    r"|migrat(?:e|ion)|concurren(?:cy|t)|race\s+condition|deadlock"
    r"|distributed\s+system|performance\s+investigation|profil(?:e|ing)"
    r"|ci/?cd\s+redesign|pipeline\s+redesign|redesign\s+(?:the\s+|our\s+|my\s+)?(?:ci/?cd|pipeline)"
    r"|security\s+analysis|security\s+audit"
    r"|large[- ]scale\s+(?:code\s+)?generation|multi-?step\s+implementation"
    r"|repo(?:sitory)?[- ]wide\s+change|across\s+(?:the\s+)?(?:whole\s+)?(?:repo|codebase)"
    r")\b",
    re.IGNORECASE,
)


def _is_complex_task(user_input: str) -> bool:
    """Pure keyword/regex heuristic -- no model call. Deliberately narrow:
    only fires on explicit complexity signals, defaults to False (local)
    otherwise."""
    if not user_input:
        return False
    return bool(_COMPLEXITY_SIGNALS.search(user_input))


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

        if category in CODING_CATEGORIES:
            if _is_complex_task(user_input):
                return Backend.CLAUDE
            return Backend.LOCAL

        if category in ("SEARCH", "OPENAI_SEARCH"):
            return Backend.OPENAI_SEARCH

        # classification.get("requires_cloud") is intentionally NOT consulted
        # here. IntentClassifier sets it as a blanket True for any
        # keyword-unmatched request (i.e. plain INFORMATION fallthrough) and
        # RESERVATION, and IntentClassifier.classify()'s Ollama path can echo
        # whatever value the local model guesses for it -- neither is a
        # reliable signal of actual complexity, and honoring it defeated the
        # local-first policy for ordinary conversation. Only the explicit
        # complexity heuristic above (for CODE/DEVOPS) or SEARCH escalates;
        # default to local when in doubt.
        return Backend.LOCAL
