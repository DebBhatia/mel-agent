"""
MEL AI AGENT - Model Gateway
=============================
Single seam between orchestrator.py's routing logic and the actual model
provider clients. Wraps the existing OllamaClient/ClaudeClient instances
(does not construct its own) so PII sanitization, prompt caching, and
per-session history already implemented on ClaudeClient keep working
unchanged -- this is a thin dispatch layer, not a reimplementation.
"""

import logging

from router import Backend

logger = logging.getLogger("model_gateway")


class ModelGateway:
    """Routes a generate/generate_stream call to the backend chosen by ModelRouter."""

    def __init__(self, ollama_client, claude_client, openai_client=None):
        self.ollama = ollama_client
        self.claude = claude_client
        self.openai = openai_client

    async def generate(self, backend: Backend, session_id: str, message: str,
                        system_prompt: str = None, extra_context: str = "") -> str:
        if backend == Backend.CLAUDE:
            return await self.claude.converse(session_id, message, extra_context=extra_context)

        if backend == Backend.OPENAI_SEARCH:
            raise NotImplementedError(
                "OpenAI-search routing is not implemented yet -- SEARCH requests "
                "are handled separately by orchestrator._handle_search()."
            )

        # Backend.LOCAL (default)
        return await self.ollama.chat(message, system_prompt=system_prompt)

    async def generate_stream(self, backend: Backend, session_id: str, message: str,
                               system_prompt: str = None):
        if backend == Backend.CLAUDE:
            async for token in self.claude.converse_stream(session_id, message):
                yield token
            return

        if backend == Backend.OPENAI_SEARCH:
            raise NotImplementedError(
                "OpenAI-search routing is not implemented yet -- SEARCH requests "
                "are handled separately by orchestrator._handle_search()."
            )

        # Backend.LOCAL (default)
        async for token in self.ollama.chat_stream(message, system_prompt=system_prompt):
            yield token
