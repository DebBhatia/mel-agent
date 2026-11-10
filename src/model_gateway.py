"""
MEL AI AGENT - Model Gateway
=============================
Single seam between orchestrator.py's routing logic and the actual model
provider clients. Wraps the existing OllamaClient/ClaudeClient/OpenAIClient
instances (does not construct its own) so PII sanitization, prompt
caching, and per-session history already implemented on ClaudeClient keep
working unchanged -- this is a thin dispatch layer, not a reimplementation.

Every non-LOCAL call (CLAUDE, OPENAI_SEARCH) is sanitized and size-capped
via cloud_context.CloudContextPolicy before it leaves this process -- this
is the one centralized enforcement point for all current and future
callers of the gateway, local calls are left untouched since they never
leave the machine.
"""

import logging

from router import Backend
from cloud_context import CloudContextPolicy

logger = logging.getLogger("model_gateway")


class ModelGateway:
    """Routes a generate/generate_stream call to the backend chosen by ModelRouter."""

    def __init__(self, ollama_client, claude_client, openai_client=None):
        self.ollama = ollama_client
        self.claude = claude_client
        self.openai = openai_client
        self._cloud_policy = CloudContextPolicy()

    async def generate(self, backend: Backend, session_id: str, message: str,
                        system_prompt: str = None, extra_context: str = "") -> str:
        if backend == Backend.CLAUDE:
            safe_message, _ = self._cloud_policy.sanitize(message)
            safe_context, _ = self._cloud_policy.sanitize(extra_context) if extra_context else ("", [])
            return await self.claude.converse(session_id, safe_message, extra_context=safe_context)

        if backend == Backend.OPENAI_SEARCH:
            if self.openai is None:
                return ""
            safe_message, _ = self._cloud_policy.sanitize(message)
            return await self.openai.web_search(safe_message)

        # Backend.LOCAL (default) -- never leaves the machine, not sanitized
        return await self.ollama.chat(message, system_prompt=system_prompt)

    async def generate_stream(self, backend: Backend, session_id: str, message: str,
                               system_prompt: str = None):
        if backend == Backend.CLAUDE:
            safe_message, _ = self._cloud_policy.sanitize(message)
            async for token in self.claude.converse_stream(session_id, safe_message):
                yield token
            return

        if backend == Backend.OPENAI_SEARCH:
            if self.openai is None:
                return
            safe_message, _ = self._cloud_policy.sanitize(message)
            answer = await self.openai.web_search(safe_message)
            if answer:
                yield answer
            return

        # Backend.LOCAL (default) -- never leaves the machine, not sanitized
        async for token in self.ollama.chat_stream(message, system_prompt=system_prompt):
            yield token

    async def generate_structured(self, backend: Backend, session_id: str, message: str,
                                   system_prompt: str, max_tokens: int = 1024) -> str:
        """One-shot structured completion with a fully caller-supplied system
        prompt and token budget -- for non-conversational cloud tasks (e.g.
        code generation) that must not use ClaudeClient's Mel-persona system
        prompt or accumulate turn history. Still goes through the same
        CloudContextPolicy sanitization as generate() for any non-LOCAL
        backend. Callers (e.g. codegen.py via the orchestrator) never talk
        to a provider client directly -- this is the only entry point."""
        if backend == Backend.CLAUDE:
            safe_message, _ = self._cloud_policy.sanitize(message)
            return await self.claude.complete(session_id, safe_message, system_prompt, max_tokens=max_tokens)

        if backend == Backend.OPENAI_SEARCH:
            raise NotImplementedError("OPENAI_SEARCH does not support structured generation")

        # Backend.LOCAL (default) -- never leaves the machine, not sanitized
        return await self.ollama.chat(message, system_prompt=system_prompt)

    async def generate_image(self, prompt: str) -> str | None:
        """Image generation via OpenAI, with the prompt sanitized/capped the
        same as every other cloud call. Returns the saved filename, or None
        when OpenAI isn't configured or generation failed."""
        if self.openai is None:
            return None
        safe_prompt, _ = self._cloud_policy.sanitize(prompt)
        return await self.openai.generate_image(safe_prompt)
