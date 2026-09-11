"""Tests for model_gateway.py — ModelGateway dispatch to the right client."""
import os
import sys
import pytest
from unittest.mock import AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from router import Backend
from model_gateway import ModelGateway


def make_gateway():
    ollama = AsyncMock()
    ollama.chat = AsyncMock(return_value="local reply")

    async def _ollama_stream(message, system_prompt=None):
        for tok in ["local ", "stream"]:
            yield tok
    ollama.chat_stream = _ollama_stream

    claude = AsyncMock()
    claude.converse = AsyncMock(return_value="claude reply")

    async def _claude_stream(session_id, message):
        for tok in ["claude ", "stream"]:
            yield tok
    claude.converse_stream = _claude_stream

    return ModelGateway(ollama, claude), ollama, claude


class TestModelGatewayGenerate:
    @pytest.mark.asyncio
    async def test_local_backend_calls_ollama(self):
        gateway, ollama, claude = make_gateway()
        result = await gateway.generate(Backend.LOCAL, "sess", "hi", system_prompt="persona")
        assert result == "local reply"
        ollama.chat.assert_awaited_once_with("hi", system_prompt="persona")
        claude.converse.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_claude_backend_calls_claude(self):
        gateway, ollama, claude = make_gateway()
        result = await gateway.generate(Backend.CLAUDE, "sess", "hi", extra_context="ctx")
        assert result == "claude reply"
        claude.converse.assert_awaited_once_with("sess", "hi", extra_context="ctx")
        ollama.chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_openai_search_backend_not_implemented(self):
        gateway, _, _ = make_gateway()
        with pytest.raises(NotImplementedError):
            await gateway.generate(Backend.OPENAI_SEARCH, "sess", "search this")


class TestModelGatewayGenerateStream:
    @pytest.mark.asyncio
    async def test_local_backend_streams_from_ollama(self):
        gateway, _, _ = make_gateway()
        chunks = [c async for c in gateway.generate_stream(Backend.LOCAL, "sess", "hi")]
        assert chunks == ["local ", "stream"]

    @pytest.mark.asyncio
    async def test_claude_backend_streams_from_claude(self):
        gateway, _, _ = make_gateway()
        chunks = [c async for c in gateway.generate_stream(Backend.CLAUDE, "sess", "hi")]
        assert chunks == ["claude ", "stream"]

    @pytest.mark.asyncio
    async def test_openai_search_stream_not_implemented(self):
        gateway, _, _ = make_gateway()
        with pytest.raises(NotImplementedError):
            async for _ in gateway.generate_stream(Backend.OPENAI_SEARCH, "sess", "hi"):
                pass
