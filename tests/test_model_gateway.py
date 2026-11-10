"""Tests for model_gateway.py — ModelGateway dispatch to the right client,
and centralized cloud-context sanitization for non-local backends."""
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

    openai = AsyncMock()
    openai.web_search = AsyncMock(return_value="search answer")

    return ModelGateway(ollama, claude, openai), ollama, claude, openai


class TestModelGatewayGenerate:
    @pytest.mark.asyncio
    async def test_local_backend_calls_ollama(self):
        gateway, ollama, claude, openai = make_gateway()
        result = await gateway.generate(Backend.LOCAL, "sess", "hi", system_prompt="persona")
        assert result == "local reply"
        ollama.chat.assert_awaited_once_with("hi", system_prompt="persona")
        claude.converse.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_claude_backend_calls_claude(self):
        gateway, ollama, claude, openai = make_gateway()
        result = await gateway.generate(Backend.CLAUDE, "sess", "hi", extra_context="ctx")
        assert result == "claude reply"
        claude.converse.assert_awaited_once_with("sess", "hi", extra_context="ctx")
        ollama.chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_openai_search_backend_calls_openai_web_search(self):
        gateway, ollama, claude, openai = make_gateway()
        result = await gateway.generate(Backend.OPENAI_SEARCH, "sess", "latest news on rates")
        assert result == "search answer"
        openai.web_search.assert_awaited_once_with("latest news on rates")
        claude.converse.assert_not_awaited()
        ollama.chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_openai_search_with_no_openai_client_returns_empty(self):
        ollama = AsyncMock()
        claude = AsyncMock()
        gateway = ModelGateway(ollama, claude, openai_client=None)
        result = await gateway.generate(Backend.OPENAI_SEARCH, "sess", "search this")
        assert result == ""


class TestModelGatewayGenerateStream:
    @pytest.mark.asyncio
    async def test_local_backend_streams_from_ollama(self):
        gateway, ollama, claude, openai = make_gateway()
        chunks = [c async for c in gateway.generate_stream(Backend.LOCAL, "sess", "hi")]
        assert chunks == ["local ", "stream"]

    @pytest.mark.asyncio
    async def test_claude_backend_streams_from_claude(self):
        gateway, ollama, claude, openai = make_gateway()
        chunks = [c async for c in gateway.generate_stream(Backend.CLAUDE, "sess", "hi")]
        assert chunks == ["claude ", "stream"]

    @pytest.mark.asyncio
    async def test_openai_search_stream_yields_single_answer(self):
        gateway, ollama, claude, openai = make_gateway()
        chunks = [c async for c in gateway.generate_stream(Backend.OPENAI_SEARCH, "sess", "hi")]
        assert chunks == ["search answer"]

    @pytest.mark.asyncio
    async def test_openai_search_stream_with_no_openai_client_yields_nothing(self):
        ollama = AsyncMock()
        claude = AsyncMock()
        gateway = ModelGateway(ollama, claude, openai_client=None)
        chunks = [c async for c in gateway.generate_stream(Backend.OPENAI_SEARCH, "sess", "hi")]
        assert chunks == []


class TestCloudContextSanitizationAtGateway:
    """The gateway is the one centralized point every non-LOCAL call passes
    through -- sanitization must happen here regardless of caller."""

    @pytest.mark.asyncio
    async def test_claude_call_is_sanitized(self):
        gateway, ollama, claude, openai = make_gateway()
        await gateway.generate(
            Backend.CLAUDE, "sess",
            "email me at deb@example.com about this",
            extra_context="call me at 555-123-4567",
        )
        args, kwargs = claude.converse.call_args
        assert "deb@example.com" not in args[1]
        assert "[EMAIL_REDACTED]" in args[1]
        assert "555-123-4567" not in kwargs["extra_context"]

    @pytest.mark.asyncio
    async def test_openai_search_call_is_sanitized(self):
        gateway, ollama, claude, openai = make_gateway()
        await gateway.generate(Backend.OPENAI_SEARCH, "sess", "look up deb@example.com's account")
        sent_query = openai.web_search.call_args.args[0]
        assert "deb@example.com" not in sent_query
        assert "[EMAIL_REDACTED]" in sent_query

    @pytest.mark.asyncio
    async def test_local_call_is_not_sanitized(self):
        # Local never leaves the machine -- redacting it would just make
        # Mel talk in placeholders to the user for no privacy benefit.
        gateway, ollama, claude, openai = make_gateway()
        await gateway.generate(Backend.LOCAL, "sess", "email me at deb@example.com")
        args, kwargs = ollama.chat.call_args
        assert args[0] == "email me at deb@example.com"

    @pytest.mark.asyncio
    async def test_clean_text_is_unaffected_by_sanitization(self):
        gateway, ollama, claude, openai = make_gateway()
        await gateway.generate(Backend.CLAUDE, "sess", "hi", extra_context="ctx")
        claude.converse.assert_awaited_once_with("sess", "hi", extra_context="ctx")


class TestGenerateImage:
    @pytest.mark.asyncio
    async def test_image_prompt_is_sanitized(self):
        gateway, ollama, claude, openai = make_gateway()
        openai.generate_image = AsyncMock(return_value="img.png")
        result = await gateway.generate_image("a poster for bob@example.com")
        assert result == "img.png"
        sent = openai.generate_image.call_args.args[0]
        assert "bob@example.com" not in sent
        assert "EMAIL_REDACTED" in sent

    @pytest.mark.asyncio
    async def test_no_openai_client_returns_none(self):
        ollama, claude = AsyncMock(), AsyncMock()
        gateway = ModelGateway(ollama, claude, None)
        assert await gateway.generate_image("a donkey") is None
