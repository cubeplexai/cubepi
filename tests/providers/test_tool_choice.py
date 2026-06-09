"""Tests for tool_choice wire format mapping across all providers.

These tests verify that each provider's ``_map_tool_choice`` static method
correctly maps CubePi's ``ToolChoice`` values to provider-specific wire
formats, and that ``stream()`` actually forwards tool_choice to the API.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from cubepi.providers.anthropic import AnthropicProvider
from cubepi.providers.base import Model, ToolDefinition, UserMessage, TextContent
from cubepi.providers.faux import FauxProvider, faux_assistant_message
from cubepi.providers.openai import OpenAIProvider
from cubepi.providers.openai_responses import OpenAIResponsesProvider


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class TestAnthropicToolChoice:
    """AnthropicProvider._map_tool_choice maps ToolChoice to Anthropic wire format."""

    def test_auto_maps_to_type_auto(self):
        result = AnthropicProvider._map_tool_choice("auto")
        assert result == {"type": "auto"}

    def test_required_maps_to_type_any(self):
        result = AnthropicProvider._map_tool_choice("required")
        assert result == {"type": "any"}

    def test_none_maps_to_python_none(self):
        result = AnthropicProvider._map_tool_choice("none")
        assert result is None

    def test_named_tool_maps_to_type_tool_with_name(self):
        result = AnthropicProvider._map_tool_choice("structured_output")
        assert result == {"type": "tool", "name": "structured_output"}


# ---------------------------------------------------------------------------
# OpenAI (Chat Completions)
# ---------------------------------------------------------------------------


class TestOpenAIToolChoice:
    """OpenAIProvider._map_tool_choice maps ToolChoice to OpenAI wire format."""

    def test_auto_maps_to_string_auto(self):
        result = OpenAIProvider._map_tool_choice("auto")
        assert result == "auto"

    def test_required_maps_to_string_required(self):
        result = OpenAIProvider._map_tool_choice("required")
        assert result == "required"

    def test_none_maps_to_string_none(self):
        result = OpenAIProvider._map_tool_choice("none")
        assert result == "none"

    def test_named_tool_maps_to_function_object(self):
        result = OpenAIProvider._map_tool_choice("structured_output")
        assert result == {"type": "function", "function": {"name": "structured_output"}}


# ---------------------------------------------------------------------------
# OpenAI Responses
# ---------------------------------------------------------------------------


class TestOpenAIResponsesToolChoice:
    """OpenAIResponsesProvider._map_tool_choice maps ToolChoice to Responses API wire format."""

    def test_required_maps_to_string_required(self):
        result = OpenAIResponsesProvider._map_tool_choice("required")
        assert result == "required"

    def test_named_tool_maps_to_function_object(self):
        result = OpenAIResponsesProvider._map_tool_choice("structured_output")
        assert result == {"type": "function", "name": "structured_output"}


# ---------------------------------------------------------------------------
# FauxProvider — integration test
# ---------------------------------------------------------------------------


class TestFauxProviderToolChoice:
    """FauxProvider accepts tool_choice without error (integration test)."""

    def _make_model(self) -> Model:
        return Model(id="faux-1", provider_id="faux")

    async def test_generate_accepts_tool_choice_auto(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("ok")])
        model = provider.model("faux-1")

        result = await model.generate(
            [UserMessage(content=[TextContent(text="hello")])],
            tool_choice="auto",
        )

        assert result.stop_reason == "stop"

    async def test_generate_accepts_tool_choice_required(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("ok")])
        model = provider.model("faux-1")

        result = await model.generate(
            [UserMessage(content=[TextContent(text="hello")])],
            tool_choice="required",
        )

        assert result.stop_reason == "stop"

    async def test_generate_accepts_tool_choice_none(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("ok")])
        model = provider.model("faux-1")

        result = await model.generate(
            [UserMessage(content=[TextContent(text="hello")])],
            tool_choice="none",
        )

        assert result.stop_reason == "stop"

    async def test_generate_accepts_named_tool_choice(self):
        provider = FauxProvider()
        provider.set_responses([faux_assistant_message("ok")])
        model = provider.model("faux-1")

        result = await model.generate(
            [UserMessage(content=[TextContent(text="hello")])],
            tool_choice="structured_output",
        )

        assert result.stop_reason == "stop"


# ---------------------------------------------------------------------------
# Anthropic stream() forwards tool_choice to SDK kwargs
# ---------------------------------------------------------------------------

_TOOL = ToolDefinition(
    name="search",
    description="Search",
    parameters={"type": "object", "properties": {}},
)

_MSGS = [UserMessage(content=[TextContent(text="hi")])]


def _anthropic_model() -> Model:
    return Model(
        id="claude-sonnet-4-20250514", provider_id="anthropic", api="anthropic"
    )


class _MockAnthropicStream:
    def __init__(self) -> None:
        self.response = SimpleNamespace(
            status_code=200, headers={"x-request-id": "req-abc"}
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self._events()

    async def _events(self):
        return
        yield  # noqa: RET504  — make this an async generator

    async def get_final_message(self):
        return SimpleNamespace(
            id="msg_1",
            content=[],
            stop_reason="end_turn",
            usage=SimpleNamespace(
                input_tokens=1,
                output_tokens=1,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
        )


class TestAnthropicStreamToolChoice:
    async def test_stream_forwards_required(self):
        provider = AnthropicProvider(api_key="test-key", cache_retention="none")
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=_MockAnthropicStream())
        provider._client = mock_client

        ms = await provider.stream(
            _anthropic_model(), _MSGS, tools=[_TOOL], tool_choice="required"
        )
        async for _ in ms:
            pass
        await ms.result()

        call_kwargs = mock_client.messages.stream.call_args[1]
        assert call_kwargs["tool_choice"] == {"type": "any"}

    async def test_stream_forwards_named_tool(self):
        provider = AnthropicProvider(api_key="test-key", cache_retention="none")
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=_MockAnthropicStream())
        provider._client = mock_client

        ms = await provider.stream(
            _anthropic_model(), _MSGS, tools=[_TOOL], tool_choice="search"
        )
        async for _ in ms:
            pass
        await ms.result()

        call_kwargs = mock_client.messages.stream.call_args[1]
        assert call_kwargs["tool_choice"] == {"type": "tool", "name": "search"}

    async def test_stream_none_omits_tool_choice(self):
        provider = AnthropicProvider(api_key="test-key", cache_retention="none")
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=_MockAnthropicStream())
        provider._client = mock_client

        ms = await provider.stream(
            _anthropic_model(), _MSGS, tools=[_TOOL], tool_choice="none"
        )
        async for _ in ms:
            pass
        await ms.result()

        call_kwargs = mock_client.messages.stream.call_args[1]
        assert "tool_choice" not in call_kwargs
        assert "tools" not in call_kwargs


# ---------------------------------------------------------------------------
# OpenAI stream() forwards tool_choice to SDK kwargs
# ---------------------------------------------------------------------------


def _openai_model() -> Model:
    return Model(id="gpt-4o", provider_id="openai", api="openai")


async def _async_iter(items):
    for item in items:
        yield item


def _make_chunk(content=None, finish_reason=None, id="chatcmpl-1"):
    choice = SimpleNamespace(
        index=0,
        delta=SimpleNamespace(content=content, tool_calls=None, role=None),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(id=id, choices=[choice])


class TestOpenAIStreamToolChoice:
    async def test_stream_forwards_required(self):
        chunks = [_make_chunk(content="hi"), _make_chunk(finish_reason="stop")]
        provider = OpenAIProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_iter(chunks)
        )
        provider._client = mock_client

        ms = await provider.stream(
            _openai_model(), _MSGS, tools=[_TOOL], tool_choice="required"
        )
        async for _ in ms:
            pass
        await ms.result()

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["tool_choice"] == "required"


# ---------------------------------------------------------------------------
# OpenAI Responses stream() forwards tool_choice to SDK kwargs
# ---------------------------------------------------------------------------


def _responses_model() -> Model:
    return Model(id="gpt-4o", provider_id="openai-responses", api="openai_responses")


def _make_response_event(type, **kwargs):
    return SimpleNamespace(type=type, **kwargs)


class TestOpenAIResponsesStreamToolChoice:
    async def test_stream_forwards_required(self):
        events = [
            _make_response_event(
                "response.completed",
                response=SimpleNamespace(
                    id="resp-1",
                    output=[
                        SimpleNamespace(
                            type="message",
                            content=[SimpleNamespace(type="output_text", text="hi")],
                        )
                    ],
                    status="completed",
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                ),
            ),
        ]
        provider = OpenAIResponsesProvider(api_key="test-key")
        mock_client = MagicMock()
        mock_client.responses = MagicMock()
        mock_client.responses.create = AsyncMock(return_value=_async_iter(events))
        provider._client = mock_client

        ms = await provider.stream(
            _responses_model(), _MSGS, tools=[_TOOL], tool_choice="required"
        )
        async for _ in ms:
            pass
        await ms.result()

        call_kwargs = mock_client.responses.create.call_args[1]
        assert call_kwargs["tool_choice"] == "required"
