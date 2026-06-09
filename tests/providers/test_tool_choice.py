"""Tests for tool_choice wire format mapping across all providers.

These tests verify that each provider's ``_map_tool_choice`` static method
correctly maps CubePi's ``ToolChoice`` values to provider-specific wire
formats.

NOTE: These tests are intentionally *failing* until the ``_map_tool_choice``
methods and ``tool_choice`` parameter are implemented (TDD red phase).
"""

from __future__ import annotations


from cubepi.providers.anthropic import AnthropicProvider
from cubepi.providers.base import Model, UserMessage, TextContent
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
