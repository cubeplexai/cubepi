"""OpenAIProvider OSS reasoning extraction tests (D4)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cubepi.providers.openai import OpenAIProvider
from cubepi.providers.base import (
    Model,
    TextContent,
    ThinkingContent,
    UserMessage,
)


# ---------------------------------------------------------------------------
# Helpers (same style as test_openai.py)
# ---------------------------------------------------------------------------


def _model() -> Model:
    return Model(id="deepseek-chat", provider_id="deepseek", api="openai")


def _make_chunk(
    content=None,
    reasoning_content=None,
    reasoning=None,
    reasoning_details=None,
    finish_reason=None,
    id=None,
):
    """Build a mock OpenAI chat-completion chunk with optional reasoning fields."""
    delta = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=reasoning_content,
        reasoning=reasoning,
        reasoning_details=reasoning_details,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(id=id, choices=[choice])


async def _async_iter(chunks):
    for chunk in chunks:
        yield chunk


def _run_stream(chunks):
    """Helper: set up mock client, return (provider, mock_client)."""
    with patch("openai.AsyncOpenAI") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_async_iter(chunks)
        )
        provider = OpenAIProvider(api_key="test-key")
        provider._client = mock_client
        return provider, mock_client


# ---------------------------------------------------------------------------
# Reasoning extraction tests
# ---------------------------------------------------------------------------


class TestOpenAIReasoningContent:
    """delta.reasoning_content variant: DeepSeek/Qwen/DouBao."""

    @pytest.mark.asyncio
    async def test_extracts_reasoning_content_variant(self):
        """reasoning_content chunks produce thinking_start/_delta/_end events."""
        chunks = [
            _make_chunk(reasoning_content="Let me "),
            _make_chunk(reasoning_content="think"),
            _make_chunk(content="Answer"),
            _make_chunk(finish_reason="stop"),
        ]
        provider, _ = _run_stream(chunks)
        ms = await provider.stream(
            _model(), [UserMessage(content=[TextContent(text="hi")])]
        )

        events = []
        async for event in ms:
            events.append(event)

        types = [e.type for e in events]
        assert "thinking_start" in types
        assert "thinking_delta" in types
        assert "thinking_end" in types

        # Concatenated thinking deltas
        thinking_deltas = [e.delta for e in events if e.type == "thinking_delta"]
        assert "".join(thinking_deltas) == "Let me think"

        # Final result has ThinkingContent and TextContent
        result = await ms.result()
        assert any(isinstance(c, ThinkingContent) for c in result.content)
        thinking_blocks = [c for c in result.content if isinstance(c, ThinkingContent)]
        assert thinking_blocks[0].thinking == "Let me think"

    @pytest.mark.asyncio
    async def test_thinking_end_fires_on_finish(self):
        """thinking_end is emitted when finish_reason arrives, not before."""
        chunks = [
            _make_chunk(reasoning_content="Thought"),
            _make_chunk(finish_reason="stop"),
        ]
        provider, _ = _run_stream(chunks)
        ms = await provider.stream(
            _model(), [UserMessage(content=[TextContent(text="hi")])]
        )

        events = []
        async for event in ms:
            events.append(event)

        types = [e.type for e in events]
        assert types.index("thinking_end") < types.index("done")

    @pytest.mark.asyncio
    async def test_thinking_start_fires_once(self):
        """thinking_start fires exactly once even with multiple reasoning_content chunks."""
        chunks = [
            _make_chunk(reasoning_content="A"),
            _make_chunk(reasoning_content="B"),
            _make_chunk(reasoning_content="C"),
            _make_chunk(finish_reason="stop"),
        ]
        provider, _ = _run_stream(chunks)
        ms = await provider.stream(
            _model(), [UserMessage(content=[TextContent(text="hi")])]
        )

        events = []
        async for event in ms:
            events.append(event)

        types = [e.type for e in events]
        assert types.count("thinking_start") == 1
        assert types.count("thinking_end") == 1
        assert types.count("thinking_delta") == 3


class TestOpenAIReasoningVariantVllm:
    """delta.reasoning variant: vLLM."""

    @pytest.mark.asyncio
    async def test_extracts_reasoning_variant_vllm(self):
        """vLLM delta.reasoning field is extracted as thinking events."""
        chunks = [
            _make_chunk(reasoning="Step 1"),
            _make_chunk(reasoning=" Step 2"),
            _make_chunk(content="Result"),
            _make_chunk(finish_reason="stop"),
        ]
        provider, _ = _run_stream(chunks)
        ms = await provider.stream(
            _model(), [UserMessage(content=[TextContent(text="hi")])]
        )

        events = []
        async for event in ms:
            events.append(event)

        types = [e.type for e in events]
        assert "thinking_start" in types
        assert "thinking_delta" in types
        assert "thinking_end" in types

        thinking_deltas = [e.delta for e in events if e.type == "thinking_delta"]
        assert "".join(thinking_deltas) == "Step 1 Step 2"

    @pytest.mark.asyncio
    async def test_reasoning_content_takes_priority_over_reasoning(self):
        """When both reasoning_content and reasoning are present, use reasoning_content."""
        chunks = [
            _make_chunk(reasoning_content="from_rc", reasoning="from_r"),
            _make_chunk(finish_reason="stop"),
        ]
        provider, _ = _run_stream(chunks)
        ms = await provider.stream(
            _model(), [UserMessage(content=[TextContent(text="hi")])]
        )

        events = []
        async for event in ms:
            events.append(event)

        thinking_deltas = [e.delta for e in events if e.type == "thinking_delta"]
        assert thinking_deltas == ["from_rc"]


class TestOpenAIReasoningDetailsVariantMinimax:
    """delta.reasoning_details variant: MiniMax."""

    @pytest.mark.asyncio
    async def test_extracts_reasoning_details_list_of_objects(self):
        """MiniMax reasoning_details as list of SimpleNamespace with .text."""
        chunks = [
            _make_chunk(
                reasoning_details=[
                    SimpleNamespace(text="Part A"),
                    SimpleNamespace(text="Part B"),
                ]
            ),
            _make_chunk(content="Final"),
            _make_chunk(finish_reason="stop"),
        ]
        provider, _ = _run_stream(chunks)
        ms = await provider.stream(
            _model(), [UserMessage(content=[TextContent(text="hi")])]
        )

        events = []
        async for event in ms:
            events.append(event)

        types = [e.type for e in events]
        assert "thinking_start" in types
        assert "thinking_delta" in types

        thinking_deltas = [e.delta for e in events if e.type == "thinking_delta"]
        assert "".join(thinking_deltas) == "Part APart B"

    @pytest.mark.asyncio
    async def test_extracts_reasoning_details_list_of_dicts(self):
        """MiniMax reasoning_details as list of plain dicts with 'text' key."""
        chunks = [
            _make_chunk(reasoning_details=[{"text": "Dict part"}]),
            _make_chunk(finish_reason="stop"),
        ]
        provider, _ = _run_stream(chunks)
        ms = await provider.stream(
            _model(), [UserMessage(content=[TextContent(text="hi")])]
        )

        events = []
        async for event in ms:
            events.append(event)

        thinking_deltas = [e.delta for e in events if e.type == "thinking_delta"]
        assert thinking_deltas == ["Dict part"]

    @pytest.mark.asyncio
    async def test_reasoning_details_skips_empty_text(self):
        """reasoning_details entries with no text are skipped silently."""
        chunks = [
            _make_chunk(
                reasoning_details=[
                    SimpleNamespace(text=""),
                    SimpleNamespace(text="Only this"),
                ]
            ),
            _make_chunk(finish_reason="stop"),
        ]
        provider, _ = _run_stream(chunks)
        ms = await provider.stream(
            _model(), [UserMessage(content=[TextContent(text="hi")])]
        )

        events = []
        async for event in ms:
            events.append(event)

        thinking_deltas = [e.delta for e in events if e.type == "thinking_delta"]
        assert thinking_deltas == ["Only this"]


class TestOpenAINoReasoning:
    """When no reasoning fields present, no thinking events should be emitted."""

    @pytest.mark.asyncio
    async def test_no_reasoning_no_thinking_events(self):
        """Plain content stream with no reasoning fields emits no thinking events."""
        chunks = [
            _make_chunk(content="Just text"),
            _make_chunk(finish_reason="stop"),
        ]
        provider, _ = _run_stream(chunks)
        ms = await provider.stream(
            _model(), [UserMessage(content=[TextContent(text="hi")])]
        )

        events = []
        async for event in ms:
            events.append(event)

        types = [e.type for e in events]
        assert "thinking_start" not in types
        assert "thinking_delta" not in types
        assert "thinking_end" not in types

    @pytest.mark.asyncio
    async def test_reasoning_content_none_does_not_trigger(self):
        """reasoning_content=None should not produce thinking events."""
        chunks = [
            _make_chunk(reasoning_content=None, content="Text"),
            _make_chunk(finish_reason="stop"),
        ]
        provider, _ = _run_stream(chunks)
        ms = await provider.stream(
            _model(), [UserMessage(content=[TextContent(text="hi")])]
        )

        events = []
        async for event in ms:
            events.append(event)

        types = [e.type for e in events]
        assert "thinking_start" not in types
