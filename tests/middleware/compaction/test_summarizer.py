from __future__ import annotations

import asyncio
from typing import Any

from cubepi.middleware.compaction import CompactionState
from cubepi.middleware.compaction.summarizer import (
    _format_message_for_summary,
    summarize,
)
from cubepi.providers.base import (
    AssistantMessage,
    Message,
    Model,
    StreamOptions,
    TextContent,
    ToolCall,
    ToolDefinition,
    UserMessage,
)


class _FakeProvider:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        thinking=None,
        thinking_budgets=None,
    ) -> AssistantMessage:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "system_prompt": system_prompt,
                "tools": tools,
                "options": options,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "thinking": thinking,
                "thinking_budgets": thinking_budgets,
            }
        )
        return AssistantMessage(content=[TextContent(text=self.reply)])


async def test_summarize_uses_provider_generate_with_common_overrides() -> None:
    provider = _FakeProvider(" Compressed summary. ")
    model = Model(id="summary-model", provider_id="faux")
    signal = asyncio.Event()

    result = await summarize(
        provider=provider,
        model=model,
        messages_to_summarize=[
            UserMessage(content=[TextContent(text="hello")]),
            AssistantMessage(content=[TextContent(text="hi")]),
        ],
        existing=None,
        max_summary_tokens=512,
        abort_signal=signal,
    )

    assert isinstance(result, CompactionState)
    assert result.summary == "Compressed summary."
    assert len(result.summarized_message_refs) == 2
    assert provider.calls[0]["max_output_tokens"] == 512
    assert provider.calls[0]["temperature"] == 0.0
    assert provider.calls[0]["thinking"] == "off"
    assert provider.calls[0]["options"].signal is signal


async def test_summarize_merges_existing_state() -> None:
    provider = _FakeProvider("Merged summary.")
    existing = CompactionState(summary="Older context.")

    result = await summarize(
        provider=provider,
        model=Model(id="summary-model", provider_id="faux"),
        messages_to_summarize=[UserMessage(content=[TextContent(text="new")])],
        existing=existing,
    )

    assert "Older context." in provider.calls[0]["system_prompt"]
    assert result.summary == "Merged summary."


async def test_summarize_raises_on_provider_error_message() -> None:
    class _ErrorProvider(_FakeProvider):
        async def generate(
            self,
            model: Model,
            messages: list[Message],
            *,
            system_prompt: str = "",
            tools: list[ToolDefinition] | None = None,
            options: StreamOptions | None = None,
            max_output_tokens: int | None = None,
            temperature: float | None = None,
            thinking=None,
            thinking_budgets=None,
        ) -> AssistantMessage:
            del model, messages, system_prompt, tools, options
            del max_output_tokens, temperature, thinking, thinking_budgets
            return AssistantMessage(
                content=[],
                stop_reason="error",
                error_message="summary failed",
            )

    try:
        await summarize(
            provider=_ErrorProvider(""),
            model=Model(id="summary-model", provider_id="faux"),
            messages_to_summarize=[UserMessage(content=[TextContent(text="new")])],
            existing=None,
        )
    except RuntimeError as exc:
        assert str(exc) == "summary failed"
    else:  # pragma: no cover
        raise AssertionError("provider error was not raised")


def test_format_message_for_summary_includes_tool_calls_and_text_like_blocks() -> None:
    class _TextLike:
        text = "extra text"

    class _Transcript:
        content = [
            TextContent(text="checking"),
            ToolCall(id="t1", name="lookup", arguments={"q": "x"}),
            _TextLike(),
        ]

    message = _Transcript()

    formatted = _format_message_for_summary(message)  # type: ignore[arg-type]

    assert "[_transcript]" in formatted
    assert "checking" in formatted
    assert "[tool_call:lookup]" in formatted
    assert "extra text" in formatted
