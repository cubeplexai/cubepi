from __future__ import annotations

from typing import Any

from cubepi.middleware.compaction import CompactionState
from cubepi.middleware.compaction.summarizer import summarize
from cubepi.providers.base import (
    AssistantMessage,
    Message,
    Model,
    StreamOptions,
    TextContent,
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
    model = Model(id="summary-model", provider="faux")

    result = await summarize(
        provider=provider,
        model=model,
        messages_to_summarize=[
            UserMessage(content=[TextContent(text="hello")]),
            AssistantMessage(content=[TextContent(text="hi")]),
        ],
        existing=None,
        max_summary_tokens=512,
    )

    assert isinstance(result, CompactionState)
    assert result.summary == "Compressed summary."
    assert provider.calls[0]["max_output_tokens"] == 512
    assert provider.calls[0]["temperature"] == 0.0
    assert provider.calls[0]["thinking"] == "off"


async def test_summarize_merges_existing_state() -> None:
    provider = _FakeProvider("Merged summary.")
    existing = CompactionState(summary="Older context.")

    result = await summarize(
        provider=provider,
        model=Model(id="summary-model", provider="faux"),
        messages_to_summarize=[UserMessage(content=[TextContent(text="new")])],
        existing=existing,
    )

    assert "Older context." in provider.calls[0]["system_prompt"]
    assert result.summary == "Merged summary."
