from __future__ import annotations

from typing import Any

import pytest

from cubepi.providers.base import (
    AssistantMessage,
    BaseProvider,
    Message,
    Model,
    StreamOptions,
    TextContent,
    ToolDefinition,
    UserMessage,
)
from cubepi.providers.faux import FauxProvider, faux_assistant_message


class _RecordingProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(provider_id="rec")
        self.generate_calls: list[dict[str, Any]] = []

    async def generate(  # type: ignore[override]
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        thinking: Any = None,
        thinking_budgets: Any = None,
    ) -> AssistantMessage:
        self.generate_calls.append(
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
        return AssistantMessage(
            content=[TextContent(text="ok")],
            provider_id=model.provider_id,
            model_id=model.id,
        )


@pytest.mark.asyncio
async def test_bound_model_generate_forwards_to_provider() -> None:
    provider = _RecordingProvider()
    bound = provider.model("model-x", temperature=0.5)
    messages = [UserMessage(content=[TextContent(text="hi")])]

    response = await bound.generate(
        messages=messages,
        system_prompt="be brief",
        max_output_tokens=64,
        temperature=0.0,
        thinking="off",
    )

    assert isinstance(response, AssistantMessage)
    assert response.provider_id == "rec"
    assert response.model_id == "model-x"

    assert len(provider.generate_calls) == 1
    call = provider.generate_calls[0]
    assert call["model"] is bound.spec
    assert call["messages"] is messages
    assert call["system_prompt"] == "be brief"
    assert call["max_output_tokens"] == 64
    assert call["temperature"] == 0.0
    assert call["thinking"] == "off"


@pytest.mark.asyncio
async def test_bound_model_stream_forwards_to_provider() -> None:
    provider = FauxProvider(provider_id="faux")
    provider.set_responses([faux_assistant_message("hello")])
    bound = provider.model("faux-1")

    stream = await bound.stream(
        messages=[UserMessage(content=[TextContent(text="hi")])],
        system_prompt="be brief",
    )

    events: list[str] = []
    async for event in stream:
        events.append(event.type)
        if event.type in ("done", "error"):
            break
    result = await stream.result()

    assert "start" in events
    assert "done" in events
    assert result.model_id == "faux-1"
    assert result.provider_id == "faux"
