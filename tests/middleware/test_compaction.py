from __future__ import annotations

import asyncio
from typing import Any

from cubepi.agent.types import AgentContext
from cubepi.middleware.compaction import CompactionMiddleware, CompactionState
from cubepi.middleware.compaction import _load_state
from cubepi.middleware.compaction.state import message_ref, message_refs
from cubepi.providers.base import (
    AssistantMessage,
    Message,
    Model,
    StreamOptions,
    TextContent,
    ToolDefinition,
    UserMessage,
)


def _user(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)])


def _assistant(text: str) -> AssistantMessage:
    return AssistantMessage(content=[TextContent(text=text)])


class _FakeSummaryProvider:
    def __init__(
        self, *, reply: str = "summary text", raises: Exception | None = None
    ) -> None:
        self.reply = reply
        self.raises = raises
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
        if self.raises is not None:
            raise self.raises
        return AssistantMessage(content=[TextContent(text=self.reply)])


def _make_middleware(
    provider: _FakeSummaryProvider,
    *,
    max_tokens_before: int = 1000,
) -> CompactionMiddleware:
    return CompactionMiddleware(
        summary_provider=provider,
        summary_model=Model(id="summary-model", provider="faux"),
        max_tokens_before_compact=max_tokens_before,
        keep_recent_messages=2,
        max_summary_tokens=512,
        min_compact_messages=2,
    )


async def test_under_threshold_returns_existing_compressed_view() -> None:
    provider = _FakeSummaryProvider()
    middleware = _make_middleware(provider, max_tokens_before=100_000)
    messages: list[Message] = [
        _user("old"),
        _assistant("old reply"),
        _user("recent"),
        _assistant("recent reply"),
    ]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={
            "compaction": CompactionState(
                summary="old summary",
                summarized_message_refs=message_refs(messages[:2]),
            ).model_dump(),
            "compaction_until_msg_index": 2,
        },
    )

    result = await middleware.transform_context(messages, ctx=ctx)

    assert len(result) == 3
    assert isinstance(result[0], UserMessage)
    assert "old summary" in result[0].content[0].text
    assert result[1:] == messages[2:]
    assert provider.calls == []


def test_load_state_accepts_state_and_ignores_unknown_values() -> None:
    state = CompactionState(summary="cached")

    assert _load_state(state) is state
    assert _load_state("not-state") is None


def test_message_ref_prefers_explicit_message_id() -> None:
    class _MessageWithId:
        id = "msg-123"

        def model_dump(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            return {"content": "unused"}

    assert message_ref(_MessageWithId()) == "id:msg-123"  # type: ignore[arg-type]


async def test_over_threshold_writes_json_safe_state_to_ctx_extra() -> None:
    provider = _FakeSummaryProvider(reply="New summary")
    middleware = _make_middleware(provider, max_tokens_before=1)
    signal = asyncio.Event()
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        _user("turn 2"),
        _assistant("reply 2"),
        _user("turn 3"),
        _assistant("reply 3"),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})

    result = await middleware.transform_context(messages, ctx=ctx, signal=signal)

    assert isinstance(ctx.extra["compaction"], dict)
    state = CompactionState.model_validate(ctx.extra["compaction"])
    assert state.summary == "New summary"
    assert ctx.extra["compaction_until_msg_index"] > 0
    assert isinstance(result[0], UserMessage)
    assert provider.calls[0]["options"].signal is signal


async def test_over_threshold_without_safe_boundary_returns_compressed_view() -> None:
    provider = _FakeSummaryProvider()
    middleware = _make_middleware(provider, max_tokens_before=1)
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})

    result = await middleware.transform_context(messages, ctx=ctx)

    assert result == messages
    assert provider.calls == []


async def test_summarizer_failure_returns_current_view_without_writing_state() -> None:
    provider = _FakeSummaryProvider(raises=RuntimeError("LLM unavailable"))
    middleware = _make_middleware(provider, max_tokens_before=1)
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        _user("turn 2"),
        _assistant("reply 2"),
        _user("turn 3"),
        _assistant("reply 3"),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})

    result = await middleware.transform_context(messages, ctx=ctx)

    assert result == messages
    assert "compaction" not in ctx.extra


async def test_stale_boundary_larger_than_history_is_ignored() -> None:
    provider = _FakeSummaryProvider()
    middleware = _make_middleware(provider, max_tokens_before=100_000)
    messages: list[Message] = [_user("new question")]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={
            "compaction": CompactionState(summary="old summary").model_dump(),
            "compaction_until_msg_index": 10,
        },
    )

    result = await middleware.transform_context(messages, ctx=ctx)

    assert result == messages
    assert "compaction" not in ctx.extra
    assert "compaction_until_msg_index" not in ctx.extra


async def test_stale_boundary_from_replaced_history_is_ignored() -> None:
    provider = _FakeSummaryProvider()
    middleware = _make_middleware(provider, max_tokens_before=100_000)
    old_messages: list[Message] = [
        _user("old turn 1"),
        _assistant("old reply 1"),
        _user("old turn 2"),
        _assistant("old reply 2"),
    ]
    new_messages: list[Message] = [
        _user("new turn 1"),
        _assistant("new reply 1"),
        _user("new turn 2"),
        _assistant("new reply 2"),
    ]
    ctx = AgentContext(
        system_prompt="",
        messages=new_messages,
        extra={
            "compaction": CompactionState(
                summary="old summary",
                summarized_message_refs=message_refs(old_messages[:2]),
            ).model_dump(),
            "compaction_until_msg_index": 2,
        },
    )

    result = await middleware.transform_context(new_messages, ctx=ctx)

    assert result == new_messages
    assert "compaction" not in ctx.extra
    assert "compaction_until_msg_index" not in ctx.extra


async def test_stale_boundary_with_mismatched_ref_count_is_ignored() -> None:
    provider = _FakeSummaryProvider()
    middleware = _make_middleware(provider, max_tokens_before=100_000)
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        _user("turn 2"),
        _assistant("reply 2"),
    ]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={
            "compaction": CompactionState(
                summary="old summary",
                summarized_message_refs=message_refs(messages[:1]),
            ).model_dump(),
            "compaction_until_msg_index": 2,
        },
    )

    result = await middleware.transform_context(messages, ctx=ctx)

    assert result == messages
    assert "compaction" not in ctx.extra
    assert "compaction_until_msg_index" not in ctx.extra


async def test_malformed_persisted_state_is_cleared() -> None:
    provider = _FakeSummaryProvider()
    middleware = _make_middleware(provider, max_tokens_before=100_000)
    messages: list[Message] = [_user("new question")]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={
            "compaction": {},
            "compaction_until_msg_index": 1,
        },
    )

    result = await middleware.transform_context(messages, ctx=ctx)

    assert result == messages
    assert "compaction" not in ctx.extra
    assert "compaction_until_msg_index" not in ctx.extra
