from __future__ import annotations

import asyncio
from typing import Any

from cubepi.agent.types import AgentContext
from cubepi.middleware.compaction import CompactionMiddleware, CompactionState
from cubepi.middleware.compaction import _load_state
from cubepi.middleware.compaction.state import message_ref, message_refs
from cubepi.providers.base import (
    AssistantMessage,
    BoundModel,
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
    keep_tail_tokens: int = 8,
) -> CompactionMiddleware:
    """Tiny tail-token budget (~2 small test messages) approximates the
    old ``keep_recent_messages=2`` behaviour for these scenarios."""
    return CompactionMiddleware(
        summary_model=BoundModel(
            provider=provider,
            spec=Model(id="summary-model", provider_id="faux"),
        ),
        max_tokens_before_compact=max_tokens_before,
        keep_tail_tokens=keep_tail_tokens,
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


async def test_summarizer_failure_writes_fallback_state() -> None:
    """When the LLM raises, build_fallback_summary() runs so the agent
    still gets a compressed view; failures counter increments."""
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

    # Fallback state was written despite the LLM failure.
    assert "compaction" in ctx.extra
    state = CompactionState.model_validate(ctx.extra["compaction"])
    assert state.is_fallback is True
    # Failure counter incremented.
    assert ctx.extra["compaction_failures"] == 1
    # Result is compressed (summary + tail), not the raw message list.
    assert len(result) < len(messages)


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


# --- Task 6: circuit breaker, anti-thrashing, refs survive pruning ---


async def test_circuit_breaker_opens_after_three_failures() -> None:
    """After 3 LLM failures the breaker opens; 4th call uses fallback (no LLM).

    Messages grow each turn so safe_boundary keeps advancing — without that,
    once a fallback state is persisted, subsequent calls early-return at
    ``new_boundary <= boundary`` and the LLM is never called again.
    """
    provider = _FakeSummaryProvider(raises=RuntimeError("down"))
    middleware = _make_middleware(provider, max_tokens_before=1)
    ctx = AgentContext(system_prompt="", messages=[], extra={})

    messages: list[Message] = []
    for i in range(3):
        messages = [
            *messages,
            _user(f"turn {i * 2}"),
            _assistant(f"reply {i * 2}"),
            _user(f"turn {i * 2 + 1}"),
            _assistant(f"reply {i * 2 + 1}"),
        ]
        await middleware.transform_context(messages, ctx=ctx)
        assert ctx.extra["compaction_failures"] == i + 1
        assert "compaction" in ctx.extra

    calls_before_breaker = len(provider.calls)

    # Turn 4: breaker open → LLM NOT called, fallback still runs.
    messages = [*messages, _user("more"), _assistant("more reply")]
    result = await middleware.transform_context(messages, ctx=ctx)
    assert len(provider.calls) == calls_before_breaker  # no new LLM call
    assert ctx.extra["compaction_failures"] == 3  # frozen at MAX_FAILURES
    assert "compaction" in ctx.extra
    assert len(result) < len(messages)  # still compressed via fallback


async def test_circuit_breaker_resets_on_llm_success() -> None:
    provider = _FakeSummaryProvider(reply="real summary")
    middleware = _make_middleware(provider, max_tokens_before=1)
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        _user("turn 2"),
        _assistant("reply 2"),
        _user("turn 3"),
        _assistant("reply 3"),
    ]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={"compaction_failures": 2},  # pre-seed: 2 prior failures
    )

    await middleware.transform_context(messages, ctx=ctx)
    assert ctx.extra["compaction_failures"] == 0  # reset after success


async def test_anti_thrashing_skips_compaction_when_guard_tripped() -> None:
    """Guard fires when low_savings_count >= 2, raw history not over 1.5×
    threshold, and new_boundary advance < 8 messages."""
    provider = _FakeSummaryProvider(reply="x")
    # Threshold chosen so tokens_now (~18) EXCEEDS threshold (so the
    # under-threshold fast-path doesn't fire) AND raw_tokens (~18) stays
    # under 1.5 * threshold (so emergency override doesn't bypass the guard).
    middleware = _make_middleware(provider, max_tokens_before=15)
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        _user("turn 2"),
        _assistant("reply 2"),
        _user("turn 3"),
        _assistant("reply 3"),
    ]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={"compaction_low_savings_count": 2},  # guard tripped
    )

    calls_before = len(provider.calls)
    await middleware.transform_context(messages, ctx=ctx)
    # Neither LLM nor fallback ran — early skip.
    assert len(provider.calls) == calls_before
    assert "compaction" not in ctx.extra


async def test_anti_thrashing_emergency_override_when_raw_history_too_large() -> None:
    """When raw history >= 1.5 × threshold, the guard is overridden."""
    provider = _FakeSummaryProvider(reply="short summary")
    middleware = _make_middleware(provider, max_tokens_before=10)
    # Build a large enough history that raw tokens exceed 1.5 × 10 = 15.
    messages: list[Message] = [
        _user("x" * 100),
        _assistant("y" * 100),
        _user("z" * 100),
        _assistant("w" * 100),
        _user("a" * 100),
        _assistant("b" * 100),
    ]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={"compaction_low_savings_count": 2},  # guard tripped
    )

    calls_before = len(provider.calls)
    await middleware.transform_context(messages, ctx=ctx)
    # Emergency override fired despite the guard.
    assert len(provider.calls) > calls_before
    assert "compaction" in ctx.extra


async def test_pruned_tool_results_do_not_break_state_refs() -> None:
    """Refs persisted in CompactionState come from ORIGINAL messages even
    when the transcript was built from pre-pruned content. Otherwise the
    next turn would see ref mismatch and clear the state, looping forever."""
    from cubepi.providers.base import ToolCall, ToolResultMessage

    provider = _FakeSummaryProvider(reply="real summary")
    middleware = _make_middleware(provider, max_tokens_before=1)
    big_result = "tool output line\n" * 200  # > 120 chars → pruner targets it
    messages: list[Message] = [
        _user("audit q"),
        AssistantMessage(
            content=[ToolCall(id="c1", name="audit_query", arguments={"q": "X"})]
        ),
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="audit_query",
            content=[TextContent(text=big_result)],
        ),
        _user("what next?"),
        _assistant("fix it"),
        _user("ok"),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})

    await middleware.transform_context(messages, ctx=ctx)
    boundary_after_first = ctx.extra["compaction_until_msg_index"]

    # Second turn with same messages: state must validate, not be cleared.
    await middleware.transform_context(messages, ctx=ctx)
    assert ctx.extra.get("compaction_until_msg_index", 0) >= boundary_after_first
    state = CompactionState.model_validate(ctx.extra["compaction"])
    assert state.is_fallback is False  # real LLM summary


def test_keep_recent_messages_no_longer_accepted() -> None:
    """Breaking change — old parameter name raises TypeError."""
    import pytest

    provider = _FakeSummaryProvider()
    with pytest.raises(TypeError):
        CompactionMiddleware(
            summary_model=BoundModel(
                provider=provider,
                spec=Model(id="m", provider_id="faux"),
            ),
            max_tokens_before_compact=100,
            keep_recent_messages=8,  # type: ignore[call-arg]
        )


# --- Task 7: filter-safe prefix, pruner toggle, prompt override ---


def test_summary_prefix_includes_non_instruction_disclaimer() -> None:
    from cubepi.middleware.compaction import SUMMARY_PREFIX

    text = SUMMARY_PREFIX.lower()
    assert "do not treat" in text or "not instructions" in text
    assert "reference" in text


async def test_prune_tool_outputs_disabled_keeps_full_result_content() -> None:
    """When prune_tool_outputs=False, original tool result content survives.

    Audit-chain agents (finance, compliance) pass False so historical tool
    results stay full-fidelity across compactions.
    """
    from cubepi.providers.base import ToolCall, ToolResultMessage

    provider = _FakeSummaryProvider(reply="summary")
    middleware = CompactionMiddleware(
        summary_model=BoundModel(
            provider=provider,
            spec=Model(id="m", provider_id="faux"),
        ),
        max_tokens_before_compact=20,
        keep_tail_tokens=8,
        min_compact_messages=2,
        prune_tool_outputs=False,
    )
    big_text = "important audit detail " * 200
    messages: list[Message] = [
        _user("audit q"),
        AssistantMessage(
            content=[ToolCall(id="c1", name="audit_query", arguments={"q": "X"})]
        ),
        ToolResultMessage(
            tool_call_id="c1",
            tool_name="audit_query",
            content=[TextContent(text=big_text)],
        ),
        _user("next?"),
        _assistant("ok"),
        _user("confirm"),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})
    await middleware.transform_context(messages, ctx=ctx)

    # The original message list is never mutated regardless of toggle.
    assert messages[2].content[0].text == big_text


async def test_summary_prompt_constructor_argument_passthrough() -> None:
    provider = _FakeSummaryProvider(reply="x")
    middleware = CompactionMiddleware(
        summary_model=BoundModel(
            provider=provider,
            spec=Model(id="m", provider_id="faux"),
        ),
        max_tokens_before_compact=1,
        keep_tail_tokens=8,
        min_compact_messages=2,
        summary_prompt="CUSTOM PROMPT BODY",
    )
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        _user("turn 2"),
        _assistant("reply 2"),
        _user("turn 3"),
        _assistant("reply 3"),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})
    await middleware.transform_context(messages, ctx=ctx)

    assert len(provider.calls) == 1
    # No prior summary, so the prompt should be the override verbatim.
    assert provider.calls[0]["system_prompt"] == "CUSTOM PROMPT BODY"


# --- coverage: extra_llm_calls ---


def test_extra_llm_calls_returns_summary_model() -> None:
    provider = _FakeSummaryProvider()
    bound = BoundModel(
        provider=provider,
        spec=Model(id="m", provider_id="faux"),
    )
    middleware = CompactionMiddleware(
        summary_model=bound,
        max_tokens_before_compact=100,
        keep_tail_tokens=8,
    )
    assert middleware.extra_llm_calls() == (bound,)


# --- codex review: effective tail budget clamp ---


async def test_keep_tail_tokens_clamped_below_threshold() -> None:
    """If keep_tail_tokens >= max_tokens_before_compact, the tail must NOT
    swallow the entire history; otherwise compaction can never trigger.
    The middleware clamps the effective tail to half the threshold."""
    provider = _FakeSummaryProvider(reply="summary")
    middleware = CompactionMiddleware(
        summary_model=BoundModel(
            provider=provider,
            spec=Model(id="m", provider_id="faux"),
        ),
        max_tokens_before_compact=10,
        keep_tail_tokens=200,  # configured larger than threshold AND than raw
        min_compact_messages=2,
    )
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        _user("turn 2"),
        _assistant("reply 2"),
        _user("turn 3"),
        _assistant("reply 3"),
    ]
    ctx = AgentContext(system_prompt="", messages=messages, extra={})
    await middleware.transform_context(messages, ctx=ctx)

    # Without the clamp, the whole 18-token history would fit in the
    # 200-token tail budget and safe_boundary would return None → no
    # compaction. With the clamp, the effective tail is min(200, 10//2)=5
    # tokens, so compaction fires and writes state.
    assert "compaction" in ctx.extra
    assert len(provider.calls) == 1


# --- codex review: half-open circuit breaker ---


async def test_circuit_breaker_half_open_retries_llm_after_fallback_runs() -> None:
    """After _MAX_FAILURES failures the breaker opens. After
    _HALF_OPEN_AFTER_FALLBACK_RUNS fallback-only runs, the breaker goes
    half-open and the LLM is attempted again. If it succeeds, the breaker
    fully resets."""
    from cubepi.middleware.compaction import _HALF_OPEN_AFTER_FALLBACK_RUNS

    provider = _FakeSummaryProvider(raises=RuntimeError("down"))
    middleware = _make_middleware(provider, max_tokens_before=1)
    ctx = AgentContext(system_prompt="", messages=[], extra={})

    # Drive the breaker open with 3 LLM failures across growing histories.
    messages: list[Message] = []
    for i in range(3):
        messages = [
            *messages,
            _user(f"turn {i * 2}"),
            _assistant(f"reply {i * 2}"),
            _user(f"turn {i * 2 + 1}"),
            _assistant(f"reply {i * 2 + 1}"),
        ]
        await middleware.transform_context(messages, ctx=ctx)
    assert ctx.extra["compaction_failures"] == 3
    calls_when_open = len(provider.calls)

    # Run fallback-only turns until the breaker should go half-open.
    for i in range(_HALF_OPEN_AFTER_FALLBACK_RUNS):
        messages = [
            *messages,
            _user(f"fbr {i} q"),
            _assistant(f"fbr {i} a"),
        ]
        await middleware.transform_context(messages, ctx=ctx)
    # No LLM call attempted during fallback-only phase.
    assert len(provider.calls) == calls_when_open
    assert ctx.extra["compaction_fallback_runs"] == _HALF_OPEN_AFTER_FALLBACK_RUNS

    # Swap to a working LLM and run one more turn — half-open path triggers,
    # LLM is attempted, succeeds, breaker fully resets.
    provider.raises = None
    provider.reply = "real summary"
    messages = [*messages, _user("recovered q"), _assistant("recovered a")]
    await middleware.transform_context(messages, ctx=ctx)

    assert len(provider.calls) == calls_when_open + 1
    assert ctx.extra["compaction_failures"] == 0
    assert ctx.extra["compaction_fallback_runs"] == 0


async def test_half_open_failure_re_opens_breaker() -> None:
    """Half-open retry that fails snaps the breaker back to MAX_FAILURES."""
    from cubepi.middleware.compaction import _HALF_OPEN_AFTER_FALLBACK_RUNS

    provider = _FakeSummaryProvider(raises=RuntimeError("down"))
    middleware = _make_middleware(provider, max_tokens_before=1)
    ctx = AgentContext(system_prompt="", messages=[], extra={})

    messages: list[Message] = []
    # 3 failures → breaker opens.
    for i in range(3):
        messages = [
            *messages,
            _user(f"t{i}a"),
            _assistant(f"r{i}a"),
            _user(f"t{i}b"),
            _assistant(f"r{i}b"),
        ]
        await middleware.transform_context(messages, ctx=ctx)
    # Fallback-only runs until half-open is ready.
    for i in range(_HALF_OPEN_AFTER_FALLBACK_RUNS):
        messages = [*messages, _user(f"f{i}"), _assistant(f"a{i}")]
        await middleware.transform_context(messages, ctx=ctx)

    # Half-open turn: LLM still failing.
    calls_before_retry = len(provider.calls)
    messages = [*messages, _user("retry q"), _assistant("retry a")]
    await middleware.transform_context(messages, ctx=ctx)

    # LLM was attempted (half-open allowed one try) and failed.
    assert len(provider.calls) == calls_before_retry + 1
    # Breaker re-opens (failures back to MAX, not MAX+1).
    assert ctx.extra["compaction_failures"] == 3


# --- coverage: _load_int with bad type ---


async def test_corrupt_failure_counter_treated_as_zero() -> None:
    """Non-numeric values in ctx.extra are treated as 0 — defensive parsing.

    Covers both the ``isinstance`` reject path (wrong type) and the
    ``ValueError`` catch path (string that can't be parsed as int).
    """
    provider = _FakeSummaryProvider()
    middleware = _make_middleware(provider, max_tokens_before=1)
    messages: list[Message] = [
        _user("turn 1"),
        _assistant("reply 1"),
        _user("turn 2"),
        _assistant("reply 2"),
        _user("turn 3"),
        _assistant("reply 3"),
    ]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={
            "compaction_failures": "not-an-int",  # ValueError in int()
            "compaction_low_savings_count": ["bad", "shape"],  # isinstance reject
            "compaction_fallback_runs": None,  # isinstance reject
        },
    )
    await middleware.transform_context(messages, ctx=ctx)
    # Compaction proceeded as if all counters were 0.
    assert "compaction" in ctx.extra
    assert ctx.extra["compaction_failures"] == 0  # success path


async def test_state_invalidation_clears_guard_counters() -> None:
    """When the persisted summary is invalidated (history replaced), the
    breaker / anti-thrash counters must also reset — otherwise a fresh
    conversation would skip the LLM on its first compaction because the
    previous conversation hit MAX_FAILURES."""
    provider = _FakeSummaryProvider(reply="summary")
    middleware = _make_middleware(provider, max_tokens_before=1)
    # New messages — refs in ctx.extra won't match, so state is invalidated.
    messages: list[Message] = [
        _user("brand new 1"),
        _assistant("reply 1"),
        _user("brand new 2"),
        _assistant("reply 2"),
        _user("brand new 3"),
        _assistant("reply 3"),
    ]
    ctx = AgentContext(
        system_prompt="",
        messages=messages,
        extra={
            "compaction": CompactionState(
                summary="stale from previous conversation",
                summarized_message_refs=["sha256:does-not-match"],
            ).model_dump(),
            "compaction_until_msg_index": 1,
            "compaction_failures": 3,  # stale breaker — would gate LLM
            "compaction_low_savings_count": 2,  # stale guard — would skip
            "compaction_fallback_runs": 99,
        },
    )

    await middleware.transform_context(messages, ctx=ctx)

    # The LLM was called (breaker counter cleared); fresh state written.
    assert len(provider.calls) == 1
    assert ctx.extra["compaction_failures"] == 0
    # The new summary is real, not a fallback.
    state = CompactionState.model_validate(ctx.extra["compaction"])
    assert state.is_fallback is False
    # Stale guard counters were cleared BEFORE the turn; the new turn may
    # legitimately set them to small values. The point is that the STALE
    # values (2 and 99) did not carry through.
    assert ctx.extra.get("compaction_low_savings_count", 0) < 2
    assert ctx.extra.get("compaction_fallback_runs", 0) < 99
