"""Tests for AnthropicProvider's capability-driven thinking + temperature path.

The Anthropic provider always runs through the capability path; when
``capability=None`` the provider falls back to ``_ANTHROPIC_DEFAULT_CAPABILITY``
which mirrors today's wire bytes exactly. See Task 10 of the capability
descriptor plan and ``cubepi/providers/anthropic.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from cubepi.providers.anthropic import AnthropicProvider
from cubepi.providers.base import (
    Model,
    StreamOptions,
    TextContent,
    UserMessage,
)
from cubepi.providers.capability import (
    CapabilityDescriptor,
    ReasoningLevelSpec,
    TemperatureSpec,
)


def _model() -> Model:
    return Model(
        id="claude-sonnet-test",
        provider="anthropic",
        context_window=200000,
        max_tokens=8192,
        reasoning=True,
        temperature=1.0,
    )


async def _capture_anthropic(
    p: AnthropicProvider, opts: StreamOptions
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def listener(kw: dict[str, Any], m: Model) -> None:
        captured.update(kw)

    p._request_listeners.append(listener)

    class _FakeStream:
        response = None

        async def __aenter__(self) -> "_FakeStream":
            return self

        async def __aexit__(self, *a: Any) -> bool:
            return False

        def __aiter__(self) -> Any:
            async def gen() -> Any:
                return
                yield  # pragma: no cover - generator marker

            return gen()

        async def get_final_message(self) -> Any:
            from anthropic.types import Message

            return Message.model_construct(
                id="m_test",
                model="claude-sonnet-test",
                role="assistant",
                content=[],
                stop_reason="end_turn",
                stop_sequence=None,
                type="message",
                usage={"input_tokens": 1, "output_tokens": 1},
            )

    def fake_stream(**kw: Any) -> _FakeStream:
        return _FakeStream()

    p._client.messages.stream = fake_stream  # type: ignore[attr-defined]

    s = await p.stream(
        model=_model(),
        messages=[UserMessage(content=[TextContent(text="hi")])],
        options=opts,
    )
    async for _ in s:
        pass
    await s.result()
    return captured


@pytest.mark.asyncio
async def test_default_capability_matches_legacy_thinking_off() -> None:
    p = AnthropicProvider(api_key="x")
    payload = await _capture_anthropic(p, StreamOptions(thinking="off"))
    # Legacy behavior: no "thinking" key when thinking=off.
    # The default capability's reasoning_off_payload is empty, so the
    # absent-key case is what we get; tolerate {"type": "disabled"} for
    # callers who override the default capability.
    assert payload.get("thinking") == {"type": "disabled"} or "thinking" not in payload
    # Temperature is allowed when thinking is off.
    assert payload.get("temperature") == 1.0


@pytest.mark.asyncio
async def test_default_capability_thinking_medium_writes_budget() -> None:
    p = AnthropicProvider(api_key="x")
    payload = await _capture_anthropic(p, StreamOptions(thinking="medium"))
    assert payload["thinking"]["type"] == "enabled"
    # Mirrors ThinkingBudgets.medium (8192) in cubepi/providers/base.py.
    assert payload["thinking"]["budget_tokens"] == 8192
    # Anthropic rejects custom temperature with thinking on.
    assert "temperature" not in payload


@pytest.mark.asyncio
async def test_custom_capability_overrides_default() -> None:
    custom = CapabilityDescriptor(
        reasoning_off_payload={"thinking": {"type": "disabled"}},
        reasoning_on_payload={"thinking": {"type": "enabled"}},
        reasoning_level=ReasoningLevelSpec(
            path="thinking.budget_tokens",
            kind="int_budget",
            level_budgets={"medium": 99999},
        ),
        temperature=TemperatureSpec(mode="ignored"),
    )
    p = AnthropicProvider(api_key="x", capability=custom)
    payload = await _capture_anthropic(p, StreamOptions(thinking="medium"))
    assert payload["thinking"]["budget_tokens"] == 99999
    assert "temperature" not in payload


@pytest.mark.asyncio
async def test_model_capability_overrides_take_precedence() -> None:
    """Per-model override beats the instance-level capability."""
    override = CapabilityDescriptor(
        reasoning_off_payload={},
        reasoning_on_payload={"thinking": {"type": "enabled"}},
        reasoning_level=ReasoningLevelSpec(
            path="thinking.budget_tokens",
            kind="int_budget",
            level_budgets={"medium": 4242},
        ),
        temperature=TemperatureSpec(mode="ignored"),
    )
    p = AnthropicProvider(
        api_key="x",
        model_capability_overrides={"claude-sonnet-test": override},
    )
    payload = await _capture_anthropic(p, StreamOptions(thinking="medium"))
    assert payload["thinking"]["budget_tokens"] == 4242


@pytest.mark.asyncio
async def test_custom_high_budget_capability_bumps_max_tokens():
    """Regression for max_tokens/budget_tokens desync: when custom level_budgets
    writes a budget larger than ThinkingBudgets defaults, max_tokens must be
    expanded to accommodate it (else Anthropic API rejects with
    budget_tokens >= max_tokens)."""
    custom = CapabilityDescriptor(
        reasoning_on_payload={"thinking": {"type": "enabled"}},
        reasoning_level=ReasoningLevelSpec(
            path="thinking.budget_tokens",
            kind="int_budget",
            level_budgets={"medium": 50000},  # WAY above ThinkingBudgets.medium=8192
        ),
    )
    p = AnthropicProvider(api_key="x", capability=custom)
    payload = await _capture_anthropic(p, StreamOptions(thinking="medium"))
    assert payload["thinking"]["budget_tokens"] == 50000
    # max_tokens must be at least budget + something (model.max_tokens=8192,
    # so min(8192+50000, 200000) = 58192). Anthropic rejects budget>=max_tokens.
    assert payload["max_tokens"] >= 50000 + 1


@pytest.mark.asyncio
async def test_capability_clamps_budget_when_context_too_small() -> None:
    """Regression: when context_window can't fit max_tokens + budget,
    budget is reduced to fit. Anthropic rejects budget >= max_tokens."""
    custom = CapabilityDescriptor(
        reasoning_on_payload={"thinking": {"type": "enabled"}},
        reasoning_level=ReasoningLevelSpec(
            path="thinking.budget_tokens",
            kind="int_budget",
            level_budgets={"high": 16384},
        ),
    )
    p = AnthropicProvider(api_key="x", capability=custom)
    # Tight model: context_window only 10000, max_tokens 8192.
    m = Model(
        id="claude-tight-test",
        provider="anthropic",
        context_window=10000,
        max_tokens=8192,
        reasoning=True,
        temperature=1.0,
    )
    captured: dict[str, Any] = {}

    async def listener(kw: dict[str, Any], model: Model) -> None:
        captured.update(kw)

    p._request_listeners.append(listener)

    class _FakeStream:
        response = None

        async def __aenter__(self) -> "_FakeStream":
            return self

        async def __aexit__(self, *a: Any) -> bool:
            return False

        def __aiter__(self) -> Any:
            async def gen() -> Any:
                return
                yield  # pragma: no cover - generator marker

            return gen()

        async def get_final_message(self) -> Any:
            from anthropic.types import Message

            return Message.model_construct(
                id="m_test",
                model="claude-tight-test",
                role="assistant",
                content=[],
                stop_reason="end_turn",
                stop_sequence=None,
                type="message",
                usage={"input_tokens": 1, "output_tokens": 1},
            )

    def fake_stream(**kw: Any) -> _FakeStream:
        return _FakeStream()

    p._client.messages.stream = fake_stream  # type: ignore[attr-defined]

    s = await p.stream(
        model=m,
        messages=[UserMessage(content=[TextContent(text="hi")])],
        options=StreamOptions(thinking="high"),
    )
    async for _ in s:
        pass
    await s.result()

    # Either budget was reduced below max_tokens, or thinking was disabled.
    thinking_block = captured.get("thinking", {})
    if thinking_block.get("type") == "enabled":
        assert thinking_block["budget_tokens"] < captured["max_tokens"]
