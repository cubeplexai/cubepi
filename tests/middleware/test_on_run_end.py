"""on_run_end hook tests."""

from __future__ import annotations

import pytest

from cubepi import Agent, Model
from cubepi.agent.types import AgentContext
from cubepi.middleware.base import Middleware, compose_middleware
from cubepi.providers.base import TextContent, UserMessage
from cubepi.providers.faux import FauxProvider, faux_assistant_message


def _mk_ctx() -> AgentContext:
    return AgentContext(system_prompt="", messages=[])


# ---------------------------------------------------------------------------
# compose_middleware unit tests
# ---------------------------------------------------------------------------


class _InjectOne(Middleware):
    async def on_run_end(self, ctx, *, signal=None):
        return [UserMessage(content=[TextContent(text="reflect")])]


class _InjectTwo(Middleware):
    async def on_run_end(self, ctx, *, signal=None):
        return [
            UserMessage(content=[TextContent(text="a")]),
            UserMessage(content=[TextContent(text="b")]),
        ]


class _ReturnNone(Middleware):
    async def on_run_end(self, ctx, *, signal=None):
        return None


class _Plain(Middleware):
    pass


def test_no_middleware_hook_absent() -> None:
    hooks = compose_middleware([_Plain()])
    assert "on_run_end" not in hooks


def test_returns_none_hook_present_when_implemented() -> None:
    """compose includes on_run_end even when middleware returns None."""
    hooks = compose_middleware([_ReturnNone()])
    assert "on_run_end" in hooks


@pytest.mark.asyncio
async def test_single_middleware_returns_messages() -> None:
    hooks = compose_middleware([_InjectOne()])
    result = await hooks["on_run_end"](_mk_ctx())
    assert result is not None
    assert len(result) == 1
    assert result[0].content[0].text == "reflect"


@pytest.mark.asyncio
async def test_multiple_middleware_concatenate() -> None:
    hooks = compose_middleware([_InjectOne(), _InjectTwo()])
    result = await hooks["on_run_end"](_mk_ctx())
    assert result is not None
    assert len(result) == 3


@pytest.mark.asyncio
async def test_all_none_returns_none() -> None:
    hooks = compose_middleware([_ReturnNone()])
    result = await hooks["on_run_end"](_mk_ctx())
    assert result is None


# ---------------------------------------------------------------------------
# Agent integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_run_end_fires_after_main_run() -> None:
    """on_run_end injects a message and the agent runs one more model call."""
    provider = FauxProvider()
    provider.set_responses(
        [
            faux_assistant_message("main"),
            faux_assistant_message("reflected"),
        ]
    )

    fired: list[str] = []

    class _Reflect(Middleware):
        async def on_run_end(self, ctx, *, signal=None):
            fired.append("fired")
            return [UserMessage(content=[TextContent(text="reflect now")])]

    agent = Agent(
        model=Model(id="test", provider="faux"),
        provider=provider,
        middleware=[_Reflect()],
    )
    await agent.prompt("hi")

    assert provider.call_count == 2
    assert fired == ["fired"]


@pytest.mark.asyncio
async def test_on_run_end_fires_exactly_once() -> None:
    """Reflection pass does NOT trigger another on_run_end (_reflection_fired guard)."""
    provider = FauxProvider()
    provider.set_responses(
        [
            faux_assistant_message("main"),
            faux_assistant_message("reflected"),
        ]
    )

    fire_count = 0

    class _CountFires(Middleware):
        async def on_run_end(self, ctx, *, signal=None):
            nonlocal fire_count
            fire_count += 1
            return [UserMessage(content=[TextContent(text="reflect")])]

    agent = Agent(
        model=Model(id="test", provider="faux"),
        provider=provider,
        middleware=[_CountFires()],
    )
    await agent.prompt("hi")

    assert fire_count == 1
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_on_run_end_none_does_not_add_turn() -> None:
    """Returning None from on_run_end does not trigger an extra model call."""
    provider = FauxProvider()
    provider.set_responses([faux_assistant_message("main")])

    class _NoOp(Middleware):
        async def on_run_end(self, ctx, *, signal=None):
            return None

    agent = Agent(
        model=Model(id="test", provider="faux"),
        provider=provider,
        middleware=[_NoOp()],
    )
    await agent.prompt("hi")

    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_on_run_end_injected_messages_in_history() -> None:
    """Messages injected by on_run_end appear in agent.state.messages."""
    from cubepi.providers.base import AssistantMessage

    provider = FauxProvider()
    provider.set_responses(
        [
            faux_assistant_message("main"),
            faux_assistant_message("reflection response"),
        ]
    )

    class _Inject(Middleware):
        async def on_run_end(self, ctx, *, signal=None):
            return [UserMessage(content=[TextContent(text="reflect")])]

    agent = Agent(
        model=Model(id="test", provider="faux"),
        provider=provider,
        middleware=[_Inject()],
    )
    await agent.prompt("hi")

    texts = [
        c.text
        for m in agent.state.messages
        if isinstance(m, AssistantMessage)
        for c in m.content
        if hasattr(c, "text")
    ]
    assert "main" in texts
    assert "reflection response" in texts


@pytest.mark.asyncio
async def test_on_run_end_fires_via_should_stop_after_turn() -> None:
    """on_run_end fires when should_stop_after_turn exits the inner loop."""
    from cubepi.agent.types import ShouldStopAfterTurnContext

    provider = FauxProvider()
    provider.set_responses(
        [
            faux_assistant_message("main"),
            faux_assistant_message("reflected"),
        ]
    )

    fired: list[str] = []
    _seen = 0

    class _StopAfterFirst(Middleware):
        async def should_stop_after_turn(self, ctx: ShouldStopAfterTurnContext) -> bool:
            nonlocal _seen
            _seen += 1
            return _seen == 1  # stop after first turn

    class _Reflect(Middleware):
        async def on_run_end(self, ctx, *, signal=None):
            fired.append("fired")
            return [UserMessage(content=[TextContent(text="reflect")])]

    agent = Agent(
        model=Model(id="test", provider="faux"),
        provider=provider,
        middleware=[_StopAfterFirst(), _Reflect()],
    )
    await agent.prompt("hi")

    assert fired == ["fired"]
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_on_run_end_skipped_on_error() -> None:
    """on_run_end does NOT fire when stop_reason is error/aborted."""
    provider = FauxProvider()
    err_msg = faux_assistant_message("oops")
    err_msg = err_msg.model_copy(update={"stop_reason": "error"})
    provider.set_responses([err_msg])

    fired: list[str] = []

    class _Reflect(Middleware):
        async def on_run_end(self, ctx, *, signal=None):
            fired.append("fired")
            return None

    agent = Agent(
        model=Model(id="test", provider="faux"),
        provider=provider,
        middleware=[_Reflect()],
    )
    await agent.prompt("hi")

    assert fired == []
