"""after_model_response hook + TurnAction tests (D8)."""

import pytest

from cubepi import Agent, Model
from cubepi.agent.types import AgentContext
from cubepi.middleware.base import Middleware, TurnAction, compose_middleware
from cubepi.providers.base import (
    AssistantMessage,
    TextContent,
    Usage,
    UserMessage,
)
from cubepi.providers.faux import FauxProvider, faux_assistant_message


def _mk_response(text: str = "hi") -> AssistantMessage:
    return AssistantMessage(content=[TextContent(text=text)], usage=Usage())


def _mk_ctx() -> AgentContext:
    return AgentContext(system_prompt="", messages=[])


class _MutateResponse(Middleware):
    async def after_model_response(self, response, ctx, *, signal=None):
        return TurnAction(response=_mk_response(text="mutated"))


class _InjectMessages(Middleware):
    async def after_model_response(self, response, ctx, *, signal=None):
        return TurnAction(
            inject_messages=[UserMessage(content=[TextContent(text="injected")])]
        )


class _Stop(Middleware):
    async def after_model_response(self, response, ctx, *, signal=None):
        return TurnAction(decision="stop")


class _Loop(Middleware):
    async def after_model_response(self, response, ctx, *, signal=None):
        return TurnAction(decision="loop_to_model")


class _NoOp(Middleware):
    async def after_model_response(self, response, ctx, *, signal=None):
        return None


def test_turn_action_defaults() -> None:
    ta = TurnAction()
    assert ta.response is None
    assert ta.inject_messages == []
    assert ta.decision == "natural"


@pytest.mark.asyncio
async def test_single_middleware_mutates_response() -> None:
    hooks = compose_middleware([_MutateResponse()])
    result = await hooks["after_model_response"](_mk_response("orig"), _mk_ctx())
    assert isinstance(result.response, AssistantMessage)
    assert result.response.content[0].text == "mutated"


@pytest.mark.asyncio
async def test_chain_last_response_wins() -> None:
    """Two mutators; last one in chain wins for response."""

    class _MutateAgain(Middleware):
        async def after_model_response(self, response, ctx, *, signal=None):
            return TurnAction(response=_mk_response(text="final"))

    hooks = compose_middleware([_MutateResponse(), _MutateAgain()])
    result = await hooks["after_model_response"](_mk_response("orig"), _mk_ctx())
    assert result.response.content[0].text == "final"


@pytest.mark.asyncio
async def test_inject_messages_concatenate() -> None:
    """inject_messages from multiple middleware concatenate."""

    class _InjectMore(Middleware):
        async def after_model_response(self, response, ctx, *, signal=None):
            return TurnAction(
                inject_messages=[UserMessage(content=[TextContent(text="more")])]
            )

    hooks = compose_middleware([_InjectMessages(), _InjectMore()])
    result = await hooks["after_model_response"](_mk_response(), _mk_ctx())
    assert len(result.inject_messages) == 2


@pytest.mark.asyncio
async def test_decision_last_wins() -> None:
    """Last middleware's decision wins."""
    hooks = compose_middleware([_Stop(), _Loop()])
    result = await hooks["after_model_response"](_mk_response(), _mk_ctx())
    assert result.decision == "loop_to_model"


@pytest.mark.asyncio
async def test_none_return_treated_as_natural() -> None:
    """Middleware returning None doesn't affect the composed TurnAction."""
    hooks = compose_middleware([_NoOp(), _Stop()])
    result = await hooks["after_model_response"](_mk_response(), _mk_ctx())
    assert result.decision == "stop"


@pytest.mark.asyncio
async def test_default_implementation_raises() -> None:
    mw = Middleware()
    with pytest.raises(NotImplementedError):
        await mw.after_model_response(_mk_response(), _mk_ctx())


def test_no_middleware_hook_absent() -> None:
    class Plain(Middleware):
        pass

    hooks = compose_middleware([Plain()])
    assert "after_model_response" not in hooks


@pytest.mark.asyncio
async def test_agent_stops_when_middleware_returns_stop() -> None:
    """decision='stop' terminates after the first model response."""
    provider = FauxProvider()
    # Two responses queued; second should never fire because we stop after first.
    provider.set_responses([
        faux_assistant_message("first"),
        faux_assistant_message("should not fire"),
    ])
    agent = Agent(
        model=Model(id="test", provider="faux"),
        provider=provider,
        middleware=[_Stop()],
    )
    await agent.prompt("hi")
    # Only the first response was consumed; one remains in the queue.
    assert provider.call_count == 1
    assert provider.pending_response_count == 1


@pytest.mark.asyncio
async def test_agent_loops_when_middleware_returns_loop_to_model() -> None:
    """decision='loop_to_model' re-invokes the model with inject_messages."""
    provider = FauxProvider()
    provider.set_responses([
        faux_assistant_message("first"),
        faux_assistant_message("second"),
    ])

    call_count = 0

    class _LoopOnce(Middleware):
        async def after_model_response(self, response, ctx, *, signal=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return TurnAction(
                    decision="loop_to_model",
                    inject_messages=[
                        UserMessage(content=[TextContent(text="retry")])
                    ],
                )
            return None

    agent = Agent(
        model=Model(id="test", provider="faux"),
        provider=provider,
        middleware=[_LoopOnce()],
    )
    await agent.prompt("hi")
    # Both responses consumed: provider called twice.
    assert provider.call_count == 2
    # Hook ran twice: once forced loop, once natural.
    assert call_count == 2


@pytest.mark.asyncio
async def test_agent_no_middleware_natural_flow() -> None:
    """Without after_model_response middleware, natural flow proceeds."""
    provider = FauxProvider()
    provider.set_responses([faux_assistant_message("ok")])
    agent = Agent(
        model=Model(id="test", provider="faux"),
        provider=provider,
    )

    events: list[str] = []

    def _listener(event, signal):
        events.append(event.type)

    agent.subscribe(_listener)
    await agent.prompt("hi")
    # Natural flow: agent_end fires and no second provider call.
    assert "agent_end" in events
    assert provider.call_count == 1
