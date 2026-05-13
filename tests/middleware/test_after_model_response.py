"""after_model_response hook + TurnAction tests (D8)."""

import pytest

from cubepi.agent.types import AgentContext
from cubepi.middleware.base import Middleware, TurnAction, compose_middleware
from cubepi.providers.base import (
    AssistantMessage,
    TextContent,
    Usage,
    UserMessage,
)


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
