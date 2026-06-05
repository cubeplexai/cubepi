"""transform_system_prompt hook tests (D7)."""

import pytest

from cubepi import Agent
from cubepi.agent.types import AgentContext
from cubepi.middleware.base import Middleware, compose_middleware
from cubepi.providers.faux import FauxProvider, faux_assistant_message


class _AppendA(Middleware):
    async def transform_system_prompt(self, sp, *, ctx, signal=None):
        del ctx, signal
        return sp + "\n[A]"


class _AppendB(Middleware):
    async def transform_system_prompt(self, sp, *, ctx, signal=None):
        del ctx, signal
        return sp + "\n[B]"


@pytest.mark.asyncio
async def test_single_middleware_appends() -> None:
    hooks = compose_middleware([_AppendA()])
    fn = hooks["transform_system_prompt"]
    out = await fn("base", ctx=AgentContext(system_prompt="", messages=[]))
    assert out == "base\n[A]"


@pytest.mark.asyncio
async def test_chain_order_preserved() -> None:
    """A then B → A first, then B sees A's output."""
    hooks = compose_middleware([_AppendA(), _AppendB()])
    fn = hooks["transform_system_prompt"]
    out = await fn("base", ctx=AgentContext(system_prompt="", messages=[]))
    assert out == "base\n[A]\n[B]"


def test_no_middleware_hook_absent() -> None:
    """If no middleware implements transform_system_prompt, the hook is not in the dict."""

    class Plain(Middleware):
        pass

    hooks = compose_middleware([Plain()])
    assert "transform_system_prompt" not in hooks


@pytest.mark.asyncio
async def test_default_implementation_raises() -> None:
    """Default Middleware.transform_system_prompt raises NotImplementedError."""
    mw = Middleware()
    with pytest.raises(NotImplementedError):
        await mw.transform_system_prompt(
            "any", ctx=AgentContext(system_prompt="", messages=[])
        )


@pytest.mark.asyncio
async def test_agent_applies_transform_system_prompt() -> None:
    """system_prompt sent to provider must reflect the middleware chain."""
    captured: list[str] = []

    class _Capturing(Middleware):
        async def transform_system_prompt(self, sp: str, *, ctx, signal=None) -> str:
            del ctx, signal
            # last middleware in chain captures what it saw
            captured.append(sp)
            return sp + "\n[C]"

    provider = FauxProvider(provider_id="faux")
    provider.set_responses([faux_assistant_message("ok")])
    agent = Agent(
        model=provider.model("test"),
        system_prompt="base",
        middleware=[_AppendA(), _AppendB(), _Capturing()],
    )
    await agent.prompt("hi")

    # _Capturing saw the post-A, post-B prompt
    assert len(captured) == 1
    assert captured[0] == "base\n[A]\n[B]"


@pytest.mark.asyncio
async def test_agent_without_middleware_passes_system_prompt_unchanged() -> None:
    """No transform_system_prompt middleware → provider sees raw system_prompt."""
    received: list[str] = []

    orig_stream = FauxProvider.stream

    async def _capturing_stream(self, model, messages, *, system_prompt="", **kw):  # type: ignore[override]
        received.append(system_prompt)
        return await orig_stream(
            self, model, messages, system_prompt=system_prompt, **kw
        )

    provider = FauxProvider(provider_id="faux")
    provider.set_responses([faux_assistant_message("ok")])
    provider.stream = _capturing_stream.__get__(provider, FauxProvider)  # type: ignore[method-assign]

    agent = Agent(
        model=provider.model("test"),
        system_prompt="base",
    )
    await agent.prompt("hi")
    assert received == ["base"]
