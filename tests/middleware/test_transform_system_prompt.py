"""transform_system_prompt hook tests (D7)."""

import pytest

from cubepi.middleware.base import Middleware, compose_middleware


class _AppendA(Middleware):
    async def transform_system_prompt(self, sp, *, signal=None):
        return sp + "\n[A]"


class _AppendB(Middleware):
    async def transform_system_prompt(self, sp, *, signal=None):
        return sp + "\n[B]"


@pytest.mark.asyncio
async def test_single_middleware_appends() -> None:
    hooks = compose_middleware([_AppendA()])
    fn = hooks["transform_system_prompt"]
    out = await fn("base")
    assert out == "base\n[A]"


@pytest.mark.asyncio
async def test_chain_order_preserved() -> None:
    """A then B → A first, then B sees A's output."""
    hooks = compose_middleware([_AppendA(), _AppendB()])
    fn = hooks["transform_system_prompt"]
    out = await fn("base")
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
        await mw.transform_system_prompt("any")
