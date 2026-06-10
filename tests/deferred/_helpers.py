"""Shared test fixtures for the deferred-tools suite."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.deferred import DeferredToolGroup, DeferredToolsMiddleware
from cubepi.providers.base import TextContent


class _Empty(BaseModel):
    pass


class _EchoArgs(BaseModel):
    value: str


def _dummy_tool(
    name: str, description: str = "dummy", *, expose: bool = True
) -> AgentTool:
    async def _exec(tool_call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="ok")])

    return AgentTool(
        name=name,
        description=description,
        parameters=_Empty,
        execute=_exec,
        expose_to_model=expose,
    )


def _echo_tool(name: str) -> AgentTool:
    async def _exec(tool_call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=f"echo:{args.value}")])

    return AgentTool(name=name, description="echo", parameters=_EchoArgs, execute=_exec)


def _make_group(
    group_id: str,
    tool_names: list[str],
    *,
    display_name: str = "Test",
    description: str = "desc",
    tool_factory: Callable[[str], AgentTool] = _dummy_tool,
    loader_tools: list[AgentTool] | None = None,
    loader_call_count: list[int] | None = None,
) -> DeferredToolGroup:
    tools = loader_tools or [tool_factory(n) for n in tool_names]
    call_count = loader_call_count if loader_call_count is not None else [0]

    async def _loader() -> list[AgentTool]:
        call_count[0] += 1
        return list(tools)

    return DeferredToolGroup(
        group_id=group_id,
        display_name=display_name,
        description=description,
        tool_names=tool_names,
        loader=_loader,
    )


def _mw(groups, *, strategy="dispatch", extra=None) -> DeferredToolsMiddleware:
    extra = extra if extra is not None else {}
    return DeferredToolsMiddleware(
        groups=groups, extra_ref=lambda: extra, strategy=strategy
    )


def _make_faux_model():
    """Create a minimal BoundModel for Agent construction."""
    from cubepi.providers.faux import FauxProvider

    provider = FauxProvider()
    return provider.model("faux")
