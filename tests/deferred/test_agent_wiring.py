from __future__ import annotations

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.deferred import DeferredToolGroup, DeferredToolsMiddleware
from cubepi.providers.base import TextContent
from cubepi.middleware.base import Middleware
from pydantic import BaseModel


class _Empty(BaseModel):
    pass


def _dummy_tool(name: str) -> AgentTool:
    async def _exec(tool_call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="ok")])

    return AgentTool(name=name, description="dummy", parameters=_Empty, execute=_exec)


def _make_group(group_id: str, tool_names: list[str]) -> DeferredToolGroup:
    async def _loader():
        return [_dummy_tool(n) for n in tool_names]

    return DeferredToolGroup(
        group_id=group_id,
        display_name="Test",
        description="desc",
        tool_names=tool_names,
        loader=_loader,
    )


def _make_faux_model():
    """Create a minimal BoundModel for Agent construction."""
    from cubepi.providers.faux import FauxProvider

    provider = FauxProvider()
    return provider.model("faux")


class TestAgentDeferredToolGroups:
    def test_agent_accepts_deferred_tool_groups(self) -> None:
        model = _make_faux_model()
        group = _make_group("mcp:github", ["t1", "t2"])
        agent = Agent(
            model=model,
            tools=[_dummy_tool("builtin")],
            deferred_tool_groups=[group],
        )
        tool_names = [t.name for t in agent._state.tools]
        assert "builtin" in tool_names
        assert "expand_tools" in tool_names

    def test_middleware_auto_created(self) -> None:
        model = _make_faux_model()
        group = _make_group("mcp:github", ["t1"])
        agent = Agent(
            model=model,
            deferred_tool_groups=[group],
        )
        assert any(isinstance(mw, DeferredToolsMiddleware) for mw in agent._middleware)

    def test_extra_ref_bound_to_agent_extra(self) -> None:
        model = _make_faux_model()
        group = _make_group("mcp:github", ["t1"])
        agent = Agent(
            model=model,
            deferred_tool_groups=[group],
        )
        deferred_mw = next(
            mw for mw in agent._middleware if isinstance(mw, DeferredToolsMiddleware)
        )
        assert deferred_mw._extra_ref() is agent._extra

    def test_no_deferred_groups_no_middleware(self) -> None:
        model = _make_faux_model()
        agent = Agent(model=model, tools=[_dummy_tool("t1")])
        assert not any(
            isinstance(mw, DeferredToolsMiddleware) for mw in agent._middleware
        )

    def test_explicit_middleware_still_works(self) -> None:
        model = _make_faux_model()
        group = _make_group("mcp:github", ["t1"])
        extra: dict = {}
        mw = DeferredToolsMiddleware(
            groups=[group],
            extra_ref=lambda: extra,
            catalog_header="Custom",
        )
        agent = Agent(
            model=model,
            middleware=[mw],
        )
        assert any(isinstance(m, DeferredToolsMiddleware) for m in agent._middleware)
        tool_names = [t.name for t in agent._state.tools]
        assert "expand_tools" in tool_names

    def test_combined_explicit_middleware_and_deferred_groups(self) -> None:
        model = _make_faux_model()
        group = _make_group("mcp:github", ["t1"])

        class _Marker(Middleware):
            pass

        marker = _Marker()
        agent = Agent(
            model=model,
            middleware=[marker],
            deferred_tool_groups=[group],
        )
        assert agent._middleware[0] is marker
        assert any(
            isinstance(mw, DeferredToolsMiddleware)
            for mw in agent._middleware
        )

    def test_empty_deferred_groups_no_middleware(self) -> None:
        model = _make_faux_model()
        agent = Agent(model=model, deferred_tool_groups=[])
        assert not any(
            isinstance(mw, DeferredToolsMiddleware)
            for mw in agent._middleware
        )
