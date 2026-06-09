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
        assert "load_tools" in tool_names

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
        assert "load_tools" in tool_names

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
        assert any(isinstance(mw, DeferredToolsMiddleware) for mw in agent._middleware)

    def test_empty_deferred_groups_no_middleware(self) -> None:
        model = _make_faux_model()
        agent = Agent(model=model, deferred_tool_groups=[])
        assert not any(
            isinstance(mw, DeferredToolsMiddleware) for mw in agent._middleware
        )

    def test_on_tools_expanded_wired(self) -> None:
        model = _make_faux_model()
        group = _make_group("mcp:github", ["t1"])
        agent = Agent(
            model=model,
            tools=[_dummy_tool("builtin")],
            deferred_tool_groups=[group],
        )
        deferred_mw = next(
            mw for mw in agent._middleware if isinstance(mw, DeferredToolsMiddleware)
        )
        assert deferred_mw._on_tools_expanded is not None

    async def test_on_tools_expanded_deduplicates(self) -> None:
        """Pre-loaded tools from resume are not duplicated by on_tools_expanded."""
        from cubepi.agent.types import AgentContext

        model = _make_faux_model()
        t1 = _dummy_tool("t1")
        group = _make_group("mcp:github", ["t1", "t2"])
        agent = Agent(
            model=model,
            tools=[t1],
            deferred_tool_groups=[group],
        )
        deferred_mw = next(
            mw for mw in agent._middleware if isinstance(mw, DeferredToolsMiddleware)
        )
        ctx = AgentContext(
            system_prompt="",
            messages=[],
            tools=list(agent._state.tools),
            extra=agent._extra,
        )
        await deferred_mw._expand(group_id="mcp:github", tool_names=None, context=ctx)
        names = [t.name for t in agent._state._tools]
        assert names.count("t1") == 1
        assert "t2" in names


class TestForkOnceDeniesMiddlewareTools:
    def test_fork_keeps_load_tools_in_schema(self) -> None:
        """load_tools stays in the tool list for prompt-cache parity."""
        from cubepi.agent.agent import _deny_in_fork

        model = _make_faux_model()
        group = _make_group("mcp:github", ["t1", "t2"])
        agent = Agent(
            model=model,
            tools=[_dummy_tool("builtin")],
            deferred_tool_groups=[group],
        )
        mw_tool_ids = {
            id(t) for mw in agent._middleware for t in getattr(mw, "tools", []) or []
        }
        fork_tools = [
            _deny_in_fork(t) if id(t) in mw_tool_ids else t for t in agent._state.tools
        ]
        fork_tool_names = [t.name for t in fork_tools]
        assert "builtin" in fork_tool_names
        assert "load_tools" in fork_tool_names

    async def test_fork_load_tools_returns_error(self) -> None:
        """load_tools in fork returns is_error=True instead of executing."""
        from cubepi.agent.agent import _deny_in_fork

        model = _make_faux_model()
        group = _make_group("mcp:github", ["t1"])
        agent = Agent(
            model=model,
            deferred_tool_groups=[group],
        )
        expand_tool = next(t for t in agent._state.tools if t.name == "load_tools")
        denied = _deny_in_fork(expand_tool)
        assert denied.name == "load_tools"
        result = await denied.execute("call-1", _Empty())
        assert result.is_error is True
        assert "not available in a forked agent" in result.content[0].text
