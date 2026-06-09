from __future__ import annotations

from cubepi.agent.types import AgentContext, AgentTool, AgentToolResult
from cubepi.deferred.middleware import DeferredToolsMiddleware
from cubepi.deferred.types import DeferredToolGroup
from cubepi.providers.base import TextContent


def _dummy_tool(name: str, description: str = "dummy") -> AgentTool:
    from pydantic import BaseModel

    class _Empty(BaseModel):
        pass

    async def _exec(tool_call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="ok")])

    return AgentTool(
        name=name, description=description, parameters=_Empty, execute=_exec
    )


def _make_group(
    group_id: str,
    tool_names: list[str],
    *,
    display_name: str = "Test",
    description: str = "desc",
    loader_tools: list[AgentTool] | None = None,
    loader_call_count: list[int] | None = None,
) -> DeferredToolGroup:
    tools = loader_tools or [_dummy_tool(n) for n in tool_names]
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


class TestMiddlewareConstruction:
    def test_tools_attribute_contains_expand_tools(self) -> None:
        extra: dict = {}
        group = _make_group("mcp:a", ["t1"])
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)
        assert len(mw.tools) == 1
        assert mw.tools[0].name == "expand_tools"


class TestTransformSystemPrompt:
    async def test_appends_catalog(self) -> None:
        extra: dict = {}
        group = _make_group("mcp:github", ["create_issue", "search_repos"])
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)
        ctx = AgentContext(system_prompt="base", messages=[])
        result = await mw.transform_system_prompt("base prompt", ctx=ctx)
        assert "mcp:github" in result
        assert "create_issue" in result
        assert "base prompt" in result

    async def test_fully_expanded_group_omitted_from_catalog(self) -> None:
        extra: dict = {"expanded_groups": {"mcp:github": None}}
        group = _make_group("mcp:github", ["t1"])
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)
        ctx = AgentContext(system_prompt="", messages=[])
        result = await mw.transform_system_prompt("base", ctx=ctx)
        assert "mcp:github" not in result or "Expanded tool groups" in result

    async def test_expanded_schemas_appended_after_catalog(self) -> None:
        extra: dict = {"expanded_groups": {"mcp:a": None}}
        group = _make_group("mcp:a", ["t1"])
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)
        mw._expanded_schemas.append(
            ("mcp:a", [{"name": "t1", "description": "Tool 1", "parameters": {}}])
        )
        ctx = AgentContext(system_prompt="", messages=[])
        result = await mw.transform_system_prompt("base", ctx=ctx)
        assert "Expanded tool groups" in result
        assert "t1" in result


class TestAfterToolCallExpansion:
    async def test_expand_all_injects_tools(self) -> None:
        extra: dict = {}
        group = _make_group("mcp:github", ["create_issue", "search_repos"])
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)

        context_tools: list[AgentTool] = [mw.tools[0]]
        ctx = AgentContext(
            system_prompt="",
            messages=[],
            tools=context_tools,
            extra=extra,
        )

        output = await mw._expand(
            group_id="mcp:github",
            tool_names=None,
            context=ctx,
        )
        assert output.expanded is True
        assert len(output.tool_names) == 2
        assert output.remaining == 0
        assert len(context_tools) == 3  # expand_tools + 2 new
        assert extra["expanded_groups"] == {"mcp:github": None}

    async def test_expand_selective_injects_only_requested(self) -> None:
        extra: dict = {}
        group = _make_group(
            "mcp:github",
            ["create_issue", "search_repos", "create_pr"],
        )
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)

        context_tools: list[AgentTool] = [mw.tools[0]]
        ctx = AgentContext(
            system_prompt="",
            messages=[],
            tools=context_tools,
            extra=extra,
        )

        output = await mw._expand(
            group_id="mcp:github",
            tool_names=["create_issue"],
            context=ctx,
        )
        assert output.expanded is True
        assert output.tool_names == ["create_issue"]
        assert output.remaining == 2
        assert len(context_tools) == 2  # expand_tools + 1 new
        assert extra["expanded_groups"] == {"mcp:github": ["create_issue"]}

    async def test_incremental_expand_same_group(self) -> None:
        extra: dict = {}
        call_count = [0]
        group = _make_group(
            "mcp:github",
            ["t1", "t2", "t3"],
            loader_call_count=call_count,
        )
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)

        context_tools: list[AgentTool] = [mw.tools[0]]
        ctx = AgentContext(
            system_prompt="",
            messages=[],
            tools=context_tools,
            extra=extra,
        )

        await mw._expand(group_id="mcp:github", tool_names=["t1"], context=ctx)
        assert len(context_tools) == 2
        assert extra["expanded_groups"] == {"mcp:github": ["t1"]}

        await mw._expand(group_id="mcp:github", tool_names=["t2"], context=ctx)
        assert len(context_tools) == 3
        assert extra["expanded_groups"] == {"mcp:github": ["t1", "t2"]}

        assert call_count[0] == 1

    async def test_expand_unknown_group_returns_error(self) -> None:
        extra: dict = {}
        mw = DeferredToolsMiddleware(groups=[], extra_ref=lambda: extra)
        ctx = AgentContext(
            system_prompt="",
            messages=[],
            tools=[],
            extra=extra,
        )

        output = await mw._expand(
            group_id="bad:id",
            tool_names=None,
            context=ctx,
        )
        assert output.expanded is False
        assert output.error is not None
        assert "expanded_groups" not in extra

    async def test_expand_idempotent_no_duplicate(self) -> None:
        extra: dict = {}
        group = _make_group("mcp:github", ["t1"])
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)

        context_tools: list[AgentTool] = [mw.tools[0]]
        ctx = AgentContext(
            system_prompt="",
            messages=[],
            tools=context_tools,
            extra=extra,
        )

        await mw._expand(
            group_id="mcp:github",
            tool_names=None,
            context=ctx,
        )
        assert len(context_tools) == 2

        await mw._expand(
            group_id="mcp:github",
            tool_names=None,
            context=ctx,
        )
        assert len(context_tools) == 2

    async def test_loader_failure_returns_error(self) -> None:
        async def _failing_loader() -> list[AgentTool]:
            raise RuntimeError("connection refused")

        group = DeferredToolGroup(
            group_id="mcp:broken",
            display_name="Broken",
            description="desc",
            tool_names=["t1"],
            loader=_failing_loader,
        )
        extra: dict = {}
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)
        ctx = AgentContext(
            system_prompt="",
            messages=[],
            tools=[],
            extra=extra,
        )

        output = await mw._expand(
            group_id="mcp:broken",
            tool_names=None,
            context=ctx,
        )
        assert output.expanded is False
        assert "connection refused" in (output.error or "")
        assert "expanded_groups" not in extra


class TestExpansionOrderPreserved:
    async def test_expansion_order_in_schemas(self) -> None:
        extra: dict = {}
        g1 = _make_group("mcp:z", ["tz"])
        g2 = _make_group("mcp:a", ["ta"])
        mw = DeferredToolsMiddleware(groups=[g1, g2], extra_ref=lambda: extra)

        ctx = AgentContext(
            system_prompt="",
            messages=[],
            tools=list(mw.tools),
            extra=extra,
        )

        await mw._expand(group_id="mcp:z", tool_names=None, context=ctx)
        await mw._expand(group_id="mcp:a", tool_names=None, context=ctx)

        assert list(extra["expanded_groups"].keys()) == ["mcp:z", "mcp:a"]
        assert len(mw._expanded_schemas) == 2
        assert mw._expanded_schemas[0][0] == "mcp:z"
        assert mw._expanded_schemas[1][0] == "mcp:a"


class TestPrepareResumedState:
    async def test_fully_expanded_group(self) -> None:
        group = _make_group("mcp:github", ["t1", "t2"])
        expanded: dict[str, list[str] | None] = {"mcp:github": None}
        resumed = await DeferredToolsMiddleware.prepare_resumed_state(
            groups=[group],
            expanded=expanded,
        )
        assert len(resumed.pre_loaded_tools) == 2
        assert len(resumed.remaining_groups) == 0
        assert len(resumed.expanded_schemas) == 1
        assert resumed.expanded_schemas[0][0] == "mcp:github"
        assert len(resumed.expanded_schemas[0][1]) == 2

    async def test_partially_expanded_group(self) -> None:
        group = _make_group("mcp:github", ["t1", "t2", "t3"])
        expanded: dict[str, list[str] | None] = {"mcp:github": ["t1"]}
        resumed = await DeferredToolsMiddleware.prepare_resumed_state(
            groups=[group],
            expanded=expanded,
        )
        assert len(resumed.pre_loaded_tools) == 1
        assert resumed.pre_loaded_tools[0].name == "t1"
        assert len(resumed.remaining_groups) == 1
        assert len(resumed.expanded_schemas) == 1
        assert len(resumed.expanded_schemas[0][1]) == 1

    async def test_unexpanded_group_stays_deferred(self) -> None:
        group = _make_group("mcp:github", ["t1"])
        expanded: dict[str, list[str] | None] = {}
        resumed = await DeferredToolsMiddleware.prepare_resumed_state(
            groups=[group],
            expanded=expanded,
        )
        assert len(resumed.pre_loaded_tools) == 0
        assert len(resumed.remaining_groups) == 1
        assert len(resumed.expanded_schemas) == 0

    async def test_resumed_schemas_seed_middleware(self) -> None:
        group_a = _make_group("mcp:github", ["t1", "t2"])
        group_b = _make_group("mcp:slack", ["s1"])
        expanded: dict[str, list[str] | None] = {
            "mcp:github": None,
            "mcp:slack": ["s1"],
        }
        resumed = await DeferredToolsMiddleware.prepare_resumed_state(
            groups=[group_a, group_b],
            expanded=expanded,
        )
        extra: dict[str, object] = {"expanded_groups": dict(expanded)}
        mw = DeferredToolsMiddleware(
            groups=resumed.remaining_groups,
            extra_ref=lambda: extra,
            resumed_schemas=resumed.expanded_schemas,
        )
        ctx = AgentContext(
            system_prompt="base",
            messages=[],
            tools=list(mw.tools) + resumed.pre_loaded_tools,
            extra=extra,
        )
        prompt = await mw.transform_system_prompt("base", ctx=ctx)
        assert "mcp:github" in prompt
        assert "mcp:slack" in prompt
        assert "t1" in prompt
