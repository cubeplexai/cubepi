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
    def test_inject_tools_attribute_contains_load_tools_only(self) -> None:
        extra: dict = {}
        group = _make_group("mcp:a", ["t1"])
        mw = DeferredToolsMiddleware(
            groups=[group], extra_ref=lambda: extra, strategy="inject"
        )
        assert len(mw.tools) == 1
        assert mw.tools[0].name == "load_tools"


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
        mw = DeferredToolsMiddleware(
            groups=[group], extra_ref=lambda: extra, strategy="inject"
        )
        ctx = AgentContext(system_prompt="", messages=[])
        result = await mw.transform_system_prompt("base", ctx=ctx)
        assert "mcp:github" not in result

    async def test_inject_system_prompt_has_no_schema_section(self) -> None:
        """Schemas live solely in the tools array — no double rendering."""
        extra: dict = {}
        group = _make_group("mcp:a", ["t1"])
        mw = DeferredToolsMiddleware(
            groups=[group], extra_ref=lambda: extra, strategy="inject"
        )
        ctx = AgentContext(
            system_prompt="base", messages=[], tools=list(mw.tools), extra=extra
        )
        await mw._expand(group_id="mcp:a", tool_names=None, context=ctx)
        prompt = await mw.transform_system_prompt("base", ctx=ctx)
        assert "Expanded tool groups" not in prompt
        assert '"parameters"' not in prompt
        # Injection itself still works — tools are model-visible.
        assert any(t.name == "t1" and t.expose_to_model for t in ctx.tools)


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
        assert len(context_tools) == 3  # load_tools + 2 new
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
        assert len(context_tools) == 2  # load_tools + 1 new
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

        # Insertion-ordered expansion state survives for resume fidelity.
        assert list(extra["expanded_groups"].keys()) == ["mcp:z", "mcp:a"]


class TestPrepareResumedState:
    async def test_fully_expanded_group(self) -> None:
        group = _make_group("mcp:github", ["t1", "t2"])
        expanded: dict[str, list[str] | None] = {"mcp:github": None}
        resumed = await DeferredToolsMiddleware.prepare_resumed_state(
            groups=[group],
            expanded=expanded,
            strategy="inject",
        )
        assert len(resumed.pre_loaded_tools) == 2
        assert all(t.expose_to_model for t in resumed.pre_loaded_tools)
        assert len(resumed.remaining_groups) == 0

    async def test_partially_expanded_group(self) -> None:
        group = _make_group("mcp:github", ["t1", "t2", "t3"])
        expanded: dict[str, list[str] | None] = {"mcp:github": ["t1"]}
        resumed = await DeferredToolsMiddleware.prepare_resumed_state(
            groups=[group],
            expanded=expanded,
            strategy="inject",
        )
        assert len(resumed.pre_loaded_tools) == 1
        assert resumed.pre_loaded_tools[0].name == "t1"
        assert len(resumed.remaining_groups) == 1

    async def test_unexpanded_group_stays_deferred(self) -> None:
        group = _make_group("mcp:github", ["t1"])
        expanded: dict[str, list[str] | None] = {}
        resumed = await DeferredToolsMiddleware.prepare_resumed_state(
            groups=[group],
            expanded=expanded,
            strategy="inject",
        )
        assert len(resumed.pre_loaded_tools) == 0
        assert len(resumed.remaining_groups) == 1

    async def test_resumed_state_seeds_middleware(self) -> None:
        group_a = _make_group("mcp:github", ["t1", "t2"])
        group_b = _make_group("mcp:slack", ["s1", "s2"])
        expanded: dict[str, list[str] | None] = {
            "mcp:github": None,
            "mcp:slack": ["s1"],
        }
        resumed = await DeferredToolsMiddleware.prepare_resumed_state(
            groups=[group_a, group_b],
            expanded=expanded,
            strategy="inject",
        )
        assert {t.name for t in resumed.pre_loaded_tools} == {"t1", "t2", "s1"}
        # Fully expanded github leaves the catalog; partial slack stays.
        assert [g.group_id for g in resumed.remaining_groups] == ["mcp:slack"]

        extra: dict[str, object] = {"expanded_groups": dict(expanded)}
        mw = DeferredToolsMiddleware(
            groups=resumed.remaining_groups,
            extra_ref=lambda: extra,
            strategy="inject",
            resumed_loader_cache=resumed.loader_cache,
        )
        ctx = AgentContext(
            system_prompt="base",
            messages=[],
            tools=list(mw.tools) + resumed.pre_loaded_tools,
            extra=extra,
        )
        prompt = await mw.transform_system_prompt("base", ctx=ctx)
        assert "s2" in prompt  # remaining tool still advertised
        assert "Expanded tool groups" not in prompt

    async def test_loader_cache_populated(self) -> None:
        group = _make_group("mcp:github", ["t1", "t2"])
        expanded: dict[str, list[str] | None] = {"mcp:github": None}
        resumed = await DeferredToolsMiddleware.prepare_resumed_state(
            groups=[group],
            expanded=expanded,
            strategy="inject",
        )
        assert "mcp:github" in resumed.loader_cache
        assert len(resumed.loader_cache["mcp:github"]) == 2

    async def test_resumed_loader_cache_skips_reload(self) -> None:
        call_count = [0]
        group = _make_group("mcp:github", ["t1"], loader_call_count=call_count)
        expanded: dict[str, list[str] | None] = {"mcp:github": ["t1"]}
        resumed = await DeferredToolsMiddleware.prepare_resumed_state(
            groups=[group],
            expanded=expanded,
            strategy="inject",
        )
        assert call_count[0] == 1

        extra: dict = {"expanded_groups": dict(expanded)}
        mw = DeferredToolsMiddleware(
            groups=resumed.remaining_groups,
            extra_ref=lambda: extra,
            strategy="inject",
            resumed_loader_cache=resumed.loader_cache,
        )
        ctx = AgentContext(
            system_prompt="",
            messages=[],
            tools=list(mw.tools) + resumed.pre_loaded_tools,
            extra=extra,
        )
        # Expanding remaining tools should NOT call loader again.
        await mw._expand(group_id="mcp:github", tool_names=None, context=ctx)
        assert call_count[0] == 1  # still 1, not 2


class TestOnToolsExpandedCallback:
    async def test_callback_invoked_on_expansion(self) -> None:
        expanded_tools: list[AgentTool] = []
        extra: dict = {}
        group = _make_group("mcp:github", ["t1", "t2"])
        mw = DeferredToolsMiddleware(
            groups=[group],
            extra_ref=lambda: extra,
            on_tools_expanded=lambda tools: expanded_tools.extend(tools),
        )
        ctx = AgentContext(
            system_prompt="",
            messages=[],
            tools=list(mw.tools),
            extra=extra,
        )
        await mw._expand(group_id="mcp:github", tool_names=None, context=ctx)
        assert len(expanded_tools) == 2
        assert {t.name for t in expanded_tools} == {"t1", "t2"}

    async def test_callback_not_invoked_on_idempotent(self) -> None:
        call_count = [0]

        def _on_expanded(tools: list[AgentTool]) -> None:
            call_count[0] += 1

        extra: dict = {}
        group = _make_group("mcp:github", ["t1"])
        mw = DeferredToolsMiddleware(
            groups=[group],
            extra_ref=lambda: extra,
            on_tools_expanded=_on_expanded,
        )
        ctx = AgentContext(
            system_prompt="",
            messages=[],
            tools=list(mw.tools),
            extra=extra,
        )
        await mw._expand(group_id="mcp:github", tool_names=None, context=ctx)
        assert call_count[0] == 1
        await mw._expand(group_id="mcp:github", tool_names=None, context=ctx)
        assert call_count[0] == 1  # no new tools, no callback


class TestAfterToolCallHook:
    """Tests that exercise the after_tool_call hook directly (not via _expand)."""

    async def test_drains_pending_on_load_tools_call(self) -> None:
        from cubepi.agent.types import AfterToolCallContext
        from cubepi.deferred._expand_tool import TOOL_NAME
        from cubepi.providers.base import AssistantMessage, ToolCall

        extra: dict = {}
        group = _make_group("mcp:github", ["t1", "t2"])
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)

        context_tools: list[AgentTool] = [mw.tools[0]]
        ctx = AgentContext(
            system_prompt="",
            messages=[],
            tools=context_tools,
            extra=extra,
        )

        await mw._expand_callback("mcp:github", None)
        assert len(mw._pending_injection) == 2

        atc_ctx = AfterToolCallContext(
            assistant_message=AssistantMessage(content=[]),
            tool_call=ToolCall(id="c1", name=TOOL_NAME, arguments={}),
            args={},
            result=AgentToolResult(content=[TextContent(text="ok")]),
            is_error=False,
            context=ctx,
        )
        await mw.after_tool_call(atc_ctx)
        assert len(context_tools) == 3
        assert len(mw._pending_injection) == 0

    async def test_skips_non_load_tools_call(self) -> None:
        from cubepi.agent.types import AfterToolCallContext
        from cubepi.providers.base import AssistantMessage, ToolCall

        extra: dict = {}
        group = _make_group("mcp:github", ["t1"])
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)
        mw._pending_injection.append(_dummy_tool("t1"))

        ctx = AgentContext(
            system_prompt="",
            messages=[],
            tools=list(mw.tools),
            extra=extra,
        )
        atc_ctx = AfterToolCallContext(
            assistant_message=AssistantMessage(content=[]),
            tool_call=ToolCall(id="c1", name="other_tool", arguments={}),
            args={},
            result=AgentToolResult(content=[TextContent(text="ok")]),
            is_error=False,
            context=ctx,
        )
        await mw.after_tool_call(atc_ctx)
        assert len(mw._pending_injection) == 1  # not drained

    async def test_skips_error_result(self) -> None:
        from cubepi.agent.types import AfterToolCallContext
        from cubepi.deferred._expand_tool import TOOL_NAME
        from cubepi.providers.base import AssistantMessage, ToolCall

        extra: dict = {}
        group = _make_group("mcp:github", ["t1"])
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)
        mw._pending_injection.append(_dummy_tool("t1"))

        ctx = AgentContext(
            system_prompt="",
            messages=[],
            tools=list(mw.tools),
            extra=extra,
        )
        atc_ctx = AfterToolCallContext(
            assistant_message=AssistantMessage(content=[]),
            tool_call=ToolCall(id="c1", name=TOOL_NAME, arguments={}),
            args={},
            result=AgentToolResult(content=[TextContent(text="err")], is_error=True),
            is_error=True,
            context=ctx,
        )
        await mw.after_tool_call(atc_ctx)
        assert len(mw._pending_injection) == 1  # not drained


class TestSelectiveExpandAfterFullExpand:
    async def test_selective_after_full_is_noop(self) -> None:
        """Selective expand on a fully-expanded group is a no-op (L158 coverage)."""
        extra: dict = {}
        group = _make_group("mcp:github", ["t1", "t2"])
        mw = DeferredToolsMiddleware(groups=[group], extra_ref=lambda: extra)
        context_tools: list[AgentTool] = [mw.tools[0]]
        ctx = AgentContext(
            system_prompt="",
            messages=[],
            tools=context_tools,
            extra=extra,
        )
        await mw._expand(group_id="mcp:github", tool_names=None, context=ctx)
        assert extra["expanded_groups"]["mcp:github"] is None
        assert len(context_tools) == 3

        output = await mw._expand(
            group_id="mcp:github",
            tool_names=["t1"],
            context=ctx,
        )
        assert output.expanded is True
        assert output.remaining == 0
        assert extra["expanded_groups"]["mcp:github"] is None
        assert len(context_tools) == 3


class TestExpandedGroupsDictStability:
    async def test_shared_dict_mutated_in_place(self) -> None:
        """expanded_groups dict in extra is mutated in place, not replaced."""
        extra: dict = {}
        g1 = _make_group("mcp:a", ["t1"])
        g2 = _make_group("mcp:b", ["t2"])
        mw = DeferredToolsMiddleware(groups=[g1, g2], extra_ref=lambda: extra)
        ctx = AgentContext(
            system_prompt="",
            messages=[],
            tools=list(mw.tools),
            extra=extra,
        )
        await mw._expand(group_id="mcp:a", tool_names=None, context=ctx)
        dict_ref = extra["expanded_groups"]
        await mw._expand(group_id="mcp:b", tool_names=None, context=ctx)
        # Same dict object, not a new one.
        assert extra["expanded_groups"] is dict_ref
        assert set(dict_ref.keys()) == {"mcp:a", "mcp:b"}
