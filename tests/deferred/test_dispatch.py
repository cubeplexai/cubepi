from __future__ import annotations

from pydantic import BaseModel

from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.providers.base import TextContent


class _Empty(BaseModel):
    pass


def _dummy_tool(name: str, *, expose: bool = True) -> AgentTool:
    async def _exec(tool_call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="ok")])

    return AgentTool(
        name=name,
        description="dummy",
        parameters=_Empty,
        execute=_exec,
        expose_to_model=expose,
    )


class TestExposeToModel:
    def test_default_is_true(self) -> None:
        async def _exec(tool_call_id, args, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="ok")])

        tool = AgentTool(name="t", description="d", parameters=_Empty, execute=_exec)
        assert tool.expose_to_model is True

    def test_hidden_tool_excluded_from_payload_filter(self) -> None:
        tools = [_dummy_tool("visible"), _dummy_tool("hidden", expose=False)]
        visible = [t.to_definition() for t in tools if t.expose_to_model]
        assert [d.name for d in visible] == ["visible"]


import json

from cubepi.agent.types import AgentContext
from cubepi.deferred import DeferredToolGroup, DeferredToolsMiddleware
from cubepi.providers.base import ToolCall


class _EchoArgs(BaseModel):
    value: str


def _echo_tool(name: str) -> AgentTool:
    async def _exec(tool_call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=f"echo:{args.value}")])

    return AgentTool(name=name, description="echo", parameters=_EchoArgs, execute=_exec)


def _make_group(gid: str, names: list[str]) -> DeferredToolGroup:
    async def _loader():
        return [_echo_tool(n) for n in names]

    return DeferredToolGroup(
        group_id=gid,
        display_name="Test",
        description="desc",
        tool_names=names,
        loader=_loader,
    )


def _mw(groups, *, strategy="dispatch", extra=None):
    extra = extra if extra is not None else {}
    return DeferredToolsMiddleware(
        groups=groups, extra_ref=lambda: extra, strategy=strategy
    )


class TestDispatchStrategy:
    def test_tools_attr_has_both_builtins(self) -> None:
        mw = _mw([_make_group("g", ["t1"])])
        names = [t.name for t in mw.tools]
        assert names == ["load_tools", "deferred_tool_call"]

    def test_inject_strategy_has_only_load_tools(self) -> None:
        mw = _mw([_make_group("g", ["t1"])], strategy="inject")
        assert [t.name for t in mw.tools] == ["load_tools"]

    async def test_system_prompt_static_across_expansion(self) -> None:
        mw = _mw([_make_group("g", ["t1", "t2"])])
        ctx = AgentContext(system_prompt="base", messages=[], tools=list(mw.tools))
        before = await mw.transform_system_prompt("base", ctx=ctx)
        await mw._expand_callback("g", None)
        after = await mw.transform_system_prompt("base", ctx=ctx)
        assert before == after  # byte-identical — the headline property

    async def test_load_tools_result_carries_schemas(self) -> None:
        mw = _mw([_make_group("g", ["t1"])])
        out = await mw._expand_callback("g", None)
        assert out.expanded is True
        assert out.schemas is not None
        assert out.schemas[0]["name"] == "t1"

    async def test_load_tools_idempotent_and_deterministic(self) -> None:
        mw = _mw([_make_group("g", ["t1"])])
        first = await mw._expand_callback("g", None)
        second = await mw._expand_callback("g", None)
        # Compaction self-rescue: repeat calls serialize byte-identically.
        assert json.dumps(first.schemas, sort_keys=True) == json.dumps(
            second.schemas, sort_keys=True
        )

    async def test_loaded_tools_enter_context_hidden(self) -> None:
        mw = _mw([_make_group("g", ["t1"])])
        ctx = AgentContext(system_prompt="", messages=[], tools=list(mw.tools))
        await mw._expand(group_id="g", tool_names=None, context=ctx)
        loaded = next(t for t in ctx.tools if t.name == "t1")
        assert loaded.expose_to_model is False

    async def test_resolver_dispatches_with_implicit_load(self) -> None:
        mw = _mw([_make_group("g", ["t1"])])
        ctx = AgentContext(system_prompt="", messages=[], tools=list(mw.tools))
        call = ToolCall(
            id="tc-1",
            name="deferred_tool_call",
            arguments={"tool_name": "t1", "arguments": {"value": "hi"}},
        )
        rewritten = await mw.resolve_tool_call(call, context=ctx)
        assert rewritten is not None
        assert rewritten.id == "tc-1"
        assert rewritten.name == "t1"
        assert rewritten.arguments == {"value": "hi"}
        # Implicit load appended the hidden tool so the pipeline can find it.
        assert any(t.name == "t1" and not t.expose_to_model for t in ctx.tools)

    async def test_resolver_ignores_other_tools(self) -> None:
        mw = _mw([_make_group("g", ["t1"])])
        ctx = AgentContext(system_prompt="", messages=[], tools=list(mw.tools))
        call = ToolCall(id="tc-2", name="load_tools", arguments={"group_id": "g"})
        assert await mw.resolve_tool_call(call, context=ctx) is None

    async def test_resolver_unknown_tool_returns_none(self) -> None:
        mw = _mw([_make_group("g", ["t1"])])
        ctx = AgentContext(system_prompt="", messages=[], tools=list(mw.tools))
        call = ToolCall(
            id="tc-3",
            name="deferred_tool_call",
            arguments={"tool_name": "nope", "arguments": {}},
        )
        assert await mw.resolve_tool_call(call, context=ctx) is None

    async def test_dispatcher_execute_is_error_fallback(self) -> None:
        mw = _mw([_make_group("g", ["t1"])])
        dispatcher = next(t for t in mw.tools if t.name == "deferred_tool_call")
        args = dispatcher.parameters.model_validate(
            {"tool_name": "nope", "arguments": {}}
        )
        result = await dispatcher.execute("tc-4", args)
        assert result.is_error is True
        assert "t1" in result.content[0].text  # lists valid names
        assert "load_tools" in result.content[0].text  # recovery hint

    async def test_direct_native_call_to_hidden_tool_executes(self) -> None:
        """If the model hallucinates a direct tool_use with the real name,
        the engine resolves it from context.tools despite expose_to_model=False."""
        from cubepi.agent.tools import execute_tool_calls
        from cubepi.providers.faux import faux_assistant_message

        mw = _mw([_make_group("g", ["t1"])])
        ctx = AgentContext(system_prompt="", messages=[], tools=list(mw.tools))
        await mw._expand(group_id="g", tool_names=None, context=ctx)
        call = ToolCall(id="tc-direct", name="t1", arguments={"value": "hi"})
        batch = await execute_tool_calls(
            ctx,
            faux_assistant_message(call, stop_reason="tool_use"),
            emit=lambda e: None,
        )
        assert batch.messages[0].content[0].text == "echo:hi"

    async def test_inject_strategy_resolver_is_noop(self) -> None:
        mw = _mw([_make_group("g", ["t1"])], strategy="inject")
        ctx = AgentContext(system_prompt="", messages=[], tools=list(mw.tools))
        call = ToolCall(
            id="tc-5",
            name="deferred_tool_call",
            arguments={"tool_name": "t1", "arguments": {"value": "x"}},
        )
        assert await mw.resolve_tool_call(call, context=ctx) is None


class TestDispatchResume:
    async def test_resume_restores_loader_cache_and_hidden_tools(self) -> None:
        group = _make_group("g", ["t1", "t2"])
        state = await DeferredToolsMiddleware.prepare_resumed_state(
            [group], {"g": ["t1"]}, strategy="dispatch"
        )
        assert [t.name for t in state.pre_loaded_tools] == ["t1"]
        assert all(not t.expose_to_model for t in state.pre_loaded_tools)
        assert "g" in state.loader_cache
        # Partially expanded groups remain deferrable.
        assert [g.group_id for g in state.remaining_groups] == ["g"]

        extra: dict = {"expanded_groups": {"g": ["t1"]}}
        mw = DeferredToolsMiddleware(
            groups=state.remaining_groups,
            extra_ref=lambda: extra,
            strategy="dispatch",
            resumed_loader_cache=state.loader_cache,
        )
        ctx = AgentContext(
            system_prompt="",
            messages=[],
            tools=[*mw.tools, *state.pre_loaded_tools],
        )
        call = ToolCall(
            id="tc-r1",
            name="deferred_tool_call",
            arguments={"tool_name": "t1", "arguments": {"value": "hi"}},
        )
        rewritten = await mw.resolve_tool_call(call, context=ctx)
        assert rewritten is not None and rewritten.name == "t1"

    async def test_resume_dispatch_keeps_fully_expanded_group_loadable(self) -> None:
        """Fully expanded groups stay in remaining so load_tools can re-serve
        schemas after compaction."""
        group = _make_group("g", ["t1"])
        state = await DeferredToolsMiddleware.prepare_resumed_state(
            [group], {"g": None}, strategy="dispatch"
        )
        assert [g.group_id for g in state.remaining_groups] == ["g"]


class _CapturingFaux:
    """FauxProvider subclass recording every request's (system_prompt, tools)."""

    def __new__(cls, **kwargs):
        from cubepi.providers.faux import FauxProvider

        class _Capturing(FauxProvider):
            def __init__(self, **kw):
                super().__init__(**kw)
                self.captured: list[tuple[str, list[dict] | None]] = []

            async def stream(
                self,
                model,
                messages,
                *,
                system_prompt="",
                tools=None,
                tool_choice=None,
                options=None,
            ):
                self.captured.append(
                    (
                        system_prompt,
                        [t.model_dump() for t in tools] if tools else None,
                    )
                )
                return await super().stream(
                    model,
                    messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    tool_choice=tool_choice,
                    options=options,
                )

        return _Capturing(**kwargs)


class TestByteStability:
    async def test_prefix_static_across_load_and_dispatch(self) -> None:
        from cubepi.agent.agent import Agent
        from cubepi.providers.faux import faux_assistant_message, faux_tool_call

        provider = _CapturingFaux()
        provider.set_responses(
            [
                faux_assistant_message(
                    faux_tool_call("load_tools", {"group_id": "g"}, id="tc-1"),
                    stop_reason="tool_use",
                ),
                faux_assistant_message(
                    faux_tool_call(
                        "deferred_tool_call",
                        {"tool_name": "t1", "arguments": {"value": "hi"}},
                        id="tc-2",
                    ),
                    stop_reason="tool_use",
                ),
                faux_assistant_message("done"),
            ]
        )
        agent = Agent(
            model=provider.model("faux"),
            system_prompt="base prompt",
            deferred_tool_groups=[_make_group("g", ["t1", "t2"])],
        )
        await agent.prompt("go")

        assert len(provider.captured) == 3
        systems = {c[0] for c in provider.captured}
        assert len(systems) == 1  # system prompt byte-identical every turn

        tool_payloads = {
            json.dumps(c[1], sort_keys=True) for c in provider.captured
        }
        assert len(tool_payloads) == 1  # tools param byte-identical every turn

        # And the dispatched tool actually ran, keyed to the dispatcher's id
        # but carrying the real tool's output.
        result_texts = [
            m.content[0].text
            for m in agent._state.messages
            if getattr(m, "tool_call_id", None) == "tc-2"
        ]
        assert result_texts == ["echo:hi"]


class TestDispatchFork:
    async def test_fork_dispatches_loaded_tool(self) -> None:
        """Forks forward the resolver, so deferred_tool_call works on tools
        the parent already loaded (regression guard: v1 forks could call
        expanded tools natively)."""
        from cubepi.agent.agent import Agent
        from cubepi.checkpointer.memory import MemoryCheckpointer
        from cubepi.providers.faux import (
            FauxProvider,
            faux_assistant_message,
            faux_tool_call,
        )

        cp = MemoryCheckpointer()
        provider = FauxProvider()
        provider.set_responses(
            [
                # parent run R1: load the group, then finish
                faux_assistant_message(
                    faux_tool_call("load_tools", {"group_id": "g"}, id="tc-1"),
                    stop_reason="tool_use",
                ),
                faux_assistant_message("loaded"),
                # fork probe: dispatch the loaded tool, then finish
                faux_assistant_message(
                    faux_tool_call(
                        "deferred_tool_call",
                        {"tool_name": "t1", "arguments": {"value": "fork"}},
                        id="tc-2",
                    ),
                    stop_reason="tool_use",
                ),
                faux_assistant_message("fork-done"),
            ]
        )
        agent = Agent(
            model=provider.model("faux"),
            deferred_tool_groups=[_make_group("g", ["t1"])],
            checkpointer=cp,
            thread_id="src",
        )
        await agent.prompt("load it", run_id="R1")

        result = await agent.fork_once("src", "use t1", after_run_id="R1")
        assert result.text == "fork-done"
        # The dispatched tool really executed inside the fork.
        echo_results = [
            m.content[0].text
            for m in result.messages
            if getattr(m, "tool_call_id", None) == "tc-2"
        ]
        assert echo_results == ["echo:fork"]
