"""Pin the new `ToolExecutionEndEvent` fields: ``terminate``,
``blocked_by_hook``, ``block_reason``.

Phase 0b of the cubepi tracing plan — these fields let observers
(tracing, dashboards) recognize tool-driven turn termination and
hook-driven blocks without unwrapping ``result``.
"""

from __future__ import annotations

from pydantic import BaseModel

from cubepi.agent.tools import execute_tool_calls
from cubepi.agent.types import (
    AgentContext,
    AgentTool,
    AgentToolResult,
    BeforeToolCallResult,
)
from cubepi.providers.base import AssistantMessage, TextContent, ToolCall


class Params(BaseModel):
    value: str = ""


def _tool(name: str, fn) -> AgentTool:
    return AgentTool(
        name=name,
        description="test tool",
        parameters=Params,
        execute=fn,
    )


def _ctx(tools: list[AgentTool]) -> AgentContext:
    return AgentContext(system_prompt="", messages=[], tools=tools)


def _msg(calls: list[ToolCall]) -> AssistantMessage:
    return AssistantMessage(content=list(calls))


def _tool_end_event(events: list):
    for e in events:
        if getattr(e, "type", None) == "tool_execution_end":
            return e
    raise AssertionError("no tool_execution_end event captured")


class TestTerminate:
    async def test_terminate_true_propagates(self):
        async def run(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="ok")], terminate=True)

        events: list = []
        await execute_tool_calls(
            _ctx([_tool("term", run)]),
            _msg([ToolCall(id="t1", name="term", arguments={})]),
            tool_execution="sequential",
            emit=lambda e: events.append(e),
        )

        ev = _tool_end_event(events)
        assert ev.terminate is True
        assert ev.is_error is False
        assert ev.blocked_by_hook is False
        assert ev.block_reason is None

    async def test_terminate_default_false(self):
        async def run(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="ok")])

        events: list = []
        await execute_tool_calls(
            _ctx([_tool("plain", run)]),
            _msg([ToolCall(id="t1", name="plain", arguments={})]),
            tool_execution="sequential",
            emit=lambda e: events.append(e),
        )

        assert _tool_end_event(events).terminate is False


class TestBlockedByHook:
    async def test_before_hook_block_sets_fields(self):
        async def run(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="should not run")])

        async def before(ctx, *, signal=None):
            return BeforeToolCallResult(block=True, reason="not allowed by policy")

        events: list = []
        await execute_tool_calls(
            _ctx([_tool("guarded", run)]),
            _msg([ToolCall(id="t1", name="guarded", arguments={})]),
            tool_execution="sequential",
            before_tool_call=before,
            emit=lambda e: events.append(e),
        )

        ev = _tool_end_event(events)
        assert ev.is_error is True
        assert ev.blocked_by_hook is True
        assert ev.block_reason == "not allowed by policy"
        assert ev.terminate is False

    async def test_before_hook_block_with_no_reason(self):
        async def run(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[])

        async def before(ctx, *, signal=None):
            return BeforeToolCallResult(block=True)  # reason omitted

        events: list = []
        await execute_tool_calls(
            _ctx([_tool("guarded", run)]),
            _msg([ToolCall(id="t1", name="guarded", arguments={})]),
            tool_execution="sequential",
            before_tool_call=before,
            emit=lambda e: events.append(e),
        )

        ev = _tool_end_event(events)
        assert ev.blocked_by_hook is True
        assert ev.block_reason is None


class TestOtherImmediateErrorsNotBlocked:
    async def test_tool_not_found(self):
        # No tools registered → "Tool xxx not found" path; is_error=True
        # but NOT blocked_by_hook.
        events: list = []
        await execute_tool_calls(
            _ctx([]),
            _msg([ToolCall(id="t1", name="missing", arguments={})]),
            tool_execution="sequential",
            emit=lambda e: events.append(e),
        )

        ev = _tool_end_event(events)
        assert ev.is_error is True
        assert ev.blocked_by_hook is False
        assert ev.block_reason is None

    async def test_arg_validation_failure(self):
        async def run(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[])

        class StrictParams(BaseModel):
            value: int  # require int

        tool = AgentTool(
            name="strict",
            description="strict",
            parameters=StrictParams,
            execute=run,
        )

        events: list = []
        await execute_tool_calls(
            _ctx([tool]),
            _msg([ToolCall(id="t1", name="strict", arguments={"value": "abc"})]),
            tool_execution="sequential",
            emit=lambda e: events.append(e),
        )

        ev = _tool_end_event(events)
        assert ev.is_error is True
        assert ev.blocked_by_hook is False
        assert ev.block_reason is None


class TestParallelPath:
    async def test_parallel_path_populates_fields_on_block(self):
        async def run(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="ran")], terminate=False)

        async def before(ctx, *, signal=None):
            if ctx.tool_call.id == "blocked_one":
                return BeforeToolCallResult(block=True, reason="parallel-block")
            return None

        events: list = []
        await execute_tool_calls(
            _ctx([_tool("p", run)]),
            _msg(
                [
                    ToolCall(id="blocked_one", name="p", arguments={}),
                    ToolCall(id="passed_one", name="p", arguments={}),
                ]
            ),
            tool_execution="parallel",
            before_tool_call=before,
            emit=lambda e: events.append(e),
        )

        ends = [e for e in events if getattr(e, "type", None) == "tool_execution_end"]
        assert len(ends) == 2
        by_id = {e.tool_call_id: e for e in ends}
        assert by_id["blocked_one"].blocked_by_hook is True
        assert by_id["blocked_one"].block_reason == "parallel-block"
        assert by_id["blocked_one"].is_error is True
        assert by_id["passed_one"].blocked_by_hook is False
        assert by_id["passed_one"].block_reason is None
        assert by_id["passed_one"].is_error is False
