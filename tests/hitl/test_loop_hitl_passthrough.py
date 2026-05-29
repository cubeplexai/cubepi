import pytest
from cubepi.hitl.exceptions import HitlAborted, HitlCancelled, HitlDetached
from cubepi.agent.tools import execute_tool_calls
from cubepi.agent.types import AgentContext, AgentTool, AgentToolResult
from cubepi.providers.base import AssistantMessage, TextContent, ToolCall
from pydantic import BaseModel


class _NoParams(BaseModel):
    pass


def _make_tool(name: str, executor, execution_mode: str = "sequential"):
    return AgentTool(
        name=name,
        description="t",
        parameters=_NoParams,
        execute=executor,
        execution_mode=execution_mode,
    )


async def test_hitl_control_exception_in_tool_propagates():
    async def raises(call_id, args, *, signal=None, on_update=None):
        raise HitlAborted()

    tool = _make_tool("t1", raises)
    ctx = AgentContext(system_prompt="", messages=[], tools=[tool])
    msg = AssistantMessage(
        content=[TextContent(text=""), ToolCall(id="tc-1", name="t1", arguments={})],
        stop_reason="tool_use",
    )
    with pytest.raises(HitlAborted):
        await execute_tool_calls(ctx, msg, emit=lambda e: None)


async def test_regular_exception_in_tool_becomes_tool_error():
    async def raises(call_id, args, *, signal=None, on_update=None):
        raise ValueError("oops")

    tool = _make_tool("t1", raises)
    ctx = AgentContext(system_prompt="", messages=[], tools=[tool])
    msg = AssistantMessage(
        content=[TextContent(text=""), ToolCall(id="tc-1", name="t1", arguments={})],
        stop_reason="tool_use",
    )
    batch = await execute_tool_calls(ctx, msg, emit=lambda e: None)
    assert batch.messages[0].is_error is True
    assert "oops" in batch.messages[0].content[0].text


async def test_hitl_control_in_before_tool_call_propagates():
    async def runs(call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="ok")])

    tool = _make_tool("t1", runs)
    ctx = AgentContext(system_prompt="", messages=[], tools=[tool])
    msg = AssistantMessage(
        content=[TextContent(text=""), ToolCall(id="tc-1", name="t1", arguments={})],
        stop_reason="tool_use",
    )

    async def before(_ctx, *, signal=None):
        raise HitlCancelled("user cancelled")

    with pytest.raises(HitlCancelled):
        await execute_tool_calls(ctx, msg, before_tool_call=before, emit=lambda e: None)


async def test_parallel_detach_does_not_start_earlier_tools():
    """Regression: when a later before_tool_call raises HitlDetached, earlier
    parallel tools must NOT have been started — otherwise their side effects
    happen but their ToolResultMessage is never emitted/checkpointed, and
    Agent.respond() re-runs them, duplicating the side effects.

    Codex PR #127 review feedback (P1, agent/tools.py).
    """
    t1_calls = 0
    t2_calls = 0

    async def t1_execute(call_id, args, *, signal=None, on_update=None):
        nonlocal t1_calls
        t1_calls += 1
        return AgentToolResult(content=[TextContent(text="t1 ran")])

    async def t2_execute(call_id, args, *, signal=None, on_update=None):
        nonlocal t2_calls
        t2_calls += 1
        return AgentToolResult(content=[TextContent(text="t2 ran")])

    t1 = _make_tool("t1", t1_execute, execution_mode="parallel")
    t2 = _make_tool("t2", t2_execute, execution_mode="parallel")
    ctx = AgentContext(system_prompt="", messages=[], tools=[t1, t2])
    msg = AssistantMessage(
        content=[
            ToolCall(id="tc-1", name="t1", arguments={}),
            ToolCall(id="tc-2", name="t2", arguments={}),
        ],
        stop_reason="tool_use",
    )

    async def before(before_ctx, *, signal=None):
        if before_ctx.tool_call.id == "tc-2":
            raise HitlDetached()
        return None

    with pytest.raises(HitlDetached):
        await execute_tool_calls(
            ctx,
            msg,
            tool_execution="parallel",
            before_tool_call=before,
            emit=lambda e: None,
        )

    # Yield once so any leaked background task has a chance to run.
    import asyncio

    await asyncio.sleep(0)

    assert t1_calls == 0, (
        "t1 was started even though t2's prepare detached — its result will "
        "never be emitted/checkpointed, so resume will duplicate the side effect"
    )
    assert t2_calls == 0


async def test_parallel_detach_does_not_leak_start_events():
    """Regression: when t2's prepare raises HitlDetached during a parallel
    batch, no `ToolExecutionStartEvent` for t1 or t2 may have been emitted —
    otherwise `state.pending_tool_calls` and the corresponding trace span
    are stuck open with no matching ToolExecutionEndEvent.

    Codex PR #127 review feedback (P2 agent/tools.py).
    """
    from cubepi.agent.types import ToolExecutionStartEvent

    async def t1_execute(call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="ok")])

    async def t2_execute(call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="ok")])

    t1 = _make_tool("t1", t1_execute, execution_mode="parallel")
    t2 = _make_tool("t2", t2_execute, execution_mode="parallel")
    ctx = AgentContext(system_prompt="", messages=[], tools=[t1, t2])
    msg = AssistantMessage(
        content=[
            ToolCall(id="tc-1", name="t1", arguments={}),
            ToolCall(id="tc-2", name="t2", arguments={}),
        ],
        stop_reason="tool_use",
    )

    async def before(before_ctx, *, signal=None):
        if before_ctx.tool_call.id == "tc-2":
            raise HitlDetached()
        return None

    events: list = []
    with pytest.raises(HitlDetached):
        await execute_tool_calls(
            ctx,
            msg,
            tool_execution="parallel",
            before_tool_call=before,
            emit=lambda e: events.append(e),
        )

    start_events = [e for e in events if isinstance(e, ToolExecutionStartEvent)]
    assert start_events == [], (
        f"leaked {len(start_events)} ToolExecutionStartEvent(s) with no End — "
        f"state.pending_tool_calls / trace spans would be stuck open: "
        f"{[(e.tool_call_id, e.tool_name) for e in start_events]}"
    )
