import pytest
from cubepi.hitl.exceptions import HitlAborted, HitlCancelled
from cubepi.agent.tools import execute_tool_calls
from cubepi.agent.types import AgentContext, AgentTool, AgentToolResult
from cubepi.providers.base import AssistantMessage, TextContent, ToolCall
from pydantic import BaseModel


class _NoParams(BaseModel):
    pass


def _make_tool(name: str, executor):
    return AgentTool(
        name=name,
        description="t",
        parameters=_NoParams,
        execute=executor,
        execution_mode="sequential",
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
