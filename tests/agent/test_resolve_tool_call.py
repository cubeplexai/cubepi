from __future__ import annotations

from pydantic import BaseModel, Field

from cubepi.agent.tools import execute_tool_calls
from cubepi.agent.types import (
    AgentContext,
    AgentTool,
    AgentToolResult,
    ToolExecutionStartEvent,
)
from cubepi.providers.base import (
    AssistantMessage,
    TextContent,
    ToolCall,
)
from cubepi.providers.faux import faux_assistant_message


class _EchoArgs(BaseModel):
    value: str = Field(description="echoed back")


def _echo_tool(name: str, *, expose: bool = True) -> AgentTool:
    async def _exec(tool_call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=f"echo:{args.value}")])

    return AgentTool(
        name=name,
        description="echo",
        parameters=_EchoArgs,
        execute=_exec,
        expose_to_model=expose,
    )


def _assistant_with(call: ToolCall) -> AssistantMessage:
    # faux_assistant_message fills usage/timestamp with valid defaults.
    return faux_assistant_message(call, stop_reason="tool_use")


def _noop_emit(event) -> None:
    return None


async def test_resolver_rewrites_call_before_pipeline() -> None:
    real = _echo_tool("real_tool", expose=False)
    ctx = AgentContext(system_prompt="", messages=[], tools=[real])
    call = ToolCall(
        id="tc-1",
        name="deferred_tool_call",
        arguments={"tool_name": "real_tool", "arguments": {"value": "hi"}},
    )
    seen_before: list[str] = []

    async def resolver(tool_call, *, context, signal=None):
        if tool_call.name != "deferred_tool_call":
            return None
        return ToolCall(
            id=tool_call.id,
            name=tool_call.arguments["tool_name"],
            arguments=tool_call.arguments["arguments"],
        )

    async def before(hook_ctx, *, signal=None):
        seen_before.append(hook_ctx.tool_call.name)
        return None

    events: list = []
    batch = await execute_tool_calls(
        ctx,
        _assistant_with(call),
        before_tool_call=before,
        resolve_tool_call=resolver,
        emit=events.append,
    )
    # Hook saw the real name, not the dispatcher envelope.
    assert seen_before == ["real_tool"]
    # Emitted execution events carry the resolved name (tracing looks tools
    # up by event.tool_name).
    start_names = [
        e.tool_name for e in events if isinstance(e, ToolExecutionStartEvent)
    ]
    assert start_names == ["real_tool"]
    # Result keyed to the ORIGINAL tool_use id, carrying the real name.
    msg = batch.messages[0]
    assert msg.tool_call_id == "tc-1"
    assert msg.tool_name == "real_tool"
    assert msg.content[0].text == "echo:hi"


async def test_resolver_none_is_passthrough() -> None:
    tool = _echo_tool("plain")
    ctx = AgentContext(system_prompt="", messages=[], tools=[tool])
    call = ToolCall(id="tc-2", name="plain", arguments={"value": "x"})

    async def resolver(tool_call, *, context, signal=None):
        return None

    batch = await execute_tool_calls(
        ctx, _assistant_with(call), resolve_tool_call=resolver, emit=_noop_emit
    )
    assert batch.messages[0].content[0].text == "echo:x"


async def test_resolver_exception_becomes_error_result() -> None:
    tool = _echo_tool("plain")
    ctx = AgentContext(system_prompt="", messages=[], tools=[tool])
    call = ToolCall(id="tc-3", name="plain", arguments={"value": "x"})

    async def resolver(tool_call, *, context, signal=None):
        raise RuntimeError("resolver blew up")

    batch = await execute_tool_calls(
        ctx, _assistant_with(call), resolve_tool_call=resolver, emit=_noop_emit
    )
    msg = batch.messages[0]
    assert msg.is_error is True
    assert "resolver blew up" in msg.content[0].text


async def test_resolved_call_validation_error_includes_schema() -> None:
    real = _echo_tool("real_tool", expose=False)
    ctx = AgentContext(system_prompt="", messages=[], tools=[real])
    call = ToolCall(
        id="tc-4",
        name="deferred_tool_call",
        arguments={"tool_name": "real_tool", "arguments": {"wrong_field": 1}},
    )

    async def resolver(tool_call, *, context, signal=None):
        return ToolCall(
            id=tool_call.id,
            name="real_tool",
            arguments=tool_call.arguments["arguments"],
        )

    batch = await execute_tool_calls(
        ctx, _assistant_with(call), resolve_tool_call=resolver, emit=_noop_emit
    )
    msg = batch.messages[0]
    assert msg.is_error is True
    text = msg.content[0].text
    assert "Invalid arguments for tool 'real_tool'" in text
    # Full schema appended so the model can self-correct in one round trip.
    assert '"value"' in text


async def test_unresolved_call_validation_error_has_no_schema() -> None:
    tool = _echo_tool("plain")
    ctx = AgentContext(system_prompt="", messages=[], tools=[tool])
    call = ToolCall(id="tc-5", name="plain", arguments={"wrong": 1})

    batch = await execute_tool_calls(ctx, _assistant_with(call), emit=_noop_emit)
    text = batch.messages[0].content[0].text
    assert "Invalid arguments" in text
    assert "Full schema" not in text


async def test_rewritten_call_keeps_original_id() -> None:
    """The engine enforces the resolver contract: a rewritten call that
    changes the id is corrected back to the original, so provider-side
    tool_result correlation never desynchronizes."""
    real = _echo_tool("real_tool", expose=False)
    ctx = AgentContext(system_prompt="", messages=[], tools=[real])
    call = ToolCall(
        id="tc-orig",
        name="deferred_tool_call",
        arguments={"tool_name": "real_tool", "arguments": {"value": "hi"}},
    )

    async def sloppy_resolver(tool_call, *, context, signal=None):
        return ToolCall(  # buggy: invents a new id
            id="tc-INVENTED",
            name="real_tool",
            arguments=tool_call.arguments["arguments"],
        )

    batch = await execute_tool_calls(
        ctx, _assistant_with(call), resolve_tool_call=sloppy_resolver, emit=_noop_emit
    )
    msg = batch.messages[0]
    assert msg.tool_call_id == "tc-orig"
    assert msg.content[0].text == "echo:hi"
