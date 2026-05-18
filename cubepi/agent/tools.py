from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import ValidationError

from cubepi.agent.types import (
    AfterToolCallContext,
    AgentContext,
    AgentTool,
    AgentToolResult,
    BeforeToolCallContext,
    MessageEndEvent,
    MessageStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
)
from cubepi.utils import emit_event
from cubepi.providers.base import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
)


@dataclass
class ToolCallBatch:
    messages: list[ToolResultMessage]
    terminate: bool


@dataclass
class _PreparedToolCall:
    tool_call: ToolCall
    tool: AgentTool
    args: Any


@dataclass
class _ImmediateOutcome:
    result: AgentToolResult
    is_error: bool
    blocked_by_hook: bool = False
    block_reason: str | None = None


@dataclass
class _FinalizedOutcome:
    tool_call: ToolCall
    result: AgentToolResult
    is_error: bool
    blocked_by_hook: bool = False
    block_reason: str | None = None


def _error_result(message: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=message)])


def _make_tool_result_message(finalized: _FinalizedOutcome) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=finalized.tool_call.id,
        tool_name=finalized.tool_call.name,
        content=finalized.result.content,
        details=finalized.result.details,
        is_error=finalized.is_error,
        timestamp=time.time(),
    )


async def _prepare_tool_call(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    before_tool_call: Callable | None,
    signal: asyncio.Event | None,
) -> _PreparedToolCall | _ImmediateOutcome:
    tool = None
    if context.tools:
        for t in context.tools:
            if t.name == tool_call.name:
                tool = t
                break

    if tool is None:
        return _ImmediateOutcome(
            result=_error_result(f"Tool {tool_call.name} not found"),
            is_error=True,
        )

    try:
        validated_args = tool.parameters.model_validate(tool_call.arguments)
    except (ValidationError, Exception) as exc:
        return _ImmediateOutcome(result=_error_result(str(exc)), is_error=True)

    if before_tool_call:
        try:
            before_ctx = BeforeToolCallContext(
                assistant_message=assistant_message,
                tool_call=tool_call,
                args=validated_args,
                context=context,
            )
            before_result = await before_tool_call(before_ctx, signal=signal)
            if before_result and before_result.block:
                return _ImmediateOutcome(
                    result=_error_result(
                        before_result.reason or "Tool execution was blocked"
                    ),
                    is_error=True,
                    blocked_by_hook=True,
                    block_reason=before_result.reason,
                )
        except Exception as exc:
            return _ImmediateOutcome(result=_error_result(str(exc)), is_error=True)

    return _PreparedToolCall(tool_call=tool_call, tool=tool, args=validated_args)


async def _execute_prepared(
    prepared: _PreparedToolCall,
    signal: asyncio.Event | None,
    emit_fn: Callable,
) -> tuple[AgentToolResult, bool]:
    try:
        result = await prepared.tool.execute(
            prepared.tool_call.id,
            prepared.args,
            signal=signal,
            on_update=lambda partial: emit_event(
                emit_fn,
                ToolExecutionUpdateEvent(
                    tool_call_id=prepared.tool_call.id,
                    tool_name=prepared.tool_call.name,
                    args=prepared.tool_call.arguments,
                    partial_result=partial,
                ),
            ),
        )
        return result, False
    except Exception as exc:
        return _error_result(str(exc)), True


async def _finalize(
    context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: _PreparedToolCall,
    result: AgentToolResult,
    is_error: bool,
    after_tool_call: Callable | None,
    signal: asyncio.Event | None,
) -> _FinalizedOutcome:
    if after_tool_call:
        try:
            after_ctx = AfterToolCallContext(
                assistant_message=assistant_message,
                tool_call=prepared.tool_call,
                args=prepared.args,
                result=result,
                is_error=is_error,
                context=context,
            )
            after_result = await after_tool_call(after_ctx, signal=signal)
            if after_result:
                result = AgentToolResult(
                    content=(
                        after_result.content
                        if after_result.content is not None
                        else result.content
                    ),
                    details=(
                        after_result.details
                        if after_result.details is not None
                        else result.details
                    ),
                    terminate=(
                        after_result.terminate
                        if after_result.terminate is not None
                        else result.terminate
                    ),
                )
                is_error = (
                    after_result.is_error
                    if after_result.is_error is not None
                    else is_error
                )
        except Exception as exc:
            result = _error_result(str(exc))
            is_error = True

    return _FinalizedOutcome(
        tool_call=prepared.tool_call, result=result, is_error=is_error
    )


def _should_terminate(finalized: list[_FinalizedOutcome]) -> bool:
    return len(finalized) > 0 and all(f.result.terminate is True for f in finalized)


async def execute_tool_calls(
    context: AgentContext,
    assistant_message: AssistantMessage,
    *,
    tool_execution: str = "parallel",
    before_tool_call: Callable | None = None,
    after_tool_call: Callable | None = None,
    signal: asyncio.Event | None = None,
    emit: Callable,
) -> ToolCallBatch:
    tool_calls = [c for c in assistant_message.content if isinstance(c, ToolCall)]

    has_sequential = any(
        t.execution_mode == "sequential"
        for tc in tool_calls
        if context.tools
        for t in context.tools
        if t.name == tc.name
    )

    if tool_execution == "sequential" or has_sequential:
        return await _execute_sequential(
            context,
            assistant_message,
            tool_calls,
            before_tool_call,
            after_tool_call,
            signal,
            emit,
        )
    return await _execute_parallel(
        context,
        assistant_message,
        tool_calls,
        before_tool_call,
        after_tool_call,
        signal,
        emit,
    )


async def _execute_sequential(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    before_tool_call: Callable | None,
    after_tool_call: Callable | None,
    signal: asyncio.Event | None,
    emit_fn: Callable,
) -> ToolCallBatch:
    finalized_list: list[_FinalizedOutcome] = []
    messages: list[ToolResultMessage] = []

    for tc in tool_calls:
        await emit_event(
            emit_fn,
            ToolExecutionStartEvent(
                tool_call_id=tc.id, tool_name=tc.name, args=tc.arguments
            ),
        )

        preparation = await _prepare_tool_call(
            context, assistant_message, tc, before_tool_call, signal
        )

        if isinstance(preparation, _ImmediateOutcome):
            finalized = _FinalizedOutcome(
                tool_call=tc,
                result=preparation.result,
                is_error=preparation.is_error,
                blocked_by_hook=preparation.blocked_by_hook,
                block_reason=preparation.block_reason,
            )
        else:
            result, is_error = await _execute_prepared(preparation, signal, emit_fn)
            finalized = await _finalize(
                context,
                assistant_message,
                preparation,
                result,
                is_error,
                after_tool_call,
                signal,
            )

        await emit_event(
            emit_fn,
            ToolExecutionEndEvent(
                tool_call_id=tc.id,
                tool_name=tc.name,
                result=finalized.result,
                is_error=finalized.is_error,
                terminate=bool(finalized.result.terminate),
                blocked_by_hook=finalized.blocked_by_hook,
                block_reason=finalized.block_reason,
            ),
        )
        tool_msg = _make_tool_result_message(finalized)
        await emit_event(emit_fn, MessageStartEvent(message=tool_msg))
        await emit_event(emit_fn, MessageEndEvent(message=tool_msg))
        finalized_list.append(finalized)
        messages.append(tool_msg)

    return ToolCallBatch(messages=messages, terminate=_should_terminate(finalized_list))


async def _execute_parallel(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    before_tool_call: Callable | None,
    after_tool_call: Callable | None,
    signal: asyncio.Event | None,
    emit_fn: Callable,
) -> ToolCallBatch:
    entries: list[_FinalizedOutcome | asyncio.Task] = []

    for tc in tool_calls:
        await emit_event(
            emit_fn,
            ToolExecutionStartEvent(
                tool_call_id=tc.id, tool_name=tc.name, args=tc.arguments
            ),
        )

        preparation = await _prepare_tool_call(
            context, assistant_message, tc, before_tool_call, signal
        )

        if isinstance(preparation, _ImmediateOutcome):
            finalized = _FinalizedOutcome(
                tool_call=tc,
                result=preparation.result,
                is_error=preparation.is_error,
                blocked_by_hook=preparation.blocked_by_hook,
                block_reason=preparation.block_reason,
            )
            await emit_event(
                emit_fn,
                ToolExecutionEndEvent(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    result=finalized.result,
                    is_error=finalized.is_error,
                    terminate=bool(finalized.result.terminate),
                    blocked_by_hook=finalized.blocked_by_hook,
                    block_reason=finalized.block_reason,
                ),
            )
            entries.append(finalized)
        else:

            async def _run(prep=preparation):
                result, is_error = await _execute_prepared(prep, signal, emit_fn)
                fin = await _finalize(
                    context,
                    assistant_message,
                    prep,
                    result,
                    is_error,
                    after_tool_call,
                    signal,
                )
                await emit_event(
                    emit_fn,
                    ToolExecutionEndEvent(
                        tool_call_id=prep.tool_call.id,
                        tool_name=prep.tool_call.name,
                        result=fin.result,
                        is_error=fin.is_error,
                        terminate=bool(fin.result.terminate),
                        blocked_by_hook=fin.blocked_by_hook,
                        block_reason=fin.block_reason,
                    ),
                )
                return fin

            entries.append(asyncio.create_task(_run()))

    finalized_list: list[_FinalizedOutcome] = []
    for entry in entries:
        if isinstance(entry, asyncio.Task):
            finalized_list.append(await entry)
        else:
            finalized_list.append(entry)

    messages: list[ToolResultMessage] = []
    for finalized in finalized_list:
        tool_msg = _make_tool_result_message(finalized)
        await emit_event(emit_fn, MessageStartEvent(message=tool_msg))
        await emit_event(emit_fn, MessageEndEvent(message=tool_msg))
        messages.append(tool_msg)

    return ToolCallBatch(messages=messages, terminate=_should_terminate(finalized_list))
