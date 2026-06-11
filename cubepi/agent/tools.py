from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel
from pydantic import ValidationError

from cubepi.hitl.exceptions import HitlControlException

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
from cubepi.types import JsonObject, StructuredObject, StructuredValue


@dataclass
class ToolCallBatch:
    messages: list[ToolResultMessage]
    terminate: bool


@dataclass
class _PreparedToolCall:
    tool_call: ToolCall
    tool: AgentTool
    args: BaseModel | JsonObject
    hitl_trace: StructuredObject | None = None


@dataclass
class _ImmediateOutcome:
    result: AgentToolResult
    is_error: bool
    blocked_by_hook: bool = False
    block_reason: str | None = None
    hitl_trace: StructuredObject | None = None


@dataclass
class _FinalizedOutcome:
    tool_call: ToolCall
    result: AgentToolResult
    is_error: bool
    blocked_by_hook: bool = False
    block_reason: str | None = None
    hitl_trace: StructuredObject | None = None


def _error_result(message: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=message)])


def _format_validation_error(exc: ValidationError, tool_name: str) -> str:
    """Render a pydantic ValidationError as text the LLM can act on.

    The default ``str(exc)`` leaks pydantic-internal model names and a
    documentation URL — the model can read it but cannot reliably extract
    the fix from it. This produces one short line per error, naming the
    field path, the constraint, and (for discriminated unions) the allowed
    discriminator values, so the model can correct the call without
    another round-trip.
    """
    lines = [f"Invalid arguments for tool '{tool_name}':"]
    for err in exc.errors():
        loc_parts = [str(p) for p in err.get("loc", ())]
        loc = ".".join(loc_parts) if loc_parts else "<root>"
        etype = err.get("type", "")
        msg = err.get("msg", "")
        ctx = err.get("ctx") or {}

        if etype == "union_tag_not_found":
            # pydantic stores discriminator wrapped in single quotes, e.g.
            # "'operation'" — strip for readability. expected_tags is not
            # populated for this error type; the model has the JSON Schema
            # to enumerate allowed values.
            disc = str(ctx.get("discriminator", "?")).strip("'\"")
            lines.append(f"- {loc}: missing required discriminator key '{disc}'")
        elif etype == "union_tag_invalid":
            disc = str(ctx.get("discriminator", "?")).strip("'\"")
            tag = ctx.get("tag", "")
            allowed = ctx.get("expected_tags") or ""
            lines.append(
                f"- {loc}: discriminator '{disc}'={tag!r} is not one of: {allowed}"
            )
        elif etype == "missing":
            lines.append(f"- {loc}: field required")
        elif etype == "literal_error":
            expected = ctx.get("expected", "")
            lines.append(f"- {loc}: must be one of {expected}")
        elif etype == "extra_forbidden":
            lines.append(f"- {loc}: unexpected field")
        else:
            lines.append(f"- {loc}: {msg}")
    return "\n".join(lines)


def _merge_hitl_details(
    base: StructuredValue, hitl: StructuredObject | None
) -> StructuredValue:
    if hitl is None:
        return base
    if base is None:
        return {"hitl": hitl}
    if isinstance(base, dict):  # pragma: no cover — E2E tested
        merged = dict(base)
        merged["hitl"] = hitl
        return merged
    return {"_non_dict_details": base, "hitl": hitl}


def _make_tool_result_message(finalized: _FinalizedOutcome) -> ToolResultMessage:
    details = _merge_hitl_details(finalized.result.details, finalized.hitl_trace)
    return ToolResultMessage(
        tool_call_id=finalized.tool_call.id,
        tool_name=finalized.tool_call.name,
        content=finalized.result.content,
        details=details,
        is_error=finalized.is_error,
        timestamp=time.time(),
    )


async def _resolve_tool_call(
    tool_call: ToolCall,
    context: AgentContext,
    resolve_tool_call: Callable | None,
    signal: asyncio.Event | None,
) -> tuple[ToolCall, bool, _ImmediateOutcome | None]:
    """Give the resolve hook a chance to rewrite the call before anything
    else sees it — events, validation, before/after hooks, and tracing all
    operate on the rewritten call.

    Returns ``(call, was_resolved, resolver_error)``; ``resolver_error`` is
    an immediate error outcome (with the original call for attribution)
    when the resolver raised."""
    if not resolve_tool_call:
        return tool_call, False, None
    try:
        rewritten = await resolve_tool_call(tool_call, context=context, signal=signal)
    except HitlControlException:
        raise
    except Exception as exc:
        return (
            tool_call,
            False,
            _ImmediateOutcome(result=_error_result(str(exc)), is_error=True),
        )
    if rewritten is None:
        return tool_call, False, None
    return rewritten, True, None


async def _prepare_tool_call(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    before_tool_call: Callable | None,
    signal: asyncio.Event | None,
    *,
    resolved: bool = False,
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
    except ValidationError as exc:
        message = _format_validation_error(exc, tool.name)
        if resolved:
            # The model never saw a resolved tool's schema in the tools
            # block — append it so a bad call self-corrects in one round
            # trip instead of a blind retry.
            schema = json.dumps(
                tool.parameters.model_json_schema(),
                sort_keys=True,
                ensure_ascii=False,
            )
            message = f"{message}\n\nFull schema for '{tool.name}':\n{schema}"
        return _ImmediateOutcome(
            result=_error_result(message),
            is_error=True,
        )
    except Exception as exc:  # pragma: no cover — defensive
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
        except HitlControlException:
            raise
        except Exception as exc:
            return _ImmediateOutcome(result=_error_result(str(exc)), is_error=True)

        if before_result and before_result.block:
            return _ImmediateOutcome(
                result=_error_result(
                    before_result.reason or "Tool execution was blocked"
                ),
                is_error=True,
                blocked_by_hook=True,
                block_reason=before_result.deny_reason or before_result.reason,
                hitl_trace=before_result.hitl_trace,
            )

        if before_result and before_result.edited_args is not None:
            try:
                validated_args = tool.parameters.model_validate(
                    before_result.edited_args
                )
            except ValidationError as exc:  # pragma: no cover — defensive
                return _ImmediateOutcome(
                    result=_error_result(_format_validation_error(exc, tool.name)),
                    is_error=True,
                )

        hitl_trace_carry = before_result.hitl_trace if before_result else None
    else:
        hitl_trace_carry = None

    return _PreparedToolCall(
        tool_call=tool_call,
        tool=tool,
        args=validated_args,
        hitl_trace=hitl_trace_carry,
    )


async def _execute_prepared(
    prepared: _PreparedToolCall,
    signal: asyncio.Event | None,
    emit_fn: Callable,
) -> tuple[AgentToolResult, bool]:
    from cubepi.hitl.channel import _in_custom_tool_var

    is_builtin = prepared.tool.hitl_builtin
    token = None if is_builtin else _in_custom_tool_var.set(True)
    try:
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
            # Honor an explicit is_error set by the tool body: the pipeline
            # tracks error state as a separate bool, so a tool returning
            # AgentToolResult(is_error=True) without raising must surface here
            # (otherwise the model sees a successful result).
            return result, bool(result.is_error)
        except HitlControlException:
            raise
        except Exception as exc:
            return _error_result(str(exc)), True
    finally:
        if token is not None:
            _in_custom_tool_var.reset(token)


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
        tool_call=prepared.tool_call,
        result=result,
        is_error=is_error,
        hitl_trace=prepared.hitl_trace,
    )


def _should_terminate(finalized: list[_FinalizedOutcome]) -> bool:
    return len(finalized) > 0 and all(f.result.terminate is True for f in finalized)


async def execute_tool_calls(
    context: AgentContext,
    assistant_message: AssistantMessage,
    *,
    tool_execution: str = "parallel",
    before_tool_call: Callable | None = None,
    resolve_tool_call: Callable | None = None,
    after_tool_call: Callable | None = None,
    signal: asyncio.Event | None = None,
    emit: Callable,
) -> ToolCallBatch:
    tool_calls = [c for c in assistant_message.content if isinstance(c, ToolCall)]

    # Resolve up front (in order) so the execution-mode decision below sees
    # the REAL tool names — a dispatcher call targeting a sequential tool
    # must route through the sequential executor. Resolution is idempotent,
    # so hoisting it ahead of execution is safe.
    resolutions: list[tuple[ToolCall, bool, _ImmediateOutcome | None]] = []
    for tc in tool_calls:
        resolutions.append(
            await _resolve_tool_call(tc, context, resolve_tool_call, signal)
        )

    has_sequential = any(
        t.execution_mode == "sequential"
        for (rtc, _, _) in resolutions
        if context.tools
        for t in context.tools
        if t.name == rtc.name
    )

    if tool_execution == "sequential" or has_sequential:
        return await _execute_sequential(
            context,
            assistant_message,
            resolutions,
            before_tool_call,
            after_tool_call,
            signal,
            emit,
        )
    return await _execute_parallel(
        context,
        assistant_message,
        resolutions,
        before_tool_call,
        after_tool_call,
        signal,
        emit,
    )


async def _execute_sequential(
    context: AgentContext,
    assistant_message: AssistantMessage,
    resolutions: list[tuple[ToolCall, bool, _ImmediateOutcome | None]],
    before_tool_call: Callable | None,
    after_tool_call: Callable | None,
    signal: asyncio.Event | None,
    emit_fn: Callable,
) -> ToolCallBatch:
    finalized_list: list[_FinalizedOutcome] = []
    messages: list[ToolResultMessage] = []

    for rtc, was_resolved, resolver_error in resolutions:
        await emit_event(
            emit_fn,
            ToolExecutionStartEvent(
                tool_call_id=rtc.id, tool_name=rtc.name, args=rtc.arguments
            ),
        )

        preparation = resolver_error or await _prepare_tool_call(
            context,
            assistant_message,
            rtc,
            before_tool_call,
            signal,
            resolved=was_resolved,
        )

        if isinstance(preparation, _ImmediateOutcome):
            finalized = _FinalizedOutcome(
                tool_call=rtc,
                result=preparation.result,
                is_error=preparation.is_error,
                blocked_by_hook=preparation.blocked_by_hook,
                block_reason=preparation.block_reason,
                hitl_trace=preparation.hitl_trace,
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
                tool_call_id=rtc.id,
                tool_name=rtc.name,
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
    resolutions: list[tuple[ToolCall, bool, _ImmediateOutcome | None]],
    before_tool_call: Callable | None,
    after_tool_call: Callable | None,
    signal: asyncio.Event | None,
    emit_fn: Callable,
) -> ToolCallBatch:
    # Two-phase: prepare ALL tool calls (which may raise HitlDetached or other
    # HitlControlException from before_tool_call) BEFORE scheduling any
    # asyncio.create_task or emitting any ToolExecutionStartEvent. Otherwise:
    # (a) a later prepare detach would leak already-started tool tasks
    #     (side effects happen but ToolResultMessage is never emitted/
    #     checkpointed, so the resumed run duplicates the side effects);
    # (b) Start events emitted in the prepare loop would have no matching
    #     End event when prepare later raised, leaving state.pending_tool_calls
    #     and trace spans permanently open.
    entries: list[_FinalizedOutcome | _PreparedToolCall] = []
    for rtc, was_resolved, resolver_error in resolutions:
        preparation = resolver_error or await _prepare_tool_call(
            context,
            assistant_message,
            rtc,
            before_tool_call,
            signal,
            resolved=was_resolved,
        )

        if isinstance(preparation, _ImmediateOutcome):
            # Immediate outcomes get a paired Start+End right here (the
            # "execution" was the prepare step itself — e.g. blocked by hook
            # or unknown tool). Pairing keeps the event stream balanced
            # even though no real tool body runs.
            await emit_event(
                emit_fn,
                ToolExecutionStartEvent(
                    tool_call_id=rtc.id, tool_name=rtc.name, args=rtc.arguments
                ),
            )
            finalized = _FinalizedOutcome(
                tool_call=rtc,
                result=preparation.result,
                is_error=preparation.is_error,
                blocked_by_hook=preparation.blocked_by_hook,
                block_reason=preparation.block_reason,
                hitl_trace=preparation.hitl_trace,
            )
            await emit_event(
                emit_fn,
                ToolExecutionEndEvent(
                    tool_call_id=rtc.id,
                    tool_name=rtc.name,
                    result=finalized.result,
                    is_error=finalized.is_error,
                    terminate=bool(finalized.result.terminate),
                    blocked_by_hook=finalized.blocked_by_hook,
                    block_reason=finalized.block_reason,
                ),
            )
            entries.append(finalized)
        else:
            entries.append(preparation)

    async def _run(prep: _PreparedToolCall) -> _FinalizedOutcome:
        # Start event lives inside _run so it is emitted only for tools
        # that actually get scheduled. If a later prepare raises before
        # we get here, no Start is leaked.
        await emit_event(
            emit_fn,
            ToolExecutionStartEvent(
                tool_call_id=prep.tool_call.id,
                tool_name=prep.tool_call.name,
                args=prep.tool_call.arguments,
            ),
        )
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

    # Now that every prepare has succeeded, fan out the executions.
    scheduled: list[_FinalizedOutcome | asyncio.Task] = [
        entry
        if isinstance(entry, _FinalizedOutcome)
        else asyncio.create_task(_run(entry))
        for entry in entries
    ]
    finalized_list: list[_FinalizedOutcome] = []
    for entry in scheduled:
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
