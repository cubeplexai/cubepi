from __future__ import annotations

import asyncio
from typing import Any, Callable

from cubepi.agent.tools import execute_tool_calls
from cubepi.agent.types import (
    AgentContext,
    AgentEndEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    ShouldStopAfterTurnContext,
    TurnEndEvent,
    TurnStartEvent,
)
from cubepi.providers.base import (
    AssistantMessage,
    Model,
    Provider,
    StreamOptions,
    ToolCall,
    ToolResultMessage,
)


async def _emit(emit_fn: Callable, event: Any) -> None:
    result = emit_fn(event)
    if asyncio.iscoroutine(result):
        await result


async def run_agent_loop(
    *,
    prompts: list[Any],
    context: AgentContext,
    provider: Provider,
    model: Model,
    convert_to_llm: Callable,
    emit: Callable,
    transform_context: Callable | None = None,
    before_tool_call: Callable | None = None,
    after_tool_call: Callable | None = None,
    should_stop_after_turn: Callable | None = None,
    get_steering_messages: Callable | None = None,
    get_follow_up_messages: Callable | None = None,
    stream_options: StreamOptions | None = None,
    tool_execution: str = "parallel",
    system_prompt: str | None = None,
) -> list[Any]:
    new_messages: list[Any] = list(prompts)
    current_context = AgentContext(
        system_prompt=system_prompt
        if system_prompt is not None
        else context.system_prompt,
        messages=list(context.messages) + list(prompts),
        tools=context.tools,
    )

    await _emit(emit, AgentStartEvent())
    await _emit(emit, TurnStartEvent())
    for prompt in prompts:
        await _emit(emit, MessageStartEvent(message=prompt))
        await _emit(emit, MessageEndEvent(message=prompt))

    await _run_loop(
        current_context=current_context,
        new_messages=new_messages,
        provider=provider,
        model=model,
        convert_to_llm=convert_to_llm,
        transform_context=transform_context,
        before_tool_call=before_tool_call,
        after_tool_call=after_tool_call,
        should_stop_after_turn=should_stop_after_turn,
        get_steering_messages=get_steering_messages,
        get_follow_up_messages=get_follow_up_messages,
        stream_options=stream_options,
        tool_execution=tool_execution,
        emit=emit,
    )
    return new_messages


async def run_agent_loop_continue(
    *,
    context: AgentContext,
    provider: Provider,
    model: Model,
    convert_to_llm: Callable,
    emit: Callable,
    transform_context: Callable | None = None,
    before_tool_call: Callable | None = None,
    after_tool_call: Callable | None = None,
    should_stop_after_turn: Callable | None = None,
    get_steering_messages: Callable | None = None,
    get_follow_up_messages: Callable | None = None,
    stream_options: StreamOptions | None = None,
    tool_execution: str = "parallel",
    system_prompt: str | None = None,
) -> list[Any]:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")

    new_messages: list[Any] = []
    current_context = AgentContext(
        system_prompt=system_prompt
        if system_prompt is not None
        else context.system_prompt,
        messages=list(context.messages),
        tools=context.tools,
    )

    await _emit(emit, AgentStartEvent())
    await _emit(emit, TurnStartEvent())

    await _run_loop(
        current_context=current_context,
        new_messages=new_messages,
        provider=provider,
        model=model,
        convert_to_llm=convert_to_llm,
        transform_context=transform_context,
        before_tool_call=before_tool_call,
        after_tool_call=after_tool_call,
        should_stop_after_turn=should_stop_after_turn,
        get_steering_messages=get_steering_messages,
        get_follow_up_messages=get_follow_up_messages,
        stream_options=stream_options,
        tool_execution=tool_execution,
        emit=emit,
    )
    return new_messages


async def _run_loop(
    *,
    current_context: AgentContext,
    new_messages: list[Any],
    provider: Provider,
    model: Model,
    convert_to_llm: Callable,
    transform_context: Callable | None,
    before_tool_call: Callable | None,
    after_tool_call: Callable | None,
    should_stop_after_turn: Callable | None,
    get_steering_messages: Callable | None,
    get_follow_up_messages: Callable | None,
    stream_options: StreamOptions | None,
    tool_execution: str,
    emit: Callable,
) -> None:
    opts = stream_options or StreamOptions()
    first_turn = True

    # Poll for steering messages at start (user may have typed while waiting)
    if get_steering_messages:
        pending = await get_steering_messages() or []
        if pending:
            for msg in pending:
                await _emit(emit, MessageStartEvent(message=msg))
                await _emit(emit, MessageEndEvent(message=msg))
                current_context.messages.append(msg)
                new_messages.append(msg)

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls:
            if not first_turn:
                await _emit(emit, TurnStartEvent())
            else:
                first_turn = False

            message = await _stream_assistant_response(
                current_context,
                provider,
                model,
                convert_to_llm,
                transform_context,
                opts,
                emit,
            )
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                await _emit(emit, TurnEndEvent(message=message, tool_results=[]))
                await _emit(emit, AgentEndEvent(messages=new_messages))
                return

            tool_calls = [c for c in message.content if isinstance(c, ToolCall)]
            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False

            if tool_calls:
                batch = await execute_tool_calls(
                    current_context,
                    message,
                    tool_execution=tool_execution,
                    before_tool_call=before_tool_call,
                    after_tool_call=after_tool_call,
                    signal=opts.signal,
                    emit=emit,
                )
                tool_results = batch.messages
                has_more_tool_calls = not batch.terminate

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            await _emit(emit, TurnEndEvent(message=message, tool_results=tool_results))

            if should_stop_after_turn:
                stop_ctx = ShouldStopAfterTurnContext(
                    message=message,
                    tool_results=tool_results,
                    context=current_context,
                    new_messages=new_messages,
                )
                if await should_stop_after_turn(stop_ctx):
                    await _emit(emit, AgentEndEvent(messages=new_messages))
                    return

            # Check for steering messages after tool execution
            if get_steering_messages and has_more_tool_calls:
                steering = await get_steering_messages() or []
                if steering:
                    for msg in steering:
                        await _emit(emit, MessageStartEvent(message=msg))
                        await _emit(emit, MessageEndEvent(message=msg))
                        current_context.messages.append(msg)
                        new_messages.append(msg)

        # After inner loop completes, check for follow-up messages
        if get_follow_up_messages:
            follow_ups = await get_follow_up_messages() or []
            if follow_ups:
                for msg in follow_ups:
                    await _emit(emit, MessageStartEvent(message=msg))
                    await _emit(emit, MessageEndEvent(message=msg))
                    current_context.messages.append(msg)
                    new_messages.append(msg)
                first_turn = False
                continue

        break

    await _emit(emit, AgentEndEvent(messages=new_messages))


async def _stream_assistant_response(
    context: AgentContext,
    provider: Provider,
    model: Model,
    convert_to_llm: Callable,
    transform_context: Callable | None,
    options: StreamOptions,
    emit: Callable,
) -> AssistantMessage:
    messages = context.messages
    if transform_context:
        messages = await transform_context(messages, signal=options.signal)

    llm_messages = convert_to_llm(messages)
    if asyncio.iscoroutine(llm_messages):
        llm_messages = await llm_messages

    tools_defs = None
    if context.tools:
        tools_defs = [t.to_definition() for t in context.tools]

    stream = await provider.stream(
        model,
        llm_messages,
        system_prompt=context.system_prompt,
        tools=tools_defs,
        options=options,
    )

    partial_message: AssistantMessage | None = None
    added_partial = False

    async for event in stream:
        if event.type == "start":
            partial_message = event.partial
            if partial_message:
                context.messages.append(partial_message)
                added_partial = True
                await _emit(
                    emit,
                    MessageStartEvent(message=partial_message.model_copy(deep=True)),
                )

        elif event.type in (
            "text_start",
            "text_delta",
            "text_end",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
        ):
            if partial_message and event.partial:
                partial_message = event.partial
                context.messages[-1] = partial_message
                await _emit(
                    emit,
                    MessageUpdateEvent(
                        message=partial_message.model_copy(deep=True),
                        stream_event=event,
                    ),
                )

        elif event.type in ("done", "error"):
            final_message = await stream.result()
            if added_partial:
                context.messages[-1] = final_message
            else:
                context.messages.append(final_message)
            if not added_partial:
                await _emit(emit, MessageStartEvent(message=final_message))
            await _emit(emit, MessageEndEvent(message=final_message))
            return final_message

    # Fallback: stream ended without done/error event
    final_message = await stream.result()
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await _emit(emit, MessageStartEvent(message=final_message))
    await _emit(emit, MessageEndEvent(message=final_message))
    return final_message
