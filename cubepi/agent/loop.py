from __future__ import annotations

import asyncio
from typing import Callable

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
from cubepi.hitl.exceptions import HitlAborted, HitlDetached
from cubepi.utils import emit_event
from cubepi.providers.base import (
    AssistantMessage,
    Message,
    Model,
    Provider,
    StreamOptions,
    ToolCall,
    ToolResultMessage,
)


async def run_agent_loop(
    *,
    prompts: list[Message],
    context: AgentContext,
    provider: Provider,
    model: Model,
    convert_to_llm: Callable,
    emit: Callable,
    transform_context: Callable | None = None,
    transform_system_prompt: Callable | None = None,
    after_model_response: Callable | None = None,
    before_tool_call: Callable | None = None,
    after_tool_call: Callable | None = None,
    should_stop_after_turn: Callable | None = None,
    get_steering_messages: Callable | None = None,
    get_follow_up_messages: Callable | None = None,
    stream_options: StreamOptions | None = None,
    tool_execution: str = "parallel",
    system_prompt: str | None = None,
) -> list[Message]:
    new_messages: list[Message] = list(prompts)
    current_context = AgentContext(
        system_prompt=system_prompt
        if system_prompt is not None
        else context.system_prompt,
        messages=list(context.messages) + list(prompts),
        tools=context.tools,
        extra=context.extra,
    )

    await emit_event(emit, AgentStartEvent())
    await emit_event(emit, TurnStartEvent())
    for prompt in prompts:
        await emit_event(emit, MessageStartEvent(message=prompt))
        await emit_event(emit, MessageEndEvent(message=prompt))

    await _run_loop(
        current_context=current_context,
        new_messages=new_messages,
        provider=provider,
        model=model,
        convert_to_llm=convert_to_llm,
        transform_context=transform_context,
        transform_system_prompt=transform_system_prompt,
        after_model_response=after_model_response,
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
    transform_system_prompt: Callable | None = None,
    after_model_response: Callable | None = None,
    before_tool_call: Callable | None = None,
    after_tool_call: Callable | None = None,
    should_stop_after_turn: Callable | None = None,
    get_steering_messages: Callable | None = None,
    get_follow_up_messages: Callable | None = None,
    stream_options: StreamOptions | None = None,
    tool_execution: str = "parallel",
    system_prompt: str | None = None,
) -> list[Message]:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")

    new_messages: list[Message] = []
    current_context = AgentContext(
        system_prompt=system_prompt
        if system_prompt is not None
        else context.system_prompt,
        messages=list(context.messages),
        tools=context.tools,
        extra=context.extra,
    )

    await emit_event(emit, AgentStartEvent())
    await emit_event(emit, TurnStartEvent())

    await _run_loop(
        current_context=current_context,
        new_messages=new_messages,
        provider=provider,
        model=model,
        convert_to_llm=convert_to_llm,
        transform_context=transform_context,
        transform_system_prompt=transform_system_prompt,
        after_model_response=after_model_response,
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
    new_messages: list[Message],
    provider: Provider,
    model: Model,
    convert_to_llm: Callable,
    transform_context: Callable | None,
    transform_system_prompt: Callable | None,
    after_model_response: Callable | None,
    before_tool_call: Callable | None,
    after_tool_call: Callable | None,
    should_stop_after_turn: Callable | None,
    get_steering_messages: Callable | None,
    get_follow_up_messages: Callable | None,
    stream_options: StreamOptions | None,
    tool_execution: str,
    emit: Callable,
) -> None:
    try:
        await _run_loop_inner(
            current_context=current_context,
            new_messages=new_messages,
            provider=provider,
            model=model,
            convert_to_llm=convert_to_llm,
            transform_context=transform_context,
            transform_system_prompt=transform_system_prompt,
            after_model_response=after_model_response,
            before_tool_call=before_tool_call,
            after_tool_call=after_tool_call,
            should_stop_after_turn=should_stop_after_turn,
            get_steering_messages=get_steering_messages,
            get_follow_up_messages=get_follow_up_messages,
            stream_options=stream_options,
            tool_execution=tool_execution,
            emit=emit,
        )
    except (HitlDetached, HitlAborted):
        # Caller (Agent.detach / Agent.abort_pending) emitted the event
        # already. Loop exits silently — assistant message and pending
        # state remain intact for the next respond() call.
        return


async def _run_loop_inner(
    *,
    current_context: AgentContext,
    new_messages: list[Message],
    provider: Provider,
    model: Model,
    convert_to_llm: Callable,
    transform_context: Callable | None,
    transform_system_prompt: Callable | None,
    after_model_response: Callable | None,
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
                await emit_event(emit, MessageStartEvent(message=msg))
                await emit_event(emit, MessageEndEvent(message=msg))
                current_context.messages.append(msg)
                new_messages.append(msg)

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls:
            if not first_turn:
                await emit_event(emit, TurnStartEvent())
            else:
                first_turn = False

            message = await _stream_assistant_response(
                current_context,
                provider,
                model,
                convert_to_llm,
                transform_context,
                transform_system_prompt,
                opts,
                emit,
            )
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                # Emit message_end before turn/agent end events.
                await emit_event(emit, MessageEndEvent(message=message))
                await emit_event(emit, TurnEndEvent(message=message, tool_results=[]))
                await emit_event(emit, AgentEndEvent(messages=new_messages))
                return

            # Apply after_model_response hook if configured.
            # The hook returns a TurnAction that can mutate the response,
            # inject messages, or change control flow.
            # message_end is deferred until here so that a mutated response
            # is what gets persisted via Agent._process_event.
            skip_tool_execution = False
            # Messages injected by after_model_response that must be appended
            # AFTER this turn's tool_results, not immediately (see below).
            deferred_inject_messages: list[Message] = []
            if after_model_response is not None:
                turn_action = await after_model_response(
                    message,
                    current_context,
                    signal=opts.signal,
                )
                if turn_action is not None:
                    if turn_action.response is not None:
                        # Replace both the local variable and the copy in
                        # context.messages that _stream_assistant_response appended.
                        message = turn_action.response
                        new_messages[-1] = message
                        if current_context.messages and isinstance(
                            current_context.messages[-1], AssistantMessage
                        ):
                            current_context.messages[-1] = message
                    # Emit message_end now — message reflects any hook mutation.
                    await emit_event(emit, MessageEndEvent(message=message))
                    if turn_action.inject_messages:
                        # When this turn still has tool_calls to execute (the
                        # "natural" decision falls through to tool handling
                        # below), the tool_results MUST be appended before any
                        # injected messages. Otherwise an injected (e.g. user)
                        # message lands between an assistant tool_use and its
                        # tool_result, which strict Anthropic-style endpoints
                        # reject ("tool_use ... without tool_result blocks
                        # immediately after"). Defer such injects until after
                        # the tool_results are appended.
                        _will_execute_tools = turn_action.decision == "natural" and any(
                            isinstance(c, ToolCall) for c in message.content
                        )
                        if _will_execute_tools:
                            deferred_inject_messages = list(turn_action.inject_messages)
                        else:
                            for inj in turn_action.inject_messages:
                                await emit_event(emit, MessageStartEvent(message=inj))
                                await emit_event(emit, MessageEndEvent(message=inj))
                                current_context.messages.append(inj)
                                new_messages.append(inj)
                    if turn_action.decision == "stop":
                        await emit_event(
                            emit, TurnEndEvent(message=message, tool_results=[])
                        )
                        await emit_event(emit, AgentEndEvent(messages=new_messages))
                        return
                    if turn_action.decision == "loop_to_model":
                        # Skip tool execution; re-invoke the model on next iteration.
                        # inject_messages are already appended to context above.
                        await emit_event(
                            emit, TurnEndEvent(message=message, tool_results=[])
                        )
                        skip_tool_execution = True
                    # decision == "natural" falls through to normal tool handling
                else:
                    # Hook returned None — emit message_end with unmodified message.
                    await emit_event(emit, MessageEndEvent(message=message))
            else:
                # No hook configured — emit message_end directly.
                await emit_event(emit, MessageEndEvent(message=message))

            if skip_tool_execution:
                has_more_tool_calls = True
                continue

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

            # Append messages deferred from after_model_response so they land
            # AFTER the tool_results, preserving tool_use/tool_result adjacency.
            for inj in deferred_inject_messages:
                await emit_event(emit, MessageStartEvent(message=inj))
                await emit_event(emit, MessageEndEvent(message=inj))
                current_context.messages.append(inj)
                new_messages.append(inj)

            await emit_event(
                emit, TurnEndEvent(message=message, tool_results=tool_results)
            )

            if should_stop_after_turn:
                stop_ctx = ShouldStopAfterTurnContext(
                    message=message,
                    tool_results=tool_results,
                    context=current_context,
                    new_messages=new_messages,
                )
                if await should_stop_after_turn(stop_ctx):
                    await emit_event(emit, AgentEndEvent(messages=new_messages))
                    return

            # Check for steering messages after tool execution
            if get_steering_messages and has_more_tool_calls:
                steering = await get_steering_messages() or []
                if steering:
                    for msg in steering:
                        await emit_event(emit, MessageStartEvent(message=msg))
                        await emit_event(emit, MessageEndEvent(message=msg))
                        current_context.messages.append(msg)
                        new_messages.append(msg)

        # The in-loop drain only fires when more tool calls remain, so a steer
        # that arrives while the model is finishing a tool-less turn would be
        # dropped. Drain it here and re-invoke the model so "steer anytime"
        # works during a final text turn too.
        if get_steering_messages:
            steering = await get_steering_messages() or []
            if steering:
                for msg in steering:
                    await emit_event(emit, MessageStartEvent(message=msg))
                    await emit_event(emit, MessageEndEvent(message=msg))
                    current_context.messages.append(msg)
                    new_messages.append(msg)
                first_turn = False
                continue

        # After inner loop completes, check for follow-up messages
        if get_follow_up_messages:
            follow_ups = await get_follow_up_messages() or []
            if follow_ups:
                for msg in follow_ups:
                    await emit_event(emit, MessageStartEvent(message=msg))
                    await emit_event(emit, MessageEndEvent(message=msg))
                    current_context.messages.append(msg)
                    new_messages.append(msg)
                first_turn = False
                continue

        break

    await emit_event(emit, AgentEndEvent(messages=new_messages))


async def _stream_assistant_response(
    context: AgentContext,
    provider: Provider,
    model: Model,
    convert_to_llm: Callable,
    transform_context: Callable | None,
    transform_system_prompt: Callable | None,
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

    sp = context.system_prompt
    if transform_system_prompt:
        sp = await transform_system_prompt(sp, signal=options.signal)

    stream = await provider.stream(
        model,
        llm_messages,
        system_prompt=sp,
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
                await emit_event(
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
                await emit_event(
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
                await emit_event(emit, MessageStartEvent(message=final_message))
            # message_end is intentionally NOT emitted here; the caller (_run_loop)
            # emits it after running after_model_response so the persisted message
            # reflects any hook mutation.
            return final_message

    # Fallback: stream ended without done/error event
    final_message = await stream.result()
    if added_partial:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)
        await emit_event(emit, MessageStartEvent(message=final_message))
    # message_end deferred to caller — see comment above.
    return final_message
