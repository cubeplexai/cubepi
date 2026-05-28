from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from cubepi.agent.loop import (
    run_agent_loop,
    run_agent_loop_continue,
    run_agent_loop_resume,
)
from cubepi.hitl import HitlError, HitlRequest
from cubepi.hitl.channel import HitlChannel
from cubepi.hitl.exceptions import HitlDetached
from cubepi.middleware.base import Middleware, compose_middleware
from cubepi.agent.types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentTool,
    MessageEndEvent,
    MessageStartEvent,
    TurnEndEvent,
)
from cubepi.providers.base import (
    AssistantMessage,
    Message,
    Model,
    OnPayloadCallback,
    OnResponseCallback,
    Provider,
    StreamOptions,
    TextContent,
    ThinkingLevel,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)

TMessage = TypeVar("TMessage")


def _default_convert_to_llm(messages: list[Message]) -> list[Message]:
    return list(messages)


class _MessageQueue:
    def __init__(self, mode: str = "one-at-a-time") -> None:
        self.mode = mode
        self._messages: list[Message] = []

    def enqueue(self, message: Message) -> None:
        self._messages.append(message)

    def has_items(self) -> bool:
        return len(self._messages) > 0

    def drain(self) -> list[Message]:
        if self.mode == "all":
            drained = self._messages[:]
            self._messages = []
            return drained
        if not self._messages:
            return []
        first = self._messages[0]
        self._messages = self._messages[1:]
        return [first]

    def remove(self, steer_id: str) -> bool:
        kept = [
            m
            for m in self._messages
            if getattr(m, "metadata", {}).get("steer_id") != steer_id
        ]
        removed = len(kept) != len(self._messages)
        self._messages = kept
        return removed

    def clear(self) -> None:
        self._messages = []


@dataclass
class AgentState:
    system_prompt: str = ""
    model: Model = field(
        default_factory=lambda: Model(id="unknown", provider="unknown")
    )
    thinking: ThinkingLevel = "off"
    is_streaming: bool = False
    streaming_message: Message | None = None
    error_message: str | None = None
    _tools: list[AgentTool] = field(default_factory=list)
    _messages: list[Message] = field(default_factory=list)
    _pending_tool_calls: set[str] = field(default_factory=set)

    @property
    def tools(self) -> list[AgentTool]:
        return list(self._tools)

    @tools.setter
    def tools(self, value: list[AgentTool]) -> None:
        self._tools = list(value)

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @messages.setter
    def messages(self, value: list[Message]) -> None:
        self._messages = list(value)

    @property
    def pending_tool_calls(self) -> set[str]:
        return set(self._pending_tool_calls)

    @pending_tool_calls.setter
    def pending_tool_calls(self, value: set[str]) -> None:
        self._pending_tool_calls = set(value)


class Agent(Generic[TMessage]):
    def __init__(
        self,
        *,
        provider: Provider,
        model: Model,
        system_prompt: str = "",
        tools: list[AgentTool] | None = None,
        thinking: ThinkingLevel = "off",
        convert_to_llm: Callable[[list[Message]], list[Message]] | None = None,
        transform_context: Callable | None = None,
        transform_system_prompt: Callable | None = None,
        after_model_response: Callable | None = None,
        before_tool_call: Callable | None = None,
        after_tool_call: Callable | None = None,
        should_stop_after_turn: Callable | None = None,
        on_payload: OnPayloadCallback | None = None,
        on_response: OnResponseCallback | None = None,
        steering_mode: str = "one-at-a-time",
        follow_up_mode: str = "one-at-a-time",
        tool_execution: str = "parallel",
        checkpointer: Any = None,
        thread_id: str | None = None,
        middleware: list[Middleware] | None = None,
        channel: HitlChannel | None = None,
    ) -> None:
        self._provider = provider
        self._state = AgentState(
            system_prompt=system_prompt,
            model=model,
            thinking=thinking,
        )
        if tools:
            self._state.tools = tools
        # Compose middleware hooks, then let explicit callables override.
        _mw_hooks = compose_middleware(middleware or [])
        self.convert_to_llm = (
            convert_to_llm or _mw_hooks.get("convert_to_llm") or _default_convert_to_llm
        )
        self.transform_context = transform_context or _mw_hooks.get("transform_context")
        self.transform_system_prompt = transform_system_prompt or _mw_hooks.get(
            "transform_system_prompt"
        )
        self.after_model_response = after_model_response or _mw_hooks.get(
            "after_model_response"
        )
        self.before_tool_call = before_tool_call or _mw_hooks.get("before_tool_call")
        self.after_tool_call = after_tool_call or _mw_hooks.get("after_tool_call")
        self.should_stop_after_turn = should_stop_after_turn or _mw_hooks.get(
            "should_stop_after_turn"
        )
        self.on_payload = on_payload
        self.on_response = on_response
        self.tool_execution = tool_execution
        self.checkpointer = checkpointer
        self.thread_id = thread_id
        self._channel = channel
        if channel is not None:
            channel._bind_emit(lambda e: self._process_event(e))
        self._run_lock = asyncio.Lock()

        self._extra: dict[str, Any] = {}

        self._steering_queue = _MessageQueue(steering_mode)
        self._follow_up_queue = _MessageQueue(follow_up_mode)
        self._listeners: list[Callable] = []
        self._active_signal: asyncio.Event | None = None
        self._active_done: asyncio.Event | None = None

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def channel(self) -> HitlChannel | None:
        return self._channel

    @property
    def in_flight_hitl_request(self) -> HitlRequest | None:
        if self._channel is None:
            raise HitlError("agent has no channel bound; pass channel= to Agent()")
        return self._channel.pending

    def subscribe(self, listener: Callable) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: (
            self._listeners.remove(listener) if listener in self._listeners else None
        )

    def steer(self, message: Message) -> None:
        self._steering_queue.enqueue(message)

    def cancel_steer(self, steer_id: str) -> bool:
        """Remove a not-yet-drained steering message by its steer_id.

        Returns True if a queued message was removed; False if it was already
        drained or never queued (best-effort cancel).
        """
        return self._steering_queue.remove(steer_id)

    def follow_up(self, message: Message) -> None:
        self._follow_up_queue.enqueue(message)

    def abort(self) -> None:
        if self._active_signal:
            self._active_signal.set()

    async def wait_for_idle(self) -> None:
        if self._active_done:
            await self._active_done.wait()

    def reset(self) -> None:
        self._state._messages = []
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state._pending_tool_calls = set()
        self._state.error_message = None
        self._steering_queue.clear()
        self._follow_up_queue.clear()

    async def prompt(self, message: str | Message | list[Message]) -> None:
        # Fail-fast guard: if the run-lock is already held OR a stream is in
        # progress, raise immediately instead of queueing on the lock. This
        # makes two concurrent cold prompt() calls fail-fast deterministically
        # (lock.locked() is the atomic source of truth — checking is_streaming
        # alone races against the to-be-set flag inside the lock body).
        if self._run_lock.locked() or self._state.is_streaming:
            raise RuntimeError(
                "Agent is already processing a prompt. "
                "Use steer() or follow_up() to queue messages."
            )
        async with self._run_lock:
            # Re-check under the lock in case streaming flipped during acquire.
            if self._state.is_streaming:  # pragma: no cover — defensive re-check
                raise RuntimeError(
                    "Agent is already processing a prompt. "
                    "Use steer() or follow_up() to queue messages."
                )

            if isinstance(message, str):
                messages = [
                    UserMessage(
                        content=[TextContent(text=message)], timestamp=time.time()
                    )
                ]
            elif isinstance(message, list):
                messages = message
            else:
                messages = [message]

            # Restore history and extra from checkpointer if this is first prompt
            if self.checkpointer and self.thread_id and not self._state._messages:
                data = await self.checkpointer.load(self.thread_id)
                if data:
                    if data.messages:
                        self._state._messages = list(data.messages)
                    self._extra = dict(data.extra)

            await self._run_prompt(messages)

    async def resume(self) -> None:
        # Same fail-fast pattern as prompt(): lock.locked() is the atomic gate.
        if self._run_lock.locked() or self._state.is_streaming:
            raise RuntimeError(
                "Agent is already processing. Wait for completion before continuing."
            )
        async with self._run_lock:
            if self._state.is_streaming:  # pragma: no cover — defensive re-check
                raise RuntimeError(
                    "Agent is already processing. Wait for completion before continuing."
                )

            if not self._state._messages:
                raise RuntimeError("No messages to continue from")

            last = self._state._messages[-1]
            if isinstance(last, AssistantMessage):
                # Check for queued messages
                steering = self._steering_queue.drain()
                if steering:
                    await self._run_prompt(steering)
                    return

                follow_ups = self._follow_up_queue.drain()
                if follow_ups:
                    await self._run_prompt(follow_ups)
                    return

                raise RuntimeError("Cannot continue from message role: assistant")

            await self._run_continuation()

    def _build_stream_options(self, signal: asyncio.Event) -> StreamOptions:
        return StreamOptions(
            thinking=self._state.thinking,
            signal=signal,
            on_payload=self.on_payload,
            on_response=self.on_response,
        )

    async def _run_prompt(self, messages: list[Message]) -> None:
        await self._run_with_lifecycle(
            lambda signal: run_agent_loop(
                prompts=messages,
                context=self._create_context_snapshot(),
                provider=self._provider,
                model=self._state.model,
                convert_to_llm=self.convert_to_llm,
                transform_context=self.transform_context,
                transform_system_prompt=self.transform_system_prompt,
                after_model_response=self.after_model_response,
                before_tool_call=self.before_tool_call,
                after_tool_call=self.after_tool_call,
                should_stop_after_turn=self.should_stop_after_turn,
                get_steering_messages=self._make_async_drain(self._steering_queue),
                get_follow_up_messages=self._make_async_drain(self._follow_up_queue),
                stream_options=self._build_stream_options(signal),
                tool_execution=self.tool_execution,
                emit=lambda e: self._process_event(e),
            )
        )

    async def _run_continuation(self) -> None:
        await self._run_with_lifecycle(
            lambda signal: run_agent_loop_continue(
                context=self._create_context_snapshot(),
                provider=self._provider,
                model=self._state.model,
                convert_to_llm=self.convert_to_llm,
                transform_context=self.transform_context,
                transform_system_prompt=self.transform_system_prompt,
                after_model_response=self.after_model_response,
                before_tool_call=self.before_tool_call,
                after_tool_call=self.after_tool_call,
                should_stop_after_turn=self.should_stop_after_turn,
                get_steering_messages=self._make_async_drain(self._steering_queue),
                get_follow_up_messages=self._make_async_drain(self._follow_up_queue),
                stream_options=self._build_stream_options(signal),
                tool_execution=self.tool_execution,
                emit=lambda e: self._process_event(e),
            )
        )

    @staticmethod
    def _make_async_drain(queue: _MessageQueue) -> Callable:
        async def _drain() -> list[Message]:
            return queue.drain()

        return _drain

    def _create_context_snapshot(self) -> AgentContext:
        return AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state._messages),
            tools=list(self._state._tools),
            extra=self._extra,
        )

    async def detach(self) -> None:
        from cubepi.agent.types import AgentSuspendedEvent

        if self._channel is None:
            raise HitlError("agent has no channel bound")
        pending = self._channel.pending
        if (
            pending is None
            or self._channel._future is None
            or self._channel._future.done()
        ):
            return  # nothing to detach
        # Emit the suspended event BEFORE triggering the exception, so listeners
        # see the real pending payload (codex pass 2 BLOCKING: previous draft
        # emitted from the loop with pending=None — fundamentally wrong).
        await self._process_event(AgentSuspendedEvent(pending_request=pending))
        self._channel._future.set_exception(HitlDetached())

    async def load_pending_hitl_request(self) -> HitlRequest | None:
        if self.checkpointer is None or self.thread_id is None:
            return None
        load_pending = getattr(self.checkpointer, "load_pending_request", None)
        if load_pending is None:
            return None  # checkpointer doesn't support HITL — graceful None
        return await load_pending(self.thread_id)

    async def respond(
        self, *, question_id: str | None = None, answer: Any
    ) -> None:  # pragma: no cover — E2E tested
        from cubepi.hitl.exceptions import (
            HitlNoPendingRequest,
            HitlStaleAnswer,
        )

        if self._channel is None:
            raise HitlError("agent has no channel bound")
        if not (self.thread_id and self.checkpointer):
            raise RuntimeError("respond() requires thread_id + checkpointer")

        load_pending = getattr(self.checkpointer, "load_pending_request", None)
        if load_pending is None:
            raise HitlError(
                "respond() requires a checkpointer that implements "
                "load_pending_request (and save_pending_request)"
            )

        async with self._run_lock:
            if not self._state._messages:
                data = await self.checkpointer.load(self.thread_id)
                if data:
                    self._state._messages = list(data.messages or [])
                    self._extra = dict(data.extra or {})

            pending = await load_pending(self.thread_id)
            if pending is None:
                raise HitlNoPendingRequest("no pending request on this thread")
            if question_id is None:
                question_id = pending.question_id
            if question_id != pending.question_id:
                raise HitlStaleAnswer(
                    f"answer for {question_id}, pending is {pending.question_id}"
                )

            self._channel.attach_resume_answer(question_id, answer)
            await self._run_hitl_resume()

    async def abort_pending(
        self, reason: str = "aborted by host"
    ) -> None:  # pragma: no cover — E2E tested
        """Abort a pending HITL request and CLOSE the conversation.

        Per spec §5.2 "abort closes the conversation" — no new model call.
        Two-phase: Phase 1 (no lock) interrupts any in-flight HITL await via
        the agent signal; Phase 2 (with lock) appends synthetic deny
        tool_results + a terminal stop_reason="aborted" assistant message
        and emits AgentAbortedEvent.
        """
        from cubepi.agent.types import AgentAbortedEvent

        if self._channel is None:
            raise HitlError("agent has no channel bound")
        if not (self.thread_id and self.checkpointer):
            raise RuntimeError("abort_pending() requires thread_id + checkpointer")

        # ============= Phase 1: interrupt any in-flight HITL await =============
        # If prompt() is currently suspended in channel.{ask,confirm,approve},
        # set the agent signal. _BaseChannel._await_answer races signal vs
        # future and raises HitlAborted when signal wins. HitlAborted
        # propagates through tool/middleware (HitlControlException is re-raised
        # by the selective handler in _execute_prepared) up to _run_loop's
        # outer silent catch. The HITL channel's finally calls
        # _on_pending_cleared(exc=HitlAborted) which clears persisted pending
        # (HitlAborted != HitlDetached).
        #
        # CRITICAL: do NOT acquire _run_lock here — prompt() holds it.
        in_flight = self._channel.pending
        if in_flight is not None:
            if self._active_signal is not None:
                self._active_signal.set()
            else:
                # Edge: channel has pending but agent has no active signal
                # (e.g. respond() race window). Cancel directly.
                await self._channel.cancel(in_flight.question_id, reason=reason)

        # ============= Phase 2: append synthetic deny + close conversation =====
        async with self._run_lock:
            save_pending = getattr(self.checkpointer, "save_pending_request", None)
            if save_pending is None:
                raise HitlError(
                    "abort_pending() requires a checkpointer that implements "
                    "save_pending_request"
                )

            # Reload messages from checkpoint to see whatever prompt() persisted.
            data = await self.checkpointer.load(self.thread_id)
            self._state._messages = list(data.messages or []) if data else []
            if not self._state._messages:
                # Nothing to close (no in-flight, no persisted history).
                await self._process_event(AgentAbortedEvent(reason=reason))
                return

            # Scan BACKWARDS for the most recent AssistantMessage that has
            # tool_calls still unresolved. We cannot just look at the tail —
            # an in-flight execute may have partially appended ToolResultMessage(s)
            # before signal-abort fired, leaving the tail as a ToolResultMessage
            # but the originating assistant turn still needing synthetic deny
            # for the OTHER tool_calls in the same batch.
            asst_pos = -1
            last_assistant = None
            for i in range(len(self._state._messages) - 1, -1, -1):
                msg = self._state._messages[i]
                if not isinstance(msg, AssistantMessage):
                    continue
                tcs = [c for c in msg.content if isinstance(c, ToolCall)]
                if not tcs:
                    continue
                already = {
                    m.tool_call_id
                    for m in self._state._messages[i + 1 :]
                    if isinstance(m, ToolResultMessage)
                }
                if any(tc.id not in already for tc in tcs):
                    asst_pos = i
                    last_assistant = msg
                    break

            if last_assistant is None or asst_pos < 0:
                # No unresolved assistant turn — conversation already closed
                # by some other path. Still clear pending + emit for observability.
                await save_pending(self.thread_id, None)
                await self._process_event(AgentAbortedEvent(reason=reason))
                return

            last = last_assistant
            tool_call_ids = [c.id for c in last.content if isinstance(c, ToolCall)]
            already_resolved = {
                m.tool_call_id
                for m in self._state._messages[asst_pos + 1 :]
                if isinstance(m, ToolResultMessage)
            }
            unresolved = [
                tc_id for tc_id in tool_call_ids if tc_id not in already_resolved
            ]

            # Synthesize deny tool_result for each unresolved tool_call.
            for tc_id in unresolved:
                tc = next(
                    c for c in last.content if isinstance(c, ToolCall) and c.id == tc_id
                )
                synthetic = ToolResultMessage(
                    tool_call_id=tc_id,
                    tool_name=tc.name,
                    content=[TextContent(text=f"aborted: {reason}")],
                    details={"hitl": {"decision": "aborted", "reason": reason}},
                    is_error=True,
                    timestamp=time.time(),
                )
                self._state._messages.append(synthetic)
                await self.checkpointer.append(self.thread_id, [synthetic])

            # Append terminal aborted assistant only if we actually appended
            # synthetic denials — otherwise the conversation already closed.
            if unresolved:
                term = AssistantMessage(
                    content=[TextContent(text=f"Conversation aborted: {reason}")],
                    stop_reason="aborted",
                    usage=Usage(),
                    timestamp=time.time(),
                )
                self._state._messages.append(term)
                await self.checkpointer.append(self.thread_id, [term])

            # Defensive clear (Phase 1's _on_pending_cleared usually did this,
            # but cross-process abort_pending may bypass Phase 1 entirely).
            await save_pending(self.thread_id, None)
            await self._process_event(AgentAbortedEvent(reason=reason))

    async def _run_hitl_resume(self) -> None:
        await self._run_with_lifecycle(
            lambda signal: run_agent_loop_resume(
                context=self._create_context_snapshot(),
                provider=self._provider,
                model=self._state.model,
                convert_to_llm=self.convert_to_llm,
                transform_context=self.transform_context,
                transform_system_prompt=self.transform_system_prompt,
                after_model_response=self.after_model_response,
                before_tool_call=self.before_tool_call,
                after_tool_call=self.after_tool_call,
                should_stop_after_turn=self.should_stop_after_turn,
                get_steering_messages=self._make_async_drain(self._steering_queue),
                get_follow_up_messages=self._make_async_drain(self._follow_up_queue),
                stream_options=self._build_stream_options(signal),
                tool_execution=self.tool_execution,
                emit=lambda e: self._process_event(e),
                checkpointer=self.checkpointer,
                thread_id=self.thread_id,
            )
        )

    async def _run_with_lifecycle(self, executor: Callable) -> None:
        signal = asyncio.Event()
        done = asyncio.Event()
        self._active_signal = signal
        self._active_done = done
        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None

        try:
            await executor(signal)
        except asyncio.CancelledError:
            # A cancel can land after the assistant message (carrying
            # tool_calls) was checkpointed but before the tool_results were.
            # That leaves the persisted history with orphan tool_calls, which
            # every provider rejects on the next turn. Backfill synthetic
            # tool_results so the thread stays resumable, then re-raise.
            await self._complete_cancelled_tool_calls()
            raise
        except Exception as error:
            await self._handle_run_failure(error, signal.is_set())
        finally:
            self._state.is_streaming = False
            self._state.streaming_message = None
            self._state._pending_tool_calls = set()
            self._active_signal = None
            done.set()
            self._active_done = None

    async def _handle_run_failure(self, error: Exception, aborted: bool) -> None:
        failure_message = AssistantMessage(
            content=[TextContent(text="")],
            stop_reason="aborted" if aborted else "error",
            error_message=str(error),
            usage=Usage(),
            timestamp=time.time(),
        )
        await self._process_event(MessageStartEvent(message=failure_message))
        await self._process_event(MessageEndEvent(message=failure_message))
        await self._process_event(
            TurnEndEvent(message=failure_message, tool_results=[])
        )
        await self._process_event(AgentEndEvent(messages=[failure_message]))

    async def _complete_cancelled_tool_calls(self) -> None:
        """Backfill tool_results for tool_calls left dangling by a cancel.

        Best-effort: a checkpoint failure here must never mask the
        CancelledError that triggered cleanup. Only the most recent assistant
        message can have un-answered tool_calls (the loop appends tool_results
        immediately after), so we scan back to it and synthesize a result for
        every tool_call id that has no ToolResultMessage yet.
        """
        try:
            last_idx = -1
            for i in range(len(self._state._messages) - 1, -1, -1):
                if isinstance(self._state._messages[i], AssistantMessage):
                    last_idx = i
                    break
            if last_idx == -1:
                return
            last_assistant = self._state._messages[last_idx]
            assert isinstance(last_assistant, AssistantMessage)

            # Only results from this turn count as answered — tool_call ids
            # are not globally unique, so scanning all history could treat a
            # reused id from an earlier turn as already answered and skip the
            # backfill, leaving the thread wedged.
            answered = {
                m.tool_call_id
                for m in self._state._messages[last_idx + 1 :]
                if isinstance(m, ToolResultMessage)
            }
            synthetic: list[Message] = []
            for block in last_assistant.content:
                if not isinstance(block, ToolCall) or block.id in answered:
                    continue
                synthetic.append(
                    ToolResultMessage(
                        tool_call_id=block.id,
                        tool_name=block.name,
                        content=[
                            TextContent(text="[Tool execution cancelled by user]")
                        ],
                        is_error=True,
                        timestamp=time.time(),
                    )
                )

            if not synthetic:
                return
            self._state._messages.extend(synthetic)
            if self.checkpointer and self.thread_id:
                await self.checkpointer.append(self.thread_id, synthetic)
        except asyncio.CancelledError:  # pragma: no cover - re-raise the trigger
            raise
        except Exception:  # pragma: no cover - cleanup must never mask the cancel
            pass

    async def _process_event(self, event: AgentEvent) -> None:
        if event.type == "message_start":
            self._state.streaming_message = event.message
        elif event.type == "message_update":
            self._state.streaming_message = event.message
        elif event.type == "message_end":
            self._state.streaming_message = None
            self._state._messages.append(event.message)
            if self.checkpointer and self.thread_id:
                await self.checkpointer.append(self.thread_id, [event.message])
        elif event.type == "tool_execution_start":
            self._state._pending_tool_calls = self._state._pending_tool_calls | {
                event.tool_call_id
            }
        elif event.type == "tool_execution_end":
            self._state._pending_tool_calls = self._state._pending_tool_calls - {
                event.tool_call_id
            }
        elif event.type == "turn_end":
            msg = event.message
            if isinstance(msg, AssistantMessage) and msg.error_message:
                self._state.error_message = msg.error_message
        elif event.type == "agent_end":
            self._state.streaming_message = None
            if self.checkpointer and self.thread_id:
                await self.checkpointer.save_extra(self.thread_id, self._extra)

        await self._emit_to_listeners(event)

    async def _emit_to_listeners(self, event: AgentEvent) -> None:
        for listener in self._listeners:
            result = listener(event, self._active_signal)
            if asyncio.iscoroutine(result):
                await result
