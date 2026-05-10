from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from cubepi.agent.loop import run_agent_loop, run_agent_loop_continue
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
    ) -> None:
        self._provider = provider
        self._state = AgentState(
            system_prompt=system_prompt,
            model=model,
            thinking=thinking,
        )
        if tools:
            self._state.tools = tools
        self.convert_to_llm = convert_to_llm or _default_convert_to_llm
        self.transform_context = transform_context
        self.before_tool_call = before_tool_call
        self.after_tool_call = after_tool_call
        self.should_stop_after_turn = should_stop_after_turn
        self.on_payload = on_payload
        self.on_response = on_response
        self.tool_execution = tool_execution
        self.checkpointer = checkpointer
        self.thread_id = thread_id

        self._steering_queue = _MessageQueue(steering_mode)
        self._follow_up_queue = _MessageQueue(follow_up_mode)
        self._listeners: list[Callable] = []
        self._active_signal: asyncio.Event | None = None
        self._active_done: asyncio.Event | None = None

    @property
    def state(self) -> AgentState:
        return self._state

    def subscribe(self, listener: Callable) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: (
            self._listeners.remove(listener) if listener in self._listeners else None
        )

    def steer(self, message: Message) -> None:
        self._steering_queue.enqueue(message)

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
        if self._state.is_streaming:
            raise RuntimeError(
                "Agent is already processing a prompt. "
                "Use steer() or follow_up() to queue messages."
            )

        if isinstance(message, str):
            messages = [
                UserMessage(content=[TextContent(text=message)], timestamp=time.time())
            ]
        elif isinstance(message, list):
            messages = message
        else:
            messages = [message]

        await self._run_prompt(messages)

    async def resume(self) -> None:
        if self._state.is_streaming:
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

    async def _process_event(self, event: AgentEvent) -> None:
        if event.type == "message_start":
            self._state.streaming_message = event.message
        elif event.type == "message_update":
            self._state.streaming_message = event.message
        elif event.type == "message_end":
            self._state.streaming_message = None
            self._state._messages.append(event.message)
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

        await self._emit_to_listeners(event)

    async def _emit_to_listeners(self, event: AgentEvent) -> None:
        for listener in self._listeners:
            result = listener(event, self._active_signal)
            if asyncio.iscoroutine(result):
                await result
