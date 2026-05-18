from __future__ import annotations

import asyncio
import copy
import inspect
from dataclasses import dataclass, field
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Literal,
    Protocol,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field

ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]


class ThinkingBudgets(BaseModel):
    """Token budgets for each thinking level."""

    minimal: int = 1024
    low: int = 2048
    medium: int = 8192
    high: int = 16384


def adjust_max_tokens_for_thinking(
    base_max_tokens: int,
    model_max_tokens: int,
    reasoning_level: ThinkingLevel,
    custom_budgets: ThinkingBudgets | None = None,
) -> tuple[int, int]:
    """Adjust max_tokens to reserve space for a thinking budget.

    Given a base max_tokens (the desired output capacity), increases it to
    accommodate the thinking budget while respecting the model's hard cap.
    If the model cap is too small to fit both, the thinking budget is reduced
    to leave at least ``min_output_tokens`` (1024) for output.

    Returns:
        A ``(max_tokens, thinking_budget)`` tuple.
    """
    if reasoning_level == "off":
        return base_max_tokens, 0

    budgets = custom_budgets or ThinkingBudgets()
    min_output_tokens = 1024

    # Clamp "xhigh" down to "high"
    level = "high" if reasoning_level == "xhigh" else reasoning_level
    thinking_budget: int = getattr(budgets, level)

    max_tokens = min(base_max_tokens + thinking_budget, model_max_tokens)

    if max_tokens - thinking_budget < min_output_tokens:
        thinking_budget = max(0, max_tokens - min_output_tokens)

    return max_tokens, thinking_budget


class ModelCost(BaseModel):
    input: float = 0
    output: float = 0
    cache_read: float = 0
    cache_write: float = 0


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class Model(BaseModel):
    id: str
    provider: str
    api: str = ""
    reasoning: bool = False
    context_window: int = 200_000
    max_tokens: int = 8192
    temperature: float = 0.7
    cost: ModelCost | None = None
    thinking_level_map: dict[str, str | None] | None = None


class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str = ""


class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    source: str = ""
    media_type: str = ""


class ThinkingContent(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str = ""


Content = TextContent | ImageContent


class ToolCall(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any]


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: list[Content]
    timestamp: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[Content | ThinkingContent | ToolCall]
    stop_reason: str = "stop"
    error_message: str | None = None
    usage: Usage | None = None
    timestamp: float | None = None
    provider_id: str = ""
    model_id: str = ""
    response_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResultMessage(BaseModel):
    role: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: list[Content]
    details: Any = None
    is_error: bool = False
    timestamp: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


Message = UserMessage | AssistantMessage | ToolResultMessage


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class StreamEvent(BaseModel):
    type: Literal[
        "start",
        "text_start",
        "text_delta",
        "text_end",
        "thinking_start",
        "thinking_delta",
        "thinking_end",
        "toolcall_start",
        "toolcall_delta",
        "toolcall_end",
        "done",
        "error",
    ]
    content_index: int | None = None
    delta: str | None = None
    partial: AssistantMessage | None = None
    error_message: str | None = None


class MessageStream:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        self._result_future: asyncio.Future[AssistantMessage] = (
            asyncio.get_running_loop().create_future()
        )
        self._producer_task: asyncio.Task | None = None

    def attach_task(self, task: asyncio.Task) -> None:
        self._producer_task = task
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc and not self._result_future.done():
            self._result_future.set_exception(exc)
            self._queue.put_nowait(None)

    def push(self, event: StreamEvent) -> None:
        self._queue.put_nowait(event)
        if event.type in ("done", "error"):
            self._queue.put_nowait(None)

    def set_result(self, message: AssistantMessage) -> None:
        if not self._result_future.done():
            self._result_future.set_result(message)

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self

    async def __anext__(self) -> StreamEvent:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def result(self) -> AssistantMessage:
        """Return the final assistant message.

        Blocks not just on the result future, but also on the producer
        task's completion — the producer's ``finally`` block runs
        :func:`_fire_response_listeners` (after ``set_result``), so
        without waiting for the task, callers under ``asyncio.run``
        teardown could exit before async response listeners have run.
        Producer exceptions are NOT re-raised here; they were already
        surfaced via the result future or the stream's error events.
        """
        msg = await self._result_future
        task = self._producer_task
        # Don't await ourselves — providers may call result() from inside
        # the producer task (e.g. to read an aborted result set by an
        # internal helper); that path would deadlock.
        if task is not None and not task.done() and task is not asyncio.current_task():
            try:
                await task
            except BaseException:
                # Producer error or cancellation — caller already has the
                # result (or an error message synthesized by the provider).
                pass
        return msg


@dataclass
class ProviderResponse:
    """HTTP response metadata exposed to on_response callbacks."""

    status: int
    headers: dict[str, str] = field(default_factory=dict)


OnPayloadCallback = Callable[[dict, Model], Awaitable[dict | None] | dict | None]
"""Optional callback for inspecting/replacing provider payloads before sending.
Return a dict to replace the payload, or None to keep unchanged."""

OnResponseCallback = Callable[["ProviderResponse", Model], Awaitable[None] | None]
"""Optional callback invoked after an HTTP response is received."""


OnRequestCallback = Callable[[dict, Model], Awaitable[None] | None]
"""Persistent observer. Fires just before HTTP send, after any per-call
``StreamOptions.on_payload`` mutation has been applied. Receives the final
wire payload dict and the Model. Return value is ignored."""

OnChunkCallback = Callable[["StreamEvent", Model], Awaitable[None] | None]
"""Persistent observer. Fires for every StreamEvent pushed onto the stream
(start, text_delta, thinking_delta, toolcall_delta, done, error, ...).
Heavy listeners should early-return on irrelevant event types — this hook
fires hot. Return value is ignored."""

OnResponseBodyCallback = Callable[
    [dict | None, Model, BaseException | None], Awaitable[None] | None
]
"""Persistent observer. Fires exactly once per ``stream()`` call, in a
finally block, after the stream terminates.

- body: assembled provider response as a dict (same shape a non-streaming
  call to the provider would have returned), or None if the stream failed
  before a response could be assembled.
- exc: the exception that ended the stream (including
  ``asyncio.CancelledError``), or None on normal completion.
Return value is ignored."""


async def _fire_listeners(listeners: list[Callable], *args: Any) -> None:
    """Invoke each listener with ``*args``. Listener return values and
    exceptions are ignored — a buggy listener must never crash the stream.

    Iterates a snapshot (``tuple(listeners)``) so a listener that detaches
    itself mid-iteration does not silently skip subsequent listeners.
    Callers are responsible for the hot-path guard
    (``if self._chunk_listeners:`` before ``await``) — see ``_emit`` helpers
    in each provider."""
    if not listeners:
        return
    for cb in tuple(listeners):
        try:
            result = cb(*args)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 — intentional broad catch
            _log_listener_exception(cb, exc)


def _fire_listeners_sync(listeners: list[Callable], *args: Any) -> None:
    """Synchronous variant of :func:`_fire_listeners`. Used in producer
    ``finally`` blocks where awaiting another coroutine after a
    cancellation is unreliable — the outer task is already cancelling, so
    a subsequent ``await`` may not get a chance to run its callee's body.

    Sync listeners are invoked directly. Async listeners are scheduled as
    detached tasks wrapped in :func:`_safe_run_coroutine` so that any
    exception raised inside the coroutine body is logged via
    :func:`_log_listener_exception` rather than bubbling up to asyncio as
    an "unhandled task exception" warning."""
    if not listeners:
        return
    for cb in tuple(listeners):
        try:
            result = cb(*args)
            if inspect.isawaitable(result):
                wrapped = _safe_run_coroutine(cb, result)
                try:
                    asyncio.create_task(wrapped)
                except RuntimeError:
                    # No running event loop (e.g. teardown) — close both
                    # the wrapper and the inner listener coroutine so
                    # neither leaks as a "coroutine was never awaited"
                    # warning.
                    wrapped.close()
                    close = getattr(result, "close", None)
                    if callable(close):
                        close()
        except Exception as exc:  # noqa: BLE001 — intentional broad catch
            _log_listener_exception(cb, exc)


async def _fire_request_listeners(
    listeners: list[Callable], payload: dict, model: "Model"
) -> None:
    """Fire :func:`subscribe_request` listeners on a defensive deep copy
    of the payload.

    ``subscribe_request`` is documented as an **observer** — the
    mutation hook is the per-call ``StreamOptions.on_payload`` slot,
    which runs before this. Without the copy, a redacting/logging
    listener that strips fields in place (``del payload["messages"]``)
    would silently alter the dict the provider then sends over the
    wire. The cost of one deepcopy per stream is negligible compared
    to the actual HTTP call.
    """
    if not listeners:
        return
    snapshot = copy.deepcopy(payload)
    await _fire_listeners(listeners, snapshot, model)


async def _fire_response_listeners(
    listeners: list[Callable],
    body: dict | None,
    model: "Model",
    exc: BaseException | None,
) -> None:
    """Fire response listeners with the right strategy for each
    termination path.

    Normal completion (``exc is None``) or an in-stream exception that is
    NOT a cancel: ``await`` each async listener inline so the producer
    task doesn't end before the listener has run — important because
    callers that use ``asyncio.run(main())`` will tear down the loop the
    moment ``main()`` returns, cancelling any still-detached listener
    task.

    Producer task cancellation (``exc`` is :class:`asyncio.CancelledError`):
    fall back to :func:`_fire_listeners_sync`. Awaiting inside a finally
    block of a cancelled task is unreliable — the runtime may skip past
    the await without running the callee — so synchronous listeners run
    inline and async listeners are scheduled as detached best-effort
    tasks. This is the same contract the cancellation tests pin.
    """
    if not listeners:
        return
    if isinstance(exc, asyncio.CancelledError):
        _fire_listeners_sync(listeners, body, model, exc)
        return
    for cb in tuple(listeners):
        try:
            result = cb(body, model, exc)
            if inspect.isawaitable(result):
                await result
        except Exception as listener_exc:  # noqa: BLE001
            _log_listener_exception(cb, listener_exc)


async def _safe_run_coroutine(cb: Callable, coro: Any) -> None:
    """Await an arbitrary coroutine, logging and swallowing any exception
    it raises. Used by :func:`_fire_listeners_sync` so async listener
    failures don't become asyncio unhandled-task-exception warnings."""
    try:
        await coro
    except Exception as exc:  # noqa: BLE001 — intentional broad catch
        _log_listener_exception(cb, exc)


def _log_listener_exception(cb: Callable, exc: BaseException) -> None:
    try:
        from loguru import logger

        logger.opt(exception=exc).warning(
            "cubepi provider listener {} raised; swallowed", cb
        )
    except ImportError:
        import logging

        logging.getLogger("cubepi.providers").warning(
            "cubepi provider listener %r raised; swallowed", cb, exc_info=exc
        )


def _detach(listeners: list, cb: Callable) -> None:
    try:
        listeners.remove(cb)
    except ValueError:
        pass


class StreamOptions(BaseModel):
    """Options bag for Provider.stream(), transparent to the agent loop."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    thinking: ThinkingLevel = "off"
    thinking_budgets: ThinkingBudgets | None = None
    signal: asyncio.Event | None = None
    on_payload: OnPayloadCallback | None = None
    on_response: OnResponseCallback | None = None


async def invoke_on_payload(
    callback: OnPayloadCallback | None,
    payload: dict,
    model: Model,
) -> dict:
    """Call *on_payload* and return the (possibly replaced) payload dict."""
    if callback is None:
        return payload
    result = callback(payload, model)
    if inspect.isawaitable(result):
        result = await result
    return result if isinstance(result, dict) else payload


async def invoke_on_response(
    callback: OnResponseCallback | None,
    response: ProviderResponse,
    model: Model,
) -> None:
    """Call *on_response* if provided."""
    if callback is None:
        return
    result = callback(response, model)
    if inspect.isawaitable(result):
        await result


@runtime_checkable
class Provider(Protocol):
    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream: ...


class BaseProvider:
    """Concrete base class for built-in cubepi providers.

    Built-in providers (Anthropic, OpenAI, OpenAI Responses, Faux) inherit
    from this class to gain the persistent listener registry used by
    ``cubepi.tracing`` and other observers. User-defined providers may also
    inherit from ``BaseProvider`` to opt in, or remain duck-typed against
    the ``Provider`` Protocol (which only requires ``stream()``).

    Concrete subclasses must implement ``stream()`` and call
    ``_fire_listeners`` at three points: after the request payload is
    finalized, for each ``StreamEvent`` pushed onto the stream, and exactly
    once in a ``finally`` block after the stream terminates.

    Per-call mutators (``StreamOptions.on_payload``,
    ``StreamOptions.on_response``) retain their existing single-slot
    semantics and fire independently of the persistent listener registry
    below.
    """

    def __init__(self) -> None:
        self._request_listeners: list[OnRequestCallback] = []
        self._chunk_listeners: list[OnChunkCallback] = []
        self._response_listeners: list[OnResponseBodyCallback] = []

    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        raise NotImplementedError

    def subscribe_request(self, cb: OnRequestCallback) -> Callable[[], None]:
        """Register a persistent observer for request payloads.

        Returns a detach callable that removes this specific subscription.
        """
        self._request_listeners.append(cb)
        return lambda: _detach(self._request_listeners, cb)

    def subscribe_chunk(self, cb: OnChunkCallback) -> Callable[[], None]:
        """Register a persistent observer for stream chunks.

        Returns a detach callable.
        """
        self._chunk_listeners.append(cb)
        return lambda: _detach(self._chunk_listeners, cb)

    def subscribe_response(self, cb: OnResponseBodyCallback) -> Callable[[], None]:
        """Register a persistent observer for assembled responses.

        Returns a detach callable.
        """
        self._response_listeners.append(cb)
        return lambda: _detach(self._response_listeners, cb)
