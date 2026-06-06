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

from cubepi.types import JsonObject, StructuredValue

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
    provider_id: str = ""
    api: str = ""
    reasoning: bool = False
    context_window: int = 200_000
    max_tokens: int = 8192
    temperature: float = 0.7
    cost: ModelCost | None = None
    thinking_level_map: dict[str, str | None] | None = None


@dataclass(frozen=True)
class BoundModel:
    provider: Provider
    spec: Model


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
    arguments: JsonObject


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: list[Content]
    timestamp: float | None = None
    metadata: JsonObject = Field(default_factory=dict)
    run_id: str | None = None


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
    metadata: JsonObject = Field(default_factory=dict)
    run_id: str | None = None


class ToolResultMessage(BaseModel):
    role: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    content: list[Content]
    details: StructuredValue = None
    is_error: bool = False
    timestamp: float | None = None
    metadata: JsonObject = Field(default_factory=dict)
    run_id: str | None = None


Message = UserMessage | AssistantMessage | ToolResultMessage


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: JsonObject


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


def format_provider_error(
    exc: BaseException,
    model: "Model",
    base_url: str | None = None,
) -> str:
    """Build a self-describing error string for a failed provider call.

    SDK exceptions like openai's ``APIConnectionError`` stringify to bare
    ``"Connection error."`` and drop the transport failure into ``__cause__``
    / ``__context__``. Without provider/model/cause context, a logged or
    persisted error tells you nothing about *which* model failed or *why*.
    """
    target = f"{model.provider_id}/{model.id}" if model.provider_id else model.id
    if base_url:
        target = f"{target} @ {base_url}"

    chain: list[str] = [f"{type(exc).__name__}: {exc}"]
    seen: set[int] = {id(exc)}
    cur: BaseException | None = exc
    while cur is not None:
        nxt = cur.__cause__ or cur.__context__
        if nxt is None or id(nxt) in seen:
            break
        seen.add(id(nxt))
        chain.append(f"{type(nxt).__name__}: {nxt}")
        cur = nxt

    detail = " <- ".join(chain)
    return f"[{target}] {detail}"


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
            # ``asyncio.shield`` lets the producer's finally (response
            # listener cleanup) keep running independently if our caller
            # is cancelled while we're waiting. We re-raise the caller's
            # CancelledError so cooperative shutdown/timeouts still
            # propagate; the producer continues unaffected.
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                raise
            except BaseException:
                # Producer raised — caller already has the result via the
                # future. Swallow so result() returns the message.
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


async def _fire_chunk_listeners(
    listeners: list[Callable], event: "StreamEvent", model: "Model"
) -> None:
    """Fire :func:`subscribe_chunk` listeners with **per-listener** deep
    copies of the event.

    The consumer's queued ``StreamEvent`` (already pushed onto the
    ``MessageStream``) and every other listener's copy are isolated.
    Without per-listener copies, a redacting listener that mutated
    ``event.partial`` would silently alter what later chunk listeners
    observed in the same stream.
    """
    if not listeners:
        return
    for cb in tuple(listeners):
        try:
            result = cb(event.model_copy(deep=True), model)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 — intentional broad catch
            _log_listener_exception(cb, exc)


async def _fire_request_listeners(
    listeners: list[Callable], payload: dict, model: "Model"
) -> None:
    """Fire :func:`subscribe_request` listeners with **per-listener** deep
    copies of the payload.

    ``subscribe_request`` is documented as an **observer** — the
    mutation hook is the per-call ``StreamOptions.on_payload`` slot,
    which runs before this. Two isolation properties matter here:

    1. The dict the provider is about to send over the wire (the
       caller's ``kwargs``) must not be mutated by any listener.
    2. Multi-subscriber observability must be order-independent — one
       listener redacting fields in place must not affect what later
       listeners observe.

    Both are achieved by giving each listener its own ``deepcopy``.
    Cost: at most one deepcopy per registered listener per stream call,
    which is negligible compared to the HTTP call itself.
    """
    if not listeners:
        return
    for cb in tuple(listeners):
        try:
            snapshot = copy.deepcopy(payload)
            result = cb(snapshot, model)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001 — intentional broad catch
            _log_listener_exception(cb, exc)


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

    In both paths, each listener receives its own ``deepcopy`` of
    ``body`` so multi-subscriber observability is order-independent.
    """
    if not listeners:
        return
    if isinstance(exc, asyncio.CancelledError):
        _fire_listeners_sync_per_listener(listeners, body, model, exc)
        return
    for cb in tuple(listeners):
        try:
            snapshot = copy.deepcopy(body) if body is not None else None
            result = cb(snapshot, model, exc)
            if inspect.isawaitable(result):
                await result
        except Exception as listener_exc:  # noqa: BLE001
            _log_listener_exception(cb, listener_exc)


def _fire_listeners_sync_per_listener(
    listeners: list[Callable],
    body: dict | None,
    model: "Model",
    exc: BaseException | None,
) -> None:
    """Sync fanout for response listeners on the cancellation path,
    giving each listener its own deep copy of ``body``."""
    if not listeners:
        return
    for cb in tuple(listeners):
        try:
            snapshot = copy.deepcopy(body) if body is not None else None
            result = cb(snapshot, model, exc)
            if inspect.isawaitable(result):
                wrapped = _safe_run_coroutine(cb, result)
                try:
                    asyncio.create_task(wrapped)
                except RuntimeError:
                    wrapped.close()
                    close = getattr(result, "close", None)
                    if callable(close):
                        close()
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

    async def generate(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        thinking: ThinkingLevel | None = None,
        thinking_budgets: ThinkingBudgets | None = None,
    ) -> AssistantMessage: ...


class BaseProvider:
    """Concrete base class for built-in cubepi providers.

    Built-in providers (Anthropic, OpenAI, OpenAI Responses, Faux) inherit
    from this class to gain the persistent listener registry used by
    ``cubepi.tracing`` and other observers. User-defined providers should
    inherit from ``BaseProvider`` unless they implement the full ``Provider``
    protocol themselves.

    Concrete subclasses must implement ``stream()`` and call
    ``_fire_listeners`` at three points: after the request payload is
    finalized, for each ``StreamEvent`` pushed onto the stream, and exactly
    once in a ``finally`` block after the stream terminates.

    Per-call mutators (``StreamOptions.on_payload``,
    ``StreamOptions.on_response``) retain their existing single-slot
    semantics and fire independently of the persistent listener registry
    below.
    """

    def __init__(self, *, provider_id: str = "") -> None:
        self.provider_id = provider_id
        self._request_listeners: list[OnRequestCallback] = []
        self._chunk_listeners: list[OnChunkCallback] = []
        self._response_listeners: list[OnResponseBodyCallback] = []

    def model(
        self,
        id: str,
        *,
        api: str = "",
        reasoning: bool = False,
        context_window: int = 200_000,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        cost: ModelCost | None = None,
        thinking_level_map: dict[str, str | None] | None = None,
    ) -> BoundModel:
        return BoundModel(
            provider=self,
            spec=Model(
                id=id,
                provider_id=self.provider_id,
                api=api,
                reasoning=reasoning,
                context_window=context_window,
                max_tokens=max_tokens,
                temperature=temperature,
                cost=cost,
                thinking_level_map=thinking_level_map,
            ),
        )

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

    async def generate(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        thinking: ThinkingLevel | None = None,
        thinking_budgets: ThinkingBudgets | None = None,
    ) -> AssistantMessage:
        """Run a single provider call and return the final assistant message."""
        model_updates: dict[str, int | float] = {}
        if max_output_tokens is not None:
            model_updates["max_tokens"] = max_output_tokens
        if temperature is not None:
            model_updates["temperature"] = temperature
        if model_updates:
            model = model.model_copy(update=model_updates)

        option_updates: dict[str, ThinkingLevel | ThinkingBudgets] = {}
        if thinking is not None:
            option_updates["thinking"] = thinking
        if thinking_budgets is not None:
            option_updates["thinking_budgets"] = thinking_budgets
        if option_updates:
            base_options = options or StreamOptions()
            options = base_options.model_copy(update=option_updates)

        stream = await self.stream(
            model=model,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            options=options,
        )
        async for event in stream:
            if event.type in ("done", "error"):
                break
        return await stream.result()

    def _error_message(self, exc: BaseException, model: Model) -> str:
        """Format a failed-call error with provider/model/base_url + cause."""
        base_url: str | None = None
        client = getattr(self, "_client", None)
        if client is not None:
            base_url = str(getattr(client, "base_url", "") or "") or None
        return format_provider_error(exc, model, base_url)

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

    async def _emit(
        self,
        ms: "MessageStream",
        event: "StreamEvent",
        model: Model | None,
    ) -> None:
        """Push an event to the message stream and fan out to chunk
        listeners with an isolated deep copy.

        The synchronous guard on ``self._chunk_listeners`` makes the
        no-listener case zero-await — important because this fires on
        every text delta. When listeners are present, each receives a
        ``model_copy(deep=True)`` of the event so a redacting/mutating
        observer cannot edit the same object that was already enqueued
        for the ``async for`` consumer.

        ``model`` may be ``None`` when invoked from internal helpers
        that bypass ``stream()`` (e.g. ``FauxProvider._stream_with_deltas``
        is exposed for direct calls in tests). In that case the listener
        fan-out is skipped — the test isn't observing listeners anyway.
        """
        ms.push(event)
        if model is not None and self._chunk_listeners:
            await _fire_chunk_listeners(self._chunk_listeners, event, model)
