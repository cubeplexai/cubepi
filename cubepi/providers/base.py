from __future__ import annotations

import asyncio
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
        return await self._result_future


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
