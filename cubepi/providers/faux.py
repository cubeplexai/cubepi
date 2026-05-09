from __future__ import annotations

import asyncio
import json
import math
import time
from typing import Any, Awaitable, Callable

from cubepi.providers.base import (
    AssistantMessage,
    Message,
    MessageStream,
    Model,
    StreamEvent,
    TextContent,
    ThinkingBudgets,
    ThinkingContent,
    ThinkingLevel,
    ToolCall,
    ToolDefinition,
    Usage,
)

FauxContentBlock = TextContent | ThinkingContent | ToolCall

FauxResponseFactory = Callable[
    [list[Message], Model],
    AssistantMessage | Awaitable[AssistantMessage],
]

FauxResponseStep = AssistantMessage | FauxResponseFactory


def _random_id(prefix: str) -> str:
    import random

    return f"{prefix}:{int(time.time() * 1000)}:{random.randbytes(6).hex()}"


def faux_text(text: str) -> TextContent:
    return TextContent(text=text)


def faux_thinking(thinking: str) -> ThinkingContent:
    return ThinkingContent(thinking=thinking)


def faux_tool_call(
    name: str,
    arguments: dict[str, Any],
    *,
    id: str | None = None,
) -> ToolCall:
    return ToolCall(id=id or _random_id("tool"), name=name, arguments=arguments)


def faux_assistant_message(
    content: str | FauxContentBlock | list[FauxContentBlock],
    *,
    stop_reason: str = "stop",
    error_message: str | None = None,
) -> AssistantMessage:
    if isinstance(content, str):
        blocks: list[FauxContentBlock] = [faux_text(content)]
    elif isinstance(content, list):
        blocks = content
    else:
        blocks = [content]
    return AssistantMessage(
        content=blocks,
        stop_reason=stop_reason,
        error_message=error_message,
        usage=Usage(),
        timestamp=time.time(),
    )


def _split_by_token_size(text: str, min_size: int, max_size: int) -> list[str]:
    import random

    chunks: list[str] = []
    i = 0
    while i < len(text):
        token_size = random.randint(min_size, max_size)
        char_size = max(1, token_size * 4)
        chunks.append(text[i : i + char_size])
        i += char_size
    return chunks or [""]


class FauxProvider:
    def __init__(
        self,
        *,
        tokens_per_second: float | None = None,
        token_size_min: int = 3,
        token_size_max: int = 5,
    ) -> None:
        self._responses: list[FauxResponseStep] = []
        self._tokens_per_second = tokens_per_second
        self._min = max(1, min(token_size_min, token_size_max))
        self._max = max(self._min, token_size_max)
        self.call_count = 0

    def set_responses(self, responses: list[FauxResponseStep]) -> None:
        self._responses = list(responses)

    def append_responses(self, responses: list[FauxResponseStep]) -> None:
        self._responses.extend(responses)

    @property
    def pending_response_count(self) -> int:
        return len(self._responses)

    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        thinking: ThinkingLevel = "off",
        thinking_budgets: ThinkingBudgets | None = None,
        signal: asyncio.Event | None = None,
    ) -> MessageStream:
        ms = MessageStream()
        self.call_count += 1

        step = self._responses.pop(0) if self._responses else None

        async def _produce() -> None:
            try:
                if step is None:
                    error_msg = AssistantMessage(
                        content=[],
                        stop_reason="error",
                        error_message="No more faux responses queued",
                        usage=Usage(),
                        timestamp=time.time(),
                    )
                    ms.push(
                        StreamEvent(type="error", error_message=error_msg.error_message)
                    )
                    ms.set_result(error_msg)
                    return

                if callable(step):
                    import inspect

                    if inspect.iscoroutinefunction(step):
                        resolved = await step(messages, model)
                    else:
                        resolved = step(messages, model)
                else:
                    resolved = step

                await self._stream_with_deltas(ms, resolved, signal)
            except BaseException as exc:
                error_msg = AssistantMessage(
                    content=[],
                    stop_reason="error",
                    error_message=str(exc),
                    usage=Usage(),
                    timestamp=time.time(),
                )
                ms.push(StreamEvent(type="error", error_message=str(exc)))
                ms.set_result(error_msg)
                if not isinstance(exc, Exception):
                    raise

        asyncio.create_task(_produce())
        return ms

    async def _stream_with_deltas(
        self,
        stream: MessageStream,
        message: AssistantMessage,
        signal: asyncio.Event | None,
    ) -> None:
        partial = AssistantMessage(
            content=[],
            stop_reason=message.stop_reason,
            usage=message.usage,
            timestamp=message.timestamp,
        )

        if signal and signal.is_set():
            aborted = self._make_aborted(partial)
            stream.push(StreamEvent(type="error", error_message="Request was aborted"))
            stream.set_result(aborted)
            return

        stream.push(StreamEvent(type="start", partial=partial.model_copy(deep=True)))

        for block in message.content:
            if signal and signal.is_set():
                aborted = self._make_aborted(partial)
                stream.push(
                    StreamEvent(type="error", error_message="Request was aborted")
                )
                stream.set_result(aborted)
                return

            if isinstance(block, ThinkingContent):
                partial.content.append(ThinkingContent(thinking=""))
                stream.push(
                    StreamEvent(
                        type="thinking_start", partial=partial.model_copy(deep=True)
                    )
                )
                for chunk in _split_by_token_size(block.thinking, self._min, self._max):
                    await self._schedule_chunk(chunk)
                    if signal and signal.is_set():
                        aborted = self._make_aborted(partial)
                        stream.push(
                            StreamEvent(
                                type="error", error_message="Request was aborted"
                            )
                        )
                        stream.set_result(aborted)
                        return
                    last = partial.content[-1]
                    if isinstance(last, ThinkingContent):
                        partial.content[-1] = ThinkingContent(
                            thinking=last.thinking + chunk
                        )
                    stream.push(
                        StreamEvent(
                            type="thinking_delta",
                            delta=chunk,
                            partial=partial.model_copy(deep=True),
                        )
                    )
                stream.push(
                    StreamEvent(
                        type="thinking_end", partial=partial.model_copy(deep=True)
                    )
                )

            elif isinstance(block, TextContent):
                partial.content.append(TextContent(text=""))
                stream.push(
                    StreamEvent(
                        type="text_start", partial=partial.model_copy(deep=True)
                    )
                )
                for chunk in _split_by_token_size(block.text, self._min, self._max):
                    await self._schedule_chunk(chunk)
                    if signal and signal.is_set():
                        aborted = self._make_aborted(partial)
                        stream.push(
                            StreamEvent(
                                type="error", error_message="Request was aborted"
                            )
                        )
                        stream.set_result(aborted)
                        return
                    last = partial.content[-1]
                    if isinstance(last, TextContent):
                        partial.content[-1] = TextContent(text=last.text + chunk)
                    stream.push(
                        StreamEvent(
                            type="text_delta",
                            delta=chunk,
                            partial=partial.model_copy(deep=True),
                        )
                    )
                stream.push(
                    StreamEvent(type="text_end", partial=partial.model_copy(deep=True))
                )

            elif isinstance(block, ToolCall):
                partial.content.append(
                    ToolCall(id=block.id, name=block.name, arguments={})
                )
                stream.push(
                    StreamEvent(
                        type="toolcall_start", partial=partial.model_copy(deep=True)
                    )
                )
                json_str = json.dumps(block.arguments)
                for chunk in _split_by_token_size(json_str, self._min, self._max):
                    await self._schedule_chunk(chunk)
                    if signal and signal.is_set():
                        aborted = self._make_aborted(partial)
                        stream.push(
                            StreamEvent(
                                type="error", error_message="Request was aborted"
                            )
                        )
                        stream.set_result(aborted)
                        return
                    stream.push(
                        StreamEvent(
                            type="toolcall_delta",
                            delta=chunk,
                            partial=partial.model_copy(deep=True),
                        )
                    )
                last = partial.content[-1]
                if isinstance(last, ToolCall):
                    partial.content[-1] = ToolCall(
                        id=block.id, name=block.name, arguments=block.arguments
                    )
                stream.push(
                    StreamEvent(
                        type="toolcall_end", partial=partial.model_copy(deep=True)
                    )
                )

        if message.stop_reason in ("error", "aborted"):
            stream.push(StreamEvent(type="error", error_message=message.error_message))
            stream.set_result(message)
            return

        stream.push(StreamEvent(type="done"))
        stream.set_result(message)

    async def _schedule_chunk(self, chunk: str) -> None:
        if not self._tokens_per_second or self._tokens_per_second <= 0:
            await asyncio.sleep(0)
            return
        tokens = max(1, math.ceil(len(chunk) / 4))
        delay = tokens / self._tokens_per_second
        await asyncio.sleep(delay)

    @staticmethod
    def _make_aborted(partial: AssistantMessage) -> AssistantMessage:
        return partial.model_copy(
            update={
                "stop_reason": "aborted",
                "error_message": "Request was aborted",
                "timestamp": time.time(),
            }
        )
