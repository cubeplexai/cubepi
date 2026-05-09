from __future__ import annotations

import asyncio
import time
from typing import Any

from cubepi.providers.base import (
    AssistantMessage,
    Content,
    ImageContent,
    Message,
    MessageStream,
    Model,
    StreamEvent,
    TextContent,
    ThinkingContent,
    ThinkingLevel,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    Usage,
    UserMessage,
)


class AnthropicProvider:
    def __init__(self, *, api_key: str | None = None) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        thinking: ThinkingLevel = "off",
        signal: asyncio.Event | None = None,
    ) -> MessageStream:
        ms = MessageStream()

        api_messages = [self._convert_message(m) for m in messages]
        kwargs: dict[str, Any] = {
            "model": model.id,
            "messages": api_messages,
            "max_tokens": model.max_tokens,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if tools:
            kwargs["tools"] = [self._convert_tool(t) for t in tools]
        if thinking != "off":
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._thinking_budget(thinking, model),
            }

        async def _produce() -> None:
            try:
                async with self._client.messages.stream(**kwargs) as stream:
                    partial = AssistantMessage(
                        content=[], usage=Usage(), timestamp=time.time()
                    )
                    ms.push(
                        StreamEvent(
                            type="start", partial=partial.model_copy(deep=True)
                        )
                    )

                    async for event in stream:
                        if signal and signal.is_set():
                            aborted = partial.model_copy(
                                update={
                                    "stop_reason": "aborted",
                                    "error_message": "Request was aborted",
                                }
                            )
                            ms.push(
                                StreamEvent(
                                    type="error",
                                    error_message="Request was aborted",
                                )
                            )
                            ms.set_result(aborted)
                            return

                        self._handle_event(event, partial, ms)

                    final_msg = stream.get_final_message()
                    result = self._convert_response(final_msg)
                    ms.push(StreamEvent(type="done"))
                    ms.set_result(result)

            except Exception as exc:
                error_msg = AssistantMessage(
                    content=[],
                    stop_reason="error",
                    error_message=str(exc),
                    usage=Usage(),
                    timestamp=time.time(),
                )
                ms.push(StreamEvent(type="error", error_message=str(exc)))
                ms.set_result(error_msg)

        asyncio.create_task(_produce())
        return ms

    @staticmethod
    def _convert_message(msg: Message) -> dict[str, Any]:
        if isinstance(msg, UserMessage):
            content = []
            for c in msg.content:
                if isinstance(c, TextContent):
                    content.append({"type": "text", "text": c.text})
                elif isinstance(c, ImageContent):
                    content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": c.media_type,
                                "data": c.source,
                            },
                        }
                    )
            return {"role": "user", "content": content}

        elif isinstance(msg, AssistantMessage):
            content = []
            for c in msg.content:
                if isinstance(c, TextContent):
                    content.append({"type": "text", "text": c.text})
                elif isinstance(c, ThinkingContent):
                    content.append({"type": "thinking", "thinking": c.thinking})
                elif isinstance(c, ToolCall):
                    content.append(
                        {
                            "type": "tool_use",
                            "id": c.id,
                            "name": c.name,
                            "input": c.arguments,
                        }
                    )
            return {"role": "assistant", "content": content}

        elif isinstance(msg, ToolResultMessage):
            tool_content = []
            for c in msg.content:
                if isinstance(c, TextContent):
                    tool_content.append({"type": "text", "text": c.text})
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": tool_content,
                        "is_error": msg.is_error,
                    }
                ],
            }

        return {"role": "user", "content": []}

    @staticmethod
    def _convert_tool(td: ToolDefinition) -> dict[str, Any]:
        return {
            "name": td.name,
            "description": td.description,
            "input_schema": td.parameters,
        }

    @staticmethod
    def _thinking_budget(level: ThinkingLevel, model: Model) -> int:
        budgets = {
            "minimal": 1024,
            "low": 2048,
            "medium": 4096,
            "high": 8192,
            "xhigh": 16384,
        }
        return budgets.get(level, 4096)

    def _handle_event(
        self, event: Any, partial: AssistantMessage, ms: MessageStream
    ) -> None:
        etype = getattr(event, "type", "")
        if etype == "content_block_start":
            block = event.content_block
            if block.type == "text":
                partial.content.append(TextContent(text=""))
                ms.push(
                    StreamEvent(
                        type="text_start", partial=partial.model_copy(deep=True)
                    )
                )
            elif block.type == "thinking":
                partial.content.append(ThinkingContent(thinking=""))
                ms.push(
                    StreamEvent(
                        type="thinking_start",
                        partial=partial.model_copy(deep=True),
                    )
                )
            elif block.type == "tool_use":
                partial.content.append(
                    ToolCall(id=block.id, name=block.name, arguments={})
                )
                ms.push(
                    StreamEvent(
                        type="toolcall_start",
                        partial=partial.model_copy(deep=True),
                    )
                )
        elif etype == "content_block_delta":
            delta = event.delta
            if hasattr(delta, "text"):
                if partial.content and isinstance(partial.content[-1], TextContent):
                    partial.content[-1] = TextContent(
                        text=partial.content[-1].text + delta.text
                    )
                ms.push(
                    StreamEvent(
                        type="text_delta",
                        delta=delta.text,
                        partial=partial.model_copy(deep=True),
                    )
                )
            elif hasattr(delta, "thinking"):
                if partial.content and isinstance(
                    partial.content[-1], ThinkingContent
                ):
                    partial.content[-1] = ThinkingContent(
                        thinking=partial.content[-1].thinking + delta.thinking
                    )
                ms.push(
                    StreamEvent(
                        type="thinking_delta",
                        delta=delta.thinking,
                        partial=partial.model_copy(deep=True),
                    )
                )
            elif hasattr(delta, "partial_json"):
                ms.push(
                    StreamEvent(
                        type="toolcall_delta",
                        delta=delta.partial_json,
                        partial=partial.model_copy(deep=True),
                    )
                )
        elif etype == "content_block_stop":
            if partial.content:
                last = partial.content[-1]
                if isinstance(last, TextContent):
                    ms.push(
                        StreamEvent(
                            type="text_end",
                            partial=partial.model_copy(deep=True),
                        )
                    )
                elif isinstance(last, ThinkingContent):
                    ms.push(
                        StreamEvent(
                            type="thinking_end",
                            partial=partial.model_copy(deep=True),
                        )
                    )
                elif isinstance(last, ToolCall):
                    ms.push(
                        StreamEvent(
                            type="toolcall_end",
                            partial=partial.model_copy(deep=True),
                        )
                    )

    @staticmethod
    def _convert_response(response: Any) -> AssistantMessage:
        content: list[Any] = []
        for block in response.content:
            if block.type == "text":
                content.append(TextContent(text=block.text))
            elif block.type == "thinking":
                content.append(ThinkingContent(thinking=block.thinking))
            elif block.type == "tool_use":
                content.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )

        stop_reason_map = {
            "end_turn": "stop",
            "tool_use": "tool_use",
            "max_tokens": "length",
        }

        return AssistantMessage(
            content=content,
            stop_reason=stop_reason_map.get(response.stop_reason, response.stop_reason),
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_tokens=getattr(
                    response.usage, "cache_read_input_tokens", 0
                )
                or 0,
                cache_write_tokens=getattr(
                    response.usage, "cache_creation_input_tokens", 0
                )
                or 0,
            ),
            timestamp=time.time(),
        )
