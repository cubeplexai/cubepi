from __future__ import annotations

import asyncio
import time
from typing import Any, Literal, Protocol, runtime_checkable

from cubepi.providers.base import (
    AssistantMessage,
    ImageContent,
    Message,
    MessageStream,
    Model,
    ProviderResponse,
    StreamEvent,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    Usage,
    UserMessage,
    adjust_max_tokens_for_thinking,
    invoke_on_payload,
    invoke_on_response,
)
from cubepi.providers.models import clamp_thinking_level

CacheRetention = Literal["short", "long", "none"]


@runtime_checkable
class CacheMarkerPolicy(Protocol):
    """Policy controlling where Anthropic cache_control markers are inserted.

    See `cubepi/docs/specs/2026-05-13-cubepi-cubebox-readiness-design.md` § D3.
    """

    def mark_system(self) -> bool: ...
    def mark_last_tool(self) -> bool: ...
    def message_breakpoint_indices(
        self,
        messages: list[Message],
    ) -> list[int]: ...


class DefaultCacheMarkerPolicy:
    """Preserves cubepi v0.2 behavior: system + last message + last tool."""

    def mark_system(self) -> bool:
        return True

    def mark_last_tool(self) -> bool:
        return True

    def message_breakpoint_indices(self, messages: list[Message]) -> list[int]:
        return [len(messages) - 1] if messages else []


class AnthropicProvider:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        cache_retention: CacheRetention = "short",
        cache_policy: CacheMarkerPolicy | None = None,
    ) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._cache_retention = cache_retention
        self._cache_policy: CacheMarkerPolicy = cache_policy or DefaultCacheMarkerPolicy()

    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        opts = options or StreamOptions()
        ms = MessageStream()
        thinking = clamp_thinking_level(model, opts.thinking)

        cache_control = self._get_cache_control()
        api_messages = [self._convert_message(m) for m in messages]
        if cache_control:
            indices = self._cache_policy.message_breakpoint_indices(messages)
            self._apply_indices_markers(api_messages, indices, cache_control)

        max_tokens, thinking_budget = adjust_max_tokens_for_thinking(
            base_max_tokens=model.max_tokens,
            model_max_tokens=model.context_window,
            reasoning_level=thinking,
            custom_budgets=opts.thinking_budgets,
        )

        kwargs: dict[str, Any] = {
            "model": model.id,
            "messages": api_messages,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_prompt,
                    **(
                        {"cache_control": cache_control}
                        if cache_control and self._cache_policy.mark_system()
                        else {}
                    ),
                }
            ]
        if tools:
            api_tools = [self._convert_tool(t) for t in tools]
            if cache_control and api_tools and self._cache_policy.mark_last_tool():
                api_tools[-1]["cache_control"] = cache_control
            kwargs["tools"] = api_tools
        if thinking != "off" and thinking_budget > 0:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

        async def _produce() -> None:
            try:
                nonlocal kwargs
                kwargs = await invoke_on_payload(opts.on_payload, kwargs, model)

                async with self._client.messages.stream(**kwargs) as stream:
                    # Invoke on_response with HTTP metadata if available
                    http_response = getattr(stream, "response", None)
                    if http_response is not None:
                        await invoke_on_response(
                            opts.on_response,
                            ProviderResponse(
                                status=http_response.status_code,
                                headers=dict(http_response.headers),
                            ),
                            model,
                        )

                    partial = AssistantMessage(
                        content=[],
                        usage=Usage(),
                        timestamp=time.time(),
                        provider_id=model.provider,
                        model_id=model.id,
                    )
                    ms.push(
                        StreamEvent(type="start", partial=partial.model_copy(deep=True))
                    )

                    async for event in stream:
                        if opts.signal and opts.signal.is_set():
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
                    result = self._convert_response(final_msg, model)
                    ms.push(StreamEvent(type="done"))
                    ms.set_result(result)

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

        ms.attach_task(asyncio.create_task(_produce()))
        return ms

    def _get_cache_control(self) -> dict[str, str] | None:
        if self._cache_retention == "none":
            return None
        cc: dict[str, str] = {"type": "ephemeral"}
        if self._cache_retention == "long":
            cc["ttl"] = "1h"
        return cc

    def _apply_indices_markers(
        self,
        api_messages: list[dict[str, Any]],
        indices: list[int],
        cache_control: dict[str, str],
    ) -> None:
        """Apply cache_control to the last content block of each indexed message."""
        for idx in indices:
            if 0 <= idx < len(api_messages):
                msg = api_messages[idx]
                content = msg.get("content")
                if isinstance(content, list) and content:
                    last_block = content[-1]
                    if isinstance(last_block, dict):
                        content[-1] = {**last_block, "cache_control": cache_control}
                elif isinstance(content, str):
                    msg["content"] = [
                        {"type": "text", "text": content, "cache_control": cache_control}
                    ]

    @staticmethod
    def _apply_message_cache_control(
        api_messages: list[dict[str, Any]],
        cache_control: dict[str, str],
    ) -> None:
        """Apply cache_control to the last content block of the last message.

        This caches the conversation prefix so subsequent turns get cache hits
        on all prior messages.
        """
        if not api_messages:
            return

        last_msg = api_messages[-1]
        content = last_msg.get("content")
        if not content:
            return

        if isinstance(content, list) and len(content) > 0:
            last_block = content[-1]
            if isinstance(last_block, dict):
                last_block["cache_control"] = cache_control
        elif isinstance(content, str):
            # Convert bare string content to a block so we can attach cache_control
            last_msg["content"] = [
                {"type": "text", "text": content, "cache_control": cache_control}
            ]

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

    def _handle_event(
        self, event: Any, partial: AssistantMessage, ms: MessageStream
    ) -> None:
        etype = getattr(event, "type", "")
        if etype == "content_block_start":
            idx = getattr(event, "index", len(partial.content))
            block = event.content_block
            if block.type == "text":
                partial.content.append(TextContent(text=""))
                ms.push(
                    StreamEvent(
                        type="text_start",
                        content_index=idx,
                        partial=partial.model_copy(deep=True),
                    )
                )
            elif block.type == "thinking":
                partial.content.append(ThinkingContent(thinking=""))
                ms.push(
                    StreamEvent(
                        type="thinking_start",
                        content_index=idx,
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
                        content_index=idx,
                        partial=partial.model_copy(deep=True),
                    )
                )
        elif etype == "content_block_delta":
            idx = getattr(event, "index", len(partial.content) - 1)
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
                        content_index=idx,
                        partial=partial.model_copy(deep=True),
                    )
                )
            elif hasattr(delta, "thinking"):
                if partial.content and isinstance(partial.content[-1], ThinkingContent):
                    partial.content[-1] = ThinkingContent(
                        thinking=partial.content[-1].thinking + delta.thinking
                    )
                ms.push(
                    StreamEvent(
                        type="thinking_delta",
                        delta=delta.thinking,
                        content_index=idx,
                        partial=partial.model_copy(deep=True),
                    )
                )
            elif hasattr(delta, "partial_json"):
                ms.push(
                    StreamEvent(
                        type="toolcall_delta",
                        delta=delta.partial_json,
                        content_index=idx,
                        partial=partial.model_copy(deep=True),
                    )
                )
        elif etype == "content_block_stop":
            idx = getattr(event, "index", len(partial.content) - 1)
            if partial.content:
                last = partial.content[-1]
                if isinstance(last, TextContent):
                    ms.push(
                        StreamEvent(
                            type="text_end",
                            content_index=idx,
                            partial=partial.model_copy(deep=True),
                        )
                    )
                elif isinstance(last, ThinkingContent):
                    ms.push(
                        StreamEvent(
                            type="thinking_end",
                            content_index=idx,
                            partial=partial.model_copy(deep=True),
                        )
                    )
                elif isinstance(last, ToolCall):
                    ms.push(
                        StreamEvent(
                            type="toolcall_end",
                            content_index=idx,
                            partial=partial.model_copy(deep=True),
                        )
                    )

    @staticmethod
    def _convert_response(response: Any, model: Model) -> AssistantMessage:
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
                cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0)
                or 0,
                cache_write_tokens=getattr(
                    response.usage, "cache_creation_input_tokens", 0
                )
                or 0,
            ),
            timestamp=time.time(),
            provider_id=model.provider,
            model_id=model.id,
            response_id=getattr(response, "id", None),
        )
