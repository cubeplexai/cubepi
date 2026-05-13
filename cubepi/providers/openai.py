from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Literal

from cubepi.utils.json_parse import parse_streaming_json

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
    invoke_on_payload,
    invoke_on_response,
)


class OpenAIProvider:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        payload_quirks: list[Literal["max_completion_tokens_alias"]] | None = None,
    ) -> None:
        import openai

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**kwargs)
        self._payload_quirks: set[str] = set(payload_quirks or [])

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

        api_messages: list[dict[str, Any]] = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        api_messages.extend(self._convert_message(m) for m in messages)

        kwargs: dict[str, Any] = {
            "model": model.id,
            "messages": api_messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = [self._convert_tool(t) for t in tools]

        async def _produce() -> None:
            try:
                nonlocal kwargs
                kwargs = await invoke_on_payload(opts.on_payload, kwargs, model)

                if "max_completion_tokens_alias" in self._payload_quirks:
                    if "max_completion_tokens" in kwargs:
                        kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")

                response = await self._client.chat.completions.create(**kwargs)

                # Invoke on_response with HTTP metadata if available
                http_response = getattr(response, "response", None)
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

                current_text = ""
                tool_calls_in_progress: dict[int, dict[str, Any]] = {}
                text_started = False
                text_content_index = 0
                response_id: str | None = None
                thinking_started = False
                thinking_content_index: int | None = None

                async for chunk in response:
                    if response_id is None and getattr(chunk, "id", None):
                        response_id = chunk.id
                        partial.response_id = response_id
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

                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue

                    # OSS reasoning extraction: priority order is
                    #   1. delta.reasoning_content (DeepSeek/Qwen/DouBao)
                    #   2. delta.reasoning         (vLLM)
                    #   3. delta.reasoning_details (MiniMax)
                    reasoning_delta: str | None = None
                    rc = getattr(delta, "reasoning_content", None)
                    if rc:
                        reasoning_delta = rc
                    else:
                        r = getattr(delta, "reasoning", None)
                        if r:
                            reasoning_delta = r
                        else:
                            rds = getattr(delta, "reasoning_details", None)
                            if rds:
                                parts: list[str] = []
                                for d in rds:
                                    if hasattr(d, "text"):
                                        text = d.text
                                    elif isinstance(d, dict):
                                        text = d.get("text")
                                    else:
                                        text = None
                                    if text:
                                        parts.append(text)
                                if parts:
                                    reasoning_delta = "".join(parts)

                    if reasoning_delta:
                        if not thinking_started:
                            partial.content.append(ThinkingContent(thinking=""))
                            thinking_content_index = len(partial.content) - 1
                            ms.push(
                                StreamEvent(
                                    type="thinking_start",
                                    content_index=thinking_content_index,
                                    partial=partial.model_copy(deep=True),
                                )
                            )
                            thinking_started = True
                        existing = partial.content[thinking_content_index].thinking  # type: ignore[union-attr]
                        partial.content[thinking_content_index] = ThinkingContent(
                            thinking=existing + reasoning_delta
                        )
                        ms.push(
                            StreamEvent(
                                type="thinking_delta",
                                delta=reasoning_delta,
                                content_index=thinking_content_index,
                                partial=partial.model_copy(deep=True),
                            )
                        )

                    if delta.content:
                        if not text_started:
                            partial.content.append(TextContent(text=""))
                            text_content_index = len(partial.content) - 1
                            ms.push(
                                StreamEvent(
                                    type="text_start",
                                    content_index=text_content_index,
                                    partial=partial.model_copy(deep=True),
                                )
                            )
                            text_started = True
                        current_text += delta.content
                        if partial.content and isinstance(
                            partial.content[-1], TextContent
                        ):
                            partial.content[-1] = TextContent(text=current_text)
                        ms.push(
                            StreamEvent(
                                type="text_delta",
                                delta=delta.content,
                                content_index=text_content_index,
                                partial=partial.model_copy(deep=True),
                            )
                        )

                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_in_progress:
                                if text_started:
                                    ms.push(
                                        StreamEvent(
                                            type="text_end",
                                            content_index=text_content_index,
                                            partial=partial.model_copy(deep=True),
                                        )
                                    )
                                    text_started = False
                                tool_calls_in_progress[idx] = {
                                    "id": tc_delta.id or "",
                                    "name": (
                                        tc_delta.function.name
                                        if tc_delta.function
                                        else ""
                                    ),
                                    "arguments": "",
                                }
                                partial.content.append(
                                    ToolCall(
                                        id=tool_calls_in_progress[idx]["id"],
                                        name=tool_calls_in_progress[idx]["name"],
                                        arguments={},
                                    )
                                )
                                tc_content_index = len(partial.content) - 1
                                tool_calls_in_progress[idx]["content_index"] = (
                                    tc_content_index
                                )
                                ms.push(
                                    StreamEvent(
                                        type="toolcall_start",
                                        content_index=tc_content_index,
                                        partial=partial.model_copy(deep=True),
                                    )
                                )
                            if tc_delta.function and tc_delta.function.arguments:
                                tool_calls_in_progress[idx]["arguments"] += (
                                    tc_delta.function.arguments
                                )
                                ms.push(
                                    StreamEvent(
                                        type="toolcall_delta",
                                        delta=tc_delta.function.arguments,
                                        content_index=tool_calls_in_progress[idx][
                                            "content_index"
                                        ],
                                        partial=partial.model_copy(deep=True),
                                    )
                                )

                    finish_reason = (
                        chunk.choices[0].finish_reason if chunk.choices else None
                    )
                    if finish_reason:
                        if thinking_started:
                            ms.push(
                                StreamEvent(
                                    type="thinking_end",
                                    content_index=thinking_content_index,
                                    partial=partial.model_copy(deep=True),
                                )
                            )
                            thinking_started = False

                        if text_started:
                            ms.push(
                                StreamEvent(
                                    type="text_end",
                                    content_index=text_content_index,
                                    partial=partial.model_copy(deep=True),
                                )
                            )

                        for idx, tc_data in tool_calls_in_progress.items():
                            args = parse_streaming_json(tc_data["arguments"])
                            for i, c in enumerate(partial.content):
                                if isinstance(c, ToolCall) and c.id == tc_data["id"]:
                                    partial.content[i] = ToolCall(
                                        id=tc_data["id"],
                                        name=tc_data["name"],
                                        arguments=args,
                                    )
                            ms.push(
                                StreamEvent(
                                    type="toolcall_end",
                                    content_index=tc_data["content_index"],
                                    partial=partial.model_copy(deep=True),
                                )
                            )

                        stop_map = {
                            "stop": "stop",
                            "tool_calls": "tool_use",
                            "length": "length",
                        }
                        final = partial.model_copy(
                            update={
                                "stop_reason": stop_map.get(
                                    finish_reason, finish_reason
                                ),
                            }
                        )
                        ms.push(StreamEvent(type="done"))
                        ms.set_result(final)
                        return

                ms.push(StreamEvent(type="done"))
                ms.set_result(partial)

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

    @staticmethod
    def _convert_message(msg: Message) -> dict[str, Any]:
        if isinstance(msg, UserMessage):
            has_image = any(isinstance(c, ImageContent) for c in msg.content)
            if has_image:
                parts: list[dict[str, Any]] = []
                for c in msg.content:
                    if isinstance(c, TextContent):
                        parts.append({"type": "text", "text": c.text})
                    elif isinstance(c, ImageContent):
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{c.media_type};base64,{c.source}"
                                },
                            }
                        )
                return {"role": "user", "content": parts}
            text_parts = [c.text for c in msg.content if isinstance(c, TextContent)]
            return {"role": "user", "content": "\n".join(text_parts)}

        elif isinstance(msg, AssistantMessage):
            text_parts = [c.text for c in msg.content if isinstance(c, TextContent)]
            tool_calls = [c for c in msg.content if isinstance(c, ToolCall)]

            result: dict[str, Any] = {"role": "assistant"}
            result["content"] = "\n".join(text_parts) if text_parts else None
            if tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ]
            return result

        elif isinstance(msg, ToolResultMessage):
            text_parts = [c.text for c in msg.content if isinstance(c, TextContent)]
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": "\n".join(text_parts),
            }

        return {"role": "user", "content": ""}

    @staticmethod
    def _convert_tool(td: ToolDefinition) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": td.name,
                "description": td.description,
                "parameters": td.parameters,
            },
        }
