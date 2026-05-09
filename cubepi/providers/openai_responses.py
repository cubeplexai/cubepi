from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from cubepi.providers.base import (
    AssistantMessage,
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

# Map cubepi ThinkingLevel to OpenAI reasoning.effort values.
# "off" means no reasoning parameter is sent.
# "minimal" is not a valid OpenAI effort level, so we map it to "low".
_THINKING_TO_EFFORT: dict[ThinkingLevel, str | None] = {
    "off": None,
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}


class OpenAIResponsesProvider:
    """Provider that uses the OpenAI Responses API.

    The Responses API supports reasoning models (o-series) with streaming,
    tool use, and reasoning effort control.
    """

    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None
    ) -> None:
        import openai

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**kwargs)

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

        api_input = self._build_input(messages)

        kwargs: dict[str, Any] = {
            "model": model.id,
            "input": api_input,
            "stream": True,
            "store": False,
        }

        if system_prompt:
            role = "developer" if model.reasoning else "system"
            kwargs["instructions"] = system_prompt
            # The instructions param uses system/developer role implicitly.
            # For explicit role control, prepend to input instead.
            kwargs["input"] = [{"role": role, "content": system_prompt}] + api_input

            # Remove instructions since we use input-based system prompt
            del kwargs["instructions"]

        if tools:
            kwargs["tools"] = [self._convert_tool(t) for t in tools]

        # Configure reasoning effort for reasoning models
        effort = _THINKING_TO_EFFORT.get(thinking)
        if model.reasoning and effort is not None:
            kwargs["reasoning"] = {
                "effort": effort,
                "summary": "auto",
            }
            kwargs["include"] = ["reasoning.encrypted_content"]

        if model.max_tokens:
            kwargs["max_output_tokens"] = model.max_tokens

        async def _produce() -> None:
            try:
                response = await self._client.responses.create(**kwargs)
                partial = AssistantMessage(
                    content=[], usage=Usage(), timestamp=time.time()
                )
                ms.push(
                    StreamEvent(type="start", partial=partial.model_copy(deep=True))
                )

                # Track current streaming state
                current_thinking = ""
                current_text = ""
                current_item_type: str | None = None
                # Per-item tool call state keyed by item.id
                tool_state: dict[str, dict[str, str]] = {}
                active_tool_item_id: str | None = None

                async for event in response:
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

                    etype = event.type

                    # --- Output item lifecycle ---
                    if etype == "response.output_item.added":
                        item = event.item
                        if item.type == "reasoning":
                            current_item_type = "reasoning"
                            current_thinking = ""
                            partial.content.append(ThinkingContent(thinking=""))
                            ms.push(
                                StreamEvent(
                                    type="thinking_start",
                                    partial=partial.model_copy(deep=True),
                                )
                            )
                        elif item.type == "message":
                            current_item_type = "message"
                            current_text = ""
                            partial.content.append(TextContent(text=""))
                            ms.push(
                                StreamEvent(
                                    type="text_start",
                                    partial=partial.model_copy(deep=True),
                                )
                            )
                        elif item.type == "function_call":
                            current_item_type = "function_call"
                            item_id = item.id or ""
                            tc_id = (
                                f"{item.call_id}|{item_id}"
                                if item_id
                                else item.call_id
                            )
                            tool_state[item_id or item.call_id] = {
                                "call_id": item.call_id,
                                "item_id": item_id,
                                "name": item.name,
                                "json": "",
                                "tc_id": tc_id,
                            }
                            active_tool_item_id = item_id or item.call_id
                            partial.content.append(
                                ToolCall(
                                    id=tc_id,
                                    name=item.name,
                                    arguments={},
                                )
                            )
                            ms.push(
                                StreamEvent(
                                    type="toolcall_start",
                                    partial=partial.model_copy(deep=True),
                                )
                            )

                    # --- Reasoning (thinking) deltas ---
                    elif etype == "response.reasoning_summary_text.delta":
                        if current_item_type == "reasoning":
                            current_thinking += event.delta
                            if partial.content and isinstance(
                                partial.content[-1], ThinkingContent
                            ):
                                partial.content[-1] = ThinkingContent(
                                    thinking=current_thinking
                                )
                            ms.push(
                                StreamEvent(
                                    type="thinking_delta",
                                    delta=event.delta,
                                    partial=partial.model_copy(deep=True),
                                )
                            )

                    elif etype == "response.reasoning_text.delta":
                        if current_item_type == "reasoning":
                            current_thinking += event.delta
                            if partial.content and isinstance(
                                partial.content[-1], ThinkingContent
                            ):
                                partial.content[-1] = ThinkingContent(
                                    thinking=current_thinking
                                )
                            ms.push(
                                StreamEvent(
                                    type="thinking_delta",
                                    delta=event.delta,
                                    partial=partial.model_copy(deep=True),
                                )
                            )

                    elif etype == "response.reasoning_summary_part.done":
                        # Append separator between summary parts
                        if current_item_type == "reasoning":
                            current_thinking += "\n\n"
                            if partial.content and isinstance(
                                partial.content[-1], ThinkingContent
                            ):
                                partial.content[-1] = ThinkingContent(
                                    thinking=current_thinking
                                )
                            ms.push(
                                StreamEvent(
                                    type="thinking_delta",
                                    delta="\n\n",
                                    partial=partial.model_copy(deep=True),
                                )
                            )

                    # --- Text deltas ---
                    elif etype == "response.output_text.delta":
                        if current_item_type == "message":
                            current_text += event.delta
                            if partial.content and isinstance(
                                partial.content[-1], TextContent
                            ):
                                partial.content[-1] = TextContent(text=current_text)
                            ms.push(
                                StreamEvent(
                                    type="text_delta",
                                    delta=event.delta,
                                    partial=partial.model_copy(deep=True),
                                )
                            )

                    elif etype == "response.refusal.delta":
                        if current_item_type == "message":
                            current_text += event.delta
                            if partial.content and isinstance(
                                partial.content[-1], TextContent
                            ):
                                partial.content[-1] = TextContent(text=current_text)
                            ms.push(
                                StreamEvent(
                                    type="text_delta",
                                    delta=event.delta,
                                    partial=partial.model_copy(deep=True),
                                )
                            )

                    # --- Tool call argument deltas ---
                    elif etype == "response.function_call_arguments.delta":
                        if active_tool_item_id and active_tool_item_id in tool_state:
                            tool_state[active_tool_item_id]["json"] += event.delta
                            ms.push(
                                StreamEvent(
                                    type="toolcall_delta",
                                    delta=event.delta,
                                    partial=partial.model_copy(deep=True),
                                )
                            )

                    elif etype == "response.function_call_arguments.done":
                        if active_tool_item_id and active_tool_item_id in tool_state:
                            tool_state[active_tool_item_id]["json"] = event.arguments

                    # --- Output item done ---
                    elif etype == "response.output_item.done":
                        item = event.item
                        if item.type == "reasoning":
                            # Finalize thinking content from item summary
                            summary_text = ""
                            if hasattr(item, "summary") and item.summary:
                                summary_text = "\n\n".join(
                                    s.text for s in item.summary if hasattr(s, "text")
                                )
                            final_thinking = summary_text or current_thinking
                            if partial.content and isinstance(
                                partial.content[-1], ThinkingContent
                            ):
                                partial.content[-1] = ThinkingContent(
                                    thinking=final_thinking
                                )
                            ms.push(
                                StreamEvent(
                                    type="thinking_end",
                                    partial=partial.model_copy(deep=True),
                                )
                            )
                            current_item_type = None

                        elif item.type == "message":
                            # Finalize text from item content
                            final_text = ""
                            if hasattr(item, "content") and item.content:
                                parts = []
                                for c in item.content:
                                    if hasattr(c, "text"):
                                        parts.append(c.text)
                                    elif hasattr(c, "refusal"):
                                        parts.append(c.refusal)
                                final_text = "".join(parts)
                            if (
                                final_text
                                and partial.content
                                and isinstance(partial.content[-1], TextContent)
                            ):
                                partial.content[-1] = TextContent(text=final_text)
                            ms.push(
                                StreamEvent(
                                    type="text_end",
                                    partial=partial.model_copy(deep=True),
                                )
                            )
                            current_item_type = None

                        elif item.type == "function_call":
                            item_key = (item.id or "") or item.call_id
                            ts = tool_state.pop(item_key, None)
                            final_json = (
                                getattr(item, "arguments", None)
                                or (ts["json"] if ts else "")
                            )
                            try:
                                args = json.loads(final_json) if final_json else {}
                            except json.JSONDecodeError:
                                args = {}
                            item_id = item.id or ""
                            tc_id = (
                                f"{item.call_id}|{item_id}"
                                if item_id
                                else item.call_id
                            )
                            for i, c in enumerate(partial.content):
                                if isinstance(c, ToolCall) and c.id == tc_id:
                                    partial.content[i] = ToolCall(
                                        id=tc_id,
                                        name=item.name,
                                        arguments=args,
                                    )
                            ms.push(
                                StreamEvent(
                                    type="toolcall_end",
                                    partial=partial.model_copy(deep=True),
                                )
                            )
                            current_item_type = None

                    # --- Response completed ---
                    elif etype == "response.completed":
                        resp = event.response
                        usage = Usage()
                        if resp and resp.usage:
                            cached = 0
                            if hasattr(resp.usage, "input_tokens_details"):
                                details = resp.usage.input_tokens_details
                                if details and hasattr(details, "cached_tokens"):
                                    cached = details.cached_tokens or 0
                            usage = Usage(
                                input_tokens=(resp.usage.input_tokens or 0) - cached,
                                output_tokens=resp.usage.output_tokens or 0,
                                cache_read_tokens=cached,
                            )

                        stop_reason = self._map_stop_reason(
                            resp.status if resp else None, partial
                        )
                        partial = partial.model_copy(
                            update={
                                "usage": usage,
                                "stop_reason": stop_reason,
                            }
                        )
                        ms.push(StreamEvent(type="done"))
                        ms.set_result(partial)
                        return

                    # --- Errors ---
                    elif etype == "error":
                        error_msg_text = ""
                        if hasattr(event, "message"):
                            error_msg_text = event.message
                        elif hasattr(event, "code"):
                            error_msg_text = f"Error code {event.code}"
                        raise RuntimeError(error_msg_text or "Unknown error")

                    elif etype == "response.failed":
                        resp = event.response
                        error_detail = ""
                        if resp and hasattr(resp, "error") and resp.error:
                            code = getattr(resp.error, "code", "unknown")
                            msg = getattr(resp.error, "message", "no message")
                            error_detail = f"{code}: {msg}"
                        elif (
                            resp
                            and hasattr(resp, "incomplete_details")
                            and resp.incomplete_details
                        ):
                            reason = getattr(
                                resp.incomplete_details, "reason", "unknown"
                            )
                            error_detail = f"incomplete: {reason}"
                        raise RuntimeError(error_detail or "Unknown error (no details)")

                # If we get here without response.completed, finalize
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

        asyncio.create_task(_produce())
        return ms

    @staticmethod
    def _map_stop_reason(status: str | None, partial: AssistantMessage) -> str:
        """Map OpenAI response status to cubepi stop reason."""
        has_tool_calls = any(isinstance(c, ToolCall) for c in partial.content)
        reason_map = {
            "completed": "stop",
            "incomplete": "length",
            "failed": "error",
            "cancelled": "error",
            "in_progress": "stop",
            "queued": "stop",
        }
        reason = reason_map.get(status or "completed", "stop")
        if has_tool_calls and reason == "stop":
            reason = "tool_use"
        return reason

    @staticmethod
    def _build_input(messages: list[Message]) -> list[dict[str, Any]]:
        """Convert cubepi messages to OpenAI Responses API input format."""
        api_input: list[dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, UserMessage):
                content: list[dict[str, Any]] = []
                for c in msg.content:
                    if isinstance(c, TextContent):
                        content.append({"type": "input_text", "text": c.text})
                if content:
                    api_input.append({"role": "user", "content": content})

            elif isinstance(msg, AssistantMessage):
                text_blocks: list[dict[str, Any]] = []
                for c in msg.content:
                    if isinstance(c, ThinkingContent):
                        pass
                    elif isinstance(c, TextContent):
                        text_blocks.append(
                            {"type": "output_text", "text": c.text, "annotations": []}
                        )
                    elif isinstance(c, ToolCall):
                        if text_blocks:
                            api_input.append(
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": text_blocks,
                                    "status": "completed",
                                }
                            )
                            text_blocks = []
                        parts = c.id.split("|", 1)
                        call_id = parts[0]
                        item_id = parts[1] if len(parts) > 1 else None
                        fc: dict[str, Any] = {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": c.name,
                            "arguments": json.dumps(c.arguments),
                        }
                        if item_id:
                            fc["id"] = item_id
                        api_input.append(fc)
                if text_blocks:
                    api_input.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": text_blocks,
                            "status": "completed",
                        }
                    )

            elif isinstance(msg, ToolResultMessage):
                parts = msg.tool_call_id.split("|", 1)
                call_id = parts[0]
                text_parts = [c.text for c in msg.content if isinstance(c, TextContent)]
                api_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": "\n".join(text_parts),
                    }
                )

        return api_input

    @staticmethod
    def _convert_tool(td: ToolDefinition) -> dict[str, Any]:
        """Convert a cubepi ToolDefinition to OpenAI Responses API tool format."""
        return {
            "type": "function",
            "name": td.name,
            "description": td.description,
            "parameters": td.parameters,
        }
