from __future__ import annotations

import asyncio
import time
from typing import Any, Literal, Protocol, runtime_checkable

from cubepi.providers.base import (
    AssistantMessage,
    BaseProvider,
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
    _fire_request_listeners,
    _fire_response_listeners,
    invoke_on_payload,
    invoke_on_response,
)
from cubepi.providers.capability import (
    CapabilityDescriptor,
    ReasoningLevelSpec,
    TemperatureSpec,
    apply_temperature,
    merge_capability_payload,
    write_reasoning_level,
)
from cubepi.providers.models import clamp_thinking_level

CacheRetention = Literal["short", "long", "none"]


# Default capability for Anthropic. Unlike OpenAI's empty default, Anthropic
# always goes through the capability path — capability=None reproduces today's
# wire bytes exactly because:
#  - reasoning_off_payload is empty: legacy omits the "thinking" key entirely
#    when thinking=off, so the default must too (no {"type": "disabled"}).
#  - reasoning_on_payload writes {"thinking": {"type": "enabled"}}, then
#    reasoning_level adds budget_tokens at thinking.budget_tokens.
#  - level_budgets mirrors cubepi.providers.base.ThinkingBudgets defaults
#    (minimal=1024, low=2048, medium=8192, high=16384). xhigh clamps to high
#    to match adjust_max_tokens_for_thinking's "xhigh -> high" mapping.
#    "off" is included for completeness but reasoning_level is never written
#    on the off-branch.
#  - temperature mode="free" with min/max [0, 1] matches Anthropic's accepted
#    range; the stream() method itself decides whether to send temperature
#    (only when thinking is off — Anthropic rejects temperature with thinking
#    enabled).
_ANTHROPIC_DEFAULT_CAPABILITY = CapabilityDescriptor(
    reasoning_off_payload={},
    reasoning_on_payload={"thinking": {"type": "enabled"}},
    reasoning_level=ReasoningLevelSpec(
        path="thinking.budget_tokens",
        kind="int_budget",
        level_budgets={
            "off": 0,
            "minimal": 1024,
            "low": 2048,
            "medium": 8192,
            "high": 16384,
            "xhigh": 16384,
        },
    ),
    temperature=TemperatureSpec(mode="free", min=0.0, max=1.0, default=1.0),
)


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


class AnthropicProvider(BaseProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_retention: CacheRetention = "short",
        cache_policy: CacheMarkerPolicy | None = None,
        capability: CapabilityDescriptor | None = None,
        model_capability_overrides: dict[str, CapabilityDescriptor] | None = None,
    ) -> None:
        super().__init__()
        import anthropic

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)
        self._cache_retention = cache_retention
        self._cache_policy: CacheMarkerPolicy = (
            cache_policy or DefaultCacheMarkerPolicy()
        )
        # Anthropic always runs the capability path; capability=None falls back
        # to _ANTHROPIC_DEFAULT_CAPABILITY which mirrors legacy wire bytes.
        self._capability: CapabilityDescriptor = (
            capability if capability is not None else _ANTHROPIC_DEFAULT_CAPABILITY
        )
        self._model_overrides: dict[str, CapabilityDescriptor] = (
            model_capability_overrides or {}
        )

    def _resolve_capability(self, model_id: str) -> CapabilityDescriptor:
        return self._model_overrides.get(model_id, self._capability)

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
        cap = self._resolve_capability(model.id)

        cache_control = self._get_cache_control()
        api_messages = [self._convert_message(m) for m in messages]
        if cache_control:
            indices = self._cache_policy.message_breakpoint_indices(messages)
            self._apply_indices_markers(api_messages, indices, cache_control)

        kwargs: dict[str, Any] = {
            "model": model.id,
            "messages": api_messages,
        }
        if system_prompt:
            if cache_control and self._cache_policy.mark_system():
                kwargs["system"] = [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": cache_control,
                    }
                ]
            else:
                kwargs["system"] = system_prompt
        if tools:
            api_tools = [self._convert_tool(t) for t in tools]
            if cache_control and api_tools and self._cache_policy.mark_last_tool():
                api_tools[-1]["cache_control"] = cache_control
            kwargs["tools"] = api_tools

        # Capability-driven thinking + temperature. The default capability
        # (capability=None) reproduces today's wire bytes:
        #  - thinking off: reasoning_off_payload is empty, so no "thinking"
        #    key is written; temperature is sent (legacy behavior).
        #  - thinking on: reasoning_on_payload writes thinking.type=enabled,
        #    reasoning_level writes thinking.budget_tokens; temperature is
        #    stripped because Anthropic rejects custom temperature with
        #    extended thinking enabled. See
        #    https://platform.claude.com/docs/en/build-with-claude/extended-thinking#feature-compatibility
        #
        # max_tokens is computed AFTER the capability writes the budget so it
        # always accommodates whatever budget actually landed on the wire.
        # Anthropic rejects requests where budget_tokens >= max_tokens.
        if thinking == "off":
            merge_capability_payload(kwargs, cap.reasoning_off_payload)
            kwargs.setdefault("temperature", model.temperature)
            apply_temperature(kwargs, cap.temperature)
            kwargs["max_tokens"] = min(model.max_tokens, model.context_window)
        else:
            merge_capability_payload(kwargs, cap.reasoning_on_payload)
            if cap.reasoning_level is not None:
                write_reasoning_level(kwargs, cap.reasoning_level, thinking)
            kwargs.pop("temperature", None)
            budget = 0
            thinking_block = kwargs.get("thinking")
            if isinstance(thinking_block, dict):
                budget = thinking_block.get("budget_tokens", 0) or 0
            kwargs["max_tokens"] = min(model.max_tokens + budget, model.context_window)

            # If context_window clipped max_tokens such that the budget no
            # longer fits, reduce the budget in place. Anthropic rejects when
            # budget_tokens >= max_tokens; reserve at least 1024 output tokens
            # to mirror adjust_max_tokens_for_thinking's legacy policy.
            min_output_tokens = 1024
            if budget > 0 and kwargs["max_tokens"] - budget < min_output_tokens:
                new_budget = max(0, kwargs["max_tokens"] - min_output_tokens)
                if isinstance(thinking_block, dict):
                    if new_budget > 0:
                        thinking_block["budget_tokens"] = new_budget
                    else:
                        # Budget reduced to 0 — disable thinking entirely.
                        kwargs["thinking"] = {"type": "disabled"}

        async def _produce() -> None:
            body: dict | None = None
            exc: BaseException | None = None
            try:
                nonlocal kwargs
                kwargs = await invoke_on_payload(opts.on_payload, kwargs, model)
                await _fire_request_listeners(self._request_listeners, kwargs, model)

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
                    await self._emit(
                        ms,
                        StreamEvent(
                            type="start", partial=partial.model_copy(deep=True)
                        ),
                        model,
                    )

                    async for event in stream:
                        if opts.signal and opts.signal.is_set():
                            aborted = partial.model_copy(
                                update={
                                    "stop_reason": "aborted",
                                    "error_message": "Request was aborted",
                                }
                            )
                            await self._emit(
                                ms,
                                StreamEvent(
                                    type="error",
                                    error_message="Request was aborted",
                                ),
                                model,
                            )
                            ms.set_result(aborted)
                            return

                        await self._handle_event(event, partial, ms, model)

                    final_msg = await stream.get_final_message()
                    result = self._convert_response(final_msg, model)
                    body = self._assemble_response(final_msg)
                    await self._emit(ms, StreamEvent(type="done"), model)
                    ms.set_result(result)

            except BaseException as e:
                exc = e
                err_text = self._error_message(e, model)
                error_msg = AssistantMessage(
                    content=[],
                    stop_reason="error",
                    error_message=err_text,
                    usage=Usage(),
                    timestamp=time.time(),
                    provider_id=model.provider,
                    model_id=model.id,
                )
                await self._emit(
                    ms, StreamEvent(type="error", error_message=err_text), model
                )
                ms.set_result(error_msg)
                if not isinstance(e, Exception):
                    raise
            finally:
                await _fire_response_listeners(
                    self._response_listeners, body, model, exc
                )

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
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": cache_control,
                        }
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

    async def _handle_event(
        self,
        event: Any,
        partial: AssistantMessage,
        ms: MessageStream,
        model: Model,
    ) -> None:
        etype = getattr(event, "type", "")
        if etype == "content_block_start":
            idx = getattr(event, "index", len(partial.content))
            block = event.content_block
            if block.type == "text":
                partial.content.append(TextContent(text=""))
                await self._emit(
                    ms,
                    StreamEvent(
                        type="text_start",
                        content_index=idx,
                        partial=partial.model_copy(deep=True),
                    ),
                    model,
                )
            elif block.type == "thinking":
                partial.content.append(ThinkingContent(thinking=""))
                await self._emit(
                    ms,
                    StreamEvent(
                        type="thinking_start",
                        content_index=idx,
                        partial=partial.model_copy(deep=True),
                    ),
                    model,
                )
            elif block.type == "tool_use":
                partial.content.append(
                    ToolCall(id=block.id, name=block.name, arguments={})
                )
                await self._emit(
                    ms,
                    StreamEvent(
                        type="toolcall_start",
                        content_index=idx,
                        partial=partial.model_copy(deep=True),
                    ),
                    model,
                )
        elif etype == "content_block_delta":
            idx = getattr(event, "index", len(partial.content) - 1)
            delta = event.delta
            if hasattr(delta, "text"):
                if partial.content and isinstance(partial.content[-1], TextContent):
                    partial.content[-1] = TextContent(
                        text=partial.content[-1].text + delta.text
                    )
                await self._emit(
                    ms,
                    StreamEvent(
                        type="text_delta",
                        delta=delta.text,
                        content_index=idx,
                        partial=partial.model_copy(deep=True),
                    ),
                    model,
                )
            elif hasattr(delta, "thinking"):
                if partial.content and isinstance(partial.content[-1], ThinkingContent):
                    partial.content[-1] = ThinkingContent(
                        thinking=partial.content[-1].thinking + delta.thinking
                    )
                await self._emit(
                    ms,
                    StreamEvent(
                        type="thinking_delta",
                        delta=delta.thinking,
                        content_index=idx,
                        partial=partial.model_copy(deep=True),
                    ),
                    model,
                )
            elif hasattr(delta, "partial_json"):
                await self._emit(
                    ms,
                    StreamEvent(
                        type="toolcall_delta",
                        delta=delta.partial_json,
                        content_index=idx,
                        partial=partial.model_copy(deep=True),
                    ),
                    model,
                )
        elif etype == "content_block_stop":
            idx = getattr(event, "index", len(partial.content) - 1)
            if partial.content:
                last = partial.content[-1]
                if isinstance(last, TextContent):
                    await self._emit(
                        ms,
                        StreamEvent(
                            type="text_end",
                            content_index=idx,
                            partial=partial.model_copy(deep=True),
                        ),
                        model,
                    )
                elif isinstance(last, ThinkingContent):
                    await self._emit(
                        ms,
                        StreamEvent(
                            type="thinking_end",
                            content_index=idx,
                            partial=partial.model_copy(deep=True),
                        ),
                        model,
                    )
                elif isinstance(last, ToolCall):
                    await self._emit(
                        ms,
                        StreamEvent(
                            type="toolcall_end",
                            content_index=idx,
                            partial=partial.model_copy(deep=True),
                        ),
                        model,
                    )

    @staticmethod
    def _assemble_response(final_msg: Any) -> dict[str, Any]:
        """Assemble Anthropic's non-streaming ``messages.create()`` response
        shape from the final streaming Message object.

        Mirrors the REST API response keys. Used to populate the assembled
        ``body`` argument passed to ``subscribe_response`` listeners.
        """
        content: list[dict[str, Any]] = []
        for block in final_msg.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                content.append({"type": "text", "text": block.text})
            elif btype == "thinking":
                content.append({"type": "thinking", "thinking": block.thinking})
            elif btype == "tool_use":
                content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

        usage = getattr(final_msg, "usage", None)
        usage_dict: dict[str, Any] = {}
        if usage is not None:
            usage_dict = {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0)
                or 0,
                "cache_creation_input_tokens": getattr(
                    usage, "cache_creation_input_tokens", 0
                )
                or 0,
            }

        body: dict[str, Any] = {
            "id": getattr(final_msg, "id", None),
            "type": "message",
            "role": "assistant",
            "model": getattr(final_msg, "model", None),
            "content": content,
            "stop_reason": getattr(final_msg, "stop_reason", None),
            "stop_sequence": getattr(final_msg, "stop_sequence", None),
            "usage": usage_dict,
        }
        return body

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
