from __future__ import annotations

import asyncio
import inspect
import json
import math
import time
from typing import Any, Awaitable, Callable

from cubepi.providers.base import (
    AssistantMessage,
    BaseProvider,
    Message,
    MessageStream,
    Model,
    StreamEvent,
    StreamOptions,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolDefinition,
    Usage,
    _fire_listeners,
    _fire_response_listeners,
    invoke_on_payload,
)

FauxContentBlock = TextContent | ThinkingContent | ToolCall

# New extended signature: (messages, model, system_prompt, tools)
# Old signature (messages, model) is still supported for backward compatibility
FauxResponseFactory = Callable[
    ...,
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


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text (approx 4 chars per token)."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _common_prefix_length(a: str, b: str) -> int:
    """Return the length of the common prefix between two strings."""
    length = min(len(a), len(b))
    index = 0
    while index < length and a[index] == b[index]:
        index += 1
    return index


def _serialize_prompt_context(
    system_prompt: str,
    tools: list[ToolDefinition] | None,
    messages: list[Message],
) -> str:
    """Serialize prompt context for cache comparison (prefix-based)."""
    parts: list[str] = []
    if system_prompt:
        parts.append(f"system:{system_prompt}")
    if tools:
        parts.append(
            f"tools:{json.dumps([t.model_dump() for t in tools], sort_keys=True)}"
        )
    for msg in messages:
        parts.append(f"{msg.role}:{msg.model_dump_json()}")
    return "\n\n".join(parts)


def _can_accept_extended_args(factory: FauxResponseFactory) -> bool:
    """Check if a factory can accept the extended (messages, model, system_prompt, tools) signature."""
    try:
        sig = inspect.signature(factory)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        # Check for VAR_POSITIONAL (*args)
        has_var_positional = any(
            p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()
        )
        if has_var_positional:
            return True
        return len(params) >= 4
    except (ValueError, TypeError):
        return False


class FauxProvider(BaseProvider):
    def __init__(
        self,
        *,
        tokens_per_second: float | None = None,
        token_size_min: int = 3,
        token_size_max: int = 5,
    ) -> None:
        super().__init__()
        self._responses: list[FauxResponseStep] = []
        self._tokens_per_second = tokens_per_second
        self._min = max(1, min(token_size_min, token_size_max))
        self._max = max(self._min, token_size_max)
        self.call_count = 0
        self._prompt_cache: dict[str, str] = {}
        # Monotonic counter for deterministic _assemble_response ids.
        self._response_seq = 0

    async def _emit(
        self, ms: MessageStream, event: StreamEvent, model: Model | None
    ) -> None:
        """Push an event to the stream and fire chunk listeners in order.

        ``model`` may be None when invoked from internal helpers
        (``_stream_with_deltas`` is exposed for direct calls in tests). In
        that case listener fan-out is skipped — the test isn't observing
        listeners anyway.
        """
        ms.push(event)
        if model is not None and self._chunk_listeners:
            await _fire_listeners(self._chunk_listeners, event, model)

    @staticmethod
    def _assemble_response(
        *, seq: int, model: Model, message: AssistantMessage
    ) -> dict[str, Any]:
        """Pinned, deterministic dict shape for FauxProvider responses.

        The schema is intentionally minimal and stable across runs (ids use
        the per-instance monotonic seq, no random or time-based fields) so
        tests for the listener registry can assert exact equality.
        """
        content_blocks: list[dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, TextContent):
                content_blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ThinkingContent):
                content_blocks.append({"type": "thinking", "thinking": block.thinking})
            elif isinstance(block, ToolCall):
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.arguments,
                    }
                )
        usage = message.usage or Usage()
        return {
            "id": f"faux-{seq}",
            "model": model.id,
            "role": "assistant",
            "content": content_blocks,
            "stop_reason": message.stop_reason,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_tokens": usage.cache_read_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
            },
        }

    def set_responses(self, responses: list[FauxResponseStep]) -> None:
        self._responses = list(responses)

    def append_responses(self, responses: list[FauxResponseStep]) -> None:
        self._responses.extend(responses)

    @property
    def pending_response_count(self) -> int:
        return len(self._responses)

    def clear_prompt_cache(self) -> None:
        """Clear the prompt cache, useful between test scenarios."""
        self._prompt_cache.clear()

    @property
    def prompt_cache(self) -> dict[str, str]:
        """Read-only access to the prompt cache for test assertions."""
        return dict(self._prompt_cache)

    def _compute_cache_usage(
        self,
        system_prompt: str,
        tools: list[ToolDefinition] | None,
        messages: list[Message],
        usage: Usage,
    ) -> Usage:
        """Compute cache-aware usage based on prompt prefix matching."""
        prompt_text = _serialize_prompt_context(system_prompt, tools, messages)
        prompt_tokens = _estimate_tokens(prompt_text)
        # Use "default" as session key (single-session simulation)
        session_key = "default"
        previous_prompt = self._prompt_cache.get(session_key)

        if previous_prompt is not None:
            cached_chars = _common_prefix_length(previous_prompt, prompt_text)
            cache_read = _estimate_tokens(previous_prompt[:cached_chars])
            cache_write = _estimate_tokens(prompt_text[cached_chars:])
            input_tokens = max(0, prompt_tokens - cache_read)
        else:
            cache_read = 0
            cache_write = prompt_tokens
            input_tokens = prompt_tokens

        self._prompt_cache[session_key] = prompt_text

        return Usage(
            input_tokens=input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
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
        opts = options or StreamOptions()
        ms = MessageStream()
        self.call_count += 1
        self._response_seq += 1
        seq = self._response_seq

        step = self._responses.pop(0) if self._responses else None

        async def _produce() -> None:
            body: dict | None = None
            exc: BaseException | None = None
            try:
                # Faux has no real wire payload, but we synthesize one for
                # observability tests. Run it through StreamOptions.on_payload
                # so the per-call mutator can produce a final dict before
                # subscribe_request listeners observe it — same ordering
                # contract as the real providers.
                payload = {
                    "model": model.id,
                    "messages": [m.model_dump() for m in messages],
                    "system_prompt": system_prompt,
                }
                payload = await invoke_on_payload(opts.on_payload, payload, model)
                if self._request_listeners:
                    await _fire_listeners(self._request_listeners, payload, model)

                if step is None:
                    error_msg = AssistantMessage(
                        content=[],
                        stop_reason="error",
                        error_message="No more faux responses queued",
                        usage=Usage(),
                        timestamp=time.time(),
                    )
                    await self._emit(
                        ms,
                        StreamEvent(
                            type="error", error_message=error_msg.error_message
                        ),
                        model,
                    )
                    ms.set_result(error_msg)
                    body = self._assemble_response(
                        seq=seq, model=model, message=error_msg
                    )
                    return

                if callable(step):
                    if _can_accept_extended_args(step):
                        args = (messages, model, system_prompt, tools)
                    else:
                        args = (messages, model)

                    if inspect.iscoroutinefunction(step):
                        resolved = await step(*args)
                    else:
                        resolved = step(*args)
                else:
                    resolved = step

                # Compute cache-aware usage
                output_tokens = _estimate_tokens(
                    json.dumps([b.model_dump() for b in resolved.content], default=str)
                )
                base_usage = Usage(output_tokens=output_tokens)
                cache_usage = self._compute_cache_usage(
                    system_prompt, tools, messages, base_usage
                )
                resolved = resolved.model_copy(
                    update={
                        "usage": cache_usage,
                        "provider_id": model.provider,
                        "model_id": model.id,
                    }
                )

                await self._stream_with_deltas(ms, resolved, opts.signal, model)
                # If the abort signal fired during streaming,
                # _stream_with_deltas sets an aborted result on the stream
                # and returns early. The queued `resolved` message would
                # misrepresent the run as a successful full response, so
                # surface the actual aborted result instead.
                if opts.signal and opts.signal.is_set():
                    aborted_msg = await ms.result()
                    body = self._assemble_response(
                        seq=seq, model=model, message=aborted_msg
                    )
                else:
                    body = self._assemble_response(
                        seq=seq, model=model, message=resolved
                    )
            except BaseException as e:
                exc = e
                error_msg = AssistantMessage(
                    content=[],
                    stop_reason="error",
                    error_message=str(e),
                    usage=Usage(),
                    timestamp=time.time(),
                )
                await self._emit(
                    ms, StreamEvent(type="error", error_message=str(e)), model
                )
                ms.set_result(error_msg)
                if not isinstance(e, Exception):
                    raise
            finally:
                # Await async listeners inline on normal/exception paths so
                # they finish before the producer task ends. On cancellation,
                # _fire_response_listeners falls back to sync fanout.
                await _fire_response_listeners(
                    self._response_listeners, body, model, exc
                )

        ms.attach_task(asyncio.create_task(_produce()))
        return ms

    async def _stream_with_deltas(
        self,
        stream: MessageStream,
        message: AssistantMessage,
        signal: asyncio.Event | None,
        model: Model | None = None,
    ) -> None:
        # ``model`` is optional so existing tests that call this private
        # helper directly without a model still work. _emit short-circuits
        # the listener fan-out when model is None.
        partial = AssistantMessage(
            content=[],
            stop_reason=message.stop_reason,
            usage=message.usage,
            timestamp=message.timestamp,
            provider_id=message.provider_id,
            model_id=message.model_id,
        )

        if signal and signal.is_set():
            aborted = self._make_aborted(partial)
            await self._emit(
                stream,
                StreamEvent(type="error", error_message="Request was aborted"),
                model,
            )
            stream.set_result(aborted)
            return

        await self._emit(
            stream,
            StreamEvent(type="start", partial=partial.model_copy(deep=True)),
            model,
        )

        for block in message.content:
            if signal and signal.is_set():
                aborted = self._make_aborted(partial)
                await self._emit(
                    stream,
                    StreamEvent(type="error", error_message="Request was aborted"),
                    model,
                )
                stream.set_result(aborted)
                return

            if isinstance(block, ThinkingContent):
                partial.content.append(ThinkingContent(thinking=""))
                block_idx = len(partial.content) - 1
                await self._emit(
                    stream,
                    StreamEvent(
                        type="thinking_start",
                        content_index=block_idx,
                        partial=partial.model_copy(deep=True),
                    ),
                    model,
                )
                for chunk in _split_by_token_size(block.thinking, self._min, self._max):
                    await self._schedule_chunk(chunk)
                    if signal and signal.is_set():
                        aborted = self._make_aborted(partial)
                        await self._emit(
                            stream,
                            StreamEvent(
                                type="error", error_message="Request was aborted"
                            ),
                            model,
                        )
                        stream.set_result(aborted)
                        return
                    last = partial.content[-1]
                    if isinstance(last, ThinkingContent):
                        partial.content[-1] = ThinkingContent(
                            thinking=last.thinking + chunk
                        )
                    await self._emit(
                        stream,
                        StreamEvent(
                            type="thinking_delta",
                            delta=chunk,
                            content_index=block_idx,
                            partial=partial.model_copy(deep=True),
                        ),
                        model,
                    )
                await self._emit(
                    stream,
                    StreamEvent(
                        type="thinking_end",
                        content_index=block_idx,
                        partial=partial.model_copy(deep=True),
                    ),
                    model,
                )

            elif isinstance(block, TextContent):
                partial.content.append(TextContent(text=""))
                block_idx = len(partial.content) - 1
                await self._emit(
                    stream,
                    StreamEvent(
                        type="text_start",
                        content_index=block_idx,
                        partial=partial.model_copy(deep=True),
                    ),
                    model,
                )
                for chunk in _split_by_token_size(block.text, self._min, self._max):
                    await self._schedule_chunk(chunk)
                    if signal and signal.is_set():
                        aborted = self._make_aborted(partial)
                        await self._emit(
                            stream,
                            StreamEvent(
                                type="error", error_message="Request was aborted"
                            ),
                            model,
                        )
                        stream.set_result(aborted)
                        return
                    last = partial.content[-1]
                    if isinstance(last, TextContent):
                        partial.content[-1] = TextContent(text=last.text + chunk)
                    await self._emit(
                        stream,
                        StreamEvent(
                            type="text_delta",
                            delta=chunk,
                            content_index=block_idx,
                            partial=partial.model_copy(deep=True),
                        ),
                        model,
                    )
                await self._emit(
                    stream,
                    StreamEvent(
                        type="text_end",
                        content_index=block_idx,
                        partial=partial.model_copy(deep=True),
                    ),
                    model,
                )

            elif isinstance(block, ToolCall):
                partial.content.append(
                    ToolCall(id=block.id, name=block.name, arguments={})
                )
                block_idx = len(partial.content) - 1
                await self._emit(
                    stream,
                    StreamEvent(
                        type="toolcall_start",
                        content_index=block_idx,
                        partial=partial.model_copy(deep=True),
                    ),
                    model,
                )
                json_str = json.dumps(block.arguments)
                for chunk in _split_by_token_size(json_str, self._min, self._max):
                    await self._schedule_chunk(chunk)
                    if signal and signal.is_set():
                        aborted = self._make_aborted(partial)
                        await self._emit(
                            stream,
                            StreamEvent(
                                type="error", error_message="Request was aborted"
                            ),
                            model,
                        )
                        stream.set_result(aborted)
                        return
                    await self._emit(
                        stream,
                        StreamEvent(
                            type="toolcall_delta",
                            delta=chunk,
                            content_index=block_idx,
                            partial=partial.model_copy(deep=True),
                        ),
                        model,
                    )
                last = partial.content[-1]
                if isinstance(last, ToolCall):
                    partial.content[-1] = ToolCall(
                        id=block.id, name=block.name, arguments=block.arguments
                    )
                await self._emit(
                    stream,
                    StreamEvent(
                        type="toolcall_end",
                        content_index=block_idx,
                        partial=partial.model_copy(deep=True),
                    ),
                    model,
                )

        if message.stop_reason in ("error", "aborted"):
            await self._emit(
                stream,
                StreamEvent(type="error", error_message=message.error_message),
                model,
            )
            stream.set_result(message)
            return

        await self._emit(stream, StreamEvent(type="done"), model)
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
