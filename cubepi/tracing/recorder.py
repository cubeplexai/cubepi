"""Map cubepi :class:`AgentEvent` + provider listener callbacks to
OpenTelemetry spans.

Span hierarchy::

    invoke_agent <agent_name>             [INTERNAL]
    └── cubepi.turn                        [INTERNAL]
        ├── chat <model>                   [CLIENT]    (lifetime: provider listeners)
        └── execute_tool <tool_name>       [INTERNAL]

The ``chat`` span lifetime is **driven by provider listeners**, NOT by
the agent's ``MessageStartEvent`` / ``MessageEndEvent`` — those fire
after ``after_model_response`` middleware hooks and so would conflate
hook time with the LLM roundtrip. The chat span opens at
``provider.on_request`` and closes at ``provider.on_response`` (success
or error). See ``docs/specs/2026-05-18-cubepi-tracing-design.md`` §9.
"""

from __future__ import annotations

import hashlib
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from cubepi.agent.types import (
    AgentEndEvent,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    TurnEndEvent,
    TurnStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from cubepi.providers.base import (
    StreamEvent,
    ToolResultMessage,
    UserMessage,
)
from cubepi.tracing.errors import cubepi_error_type_for
from cubepi.tracing.schema import (
    CUBEPI_ABORTED,
    CUBEPI_AGENT_SYSTEM_PROMPT_SHA256,
    CUBEPI_AGENT_TOOLS,
    CUBEPI_INPUT_MESSAGES_COUNT,
    CUBEPI_LLM_THINKING_LEVEL,
    CUBEPI_OUTPUT_MESSAGES_COUNT,
    CUBEPI_RUN_ID,
    CUBEPI_TOOL_BLOCK_REASON,
    CUBEPI_TOOL_BLOCKED_BY_HOOK,
    CUBEPI_TOOL_EXECUTION_MODE,
    CUBEPI_TOOL_IS_ERROR,
    CUBEPI_TOOL_TERMINATE,
    CUBEPI_TURN_INDEX,
    CUBEPI_TURN_STOP_REASON,
    CUBEPI_TURN_TERMINATED_BY_TOOL,
    CUBEPI_TURN_TOOL_CALLS_COUNT,
    ERROR_TYPE,
    EVENT_GEN_AI_EXCEPTION,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_STREAM,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_DESCRIPTION,
    GEN_AI_TOOL_NAME,
    GEN_AI_TOOL_TYPE,
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
    OP_CHAT,
    OP_EXECUTE_TOOL,
    OP_INVOKE_AGENT,
    SPAN_NAME_CHAT,
    SPAN_NAME_EXECUTE_TOOL,
    SPAN_NAME_INVOKE_AGENT,
    SPAN_NAME_TURN,
    map_provider_name,
)

if TYPE_CHECKING:
    from cubepi.agent.agent import Agent
    from cubepi.providers.base import Model
    from cubepi.tracing.tracer import Tracer


@dataclass
class _RunState:
    """Per-run mutable state. One per attached agent run."""

    run_id: str
    agent_span: Any  # opentelemetry.trace.Span
    turn_span: Any | None = None
    turn_index: int = -1
    chat_span: Any | None = None
    chat_open_ns: int | None = None
    chat_first_chunk_recorded: bool = False
    tool_spans: dict[str, Any] = field(default_factory=dict)
    # Caller-supplied extra attrs for this run (set via run_scope, future).
    extra_attrs: dict[str, Any] = field(default_factory=dict)
    # Counts for invoke_agent attrs.
    new_message_count: int = 0
    # Whether any tool reported terminate=True in the current turn.
    turn_terminated_by_tool: bool = False


class Recorder:
    """Subscribe to agent + provider events and produce OTel spans.

    Lifetime: one :class:`Recorder` per :meth:`Tracer.attach` call. The
    recorder maintains per-run state keyed by a generated run_id; one
    agent can host many sequential runs over the recorder's lifetime.
    """

    def __init__(self, tracer: "Tracer", *, record_content: bool = False) -> None:
        self._tracer = tracer
        self._record_content = record_content
        # Active run state. cubepi agents serialize runs (one
        # ``agent.run()`` at a time per Agent), so a single slot is
        # enough — but defensively key by AgentStartEvent identity
        # in case the contract changes.
        self._run: _RunState | None = None

    # ------------------------------------------------------------------
    # Attach / detach
    # ------------------------------------------------------------------

    def attach(self, agent: "Agent") -> Callable[[], None]:
        from cubepi.providers.base import BaseProvider

        unsub_agent = agent.subscribe(self._on_agent_event)
        # ``Agent`` stores the provider as a private attribute; accept
        # either the (private) ``_provider`` or a public ``provider``
        # alias should one be added later.
        provider = getattr(agent, "_provider", None) or getattr(agent, "provider", None)
        provider_detachers: list[Callable[[], None]] = []
        if isinstance(provider, BaseProvider):
            provider_detachers.append(
                provider.subscribe_request(self._on_provider_request)
            )
            provider_detachers.append(provider.subscribe_chunk(self._on_provider_chunk))
            provider_detachers.append(
                provider.subscribe_response(self._on_provider_response)
            )

        async def _detach_async() -> None:
            unsub_agent()
            for d in provider_detachers:
                try:
                    d()
                except Exception:
                    pass
            await self._tracer.force_flush()

        def detach() -> None:
            # Synchronous shim. Tests can also call ``detach_async``.
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop — best-effort sync detach.
                unsub_agent()
                for d in provider_detachers:
                    try:
                        d()
                    except Exception:
                        pass
                return
            # Inside a loop — schedule the async detach.
            loop.create_task(_detach_async())

        return detach

    # ------------------------------------------------------------------
    # Agent event handler
    # ------------------------------------------------------------------

    async def _on_agent_event(self, event: Any, signal: Any | None = None) -> None:
        # The agent calls listeners with (event, signal). We don't use signal.
        del signal
        try:
            if isinstance(event, AgentStartEvent):
                self._on_agent_start()
            elif isinstance(event, TurnStartEvent):
                self._on_turn_start()
            elif isinstance(event, ToolExecutionStartEvent):
                self._on_tool_exec_start(event)
            elif isinstance(event, ToolExecutionEndEvent):
                self._on_tool_exec_end(event)
            elif isinstance(event, MessageStartEvent):
                self._on_message_start(event)
            elif isinstance(event, MessageEndEvent):
                self._on_message_end(event)
            elif isinstance(event, TurnEndEvent):
                self._on_turn_end(event)
            elif isinstance(event, AgentEndEvent):
                self._on_agent_end(event)
            # MessageUpdateEvent / ToolExecutionUpdateEvent: IGNORED.
        except Exception:
            # Recorders must never crash the agent.
            pass

    # ------------------------------------------------------------------
    # invoke_agent
    # ------------------------------------------------------------------

    def _on_agent_start(self) -> None:
        run_id = str(uuid.uuid4())
        # Open the root invoke_agent span. Caller-context propagation
        # (parent_trace_id / parent_span_id from a host service) lands
        # here in a future run_scope feature.
        span = self._tracer.otel_tracer.start_span(
            name=SPAN_NAME_INVOKE_AGENT,
            kind=SpanKind.INTERNAL,
            attributes={
                GEN_AI_OPERATION_NAME: OP_INVOKE_AGENT,
                CUBEPI_RUN_ID: run_id,
                # gen_ai.provider.name is required by semconv; we set a
                # placeholder here and overwrite at the first chat span
                # with the actual provider id.
                GEN_AI_PROVIDER_NAME: "cubepi",
            },
        )
        # Resource carries gen_ai.agent.name at process level — agents
        # that vary per-run set their own value via _ensure_agent_name.
        self._run = _RunState(run_id=run_id, agent_span=span)

    def _on_agent_end(self, event: AgentEndEvent) -> None:
        run = self._run
        if run is None:
            return
        run.new_message_count = len(event.messages)
        run.agent_span.set_attribute(
            CUBEPI_OUTPUT_MESSAGES_COUNT, run.new_message_count
        )
        run.agent_span.end()
        self._run = None

    # ------------------------------------------------------------------
    # cubepi.turn
    # ------------------------------------------------------------------

    def _on_turn_start(self) -> None:
        run = self._run
        if run is None:
            return
        run.turn_index += 1
        ctx = trace.set_span_in_context(run.agent_span)
        run.turn_span = self._tracer.otel_tracer.start_span(
            name=SPAN_NAME_TURN,
            kind=SpanKind.INTERNAL,
            context=ctx,
            attributes={
                CUBEPI_TURN_INDEX: run.turn_index,
            },
        )
        run.turn_terminated_by_tool = False

    def _on_turn_end(self, event: TurnEndEvent) -> None:
        run = self._run
        if run is None or run.turn_span is None:
            return
        msg = event.message
        # Map cubepi stop_reason to gen_ai.response.finish_reasons on
        # the chat span — already done in _on_provider_response. Here we
        # record the cubepi-normalized stop_reason on the turn span.
        stop_reason = getattr(msg, "stop_reason", None)
        if stop_reason:
            run.turn_span.set_attribute(CUBEPI_TURN_STOP_REASON, stop_reason)
        tool_calls_count = sum(
            1
            for c in getattr(msg, "content", [])
            if getattr(c, "type", "") == "tool_call"
        )
        run.turn_span.set_attribute(CUBEPI_TURN_TOOL_CALLS_COUNT, tool_calls_count)
        if run.turn_terminated_by_tool:
            run.turn_span.set_attribute(CUBEPI_TURN_TERMINATED_BY_TOOL, True)
        # Error handling on the turn: if assistant message stopped with
        # "error", mark turn ERROR. Abort path leaves UNSET + sets
        # cubepi.aborted on the invoke_agent root.
        if stop_reason == "error":
            err_msg = getattr(msg, "error_message", None) or "model error"
            run.turn_span.set_status(Status(StatusCode.ERROR, err_msg[:256]))
            run.turn_span.set_attribute(ERROR_TYPE, "cubepi.error")
        elif stop_reason == "aborted":
            run.turn_span.set_attribute(CUBEPI_ABORTED, True)
            run.agent_span.set_attribute(CUBEPI_ABORTED, True)
        run.turn_span.end()
        run.turn_span = None

    # ------------------------------------------------------------------
    # execute_tool
    # ------------------------------------------------------------------

    def _on_tool_exec_start(self, event: ToolExecutionStartEvent) -> None:
        run = self._run
        if run is None or run.turn_span is None:
            return
        ctx = trace.set_span_in_context(run.turn_span)
        attrs: dict[str, Any] = {
            GEN_AI_OPERATION_NAME: OP_EXECUTE_TOOL,
            GEN_AI_TOOL_NAME: event.tool_name,
            GEN_AI_TOOL_CALL_ID: event.tool_call_id,
            GEN_AI_TOOL_TYPE: "function",
        }
        # Lookup AgentTool for description + execution_mode.
        tool_obj = self._find_tool(event.tool_name)
        if tool_obj is not None:
            if tool_obj.description:
                attrs[GEN_AI_TOOL_DESCRIPTION] = tool_obj.description
            mode = tool_obj.execution_mode or "parallel"
            attrs[CUBEPI_TOOL_EXECUTION_MODE] = mode
        span = self._tracer.otel_tracer.start_span(
            name=f"{SPAN_NAME_EXECUTE_TOOL} {event.tool_name}",
            kind=SpanKind.INTERNAL,
            context=ctx,
            attributes=attrs,
        )
        run.tool_spans[event.tool_call_id] = span

    def _on_tool_exec_end(self, event: ToolExecutionEndEvent) -> None:
        run = self._run
        if run is None:
            return
        span = run.tool_spans.pop(event.tool_call_id, None)
        if span is None:
            return
        span.set_attribute(CUBEPI_TOOL_IS_ERROR, event.is_error)
        if event.terminate:
            span.set_attribute(CUBEPI_TOOL_TERMINATE, True)
            run.turn_terminated_by_tool = True
        if event.blocked_by_hook:
            span.set_attribute(CUBEPI_TOOL_BLOCKED_BY_HOOK, True)
            if event.block_reason is not None:
                span.set_attribute(CUBEPI_TOOL_BLOCK_REASON, event.block_reason)
        if event.is_error:
            span.set_status(Status(StatusCode.ERROR, "tool error"))
            err_type = (
                "cubepi.tool.blocked_by_hook"
                if event.blocked_by_hook
                else "cubepi.tool.error"
            )
            span.set_attribute(ERROR_TYPE, err_type)
        span.end()

    # ------------------------------------------------------------------
    # MessageStart/End — used to enrich invoke_agent attrs but NOT to
    # drive the chat span lifetime (that's the provider listeners' job).
    # ------------------------------------------------------------------

    def _on_message_start(self, event: MessageStartEvent) -> None:
        run = self._run
        if run is None:
            return
        msg = event.message
        # Count user-provided + tool-result messages as input.
        if isinstance(msg, (UserMessage, ToolResultMessage)):
            run.agent_span.set_attribute(
                CUBEPI_INPUT_MESSAGES_COUNT,
                int(
                    (run.agent_span.attributes or {}).get(
                        CUBEPI_INPUT_MESSAGES_COUNT, 0
                    )
                )
                + 1,
            )

    def _on_message_end(self, event: MessageEndEvent) -> None:
        # Reserved for future use. Currently no-op — chat span is
        # closed by the provider response listener, and turn span
        # gets its stop_reason from TurnEndEvent.
        del event

    # ------------------------------------------------------------------
    # Provider listeners — drive the chat span lifetime
    # ------------------------------------------------------------------

    def _on_provider_request(self, payload: dict, model: "Model") -> None:
        run = self._run
        if run is None or run.turn_span is None:
            return
        ctx = trace.set_span_in_context(run.turn_span)
        attrs: dict[str, Any] = {
            GEN_AI_OPERATION_NAME: OP_CHAT,
            GEN_AI_PROVIDER_NAME: map_provider_name(model.provider),
            GEN_AI_REQUEST_MODEL: model.id,
            GEN_AI_REQUEST_STREAM: True,
        }
        # Pull StreamOptions-derived params from the payload where the
        # provider exposed them as final kwargs.
        for key, attr in (
            ("max_tokens", GEN_AI_REQUEST_MAX_TOKENS),
            ("temperature", GEN_AI_REQUEST_TEMPERATURE),
            ("top_p", GEN_AI_REQUEST_TOP_P),
        ):
            if key in payload and payload[key] is not None:
                attrs[attr] = payload[key]

        # Thinking level on cubepi.* namespace.
        thinking = payload.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") == "enabled":
            attrs[CUBEPI_LLM_THINKING_LEVEL] = "on"
        elif thinking is None and (
            payload.get("reasoning", {}).get("effort")
            if payload.get("reasoning")
            else None
        ):
            attrs[CUBEPI_LLM_THINKING_LEVEL] = payload["reasoning"]["effort"]

        # System prompt sha256 on root agent span (once per run).
        self._maybe_record_system_prompt_hash(payload, run)

        # Update root's gen_ai.provider.name with the first concrete one.
        if (run.agent_span.attributes or {}).get(GEN_AI_PROVIDER_NAME) == "cubepi":
            run.agent_span.set_attribute(
                GEN_AI_PROVIDER_NAME, attrs[GEN_AI_PROVIDER_NAME]
            )

        # Tool count on root.
        tools = payload.get("tools")
        if isinstance(tools, list) and tools:
            run.agent_span.set_attribute(
                CUBEPI_AGENT_TOOLS, [_safe_tool_name(t) for t in tools]
            )

        chat_span = self._tracer.otel_tracer.start_span(
            name=f"{SPAN_NAME_CHAT} {model.id}",
            kind=SpanKind.CLIENT,
            context=ctx,
            attributes=attrs,
        )
        run.chat_span = chat_span
        run.chat_open_ns = time.time_ns()
        run.chat_first_chunk_recorded = False
        # cubepi.llm.raw_request is content; MVP is record_content=False.

    def _on_provider_chunk(self, event: StreamEvent, model: "Model") -> None:
        del model
        run = self._run
        if run is None or run.chat_span is None or run.chat_first_chunk_recorded:
            return
        # Only count "content" chunks for time-to-first-chunk (not 'start').
        if event.type in (
            "text_delta",
            "thinking_delta",
            "toolcall_delta",
        ):
            opened = run.chat_open_ns or time.time_ns()
            ttft_s = (time.time_ns() - opened) / 1e9
            run.chat_span.set_attribute(GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK, ttft_s)
            run.chat_first_chunk_recorded = True

    def _on_provider_response(
        self,
        body: dict | None,
        model: "Model",
        exc: BaseException | None,
    ) -> None:
        del model
        run = self._run
        if run is None or run.chat_span is None:
            return
        span = run.chat_span
        try:
            if body is not None:
                self._record_chat_response_attrs(span, body)
            if exc is None:
                # Healthy completion — leave Status UNSET per OTel guidance.
                pass
            elif _is_cancelled_error(exc):
                # Abort: not a failure. Mark and leave UNSET.
                span.set_attribute(CUBEPI_ABORTED, True)
                span.set_attribute(ERROR_TYPE, "cubepi.aborted")
            else:
                span.set_status(Status(StatusCode.ERROR, str(exc)[:256]))
                span.set_attribute(ERROR_TYPE, cubepi_error_type_for(exc))
                span.add_event(
                    name=EVENT_GEN_AI_EXCEPTION,
                    attributes={
                        "exception.type": type(exc).__name__,
                        "exception.message": str(exc),
                        "exception.stacktrace": "".join(
                            traceback.format_exception(exc)
                        ),
                    },
                )
        finally:
            span.end()
            run.chat_span = None
            run.chat_open_ns = None
            run.chat_first_chunk_recorded = False

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _find_tool(self, name: str):
        # Look up the AgentTool on the active agent via context. The
        # Recorder doesn't hold an agent reference; instead we accept
        # that the tool lookup is best-effort. cubepi's _prepare_tool_call
        # already validates the tool exists; if we miss it here, the
        # span just lacks the description/execution_mode attrs.
        return None

    def _record_chat_response_attrs(self, span: Any, body: dict) -> None:
        # Anthropic-shaped body
        if "stop_reason" in body and "usage" in body:
            usage = body.get("usage") or {}
            self._set_usage_anthropic_like(span, usage)
            finish_reason = body.get("stop_reason")
            if finish_reason:
                span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason])
            if "model" in body and body["model"]:
                span.set_attribute(GEN_AI_RESPONSE_MODEL, body["model"])
            if "id" in body and body["id"]:
                span.set_attribute(GEN_AI_RESPONSE_ID, body["id"])
            return
        # OpenAI chat.completion-shaped body
        if "choices" in body and isinstance(body.get("choices"), list):
            usage = body.get("usage") or {}
            self._set_usage_openai_like(span, usage)
            choices = body["choices"]
            if choices:
                first = choices[0] or {}
                finish = first.get("finish_reason")
                if finish:
                    span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [finish])
            if body.get("id"):
                span.set_attribute(GEN_AI_RESPONSE_ID, body["id"])
            if body.get("model"):
                span.set_attribute(GEN_AI_RESPONSE_MODEL, body["model"])
            return
        # OpenAI Responses-shaped body
        if body.get("object") == "response" or "output" in body:
            usage = body.get("usage") or {}
            self._set_usage_openai_responses_like(span, usage)
            status = body.get("status")
            if status:
                span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [status])
            if body.get("id"):
                span.set_attribute(GEN_AI_RESPONSE_ID, body["id"])
            if body.get("model"):
                span.set_attribute(GEN_AI_RESPONSE_MODEL, body["model"])
            return
        # Unknown shape — record what we can defensively.
        if isinstance(body.get("model"), str):
            span.set_attribute(GEN_AI_RESPONSE_MODEL, body["model"])

    @staticmethod
    def _set_usage_anthropic_like(span: Any, usage: dict) -> None:
        # Anthropic's input_tokens excludes cached tokens; semconv expects
        # the inclusive count. Reconcile.
        input_t = int(usage.get("input_tokens", 0) or 0)
        cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
        cache_create = int(usage.get("cache_creation_input_tokens", 0) or 0)
        span.set_attribute(
            GEN_AI_USAGE_INPUT_TOKENS, input_t + cache_read + cache_create
        )
        span.set_attribute(
            GEN_AI_USAGE_OUTPUT_TOKENS, int(usage.get("output_tokens", 0) or 0)
        )
        if cache_read:
            span.set_attribute(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, cache_read)
        if cache_create:
            span.set_attribute(GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, cache_create)

    @staticmethod
    def _set_usage_openai_like(span: Any, usage: dict) -> None:
        # OpenAI uses prompt_tokens / completion_tokens; cached separately.
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        details = usage.get("prompt_tokens_details") or {}
        cache_read = (
            int(details.get("cached_tokens", 0) or 0)
            if isinstance(details, dict)
            else 0
        )
        # prompt_tokens already includes cached; semconv input_tokens is
        # the total prompt count — emit as-is.
        span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, prompt)
        span.set_attribute(
            GEN_AI_USAGE_OUTPUT_TOKENS, int(usage.get("completion_tokens", 0) or 0)
        )
        if cache_read:
            span.set_attribute(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, cache_read)

    @staticmethod
    def _set_usage_openai_responses_like(span: Any, usage: dict) -> None:
        # OpenAI Responses API surfaces input_tokens / output_tokens.
        span.set_attribute(
            GEN_AI_USAGE_INPUT_TOKENS, int(usage.get("input_tokens", 0) or 0)
        )
        span.set_attribute(
            GEN_AI_USAGE_OUTPUT_TOKENS, int(usage.get("output_tokens", 0) or 0)
        )
        details = usage.get("input_tokens_details") or {}
        if isinstance(details, dict):
            cache_read = int(details.get("cached_tokens", 0) or 0)
            if cache_read:
                span.set_attribute(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, cache_read)
        out_details = usage.get("output_tokens_details") or {}
        if isinstance(out_details, dict):
            reasoning = int(out_details.get("reasoning_tokens", 0) or 0)
            if reasoning:
                span.set_attribute(GEN_AI_USAGE_REASONING_OUTPUT_TOKENS, reasoning)

    def _maybe_record_system_prompt_hash(self, payload: dict, run: _RunState) -> None:
        if (run.agent_span.attributes or {}).get(CUBEPI_AGENT_SYSTEM_PROMPT_SHA256):
            return
        text = _extract_system_prompt(payload)
        if not text:
            return
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        run.agent_span.set_attribute(CUBEPI_AGENT_SYSTEM_PROMPT_SHA256, digest)


def _extract_system_prompt(payload: dict) -> str | None:
    """Find the system prompt text in a provider's wire payload.

    Provider shape variance is real; check the known forms in turn:

    - Anthropic: ``payload["system"]`` is a string OR a list of
      ``{"type": "text", "text": ...}`` cache-control blocks.
    - Faux: ``payload["system_prompt"]`` is a string.
    - OpenAI chat-completions: ``payload["messages"][0]`` is
      ``{"role": "system", "content": ...}`` when a prompt was set.
    - OpenAI Responses: ``payload["input"][0]`` is ``{"role":
      "developer" | "system", "content": ...}`` when a prompt was set.
    """
    system = payload.get("system")
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        joined = "".join(parts)
        if joined:
            return joined
    if isinstance(payload.get("system_prompt"), str):
        return payload["system_prompt"]
    for key in ("messages", "input"):
        seq = payload.get(key)
        if isinstance(seq, list) and seq:
            first = seq[0]
            if isinstance(first, dict) and first.get("role") in ("system", "developer"):
                content = first.get("content")
                if isinstance(content, str):
                    return content
    return None


def _safe_tool_name(t: Any) -> str:
    if isinstance(t, dict):
        return str(t.get("name") or "")
    name = getattr(t, "name", None)
    return str(name) if name else ""


def _is_cancelled_error(exc: BaseException) -> bool:
    import asyncio

    return isinstance(exc, asyncio.CancelledError)
