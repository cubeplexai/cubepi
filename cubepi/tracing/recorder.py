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
from cubepi.tracing.content import (
    messages_to_semconv,
    serialize_for_attribute,
    system_instructions_to_semconv,
    tool_definitions_to_semconv,
)
from cubepi.tracing.errors import cubepi_error_type_for
from cubepi.tracing.schema import (
    CUBEPI_ABORTED,
    CUBEPI_AGENT_SYSTEM_PROMPT_SHA256,
    CUBEPI_AGENT_TOOLS,
    CUBEPI_INPUT_MESSAGES_COUNT,
    CUBEPI_LLM_RAW_REQUEST,
    CUBEPI_LLM_RAW_RESPONSE,
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
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_SYSTEM_INSTRUCTIONS,
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
    GEN_AI_TOOL_DEFINITIONS,
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
    OPENAI_API_TYPE,
    OPENAI_REQUEST_SERVICE_TIER,
    OPENAI_RESPONSE_SERVICE_TIER,
    OPENAI_RESPONSE_SYSTEM_FINGERPRINT,
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
    # contextvars.Token per tool_call_id, returned by
    # ``cubepi.mcp._tracing.register_tool_span``. Passed back on
    # ``unregister_tool_span`` to restore the prior contextvar state.
    tool_span_tokens: dict[str, Any] = field(default_factory=dict)
    # Caller-supplied extra attrs for this run (set via run_scope, future).
    extra_attrs: dict[str, Any] = field(default_factory=dict)
    # Counts for invoke_agent attrs.
    new_message_count: int = 0
    # Whether any tool reported terminate=True in the current turn.
    turn_terminated_by_tool: bool = False
    # Content recording (only used when record_content=True).
    system_prompt: str | None = None
    # input_messages: user / tool_result messages — used by the root
    # span's ``gen_ai.input.messages``. Pure caller-provided input.
    input_messages: list[Any] = field(default_factory=list)
    # output_messages: assistant + tool_result messages produced
    # during the run — used by the root span's ``gen_ai.output.messages``.
    output_messages: list[Any] = field(default_factory=list)
    # transcript: ALL messages in chronological order, including prior
    # assistant turns. Used as the chat span's ``gen_ai.input.messages``
    # so multi-turn tool-using runs reconstruct correctly — the second
    # chat call's request includes the prior assistant tool-call
    # message plus the tool_result, and so must its input.messages
    # attribute.
    transcript: list[Any] = field(default_factory=list)
    # Per-turn message accumulators (cleared at TurnStart).
    turn_input_messages: list[Any] = field(default_factory=list)
    turn_output_messages: list[Any] = field(default_factory=list)
    # Chat-span specific tool_definitions captured at request time.
    chat_tool_definitions: list[dict] | None = None


class Recorder:
    """Subscribe to agent + provider events and produce OTel spans.

    Lifetime: one :class:`Recorder` per :meth:`Tracer.attach` call. The
    recorder maintains per-run state keyed by a generated run_id; one
    agent can host many sequential runs over the recorder's lifetime.
    """

    def __init__(
        self,
        tracer: "Tracer",
        *,
        record_content: bool = False,
        redact: "Callable[[str, Any], Any] | None" = None,
    ) -> None:
        self._tracer = tracer
        self._record_content = record_content
        self._redact = redact
        # Active run state. cubepi agents serialize runs (one
        # ``agent.run()`` at a time per Agent), so a single slot is
        # enough — but defensively key by AgentStartEvent identity
        # in case the contract changes.
        self._run: _RunState | None = None
        # Set on attach() so the recorder can look up AgentTool
        # definitions (description, execution_mode) at tool-exec span
        # open. ``None`` if the agent doesn't expose a tool registry.
        self._agent: Any | None = None

    # ------------------------------------------------------------------
    # Attach / detach
    # ------------------------------------------------------------------

    def attach(self, agent: "Agent") -> Callable[[], None]:
        from cubepi.providers.base import BaseProvider

        self._agent = agent
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

        def _sync_detach() -> None:
            """All synchronous cleanup: unsubscribe + close any spans
            still open + sweep leaked registrations. Runs eagerly in
            ``detach()`` so cleanup is observable on the next line —
            not deferred to a loop tick that may never arrive if the
            caller immediately awaits ``tracer.shutdown()`` or exits
            ``asyncio.run`` (codex round-11).
            """
            unsub_agent()
            for d in provider_detachers:
                try:
                    d()
                except Exception:
                    pass
            # Close any spans that an in-flight cancelled run left
            # open, then sweep MCP registrations (codex round-10 /
            # overall-review BLOCKING). CancelledError bypasses the
            # agent's ``except Exception`` handler so the per-event
            # end handlers never fire.
            self._close_open_spans(self._run)
            self._sweep_tool_span_tokens(self._run)

        def detach():
            """Synchronous cleanup + scheduled flush.

            The unsubscribe / open-span close / MCP-token sweep all
            run synchronously here — observable on the line after
            ``detach()`` returns. The flush is scheduled as an
            ``asyncio.Task`` and returned so callers who need the
            buffered spans persisted before they proceed can
            ``await detach()`` — the previous fire-and-forget
            ``create_task(...)`` left them in ``BatchSpanProcessor``
            indefinitely if the caller exited ``asyncio.run`` (codex
            overall-review MAJOR).

            Outside an async context (no running loop) returns
            ``None`` — sync cleanup is done but flush is the caller's
            responsibility via ``await tracer.shutdown()``.
            """
            _sync_detach()
            try:
                import asyncio

                loop = asyncio.get_running_loop()
            except RuntimeError:
                return None
            return loop.create_task(self._tracer.force_flush())

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

    def _agent_signal_is_set(self) -> bool:
        """True iff the attached agent's active abort signal is set.

        Used to disambiguate a provider response listener firing as
        ``(body=None, exc=None)``: that shape can mean either a real
        cooperative abort (we should mark the chat span aborted) or a
        provider-side non-abort fallback like
        ``OpenAIResponsesProvider`` finalizing an incomplete stream
        (we must NOT mark it aborted).
        """
        agent = self._agent
        if agent is None:
            return False
        sig = getattr(agent, "_active_signal", None)
        if sig is None:
            return False
        try:
            return bool(sig.is_set())
        except Exception:
            return False

    def _sweep_tool_span_tokens(self, run: "_RunState | None") -> None:
        """Drain ``run.tool_span_tokens`` through ``unregister_tool_span``.

        The cubepi agent loop emits ``ToolExecutionEndEvent`` for every
        tool that produced a ``ToolExecutionStartEvent`` *only on the
        ``Exception`` path*. ``asyncio.CancelledError`` inherits from
        ``BaseException`` and bypasses that handler, so a run cancelled
        while an MCP tool was in flight leaves the registration alive
        in :mod:`cubepi.mcp._tracing._active_entries`. Repeated aborts
        would leak parent-span/provider pairs for the lifetime of the
        process.

        We sweep on agent_start (next run reuses this Recorder),
        agent_end (normal completion + agent-caught errors), and on
        detach (final safety net for an aborted run that never starts
        again). This is the cleanup path codex round-10 requested.
        """
        if run is None or not run.tool_span_tokens:
            return
        try:
            from cubepi.mcp import _tracing as _mcp_tracing
        except ImportError:  # pragma: no cover — mcp module always present
            run.tool_span_tokens.clear()
            return
        for token in list(run.tool_span_tokens.values()):
            try:
                _mcp_tracing.unregister_tool_span(token)
            except Exception:
                pass
        run.tool_span_tokens.clear()

    def _close_open_spans(self, run: "_RunState | None") -> None:
        """End any spans still open on ``run``.

        Normal flow ends ``execute_tool`` / ``chat`` / ``cubepi.turn`` /
        ``invoke_agent`` spans in their respective event handlers. But
        ``asyncio.CancelledError`` bypasses cubepi's agent-loop
        ``except Exception`` handler — no ``ToolExecutionEnd`` /
        ``TurnEnd`` / ``AgentEnd`` is emitted. Without this sweep the
        spans never reach ``span.end()`` and ``BatchSpanProcessor``
        never exports them: cancelled runs simply disappear from the
        backend (codex overall-review BLOCKING).

        Each span gets ``cubepi.aborted=true`` + ``error.type`` so the
        backend sees the run was interrupted rather than silently
        succeeding. Status is left UNSET — cancellation isn't a
        failure.
        """
        if run is None:
            return
        for span in list(run.tool_spans.values()):
            try:
                span.set_attribute(CUBEPI_ABORTED, True)
                span.set_attribute(ERROR_TYPE, "cubepi.aborted")
                span.end()
            except Exception:
                pass
        run.tool_spans.clear()
        if run.chat_span is not None:
            try:
                run.chat_span.set_attribute(CUBEPI_ABORTED, True)
                run.chat_span.set_attribute(ERROR_TYPE, "cubepi.aborted")
                run.chat_span.end()
            except Exception:
                pass
            run.chat_span = None
            run.chat_open_ns = None
            run.chat_first_chunk_recorded = False
        if run.turn_span is not None:
            try:
                run.turn_span.set_attribute(CUBEPI_ABORTED, True)
                run.turn_span.end()
            except Exception:
                pass
            run.turn_span = None
        if run.agent_span is not None:
            try:
                run.agent_span.set_attribute(CUBEPI_ABORTED, True)
                run.agent_span.end()
            except Exception:
                pass
            # Don't null agent_span — _RunState gets dropped wholesale.

    def _on_agent_start(self) -> None:
        # Defensive cleanup: if a prior run was cancelled
        # (CancelledError inherits BaseException — cubepi's agent loop
        # doesn't emit ToolExecutionEnd / TurnEnd / AgentEnd in that
        # path), its open spans never reached span.end() and its MCP
        # tool-span registrations are still live. Close + sweep both
        # before opening the new _RunState so the prior run is at
        # least visible (marked aborted) in the backend and shared
        # state stays consistent.
        self._close_open_spans(self._run)
        self._sweep_tool_span_tokens(self._run)

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

        # Stamp any tags / metadata set via ``cubepi.tracing.tracing_context``
        # for this run. Per-task contextvar scoping means concurrent
        # agents each see their own values; nested ``tracing_context``
        # blocks merge before reaching here.
        #
        # User metadata is namespaced under ``cubepi.metadata.*`` so
        # that keys like ``run_id`` / ``turn_index`` / ``agent.tools``
        # — which the recorder owns under ``cubepi.*`` — can never be
        # overwritten by caller-supplied values. Reserved cubepi
        # schema keys (especially ``cubepi.run_id`` which the JSONL
        # exporter shards on) must stay recorder-controlled (codex P2
        # on PR #92).
        try:
            from cubepi.tracing.context import _current_metadata, _current_tags

            tags = _current_tags()
            if tags:
                span.set_attribute("cubepi.tags", tags)
            metadata = _current_metadata()
            for key, value in metadata.items():
                # OTel attribute values are limited to scalar primitives
                # and homogeneous sequences. Anything else is silently
                # dropped here — better to lose one tag than corrupt the
                # whole span.
                try:
                    span.set_attribute(f"cubepi.metadata.{key}", value)
                except (TypeError, ValueError):
                    pass
        except ImportError:  # pragma: no cover — context module always available
            pass

        # Seed the transcript from the agent's existing conversation
        # history. The cubepi agent loop only emits MessageStartEvent
        # for newly-introduced messages (new prompts in ``run_agent_loop``,
        # tool_results in ``execute_tool_calls``); the prior history is
        # NOT replayed. Without this seed, ``Agent.resume()`` /
        # ``run_agent_loop_continue`` (which adds no new prompt) would
        # leave ``run.transcript`` empty and the first chat span's
        # ``gen_ai.input.messages`` would omit the conversation history
        # that the provider request actually carries. For a normal
        # ``prompt()`` the pre-run history is also present here — the
        # new prompt(s) then append via MessageStart, producing the
        # full chronological context.
        #
        # Source: ``Agent.state.messages`` — that's where production
        # code (e.g. ``_create_context_snapshot``) reads the persisted
        # conversation from. ``getattr(agent, "messages", …)`` would
        # return ``None`` on real Agent instances because the
        # ``messages`` property lives on :class:`AgentState`, not on
        # :class:`Agent` (codex P2 follow-up on PR #87).
        if self._agent is not None:
            history: list[Any] = []
            try:
                state = getattr(self._agent, "state", None)
                state_messages = (
                    getattr(state, "messages", None) if state is not None else None
                )
                if state_messages:
                    history = list(state_messages)
            except Exception:
                history = []
            if history:
                self._run.transcript.extend(history)

    def _on_agent_end(self, event: AgentEndEvent) -> None:
        run = self._run
        if run is None:
            return
        run.new_message_count = len(event.messages)
        run.agent_span.set_attribute(
            CUBEPI_OUTPUT_MESSAGES_COUNT, run.new_message_count
        )
        if self._record_content:
            if run.system_prompt:
                self._set_content_attr(
                    run.agent_span,
                    GEN_AI_SYSTEM_INSTRUCTIONS,
                    system_instructions_to_semconv(run.system_prompt),
                )
            if run.input_messages:
                self._set_content_attr(
                    run.agent_span,
                    GEN_AI_INPUT_MESSAGES,
                    messages_to_semconv(run.input_messages),
                )
            # ``event.messages`` is the run's full new_messages list —
            # for a normal ``agent.prompt(...)`` it includes the user
            # prompt(s) as well as the generated assistant/tool replies.
            # ``gen_ai.output.messages`` should be model output only, so
            # use the accumulator that tracks just assistant + tool
            # results.
            if run.output_messages:
                self._set_content_attr(
                    run.agent_span,
                    GEN_AI_OUTPUT_MESSAGES,
                    messages_to_semconv(run.output_messages),
                )
        # Final sweep for normal-end (and agent-caught error) paths.
        # Cancellation skips this hook entirely — see
        # ``_sweep_tool_span_tokens`` docstring.
        self._sweep_tool_span_tokens(run)
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
                # Propagate run_id so every child span carries it —
                # OTel does NOT inherit attributes from parent spans, and
                # JsonlSpanExporter shards by ``cubepi.run_id`` per span.
                CUBEPI_RUN_ID: run.run_id,
            },
        )
        run.turn_terminated_by_tool = False
        run.turn_input_messages = []
        run.turn_output_messages = []

    def _on_turn_end(self, event: TurnEndEvent) -> None:
        run = self._run
        if run is None or run.turn_span is None:
            return
        msg = event.message
        # Track output messages: assistant + any tool_results from this turn.
        run.turn_output_messages.append(msg)
        run.output_messages.append(msg)
        for tr in getattr(event, "tool_results", []) or []:
            run.turn_output_messages.append(tr)
            run.output_messages.append(tr)
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
        # Content: record per-turn input/output messages if enabled.
        if self._record_content:
            if run.turn_input_messages:
                self._set_content_attr(
                    run.turn_span,
                    GEN_AI_INPUT_MESSAGES,
                    messages_to_semconv(run.turn_input_messages),
                )
            if run.turn_output_messages:
                self._set_content_attr(
                    run.turn_span,
                    GEN_AI_OUTPUT_MESSAGES,
                    messages_to_semconv(run.turn_output_messages),
                )
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
            # Per-span run_id so JsonlSpanExporter shards correctly.
            CUBEPI_RUN_ID: run.run_id,
        }
        # NOTE: gen_ai.tool.call.arguments is opt-in content recorded
        # after start_span so the redaction hook can run before the
        # attribute is committed.
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
        # Expose this execute_tool span (and its owning provider) to
        # ``cubepi.mcp._tracing`` via a per-task contextvar so an MCP
        # tool call running inside the AgentTool body inherits the
        # right parent. Per-task scoping avoids the global-dict
        # collision when concurrent agents reuse the same tool_call_id
        # (e.g. Faux/OpenAI-style providers minting ``tc1`` per
        # conversation — codex round-8). The recorder stashes the
        # contextvar reset token on _RunState to undo on exec_end.
        try:
            from cubepi.mcp import _tracing as _mcp_tracing

            cv_token = _mcp_tracing.register_tool_span(
                event.tool_call_id,
                span,
                provider=self._tracer._provider,
            )
            run.tool_span_tokens[event.tool_call_id] = cv_token
        except ImportError:  # pragma: no cover — mcp module always present
            pass
        if self._record_content and event.args is not None:
            self._set_content_attr(
                span, GEN_AI_TOOL_CALL_ARGUMENTS, _coerce_dict(event.args)
            )

    def _on_tool_exec_end(self, event: ToolExecutionEndEvent) -> None:
        run = self._run
        if run is None:
            return
        span = run.tool_spans.pop(event.tool_call_id, None)
        cv_token = run.tool_span_tokens.pop(event.tool_call_id, None)
        try:
            from cubepi.mcp import _tracing as _mcp_tracing

            _mcp_tracing.unregister_tool_span(cv_token)
        except ImportError:  # pragma: no cover — mcp module always present
            pass
        if span is None:
            return
        span.set_attribute(CUBEPI_TOOL_IS_ERROR, event.is_error)
        if self._record_content and event.result is not None:
            self._set_content_attr(
                span, GEN_AI_TOOL_CALL_RESULT, _coerce_dict(event.result)
            )
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
        # Count user-provided + tool-result messages as input. We also
        # accumulate them for content recording (only emitted on spans
        # when ``record_content=True``).
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
            run.input_messages.append(msg)
            run.turn_input_messages.append(msg)
            run.transcript.append(msg)

    def _on_message_end(self, event: MessageEndEvent) -> None:
        run = self._run
        if run is None:
            return
        # Append assistant messages to the transcript so later chat
        # spans see them in their ``gen_ai.input.messages``.
        #
        # tool_result messages are intentionally excluded here:
        # ``execute_tool_calls`` emits both MessageStart AND MessageEnd
        # for the same tool_result, and ``_on_message_start`` already
        # appends ToolResultMessage to the transcript. Appending again
        # here would produce two identical ``tool`` entries in the
        # next chat span's input messages even though the provider
        # context contains only one. Assistant messages have no
        # MessageStart in the cubepi event flow, so they only land
        # here.
        msg = event.message
        if getattr(msg, "role", None) == "assistant":
            run.transcript.append(msg)

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
            # Per-span run_id so JsonlSpanExporter shards correctly.
            CUBEPI_RUN_ID: run.run_id,
        }
        # Pull StreamOptions-derived params from the payload where the
        # provider exposed them as final kwargs.
        #
        # Note: OpenAI Responses uses ``max_output_tokens`` in place of
        # the chat-completions ``max_tokens`` field — accept either so
        # the resulting ``gen_ai.request.max_tokens`` attribute is
        # consistent across providers (codex overall-review MINOR).
        for key, attr in (
            ("max_tokens", GEN_AI_REQUEST_MAX_TOKENS),
            ("max_output_tokens", GEN_AI_REQUEST_MAX_TOKENS),
            ("temperature", GEN_AI_REQUEST_TEMPERATURE),
            ("top_p", GEN_AI_REQUEST_TOP_P),
        ):
            if key in payload and payload[key] is not None:
                attrs[attr] = payload[key]

        # OpenAI provider-specific request fields (semconv §openai).
        if payload.get("service_tier"):
            attrs[OPENAI_REQUEST_SERVICE_TIER] = payload["service_tier"]

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

        # Content (record_content=True): record system instructions,
        # input messages, tool definitions, and the raw wire payload.
        if self._record_content:
            sys_prompt = _extract_system_prompt(payload)
            if sys_prompt:
                run.system_prompt = sys_prompt
                sys_payload = system_instructions_to_semconv(sys_prompt)
                if sys_payload:
                    self._set_content_attr(
                        chat_span, GEN_AI_SYSTEM_INSTRUCTIONS, sys_payload
                    )
            if run.transcript:
                # chat input = full chronological context the provider
                # is about to receive (user prompts + earlier
                # assistant/tool-result turns).
                self._set_content_attr(
                    chat_span,
                    GEN_AI_INPUT_MESSAGES,
                    messages_to_semconv(run.transcript),
                )
            tool_defs = tool_definitions_to_semconv(payload)
            if tool_defs:
                run.chat_tool_definitions = tool_defs
                self._set_content_attr(chat_span, GEN_AI_TOOL_DEFINITIONS, tool_defs)
            # Raw wire payload — large; gated by record_content.
            self._set_content_attr(chat_span, CUBEPI_LLM_RAW_REQUEST, payload)

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
                if self._record_content:
                    # Output messages on chat span: the assembled body
                    # itself (provider-shaped). Record both the raw
                    # response (JSON dict) for backends that prefer it,
                    # and the normalized output messages where derivable.
                    self._set_content_attr(span, CUBEPI_LLM_RAW_RESPONSE, body)
            # Cooperative abort: providers may finish the response
            # listener with ``exc is None`` and a body whose finish
            # reason is ``"aborted"`` (faux + the agent's signal path).
            # Mark the chat span aborted in that case so it matches
            # the contract for hard cancels.
            if body is not None and _body_is_aborted(body):
                span.set_attribute(CUBEPI_ABORTED, True)
                span.set_attribute(ERROR_TYPE, "cubepi.aborted")

            if exc is None and body is None and self._agent_signal_is_set():
                # Provider abort branches (anthropic/openai/openai_responses)
                # return from the stream before assembling a body, so the
                # response listener is invoked as (None, model, None). The
                # cooperative-abort marker check above requires a body to
                # inspect, so without this branch the chat span would close
                # UNSET — out of sync with the turn/root which TurnEnd
                # marks aborted. Match the cancellation contract: leave
                # Status UNSET, set cubepi.aborted + error.type.
                #
                # Gate on ``agent._active_signal.is_set()`` so we don't
                # mistake provider-side non-abort fallbacks for aborts:
                # OpenAIResponsesProvider, for example, finalizes a
                # stream that ends without ``response.completed`` by
                # firing response listeners with body=None / exc=None
                # — a successful (if incomplete) completion that must
                # NOT be marked aborted (codex P2 follow-up on PR #87).
                span.set_attribute(CUBEPI_ABORTED, True)
                span.set_attribute(ERROR_TYPE, "cubepi.aborted")
            elif exc is None:
                # Healthy completion — leave Status UNSET per OTel guidance.
                pass
            elif _is_cancelled_error(exc):
                # Hard cancel: not a failure. Mark and leave UNSET.
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

    def _set_content_attr(self, span: Any, key: str, value: Any) -> None:
        """Set a content attribute on ``span`` with optional redaction.

        ``value`` is JSON-serialized (OTel attribute values can only be
        simple scalars/sequences, not dicts/lists-of-dicts). When the
        Tracer has a ``redact`` hook, the raw value is passed through
        it first; returning ``None`` skips the attribute entirely.
        """
        if not self._record_content:
            return
        if self._redact is not None:
            try:
                value = self._redact(key, value)
            except Exception:
                # Redactor must never crash the recorder.
                return
            if value is None:
                return
        if isinstance(value, str):
            span.set_attribute(key, value)
        else:
            span.set_attribute(key, serialize_for_attribute(value))

    def _find_tool(self, name: str):
        """Look up an :class:`AgentTool` by name on the attached agent.

        Read directly from the agent's tool registry; the Recorder is
        attached for the agent's lifetime so the reference is stable.
        Returns ``None`` if the agent exposes no tool list or the name
        isn't registered.
        """
        agent = self._agent
        if agent is None:
            return None
        tools = getattr(agent, "_state", None)
        if tools is None:
            return None
        tool_list = getattr(tools, "_tools", None) or []
        for t in tool_list:
            if getattr(t, "name", None) == name:
                return t
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
            # Provider-specific (semconv §openai).
            span.set_attribute(OPENAI_API_TYPE, "chat_completions")
            if body.get("system_fingerprint"):
                span.set_attribute(
                    OPENAI_RESPONSE_SYSTEM_FINGERPRINT, body["system_fingerprint"]
                )
            if body.get("service_tier"):
                span.set_attribute(OPENAI_RESPONSE_SERVICE_TIER, body["service_tier"])
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
            # Provider-specific (semconv §openai).
            span.set_attribute(OPENAI_API_TYPE, "responses")
            if body.get("service_tier"):
                span.set_attribute(OPENAI_RESPONSE_SERVICE_TIER, body["service_tier"])
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
    """Best-effort tool-name extractor across provider payload shapes.

    Three shapes seen in the wild:

    - Anthropic / cubepi.AgentTool: top-level ``{"name": ...}``
    - OpenAI Responses: top-level ``{"name": ..., "type": "function"}``
    - OpenAI Chat: nested ``{"type": "function", "function": {"name": ...}}``

    Falls back to an attribute lookup for plain objects. Without the
    nested-function branch, OpenAI Chat tool lists produced
    ``[""]`` on the root span attribute (codex overall-review MINOR).
    """
    if isinstance(t, dict):
        top = t.get("name")
        if top:
            return str(top)
        fn = t.get("function")
        if isinstance(fn, dict):
            nested = fn.get("name")
            if nested:
                return str(nested)
        return ""
    name = getattr(t, "name", None)
    return str(name) if name else ""


def _coerce_dict(value: Any) -> Any:
    """Coerce a pydantic model or arbitrary object into a JSON-safe
    structure. Returns the value as-is if already a dict/list/scalar.
    """
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "__dict__") and not isinstance(value, type):
        # Best-effort dataclass-like coercion.
        try:
            return dict(vars(value))
        except Exception:
            pass
    return value


def _is_cancelled_error(exc: BaseException) -> bool:
    import asyncio

    return isinstance(exc, asyncio.CancelledError)


def _body_is_aborted(body: dict) -> bool:
    """Detect cooperative-abort signal in an assembled provider body.

    Cubepi providers normalize aborts into ``stop_reason = "aborted"``
    on the assistant message (the body's ``stop_reason`` for Anthropic-
    shaped bodies; for OpenAI bodies the equivalent surfaces via the
    aborted partial response which carries no terminal finish_reason
    of its own — handled separately on the agent-side at TurnEnd).
    """
    if body.get("stop_reason") == "aborted":
        return True
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict) and first.get("finish_reason") == "aborted":
            return True
    return False
