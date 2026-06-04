"""``Tracer`` config class — the user-facing entry point.

Builds an SDK :class:`opentelemetry.sdk.trace.TracerProvider` with a
pinned schema URL, attaches one ``BatchSpanProcessor`` per exporter,
and exposes ``attach(agent)`` to wire the cubepi :class:`Recorder` into
the agent's event stream.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.trace import (
    NonRecordingSpan,
    SpanContext,
    TraceFlags,
    set_span_in_context,
)

import cubepi
from cubepi.tracing.schema import SCHEMA_URL, SCOPE_NAME

if TYPE_CHECKING:
    from cubepi.agent.agent import Agent
    from cubepi.providers.base import BaseProvider, Message, Model


class _OneShotSession:
    """Returned by :meth:`Tracer.oneshot` to issue a single traced LLM call.

    The session's provider listeners are live for the lifetime of the
    ``async with tracer.oneshot(...)`` block.  Call :meth:`generate` once
    inside that block to run the LLM and get the response text.
    """

    def __init__(
        self,
        provider: "BaseProvider",
        model: "Model",
        run: Any,
    ) -> None:
        self._provider = provider
        self._model = model
        self._run = run

    async def generate(
        self,
        *,
        system: str,
        messages: "list[Message]",
        max_output_tokens: int,
    ) -> str:
        """Run one non-tool-using completion and return the full text.

        The provider listeners wired by :meth:`Tracer.oneshot` will record
        a ``chat`` child span covering this call automatically.
        """
        import asyncio as _asyncio

        from cubepi.providers.base import (
            AssistantMessage as _AssistantMessage,
        )
        from cubepi.providers.base import (
            StreamOptions as _StreamOptions,
        )
        from cubepi.providers.base import (
            TextContent as _TextContent,
        )

        # Seed transcript so record_content captures input messages on the
        # chat span, and stash system/input on the run so the oneshot
        # finally block can also stamp them on the root invoke_agent span
        # (no AgentEnd event fires here to do it automatically).
        self._run.transcript = list(messages)
        self._run.input_messages = list(messages)
        self._run.system_prompt = system

        # Cancellation signal forwarded to the provider so a cancelled
        # consumer (e.g. asyncio.wait_for timeout) tears down the
        # producer task instead of leaking it past the oneshot context.
        # Mirrors the agent loop's ``StreamOptions(signal=...)`` use.
        abort_signal = _asyncio.Event()

        model = self._model.model_copy(update={"max_tokens": max_output_tokens})
        stream = await self._provider.stream(
            model=model,
            messages=messages,
            system_prompt=system,
            options=_StreamOptions(signal=abort_signal),
        )
        parts: list[str] = []
        error_msg: str | None = None
        try:
            async for evt in stream:
                if evt.type == "text_delta" and evt.delta:
                    parts.append(evt.delta)
                elif evt.type == "error":
                    error_msg = evt.error_message or "oneshot generation failed"
                    break
                elif evt.type == "done":
                    break
        except BaseException:
            # On cancellation/timeout, tell the producer to stop so it
            # doesn't keep running after we exit. ``stream.result()``
            # below then surfaces the (now-completed) producer's state.
            abort_signal.set()
            raise
        # Wait for the producer task to finish — its ``finally`` block
        # invokes the response listeners (including the recorder's
        # ``_on_provider_response`` that closes the chat span and records
        # usage). Without this, breaking on ``done`` lets us race ahead
        # and exit the oneshot context before the chat span is closed,
        # so the cleanup sweep would mark it ``cubepi.aborted`` and the
        # trace would be missing usage / response data. Mirrors the
        # ``await stream.result()`` pattern in cubepi.agent.loop.
        try:
            await stream.result()
        except Exception:
            # The producer's exception was already surfaced via the
            # error event above (or the listener has recorded the error
            # on the chat span). Swallow here so we report it uniformly
            # via ``error_msg``.
            pass
        if error_msg is not None:
            raise RuntimeError(error_msg)
        text = "".join(parts)
        # Stash output as a single assistant message so the oneshot finally
        # block can stamp gen_ai.output.messages on the root span.
        if text:
            self._run.output_messages = [
                _AssistantMessage(content=[_TextContent(text=text)])
            ]
        return text


class Tracer:
    """Attaches OTel-compatible tracing to a cubepi :class:`Agent`.

    Construct once per process (or per service). Each ``attach(agent)``
    call wires the cubepi recorder to the agent's event stream and
    provider listener registry; the returned callable detaches.

    Example
    -------
    ::

        from cubepi.tracing import Tracer
        from cubepi.tracing.exporters import JsonlSpanExporter

        tracer = Tracer(
            service_name="my-bot",
            agent_name="coding-agent",
            exporters=[JsonlSpanExporter(directory="./cubepi-traces")],
        )
        detach = tracer.attach(agent)
        try:
            ...
        finally:
            detach()
            await tracer.shutdown()
    """

    def __init__(
        self,
        *,
        service_name: str | None = None,
        service_version: str | None = None,
        service_namespace: str | None = None,
        service_instance_id: str | None = None,
        deployment_environment: str | None = None,
        agent_name: str | None = None,
        agent_id: str | None = None,
        agent_description: str | None = None,
        agent_version: str | None = None,
        exporters: list[SpanExporter] | None = None,
        record_content: bool = False,
        record_stream: bool = False,
        stream_dir: "str | Path | None" = None,
        redact: "Callable[[str, Any], Any] | None" = None,
        resource: Resource | None = None,
        atexit_flush: bool = True,
        atexit_flush_timeout_seconds: float = 5.0,
    ) -> None:
        self._record_content = record_content
        self._record_stream = record_stream
        self._stream_dir = (
            Path(stream_dir) if isinstance(stream_dir, str) else stream_dir
        )
        self._redact = redact
        self._resource = resource or _build_resource(
            service_name=service_name,
            service_version=service_version,
            service_namespace=service_namespace,
            service_instance_id=service_instance_id,
            deployment_environment=deployment_environment,
            agent_name=agent_name,
            agent_id=agent_id,
            agent_description=agent_description,
            agent_version=agent_version,
        )
        self._provider = TracerProvider(resource=self._resource)
        self._processors: list[BatchSpanProcessor] = []
        for exporter in exporters or []:
            proc = BatchSpanProcessor(exporter)
            self._provider.add_span_processor(proc)
            self._processors.append(proc)

        self._otel_tracer = self._provider.get_tracer(
            instrumenting_module_name=SCOPE_NAME,
            instrumenting_library_version=cubepi.__version__,
            schema_url=SCHEMA_URL,
        )
        self._shutdown = False
        self._atexit_flush_timeout_ms = int(atexit_flush_timeout_seconds * 1000)
        self._atexit_unregister: Callable[[], None] | None = None
        if atexit_flush:
            # Safety net for callers that forget ``await tracer.shutdown()``:
            # at process exit, sync-flush any buffered spans through
            # BatchSpanProcessor. Idempotent — ``shutdown()`` flips
            # ``self._shutdown`` so this becomes a no-op when the user
            # cleaned up properly. Matches the Traceloop SDK pattern.
            #
            # Limitations: ``atexit`` does NOT run on SIGKILL, kernel
            # OOM kill, or os._exit(). For guaranteed delivery under
            # those conditions, use SimpleSpanProcessor instead.
            atexit.register(self._atexit_flush)
            self._atexit_unregister = lambda: atexit.unregister(self._atexit_flush)

    # -- public API ---------------------------------------------------

    @property
    def resource(self) -> Resource:
        return self._resource

    @property
    def otel_tracer(self) -> Any:
        """The underlying ``opentelemetry.trace.Tracer`` instance.

        Exposed so callers can write their own spans alongside the
        cubepi-generated ones if desired.
        """
        return self._otel_tracer

    @property
    def redact(self) -> "Callable[[str, Any], Any] | None":
        """Optional ``(key, value) -> value`` filter applied at every
        ``set_attribute`` site for content attributes.

        Return ``None`` to drop the attribute entirely. Return a value
        of the same type to substitute. Useful for redacting PII inside
        ``gen_ai.input.messages`` and friends before they leave the
        process.
        """
        return self._redact

    def attach(self, agent: "Agent") -> Callable[[], Any]:
        """Wire the cubepi recorder to ``agent``.

        Returns a ``detach()`` callable. When invoked:

        - Synchronously: unsubscribes every hook, closes any spans
          still open from a cancelled run, sweeps MCP tool-span
          registrations — observable on the next line.
        - Schedules a flush on the running event loop and returns the
          resulting ``asyncio.Task`` so callers can ``await detach()``
          to block until buffered spans have been exported. Outside
          an async context returns ``None`` — call
          ``await tracer.shutdown()`` separately to flush.

        Either ``await detach()`` or ``await tracer.shutdown()`` (or
        both) must be used in the caller's ``finally`` block; the
        synchronous ``detach()`` alone does not guarantee that ended
        spans have left ``BatchSpanProcessor`` (codex overall-review
        MAJOR).

        Also registers this Tracer's private TracerProvider with
        :mod:`cubepi.mcp._tracing` so MCP CLIENT spans flow through
        the same exporters — without this step, the MCP module falls
        back to the OTel global default (a no-op provider when the
        caller didn't separately call ``trace.set_tracer_provider``).
        The detach callable unregisters.
        """
        from cubepi.tracing.recorder import Recorder

        recorder = Recorder(
            self,
            record_content=self._record_content,
            record_stream=self._record_stream,
            stream_dir=self._stream_dir,
            redact=self._redact,
        )
        recorder_detach = recorder.attach(agent)

        # Route MCP spans through this Tracer's provider. Capture the
        # token returned by register_provider so the detach below only
        # removes this attach's registration — multiple attaches of the
        # same Tracer to different agents must not clobber each other
        # (see codex round-6 review on PR #86).
        #
        # Keep attach all-or-nothing: if MCP registration fails for any reason
        # other than the optional module being absent, undo the recorder attach
        # before re-raising so we don't leak its subscriptions.
        mcp_token: object | None = None
        try:
            from cubepi.mcp import _tracing as mcp_tracing

            mcp_token = mcp_tracing.register_provider(self._provider)
        except ImportError:  # pragma: no cover — mcp module always present
            mcp_tracing = None
        except BaseException:
            try:
                recorder_detach()
            except Exception:
                pass
            raise

        def detach():
            # Forward the recorder's detach return — when called from
            # an async context it's an ``asyncio.Task`` for the
            # flush; callers can ``await detach()`` to wait for
            # buffered spans to land (codex overall-review MAJOR).
            #
            # Unregister MCP in a ``finally`` so a raising recorder detach
            # can't leak the provider registration.
            try:
                return recorder_detach()
            finally:
                if mcp_tracing is not None and mcp_token is not None:
                    try:
                        mcp_tracing.unregister_provider(mcp_token)
                    except Exception:
                        pass

        return detach

    async def force_flush(self, timeout_seconds: float = 30.0) -> bool:
        """Block until all currently buffered spans are exported.

        Returns ``False`` on timeout.
        """
        timeout_millis = int(timeout_seconds * 1000)
        return self._provider.force_flush(timeout_millis=timeout_millis)

    async def shutdown(self, timeout_seconds: float = 30.0) -> None:
        """Flush and close all exporters. Tracer is unusable after this."""
        if self._shutdown:
            return
        # SpanProcessor.shutdown is sync; flush first to bound wait.
        await self.force_flush(timeout_seconds=timeout_seconds)
        self._provider.shutdown()
        self._shutdown = True
        # Remove the atexit hook now that the user has cleaned up
        # explicitly — keeps the interpreter's atexit table from
        # growing unbounded when many Tracers are constructed (e.g.
        # in tests) and prevents a redundant flush at interpreter
        # shutdown.
        if self._atexit_unregister is not None:
            try:
                self._atexit_unregister()
            except Exception:
                pass
            self._atexit_unregister = None

    def _atexit_flush(self) -> None:
        """Process-exit safety net: sync-flush any buffered spans.

        Runs in the interpreter's atexit handler chain — no event
        loop available, so we call ``provider.force_flush``
        synchronously (BatchSpanProcessor's flush is sync; it blocks
        the calling thread until the queue drains or the timeout
        elapses).

        Idempotent with ``shutdown()`` — if the user cleaned up
        explicitly, ``_shutdown`` is True and we no-op.
        """
        if self._shutdown:
            return
        try:
            self._provider.force_flush(timeout_millis=self._atexit_flush_timeout_ms)
        except Exception:
            # atexit hooks must never raise — would corrupt interpreter
            # shutdown for everything else.
            pass

    @contextlib.asynccontextmanager
    async def attached(self, agent: "Agent") -> AsyncIterator["Tracer"]:
        """RAII wrapper around :meth:`attach`.

        ``async with`` enters by attaching the recorder, exits by
        running detach and (in an async context) awaiting its returned
        flush ``Task``. Equivalent to::

            detach = tracer.attach(agent)
            try:
                ...
            finally:
                result = detach()
                if result is not None:  # async context: it's a Task
                    await result

        Use this instead of the bare ``attach`` / ``finally: detach()``
        pattern when you want the cleanup tied to a single ``async
        with`` block. Combines with :class:`Tracer`'s own context
        manager:

        ::

            async with Tracer(...) as tracer, tracer.attached(agent):
                await agent.prompt("...")
            # auto: detach (closes cancelled spans) + shutdown (flush + close)
        """
        import sys

        detach = self.attach(agent)
        try:
            yield self
        finally:
            result = detach()
            if result is not None and hasattr(result, "__await__"):
                # Surface flush failures to the caller when the body
                # didn't raise — matches what ``await detach()`` would
                # do manually, so users don't continue past the block
                # believing spans were exported when they weren't.
                # When the body did raise, suppress flush errors so
                # the original (more diagnostic) exception isn't
                # masked by a cleanup failure (codex review on PR #90).
                if sys.exc_info()[1] is None:
                    await result
                else:
                    try:
                        await result
                    except BaseException:
                        pass

    @contextlib.asynccontextmanager
    async def oneshot(
        self,
        *,
        provider: "BaseProvider",
        model: "Model",
        operation: str = "oneshot",
        metadata: "dict[str, str | int | float | bool] | None" = None,
        record_content: bool | None = None,
    ) -> AsyncIterator["_OneShotSession"]:
        """Instrumented one-shot LLM call that produces a trace without a full Agent.

        Creates an ``invoke_agent`` root span (so the ``cubepi trace`` CLI
        indexes it alongside normal agent runs) and wires the provider's
        listeners for the duration of the block, producing a ``chat`` child
        span automatically when :meth:`_OneShotSession.generate` is called.

        ``metadata`` keys are stamped as ``cubepi.metadata.<key>`` attributes on
        the root span and are queryable with
        ``cubepi trace ls --meta <key>=<value>``.

        ``operation`` is recorded in two places:

        - ``cubepi.oneshot.operation`` — for span introspection and dashboards.
        - ``cubepi.metadata.oneshot_operation`` — so the CLI ``--meta`` filter
          (which only reads ``cubepi.metadata.*`` attributes) can reach it::

              uv run cubepi trace ls --meta oneshot_operation=consolidate_memory
              uv run cubepi trace ls --meta conversation_id=conv-123

        Example::

            async with tracer.oneshot(
                provider=provider,
                model=model,
                operation="consolidate_memory",
                metadata={"conversation_id": conv_id, "user_id": user_id},
            ) as session:
                text = await session.generate(
                    system=SYSTEM_PROMPT,
                    messages=[UserMessage(...)],
                    max_output_tokens=1500,
                )
        """
        from opentelemetry.trace import SpanKind

        from cubepi.providers.base import BaseProvider as _BaseProvider
        from cubepi.tracing.recorder import Recorder, _RunState, _active_run
        from cubepi.tracing.schema import (
            CUBEPI_RUN_ID,
            GEN_AI_OPERATION_NAME,
            GEN_AI_PROVIDER_NAME,
            OP_INVOKE_AGENT,
            SPAN_NAME_INVOKE_AGENT,
        )

        do_record = (
            record_content if record_content is not None else self._record_content
        )
        run_id = str(uuid.uuid4())

        root_attrs: dict[str, Any] = {
            GEN_AI_OPERATION_NAME: OP_INVOKE_AGENT,
            CUBEPI_RUN_ID: run_id,
            GEN_AI_PROVIDER_NAME: "cubepi",
            "cubepi.oneshot.operation": operation,
            # Also expose operation under cubepi.metadata.* so the
            # `cubepi trace ls --meta oneshot_operation=...` filter works
            # (the CLI's --meta filter only reads cubepi.metadata.* attrs).
            "cubepi.metadata.oneshot_operation": operation,
        }
        for k, v in (metadata or {}).items():
            root_attrs[f"cubepi.metadata.{k}"] = v

        root_span = self._otel_tracer.start_span(
            name=SPAN_NAME_INVOKE_AGENT,
            kind=SpanKind.INTERNAL,
            attributes=root_attrs,
        )

        # _RunState with turn_span = root_span so chat spans nest directly
        # under the root (no cubepi.turn wrapper — one-shot has no loop).
        run = _RunState(
            run_id=run_id,
            agent_span=root_span,
            turn_span=root_span,
        )

        # Minimal Recorder to handle the 3 provider lifecycle listeners.
        # We don't subscribe agent events — there's no agent loop.
        recorder = Recorder(
            self,
            record_content=do_record,
            record_stream=self._record_stream,
            stream_dir=self._stream_dir,
            redact=self._redact,
        )
        recorder._run = run

        # Open the per-run stream file when record_stream is enabled, so
        # ``_on_provider_chunk`` writes ``<run_id>.stream.jsonl`` for the
        # oneshot the same way the agent path does. Closed in the finally
        # block below.
        if self._record_stream and self._stream_dir is not None:
            try:
                import time as _time

                self._stream_dir.mkdir(parents=True, exist_ok=True)
                stream_path = self._stream_dir / f"{run_id}.stream.jsonl"
                run.stream_file = stream_path.open("w", encoding="utf-8")
                run.stream_start_time = _time.time()
            except Exception:
                pass

        detachers: list[Callable[[], None]] = []
        if isinstance(provider, _BaseProvider):
            try:
                detachers.append(
                    provider.subscribe_request(recorder._on_provider_request)
                )
                detachers.append(provider.subscribe_chunk(recorder._on_provider_chunk))
                detachers.append(
                    provider.subscribe_response(recorder._on_provider_response)
                )
            except BaseException:
                # Unwind any partial subscriptions before re-raising so no
                # dangling listeners are left on the provider.
                for d in detachers:
                    try:
                        d()
                    except Exception:
                        pass
                root_span.end()
                raise

        # Set the per-task active-run gate so provider listeners recognise
        # calls that belong to this oneshot session.
        token = _active_run.set(run)
        try:
            yield _OneShotSession(provider=provider, model=model, run=run)
        except BaseException as _exc:
            # Mark the root span as failed so `cubepi trace ls` (which
            # reads status off the root) lists this run as an error
            # instead of a successful invoke_agent. Mirrors what the
            # agent path does on turn-level errors. Cancellation is
            # tracked as cubepi.aborted (UNSET status, semconv-aligned)
            # rather than ERROR.
            try:
                from opentelemetry.trace import Status, StatusCode

                from cubepi.tracing.schema import CUBEPI_ABORTED, ERROR_TYPE

                import asyncio as _asyncio_for_cancel

                if isinstance(_exc, _asyncio_for_cancel.CancelledError):
                    root_span.set_attribute(CUBEPI_ABORTED, True)
                    root_span.set_attribute(ERROR_TYPE, "cubepi.aborted")
                else:
                    root_span.set_status(Status(StatusCode.ERROR, str(_exc)[:256]))
                    root_span.set_attribute(ERROR_TYPE, type(_exc).__name__)
                root_span.record_exception(_exc)
            except Exception:
                pass
            raise
        finally:
            _active_run.reset(token)
            for d in detachers:
                try:
                    d()
                except Exception:
                    pass
            # Close any chat span left open by a cancelled/timed-out stream.
            # We don't reuse Recorder._close_open_spans because that helper
            # also walks turn_span / agent_span — both of which point at
            # root_span here, so it would mark a successful one-shot's root
            # as aborted before we end it normally below. Tool spans don't
            # exist for one-shot (no tool execution path).
            if run.chat_span is not None:
                try:
                    from cubepi.tracing.schema import CUBEPI_ABORTED, ERROR_TYPE

                    run.chat_span.set_attribute(CUBEPI_ABORTED, True)
                    run.chat_span.set_attribute(ERROR_TYPE, "cubepi.aborted")
                    run.chat_span.end()
                except Exception:
                    pass
                run.chat_span = None
            # Stamp content on the root invoke_agent span. The agent path
            # does this in Recorder._on_agent_end, which never fires for
            # oneshot. Without it, ``cubepi trace ls`` shows a blank input
            # column for one-shot traces because it reads
            # ``gen_ai.input.messages`` off the root.
            if do_record:
                try:
                    from cubepi.tracing.content import (
                        messages_to_semconv,
                        system_instructions_to_semconv,
                    )
                    from cubepi.tracing.schema import (
                        GEN_AI_INPUT_MESSAGES,
                        GEN_AI_OUTPUT_MESSAGES,
                        GEN_AI_SYSTEM_INSTRUCTIONS,
                    )

                    if run.system_prompt:
                        recorder._set_content_attr(
                            root_span,
                            GEN_AI_SYSTEM_INSTRUCTIONS,
                            system_instructions_to_semconv(run.system_prompt),
                        )
                    if run.input_messages:
                        recorder._set_content_attr(
                            root_span,
                            GEN_AI_INPUT_MESSAGES,
                            messages_to_semconv(run.input_messages),
                        )
                    if run.output_messages:
                        recorder._set_content_attr(
                            root_span,
                            GEN_AI_OUTPUT_MESSAGES,
                            messages_to_semconv(run.output_messages),
                        )
                except Exception:
                    pass
            # Close the stream file opened above (mirrors the agent
            # _on_agent_end / _close_open_spans path).
            if run.stream_file is not None:
                try:
                    run.stream_file.close()
                except Exception:
                    pass
                run.stream_file = None
            recorder._run = None
            root_span.end()
            # Flush synchronously so the trace is visible after the block.
            # Unlike attach()'s detach, we have no detach Task to hand back
            # to the caller — if we don't await here, a short-lived
            # asyncio.run() can exit before the BatchSpanProcessor drains.
            try:
                await self.force_flush(timeout_seconds=5.0)
            except Exception:
                pass

    async def __aenter__(self) -> "Tracer":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.shutdown()

    # -- internals used by Recorder -----------------------------------

    def _make_parent_context(
        self,
        *,
        parent_trace_id: int | None = None,
        parent_span_id: int | None = None,
        trace_flags: int = 0x01,
    ):
        """Build an OTel ``Context`` that wraps an inbound W3C
        ``traceparent``. Used by ``run_scope`` to root the cubepi spans
        under a caller-supplied trace.
        """
        if parent_trace_id is None or parent_span_id is None:
            return None
        ctx_obj = SpanContext(
            trace_id=parent_trace_id,
            span_id=parent_span_id,
            is_remote=True,
            trace_flags=TraceFlags(trace_flags),
        )
        return set_span_in_context(NonRecordingSpan(ctx_obj))


def _build_resource(
    *,
    service_name: str | None,
    service_version: str | None,
    service_namespace: str | None,
    service_instance_id: str | None,
    deployment_environment: str | None,
    agent_name: str | None,
    agent_id: str | None,
    agent_description: str | None,
    agent_version: str | None,
) -> Resource:
    attrs: dict[str, Any] = {}
    if service_name is not None:
        attrs["service.name"] = service_name
    if service_namespace is not None:
        attrs["service.namespace"] = service_namespace
    if service_version is not None:
        attrs["service.version"] = service_version
    attrs["service.instance.id"] = service_instance_id or str(uuid.uuid4())
    if deployment_environment is not None:
        attrs["deployment.environment.name"] = deployment_environment
    if agent_name is not None:
        attrs["gen_ai.agent.name"] = agent_name
    if agent_id is not None:
        attrs["gen_ai.agent.id"] = agent_id
    if agent_description is not None:
        attrs["gen_ai.agent.description"] = agent_description
    if agent_version is not None:
        attrs["gen_ai.agent.version"] = agent_version
    # The SDK adds telemetry.sdk.* automatically.
    return Resource.create(attrs, schema_url=SCHEMA_URL)


@contextlib.asynccontextmanager
async def trace(
    tracer: "Tracer | None",
    agent: "Agent",
) -> AsyncIterator[None]:
    """Best-effort tracing scope for one agent run.

    Attaches ``tracer`` to ``agent`` on enter and detaches + flushes its
    buffered spans on exit. Every tracing fault — a failed attach, detach, or
    flush — is logged and swallowed, so tracing can never break or fail the
    work inside the ``async with`` block. Passing ``tracer=None`` makes the
    block a no-op, which lets callers gate tracing on config without branching
    at the call site.

    This does **not** shut the tracer down: the tracer is reusable across runs,
    so build it once (e.g. per process) and call ``await tracer.shutdown()``
    when the owning process stops.

    Unlike :meth:`Tracer.attached`, which surfaces flush failures to the caller,
    this helper swallows them — use it when tracing is auxiliary to the work and
    must never affect its outcome.

    Example
    -------
    ::

        async with trace(tracer, agent):
            await agent.prompt("...")
    """
    if tracer is None:
        yield
        return

    detach: "Callable[[], Any] | None" = None
    try:
        detach = tracer.attach(agent)
    except Exception as exc:  # noqa: BLE001 — tracing must never break the run
        _log_tracing_warning("attach failed; running untraced", exc)
        detach = None

    try:
        yield
    finally:
        if detach is not None:
            try:
                result = detach()
                # In an async context ``detach()`` returns a flush Task; await
                # it so this run's spans are exported before the block exits.
                if result is not None and hasattr(result, "__await__"):
                    await result
            except Exception as exc:  # noqa: BLE001 — flush/detach must never break the run
                _log_tracing_warning("detach/flush failed", exc)


def _log_tracing_warning(message: str, exc: BaseException) -> None:
    """Log a swallowed tracing fault via stdlib ``logging`` — cubepi does not
    depend on loguru. Hosts that use loguru can intercept stdlib logging to
    route these records through it. The log call itself is guarded so a raising
    logging handler can't defeat the best-effort guarantee."""
    try:
        logging.getLogger("cubepi.tracing").warning(
            "cubepi tracing: %s", message, exc_info=exc
        )
    except Exception:  # noqa: BLE001 — logging must never break the run  # pragma: no cover
        pass
