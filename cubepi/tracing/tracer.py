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
