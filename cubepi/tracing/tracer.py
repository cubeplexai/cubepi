"""``Tracer`` config class — the user-facing entry point.

Builds an SDK :class:`opentelemetry.sdk.trace.TracerProvider` with a
pinned schema URL, attaches one ``BatchSpanProcessor`` per exporter,
and exposes ``attach(agent)`` to wire the cubepi :class:`Recorder` into
the agent's event stream.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Callable

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
        resource: Resource | None = None,
    ) -> None:
        if record_content:
            raise NotImplementedError(
                "record_content=True is Phase 2 of cubepi.tracing. "
                "Construct with record_content=False (default) for MVP."
            )

        self._record_content = record_content
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
            instrumenting_library_version=cubepi.__version__
            if hasattr(cubepi, "__version__")
            else None,
            schema_url=SCHEMA_URL,
        )
        self._shutdown = False

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

    def attach(self, agent: "Agent") -> Callable[[], None]:
        """Wire the cubepi recorder to ``agent``.

        Returns a ``detach()`` callable that unsubscribes every hook
        installed by this attach, then calls :meth:`force_flush` to
        ensure spans produced by the agent's runs are persisted before
        the caller proceeds.
        """
        from cubepi.tracing.recorder import Recorder

        recorder = Recorder(self, record_content=self._record_content)
        return recorder.attach(agent)

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
