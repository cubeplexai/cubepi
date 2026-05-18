"""``Meter`` — OTel histograms for cubepi.tracing.

Mirrors :class:`cubepi.tracing.tracer.Tracer` in shape: construct once,
``attach(agent)`` to wire metrics emission to the agent's event stream
+ provider listeners.

Histograms emitted (per OTel GenAI semconv):

- ``gen_ai.client.operation.duration`` (seconds) — on chat /
  execute_tool / invoke_agent span close
- ``gen_ai.client.token.usage`` ({token}) — one observation per token
  type on chat close (``input`` and ``output``)
- ``gen_ai.client.operation.time_to_first_chunk`` (seconds) — on chat
  close, when a first chunk was observed
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    MetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource

import cubepi
from cubepi.tracing.schema import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_RESPONSE_MODEL,
    OP_CHAT,
    OP_EXECUTE_TOOL,
    OP_INVOKE_AGENT,
    SCHEMA_URL,
    SCOPE_NAME,
    map_provider_name,
)

if TYPE_CHECKING:
    from cubepi.agent.agent import Agent
    from cubepi.providers.base import Model


#: Recommended GenAI metric bucket boundaries per the semconv spec
#: (seconds). Used for both \\``operation.duration\\`` and the latency
#: histograms.
_DEFAULT_DURATION_BUCKETS: tuple[float, ...] = (
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1,
    2,
    5,
    10,
    30,
    60,
    120,
    300,
)


class Meter:
    """Emit OTel GenAI histograms alongside the cubepi Tracer.

    Construct with a list of :class:`MetricExporter` instances (e.g. the
    OTLP metric exporter); call :meth:`attach` to start emitting.

    Example::

        from cubepi.tracing import Tracer, Meter
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )

        tracer = Tracer(service_name="my-bot", exporters=[...])
        meter = Meter(
            resource=tracer.resource,
            exporters=[OTLPMetricExporter(endpoint="http://collector:4318/v1/metrics")],
        )
        tracer.attach(agent)
        meter.attach(agent)
    """

    def __init__(
        self,
        *,
        resource: Resource | None = None,
        exporters: list[MetricExporter] | None = None,
        export_interval_millis: int = 60_000,
    ) -> None:
        self._resource = resource or Resource.create(
            {"service.name": "cubepi"}, schema_url=SCHEMA_URL
        )
        readers = []
        for exporter in exporters or []:
            readers.append(
                PeriodicExportingMetricReader(
                    exporter=exporter,
                    export_interval_millis=export_interval_millis,
                )
            )
        self._provider = MeterProvider(resource=self._resource, metric_readers=readers)
        self._otel_meter = self._provider.get_meter(
            name=SCOPE_NAME,
            version=cubepi.__version__ if hasattr(cubepi, "__version__") else None,
            schema_url=SCHEMA_URL,
        )
        self._duration_hist = self._otel_meter.create_histogram(
            "gen_ai.client.operation.duration",
            unit="s",
            description="GenAI operation duration",
            explicit_bucket_boundaries_advisory=list(_DEFAULT_DURATION_BUCKETS),
        )
        self._token_hist = self._otel_meter.create_histogram(
            "gen_ai.client.token.usage",
            unit="{token}",
            description="Number of input/output tokens used",
        )
        self._ttfc_hist = self._otel_meter.create_histogram(
            "gen_ai.client.operation.time_to_first_chunk",
            unit="s",
            description="Time elapsed from client request to first response chunk",
            explicit_bucket_boundaries_advisory=list(_DEFAULT_DURATION_BUCKETS),
        )
        self._shutdown = False
        # Per-run state for timing.
        self._chat_open_ns: int | None = None
        self._chat_first_chunk_ns: int | None = None
        self._chat_attrs: dict[str, Any] = {}
        self._tool_open_ns: dict[str, int] = {}
        self._tool_attrs: dict[str, dict[str, Any]] = {}
        self._agent_open_ns: int | None = None
        self._agent_attrs: dict[str, Any] = {}

    # -- public API ---------------------------------------------------

    @property
    def otel_meter(self) -> Any:
        """The underlying ``opentelemetry.metrics.Meter`` — exposed so
        callers can register their own instruments."""
        return self._otel_meter

    def attach(self, agent: "Agent") -> Callable[[], None]:
        """Subscribe to ``agent`` and start emitting metrics."""
        from cubepi.providers.base import BaseProvider

        unsub_agent = agent.subscribe(self._on_agent_event)
        provider = getattr(agent, "_provider", None) or getattr(agent, "provider", None)
        detachers: list[Callable[[], None]] = []
        if isinstance(provider, BaseProvider):
            detachers.append(provider.subscribe_request(self._on_provider_request))
            detachers.append(provider.subscribe_chunk(self._on_provider_chunk))
            detachers.append(provider.subscribe_response(self._on_provider_response))

        def detach() -> None:
            unsub_agent()
            for d in detachers:
                try:
                    d()
                except Exception:
                    pass

        return detach

    async def force_flush(self, timeout_seconds: float = 30.0) -> bool:
        return self._provider.force_flush(timeout_millis=int(timeout_seconds * 1000))

    async def shutdown(self, timeout_seconds: float = 30.0) -> None:
        if self._shutdown:
            return
        await self.force_flush(timeout_seconds=timeout_seconds)
        self._provider.shutdown()
        self._shutdown = True

    async def __aenter__(self) -> "Meter":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.shutdown()

    # -- event handlers -----------------------------------------------

    async def _on_agent_event(self, event: Any, signal: Any | None = None) -> None:
        del signal
        try:
            type_name = getattr(event, "type", "")
            if type_name == "agent_start":
                self._agent_open_ns = time.time_ns()
                self._agent_attrs = {GEN_AI_OPERATION_NAME: OP_INVOKE_AGENT}
            elif type_name == "agent_end":
                if self._agent_open_ns is not None:
                    duration = (time.time_ns() - self._agent_open_ns) / 1e9
                    self._duration_hist.record(duration, self._agent_attrs)
                self._agent_open_ns = None
                self._agent_attrs = {}
            elif type_name == "tool_execution_start":
                self._tool_open_ns[event.tool_call_id] = time.time_ns()
                self._tool_attrs[event.tool_call_id] = {
                    GEN_AI_OPERATION_NAME: OP_EXECUTE_TOOL,
                }
            elif type_name == "tool_execution_end":
                start = self._tool_open_ns.pop(event.tool_call_id, None)
                attrs = self._tool_attrs.pop(event.tool_call_id, None)
                if start is not None and attrs is not None:
                    duration = (time.time_ns() - start) / 1e9
                    self._duration_hist.record(duration, attrs)
        except Exception:
            pass

    def _on_provider_request(self, payload: dict, model: "Model") -> None:
        provider_name = map_provider_name(model.provider)
        self._chat_open_ns = time.time_ns()
        self._chat_first_chunk_ns = None
        self._chat_attrs = {
            GEN_AI_OPERATION_NAME: OP_CHAT,
            GEN_AI_PROVIDER_NAME: provider_name,
        }
        # Stamp the agent + tool attrs with provider too, so all metrics
        # from this run can be filtered by provider in one shot.
        if self._agent_attrs:
            self._agent_attrs.setdefault(GEN_AI_PROVIDER_NAME, provider_name)
        for _, attrs in self._tool_attrs.items():
            attrs.setdefault(GEN_AI_PROVIDER_NAME, provider_name)

    def _on_provider_chunk(self, event: Any, model: "Model") -> None:
        del model
        if self._chat_first_chunk_ns is not None or self._chat_open_ns is None:
            return
        ev_type = getattr(event, "type", "")
        if ev_type in ("text_delta", "thinking_delta", "toolcall_delta"):
            self._chat_first_chunk_ns = time.time_ns()

    def _on_provider_response(
        self,
        body: dict | None,
        model: "Model",
        exc: BaseException | None,
    ) -> None:
        if self._chat_open_ns is None:
            return
        try:
            now = time.time_ns()
            duration = (now - self._chat_open_ns) / 1e9
            response_model: str | None = None
            if isinstance(body, dict):
                response_model = body.get("model") or None
            attrs = dict(self._chat_attrs)
            if response_model:
                attrs[GEN_AI_RESPONSE_MODEL] = response_model
            self._duration_hist.record(duration, attrs)
            if self._chat_first_chunk_ns is not None:
                ttfc = (self._chat_first_chunk_ns - self._chat_open_ns) / 1e9
                self._ttfc_hist.record(ttfc, attrs)
            # Token usage — one observation per type.
            if isinstance(body, dict) and not exc:
                self._record_token_usage(body, attrs)
        finally:
            self._chat_open_ns = None
            self._chat_first_chunk_ns = None
            self._chat_attrs = {}

    def _record_token_usage(self, body: dict, attrs: dict) -> None:
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
        if not usage:
            return
        # Anthropic-shaped: input_tokens / output_tokens / cache_*_input_tokens
        if "input_tokens" in usage or "output_tokens" in usage:
            input_t = int(usage.get("input_tokens", 0) or 0)
            cache_r = int(usage.get("cache_read_input_tokens", 0) or 0)
            cache_c = int(usage.get("cache_creation_input_tokens", 0) or 0)
            total_input = input_t + cache_r + cache_c
            output_t = int(usage.get("output_tokens", 0) or 0)
            if total_input:
                self._token_hist.record(
                    total_input, {**attrs, "gen_ai.token.type": "input"}
                )
            if output_t:
                self._token_hist.record(
                    output_t, {**attrs, "gen_ai.token.type": "output"}
                )
            return
        # OpenAI chat.completion shape
        if "prompt_tokens" in usage or "completion_tokens" in usage:
            prompt_t = int(usage.get("prompt_tokens", 0) or 0)
            compl_t = int(usage.get("completion_tokens", 0) or 0)
            if prompt_t:
                self._token_hist.record(
                    prompt_t, {**attrs, "gen_ai.token.type": "input"}
                )
            if compl_t:
                self._token_hist.record(
                    compl_t, {**attrs, "gen_ai.token.type": "output"}
                )
