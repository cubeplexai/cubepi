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

import contextlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable

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
    GEN_AI_REQUEST_MODEL,
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
            version=cubepi.__version__,
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

    # -- public API ---------------------------------------------------

    @property
    def otel_meter(self) -> Any:
        """The underlying ``opentelemetry.metrics.Meter`` — exposed so
        callers can register their own instruments."""
        return self._otel_meter

    def attach(self, agent: "Agent") -> Callable[[], None]:
        """Subscribe to ``agent`` and start emitting metrics.

        Each ``attach()`` call gets its own :class:`_MeterState` so
        multiple concurrent agents attached to the same Meter don't
        clobber each other's open-ns timestamps and attribute dicts
        (codex overall-review MAJOR — previously all state lived on
        the Meter instance, so a second agent's ``provider_request``
        overwrote the first agent's ``_chat_open_ns`` / ``_chat_attrs``
        before its response landed).
        """
        from cubepi.providers.base import BaseProvider, collect_agent_providers

        state = _MeterState()

        async def _on_agent_event(event: Any, signal: Any | None = None) -> None:
            del signal
            self._handle_agent_event(state, event)

        def _on_request(payload: dict, model: "Model") -> None:
            self._handle_provider_request(state, payload, model)

        def _on_chunk(event: Any, model: "Model") -> None:
            self._handle_provider_chunk(state, event, model)

        def _on_response(
            body: dict | None,
            model: "Model",
            exc: BaseException | None,
        ) -> None:
            self._handle_provider_response(state, body, model, exc)

        unsub_agent = agent.subscribe(_on_agent_event)
        # When the bound model is a FallbackBoundModel, ``collect_agent_providers``
        # returns every unique BaseProvider in the chain so post-failover token
        # usage / cost / cache-hit metrics are emitted for chain[1..] calls the
        # same way as chain[0]. Non-fallback models yield a single-entry list.
        agent_providers: list[BaseProvider] = collect_agent_providers(agent)
        detachers: list[Callable[[], None]] = []
        for provider in agent_providers:
            # Defensive guard: collect_agent_providers already filters to
            # BaseProvider, but the isinstance check is cheap and protects
            # against a future helper change that loosens the contract.
            if isinstance(provider, BaseProvider):
                detachers.append(provider.subscribe_request(_on_request))
                detachers.append(provider.subscribe_chunk(_on_chunk))
                detachers.append(provider.subscribe_response(_on_response))

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

    @contextlib.asynccontextmanager
    async def attached(self, agent: "Agent") -> AsyncIterator["Meter"]:
        """RAII wrapper around :meth:`attach`.

        ``async with`` enters by attaching the per-attach state, exits
        by calling the detach callable returned from :meth:`attach`.
        Mirrors :meth:`Tracer.attached` — use both for the cleanest
        end-to-end shutdown:

        ::

            async with (
                Tracer(...) as tracer,
                Meter(resource=tracer.resource, ...) as meter,
                tracer.attached(agent),
                meter.attached(agent),
            ):
                await agent.prompt("...")
            # auto: detach both + shutdown both
        """
        detach = self.attach(agent)
        try:
            yield self
        finally:
            try:
                detach()
            except Exception:
                pass

    async def __aenter__(self) -> "Meter":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.shutdown()

    # -- event handlers -----------------------------------------------
    #
    # Each handler takes a per-attach ``_MeterState`` so concurrent
    # agents attached to one Meter never share timing/attribute state.

    def _handle_agent_event(self, state: "_MeterState", event: Any) -> None:
        try:
            type_name = getattr(event, "type", "")
            if type_name == "agent_start":
                state.agent_open_ns = time.time_ns()
                state.agent_attrs = {GEN_AI_OPERATION_NAME: OP_INVOKE_AGENT}
            elif type_name == "agent_end":
                if state.agent_open_ns is not None:
                    duration = (time.time_ns() - state.agent_open_ns) / 1e9
                    self._duration_hist.record(duration, state.agent_attrs)
                state.agent_open_ns = None
                state.agent_attrs = {}
            elif type_name == "tool_execution_start":
                state.tool_open_ns[event.tool_call_id] = time.time_ns()
                # Carry the most-recent provider onto tool metrics so
                # ``execute_tool`` duration observations can be filtered
                # alongside chat metrics. ``_on_provider_request`` fires
                # before any tool in a normal run, so its setdefault
                # loop has nothing to update by then; stamp at creation
                # time here from the latest chat attrs instead
                # (codex P2 finding on PR #85).
                attrs: dict[str, Any] = {GEN_AI_OPERATION_NAME: OP_EXECUTE_TOOL}
                provider_name = state.chat_attrs.get(GEN_AI_PROVIDER_NAME) or (
                    state.agent_attrs.get(GEN_AI_PROVIDER_NAME)
                    if state.agent_attrs
                    else None
                )
                if provider_name is not None:
                    attrs[GEN_AI_PROVIDER_NAME] = provider_name
                state.tool_attrs[event.tool_call_id] = attrs
            elif type_name == "tool_execution_end":
                start = state.tool_open_ns.pop(event.tool_call_id, None)
                attrs = state.tool_attrs.pop(event.tool_call_id, {})
                if start is not None and attrs:
                    duration = (time.time_ns() - start) / 1e9
                    self._duration_hist.record(duration, attrs)
        except Exception:
            pass

    def _handle_provider_request(
        self, state: "_MeterState", payload: dict, model: "Model"
    ) -> None:
        del payload
        provider_name = map_provider_name(model.provider_id)
        state.chat_open_ns = time.time_ns()
        state.chat_first_chunk_ns = None
        # Include gen_ai.request.model so failed/cancelled chat metrics
        # (where the response body never lands and gen_ai.response.model
        # cannot be set) can still be grouped by the requested model
        # (codex P2 finding on PR #85).
        state.chat_attrs = {
            GEN_AI_OPERATION_NAME: OP_CHAT,
            GEN_AI_PROVIDER_NAME: provider_name,
            GEN_AI_REQUEST_MODEL: model.id,
        }
        # Stamp the agent + tool attrs with provider too, so all metrics
        # from this run can be filtered by provider in one shot.
        if state.agent_attrs:
            state.agent_attrs.setdefault(GEN_AI_PROVIDER_NAME, provider_name)
        for _, attrs in state.tool_attrs.items():
            attrs.setdefault(GEN_AI_PROVIDER_NAME, provider_name)

    def _handle_provider_chunk(
        self, state: "_MeterState", event: Any, model: "Model"
    ) -> None:
        del model
        if state.chat_first_chunk_ns is not None or state.chat_open_ns is None:
            return
        ev_type = getattr(event, "type", "")
        if ev_type in ("text_delta", "thinking_delta", "toolcall_delta"):
            state.chat_first_chunk_ns = time.time_ns()

    def _handle_provider_response(
        self,
        state: "_MeterState",
        body: dict | None,
        model: "Model",
        exc: BaseException | None,
    ) -> None:
        del model
        if state.chat_open_ns is None:
            return
        try:
            now = time.time_ns()
            duration = (now - state.chat_open_ns) / 1e9
            response_model: str | None = None
            if isinstance(body, dict):
                response_model = body.get("model") or None
            attrs = dict(state.chat_attrs)
            if response_model:
                attrs[GEN_AI_RESPONSE_MODEL] = response_model
            self._duration_hist.record(duration, attrs)
            if state.chat_first_chunk_ns is not None:
                ttfc = (state.chat_first_chunk_ns - state.chat_open_ns) / 1e9
                self._ttfc_hist.record(ttfc, attrs)
            # Token usage — one observation per type.
            if isinstance(body, dict) and not exc:
                self._record_token_usage(body, attrs)
        finally:
            state.chat_open_ns = None
            state.chat_first_chunk_ns = None
            state.chat_attrs = {}

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


@dataclass
class _MeterState:
    """Per-attach mutable state. One per :meth:`Meter.attach` call so
    concurrent agents on the same Meter never share their open-ns
    timestamps or attribute dicts (codex overall-review MAJOR).

    Previously these fields lived on the Meter instance — a second
    agent's ``provider_request`` would overwrite the first agent's
    ``chat_open_ns`` / ``chat_attrs`` before its response listener
    fired, corrupting the recorded duration and token-usage points.
    """

    chat_open_ns: int | None = None
    chat_first_chunk_ns: int | None = None
    chat_attrs: dict[str, Any] = field(default_factory=dict)
    tool_open_ns: dict[str, int] = field(default_factory=dict)
    tool_attrs: dict[str, dict[str, Any]] = field(default_factory=dict)
    agent_open_ns: int | None = None
    agent_attrs: dict[str, Any] = field(default_factory=dict)
