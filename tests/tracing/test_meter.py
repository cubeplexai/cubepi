"""Phase 4: pin the OTel GenAI metric emissions.

The Meter mirrors Tracer in shape; attach it to an agent and metric
observations flow to the configured MetricExporter.
"""

from __future__ import annotations

from typing import Any

from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
    MetricsData,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource

from cubepi.agent.agent import Agent
from cubepi.providers.base import Model
from cubepi.providers.faux import FauxProvider, faux_assistant_message
from cubepi.tracing import Meter
from cubepi.tracing.schema import SCHEMA_URL, SCOPE_NAME


MODEL = Model(id="faux-1", provider="faux")


def _build_meter() -> tuple[Meter, InMemoryMetricReader]:
    """Construct a Meter that buffers to an InMemoryMetricReader for
    test inspection. Bypasses Meter.__init__'s exporter-based setup so
    we get synchronous reads via reader.get_metrics_data()."""
    reader = InMemoryMetricReader()
    resource = Resource.create({"service.name": "test"}, schema_url=SCHEMA_URL)
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    meter = Meter.__new__(Meter)
    meter._resource = resource  # type: ignore[attr-defined]
    meter._provider = provider  # type: ignore[attr-defined]
    meter._otel_meter = provider.get_meter(name=SCOPE_NAME, schema_url=SCHEMA_URL)
    meter._duration_hist = meter._otel_meter.create_histogram(
        "gen_ai.client.operation.duration", unit="s"
    )
    meter._token_hist = meter._otel_meter.create_histogram(
        "gen_ai.client.token.usage", unit="{token}"
    )
    meter._ttfc_hist = meter._otel_meter.create_histogram(
        "gen_ai.client.operation.time_to_first_chunk", unit="s"
    )
    meter._shutdown = False
    meter._chat_open_ns = None
    meter._chat_first_chunk_ns = None
    meter._chat_attrs = {}
    meter._tool_open_ns = {}
    meter._tool_attrs = {}
    meter._agent_open_ns = None
    meter._agent_attrs = {}
    return meter, reader


def _all_metric_points(reader: InMemoryMetricReader):
    data: MetricsData | None = reader.get_metrics_data()
    if data is None:
        return []
    points: list[tuple[str, Any]] = []
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                for dp in metric.data.data_points:
                    points.append((metric.name, dp))
    return points


def _by_name(points: list[tuple[str, Any]], name: str) -> list[Any]:
    return [dp for n, dp in points if n == name]


class TestDurationMetric:
    async def test_chat_and_invoke_agent_duration(self):
        provider = FauxProvider()
        provider.append_responses([faux_assistant_message("ok")])
        agent = Agent(provider=provider, model=MODEL, system_prompt="s")
        meter, reader = _build_meter()
        meter.attach(agent)

        await agent.prompt("hello")
        await agent.wait_for_idle()

        points = _all_metric_points(reader)
        durations = _by_name(points, "gen_ai.client.operation.duration")
        # At least one chat duration + one invoke_agent duration.
        assert len(durations) >= 2
        op_names = {
            dict(dp.attributes).get("gen_ai.operation.name") for dp in durations
        }
        assert "chat" in op_names
        assert "invoke_agent" in op_names


class TestTokenUsageMetric:
    async def test_token_usage_input_and_output(self):
        provider = FauxProvider()
        provider.append_responses([faux_assistant_message("hi there")])
        agent = Agent(provider=provider, model=MODEL, system_prompt="s")
        meter, reader = _build_meter()
        meter.attach(agent)

        await agent.prompt("hi")
        await agent.wait_for_idle()

        points = _all_metric_points(reader)
        tokens = _by_name(points, "gen_ai.client.token.usage")
        types = {dict(dp.attributes).get("gen_ai.token.type") for dp in tokens}
        # Faux body carries input_tokens + output_tokens — both types emit.
        assert "input" in types
        assert "output" in types


class TestTimeToFirstChunkMetric:
    async def test_ttfc_emitted_when_first_chunk_seen(self):
        provider = FauxProvider(tokens_per_second=200.0)
        provider.append_responses([faux_assistant_message("hello world")])
        agent = Agent(provider=provider, model=MODEL, system_prompt="s")
        meter, reader = _build_meter()
        meter.attach(agent)

        await agent.prompt("x")
        await agent.wait_for_idle()

        points = _all_metric_points(reader)
        ttfc = _by_name(points, "gen_ai.client.operation.time_to_first_chunk")
        assert len(ttfc) == 1


class TestRequestModelOnChatMetrics:
    async def test_chat_duration_carries_request_model(self):
        """Every chat metric must carry ``gen_ai.request.model`` so that
        failed / cancelled requests (no response body, no response.model)
        can still be grouped by the requested model. Codex P2 finding
        on PR #85."""
        provider = FauxProvider()
        provider.append_responses([faux_assistant_message("ok")])
        agent = Agent(provider=provider, model=MODEL, system_prompt="s")
        meter, reader = _build_meter()
        meter.attach(agent)

        await agent.prompt("hello")
        await agent.wait_for_idle()

        points = _all_metric_points(reader)
        chat_durations = [
            dp
            for n, dp in points
            if n == "gen_ai.client.operation.duration"
            and dict(dp.attributes).get("gen_ai.operation.name") == "chat"
        ]
        assert chat_durations
        for dp in chat_durations:
            attrs = dict(dp.attributes)
            assert attrs.get("gen_ai.request.model") == MODEL.id, (
                f"chat metric missing gen_ai.request.model; got {attrs}"
            )


class TestProviderOnToolMetrics:
    async def test_tool_duration_carries_provider_name(self):
        """``execute_tool`` duration observations must carry
        ``gen_ai.provider.name`` so they can be filtered alongside
        chat metrics. Tool execution starts AFTER the chat request,
        so the meter must read the active provider at tool-start time
        instead of relying on the chat-request stamp loop (which by
        then has no tool entries to update). Codex P2 finding on PR #85."""
        from pydantic import BaseModel

        from cubepi.agent.types import AgentTool, AgentToolResult
        from cubepi.providers.base import TextContent, ToolCall

        class P(BaseModel):
            pass

        async def run(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="ok")])

        tool = AgentTool(name="t", description="t", parameters=P, execute=run)
        provider = FauxProvider()
        provider.append_responses(
            [
                faux_assistant_message(
                    [ToolCall(id="c1", name="t", arguments={})],
                    stop_reason="tool_use",
                ),
                faux_assistant_message("final"),
            ]
        )
        agent = Agent(provider=provider, model=MODEL, system_prompt="s", tools=[tool])
        meter, reader = _build_meter()
        meter.attach(agent)

        await agent.prompt("kick off")
        await agent.wait_for_idle()

        points = _all_metric_points(reader)
        tool_durations = [
            dp
            for n, dp in points
            if n == "gen_ai.client.operation.duration"
            and dict(dp.attributes).get("gen_ai.operation.name") == "execute_tool"
        ]
        assert tool_durations
        for dp in tool_durations:
            attrs = dict(dp.attributes)
            assert attrs.get("gen_ai.provider.name") == "faux", (
                f"execute_tool metric missing gen_ai.provider.name; got {attrs}"
            )


class TestNoMetricsWithoutListeners:
    async def test_detach_stops_emission(self):
        provider = FauxProvider()
        agent = Agent(provider=provider, model=MODEL, system_prompt="s")
        meter, reader = _build_meter()
        detach = meter.attach(agent)
        detach()

        provider.append_responses([faux_assistant_message("ok")])
        await agent.prompt("x")
        await agent.wait_for_idle()

        points = _all_metric_points(reader)
        assert _by_name(points, "gen_ai.client.operation.duration") == []
