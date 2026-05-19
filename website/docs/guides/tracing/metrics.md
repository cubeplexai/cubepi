---
title: Metrics
sidebar_position: 5
---

# Metrics with `Meter`

Spans tell you the shape of one run; histograms tell you the shape of the
fleet. `cubepi.tracing.Meter` mirrors `Tracer` and emits the OTel GenAI
metric set so dashboards work out of the box.

## What it emits

| Instrument | Description |
|---|---|
| `gen_ai.client.operation.duration` | Histogram (seconds) — recorded for `chat`, `execute_tool`, and `invoke_agent` on close |
| `gen_ai.client.operation.time_to_first_chunk` | Histogram (seconds) — recorded for `chat` when at least one content chunk arrived |
| `gen_ai.client.token.usage` | Histogram (`{token}`) — one observation per `gen_ai.token.type` (`input`, `output`) per chat response |

Each point carries the operation, provider, and request model attributes
so failed / cancelled requests (where no response body / response model
landed) are still groupable by what was asked for:

- `gen_ai.operation.name` — `chat` / `execute_tool` / `invoke_agent`
- `gen_ai.provider.name` — `anthropic`, `openai`, `openai_responses`, …
- `gen_ai.request.model` — e.g. `claude-sonnet-4-5-20250929`
- `gen_ai.response.model` — e.g. `claude-sonnet-4-5-20250929` (success only)
- `gen_ai.token.type` — `input` or `output` (token usage only)

## Attaching a Meter

```python
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from cubepi.tracing import Tracer, Meter
from cubepi.tracing.exporters import JsonlSpanExporter

tracer = Tracer(
    service_name="my-bot",
    agent_name="assistant",
    exporters=[JsonlSpanExporter(directory="./cubepi-traces")],
)
meter = Meter(
    resource=tracer.resource,        # share Resource so service.* matches spans
    exporters=[
        OTLPMetricExporter(endpoint="http://otel-collector:4318/v1/metrics"),
    ],
)

tracer.attach(agent)
meter.attach(agent)
```

`Meter.attach()` is independent from `Tracer.attach()`. You can run either
on its own; the recommended setup is both, sharing one Resource so the
backend treats them as the same service.

## Bucket boundaries

The duration histograms ship with the OTel GenAI semconv's recommended
boundaries (in seconds):

```text
0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60, 120, 300
```

OTel exposes these as the `_advisory` boundaries; backends are free to use
them as-is or override.

## Concurrent agents on one Meter

Like `Tracer`, one `Meter` instance is safe to attach to multiple agents in
the same process. Each `attach()` gets its own internal state so concurrent
agents never share timing or attribute dicts.

## Shutting down

```python
finally:
    detach()                  # detach from Tracer + Meter
    await tracer.shutdown()
    await meter.shutdown()
```

`Meter.shutdown()` awaits a flush of the metric reader, then closes it.
`PeriodicExportingMetricReader` exports on a fixed interval (60 s by
default) — `shutdown` is the only way to guarantee the final window lands
before the process exits.

## Querying example (Honeycomb)

p95 chat latency by provider over the last hour:

```text
VISUALIZE  P95(duration_s)
GROUP BY   gen_ai.provider.name
WHERE      gen_ai.operation.name = "chat"
TIME       last 1 hour
```

Token usage by model:

```text
VISUALIZE  SUM(token_count)
GROUP BY   gen_ai.request.model, gen_ai.token.type
WHERE      gen_ai.operation.name = "chat"
```

Substitute your backend's query DSL — same attributes, same aggregations.
