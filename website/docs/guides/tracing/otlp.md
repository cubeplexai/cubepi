---
title: OTLP & Backends
sidebar_position: 3
---

# Exporting to OTLP Backends

`cubepi.tracing.Tracer` accepts any `opentelemetry.sdk.trace.export.SpanExporter`,
so anything in the OpenTelemetry ecosystem works. Pick the wire format
(HTTP or gRPC), point it at your collector, hand the exporter to the Tracer.

## HTTP (OTLP/HTTP)

```bash
pip install "cubepi[tracing]" opentelemetry-exporter-otlp-proto-http
```

```python
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)
from cubepi.tracing import Tracer

tracer = Tracer(
    service_name="my-bot",
    service_version="1.4.2",
    deployment_environment="prod",
    agent_name="assistant",
    exporters=[
        OTLPSpanExporter(
            endpoint="http://otel-collector:4318/v1/traces",
            headers={"x-api-key": "…"},  # backend-specific
        ),
    ],
)
```

`service_name`, `service_version`, `deployment_environment`, and `agent_name`
flow through as OTel Resource attributes (`service.*`, `gen_ai.agent.name`,
`deployment.environment.name`) so backends can group runs without further
config.

## gRPC (OTLP/gRPC)

```bash
pip install "cubepi[tracing]" opentelemetry-exporter-otlp-proto-grpc
```

```python
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)

exporter = OTLPSpanExporter(endpoint="otel-collector:4317", insecure=True)
tracer = Tracer(service_name="my-bot", exporters=[exporter])
```

## Backend recipes

These all consume OTLP — the only thing that differs is the endpoint and
auth header.

### Jaeger (>=1.50)

Jaeger natively accepts OTLP/HTTP on port 4318:

```python
OTLPSpanExporter(endpoint="http://jaeger:4318/v1/traces")
```

### Grafana Tempo

Send to your collector, or directly to Tempo's OTLP endpoint:

```python
OTLPSpanExporter(endpoint="http://tempo:4318/v1/traces")
```

### Honeycomb

```python
OTLPSpanExporter(
    endpoint="https://api.honeycomb.io/v1/traces",
    headers={"x-honeycomb-team": HONEYCOMB_API_KEY},
)
```

### Datadog (via the OTel collector)

Configure the collector with the Datadog exporter, then ship to it:

```python
OTLPSpanExporter(endpoint="http://otel-collector:4318/v1/traces")
```

Datadog also accepts native OTLP HTTP directly — same shape, different URL.

### AWS X-Ray (via collector)

The OTel collector includes the AWS X-Ray exporter; treat it like any other
OTLP target.

## Continuing an upstream trace

When cubepi runs inside a service that already has its own traces — e.g. an
HTTP handler — you usually want the agent run rooted under the inbound trace
rather than starting a new one. Pass the parent context when attaching:

```python
from opentelemetry import trace

parent_span = trace.get_current_span()  # set by your web framework's middleware
parent_ctx = parent_span.get_span_context()

tracer = Tracer(service_name="my-bot", exporters=[exporter])

# Pass parent_trace_id / parent_span_id to root the agent's spans under it.
# (See cubepi.tracing.Tracer for the internal helper used by run_scope.)
detach = tracer.attach(agent)
```

On the way out, MCP `tools/call` automatically injects W3C `traceparent` as
an HTTP header so an instrumented MCP server can continue the trace
through to its own backend.

## Combining exporters

You can pass multiple exporters and they'll receive every span. Common
pattern — JSONL for local debugging plus OTLP for the production backend:

```python
tracer = Tracer(
    service_name="my-bot",
    exporters=[
        JsonlSpanExporter(directory="./cubepi-traces"),
        OTLPSpanExporter(endpoint="https://api.honeycomb.io/v1/traces", headers={…}),
    ],
)
```

## Flushing

`Tracer` uses `BatchSpanProcessor` under the hood, so spans are exported in
the background. To make sure buffered spans land before your process exits:

```python
finally:
    detach()                    # closes any spans a cancelled run left open
    await tracer.shutdown()     # awaits force_flush, then shuts the SDK down
```

`shutdown()` is idempotent; calling it twice is a no-op.

Alternatively, `await detach()` directly — the Task it returns awaits the
flush, so a `finally: await detach()` block is enough without calling
`shutdown()` separately. The two together is the safest pattern.
