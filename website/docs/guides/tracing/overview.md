---
title: Tracing Overview
sidebar_position: 1
---

# Tracing Overview

CubePi emits [OpenTelemetry](https://opentelemetry.io/) spans that follow the
[GenAI Semantic Conventions v1.41](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
so any OTel-compatible backend (Jaeger, Tempo, Honeycomb, Datadog, AWS X-Ray,
Azure Monitor, …) can ingest agent runs without custom instrumentation.

Attach a `Tracer` to an `Agent` and every prompt produces a tree of spans you
can pivot, query, and join with the rest of your service traces:

```text
invoke_agent <agent_name>              [INTERNAL]    one per agent.prompt()
└── cubepi.turn                        [INTERNAL]    one per LLM round-trip
    ├── chat <model>                   [CLIENT]      the LLM call itself
    └── execute_tool <tool_name>       [INTERNAL]    each tool invocation
        └── tools/call <tool_name>     [CLIENT]      (MCP tools only)
```

Each layer carries standard `gen_ai.*` attributes — `gen_ai.operation.name`,
`gen_ai.request.model`, `gen_ai.provider.name`, `gen_ai.usage.input_tokens`,
`gen_ai.usage.output_tokens`, `gen_ai.response.finish_reasons`, …

## What ships out of the box

- **Tracer** — builds an SDK `TracerProvider`, attaches one
  `BatchSpanProcessor` per exporter, wires the cubepi event stream into spans.
- **Meter** — sibling for OTel histograms:
  `gen_ai.client.operation.duration`, `gen_ai.client.operation.time_to_first_chunk`,
  `gen_ai.client.token.usage`.
- **JsonlSpanExporter** — write one JSON line per span to
  `./cubepi-traces/<date>/<run_id>.jsonl`. Useful for local dev and offline
  debugging; works with any OTel viewer that reads JSONL.
- **OTLP** — bring your own exporter via `opentelemetry-exporter-otlp-proto-http`
  (HTTP) or `…-grpc` and hand it to `Tracer(exporters=[…])`.
- **W3C trace context propagation** — outgoing MCP calls inject the active
  `traceparent` as an HTTP header so an instrumented MCP server can continue
  the trace.

## What it costs

- One pure-Python recorder per agent run subscribing to the agent's event
  stream and the provider's listener registry — no monkey-patching, no extra
  threads.
- One OTel SDK span per layer above. `BatchSpanProcessor` batches export off
  the hot path.
- **No payloads are recorded by default.** `gen_ai.input.messages`,
  `gen_ai.output.messages`, raw request/response, and tool args/results all
  require explicit opt-in via `record_content=True` so you don't accidentally
  ship PII to your backend. See [Content & Redaction](./content-recording).

## When to use each piece

| You want | Use |
|---|---|
| Trace one local agent run and inspect a JSONL file | `Tracer` + `JsonlSpanExporter` |
| Ship to Jaeger / Tempo / Honeycomb / Datadog | `Tracer` + OTLP exporter |
| Latency + token histograms next to the spans | `Meter` alongside `Tracer` |
| Record prompts / model outputs for evaluation | `Tracer(record_content=True)` |
| Redact PII before it leaves the process | `Tracer(redact=…)` |
| Continue a trace from an upstream service | `Tracer(resource=…)` + W3C `traceparent` (auto for MCP, manual for HTTP) |

## Where to go next

- [Getting Started](./getting-started) — install the extra and emit your
  first spans
- [OTLP & Backends](./otlp) — point cubepi at Jaeger, Tempo, Honeycomb, …
- [Content Recording & Redaction](./content-recording) — record prompts and
  responses safely
- [Metrics](./metrics) — histograms via `Meter`
