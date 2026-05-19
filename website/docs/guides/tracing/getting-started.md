---
title: Getting Started
sidebar_position: 2
---

# Getting Started with Tracing

## Install the extra

CubePi keeps OpenTelemetry an optional dependency:

```bash
pip install "cubepi[tracing]"
```

This pulls in `opentelemetry-sdk` and friends. Without the extra, the
`cubepi.tracing` import raises a clear error so you find out at import time
rather than mid-run.

## Attach a Tracer

The minimal end-to-end setup — local JSONL export:

```python
import asyncio
from cubepi import Agent, Model
from cubepi.providers.anthropic import AnthropicProvider
from cubepi.tracing import Tracer
from cubepi.tracing.exporters import JsonlSpanExporter


async def main() -> None:
    agent = Agent(
        provider=AnthropicProvider(api_key="…"),
        model=Model(id="claude-sonnet-4-5-20250929", provider="anthropic"),
        system_prompt="Be helpful.",
    )

    tracer = Tracer(
        service_name="my-bot",
        agent_name="assistant",
        exporters=[JsonlSpanExporter(directory="./cubepi-traces")],
    )
    detach = tracer.attach(agent)
    try:
        await agent.prompt("Say hello.")
        await agent.wait_for_idle()
    finally:
        # Either is enough on its own:
        #   await detach()                  # awaits the post-detach flush
        #   await tracer.shutdown()          # flushes + closes exporters
        detach()
        await tracer.shutdown()


asyncio.run(main())
```

The run produces one JSONL file per agent run:

```text
./cubepi-traces/
  2026-05-19/
    8e1c…-…-…-….jsonl       ← one run, one file, one span per line
```

Open it with any tool that reads OTLP/JSON or with `jq` directly:

```bash
jq -r '"\(.name)  \(.attributes."gen_ai.operation.name" // "")"' \
   cubepi-traces/2026-05-19/*.jsonl
# invoke_agent  invoke_agent
# cubepi.turn
# chat claude-sonnet-4-5-20250929  chat
```

## Span hierarchy

For a single prompt with one LLM round-trip, the recorder produces three spans:

```text
invoke_agent assistant              [INTERNAL]   gen_ai.operation.name=invoke_agent
└── cubepi.turn                     [INTERNAL]   cubepi.turn.index=0
    └── chat <model>                [CLIENT]     gen_ai.operation.name=chat
```

When the model calls a tool, you get an extra layer per tool:

```text
invoke_agent assistant
└── cubepi.turn                     ← turn index 0
    ├── chat <model>                ← first round trip
    └── execute_tool <tool_name>    ← gen_ai.tool.name, gen_ai.tool.call.id
└── cubepi.turn                     ← turn index 1 (response after tool result)
    └── chat <model>
```

For MCP tools the `execute_tool` span gets a CLIENT child:

```text
execute_tool <tool_name>            [INTERNAL]   cubepi-side wrapper
└── tools/call <tool_name>          [CLIENT]     gen_ai.operation.name=execute_tool
                                                  mcp.method.name=tools/call
                                                  mcp.session.id=…
                                                  server.address / server.port
```

The CLIENT span injects W3C `traceparent` into outgoing HTTP headers, so an
instrumented MCP server can continue the trace.

## Cancellation, errors, aborts

The recorder treats cancellation as a control signal, not a failure:

- `agent.abort()` mid-stream → spans close with `cubepi.aborted=true` and
  `error.type=cubepi.aborted`, **status UNSET** (per OTel guidance — cancellation
  isn't an error).
- A provider raising → chat/turn/root close with **status ERROR**, an
  `exception` event on the chat span, and `error.type` derived from the
  exception class (`timeout`, `connection_error`, fully-qualified class name, …).
- An MCP `tools/call` returning `isError=true` → CLIENT span closes
  ERROR + `error.type=mcp.is_error`.

Either way, `detach()` and `tracer.shutdown()` always close any span the run
left open, so cancelled runs are still visible in your backend rather than
silently disappearing.

## What's on each span

Defaults (no opt-in needed):

- `invoke_agent` (root) — `gen_ai.operation.name`, `gen_ai.provider.name`,
  `gen_ai.agent.name`, `cubepi.run_id`, `cubepi.agent.system_prompt.sha256`,
  `cubepi.agent.tools` (names list), `cubepi.input_messages.count`,
  `cubepi.output_messages.count`
- `cubepi.turn` — `cubepi.turn.index`, `cubepi.turn.stop_reason`,
  `cubepi.turn.tool_calls.count`, `cubepi.turn.terminated_by_tool`,
  `cubepi.run_id`
- `chat <model>` — `gen_ai.operation.name`, `gen_ai.provider.name`,
  `gen_ai.request.model`, `gen_ai.request.max_tokens` / `temperature` /
  `top_p`, `gen_ai.request.stream`, `gen_ai.usage.input_tokens` /
  `output_tokens` / `cache_read_input_tokens` / `cache_creation_input_tokens` /
  `reasoning_output_tokens`, `gen_ai.response.model` /
  `finish_reasons` / `id`, `gen_ai.response.time_to_first_chunk`, plus
  OpenAI-specific extras (`openai.api.type`, service tier, system fingerprint)
- `execute_tool <tool_name>` — `gen_ai.operation.name=execute_tool`,
  `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.description`,
  `gen_ai.tool.type`, `cubepi.tool.is_error`, `cubepi.tool.execution_mode`
- `tools/call <tool_name>` (MCP only) — `mcp.method.name`, `mcp.session.id`,
  `mcp.protocol.version`, `server.address`, `server.port`, `gen_ai.tool.name`

Optional, opt-in via `Tracer(record_content=True)`:
`gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`,
`gen_ai.tool.definitions`, `gen_ai.tool.call.arguments`,
`gen_ai.tool.call.result`, `cubepi.llm.raw_request`,
`cubepi.llm.raw_response`. See [Content & Redaction](./content-recording).

## Multiple agents, one process

`Tracer` is fine to share across agents — call `attach(agent)` multiple
times. Each attach gets its own recorder so concurrent agents don't share
span state, and MCP CLIENT spans route through the right Tracer based on
which agent's `execute_tool` span is the parent. The same applies to
`Meter`.

## Next

- [OTLP & Backends](./otlp) — Jaeger, Tempo, Honeycomb, Datadog, …
- [Content Recording & Redaction](./content-recording)
- [Metrics](./metrics)
