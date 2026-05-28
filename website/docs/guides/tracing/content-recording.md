---
title: Content & Redaction
description: "Configure prompt content recording and redaction in CubePi's OpenTelemetry tracing."
sidebar_position: 4
---

# Recording Prompts, Responses, and Tool Payloads

By default cubepi's tracing emits structural attributes only — operation
names, models, token counts, finish reasons, durations. **No prompt content,
no model output, no tool arguments or results leave the process.** This is
deliberate: many agent setups handle PII, customer data, or trade-secret
prompts that don't belong in a third-party observability backend.

When you _do_ want content captured — for offline evaluation, debugging a
flaky tool call, or building a labelled dataset — opt in explicitly with
`record_content=True`, and combine it with a `redact` callback to strip the
sensitive parts before they leave the process.

## Turning content recording on

```python
tracer = Tracer(
    service_name="my-bot",
    agent_name="assistant",
    record_content=True,            # ← opt-in
    exporters=[JsonlSpanExporter(directory="./cubepi-traces")],
)
```

With `record_content=True`, each span layer carries the relevant content
attributes per the OTel GenAI semconv:

| Span | Content attributes added |
|---|---|
| `invoke_agent` | `gen_ai.system_instructions`, `gen_ai.input.messages`, `gen_ai.output.messages` |
| `cubepi.turn` | `gen_ai.input.messages` (per-turn slice), `gen_ai.output.messages` (per-turn slice) |
| `chat <model>` | `gen_ai.system_instructions`, `gen_ai.input.messages`, `gen_ai.tool.definitions`, `cubepi.llm.raw_request`, `cubepi.llm.raw_response` |
| `execute_tool <tool_name>` | `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result` |

The `chat` span's `gen_ai.input.messages` contains the **full chronological
context** the provider request actually carried — including prior assistant
turns and tool results — not just the new user prompt. This matters for
multi-turn tool-using runs: trace consumers can reconstruct exactly what
the model saw at each call.

## Redacting before export

`redact` is a `(key, value) -> value | None` callback invoked at every
content-attribute set site. Return:

- The original value unchanged → keep as-is
- A modified value of the same shape → substitute
- `None` → drop the attribute entirely

```python
def redact(key: str, value):
    # Strip secrets from prompts before they leave the process.
    if key in ("gen_ai.input.messages", "gen_ai.output.messages"):
        return _scrub_messages(value)
    # Don't ship raw bodies at all in prod — keep only the normalised shape.
    if key in ("cubepi.llm.raw_request", "cubepi.llm.raw_response"):
        return None
    return value


tracer = Tracer(
    service_name="my-bot",
    record_content=True,
    redact=redact,
    exporters=[…],
)
```

`redact` is the single chokepoint for content — the recorder calls it once
per attribute before serializing into the OTel attribute, so anything the
function returns is what hits the wire. Exceptions inside `redact` are
swallowed (the attribute is dropped in that case), so a buggy redactor
fails closed rather than leaking.

### Common patterns

Drop everything but per-message length so dashboards still work without
shipping content:

```python
def redact(key, value):
    if key in ("gen_ai.input.messages", "gen_ai.output.messages"):
        return [{"role": m["role"], "parts": [{"type": "text", "content": "<redacted>",
                                               "length": sum(len(p.get("content", "")) for p in m["parts"])}]}
                for m in value]
    return value
```

Tag-based selective recording — strip everything unless a thread is opted in:

```python
import contextvars
RECORD = contextvars.ContextVar("trace.record_content", default=False)

def redact(key, value):
    return value if RECORD.get() else None
```

then `RECORD.set(True)` for the runs you want captured.

## Size budgets

OTel attribute values are JSON-serialized inside the recorder. Most backends
truncate or reject attributes over a few hundred KB. If you're recording
the raw provider response on every chat span, multi-turn agentic runs can
get large fast. Drop or summarise via `redact` for any field over your
budget.

## Auditing what's recorded

The recorder always sets `service.name`, `gen_ai.agent.name`, and
`cubepi.run_id` on every span — regardless of `record_content`. Use these
to filter the trace backend to a single run and visually confirm what
landed.

For deeper audits, `JsonlSpanExporter` writes one line per span, so you can
grep / `jq` the local files before pointing the same exporter at a remote
backend:

```bash
jq -r 'select(.attributes["gen_ai.input.messages"]) | .attributes["gen_ai.input.messages"]' \
   cubepi-traces/2026-05-19/*.jsonl
```
