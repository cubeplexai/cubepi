# CubePi Tracing — Design Spec

**Date:** 2026-05-18
**Author:** xf gong (gxf.alpha@gmail.com)
**Status:** Draft — awaiting review
**Spec target:** OpenTelemetry Semantic Conventions, GenAI working set as of 2026-05 (SemConv v1.41.0)

## 1. Goal & Scope

Give CubePi a first-class debugging/observability story that is **fully compatible with the OpenTelemetry data model and the GenAI semantic conventions**. A developer should be able to:

- Inspect any `agent.run()` after the fact: full prompt sent, full response received, per-call timing, token usage, tool args/results, errors with stacktraces.
- Pipe traces into any OTel-aware backend (Jaeger, Tempo, Phoenix, Honeycomb, Datadog, …) with zero translation.
- Run cubepi locally with zero infrastructure (jsonl on disk).

Two exporters in this scope:

| Exporter | Purpose |
|---|---|
| **`JsonlSpanExporter`** | Append-only file, one OTLP/JSON span per line. For local dev, CI repro. |
| **`OTLPSpanExporter`** (re-export) | OTLP/HTTP over protobuf. For production, OTel Collector, Jaeger, SaaS. |

Deferred to a later phase: SQLite exporter, OpenSearch exporter, sampling.

Out of scope:
- Query backend / UI (separate repo, separate spec).
- Auto-instrumentation of third-party libraries.

## 2. Non-Goals & Explicit Rejections

| Option | Rejected because |
|---|---|
| Read `CheckpointData` to drive debugging UI | Checkpoint stores final messages + extra dict only — no timing, tokens, durations, payloads. |
| Adopt Traceloop / OpenLLMetry SDK | LangChain-flavored autoinstrumentation; cubepi is not LangChain. |
| Implement as cubepi `Middleware` | Middleware is a mutation API. Tracing must not have that power. |
| Roll our own span / resource / scope dataclasses | Duplicating `opentelemetry-sdk`. See §3.3. |
| Persist every `MessageUpdateEvent` / `ToolExecutionUpdateEvent` | Streaming deltas are useful for *live* UI, not for *trace* storage. Record final state only. |
| Use ad-hoc field names (`cubepi.tool.args`, `gen_ai.system`) | Not OTel-compatible. Use standardized `gen_ai.tool.call.arguments`, `gen_ai.provider.name`. |

## 3. Design Decisions

### 3.1 Instrumentation surface: `agent.subscribe()` + provider listener registry

cubepi exposes one observation-only hook today and requires one new one:

- `Agent.subscribe(listener)` in `cubepi/agent/agent.py` — emits the 10-event `AgentEvent` union (agent/turn/message/tool-execution × start/[update]/end). **Already exists.**
- A persistent multi-subscriber listener registry on `Provider` — exposes request payload, per-chunk timing, and assembled response body to the tracer. **Requires a small extension to `cubepi/providers/base.py`**; specified in §3.4 below.

The existing per-call `StreamOptions.on_payload` / `on_response` callbacks are insufficient for tracing — `on_response` carries only HTTP status + headers, not the response body — and they are single-slot, so a user-supplied callback would clobber the tracer. The new registry coexists with these per-call slots.

No monkey-patching, no SDK auto-instrumentation, no `Middleware`.

### 3.2 Span hierarchy using semconv operation names

All spans use names from the semconv `gen_ai.operation.name` enum:

| Span | `gen_ai.operation.name` | `SpanKind` | Source event |
|---|---|---|---|
| Root | `invoke_agent` | `INTERNAL` | `AgentStartEvent` / `AgentEndEvent` |
| Per turn | (cubepi-namespaced; no `gen_ai.operation.name`) | `INTERNAL` | `TurnStartEvent` / `TurnEndEvent` |
| Per LLM call | `chat` | `CLIENT` | assistant `MessageStartEvent` / `MessageEndEvent` |
| Per tool call (cubepi-side) | `execute_tool` | `INTERNAL` | `ToolExecutionStartEvent` / `ToolExecutionEndEvent` |
| Per MCP call (when tool is MCP-backed) | `execute_tool` (with `mcp.method.name="tools/call"`) | `CLIENT` | inside `cubepi/mcp/` client |

Rationale for **not** using `gen_ai.operation.name = "invoke_workflow"` on the turn span: the semconv agent-spans page states `invoke_workflow` "should not be reported when instrumentation cannot distinguish workflow from agent invocation," and the meaning of "workflow" in cubepi (one model→tool roundtrip) is narrower than the semconv intent (orchestration of multiple agents and/or tools). To stay conservatively correct, the turn span uses the cubepi-namespaced name `cubepi.turn` and carries only `cubepi.turn.*` attributes — no `gen_ai.operation.name`, no `gen_ai.workflow.name`. Standard OTel UIs will still render the parent-child structure correctly via the `invoke_agent → cubepi.turn → chat / execute_tool` tree; only the operation-name-keyed dashboards skip the intermediate layer, which is the right default.

### 3.3 Build on `opentelemetry-sdk`, do not reimplement

cubepi's tracing module **depends on `opentelemetry-sdk`** for:

- `Span` / `Resource` / `InstrumentationScope` / `Status` / `SpanKind` data types — full OTLP proto fidelity, `AnyValue` encoding, `dropped_*_count` bookkeeping all handled.
- `TracerProvider` / `Tracer` — span lifecycle, schema URL plumbing, `IdGenerator`, `SpanLimits`.
- `BatchSpanProcessor` / `SimpleSpanProcessor` — async batching, retries, shutdown.
- `SpanExporter` base class — pluggable export pipeline.
- `record_exception()`, `set_status()`, `add_event()` — all standardized formats.
- `MeterProvider` / `Histogram` — for the metrics in §13.
- `TraceContextTextMapPropagator` — W3C `traceparent` injection (used by MCP integration).
- `OTLPSpanExporter` (from `opentelemetry-exporter-otlp-proto-http`) — re-exported as-is.

**Library/application boundary.** cubepi as a library declares OTel as **optional**. Users opt in via `pip install cubepi[tracing]`. Without that extra, `cubepi.tracing` raises `ImportError` cleanly; the rest of cubepi works untouched.

We use SDK in **explicit-span mode**, not in `with start_as_current_span` mode. Reason: cubepi's event stream gives us discrete start/end callbacks, not bracketed scopes — so we manually call `tracer.start_span()`, hold the `Span` reference on a per-run stack, then call `span.end()` at the matching end event. The SDK supports this first-class via `start_span(context=...)` with an explicit parent. This also avoids any `contextvars` ambiguity when parallel tools run concurrently.

What we still write ourselves (~400-600 lines, including Phase 0):

- **Phase 0 — prerequisite cubepi changes** (§3.4): (a) Provider listener registry — converts `Provider` from `Protocol` to a base class, adds three persistent subscription methods, threads listener invocations through every provider implementation; (b) extends `ToolExecutionEndEvent` with `terminate` / `blocked_by_hook` / `block_reason` fields.
- `Recorder`: AgentEvent + provider listeners → SDK API calls.
- `schema.py`: `gen_ai.*` / `cubepi.*` field-name constants.
- `Tracer` (config class): wraps `TracerProvider` setup, exporter registration, `attach(agent)`.
- `JsonlSpanExporter`: ~30-line `SpanExporter` subclass.
- `mcp.py`: helper for the MCP client to wrap calls in a CLIENT span.
- Provider-specific attribute extraction (Anthropic token reconciliation, OpenAI service tier).

### 3.4 Required cubepi extensions (Phase 0)

Two small changes to cubepi are prerequisites for Phase 1.

#### 3.4.1 Provider listener registry

The tracer cannot capture the wire payload, time-to-first-chunk, or assembled response body using the current `Provider` interface.

**Current state** (`cubepi/providers/base.py`):

- `Provider` is a `Protocol` with no instance state.
- Per-call `StreamOptions.on_payload` / `on_response` are single-slot callbacks.
- `OnResponseCallback` receives `ProviderResponse(status, headers), Model` — HTTP metadata only, no body.

**Required additions:**

1. **Make `Provider` a base class** (or introduce a mixin) carrying a listener registry so multiple subscribers (tracer + user code + future extensions) coexist without clobbering each other.

2. **New persistent subscription API**, three methods, each returning a detach callable:

   ```python
   class Provider:
       def subscribe_request(self, cb: OnRequestCallback) -> Callable[[], None]: ...
       def subscribe_chunk(self, cb: OnChunkCallback) -> Callable[[], None]: ...
       def subscribe_response(self, cb: OnResponseBodyCallback) -> Callable[[], None]: ...
   ```

   These fire on **every** `provider.stream()` call. They are observers, never mutators. The existing per-call `StreamOptions.on_payload` (mutate the request) and `on_response` (inspect HTTP metadata) retain their current semantics; the new listeners fire **after** any `StreamOptions.on_payload` mutation, so they see the final wire payload.

3. **New callback signatures:**

   ```python
   OnRequestCallback      = Callable[[dict, Model], None | Awaitable[None]]
   # fires just before HTTP send; payload is the final dict after any
   # StreamOptions.on_payload mutation.

   OnChunkCallback        = Callable[[StreamEvent, Model], None | Awaitable[None]]
   # fires for each StreamEvent (start / text_delta / done / error / ...).
   # The tracer uses this ONLY to time TTFT and count chunks; content is not
   # stored. Heavy listeners should early-return on irrelevant event types.

   OnResponseBodyCallback = Callable[
       [dict | None, Model, Exception | None],
       None | Awaitable[None],
   ]
   # fires after the stream terminates (normal completion or error).
   #   - dict: the assembled provider response (the same structure a buffered
   #     non-streaming call would have returned), or None on early failure.
   #   - Exception: present iff the stream raised; None on normal completion.
   ```

4. **Provider implementations** invoke listeners at these points (anthropic.py / openai.py / openai_responses.py):

   ```python
   # inside provider.stream()
   request_dict = build_request(...)
   request_dict = await invoke_on_payload(opts.on_payload, request_dict, model)
   await _fire_listeners(self._request_listeners, request_dict, model)

   try:
       async for raw_chunk in http_stream:
           event = parse_chunk(raw_chunk)
           stream.push(event)
           await _fire_listeners(self._chunk_listeners, event, model)
       final_response_dict = assemble_response(...)
       await _fire_listeners(self._response_listeners, final_response_dict, model, None)
   except Exception as exc:
       await _fire_listeners(self._response_listeners, None, model, exc)
       raise
   ```

**What this enables for the tracer:**

| Need | Hook |
|---|---|
| `cubepi.llm.raw_request` | `subscribe_request` |
| `chat` span **open** | `subscribe_request` |
| `gen_ai.response.time_to_first_chunk` | `subscribe_chunk` (timestamp of first non-`start` chunk) |
| `gen_ai.client.operation.time_per_output_chunk` metric | `subscribe_chunk` (consecutive non-`start` chunks) |
| `cubepi.llm.raw_response`, all `gen_ai.response.*`, all `gen_ai.usage.*`, provider-specific attrs | `subscribe_response` |
| `chat` span **close** (success or error) | `subscribe_response` |

**Listener invocation contract:**

- Listeners run on the asyncio event loop in registration order.
- Exceptions in listeners are logged and **swallowed** by the provider — a buggy tracer must never crash the stream. The contract is enforced inside `_fire_listeners`, not at each call site.
- **`subscribe_response` listeners MUST be invoked exactly once per `stream()` call, before that call returns or raises.** This is the contract that lets the tracer guarantee `chat` span closure. Provider implementations enforce it with a `finally` block:

  ```python
  request_dict = build_request(...)
  request_dict = await invoke_on_payload(opts.on_payload, request_dict, model)
  await _fire_listeners(self._request_listeners, request_dict, model)

  body: dict | None = None
  exc: BaseException | None = None
  try:
      async for raw_chunk in http_stream:
          event = parse_chunk(raw_chunk)
          stream.push(event)
          await _fire_listeners(self._chunk_listeners, event, model)
      body = assemble_response(...)
  except BaseException as e:           # includes asyncio.CancelledError
      exc = e
      raise
  finally:
      # Fires exactly once, even on abort / cancellation / network drop.
      # Body may be partial-and-not-None if the provider managed to assemble
      # a final message with stop_reason="aborted" before the stream broke.
      await _fire_listeners(self._response_listeners, body, model, exc)
  ```

  Note `except BaseException`: `asyncio.CancelledError` is a `BaseException` in Python 3.8+, not an `Exception`. The provider must still fire the listener on cancel; it then re-raises to honor cancellation semantics.

#### 3.4.2 `ToolExecutionEndEvent` extension

Current `ToolExecutionEndEvent` carries only `tool_call_id`, `tool_name`, `result`, `is_error` — insufficient to populate `cubepi.tool.terminate`, `cubepi.tool.blocked_by_hook`, `cubepi.tool.block_reason` cleanly. `result.terminate` is reachable by unwrapping the `AgentToolResult`, but block reasons today are only visible as a string buried inside the result's content.

Add three fields:

```python
class ToolExecutionEndEvent(BaseModel):
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: Any = None
    is_error: bool = False
    # New:
    terminate: bool = False
    blocked_by_hook: bool = False
    block_reason: str | None = None
```

The tool-execution layer (`cubepi/agent/tools.py`) populates these:
- `terminate` is copied from the finalized `AgentToolResult.terminate` (already plumbed through `after_tool_call`).
- `blocked_by_hook` is `True` only when the `_ImmediateOutcome` was produced by `before_tool_call` returning `block=True`. Distinguish this from other immediate errors (tool not found, arg validation failure) so the recorder can tag the span correctly.
- `block_reason` is the `BeforeToolCallResult.reason` string when blocked, else `None`.

This change is additive — existing consumers continue to work; defaults are conservative.

#### 3.4.3 No checkpointer changes

The `extra` field on `AgentContext` and `CheckpointData` remains tracing-free. Nested-run parent-span propagation uses OTel's standard context propagation (see §8), not `extra`.

### 3.5 Repository split

Recorder/tracer/exporters live in cubepi repo. Query backend + UI live in a separate repo (`cubepi-trace`), reading exported documents through a storage boundary. No shared schema package.

## 4. Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                         cubepi (this repo)                     │
│                                                                │
│   Agent ──┬─ subscribe(listener) ──▶ Recorder                 │
│           │                              │                     │
│   Provider ─ on_payload / on_response ───┤                     │
│   MCP client ─ wraps call site ──────────┤                     │
│                                          ▼                     │
│                                  opentelemetry-sdk Tracer      │
│                                          │                     │
│                                          ▼                     │
│                                  TracerProvider                │
│                                          │                     │
│                                  ┌───────┴───────┐             │
│                                  ▼               ▼             │
│                            JsonlSpanExporter   OTLPSpanExporter│
│                                                                │
│                       MeterProvider (optional)                 │
│                         ──▶ OTLPMetricExporter                 │
└───────────────────────────────────────────────────────────────┘
            │                                       │
            ▼                                       ▼
   ┌────────────────────┐              ┌──────────────────────────┐
   │  cubepi-traces/    │              │  OTel Collector / Jaeger /│
   │   YYYY-MM-DD/      │              │  Tempo / Phoenix /        │
   │     <run_id>.jsonl │              │  Honeycomb / Datadog / …  │
   └────────────────────┘              └──────────────────────────┘
```

## 5. Module Layout

```
cubepi/tracing/
├── __init__.py               # public: Tracer, attach helpers, schema constants
├── tracer.py                 # Tracer config class: builds TracerProvider, wires exporters
├── recorder.py               # consumes AgentEvent + provider callbacks → SDK API calls
├── schema.py                 # field-name constants (gen_ai.*, cubepi.*, mcp.*, openai.*)
├── errors.py                 # error.type derivation from exceptions / stop reasons
├── meter.py                  # MeterProvider setup + histogram registry (phase 4)
├── mcp.py                    # helper for cubepi.mcp client: open tools/call CLIENT spans
└── exporters/
    ├── __init__.py           # re-exports OTLPSpanExporter for convenience
    └── jsonl.py              # JsonlSpanExporter: SpanExporter subclass, span.to_json() per line
```

`pyproject.toml`:

```toml
[project.optional-dependencies]
tracing      = ["opentelemetry-sdk>=1.30"]
tracing-otlp = ["opentelemetry-exporter-otlp-proto-http>=1.30"]
```

A user who does not install `cubepi[tracing]` cannot import `cubepi.tracing` (the package's `__init__` does an SDK import at module level and will fail with a clear message). All other cubepi functionality is unaffected.

## 6. Public API

```python
from cubepi.tracing import Tracer
from cubepi.tracing.exporters.jsonl import JsonlSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

tracer = Tracer(
    service_name="my-bot",
    service_version="0.1.0",
    deployment_environment="dev",
    agent_name="coding-agent",
    agent_id="agent-abc",
    exporters=[
        JsonlSpanExporter(directory="./cubepi-traces"),
        OTLPSpanExporter(endpoint="http://collector:4318/v1/traces"),
    ],
    record_content=False,   # default: do not record prompt/completion bodies
)
tracer.attach(agent)

# Per-run override:
async with tracer.run_scope(extra_attrs={"cubepi.user.id": user_id}):
    await agent.run([UserMessage(...)])
```

What `Tracer.__init__` does:
1. Builds a `Resource` with `service.name`, `service.version`, `deployment.environment.name`, `gen_ai.agent.*`, `telemetry.sdk.*` (auto by SDK).
2. Builds a `TracerProvider(resource=resource)`.
3. For each exporter, wraps it in `BatchSpanProcessor(exporter)` and registers via `provider.add_span_processor(...)`.
4. Calls `provider.get_tracer(name="cubepi.tracing", version=cubepi.__version__, schema_url=SCHEMA_URL)`; holds the `Tracer` instance.

What `tracer.attach(agent)` does:
1. `agent.subscribe(self._recorder.on_agent_event)` — high-level lifecycle (agent / turn / tool spans).
2. `agent.provider.subscribe_request(self._recorder.on_provider_request)` — opens `chat` span and records `cubepi.llm.raw_request`.
3. `agent.provider.subscribe_chunk(self._recorder.on_provider_chunk)` — times TTFT and chunk metric.
4. `agent.provider.subscribe_response(self._recorder.on_provider_response)` — records `gen_ai.response.*` / `gen_ai.usage.*` / `cubepi.llm.raw_response` and closes `chat` span.
5. Returns a `detach()` callable that reverses all four subscriptions and forces a final flush.

### 6.1 Lifecycle and flush

`BatchSpanProcessor` exports spans asynchronously — `AgentEnd` does **not** guarantee that the corresponding spans have hit disk or the wire. cubepi exposes three controls:

```python
class Tracer:
    async def force_flush(self, timeout_seconds: float = 30) -> bool:
        """Block until all currently-buffered spans are exported.
        Returns False on timeout."""

    async def shutdown(self, timeout_seconds: float = 30) -> None:
        """Flush, then close all exporters. Tracer is unusable afterwards."""

    async def __aenter__(self) -> Tracer: ...
    async def __aexit__(self, *exc) -> None:
        """Calls shutdown() on context exit."""
```

Common patterns:

```python
# CI / short-lived script — guarantee data is on disk before process exits
async with Tracer(...) as tracer:
    tracer.attach(agent)
    await agent.run(...)
    # __aexit__ flushes and closes exporters here

# Long-running service — flush at known checkpoints
tracer = Tracer(...)
detach = tracer.attach(agent)
await agent.run(...)
await tracer.force_flush()              # spans guaranteed exported now
# ... later, on shutdown signal:
await tracer.shutdown()
```

`detach()` returned from `attach()` calls `force_flush()` internally before unsubscribing, so any in-flight spans for that agent settle before the listener chain disconnects.

Under the hood, `Tracer.force_flush()` / `shutdown()` delegate to `provider.force_flush()` / `provider.shutdown()`, which propagate to every registered `SpanProcessor`. `JsonlSpanExporter` honors flush by `fsync`'ing the active file; `OTLPSpanExporter` drains its retry queue.

Note we **do not** call `trace.set_tracer_provider(provider)` — that mutates global state. cubepi's `Tracer` owns its provider so multiple agents can coexist in one process with different exporter configs.

## 7. Schema URL & SDK Setup

```python
SCHEMA_URL = "https://opentelemetry.io/schemas/1.41.0"
```

Pinned to the semconv version we promise to follow. Set in two places:

```python
resource = Resource.create(
    attributes={"service.name": ..., ...},
    schema_url=SCHEMA_URL,
)
tracer_internal = provider.get_tracer(
    instrumenting_module_name="cubepi.tracing",
    instrumenting_library_version=cubepi.__version__,
    schema_url=SCHEMA_URL,
)
```

`cubepi.*` extensions are versioned by cubepi releases. `gen_ai.*` keys follow OTel deprecation policy.

## 8. Trace & Span ID Inheritance

The SDK's `RandomIdGenerator` generates `trace_id` (16 bytes) and `span_id` (8 bytes) by default. cubepi does not override it.

**Inbound trace context.** When a host service wants to make cubepi a child of its own trace:

```python
async with tracer.run_scope(
    parent_trace_id=incoming_trace_id_int,   # int128
    parent_span_id=incoming_span_id_int,     # int64
    trace_flags=0x01,
):
    await agent.run(...)
```

Implementation:

```python
from opentelemetry.trace import SpanContext, TraceFlags, NonRecordingSpan, set_span_in_context

parent_ctx = SpanContext(
    trace_id=parent_trace_id,
    span_id=parent_span_id,
    is_remote=True,
    trace_flags=TraceFlags(trace_flags),
)
context = set_span_in_context(NonRecordingSpan(parent_ctx))
root = self._sdk_tracer.start_span(
    name=f"invoke_agent {agent_name}",
    context=context,
    kind=SpanKind.INTERNAL,
)
```

**Nested cubepi runs** (an inner `agent.run()` inside a tool): propagation uses OTel's standard context, **not** `AgentContext.extra`. `extra` is JSON-serialized by the checkpointer (see `cubepi/checkpointer/sqlite.py`), and an SDK `Span` is not serializable; storing one there would crash on first checkpoint flush.

Mechanism: when the recorder opens `execute_tool`, it attaches the new span to the OTel context for the duration of that tool's body:

```python
import opentelemetry.context as context_api
from opentelemetry import trace

# inside Recorder.on_tool_exec_start
tool_span = self._sdk_tracer.start_span(...)
ctx_with_tool = trace.set_span_in_context(tool_span)
token = context_api.attach(ctx_with_tool)
self._tool_contexts[tool_call_id] = (tool_span, token)

# inside Recorder.on_tool_exec_end
tool_span, token = self._tool_contexts.pop(tool_call_id)
context_api.detach(token)
tool_span.end()
```

When the tool body calls `agent.run()` on an inner agent (with its own `Tracer`), that inner Tracer's root `start_span()` (called with no explicit `context=`) picks up the outer tool span as parent via SDK's default context lookup. `trace_id` is inherited automatically; `parent_span_id` becomes the outer tool span's id.

Two cubepi processes / agents that share the same Python process share the OTel `contextvars`, so this works across `Tracer` instances. For inner agents that run in a separate `asyncio` task (`asyncio.create_task(...)` inside a tool) the context inheritance is the standard Python contextvars behavior — the new task inherits the current context at creation time.

For cross-process nesting (inner agent runs in a subprocess or worker), the outer Tracer must serialize the W3C `traceparent` string and the inner host must call `tracer.run_scope(parent_trace_id=..., parent_span_id=...)` explicitly. See "Inbound trace context" above.

## 9. Event → Span Mapping

The `chat` span is **driven by provider listeners**, not by `MessageStartEvent` / `MessageEndEvent`. Reason: `MessageEndEvent` fires after `after_model_response` middleware hooks have run, so closing the span there would include hook time in the duration and would let stream errors leak (an exception during streaming can synthesize a failure message via the loop, but the partial `chat` span would never be closed). Driving from provider listeners ties the span lifetime to the actual HTTP roundtrip.

```
AgentStart                  → start_span("invoke_agent ...",        kind=INTERNAL, context=root_ctx)
TurnStart                   → start_span("cubepi.turn",              kind=INTERNAL, context=in(agent_span))
provider.on_request         → start_span("chat <model>",            kind=CLIENT,   context=in(turn_span))
                              + set request attrs, raw_request
provider.on_chunk (1st)     → record TTFT on chat span
provider.on_chunk (subseq.) → optional: emit time_per_output_chunk metric observation
provider.on_response (ok)   → set response/usage/raw_response attrs; chat_span.end()
provider.on_response (err)  → if isinstance(exc, asyncio.CancelledError):
                                  set chat_span.attribute("cubepi.aborted", true)
                                  set error.type = "cubepi.aborted"
                                  Status stays UNSET   (abort is not a failure)
                              else:
                                  add_event("gen_ai.client.operation.exception",...)
                                  set Status.ERROR; set error.type
                              chat_span.end()                          ← always closes
MessageStart [a]            → no-op for spans (already opened by on_request)
MessageEnd   [a]            → no-op for chat span (already closed by on_response).
                              Used by turn span to collect post-hook output.
ToolExecStart               → start_span("execute_tool <name>",     kind=INTERNAL, context=in(turn_span))
ToolExecEnd                 → set tool attrs; tool_span.end()
TurnEnd                     → set workflow output attrs; turn_span.end()
AgentEnd                    → set agent output attrs; agent_span.end()

MessageStart [u/t]          → recorded into turn span's `gen_ai.input.messages` buffer
MessageUpdate               → IGNORED (TTFT and chunk timing are sourced from provider.on_chunk, not from agent-level update events)
ToolExecUpdate              → IGNORED
```

`in(span)` is shorthand for `trace.set_span_in_context(span)`.

**`chat` span recorded output is pre-hook.** `gen_ai.output.messages` and `cubepi.llm.raw_response` on the `chat` span reflect what the provider actually returned. Any mutation performed by an `after_model_response` middleware is captured on the parent `cubepi.turn` span's `gen_ai.output.messages` instead. This preserves the rule "chat span describes the provider call; turn span describes cubepi's processed turn."

**No-provider-call case.** A middleware can inject a synthetic assistant message via `TurnAction.inject_messages` without ever calling the provider. In that case no `chat` span is opened — the `MessageStart` / `MessageEnd` pair has no associated provider events. The synthetic message still appears on the `turn span's `gen_ai.output.messages``.

**Recorder span stack.** The Recorder keeps a per-run map of open spans:
- `agent_span` (1 per run)
- `turn_span` (1 at a time per run)
- `chat_span` (1 at a time per run; opened by `on_request`, closed by `on_response`)
- `tool_spans: dict[tool_call_id, Span]` — parallel tools are keyed by `tool_call_id` so each execution has an unambiguous lifetime regardless of `contextvars` state.

**Provider listener implementations** (sketch):

```python
def on_provider_request(self, payload: dict, model: Model) -> None:
    turn_span = self._current_turn_span()
    if turn_span is None:
        return  # not in an attached run
    chat_span = self._sdk_tracer.start_span(
        name=f"chat {model.id}",
        kind=SpanKind.CLIENT,
        context=trace.set_span_in_context(turn_span),
    )
    self._chat_span = chat_span
    self._chat_open_ns = time.time_ns()
    chat_span.set_attribute("gen_ai.operation.name", "chat")
    chat_span.set_attribute("gen_ai.provider.name", map_provider(model.provider))
    chat_span.set_attribute("gen_ai.request.model", model.id)
    # ... other gen_ai.request.* from payload
    if self._record_content:
        chat_span.set_attribute("cubepi.llm.raw_request", json.dumps(payload))

def on_provider_chunk(self, event: StreamEvent, model: Model) -> None:
    if self._chat_span is None or self._first_chunk_recorded:
        return
    if event.type in ("text_delta", "thinking_delta", "toolcall_delta"):
        ttft_s = (time.time_ns() - self._chat_open_ns) / 1e9
        self._chat_span.set_attribute("gen_ai.response.time_to_first_chunk", ttft_s)
        self._first_chunk_recorded = True

def on_provider_response(
    self,
    body: dict | None,
    model: Model,
    exc: BaseException | None,    # BaseException to catch CancelledError
) -> None:
    if self._chat_span is None:
        return
    span = self._chat_span
    try:
        if body is not None:
            # Even on abort, the provider may have assembled a partial body
            # with stop_reason="aborted". Record what we have.
            self._record_response_attrs(span, body, model)
        if isinstance(exc, asyncio.CancelledError):
            span.set_attribute("cubepi.aborted", True)
            span.set_attribute("error.type", "cubepi.aborted")
            # status stays UNSET — abort is a control signal, not a failure
        elif exc is not None:
            span.set_status(Status(StatusCode.ERROR, str(exc)[:256]))
            span.set_attribute("error.type", cubepi_error_type_for(exc))
            span.add_event(
                name="gen_ai.client.operation.exception",
                attributes={
                    "exception.type": type(exc).__name__,
                    "exception.message": str(exc),
                    "exception.stacktrace": "".join(traceback.format_exception(exc)),
                },
            )
    finally:
        span.end()
        self._chat_span = None
        self._first_chunk_recorded = False
```

## 10. Span Attribute Schemas

Notation: **R** = Required, **CR** = Conditionally Required, **Rec** = Recommended, **Opt** = Opt-In.

### 10.1 `invoke_agent` (root)

| Attribute | Type | Status | Source |
|---|---|---|---|
| `gen_ai.operation.name` | string | **R** | `"invoke_agent"` |
| `gen_ai.provider.name` | string | **R** | provider id of first chat call, or `"cubepi"` if undetermined at root open |
| `gen_ai.agent.name` | string | CR | from `Tracer.agent_name` or `run_scope(extra_attrs=...)` |
| `gen_ai.agent.id` | string | CR | caller-supplied |
| `gen_ai.agent.description` | string | CR | caller-supplied |
| `gen_ai.agent.version` | string | CR | caller-supplied |
| `gen_ai.conversation.id` | string | CR | `thread_id` |
| `gen_ai.request.model` | string | CR | model id of first chat call (if pre-known) |
| `error.type` | string | CR (on error) | see §12.3 |
| `cubepi.run_id` | string | **R** | uuid generated at root open |
| `cubepi.thread_id` | string | Rec | same as `gen_ai.conversation.id`, duplicated for `cubepi.*` namespace ergonomics |
| `cubepi.agent.tools` | string[] | Opt | tool names registered |
| `cubepi.agent.system_prompt.sha256` | string | Opt | first 16 hex of `sha256(system_prompt)` |
| `cubepi.input.messages.count` | int | Rec | input message count |
| `cubepi.output.messages.count` | int | Rec | new messages produced |
| `cubepi.aborted` | bool | Opt | `true` if the run terminated via `asyncio.CancelledError` or assistant `stop_reason == "aborted"`. Status stays `UNSET`. |
| `gen_ai.system_instructions` | array | Opt | gated by `record_content` |
| `gen_ai.input.messages` | array | Opt | gated by `record_content` |
| `gen_ai.output.messages` | array | Opt | gated by `record_content` |

**Span name:** `invoke_agent {gen_ai.agent.name}` or `invoke_agent`. **Kind:** `INTERNAL`.

### 10.2 `cubepi.turn` (one turn)

cubepi-specific INTERNAL span; carries no `gen_ai.operation.name`. See §3.2 rationale.

| Attribute | Type | Status | Source |
|---|---|---|---|
| `error.type` | string | CR (on error) | see §12.3 |
| `gen_ai.input.messages` | array | Opt | messages going into this turn (cubepi extends opt-in content attrs to non-semconv spans) |
| `gen_ai.output.messages` | array | Opt | assistant message + any tool results (post-hook view, includes middleware mutations) |
| `cubepi.turn.index` | int | **R** | 0-based |
| `cubepi.turn.stop_reason` | string | Rec | cubepi-normalized: `stop` / `tool_use` / `length` / `error` / `aborted` (see `cubepi/providers/anthropic.py` `stop_reason_map`) |
| `cubepi.turn.tool_calls.count` | int | Rec | count |
| `cubepi.turn.terminated_by_tool` | bool | Opt | a tool returned `terminate=True` |

**Span name:** `cubepi.turn`. **Kind:** `INTERNAL`.

### 10.3 `chat` (LLM call)

| Attribute | Type | Status | Source |
|---|---|---|---|
| `gen_ai.operation.name` | string | **R** | `"chat"` |
| `gen_ai.provider.name` | string | **R** | mapped `Model.provider` (see §10.6) |
| `gen_ai.request.model` | string | CR | `Model.id` |
| `gen_ai.response.model` | string | Rec | from provider response |
| `gen_ai.response.id` | string | Rec | from provider response |
| `gen_ai.response.finish_reasons` | string[] | Rec | semconv-standard values: `["stop"]`, `["tool_use"]`, `["length"]`, `["content_filter"]`, `["error"]`. cubepi providers already normalize to these (see `stop_reason_map` in `anthropic.py`). The cubepi-synthetic value `"aborted"` is NOT emitted here — abort is tracked separately via the `cubepi.aborted` attribute on `chat` and `invoke_agent` spans. |
| `gen_ai.request.max_tokens` | int | Rec | `StreamOptions` |
| `gen_ai.request.temperature` | double | Rec | `StreamOptions` |
| `gen_ai.request.top_p` | double | Rec | `StreamOptions` |
| `gen_ai.request.top_k` | double | Rec | `StreamOptions` |
| `gen_ai.request.stop_sequences` | string[] | Rec | `StreamOptions` |
| `gen_ai.request.frequency_penalty` | double | Rec | `StreamOptions` |
| `gen_ai.request.presence_penalty` | double | Rec | `StreamOptions` |
| `gen_ai.request.seed` | int | CR | `StreamOptions` |
| `gen_ai.request.stream` | bool | CR | `true` |
| `gen_ai.request.choice.count` | int | CR | if `≠ 1` |
| `gen_ai.output.type` | string | CR | `"text"` / `"json"` / `"image"` |
| `gen_ai.response.time_to_first_chunk` | double | Rec | seconds between request send and first delta |
| `gen_ai.usage.input_tokens` | int | Rec | from `Usage` (Anthropic: reconciled, see §11.1) |
| `gen_ai.usage.output_tokens` | int | Rec | from `Usage` |
| `gen_ai.usage.reasoning.output_tokens` | int | Opt | extended-thinking tokens |
| `gen_ai.usage.cache_read.input_tokens` | int | Opt | Anthropic / OpenAI cache hit |
| `gen_ai.usage.cache_creation.input_tokens` | int | Opt | Anthropic cache write |
| `server.address` | string | Rec | host of LLM API endpoint |
| `server.port` | int | CR | if non-default port |
| `error.type` | string | CR (on error) | see §12.3 |
| `gen_ai.input.messages` | array | Opt | gated by `record_content` (§10.5) |
| `gen_ai.output.messages` | array | Opt | gated by `record_content` (§10.5) |
| `gen_ai.system_instructions` | array | Opt | gated by `record_content` |
| `gen_ai.tool.definitions` | array | Opt | gated by `record_content` |
| `cubepi.llm.thinking_level` | string | Opt | `off` / `low` / `medium` / `high` |
| `cubepi.llm.raw_request` | string | Opt | full provider request, JSON-serialized — gated by `record_content` |
| `cubepi.llm.raw_response` | string | Opt | full provider response, JSON-serialized — gated by `record_content` |

Plus provider-specific opt-in attributes from §11.

**Span name:** `chat {gen_ai.request.model}`. **Kind:** `CLIENT`.
**Lifetime:** opened by `provider.on_request`, closed by `provider.on_response` (success or failure). Duration reflects the actual HTTP roundtrip, **not** post-stream middleware processing time. See §9.

### 10.4 `execute_tool` (cubepi-side)

| Attribute | Type | Status | Source |
|---|---|---|---|
| `gen_ai.operation.name` | string | **R** | `"execute_tool"` |
| `gen_ai.tool.name` | string | **R** | tool name |
| `gen_ai.tool.call.id` | string | Rec | `tool_call_id` from assistant message |
| `gen_ai.tool.type` | string | Rec | `"function"` |
| `gen_ai.tool.description` | string | Rec | from `AgentTool.description` |
| `gen_ai.tool.call.arguments` | any | Opt | gated by `record_content` |
| `gen_ai.tool.call.result` | any | Opt | gated by `record_content` |
| `error.type` | string | CR (on error) | see §12.3 |
| `cubepi.tool.execution_mode` | string | Rec | `"parallel"` / `"sequential"`. Recorder reads `AgentTool.execution_mode` from the agent's tool registry at `ToolExecutionStartEvent`, falling back to the agent-level `tool_execution` parameter. Not carried on the event itself. |
| `cubepi.tool.is_error` | bool | Rec | mirrors `AgentToolResult.is_error` |
| `cubepi.tool.terminate` | bool | Opt | tool returned `terminate=True` |
| `cubepi.tool.blocked_by_hook` | bool | Opt | `before_tool_call` blocked the call |
| `cubepi.tool.block_reason` | string | Opt | reason string |

**Span name:** `execute_tool {gen_ai.tool.name}`. **Kind:** `INTERNAL`.

### 10.5 `gen_ai.input.messages` / `output.messages` content schema

Opt-in. When `Tracer(record_content=True)`, the recorder serializes messages into the OTel GenAI messages JSON schema:

```jsonc
[
  {
    "role": "user" | "assistant" | "tool" | "system",
    "parts": [
      { "type": "text", "content": "..." },
      { "type": "tool_call", "id": "...", "name": "...", "arguments": { ... } },
      { "type": "tool_call_response", "id": "...", "result": "..." },
      { "type": "reasoning", "content": "..." }
    ]
  }
]
```

Mapping from cubepi `Content` union (`cubepi/providers/base.py`):

| cubepi type | semconv `part` |
|---|---|
| `TextContent` | `{"type": "text", "content": <text>}` |
| `ThinkingContent` | `{"type": "reasoning", "content": <text>}` — `reasoning` is **not yet a standardized part type** in semconv; documented as a cubepi extension (the spec does standardize `gen_ai.usage.reasoning.output_tokens`). |
| `ToolCall` | `{"type": "tool_call", "id": <id>, "name": <name>, "arguments": <args>}` |
| `ToolResultMessage` content | `{"type": "tool_call_response", "id": <tool_call_id>, "result": <content>}` |

### 10.6 `gen_ai.provider.name` value mapping

| cubepi `Model.provider` | `gen_ai.provider.name` |
|---|---|
| `anthropic` | `"anthropic"` |
| `openai` | `"openai"` |
| `azure_openai` | `"azure.ai.openai"` |
| `gemini` | `"gcp.gemini"` |
| `vertex_ai` | `"gcp.vertex_ai"` |
| `bedrock` | `"aws.bedrock"` |
| `cohere` | `"cohere"` |
| `mistral` | `"mistral_ai"` |
| `groq` | `"groq"` |
| `xai` | `"x_ai"` |
| `deepseek` | `"deepseek"` |
| `perplexity` | `"perplexity"` |
| `watsonx` | `"ibm.watsonx.ai"` |
| (unknown) | the raw `Model.provider` string, prefixed with `"unknown:"` |

## 11. Provider-Specific Attributes

### 11.1 Anthropic

`gen_ai.provider.name = "anthropic"`.

**Token reconciliation** — per the Anthropic semconv page, `input_tokens` from the API **excludes** cached tokens. The recorder must compute:

```python
gen_ai_input_tokens = (
    response.usage.input_tokens
    + response.usage.cache_read_input_tokens
    + response.usage.cache_creation_input_tokens
)
```

This is the value emitted on `gen_ai.usage.input_tokens`. Cache fields go on their own attributes unchanged.

### 11.2 OpenAI

`gen_ai.provider.name = "openai"` (or `"azure.ai.openai"`).

| Attribute | Type | Source |
|---|---|---|
| `openai.api.type` | string | `"chat_completions"` / `"responses"` |
| `openai.request.service_tier` | string | `"auto"` / `"default"` |
| `openai.response.service_tier` | string | `"scale"` / `"default"` |
| `openai.response.system_fingerprint` | string | from response |

OpenAI uses `gen_ai.usage.cache_read.input_tokens` for prompt caching. The OpenAI semconv page also lists `gen_ai.usage.cache_creation.input_tokens`; emit it only when the provider response actually exposes the field (current OpenAI responses do not — Anthropic does).

### 11.3 Other providers

Common `gen_ai.*` attributes only. Provider-specific raw fields are preserved in `cubepi.llm.raw_response`.

## 12. Exceptions & Errors

### 12.1 LLM API errors

When the provider returns an error, raises a typed exception, or times out, the `chat` span:

```python
chat_span.set_status(Status(StatusCode.ERROR, description=str(exc)[:256]))
chat_span.set_attribute("error.type", cubepi_error_type_for(exc))
chat_span.add_event(
    name="gen_ai.client.operation.exception",
    attributes={
        "exception.type": type(exc).__name__,
        "exception.message": str(exc),
        "exception.stacktrace": traceback.format_exc(),
    },
)
```

Note: we add a **named event** (`gen_ai.client.operation.exception`), distinct from the standard `exception` event. Per the GenAI semconv exceptions page, this is the dedicated event for LLM client failures.

### 12.2 cubepi-internal Python exceptions

For exceptions inside tool execution or cubepi-internal code (not LLM-provider failures), use SDK's built-in:

```python
tool_span.record_exception(exc, escaped=True)
tool_span.set_status(Status(StatusCode.ERROR, description=str(exc)[:256]))
tool_span.set_attribute("error.type", type(exc).__name__)
```

`record_exception` produces the standard `exception` event with `exception.type` / `exception.message` / `exception.stacktrace` / `exception.escaped`.

### 12.3 `error.type` value conventions

The semconv does not prescribe a closed enum. cubepi conventions:

| Situation | `error.type` value |
|---|---|
| Provider HTTP 4xx/5xx | `"<provider>.<status_code>"` e.g. `"anthropic.429"` |
| Provider client class raised | fully-qualified Python class name, e.g. `"anthropic.RateLimitError"` |
| Network timeout | `"timeout"` |
| Connection refused | `"connection_error"` |
| Tool raised business error | tool's exception class name |
| Agent aborted | `"cubepi.aborted"` |
| Agent error stop_reason | `"cubepi.<stop_reason>"` |

### 12.4 Status mapping

| Situation | Span `status.code` |
|---|---|
| Healthy completion | `UNSET` (do not affirmatively set `OK`) |
| Tool `is_error == true` | `ERROR` on `execute_tool`; parent stays `UNSET` |
| Assistant `stop_reason == "error"` | `ERROR` on `chat`, bubble to parent `cubepi.turn` |
| Assistant `stop_reason == "aborted"` | `UNSET` + `cubepi.aborted = true` attribute on both `chat` and `invoke_agent` |
| `asyncio.CancelledError` during stream | `UNSET` + `cubepi.aborted = true` + `error.type = "cubepi.aborted"` on `chat`; span closes via provider's `finally` block (§3.4); the `CancelledError` is re-raised after listeners fire |
| Exception during provider call | `ERROR` on `chat` |
| Exception during tool execution | `ERROR` on `execute_tool` |

## 13. Metrics

Histograms emitted independently of spans, via `opentelemetry-sdk-metrics`. Phase 4 in the roadmap; pinned here so consumers know the field names ahead of time.

```python
from cubepi.tracing import Meter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

meter = Meter(
    resource=tracer.resource,
    exporter=OTLPMetricExporter(endpoint="http://collector:4318/v1/metrics"),
    export_interval_millis=60_000,
)
meter.attach(tracer)
```

Internally `Meter` wires the exporter through a `PeriodicExportingMetricReader` (this is the SDK's required wrapper; passing a raw exporter to `MeterProvider` doesn't work):

```python
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

reader = PeriodicExportingMetricReader(
    exporter=exporter,
    export_interval_millis=export_interval_millis,
)
provider = MeterProvider(resource=resource, metric_readers=[reader])
self._otel_meter = provider.get_meter(
    name="cubepi.tracing",
    version=cubepi.__version__,
    schema_url=SCHEMA_URL,
)
```

`Meter` has its own `force_flush()` / `shutdown()` mirroring `Tracer.6.1` and is closed automatically when `Tracer.shutdown()` is called if `meter.attach(tracer)` was invoked.

### 13.1 Histograms emitted

| Metric | Unit | When | Required attrs |
|---|---|---|---|
| `gen_ai.client.operation.duration` | `s` | every `chat` / `execute_tool` / `invoke_agent` close (NOT `cubepi.turn` — turn span has no `gen_ai.operation.name`) | `gen_ai.operation.name`, `gen_ai.provider.name` |
| `gen_ai.client.token.usage` | `{token}` | one observation per token type on `chat` close | `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.token.type` (`input` / `output`) |
| `gen_ai.client.operation.time_to_first_chunk` | `s` | `chat` close (streaming) | `gen_ai.operation.name`, `gen_ai.provider.name` |
| `gen_ai.client.operation.time_per_output_chunk` | `s` | `chat` close (streaming, > 1 chunk) | `gen_ai.operation.name`, `gen_ai.provider.name` |

Recommended attrs on all: `gen_ai.response.model`, `server.address`.

### 13.2 MCP histograms

When MCP is active:

| Metric | Unit | When | Required attrs |
|---|---|---|---|
| `mcp.client.operation.duration` | `s` | every `tools/call` close | `mcp.method.name` |
| `mcp.client.session.duration` | `s` | MCP session close | `mcp.session.id`, `mcp.protocol.version` |

Suggested bucket boundaries (from semconv): `[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60, 120, 300]` s.

## 14. MCP Integration

cubepi has an MCP client at `cubepi/mcp/`. When the agent calls a tool backed by MCP, a child `tools/call` CLIENT span wraps the network call.

```
execute_tool {tool_name}            INTERNAL          ← cubepi side
└── tools/call {tool_name}          CLIENT            ← MCP wire
```

The MCP client uses a helper from `cubepi.tracing.mcp`:

```python
from cubepi.tracing.mcp import mcp_client_span

async with mcp_client_span(
    method="tools/call",
    target=tool_name,
    session_id=session.id,
    protocol_version=session.protocol_version,
    server_address=session.server_url,
) as span:
    # The helper has already injected traceparent into params._meta via
    # TraceContextTextMapPropagator before sending.
    span.set_attribute("jsonrpc.request.id", request_id)
    response = await mcp_client.send(...)
```

Span attributes:

| Attribute | Status |
|---|---|
| `mcp.method.name` | **R** |
| `gen_ai.tool.name` | CR (when method is `tools/call`) |
| `mcp.protocol.version` | Rec |
| `mcp.session.id` | Rec |
| `jsonrpc.request.id` | CR |
| `gen_ai.operation.name` | Rec (`"execute_tool"`) |
| `server.address` | Rec |
| `server.port` | CR |
| `error.type` | CR (on error) |

The helper is a no-op when no `Tracer` is attached. Discovery is via a contextvar that `Tracer.attach` sets to the current SDK `TracerProvider`.

**W3C propagation.** The helper injects `traceparent` (and `tracestate` if set) into the MCP message's `params._meta` field, per the MCP semconv recommendation, so an instrumented MCP server can continue the trace.

## 15. Exporters

### 15.1 `JsonlSpanExporter`

```python
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace import ReadableSpan

class JsonlSpanExporter(SpanExporter):
    def __init__(self, directory: str = "./cubepi-traces"):
        self._dir = Path(directory)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        # Group by run_id (read from cubepi.run_id attribute on each span).
        # Append each span's OTLP/JSON encoding to {dir}/{date}/{run_id}.jsonl.
        for span in spans:
            run_id = span.attributes.get("cubepi.run_id", "no-run")
            path = self._dir / date.today().isoformat() / f"{run_id}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                f.write(span.to_json(indent=None))
                f.write("\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None: ...
```

`Span.to_json()` is provided by the SDK and produces OTLP/JSON-shaped output. Each line is **one span** (not a `ResourceSpans` envelope) — simpler and the dominant convention for jsonl trace storage. Consumers (cubepi-trace) reconstruct the trace by joining lines with the same `traceId`.

For full OTLP/JSON ResourceSpans-per-line format, consumers can run the file through the OTel Collector's `fileexporter` reader.

### 15.2 `OTLPSpanExporter`

Direct re-export from `opentelemetry-exporter-otlp-proto-http`. cubepi adds no wrapping:

```python
# in cubepi/tracing/exporters/__init__.py
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

__all__ = ["JsonlSpanExporter", "OTLPSpanExporter"]
```

This exporter handles protobuf encoding, retries (with exponential backoff), gzip compression, auth headers, and gRPC alternative endpoint configuration — none of which we reimplement.

## 16. Resource Attributes

| Attribute | Status | Note |
|---|---|---|
| `service.name` | Rec | from `Tracer.service_name` |
| `service.namespace` | Opt | from `Tracer.service_namespace` |
| `service.version` | Opt | from `Tracer.service_version` |
| `service.instance.id` | Opt | uuid generated at Tracer init if not supplied |
| `deployment.environment.name` | Opt | from `Tracer.deployment_environment` |
| `gen_ai.agent.name` | Opt | process-level |
| `gen_ai.agent.id` | Opt | process-level |
| `gen_ai.agent.description` | Opt | process-level |
| `gen_ai.agent.version` | Opt | process-level |
| `telemetry.sdk.name` | auto | `"opentelemetry"` (set by SDK) |
| `telemetry.sdk.language` | auto | `"python"` (set by SDK) |
| `telemetry.sdk.version` | auto | set by SDK |

cubepi could add `telemetry.distro.name = "cubepi.tracing"` and `telemetry.distro.version = cubepi.__version__` to be discoverable as a custom distro — TBD.

## 17. Storage / Query Boundary

cubepi writes; cubepi-trace (separate repo) reads. Wire format:

- jsonl files (one OTLP-JSON span per line) for local.
- OTLP protobuf over HTTP for production.

Schema reference: this document + the pinned `SCHEMA_URL`.

cubepi-trace **must not** import from `cubepi.tracing`. It parses OTLP/JSON with its own pydantic models. When a breaking change is needed: bump cubepi major version, update cubepi-trace in lockstep.

## 18. Repository Plan

| Item | Where |
|---|---|
| `cubepi.tracing.*` | this repo, `cubepi/tracing/` |
| Schema constants | this repo, `cubepi/tracing/schema.py` + this doc |
| Query backend (FastAPI, reads jsonl/OTLP) | new repo `cubepi-trace` |
| Query frontend (React) | same `cubepi-trace` repo |

## 19. Implementation Phases

0. **Phase 0 (prerequisite cubepi changes):** Both extensions in §3.4 ship before any of `cubepi/tracing/` is written.
   - **3.4.1 Provider listener registry** in `cubepi/providers/`: convert `Provider` from Protocol to base class with listener registry; add `subscribe_request` / `subscribe_chunk` / `subscribe_response`; thread listener invocations through `anthropic.py`, `openai.py`, `openai_responses.py`, `faux.py`. Tests verifying listeners fire in the right order and exceptions don't leak.
   - **3.4.2 `ToolExecutionEndEvent` extension** in `cubepi/agent/`: add `terminate`, `blocked_by_hook`, `block_reason` fields; populate from `cubepi/agent/tools.py`. Backward compatible (additive, conservative defaults).
1. **Phase 1 (MVP):** `Tracer`, `Recorder`, `JsonlSpanExporter`. Span lifecycle for `invoke_agent` / `cubepi.turn` / `chat` / `execute_tool`. `gen_ai.*` attribute set complete; `record_content=False` only. Provider listener subscriptions wired.
2. **Phase 2:** `record_content=True` path. `gen_ai.input.messages` / `output.messages` / `system_instructions` / `tool.definitions`. Redaction hook.
3. **Phase 3:** `OTLPSpanExporter` wiring + docs. Provider-specific Anthropic / OpenAI attributes.
4. **Phase 4:** `Meter` + metrics histograms. `cubepi.tracing.mcp` helper. MCP CLIENT spans + W3C propagation.
5. **Phase 5 (deferred):** `SqliteSpanExporter`, OpenSearch exporter, sampling, span links for cross-run causality.

## 20. Open Questions

- **Redaction hook signature.** `redact: Callable[[Span], None]` invoked just before export, mutating in place? Or per-attribute callback at write time? Phase 2 will decide.
- **Span linking for fire-and-forget inner agents.** Currently we use parent-child. If a tool spawns a background inner agent without `await`, parent-child is wrong — that's `SpanLink` territory. Add when use case appears.
- **Async tool spans in parallel mode.** The recorder keys its span stack by `tool_call_id` rather than relying on `contextvars`. Needs explicit test coverage.
- **`MetricsExporter` lifecycle.** Phase 4 will decide whether `Meter` owns its own background flush loop separate from `Tracer`'s per-run flush. Likely yes.
- **Schema URL bump cadence.** Pin and audit on bump once GenAI semconv graduates from Experimental.
- **Should `record_exception` be used for `gen_ai.client.operation.exception` too?** SDK's `record_exception` always emits the event name `"exception"`. To emit `"gen_ai.client.operation.exception"`, we call `span.add_event(...)` manually. We do not extend `record_exception` — both paths are explicit and the recorder picks based on whether the error originated from the provider client.
