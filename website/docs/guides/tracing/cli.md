---
title: The trace CLI
---

# Inspecting traces with `cubepi trace`

The `JsonlSpanExporter` writes one file per trace under
`./cubepi-traces/<date>/<trace_id>.jsonl`. The `cubepi trace` CLI (provided by
the `trace-cli` extra) reads those files so you can see exactly what a run did
— which LLM and tool calls fired, in what order, what each returned, where it
errored, and the token counts — without re-running it.

```bash
pip install 'cubepi[trace-cli]'      # or: uv sync --extra trace-cli
cubepi trace --help
```

`--dir` defaults to `./cubepi-traces`; pass `--dir <path>` if your traces live
elsewhere. Each file is one **trace**: the run plus any nested subagent runs,
which inherit the parent's `trace_id` and so land in the same file.

## `ls` — list recent traces

```bash
cubepi trace ls          # newest first; -n N to limit
```

| column | meaning |
|---|---|
| `started` | trace start time (UTC) |
| `trace_id` | the id you pass to `view` / `follow` / `stats` |
| `spans` | span count for the whole trace (incl. subagents) |
| `status` | `ok` or `error` |
| `duration` | wall-clock span of the trace |
| `input` | the user's prompt, to identify the run |

### Filter by run metadata (`--meta`)

If the host stamped run-scoped metadata onto the trace (via
`tracing_context(metadata=…)` — e.g. cubebox records `conversation_id`,
`user_id`, `org_id`, `workspace_id` on the root `invoke_agent` span), filter to
just those traces:

```bash
cubepi trace ls --meta conversation_id=conv_123
cubepi trace ls --meta user_id=usr_9 --meta org_id=org_1   # repeatable = AND, exact match
```

Each `--meta KEY=VALUE` is matched exactly against the trace's root metadata;
repeating the flag ANDs the conditions.

To **display** metadata values as columns (rather than only filter by them),
add `--show-meta KEY[,KEY…]`:

```bash
cubepi trace ls --show-meta conversation_id,user_id
cubepi trace ls --meta org_id=org_1 --show-meta conversation_id   # filter + show
```

(Or see all of a single trace's metadata with `cubepi trace view <id> -v`.)

## `view` — render a trace as a span tree

A trace-id **prefix** is enough (the table truncates ids); an ambiguous prefix
lists candidates.

```bash
cubepi trace view 1cd97cdb
```

```
trace
└── invoke_agent  14425.8ms  [0x1cd97cdb]
    ├── cubepi.turn  1283.1ms  [0x5cfda93e]
    │   ├── chat claude-sonnet-4-5  1208.7ms  tok 6845/68  [0x0d130229]
    │   └── execute_tool subagent  9610.2ms  subagent  [0x38bdd10a]
    │       └── invoke_agent  9601.0ms  [0x8094f99b]      ← subagent run, nested
    │           └── cubepi.turn  9598.4ms  [0x57c5cfc7]
    │               ├── chat claude-sonnet-4-5  1190.3ms  [0x8205ca6b]
    │               └── execute_tool web_search  6500.2ms  web_search  [0xca4e59fc]
    └── cubepi.turn  491.9ms  [0xce25f242]
        └── chat claude-sonnet-4-5  427.2ms  [0x0bff68ec]
```

Read it top-down: `invoke_agent` (a run) → `cubepi.turn` (one agent-loop turn)
→ `chat <model>` (an LLM call, with `tok <input>/<output>`) and
`execute_tool <name>` (a tool call). A **subagent** shows up as
`execute_tool subagent` with its own `invoke_agent → cubepi.turn → …` nested
directly beneath it. The `[0x…]` suffix on each node is the span's `span_id` —
grep it in the raw JSONL to inspect that exact span. Errors print inline under
the failing span.

Flags:

```bash
cubepi trace view <id> --content   # expand gen_ai prompts / tool args / results
cubepi trace view <id> -v          # expand ALL span attributes (verbose, large)
```

`--content` requires the run to have been recorded with
`record_content=True` (see [Content & Redaction](./content-recording)).

## `follow` — watch a trace live

```bash
cubepi trace follow <id>           # polls as spans complete; good for a run in progress
```

## `stats` — aggregate across traces

```bash
cubepi trace stats --by model                  # latency p50/p95, error rate, tokens
cubepi trace stats --by tool --since 2026-05-20
```

`stats` also accepts `--meta KEY=VALUE` (same semantics as `ls`) to aggregate
only the traces that match — e.g. latency / error-rate / tokens for one user or
conversation:

```bash
cubepi trace stats --by model --meta user_id=usr_9
cubepi trace stats --by tool --meta conversation_id=conv_123
```

## Beyond the CLI

The files are plain JSONL — one span per line — so you can parse them directly
(`jq`, `python -c`) to pull a specific attribute (`gen_ai.usage.*`,
`gen_ai.tool.call.result`, `gen_ai.input.messages`, …). Error detail lives in a
span **event** named `gen_ai.client.operation.exception`.

:::tip For AI agents
A bundled `cubepi-trace` skill drives this CLI for debugging ("why did the run
end with no reply?", "the tool result is wrong"). It encodes the fast path
(`ls` → `view <prefix>`) and the token/cache-rate conventions.
:::
