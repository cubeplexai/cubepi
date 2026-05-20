---
name: cubepi-trace
description: Use when debugging a cubepi/cubebox agent run — a missing final reply, a tool that misbehaved, a 4xx from the model, wrong token/cache numbers, or "why did the agent do that?". Reads the JSONL traces cubepi writes per run via the `cubepi trace` CLI (ls / view / follow / stats) to inspect the span tree, errors, tool inputs/outputs, and token usage. Triggers on phrases like "查一下这个 run", "trace 一下", "为什么最后一轮没回复", "工具结果不对", "看 trace".
---

# Debugging cubepi runs with `cubepi trace`

cubepi can record every agent run as an OpenTelemetry span tree, written to
local JSONL files. The `cubepi trace` CLI reads those files so you can see
exactly what happened inside a run without re-running it: which LLM calls and
tool calls fired, in what order, what each returned, where it errored, and the
token/cache counts. Reach for this **before** guessing at a bug.

## When this applies

- The final answer is missing, truncated, or the run "ended" with no reply.
- A tool (web_search, web_fetch, execute, MCP tool, …) returned something wrong
  or empty, or the agent looped.
- The model returned a 4xx/5xx (e.g. `BadRequestError`), or the run status is
  `error`.
- Token / cache / cost numbers look wrong.
- You just need to understand the agent's actual trajectory for a given input.

## Prerequisites (one-time)

Tracing must be **on** and **recording content** for the run you want to
inspect. In cubebox it's config-driven (dynaconf):

- `tracing.enabled: true`
- `tracing.directory: ./cubepi-traces`  (default; relative to the backend cwd)
- `tracing.record_content: true`  (needed to see prompts / tool args / results;
  off = you get the span tree and timings but no content)

Env override form: `CUBEBOX_TRACING__ENABLED=true`,
`CUBEBOX_TRACING__RECORD_CONTENT=true`, `CUBEBOX_TRACING__DIRECTORY=...`.
cubebox's `config.development` already enables this. Files land at
`<directory>/<YYYY-MM-DD>/<run_id>.jsonl` (a run that crosses UTC midnight is
split across two date dirs; the CLI merges them).

Run the CLI from the cubebox backend dir so it picks up the venv that has
cubepi installed (the `trace-cli` extra provides it):

```bash
uv run cubepi trace --help
```

`--dir` defaults to `./cubepi-traces`; pass `--dir <path>` if your traces live
elsewhere (e.g. a worktree).

## The fast path (this is 90% of debugging)

```bash
# 1. List recent runs, newest first. The `input` column shows the user's
#    message so you can find the right run; `status` flags errors.
uv run cubepi trace ls

# 2. View one run as a span tree. A run id PREFIX is enough (ls truncates ids).
#    Errors are printed inline under the failing span — no flags needed.
uv run cubepi trace view 66f1806f
```

`view` output looks like:

```
trace
└── invoke_agent  14425.8ms
    ├── cubepi.turn  1283.1ms
    │   ├── chat deepseek-v4-flash  1208.7ms  tok 6845/68
    │   └── execute_tool datetime  0.3ms  datetime
    └── cubepi.turn  491.9ms  ERROR
        └── chat deepseek-v4-flash  427.2ms  ERROR
            └── error: Error code: 400 - ... `tool_use` ids were found without
                `tool_result` blocks immediately after: call_01_...
```

Read the tree top-down: `invoke_agent` (whole run) → `cubepi.turn` (one
agent loop turn) → `chat <model>` (an LLM call, with `tok <input>/<output>`)
and `execute_tool <name>` (a tool call). An `ERROR` marker plus the inline
`error:` line usually tells you the root cause directly.

## Going deeper

```bash
# Expand the actual gen_ai content (prompts, tool args, tool results) —
# useful when the error isn't enough and you need to see what was sent.
uv run cubepi trace view <run> --content

# Expand ALL span attributes (verbose; includes full request bodies/tool
# schemas — large). Use when --content isn't enough.
uv run cubepi trace view <run> -v

# Watch a run live as spans complete (poll). Good for a run in progress.
uv run cubepi trace follow <run>

# Aggregate across runs: latency p50/p95, error rate, tokens — by model or tool.
uv run cubepi trace stats --by model
uv run cubepi trace stats --by tool --since 2026-05-20
```

If the CLI view still isn't enough, the files are plain JSONL — one span per
line — so you can parse them directly (e.g. with `python -c`/`jq`) to pull a
specific attribute. Useful attribute keys:

- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
  `gen_ai.usage.cache_read.input_tokens`, `gen_ai.usage.cache_creation.input_tokens`
- `gen_ai.tool.name`, `gen_ai.tool.call.result` (the tool's output),
  `gen_ai.input.messages`, `gen_ai.output.messages`
- error detail lives in a span **event** named `gen_ai.client.operation.exception`
  (`exception.type`, `exception.message`, `exception.stacktrace`); the span's
  `status.description` carries a truncated copy.

## Reading token / cache numbers

`input_tokens` is the **uncached** input; `cache_read.input_tokens` is what was
served from cache; they're disjoint. Total prompt = input + cache_read
(+ cache_creation). So cache hit rate = `cache_read / (input + cache_read)` —
dividing by `input` alone can exceed 100% and is wrong.

## Tips

- **Prefix match**: `view`/`follow`/`stats` accept a unique run-id prefix;
  copy the visible chars from `ls`. An ambiguous prefix lists candidates.
- **Find the run** by the `input` column in `ls` rather than guessing ids.
- **Backend vs frontend**: the trace shows what the backend/agent did. If the
  trace shows correct data but the UI is wrong, the bug is in the frontend
  (SSE handling / rendering) — the trace won't see that; use a browser instead.
