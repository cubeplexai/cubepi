# Changelog

All notable changes to CubePi are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`on_run_end` middleware hook** — fires exactly once after all turns and tool
  calls complete, before `AgentEndEvent`. Return a `list[Message]` to inject
  additional messages and run one extra model turn (e.g. a memory-reflection
  pass); return `None` to do nothing.
  - `_reflection_fired` guard in the loop prevents the injected turn from
    re-triggering `on_run_end`.
  - `should_stop_after_turn` and `turn_action.decision == "stop"` paths now
    route through `on_run_end` before emitting `AgentEndEvent`. Error/aborted
    runs (`stop_reason in ("error", "aborted")`) skip the hook.
  - HITL-interrupted runs (HitlDetached / HitlAborted) also skip the hook —
    the conversation is paused, not finished.

## [0.6.0] - 2026-05-31

### Added

- **Human-in-the-Loop (HITL)** — first-class suspend/resume for agent runs that
  need a human decision mid-flight:
  - `HitlChannel` protocol with `InMemoryChannel` and `CheckpointedChannel`
    implementations. `CheckpointedChannel` persists pending requests so an
    agent can be resumed after a process restart.
  - `Agent(channel=...)` wiring; `agent.respond()`, `agent.detach()`, and
    `agent.abort_pending()` for external controllers.
  - `ApprovalPolicyMiddleware` and `ConfirmToolCallMiddleware` for declarative
    tool-call gating; `ask_user` built-in tool for open-ended prompts.
  - `ScriptedChannel` and `NoopChannel` for deterministic testing.
  - `HitlRequestEvent`, `HitlAnswerEvent`, `AgentSuspendedEvent`,
    `AgentAbortedEvent` stream events.
  - Lazy OTel `hitl.ask` / `hitl.confirm` spans with outcome attributes.
  - Pending-request persistence on Memory, SQLite, Postgres, and MySQL
    checkpointers (schema v2 — additive migration, backwards compatible).
- **`MySQLCheckpointer`** — full-featured MySQL/MariaDB checkpointer with
  Alembic schema management, matching the Postgres implementation.
- **Stream recording + `trace convert`** — `record_stream()` captures a raw
  provider `MessageStream` to JSONL; `cubepi trace convert` replays the
  recording as a structured trace. Useful for offline debugging and testing
  without live API calls.
- **Trace CLI — run-metadata filtering** (`--meta` / `--show-meta`): filter
  and display `tracing_context` key/value tags attached to a run.
- **Trace CLI — span IDs in `trace view`** node labels for easier
  cross-referencing with external OTLP backends.
- **Agent steering by ID** — `agent.steer(...)` returns a `steer_id`; pass it
  to `agent.cancel_steer(steer_id)` to cancel a not-yet-drained steering
  message.

### Fixed

- **OpenAI provider**: deduplicate `finish_reason` processing that caused
  spurious extra events on streamed responses.
- **Agent**: `ToolExecutionStartEvent` is now deferred until the tool coroutine
  is actually scheduled, preventing early events for tools that are never run.
- **Tracing**: clear `stream_tool_accumulated` at turn start to avoid stale
  tool-call data leaking across turns.

## [0.5.0] - 2026-05-25

### Added

- **Image generation subsystem** (`cubepi.providers.images`): a pluggable,
  per-vendor model interface for image generation, with a class factory so new
  backends slot in without touching call sites.
- **`CapabilityDescriptor`**: a declarative, per-model description of what a
  model supports — temperature mode (free / fixed / ignored), reasoning-level
  mapping (int budget / effort / enum), and the `max_tokens` field name. It now
  drives the OpenAI, OpenAI Responses, Anthropic, and DeepSeek providers, and is
  exported from the top-level package.
- **`cubepi trace` CLI** (install with the `trace-cli` extra): discover, list,
  view, follow, and aggregate stats over local agent-run traces, with rich
  rendering and run-id prefix matching.
- **Tracing**: an OTLP exporter and a best-effort `trace()` scope helper. The
  tracing package now imports cleanly without `opentelemetry` installed.
- **Self-describing provider errors** that carry provider / model / cause
  context for easier debugging.

### Changed

- Provider reasoning/thinking and temperature handling is now driven by
  `CapabilityDescriptor` instead of per-provider ad-hoc payload quirks, giving
  consistent behavior across OpenAI, Anthropic, and DeepSeek.

### Fixed

- **Anthropic**: merge parallel tool results into a single user message; carry
  parsed tool arguments through the streaming `toolcall_end` event; compute
  `max_tokens` from the actual capability budget; honor per-request
  `thinking_budgets` overrides.
- **Agent loop / steering**: drain steering at the turn boundary;
  `after_model_response` now injects after tool results; backfill tool results
  for tool calls orphaned by a cancel.
- **DeepSeek**: correct reasoning-effort path and temperature range handling.

## Earlier releases

- **[0.4.0]** - 2026-05-19 — see the [release notes](https://github.com/cubeplexai/cubepi/releases/tag/v0.4.0).
- **[0.3.0]** - 2026-05-14 — see the [release notes](https://github.com/cubeplexai/cubepi/releases/tag/v0.3.0).
- **[0.2.0]** - 2026-05-10 — see the [release notes](https://github.com/cubeplexai/cubepi/releases/tag/v0.2.0).
- **[0.1.0]** - 2026-05-09 — initial release. See the [release notes](https://github.com/cubeplexai/cubepi/releases/tag/v0.1.0).

[Unreleased]: https://github.com/cubeplexai/cubepi/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/cubeplexai/cubepi/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/cubeplexai/cubepi/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/cubeplexai/cubepi/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cubeplexai/cubepi/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/cubeplexai/cubepi/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cubeplexai/cubepi/releases/tag/v0.1.0
