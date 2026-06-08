# Changelog

All notable changes to CubePi are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`FallbackBoundModel`** — built-in failover chain at the `BoundModel` level.
  Wrap an ordered `chain` of `BoundModel` instances; on `RateLimited`,
  `ProviderUnavailable`, or `ContextLengthExceeded` (configurable via
  `trigger_errors`), or on a first-event stream error, the next model in the
  chain is tried transparently. Optional `on_failover` callback for
  billing/metrics hooks. Exported from `cubepi` and `cubepi.providers`.

- **`DEFAULT_TRIGGER_ERRORS`** — `frozenset({RateLimited, ProviderUnavailable,
  ContextLengthExceeded})`. The default set of error types that trigger failover
  in `FallbackBoundModel`.

- **`BoundModel.generate()` / `BoundModel.stream()`** — the handle returned by
  `provider.model(...)` now drives a provider call directly. Useful for
  utilities (summarizers, classifiers) where you already hold a `BoundModel`
  and want to skip the agent loop:

  ```python
  bound = provider.model("claude-sonnet-4-6")
  reply = await bound.generate(
      messages=[UserMessage(content=[TextContent(text="hi")])],
      system_prompt="Be brief.",
  )
  ```

  Both methods forward to the bound provider with `model=bound.spec` and
  mirror the `Provider.generate` / `Provider.stream` signatures exactly.

### Breaking

- **`Middleware.extra_llm_calls()` returns `Iterable[BoundModel]`** instead
  of `Iterable[tuple[Provider, Model]]`. Third-party middleware overriding
  this hook must update the return shape (see Migration). The recorder
  consumer in `cubepi.tracing` was adapted in lock-step; built-in
  `CompactionMiddleware` already updated.
- **`cubepi.middleware.compaction.summarizer.summarize()` takes
  `model: BoundModel`** instead of separate `provider: Provider, model: Model`
  kwargs. Direct callers (rare — this is internal to `CompactionMiddleware`)
  must wrap the pair. The public `CompactionMiddleware(summary_model=...)`
  API is unchanged.
- **`cubepi.run_agent_loop` and `cubepi.run_agent_loop_continue` take
  `model: BoundModel`** instead of separate `provider: Provider, model: Model`
  kwargs. Stateless-loop callers driving the loop outside of `Agent` must
  update. The `Agent` API is unchanged — it already took `model: BoundModel`.

### Migration

- Middleware authors overriding `extra_llm_calls()`:

  ```python
  from cubepi.providers.base import BoundModel

  # Before
  def extra_llm_calls(self):
      return [(self._provider, self._model_spec)]

  # After — either build one explicitly…
  def extra_llm_calls(self):
      return [BoundModel(provider=self._provider, spec=self._model_spec)]

  # …or, if your middleware already holds a BoundModel (recommended),
  # just return it:
  def extra_llm_calls(self):
      return [self._bound_model]
  ```

- Direct `summarize()` callers (uncommon):

  ```python
  # Before
  await summarize(provider=provider, model=model_spec, ...)

  # After
  await summarize(model=BoundModel(provider=provider, spec=model_spec), ...)
  ```

- Stateless-loop callers (uncommon — most users build an `Agent`):

  ```python
  # Before
  await run_agent_loop(
      prompts=[...],
      context=ctx,
      provider=provider,
      model=model_spec,
      convert_to_llm=...,
      emit=...,
  )

  # After
  await run_agent_loop(
      prompts=[...],
      context=ctx,
      model=provider.model("id", ...),
      convert_to_llm=...,
      emit=...,
  )
  ```

## [0.8.0] - 2026-06-06

### Added

- **`@tool` decorator** — `cubepi.tool` builds an `AgentTool` from a plain
  async function: the input schema is generated from the typed parameters
  (honouring `Field(...)` metadata), the docstring becomes the description,
  and the loop-supplied `tool_call_id` / `signal` / `on_update` are injected
  only when declared. Tools may return a `str`, a `Content`, a `list` of
  content, or a full `AgentToolResult`. The longhand `AgentTool(...)` remains
  fully supported.
- **Conversation fork** — fork a thread at a completed-run boundary, or run
  a one-shot ephemeral continuation against a snapshot:
  - `Agent.fork(src, new, *, after_run_id, metadata=None)` — physical-copy
    fork at a completed-run boundary.
  - `Agent.fork_once(src, message, *, after_run_id) -> ForkOnceResult` —
    single-turn ephemeral continuation, no checkpointer writes.
  - `Agent.prompt(message, *, run_id=None) -> str` now accept-or-generates
    the `run_id` and returns it.
  - `Agent.state.active_run_id` exposes the in-flight `run_id`.
  - `Agent(messages=...)` constructor arg for ephemeral pre-seeded history.
  - `Checkpointer.snapshot`, `fork`, `claim_run`, `mark_run_complete`,
    `load_pending` Protocol methods.
  - `cubepi_runs` table per backend (Postgres / MySQL schema v3 → v4).
  - `Message.run_id: str | None` field on all three Message variants.
  - `HitlBinding` attribute on `AgentTool` / `Middleware`; `ask_user_tool`
    and `ApprovalPolicyMiddleware` populate it.

### Fixed

- **Tool results that set `is_error=True` are now reported as errors.** A tool
  body that returned `AgentToolResult(is_error=True)` without raising was
  surfaced to the model as a successful result; the flag is now honored on the
  execution path (affects both `@tool` and longhand `AgentTool` tools).

### Breaking

- `Agent.prompt()` return type changed from `None` to `str`. Callers ignoring
  the return value keep working.
- `Checkpointer` Protocol gained 5 new methods. Third-party v3-only
  checkpointers continue to work for vanilla `prompt()` via degraded mode;
  fork APIs raise `CheckpointerError` on such backends.

### Migration

- Postgres / MySQL: run the new alembic helper (see backend guides).
- SQLite: auto-migration at connect time.
- Legacy `run_id=NULL` messages remain readable; threads with only such
  messages are not forkable.

## [0.7.0] - 2026-06-05

### Added

- **`Provider.generate(...)` one-shot helper** — providers now expose a
  non-streaming call that consumes `stream()` and returns the final
  `AssistantMessage`. It supports per-call overrides for
  `max_output_tokens`, `temperature`, `thinking`, and `thinking_budgets`.
- **Compaction middleware** — `cubepi.middleware.CompactionMiddleware`
  summarizes older conversation turns into JSON-safe `AgentContext.extra`
  state and sends the model a compressed view while preserving full agent
  history.
- **Subagent middleware** — `cubepi.middleware.SubagentMiddleware` adds a
  `subagent` tool that runs an ephemeral child `Agent`, supports shared tools
  and middleware inheritance, captures child events, and exposes host callbacks
  for application-specific event streaming.
- **Typed provider error taxonomy** — built-in providers wrap SDK failures in
  `ProviderError` subclasses such as `ContextLengthExceeded`, `RateLimited`,
  `ProviderAuthFailed`, `ProviderUnavailable`, and `ProviderBadRequest`.
- **`Tracer.oneshot(...)`** — trace a single background LLM call without a full
  `Agent` loop; the trace CLI can filter these runs via
  `--meta oneshot_operation=...`.
- **Checkpointed HITL run IDs** — checkpointers now persist the owning
  `run_id` atomically with pending HITL requests so hosts can resume the
  correct detached run after a restart.
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
- **Compaction summarizer tracing** — `CompactionMiddleware`'s summary call is
  now first-class in the trace tree. A `cubepi.compaction.summarize` parent
  span (carrying `cubepi.compaction.message_count`) wraps the summarizer LLM
  call, and the recorder auto-subscribes the middleware's `summary_provider`
  so its `chat` span lands as a child:
  ```
  invoke_agent
  └── cubepi.turn
      ├── cubepi.compaction.summarize
      │   └── chat <summary-model>
      └── chat <main-model>
  ```
  Previously summarizer calls bypassed the trace entirely.
- **`Middleware.providers()` protocol** — middleware authors can override this
  method to expose any extra `BaseProvider` instances the middleware owns;
  `Recorder.attach()` walks `agent._middleware` and wires its listener registry
  on each one (id-deduped against `agent._provider`). Default is empty, so
  existing custom middleware needs no change.

### Changed

- **Breaking:** agent construction now takes a bound model from
  `provider.model(...)`. Replace
  `Agent(provider=provider, model=Model(...))` with
  `Agent(model=provider.model("model-id", ...))`. `provider_id` now lives on
  provider constructors and is copied into model metadata used for tracing,
  response metadata, and error messages. `Tracer.oneshot(...)`,
  `CompactionMiddleware`, and `SubagentMiddleware` use the same bound-model
  shape.
- **Breaking:** pre-model middleware hooks now receive `AgentContext` directly.
  Update custom middleware and explicit hook callables from the old signatures:
  - `transform_context(messages, *, signal=None)`
  - `transform_system_prompt(system_prompt, *, signal=None)`
  - `convert_to_llm(messages)`

  to the new signatures:
  - `transform_context(messages, *, ctx, signal=None)`
  - `transform_system_prompt(system_prompt, *, ctx, signal=None)`
  - `convert_to_llm(messages, *, ctx)`

  Use `ctx.extra` for middleware state that should survive checkpointing. This
  release does not include old-signature compatibility shims.
- **Breaking:** custom providers must implement `Provider.generate(...)` or
  inherit from `BaseProvider`, which supplies `generate()` by consuming
  `stream()`.
- **Breaking:** Postgres and MySQL checkpointer schemas are now version 3. Host
  applications must add the nullable `run_id` column to `cubepi_threads` and
  call `write_schema_version_op()` in their Alembic migration before using this
  release.
- **Breaking:** the `cubepi.providers.images` surface has been redesigned
  to align with the chat-provider 0.7 conventions: providers now take
  `provider_id` and an optional `ImagesCapabilityDescriptor`; models are
  built via `provider.model("id", ...)` (renamed `provider` field to
  `provider_id`, added `default_size/n/quality/output_format` and `cost`
  metadata); `ImagesContext` is typed (`size/n/quality/output_format/seed/
  negative_prompt/steps/guidance/extra`); per-call options live on a new
  `ImagesOptions` bag (`signal`, `on_payload`, `on_response`); failures
  raise `cubepi.errors.ProviderError` subclasses instead of in-band
  `AssistantImages.error_message`; the `create_images_provider` /
  `register_images_provider_class` registry is removed. The new shape
  reaches OpenAI, Doubao Seedream, SiliconFlow, and Together AI through
  a single `OpenAIImagesProvider` configured with the right
  `ImagesCapabilityDescriptor`.
- CI now runs `mypy cubepi` in addition to pytest and ruff.

### Fixed

- Tool argument `ValidationError`s are formatted as model-readable tool
  results, including literal and extra-field errors.
- Provider stream-level SDK exceptions are classified into typed cubepi errors
  instead of leaking raw vendor exception types.
- `Tracer.oneshot()` now closes chat spans on failure/cancellation, awaits
  stream completion and flushes, forwards abort signals, and surfaces silent
  producer failures.
- **Anthropic empty-assistant recovery** — a persisted
  `AssistantMessage(content=[], stop_reason="error")` (e.g. from a transient
  provider failure that the checkpointer recorded) no longer poisons the
  conversation:
  - Trailing empty assistant turns are dropped before send (the next request
    regenerates from the preceding user prompt instead of treating the
    placeholder as an assistant prefill).
  - Mid-history empty assistant content is replaced with an `[empty response]`
    text block so the wire payload satisfies the Anthropic API's
    `messages.N: all messages must have non-empty content` rule.
  - The trim happens in `stream()` before the cache policy resolves
    breakpoints, so the user's last-message `cache_control` marker survives
    on retry paths.
- **Tracing root attribution under middleware-driven providers** — when a
  middleware provider (e.g. `CompactionMiddleware.summary_provider`) issues
  the first chat span of a run before the agent's main call, the root
  `invoke_agent` span's `gen_ai.provider.name` / `cubepi.agent.system_prompt_sha256`
  / `cubepi.agent.tools` are no longer overwritten by the middleware's values.
  The recorder now gates root attribution by model identity (not listener
  identity), and falls back to first-call-wins when a middleware declares the
  same `(provider, model)` pair as the agent's main.
- The compaction summarizer's wrapper span now installs `turn_span` as the
  OTel current span, so `cubepi.compaction.summarize` lands as a child of
  `cubepi.turn` instead of becoming an orphan root.

### Removed

- Removed unused `PyYAML` from the default dependency set; the core dependency
  set is back to `anthropic`, `openai`, and `pydantic`.

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

[Unreleased]: https://github.com/cubeplexai/cubepi/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/cubeplexai/cubepi/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/cubeplexai/cubepi/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/cubeplexai/cubepi/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/cubeplexai/cubepi/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/cubeplexai/cubepi/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cubeplexai/cubepi/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/cubeplexai/cubepi/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cubeplexai/cubepi/releases/tag/v0.1.0
