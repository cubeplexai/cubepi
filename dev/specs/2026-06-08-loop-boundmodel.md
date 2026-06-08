# Spec: Thread `BoundModel` through the agent loop

**Status:** Draft → user-confirmed design points 2026-06-08.

## Goal

Make `cubepi/agent/loop.py` (and its callers in `Agent` and
`cubepi/tracing/tracer.py`) take a single `BoundModel` instead of a
`provider: Provider, model: Model` pair. After this change, the only
places that still see a bare `Model` are:

- `Provider` protocol / built-in provider implementations (correctly so —
  providers are below the binding layer).
- `AgentState.model` field (serialized; provider can't be pickled).
- Tracing recorder reading `model.id` / `model.provider_id` off bound
  models or state.

User-facing code (Agent construction, middleware, examples, custom loop
drivers) can stop carrying the unwrapped pair.

## Why now

The 2026-06-08 BoundModel-methods PR (#154, merged as `e917128`) added
`BoundModel.generate()` / `BoundModel.stream()` and migrated
`Middleware.extra_llm_calls()` to return `Iterable[BoundModel]`. That left
`agent/loop.py` as the last load-bearing surface still threading the
unwrapped pair. The follow-up note in
`dev/plans/2026-06-08-boundmodel-convenience-methods.md` flagged this
explicitly.

## Prior art notes

- **langgraph** and **claude-code** use string ids + a global provider
  registry ("anthropic:claude-3-5-sonnet"). cubepi 0.8 deliberately chose
  the opposite — explicit `provider.model("id")` binding, no registry — so
  spec stays serializable and there's no global state. This spec preserves
  that choice; we only push the existing `BoundModel` deeper into the
  internals.
- **pi-agent-core** has no analog of `BoundModel`; its loop takes a
  pre-bound callable. cubepi keeps explicit types instead of opaque
  closures, but the runtime effect is similar.

## Design points (user-confirmed 2026-06-08)

1. **`AgentState.model: Model` stays as-is.** State is part of the
   checkpointer schema; changing its shape forces a schema migration.
   Loop call sites in `Agent` switch from `model=self._state.model` to
   passing `self._model` (the `BoundModel`). The state's `model` field
   remains as a historical / debug record of which spec was used.

2. **Resume trusts the caller.** When the Agent restores from
   checkpointer, `self._state.model` carries the original spec but
   `self._model: BoundModel` (set at `__init__`) is the source of truth
   for the next loop iteration. No mismatch check — the Agent's current
   `BoundModel` runs; the state's spec is treated as an archived value.
   If a caller reconstructs an Agent with a different model after
   restart, that's an intentional model swap and the loop honors it.

3. **`_OneShotSession` migrates in lock-step.** `tracer.oneshot()` already
   takes a `BoundModel` at its public boundary; finish the migration by
   making `_OneShotSession` itself hold a `BoundModel` (drops the
   `provider=..., model=model_spec` constructor split inside
   `tracer.py:573`).

## Scope

### In scope

- `cubepi/agent/loop.py`: collapse `provider: Provider, model: Model`
  into `model: BoundModel` across the six function surfaces:
  - `run_agent_loop` (public)
  - `run_agent_loop_continue` (public)
  - `run_agent_loop_resume` (internal — not in top-level `__all__`)
  - `_run_loop` (private)
  - `_run_loop_inner` (private)
  - `_stream_assistant_response` (private)
  - Bodies: replace `await provider.stream(model=spec, ...)` etc. with
    `await model.stream(...)` / `await model.generate(...)`.
- `cubepi/agent/agent.py`:
  - Remove `self._provider: Provider = model.provider` (line 165) — the
    field becomes redundant once loop takes `BoundModel`.
  - Replace three loop call sites (664, 688, 959): drop
    `provider=self._provider, model=self._state.model`, pass
    `model=self._model` instead.
  - Keep `self._state.model = model.spec` at construction (line 169) and
    leave `AgentState.model` typed `Model` — see Design point 1.
- `cubepi/tracing/tracer.py:573`:
  - Migrate `_OneShotSession` (defined at `tracer.py:36`) to hold
    `BoundModel`. `_OneShotSession.generate` switches from internal
    `provider.generate(model=spec, ...)` to `bound.generate(...)`.
  - `oneshot()` no longer needs `provider = model.provider; model_spec =
    model.spec` (lines 477, 573).
- `cubepi/tracing/recorder.py:292`: rename `for bound in extra` to
  `for model in extra` per the BoundModel naming convention recorded in
  the `feedback_boundmodel_naming` memory. Update the local variables
  (`spec = model.spec`, `provider = model.provider`) to match.
- `tests/agent/test_loop.py`: migrate direct `run_agent_loop(...)` /
  `run_agent_loop_continue(...)` call sites to the new signature
  (`provider=..., model=...` → `model=bound`).
- `CHANGELOG.md` `[Unreleased]`: Breaking + Migration entries for the two
  publicly exported functions (`run_agent_loop`,
  `run_agent_loop_continue`).
- Docs: regenerate API reference (`pnpm apiref`) since the `cubepi`
  top-level module surface changed.

### Out of scope

- `Provider` protocol and built-in provider implementations
  (`anthropic.py`, `openai.py`, `openai_responses.py`, `faux.py`) — they
  correctly take `model: Model` (the spec) because providers operate
  below the binding layer.
- `cubepi/providers/models.py` helpers — operate on `Model` (the spec).
- `AgentState` shape / checkpointer schema — Design point 1.
- Mid-run model swap APIs (e.g., `agent.set_model(...)`) — out of
  scope; `self._model` remains set-once at `__init__`.
- Provider registry / string-id resolution (langgraph style) — explicitly
  rejected at 0.8 and reaffirmed here.

## Breaking surface

Two publicly exported functions change signature:

- `cubepi.run_agent_loop`
- `cubepi.run_agent_loop_continue`

Both lose the `provider: Provider` keyword and replace `model: Model`
with `model: BoundModel`. Callers using these stateless-loop entry
points (uncommon — most users build `Agent` instances) must update.
This is the same shape of breakage as the
`Middleware.extra_llm_calls() -> Iterable[BoundModel]` change in #154:
documented in `CHANGELOG.md` under Breaking + Migration, no
backwards-compat shim per project policy
(`feedback_breaking_no_shim`).

Migration for callers:

```python
# Before
await run_agent_loop(
    prompts=[...],
    context=ctx,
    provider=provider,
    model=model_spec,
    convert_to_llm=...,
    emit=...,
    ...
)

# After
await run_agent_loop(
    prompts=[...],
    context=ctx,
    model=provider.model("id", ...),
    convert_to_llm=...,
    emit=...,
    ...
)
```

(`run_agent_loop_resume` also changes signature but is not in
`cubepi/__init__.py`'s top-level `__all__`, so it's not a public API
break — internal callers in `agent.py` migrate together.)

## Naming convention

Per `feedback_boundmodel_naming`: inside the framework, name `BoundModel`
parameters and locals `model` (not `bound`). Call sites read
`await model.generate(...)` / `await model.stream(...)`, matching the
public `Agent(model=...)` / `provider.model(...)` ergonomics. This
includes the `recorder.py` `for bound in extra` cleanup.

## Non-goals / explicitly-not-changing

- `Model` (the spec class) name. Renaming to `ModelSpec` for clarity
  was discussed and deferred — the churn cost outweighs the naming
  benefit, and `Model` / `BoundModel` is a recognizable "data class vs
  runtime handle" pair (cf. `URL` vs `URLConnection`).
- Public `Model` constructor — direct use of `Model(...)` remains
  valid (custom providers, test fixtures, advanced flows).
