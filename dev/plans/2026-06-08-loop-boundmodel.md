# Loop BoundModel Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push `BoundModel` through `cubepi/agent/loop.py` and its consumers (`Agent`, `tracer.oneshot`'s `_OneShotSession`, the recorder cleanup, the loop tests) in one atomic refactor, so the only places that still see a bare `Model` are the `Provider` protocol / impls and the serializable `AgentState.model` field.

**Architecture:** Collapse `provider: Provider, model: Model` → `model: BoundModel` across six loop function signatures (`run_agent_loop`, `run_agent_loop_continue`, `run_agent_loop_resume`, `_run_loop`, `_run_loop_inner`, `_stream_assistant_response`). The single in-body provider call (`provider.stream(...)` at `loop.py:723`) becomes `model.stream(...)` via the `BoundModel.stream` convenience added in #154. All consumers (`Agent.__init__` + three loop call sites, `_OneShotSession` and `tracer.oneshot`, the `recorder.py for bound in extra` rename, `tests/agent/test_loop.py` direct call sites) flip in the same commit. Tests stay green at every commit boundary.

**Tech Stack:** Python 3.11+, dataclasses, `BoundModel` (added in #154), pytest with `asyncio_mode=auto`, `FauxProvider` for deterministic tests, ruff, mypy.

---

## File Structure

- **Modify** `cubepi/agent/loop.py` — six function signatures lose `provider: Provider`, gain `model: BoundModel`; one body call (line 723) flips to `model.stream(...)`; `Provider` / `Model` imports drop.
- **Modify** `cubepi/agent/agent.py` — remove `self._provider: Provider = model.provider` (line 165); the three loop call sites (664, 688, 959) replace `provider=self._provider, model=self._state.model` with `model=self._model`; `Provider` import drops if unused.
- **Modify** `cubepi/tracing/tracer.py` — `_OneShotSession` dataclass (~line 36) holds `BoundModel` instead of split `provider` + `model`; `_OneShotSession.generate` calls `bound.generate(...)`; the `oneshot()` body (`provider = model.provider; model_spec = model.spec`; `_OneShotSession(provider=..., model=model_spec, ...)`) simplifies to pass the `BoundModel` through.
- **Modify** `cubepi/tracing/recorder.py:292` — rename `for bound in extra` → `for model in extra`, plus the local `spec = bound.spec` / `provider = bound.provider` → `spec = model.spec` / `provider = model.provider`. (Cosmetic, per `feedback_boundmodel_naming` convention.)
- **Modify** `tests/agent/test_loop.py` — delete the `make_model()` helper; every `provider = FauxProvider(); ... provider=provider, model=make_model()` block becomes `provider = FauxProvider(provider_id="faux"); ... model=provider.model("faux-1")`. Drop unused `Model` import if applicable.
- **Modify** `CHANGELOG.md` `[Unreleased]` — Breaking + Migration entries for the two publicly exported functions (`cubepi.run_agent_loop`, `cubepi.run_agent_loop_continue`).
- **Modify** `website/docs/api/*.mdx` — regenerate via `pnpm apiref` since the `cubepi` top-level surface changed.

---

## Task 1: Atomic BoundModel migration (loop + agent + tracer + recorder + tests)

**Why atomic:** `loop.py` is the producer of the new signature; `Agent`, `_OneShotSession`, the loop tests, and (cosmetically) the recorder all consume. Splitting these commits would leave any intermediate commit with broken tests. Same rationale as Task 3 in the #154 plan.

- [ ] **Step 0: Confirm scope by greppping for any external callers we'd miss**

```bash
cd /home/chris/cubepi/.worktrees/2026-06-08-loop-boundmodel
grep -rn "run_agent_loop\|_run_loop\|_stream_assistant_response\|_OneShotSession" cubepi/ tests/ examples/ --include="*.py"
```

Expected hits: `cubepi/__init__.py` (re-exports), `cubepi/agent/__init__.py` (re-exports), `cubepi/agent/agent.py` (callers), `cubepi/agent/loop.py` (definitions), `cubepi/tracing/tracer.py` (`_OneShotSession`), `tests/agent/test_loop.py` (direct callers). If anything else shows up (e.g. an example script using `run_agent_loop` directly), fold it into the migration.

- [ ] **Step 1: Baseline test run**

```bash
uv run pytest tests/ -x 2>&1 | tail -3
```

Expected: PASS. Record the totals — must be the same after Task 1.

- [ ] **Step 2: Migrate `cubepi/agent/loop.py` signatures + body**

Edit each of the six functions: replace `provider: Provider,` + `model: Model,` (two lines) with `model: BoundModel,` (one line). Then in `_stream_assistant_response` (around line 723), replace:

```python
stream = await provider.stream(
    model=model,
    messages=messages,
    system_prompt=system_prompt,
    tools=tools,
    options=options,
)
```

with:

```python
stream = await model.stream(
    messages=messages,
    system_prompt=system_prompt,
    tools=tools,
    options=options,
)
```

Update imports at the top of the file: drop `Model` and `Provider` from the `cubepi.providers.base` import line, add `BoundModel`. Verify with `grep -n "Provider\b\|Model\b" cubepi/agent/loop.py` — after editing, neither bare name should appear in the file body (only as `BoundModel`).

Internal call sites between the six functions: each `_run_loop`/`_run_loop_inner` call previously passed `provider=provider, model=model` — switch to `model=model` (the BoundModel parameter).

- [ ] **Step 3: Migrate `cubepi/agent/agent.py`**

Three changes in `agent.py`:

**(a)** Drop `self._provider: Provider = model.provider` (line 165). The `self._model` field already holds the same provider via `self._model.provider`.

**(b)** Update the three loop call sites (around lines 664, 688, 959). Currently each looks like:

```python
lambda signal: run_agent_loop(
    ...
    provider=self._provider,
    model=self._state.model,
    ...
)
```

Replace with:

```python
lambda signal: run_agent_loop(
    ...
    model=self._model,
    ...
)
```

(`self._state.model` was the spec; `self._model` is the `BoundModel` — that's the source of truth post-migration.)

**(c)** Drop `Provider` from imports if no longer used (verify with `grep -n "Provider\b" cubepi/agent/agent.py` — only `BaseProvider`-style names should remain, if any).

`self._state.model = model.spec` at construction (around line 169) stays as-is. The `AgentState.model: Model` field stays as-is. (Design point 1 in the spec.)

- [ ] **Step 4: Migrate `_OneShotSession` + `tracer.oneshot()`**

In `cubepi/tracing/tracer.py`, `_OneShotSession` (around line 36) is a dataclass holding `provider` + `model` + `run`. Change to hold a single `model: BoundModel` + `run`. Find the call sites:

- `_OneShotSession.generate(...)` body: replace internal `provider.generate(...)` (currently passes `model=self.model`) with `self.model.generate(...)`.
- `Tracer.oneshot()` (around line 419): the lines `provider = model.provider; model_spec = model.spec` and the `_OneShotSession(provider=provider, model=model_spec, run=run)` instantiation. Replace with `_OneShotSession(model=model, run=run)`. Drop the two local variables.

The public `Tracer.oneshot(*, model: BoundModel, ...)` signature is unchanged — only internals migrate.

- [ ] **Step 5: Rename `for bound in extra` → `for model in extra` in recorder**

In `cubepi/tracing/recorder.py:292`, the current block:

```python
for bound in extra:
    spec = bound.spec
    key = (spec.provider_id, spec.id)
    if key != agent_key:
        self._extra_call_models.add(key)
    provider = bound.provider
    if id(provider) in seen:
        continue
    seen.add(id(provider))
    _subscribe(provider)
```

becomes:

```python
for model in extra:
    spec = model.spec
    key = (spec.provider_id, spec.id)
    if key != agent_key:
        self._extra_call_models.add(key)
    provider = model.provider
    if id(provider) in seen:
        continue
    seen.add(id(provider))
    _subscribe(provider)
```

Pure cosmetic — no behavior change. Matches the `feedback_boundmodel_naming` convention. No imports change.

- [ ] **Step 6: Migrate `tests/agent/test_loop.py` to the new signature**

Two edits:

**(a)** Delete the `make_model()` helper (lines 29–30).

**(b)** Every test method has a block like:

```python
provider = FauxProvider()
# ...
result = await run_agent_loop(
    ...
    provider=provider,
    model=make_model(),
    ...
)
```

Change to:

```python
provider = FauxProvider(provider_id="faux")
# ...
result = await run_agent_loop(
    ...
    model=provider.model("faux-1"),
    ...
)
```

Drop `Model` from the top-level `cubepi.providers.base` import if it's no longer referenced elsewhere in the file (verify with `grep -n "\bModel\b" tests/agent/test_loop.py`).

- [ ] **Step 7: Run the affected suites, then the full sweep + lint + types**

```bash
uv run pytest tests/agent/ tests/tracing/ -v 2>&1 | tail -10
```

Expected: PASS. Specifically `tests/agent/test_loop.py` should report the same number of tests passing as before.

```bash
uv run pytest tests/ -x 2>&1 | tail -3
uv run ruff check cubepi/ tests/
uv run ruff format --check cubepi/ tests/
uv run mypy cubepi
```

Expected: all green. If ruff format flags reformatting, run `uv run ruff format cubepi/ tests/` then re-check.

- [ ] **Step 8: Commit**

```bash
git add cubepi/agent/loop.py cubepi/agent/agent.py cubepi/tracing/tracer.py cubepi/tracing/recorder.py tests/agent/test_loop.py
git commit -m "refactor(agent): thread BoundModel through loop, agent, oneshot, recorder"
```

---

## Task 2: CHANGELOG

- [ ] **Step 1: Add Breaking + Migration entries under `[Unreleased]`**

In `CHANGELOG.md`, extend the existing `[Unreleased]` block (which already contains the BoundModel.generate/stream + extra_llm_calls entries from #154) with one new `Breaking` bullet:

```markdown
- **`cubepi.run_agent_loop` and `cubepi.run_agent_loop_continue` take
  `model: BoundModel`** instead of separate `provider: Provider, model: Model`
  kwargs. The stateless-loop entry points used by callers who drive the loop
  outside of `Agent` must update. The `Agent` API is unchanged — it has always
  taken `model: BoundModel`.
```

And one new `Migration` example:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): record run_agent_loop BoundModel migration"
```

---

## Task 3: Regenerate API reference docs

- [ ] **Step 1: Run apiref**

```bash
cd website && pnpm apiref 2>&1 | tail -10 && cd -
```

Expected: a handful of `website/docs/api/*.mdx` files update — at minimum `cubepi.mdx` (signature of `run_agent_loop` / `run_agent_loop_continue`) and `cubepi-middleware.mdx` (which had a stale `extra_llm_calls() -> Iterable[tuple[Provider, Model]]` line from #154 that this regen also fixes).

- [ ] **Step 2: Verify the stale references are gone**

```bash
grep -rn "extra_llm_calls.*tuple\|tuple\[Provider, Model\]\|provider: Provider,\s*model: Model" website/docs/api/
```

Expected: no hits.

- [ ] **Step 3: Diff sanity check + commit**

```bash
git status website/docs/api/
git diff --stat website/docs/api/
```

Confirm the diff only touches the expected mdx files (signatures involving `run_agent_loop*`, `extra_llm_calls`, and possibly `oneshot`).

```bash
git add website/docs/api/
git commit -m "docs(apiref): regenerate after BoundModel loop migration"
```

---

## Task 4: Open PR + PR codex review loop

- [ ] **Step 1: Push branch**

```bash
git push -u origin 2026-06-08-loop-boundmodel
```

- [ ] **Step 2: Open PR**

```bash
gh pr create --title "refactor(agent): thread BoundModel through loop, oneshot, recorder" --body "$(cat <<'EOF'
## Summary

- Migrate `cubepi/agent/loop.py` (6 function signatures) from `provider: Provider, model: Model` → `model: BoundModel`. The single in-body `provider.stream(...)` flips to `model.stream(...)` via the convenience added in #154.
- `Agent.__init__` drops the redundant `self._provider` field; three loop call sites pass `model=self._model` instead of the split pair. `AgentState.model: Model` field unchanged — it stays serializable for the checkpointer.
- `_OneShotSession` (used by `tracer.oneshot()`) holds a `BoundModel` internally; `oneshot()` no longer extracts `provider` / `model_spec` separately.
- Recorder cleanup: `for bound in extra` → `for model in extra` per the `feedback_boundmodel_naming` convention.
- `tests/agent/test_loop.py` migrated; `make_model()` helper removed in favor of inline `provider.model("faux-1")`.
- API reference regenerated.

## Workflow

- Spec at `dev/specs/2026-06-08-loop-boundmodel.md` (user-confirmed design: AgentState.model stays Model, resume trusts caller, OneShot migrates lock-step).
- Plan at `dev/plans/2026-06-08-loop-boundmodel.md`.
- Per user direction, skipped local codex review on spec/plan/code — going straight to PR codex review.

## Breaking

`cubepi.run_agent_loop` and `cubepi.run_agent_loop_continue` change signature. See `CHANGELOG.md` Breaking + Migration sections under `[Unreleased]`.

## Test plan

- [x] `uv run pytest tests/`
- [x] `uv run ruff check cubepi/ tests/`
- [x] `uv run ruff format --check cubepi/ tests/`
- [x] `uv run mypy cubepi`
EOF
)"
```

- [ ] **Step 3: Enter PR codex review loop**

Poll the PR every ~2 minutes; for every codex finding, evaluate against project policy (`feedback_breaking_no_shim`, `feedback_boundmodel_naming`), fix or push back with rationale, push commit(s), reply `@codex review again`. Repeat until clean. Merge after CI + codex both green.

---

## Self-Review

1. **Spec coverage** — every spec scope item maps to a Task 1 step:
   - Loop six function migration → Step 2.
   - Agent: drop `_provider`, three call sites → Step 3.
   - `_OneShotSession` + `tracer.oneshot()` → Step 4.
   - Recorder rename → Step 5.
   - `tests/agent/test_loop.py` migration → Step 6.
   - CHANGELOG → Task 2.
   - apiref → Task 3.
2. **Atomicity** — Task 1 is one commit covering loop + agent + tracer + recorder + tests; producer/consumer flip together so `pytest -x` stays green at every commit boundary.
3. **Placeholders** — none. Every step has the actual code change or a concrete grep / pytest / commit command.
4. **Naming convention** — every variable rename and parameter name uses `model: BoundModel` per `feedback_boundmodel_naming`. No leftover `bound` / `bound_model`.
5. **Out of scope** — Provider protocol & impls, `AgentState` shape, `Model` rename, mid-run model swap. All match the spec.

## Follow-ups (out of scope)

None for this work. The original Follow-up #2 from the #154 plan is what this plan executes; no further follow-ups required.
