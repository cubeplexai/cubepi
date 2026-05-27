# Subagent Trace Nesting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an inner (sub)agent's spans nest under the `execute_tool subagent` span that spawned it — its `cubepi.turn` / `chat` / `execute_tool` spans become children of the tool span instead of leaking flat onto the parent agent's turn.

**Architecture:** Two coordinated changes. (1) cubepi `Recorder`: gate provider-listener span creation on a per-task "active run" contextvar so a *shared provider* no longer routes an inner agent's LLM calls into the parent recorder's run; and parent the inner run's root span under the active `execute_tool` span when one is in scope for the task. (2) cubebox `SubAgentMiddleware`: attach the parent's `Tracer` to the inner agent so the inner run is actually recorded.

**Tech Stack:** Python 3.13, OpenTelemetry SDK, `contextvars`, cubepi event-driven `Recorder`, pytest.

---

## Background — Root Cause (verified against trace `9c6a0909`)

The `Recorder` (`cubepi/tracing/recorder.py`) records spans two ways:

- **Structural spans** (`invoke_agent` / `cubepi.turn` / `execute_tool`) — driven by the *attached* agent's events. Parents wired with explicit `context=set_span_in_context(run.agent_span | run.turn_span)`.
- **`chat` spans** — driven by **provider listeners** (`_on_provider_request/chunk/response`) registered on `agent._provider`, parented to `self._run.turn_span`.

cubebox's `SubAgentMiddleware._execute` (`backend/cubebox/middleware/subagents.py`) spawns an inner `cubepi.Agent` that (a) **reuses the parent's provider instance** and (b) is **never attached** to a recorder (only `inner.subscribe(_listener)` for SSE). Consequences:

1. Inner agent's turn/tool/agent events never reach a recorder → **no nested turn spans, no nested `execute_tool` spans**.
2. Inner LLM calls go through the shared provider → the *parent* recorder's `_on_provider_request` fires and parents the `chat` span under `self._run.turn_span` (the parent's subagent-calling turn) → **inner chats flatten under the outer turn**, siblings of `execute_tool subagent`.

The tracing design spec (`dev/specs/2026-05-18-cubepi-tracing-design.md` §6.1 / §8 "nested agents") already describes the intended behavior: an inner agent with its own `Tracer` whose root `start_span` nests under the active tool span via context propagation. The infra for per-task parent routing already exists for MCP CLIENT spans (`cubepi/mcp/_tracing.py`: `register_tool_span` + `_get_tool_span_entry` + `_current_handle` contextvar). This plan reuses that pattern for agent runs.

### Verified feasibility facts

- `Tracer.attach()` constructs a **fresh `Recorder`** per call (`cubepi/tracing/tracer.py:148`), so one `Tracer` can attach to parent **and** inner agent, each with isolated `_RunState`, both exporting to the same JSONL/OTLP destinations (same `trace_id` file).
- `Recorder._on_tool_exec_start` calls `register_tool_span(...)` **before** the tool body runs; for `parallel` tool mode the body runs in a child task created after the contextvar is set, so it inherits `_current_handle`. For `sequential` mode the body runs inline in the same task. Either way `_get_tool_span_entry()` returns the active `execute_tool` span inside the subagent tool body.

---

## File Structure

- `cubepi/tracing/recorder.py` — add a per-task "active run" contextvar; gate the three provider listeners on it; parent the `invoke_agent` root under the active tool span when present.
- `cubepi/mcp/_tracing.py` — no new code; reuse `_get_tool_span_entry()` (already public-internal) to fetch `(span, provider)` for the current task.
- `backend/cubebox/middleware/subagents.py` (cubebox repo) — accept a `tracer` and `tracer.attach(inner)` around `inner.prompt()`.
- `backend/cubebox/streams/run_manager.py` (cubebox repo) — pass the active `Tracer` into `SubAgentMiddleware`.
- Tests: `tests/tracing/test_subagent_nesting.py` (cubepi), `backend/tests/.../test_subagent_trace.py` (cubebox).

> **Cross-repo note:** cubepi reaches cubebox runtime only via a pinned `rev` in `backend/uv.lock`. After cubepi changes land (Tasks 1–3) and are pushed, bump the pin in cubebox before the cubebox change (Task 4) can be verified end-to-end. See Task 5.

---

## Task 1: Per-task "active run" gate on provider listeners (cubepi)

Stops a shared provider from routing an inner agent's LLM calls into the parent recorder's run. Each recorder marks *its* run active in the current task on `AgentStart`; provider listeners no-op when the task's active run isn't theirs.

**Files:**
- Modify: `cubepi/tracing/recorder.py` (module-level contextvar; `_on_agent_start`, `_on_agent_end`, `_close_open_spans`/abort path; `_on_provider_request`, `_on_provider_chunk`, `_on_provider_response`)
- Test: `tests/tracing/test_subagent_nesting.py`

- [ ] **Step 1: Write the failing test**

> **Why this test must drive the PARENT run (codex BLOCKING).** A naive
> repro that attaches both agents but only calls `inner.prompt()` does NOT
> reproduce the bug: the parent recorder never receives `AgentStartEvent`, so
> its `_run is None` and `_on_provider_request` early-returns (`recorder.py`
> "run is None" guard) — it never mints a duplicate. The leak only happens
> when the parent run is **live and mid-turn** (its `turn_span` is open while
> it awaits the subagent tool). So the test MUST run the parent through a turn
> that calls a tool whose body attaches+runs the inner agent — exactly the
> production shape.

```python
# tests/tracing/test_subagent_nesting.py
import pytest
from cubepi import AgentTool, AgentToolResult, TextContent
from pydantic import BaseModel
from tests.tracing.helpers import build_tracer_with_memory_exporter, FauxProvider, make_agent


class _Empty(BaseModel):
    pass


@pytest.mark.asyncio
async def test_shared_provider_does_not_double_mint_inner_chat(tmp_path):
    """Parent (mid-turn) spawns an inner agent that shares the parent's provider
    and is attached to the SAME tracer. Without the active-run gate, BOTH
    recorders' provider listeners fire for the inner LLM call -> a duplicate
    chat span under the parent turn. With the gate, only the inner recorder
    mints it."""
    tracer, exporter = build_tracer_with_memory_exporter()
    # ONE shared provider. Scripted replies are consumed in cross-agent CALL
    # order: parent turn-1 emits the tool call; the inner agent (run inside the
    # tool body) consumes the next reply; parent turn-2 finalizes last.
    provider = FauxProvider(scripted_replies=[
        {"tool_calls": [{"id": "tc1", "name": "spawn", "arguments": {}}]},  # parent T1
        "inner-final",                                                       # inner
        "parent-final",                                                      # parent T2
    ])

    async def _spawn(tool_call_id, args, *, signal=None, on_update=None):
        inner = make_agent(provider=provider, tools=[])
        detach = tracer.attach(inner)
        try:
            await inner.prompt("inner task")
        finally:
            res = detach()
            if res is not None:
                await res
        return AgentToolResult(content=[TextContent(text="ok")])

    spawn = AgentTool(name="spawn", description="spawn inner", parameters=_Empty, execute=_spawn)
    parent = make_agent(provider=provider, tools=[spawn])
    detach_parent = tracer.attach(parent)
    try:
        await parent.prompt("go")
    finally:
        res = detach_parent()
        if res is not None:
            await res
    await tracer.shutdown()

    spans = exporter.get_finished_spans()
    chat_spans = [s for s in spans if s.attributes.get("gen_ai.operation.name") == "chat"]
    # Expected: 2 parent chats (T1 + T2) + 1 inner chat = 3.
    # The bug adds a 4th: the parent recorder ALSO mints the inner call under
    # the parent's open turn_span.
    assert len(chat_spans) == 3, (
        f"expected 3 chat spans, got {len(chat_spans)} "
        "(parent recorder double-handled the inner LLM call)"
    )
```

> If `tests/tracing/helpers.py` lacks `build_tracer_with_memory_exporter` / `FauxProvider` / `make_agent`, reuse the existing fixtures used by the current recorder tests (grep `tests/tracing/` for `InMemorySpanExporter` and the faux provider). Adapt names + the `scripted_replies` shape (some fakes take `AssistantMessage`s, others a callable) to the existing harness rather than inventing new ones.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/chris/cubepi && uv run pytest tests/tracing/test_subagent_nesting.py::test_shared_provider_does_not_double_mint_inner_chat -v`
Expected: FAIL — 4 chat spans (the parent recorder mints an extra one for the inner LLM call, parented under the parent turn).

- [ ] **Step 3: Add the active-run contextvar and set/reset it**

In `cubepi/tracing/recorder.py`, near the top-level imports (after the existing module constants), add:

```python
import contextvars

# Per-task pointer to the _RunState that "owns" the current asyncio task.
# Provider listeners are registered per provider INSTANCE; when one provider
# is shared by a parent agent and an inner (sub)agent, every attached
# recorder's listener fires for every LLM call on that provider. This
# contextvar lets each listener act only for the run that is active in the
# calling task. Copies into child tasks at creation (asyncio semantics), and
# the inner agent's AgentStart overwrites it within the child task.
_active_run: contextvars.ContextVar[object | None] = contextvars.ContextVar(
    "cubepi.tracing.active_run", default=None
)
```

In `_on_agent_start`, immediately after `self._run = _RunState(run_id=run_id, agent_span=span)`:

```python
        # Mark this run active for the current task so the shared-provider
        # listeners route LLM spans to the right recorder.
        self._active_run_token = _active_run.set(self._run)
```

Add `self._active_run_token = None` to `Recorder.__init__` (alongside the other per-run fields). Reset it best-effort in a small helper and call that helper from **BOTH** `_on_agent_end` **AND** `_close_open_spans` (the detach cleanup path):

```python
    def _reset_active_run(self) -> None:
        token = getattr(self, "_active_run_token", None)
        if token is not None:
            try:
                _active_run.reset(token)
            except (ValueError, LookupError):
                pass
            self._active_run_token = None
```

> **Why reset in `_close_open_spans`, not only `_on_agent_end` (codex SHOULD-FIX).**
> `Agent._run_with_lifecycle` re-raises `asyncio.CancelledError` WITHOUT
> emitting `AgentEndEvent`, so a cancelled inner run never reaches
> `_on_agent_end`. `Tracer.attach`'s `detach()` always runs
> `_close_open_spans` synchronously (see `recorder.py` `_sync_detach`), and
> Task 4 calls `detach()` in a `finally`. In **sequential** tool mode the inner
> agent's `_on_agent_start` and the tool body's `detach()` run in the SAME task,
> so the reset token is valid and the reset succeeds — preventing a stale
> `_active_run` from gating the parent's post-tool turn in that same task. Put
> the reset in `_close_open_spans` so cancellation can't leak a stale gate.

> **Why the per-task gate is correct under parallel tool mode (codex SHOULD-FIX — document this).**
> In `parallel` mode the parent emits `ToolExecutionStartEvent` in the parent
> task BEFORE `asyncio.create_task(_run())` spawns the per-tool child task
> (`agent/tools.py` `_execute_parallel`). The child task copies the parent's
> contextvar snapshot at creation; then `inner.prompt()` fires `AgentStartEvent`
> inside that same child task, so `_active_run.set(inner_run)` shadows ONLY the
> child task's copy — the parent task's `_active_run` is untouched. The inner's
> `AgentEndEvent` (or `detach`) runs on that same child-task path, so the reset
> token is same-task. `_active_run.reset` raising on cross-task is handled
> best-effort (mirrors the existing `unregister_tool_span` pattern); the
> source of truth is `self._run`, the contextvar is only the per-task pointer.

- [ ] **Step 4: Gate the three provider listeners**

At the top of `_on_provider_request`, `_on_provider_chunk`, and `_on_provider_response`, after the existing `run = self._run` / `if run is None ...` guard, add:

```python
        if _active_run.get() is not run:
            # A different run owns this task (e.g. an inner subagent sharing
            # our provider). That run's own recorder will handle this event.
            return
```

> Place the check AFTER `run is None` so a recorder with no live run still no-ops. For `_on_provider_chunk` / `_on_provider_response`, also keep the existing `run.chat_span is None` guards.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/chris/cubepi && uv run pytest tests/tracing/test_subagent_nesting.py::test_shared_provider_does_not_double_mint_inner_chat -v`
Expected: PASS — exactly 3 chat spans.

- [ ] **Step 6: Cancellation test — a cancelled inner run must not leave a stale gate**

```python
@pytest.mark.asyncio
async def test_cancelled_inner_does_not_gate_parents_next_turn(tmp_path):
    """Inner run cancelled mid-LLM-call (no AgentEndEvent) in SEQUENTIAL tool
    mode runs in the same task as the parent. If detach()/_close_open_spans
    fails to reset _active_run, the parent's post-tool turn would be wrongly
    gated and its chat span dropped. Assert the parent's T2 chat survives."""
    tracer, exporter = build_tracer_with_memory_exporter()

    # Inner provider call raises CancelledError; parent then finalizes.
    provider = FauxProvider(scripted_replies=[
        {"tool_calls": [{"id": "tc1", "name": "spawn", "arguments": {}}]},  # parent T1
        CancelledError(),                                                    # inner -> cancel
        "parent-final",                                                      # parent T2
    ])

    async def _spawn(tool_call_id, args, *, signal=None, on_update=None):
        inner = make_agent(provider=provider, tools=[])
        detach = tracer.attach(inner)
        try:
            await inner.prompt("inner task")
        except CancelledError:
            pass  # subagent tool swallows + returns an error result in prod
        finally:
            res = detach()
            if res is not None:
                await res
        return AgentToolResult(content=[TextContent(text="[cancelled]")], is_error=True)

    # Force sequential so the inner run shares the parent/loop task.
    spawn = AgentTool(name="spawn", description="spawn inner", parameters=_Empty,
                      execute=_spawn, execution_mode="sequential")
    parent = make_agent(provider=provider, tools=[spawn])
    detach_parent = tracer.attach(parent)
    try:
        await parent.prompt("go")
    finally:
        res = detach_parent()
        if res is not None:
            await res
    await tracer.shutdown()

    spans = exporter.get_finished_spans()
    # Identify the PARENT run UNAMBIGUOUSLY via the execute_tool span's run_id.
    # `_on_tool_exec_start` stamps CUBEPI_RUN_ID = the parent recorder's
    # run.run_id on the tool span, so this is the parent run regardless of
    # task ordering. (codex SHOULD-FIX: do NOT use `next(parent is None)` — at
    # Task 1 time, before Task 2 lands, the cancelled inner run ALSO opens a
    # parentless invoke_agent span, so parentless-root selection is ambiguous.)
    tool_span = next(s for s in spans if s.name == "execute_tool spawn")
    parent_run_id = tool_span.attributes.get("cubepi.run_id")
    parent_chats = [s for s in spans
                    if s.attributes.get("gen_ai.operation.name") == "chat"
                    and s.attributes.get("cubepi.run_id") == parent_run_id]
    # Parent T1 AND T2 must both be present under the PARENT run. If the stale
    # inner gate were left set, T2's provider call would be skipped and only 1
    # parent chat would exist.
    assert len(parent_chats) == 2, (
        f"expected 2 parent-run chat spans, got {len(parent_chats)} — parent's "
        "post-tool turn was gated by a stale _active_run from the cancelled inner run"
    )
```

> Import `from asyncio import CancelledError` at the top of the test module.
> Adapt the faux provider's "raise on this reply" mechanism to the harness
> (some fakes accept an exception instance in `scripted_replies`; others need a
> side-effect callable). The invariant under test is harness-independent: after
> a cancelled inner run, the parent's next provider call in the same task is
> NOT gated.

- [ ] **Step 7: Run cancellation test**

Run: `cd /home/chris/cubepi && uv run pytest tests/tracing/test_subagent_nesting.py::test_cancelled_inner_does_not_gate_parents_next_turn -v`
Expected: PASS (with the reset wired into `_close_open_spans` per Step 3).

- [ ] **Step 8: Regression — existing tracing tests still green**

Run: `cd /home/chris/cubepi && uv run pytest tests/tracing/ -q`
Expected: PASS (no regressions in single-agent recording).

- [ ] **Step 9: Commit**

```bash
cd /home/chris/cubepi
git add cubepi/tracing/recorder.py tests/tracing/test_subagent_nesting.py
git commit -m "fix(tracing): gate provider listeners on per-task active run

A provider shared by a parent and inner agent fired every attached
recorder's chat-span listener. Route each listener to the run that owns
the calling task via a contextvar so an inner subagent's LLM calls no
longer leak into the parent recorder's run."
```

---

## Task 2: Parent the inner run's root under the active `execute_tool` span (cubepi)

When `invoke_agent` opens inside an active tool body, nest it under that tool span (inherit `trace_id`, set `parent_span_id`) instead of starting a new root.

**Files:**
- Modify: `cubepi/tracing/recorder.py` (`_on_agent_start`, around the `start_span(SPAN_NAME_INVOKE_AGENT ...)` call ~line 436)
- Test: `tests/tracing/test_subagent_nesting.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_inner_run_nests_under_active_tool_span(tmp_path):
    """An invoke_agent span opened while an execute_tool span is active for the
    task must be parented under that tool span (same trace_id)."""
    from cubepi.mcp import _tracing as mcp_tracing
    tracer, exporter = build_tracer_with_memory_exporter()

    # Open a standalone execute_tool-like parent span and register it as the
    # active tool span for this task, mimicking _on_tool_exec_start.
    parent_span = tracer.otel_tracer.start_span("execute_tool subagent")
    token = mcp_tracing.register_tool_span("tc1", parent_span, provider=None)
    try:
        provider = FauxProvider(scripted_replies=["inner-done"])
        inner = make_agent(provider=provider, tools=[])
        detach = tracer.attach(inner)
        try:
            await inner.prompt("hi")
        finally:
            await detach()
    finally:
        mcp_tracing.unregister_tool_span(token)
        parent_span.end()
    await tracer.shutdown()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    root = next(s for s in exporter.get_finished_spans()
                if s.attributes.get("gen_ai.operation.name") == "invoke_agent")
    parent_ctx = parent_span.get_span_context()
    assert root.parent is not None
    assert root.parent.span_id == parent_ctx.span_id, "invoke_agent not nested under tool span"
    assert root.context.trace_id == parent_ctx.trace_id, "trace_id not inherited"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/chris/cubepi && uv run pytest tests/tracing/test_subagent_nesting.py::test_inner_run_nests_under_active_tool_span -v`
Expected: FAIL — `root.parent is None` (invoke_agent opened as a new root).

- [ ] **Step 3: Nest the root under the active tool span**

In `_on_agent_start`, replace the unconditional root open:

```python
        span = self._tracer.otel_tracer.start_span(
            name=SPAN_NAME_INVOKE_AGENT,
            kind=SpanKind.INTERNAL,
            attributes={ ... },
        )
```

with a context-aware open:

```python
        from opentelemetry import trace as _otel_trace
        parent_ctx = None
        try:
            # TODO(tracing): relocate this "current tool span" helper out of
            # cubepi.mcp._tracing — it now serves generic nested agents, not
            # just MCP clients.
            from cubepi.mcp import _tracing as _mcp_tracing
            entry = _mcp_tracing._get_tool_span_entry()
            if entry is not None:
                tool_span, _tool_provider = entry
                parent_ctx = _otel_trace.set_span_in_context(tool_span)
        except ImportError:  # pragma: no cover — mcp module always present
            parent_ctx = None

        span = self._tracer.otel_tracer.start_span(
            name=SPAN_NAME_INVOKE_AGENT,
            kind=SpanKind.INTERNAL,
            context=parent_ctx,  # see branch note below
            attributes={
                GEN_AI_OPERATION_NAME: OP_INVOKE_AGENT,
                CUBEPI_RUN_ID: run_id,
                GEN_AI_PROVIDER_NAME: "cubepi",
            },
        )
```

> **Two branches (codex NIT — don't call `context=None` a "new root").**
> When no tool span is active for the task, `parent_ctx` stays `None` and
> `start_span` falls back to the OTel ambient current context — i.e. exactly
> today's behavior (a root span in practice, since the recorder never installs
> its own spans as ambient current). Single-agent runs are unaffected. The new
> work is the OTHER branch: when `_get_tool_span_entry()` returns a live tool
> span, `parent_ctx = set_span_in_context(tool_span)` makes the inner
> `invoke_agent` a child of the `execute_tool` span — inheriting its `trace_id`
> and taking `parent_span_id = tool_span`. This is the "run_scope /
> caller-context propagation" the existing `_on_agent_start` comment anticipates.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/chris/cubepi && uv run pytest tests/tracing/test_subagent_nesting.py::test_inner_run_nests_under_active_tool_span -v`
Expected: PASS.

- [ ] **Step 5: Regression**

Run: `cd /home/chris/cubepi && uv run pytest tests/tracing/ -q`
Expected: PASS — single-agent roots still have `parent is None`.

- [ ] **Step 6: Commit**

```bash
cd /home/chris/cubepi
git add cubepi/tracing/recorder.py tests/tracing/test_subagent_nesting.py
git commit -m "feat(tracing): nest inner-agent invoke_agent under active tool span

When a recorder opens invoke_agent while an execute_tool span is active
for the task (a subagent running inside a tool body), parent the root
under that tool span so the whole inner run nests in the same trace."
```

---

## Task 3: End-to-end cubepi test — nested subagent produces a nested subtree

Proves Tasks 1+2 compose: an inner agent attached to the same Tracer, run inside an `execute_tool` body, yields `execute_tool → invoke_agent → cubepi.turn → {chat, execute_tool}` nesting.

**Files:**
- Test: `tests/tracing/test_subagent_nesting.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_full_nested_subtree_via_real_tool(tmp_path):
    """A tool whose body attaches+runs an inner agent that ITSELF calls a tool
    yields the full nested subtree: the inner invoke_agent AND its turn / chat /
    execute_tool spans all descend from the outer execute_tool span."""
    tracer, exporter = build_tracer_with_memory_exporter()
    # Replies are consumed in cross-agent CALL order. The inner agent runs
    # INSIDE the spawn tool body, BEFORE the parent's post-tool turn, so its
    # replies come before `parent-final` (codex SHOULD-FIX: original order was
    # wrong — the inner would have eaten `parent-final`).
    provider = FauxProvider(scripted_replies=[
        {"tool_calls": [{"id": "tc1", "name": "spawn", "arguments": {}}]},   # parent T1
        {"tool_calls": [{"id": "tc2", "name": "inner_tool", "arguments": {}}]},  # inner T1
        "inner-final",                                                         # inner T2
        "parent-final",                                                        # parent T2
    ])

    async def _inner_tool(tool_call_id, args, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text="inner-tool-ok")])

    inner_tool = AgentTool(name="inner_tool", description="inner work",
                           parameters=_Empty, execute=_inner_tool)

    async def _spawn(tool_call_id, args, *, signal=None, on_update=None):
        inner = make_agent(provider=provider, tools=[inner_tool])
        detach = tracer.attach(inner)
        try:
            await inner.prompt("do the thing")
        finally:
            res = detach()
            if res is not None:
                await res
        return AgentToolResult(content=[TextContent(text="ok")])

    spawn = AgentTool(name="spawn", description="spawn inner", parameters=_Empty, execute=_spawn)
    parent = make_agent(provider=provider, tools=[spawn])
    detach_parent = tracer.attach(parent)
    try:
        await parent.prompt("go")
    finally:
        res = detach_parent()
        if res is not None:
            await res
    await tracer.shutdown()

    spans = exporter.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    tool_span = next(s for s in spans if s.name == "execute_tool spawn")
    inner_root = next(s for s in spans
                      if s.attributes.get("gen_ai.operation.name") == "invoke_agent"
                      and s.parent is not None
                      and s.parent.span_id == tool_span.context.span_id)

    def _descends_from(span, ancestor_id):
        cur = span
        while cur is not None and cur.parent is not None:
            if cur.parent.span_id == ancestor_id:
                return True
            cur = by_id.get(cur.parent.span_id)
        return False

    # (a) inner chat descends from the inner root (not the parent turn).
    inner_chats = [s for s in spans
                   if s.attributes.get("gen_ai.operation.name") == "chat"
                   and _descends_from(s, inner_root.context.span_id)]
    assert inner_chats, "inner chat span did not nest under the inner run"
    # (b) the inner agent's OWN execute_tool span exists AND nests under the
    #     inner root — this is the regression guard for the original
    #     'missing inner tool spans' bug (codex SHOULD-FIX).
    inner_tool_spans = [s for s in spans
                        if s.name == "execute_tool inner_tool"
                        and _descends_from(s, inner_root.context.span_id)]
    assert inner_tool_spans, "inner execute_tool span missing or not nested under the inner run"
```

> Adjust `FauxProvider`'s scripted-reply shape to match the existing harness (some fakes take a list of `AssistantMessage`s, others a callable). The intent: parent calls `spawn`; the inner agent then calls `inner_tool` and finalizes; the parent finalizes last. `_Empty` is defined once in Task 1's test module.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/chris/cubepi && uv run pytest tests/tracing/test_subagent_nesting.py::test_full_nested_subtree_via_real_tool -v`
Expected: PASS once Tasks 1+2 are in (this is the integration guard). If it FAILS, the active-tool-span contextvar is not visible in the tool body's task — investigate `execute_tool_calls` task creation ordering before patching further (do NOT add a second fix on top).

- [ ] **Step 3: Commit**

```bash
cd /home/chris/cubepi
git add tests/tracing/test_subagent_nesting.py
git commit -m "test(tracing): end-to-end nested-subagent subtree"
```

---

## Task 4: cubebox — attach the parent Tracer to the inner agent

Make the inner subagent run actually recorded (and, with Tasks 1–2, nested) by attaching the active `Tracer` around `inner.prompt()`.

**Files:**
- Modify: `backend/cubebox/middleware/subagents.py` (`SubAgentMiddleware.__init__` to accept `tracer`; `_execute` to attach/detach)
- Modify: `backend/cubebox/streams/run_manager.py` (pass the active `Tracer` into `SubAgentMiddleware`)
- Test: `backend/tests/middleware/test_subagent_trace.py`

- [ ] **Step 1: Confirm the Tracer source in run-manager (codex SHOULD-FIX — be exact)**

The process-level Tracer is on app state. `SubAgentMiddleware` is constructed at `backend/cubebox/streams/run_manager.py:1282` (`subagent_mw = SubAgentMiddleware(...)`), but the local `tracer = getattr(self._app.state, "tracer", None)` is only read LATER at line ~1467 (where the parent agent is attached via cubepi's best-effort `trace` scope). So at construction time the local does not yet exist — read it directly:

Run: `cd /home/chris/cubebox && grep -n "SubAgentMiddleware(\|getattr(self._app.state, \"tracer\"\|app.state.tracer" backend/cubebox/streams/run_manager.py`
Expected: construction at ~1282, tracer accessor at ~1467. The fix passes `tracer=getattr(self._app.state, "tracer", None)` into the constructor at 1282 (it's `None` when tracing is disabled — handled below).

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/middleware/test_subagent_trace.py
import pytest
from asyncio import CancelledError


class _FakeTracer:
    def __init__(self):
        self.attached = []
        self.detached = 0
    def attach(self, agent):
        self.attached.append(agent)
        def _detach():               # mirror Tracer.attach: sync call, may return awaitable
            self.detached += 1
            return None
        return _detach


@pytest.mark.asyncio
async def test_subagent_attaches_and_detaches_tracer_on_success():
    """SubAgentMiddleware attaches the provided Tracer to the inner agent and
    detaches in finally on the success path."""
    tracer = _FakeTracer()
    # Build SubAgentMiddleware(tracer=tracer, ...) with a stub subagent_map and
    # a provider whose inner run completes normally; invoke the subagent tool.
    ...  # use the existing SubAgentMiddleware test harness (Step note)
    assert len(tracer.attached) == 1, "inner agent was not attached"
    assert tracer.detached == 1, "detach() not called in finally"


@pytest.mark.asyncio
async def test_subagent_detaches_tracer_when_inner_run_fails():
    """Even when the inner run raises a normal Exception, detach() still runs
    (finally) and the tool returns an is_error result rather than propagating."""
    tracer = _FakeTracer()
    # Build the middleware with an inner provider that raises Exception mid-run;
    # invoke the subagent tool (it returns an is_error result, not raising).
    ...
    assert len(tracer.attached) == 1
    assert tracer.detached == 1, "detach() not called on the failure path"


@pytest.mark.asyncio
async def test_subagent_detaches_tracer_when_inner_run_cancelled():
    """CancelledError is BaseException — NOT caught by `except Exception` — so it
    propagates out of the tool body, but the `finally` block must still detach
    (codex SHOULD-FIX: cancellation path was untested)."""
    tracer = _FakeTracer()
    # Build the middleware with an inner provider that raises CancelledError;
    # invoke the subagent tool and expect the CancelledError to propagate.
    ...
    with pytest.raises(CancelledError):
        await _invoke_subagent_tool(...)  # adapt to the harness's invocation helper
    assert len(tracer.attached) == 1
    assert tracer.detached == 1, "detach() not called on the cancellation path"


@pytest.mark.asyncio
async def test_run_manager_passes_process_tracer_to_subagent_middleware(monkeypatch):
    """run_manager constructs SubAgentMiddleware with the app-state tracer."""
    # Arrange app.state.tracer = sentinel; drive the run-manager path that
    # builds SubAgentMiddleware; assert the middleware received tracer=sentinel.
    ...
```

> Flesh out the `...` blocks with the existing subagent middleware test harness (grep `backend/tests` for `SubAgentMiddleware`). The three invariants that matter: (1) inner agent attached, (2) detach runs on BOTH success and failure paths, (3) run-manager wires the app-state tracer in.

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /home/chris/cubebox/backend && uv run pytest tests/middleware/test_subagent_trace.py -v`
Expected: FAIL — inner agent never attached / tracer never threaded.

- [ ] **Step 4: Thread the Tracer through and attach (best-effort)**

In `SubAgentMiddleware.__init__`, add an optional `tracer: Any = None` parameter and store `self._tracer = tracer`. In `_execute`, wrap the run so tracing can NEVER break the subagent (mirrors the parent's best-effort attach in run_manager):

```python
            detach = None
            if self._tracer is not None:
                try:
                    detach = self._tracer.attach(inner)
                except Exception as exc:  # tracing must never break the run
                    logger.debug("subagent tracer.attach failed: {}", exc)
                    detach = None
            try:
                await inner.prompt(args.prompt)
            except Exception as exc:
                ...  # unchanged existing error handling -> returns is_error result
            finally:
                if detach is not None:
                    try:
                        res = detach()
                        if res is not None:
                            await res
                    except Exception as exc:
                        logger.debug("subagent tracer.detach failed: {}", exc)
```

In `run_manager.py:1282`, add the `tracer=` kwarg to the EXISTING construction (do not change the other args — the real call passes `default_provider_name`, `shared_tools`, `inherited_middleware`, NOT `metadata`):

```python
            subagent_mw = SubAgentMiddleware(
                subagent_map={},
                default_provider=provider,
                default_model_id=model_id,
                default_provider_name=provider_name,
                shared_tools=_sandbox_tools + _artifact_tools + _builtin_tools,
                inherited_middleware=_cost_mw_for_inherit,
                tracer=getattr(self._app.state, "tracer", None),  # NEW
            )
```

> Same Tracer instance ⇒ same exporters ⇒ inner spans land in the parent's trace file. Task 1's active-run gate prevents double chat spans; Task 2 nests the inner root under `execute_tool subagent`. `tracer=None` (tracing disabled) ⇒ the attach block is skipped, behavior unchanged.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/chris/cubebox/backend && uv run pytest tests/middleware/test_subagent_trace.py -v`
Expected: PASS (all three).

- [ ] **Step 6: Commit**

```bash
cd /home/chris/cubebox
git add backend/cubebox/middleware/subagents.py backend/cubebox/streams/run_manager.py backend/tests/middleware/test_subagent_trace.py
git commit -m "fix(subagents): attach Tracer to inner agent so its run is traced

Inner subagent runs were invisible to the recorder (no nested turn/tool
spans) and leaked chat spans onto the parent turn. Attach the active
Tracer to the inner agent; with the cubepi run-nesting fix its spans now
nest under the execute_tool subagent span."
```

---

## Task 5: Bump the cubepi pin in cubebox and verify end-to-end

cubepi changes (Tasks 1–3) reach cubebox runtime only via the pinned `rev`.

**Files:**
- Modify: `backend/uv.lock` (via `uv`), `backend/pyproject.toml` if the rev is pinned there

- [ ] **Step 1: Push cubepi changes and capture the new rev**

```bash
cd /home/chris/cubepi && git push && git rev-parse HEAD
```

- [ ] **Step 2: Bump the pin**

```bash
cd /home/chris/cubebox/backend && uv lock --upgrade-package cubepi
```
> Do not hand-edit `uv.lock`. Confirm the new `rev` matches the pushed cubepi HEAD: `grep -A1 'name = "cubepi"' uv.lock`.

- [ ] **Step 3: Run a real subagent task and inspect the trace**

Run a subagent-spawning prompt, then:
```bash
cd /home/chris/cubebox/backend && uv run python -m cubepi.cli trace view <run_id> --dir cubepi-traces
```
Expected: under each `execute_tool subagent [..]` node there is now a nested `invoke_agent → cubepi.turn → {chat, execute_tool}` subtree (with span_ids shown), instead of flat sibling chats. Confirm no duplicate chat spans.

- [ ] **Step 4: Commit the pin bump**

```bash
cd /home/chris/cubebox
git add backend/uv.lock backend/pyproject.toml
git commit -m "build(deps): bump cubepi pin for subagent trace nesting"
```

---

## Self-Review Notes

- **Spec coverage:** Task 1 = shared-provider routing (spec §6 provider listeners); Task 2 = nested-agent caller-context (spec §6.1 / §8); Task 4 = the cubebox side the spec assumes ("inner agent with its own Tracer"). The span_id display ask is already shipped in `render.py` (out of scope here).
- **Known risk (call out before coding Task 3):** if `_get_tool_span_entry()` is not visible inside the parallel tool body's task, the inner root won't nest. The contextvar copies at child-task creation, and `register_tool_span` runs in `_on_tool_exec_start` *before* the body task is created, so it should be visible — but Task 3's integration test is the gate. If it fails, fix the contextvar propagation, not the symptom.
- **Type consistency:** `tracer.attach(agent)` returns a `detach` callable that may return an awaitable `Task` or `None`; Tasks 1/3/4 all handle both (`res = detach(); if res is not None: await res`).
- **Worktree (per project discipline):** execute this plan in a dedicated worktree, not on `main`, in BOTH repos. The span_id `render.py` change is currently uncommitted on `~/cubepi` `main` — fold it into the worktree branch or commit it separately before starting.

### Codex review (2026-05-27) — folded in

- **[BLOCKING, fixed]** Task 1's original test only ran `inner.prompt()`, so the parent recorder's `_run` was `None` and its listener already early-returned — the test wasn't red for the real bug. Rewritten to drive the parent through a turn that spawns the inner agent (production shape); pre-fix expects **4** chat spans, post-fix **3**.
- **[SHOULD-FIX, fixed]** Cancellation: `CancelledError` bypasses `AgentEndEvent`, so the `_active_run` reset now lives in `_close_open_spans` (the detach path), with a dedicated sequential-mode cancellation test (Task 1 Step 6).
- **[SHOULD-FIX, fixed]** Parallel-mode contextvar shadowing argument is now written out explicitly in Task 1 Step 3.
- **[SHOULD-FIX, fixed]** Task 3 now makes the inner agent call a real `inner_tool` and asserts the inner `execute_tool` span descends from the inner root (regression guard for "missing inner tool spans"); FauxProvider replies reordered to tool-call → inner-tool → inner-final → parent-final.
- **[SHOULD-FIX, fixed]** Task 4 test now asserts `detach()` on success AND failure paths and that run-manager wires `app.state.tracer`; tracer source pinned to `run_manager.py:1282` via `getattr(self._app.state, "tracer", None)`, attach/detach wrapped best-effort.
- **[NIT, fixed]** Task 2 wording now distinguishes the `context=None` (ambient/today) branch from the tool-span-parent branch.
- **[NIT, noted]** `_get_tool_span_entry()` lives in `cubepi.mcp._tracing`; the recorder already depends on that module, so Task 2 adds no new layering breach. Long-term this "current tool span" helper should move into a tracing-owned module since it now serves generic nested agents, not just MCP clients — out of scope for this plan; leave a `# TODO(tracing): relocate current-tool-span helper out of mcp` where Task 2 reads it.
