# Live ToolResult `run_id` Hotfix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure every live `ToolResultMessage` created for a run-aware Agent tool turn carries the owning assistant turn's `run_id` before emission, hooks, live-context append, checkpointing, and the next provider call.

**Architecture:** Keep `execute_tool_calls` public behavior and signatures compatible. Make the current assistant message's `run_id` visible on the live assistant object at the existing Agent message-end stamping seam, then pass that value explicitly through the private tool-result construction/emission boundary. Direct calls with an unstamped assistant continue to produce `run_id=None`; no run ID is inferred from call IDs or other global state during tool-result construction.

**Tech Stack:** Python 3.13, asyncio, Pydantic models, `FauxProvider`, `MemoryCheckpointer`, pytest, Ruff, mypy, uv.

---

### Task 1: Pin the Regression at the Real Agent Boundary

**Files:**
- Modify: `tests/agent/test_agent_run_id.py`

- [ ] **Step 1: Add a production-shaped Agent/FauxProvider tool-cycle test**

Create an `Agent` with a real `AgentTool`, `MemoryCheckpointer`, and `run_id="R1"`. Queue a tool-call assistant response followed by a Faux response factory. In the second response factory, assert the just-appended `ToolResultMessage.run_id == "R1"`. Capture `should_stop_after_turn` inputs and tool-result message events so the test also verifies the live context, `new_messages`, hook input, emitted message, `Agent.state.messages`, and checkpointed message all carry exactly `"R1"`.

- [ ] **Step 2: Run the production-shaped test and record RED**

Run:

```bash
uv run pytest tests/agent/test_agent_run_id.py::test_live_tool_results_carry_owning_run_id -vv
```

Expected on base `94d0ca7380502506042390ce73a02de1060387b0`: FAIL in the second Faux response factory because the live `ToolResultMessage.run_id` is `None`, while the `MemoryCheckpointer` copy is stamped `"R1"`.

### Task 2: Pin Executor Outcome and Compatibility Contracts

**Files:**
- Modify: `tests/agent/test_tools.py`

- [ ] **Step 1: Add stamped-assistant executor coverage**

Add focused tests that pass an `AssistantMessage(run_id="R-tools")` into `execute_tool_calls` and assert both emitted `MessageStartEvent`/`MessageEndEvent` messages and returned `ToolCallBatch.messages` retain `"R-tools"` for:

- a mixed parallel batch containing success, validation failure, tool-not-found, tool-body error, and `before_tool_call` blocked outcomes;
- a sequential batch with multiple results.

Also assert message ordering, content/error classification, and tool-call IDs remain unchanged.

- [ ] **Step 2: Add the unstamped legacy control**

Call `execute_tool_calls` directly with an assistant whose `run_id` is `None` and assert every returned and emitted `ToolResultMessage.run_id` remains `None`.

- [ ] **Step 3: Run the focused executor tests and record RED**

Run the new test node IDs with `uv run pytest ... -vv`.

Expected on the pinned base: stamped-assistant assertions FAIL because `_make_tool_result_message` omits `run_id`; the unstamped control PASSes.

### Task 3: Implement the Smallest Construction-Boundary Fix

**Files:**
- Modify: `cubepi/agent/agent.py`
- Modify: `cubepi/agent/tools.py`

- [ ] **Step 1: Preserve the owning ID on the live assistant turn**

At `Agent._process_event`'s existing `MessageEndEvent` stamping seam, ensure an unstamped `AssistantMessage` that belongs to the active run is stamped on the same live object before the loop proceeds to `execute_tool_calls`. Preserve the existing mismatch rejection and persistence behavior. Avoid mutating caller-supplied user messages or adding a run ID to stateless/legacy flows.

- [ ] **Step 2: Pass the assistant ID into tool-result construction**

Change the private `_make_tool_result_message` helper to accept `run_id: str | None` and set `ToolResultMessage.run_id`. Pass `assistant_message.run_id` from sequential construction and through `_emit_tool_result_messages` for every parallel, salvage, cancellation, HITL-sibling, immediate, success, and error outcome.

- [ ] **Step 3: Run the new tests GREEN**

Run:

```bash
uv run pytest tests/agent/test_agent_run_id.py tests/agent/test_tools.py -vv
```

Expected: all tests PASS, including live provider context, emitted event, hook, multiple-result, outcome-matrix, and unstamped-assistant controls.

### Task 4: Verify Regressions and Quality Gates

**Files:**
- Verify only; no new production scope.

- [ ] **Step 1: Run focused Agent/tool/run-ID/checkpointer suites**

```bash
uv run pytest tests/agent/test_agent_run_id.py tests/agent/test_tools.py tests/agent/test_loop.py tests/agent/test_checkpointer_integration.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the full test suite**

```bash
uv run pytest tests/
```

Expected: PASS; record exact passed/skipped counts.

- [ ] **Step 3: Run static and formatting checks**

```bash
uv run ruff check cubepi/ tests/
uv run ruff format --check cubepi/ tests/
uv run mypy cubepi
```

Expected: all PASS; record exact output/counts.

### Task 5: Produce One Local Hotfix Commit and Applicability Report

**Files:**
- Commit only the plan, focused tests, and minimal source fix.

- [ ] **Step 1: Inspect the final diff**

Verify there are no unrelated refactors, public API changes, Cubemanus changes, release/tag changes, or external publishing actions.

- [ ] **Step 2: Commit once locally**

```bash
git add dev/plans/2026-07-31-live-tool-result-run-id.md cubepi/agent/agent.py cubepi/agent/tools.py tests/agent/test_agent_run_id.py tests/agent/test_tools.py
git commit -m "fix(agent): preserve tool result run ids

Co-Authored-By: Claude <noreply@anthropic.com>"
```

- [ ] **Step 3: Verify clean worktree and main applicability**

Record the exact commit SHA and changed files. Fetch no remote state unless already available; compare against current `origin/main` and use a temporary, non-publishing applicability check (`git apply --check` against the patch or an isolated temporary worktree) without mutating `main`.

Expected: clean worktree and an explicit clean/conflicting applicability result.
