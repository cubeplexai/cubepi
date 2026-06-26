# Run-scoped Compaction & Real-token Triggering — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `CompactionMiddleware` compress history *within* a single long
agentic run (not just between user turns), and trigger compaction on the *real*
context token count instead of a cache-blind char estimate.

**Spec:** [`dev/specs/2026-06-26-run-scoped-compaction.md`](../specs/2026-06-26-run-scoped-compaction.md)

**Architecture:** Two surgical changes inside `cubepi/middleware/compaction/`, no
`loop.py` / `agent.py` / public-API changes:
1. `boundary.py` — drop the `UserMessage`-only gate in `safe_boundary`; rely on
   the existing suffix self-containment check. Turn-boundary `AssistantMessage`
   cuts become legal, so the boundary advances inside a run.
2. `tokens.py` + `__init__.py` — add a real-token estimate that prefers the most
   recent `AssistantMessage.usage` (summing `input + cache_read + cache_write`)
   plus the estimated delta since; switch the trigger onto it; delete the
   `scale_factor` calibration (its only job was absolute-vs-threshold, now served
   by real numbers).

A third spec item (§2.3, oversized tool results) ships **docs only** — the
existing `after_tool_call` seam is the mechanism; no code.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest (`asyncio_mode=auto`), mypy
strict, `FauxProvider` for deterministic runs.

---

## File Structure

- **Modify** `cubepi/middleware/compaction/boundary.py:42-74` — remove
  `UserMessage` gate in `safe_boundary`; reword docstring.
- **Modify** `cubepi/middleware/compaction/tokens.py` — remove `scale_factor`
  block (lines 15, 24, 37-46, 53-54); add `real_context_estimate(messages)`.
- **Modify** `cubepi/middleware/compaction/__init__.py:209-212` — switch trigger
  from `approx_tokens(...)` to `real_context_estimate(...)`.
- **Modify** `tests/middleware/compaction/test_boundary.py` (or equivalent) — add
  run-scoped boundary cases.
- **Modify/Create** `tests/middleware/compaction/test_tokens.py` — real-token
  estimate + calibration-removal cases.
- **Modify** `tests/middleware/compaction/test_*.py` — update any test asserting a
  `UserMessage` boundary or relying on the `scale_factor`.
- **Modify** `website/docs/guides/middleware/compaction.md` — document in-run
  compaction, real-token triggering, and the `after_tool_call` seam for oversized
  tool results.

> Confirm exact test-file paths first: `ls tests/middleware/compaction/`.

---

## Task 1: Run-scoped boundary (§2.1)

**Files:**
- Modify: `cubepi/middleware/compaction/boundary.py`
- Test: `tests/middleware/compaction/test_boundary.py`

- [ ] **Step 1: Write the failing tests**

  Build a message list that mimics a long run with **no intermediate
  `UserMessage`**: `User(open)`, then repeated `Assistant(tool_use)` +
  `ToolResult` pairs, plus a multi-tool turn (`Assistant(tool_use A, tool_use B)`,
  `ToolResult A`, `ToolResult B`). Assert:
  - `safe_boundary(msgs, tail_start=…)` returns an index pointing at a turn-start
    `AssistantMessage` (not pinned to the opening user message).
  - `_suffix_is_self_contained(msgs[boundary:])` holds for the returned index.
  - The boundary is never *between* `ToolResult A` and `ToolResult B`, and never
    between a `tool_use` and its `ToolResult` (parametrize over every index;
    accepted indices must all be self-contained).
  - `ToolResultMessage` indices are never returned.
  - Existing user-boundary behaviour still works (regression: a normal multi-run
    history still cuts cleanly).

- [ ] **Step 2: Implement**

  In `safe_boundary` (`boundary.py:63-72`), delete the gate:
  ```python
  if not isinstance(messages[candidate], UserMessage):
      candidate -= 1
      continue
  ```
  Keep the `_suffix_is_self_contained` check and the `min_compact` guard. Reword
  the docstring (`boundary.py:42-53`): "searches the prefix for the latest
  **self-contained turn boundary**" (drop the "latest `UserMessage`" wording).
  Remove the now-unused `UserMessage` import if nothing else uses it.

- [ ] **Step 3: Verify** — `uv run pytest tests/middleware/compaction/test_boundary.py -v`, `uv run mypy cubepi`, `uv run ruff check cubepi/ tests/`.

---

## Task 2: Real-token trigger + remove calibration (§2.2)

**Files:**
- Modify: `cubepi/middleware/compaction/tokens.py`, `cubepi/middleware/compaction/__init__.py`
- Test: `tests/middleware/compaction/test_tokens.py`

- [ ] **Step 1: Write the failing tests**

  - `real_context_estimate(messages)`:
    - With a trailing `AssistantMessage` carrying `usage`
      (`input_tokens=1000, cache_read_tokens=50_000, cache_write_tokens=2_000`)
      and two messages appended after it, returns
      `53_000 + approx_tokens(tail_two)` — i.e. **sums all three** usage fields,
      not `input_tokens` alone.
    - With **no** usage anywhere (cold start), falls back to
      `approx_tokens(messages)`.
    - Picks the **most recent** usage-bearing assistant when several exist.
  - `approx_tokens` no longer applies a scale factor: an `AssistantMessage` with
    `usage` produces the same value as one without (pure `chars/2`).
  - End-to-end via `FauxProvider` reporting cache usage (`faux.py:262-265`):
    compaction triggers when the *combined* real count crosses the threshold even
    though `input_tokens` alone is well under it.

- [ ] **Step 2: Implement**

  - `tokens.py`: add
    ```python
    def real_context_estimate(messages: list[Message]) -> int:
        # walk backward to the most recent AssistantMessage with usage;
        # base = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
        # return base + approx_tokens(messages_after_that_assistant)
        # no usage anywhere -> approx_tokens(messages)
    ```
  - `tokens.py`: delete the calibration — `scale_factor` var (line 24), the
    `usage` lookup + clamp block (lines 37-46), the `* scale_factor` return path
    (lines 53-54), and `_SCALE_MIN_TOKENS` (line 15). `approx_tokens` becomes a
    pure char estimate. Keep the per-message-type char accounting intact.
  - `__init__.py:209-212`: replace the trigger
    ```python
    tokens_now = approx_tokens(unpruned_compressed)
    if tokens_now < self._max_tokens_before:
    ```
    with `tokens_now = real_context_estimate(unpruned_compressed)`. **Leave
    `raw_tokens` (anti-thrash, `__init__.py:256`), `tokens_after`
    (`__init__.py:346`), `tail_start_by_tokens`, and the summary budget on
    `approx_tokens`** — those are relative/budget decisions where a consistent
    bias cancels.

- [ ] **Step 3: Verify** — `uv run pytest tests/middleware/compaction/ -v`, `uv run mypy cubepi`, `uv run ruff check`. Note any tail-size drift caused by uncalibrated `approx_tokens` and confirm anti-thrash tests still pass (retune only if red — see Open Q2).

---

## Task 3 (CONDITIONAL — gated on spec Open Q1): threshold auto-binding

> Implement **only** if the user chooses to auto-default the threshold. Otherwise skip.

- [ ] Default `max_tokens_before_compact` from the bound model when omitted:
  `int(ratio * model.context_window)` (`providers/base.py:91`), ratio per the
  user's answer. Add a test that an unset threshold derives from `context_window`
  and an explicit threshold still wins. Document the default in the docs page.

---

## Task 4: Documentation (§2.3 seam + the two features)

**Files:**
- Modify: `website/docs/guides/middleware/compaction.md`

- [ ] **Step 1:** Document **in-run compaction** — compaction now advances the
  summary boundary at turn boundaries inside a single run, not only between user
  turns.
- [ ] **Step 2:** Document **real-token triggering** — the trigger uses the true
  context fill (`input + cache_read + cache_write` from the last turn's usage),
  so thresholds behave correctly under prompt caching.
- [ ] **Step 3:** Add a short **"Bounding oversized tool results"** note: this is
  the application's responsibility via an `after_tool_call` middleware (link
  `hooks.md`), because cubepi is environment-agnostic and cannot assume where to
  persist spilled content. Show a minimal sketch that previews + replaces a large
  result using `AfterToolCallResult.content`.

---

## Final verification

- [ ] `uv run pytest tests/` (full suite green).
- [ ] `uv run ruff check cubepi/ tests/` and `uv run ruff format --check cubepi/ tests/`.
- [ ] `uv run mypy cubepi`.
- [ ] Re-read the spec's §5 test-plan checklist; confirm every bullet has a test.
- [ ] Docs updated (feature is not done without docs — CLAUDE.md §4).

## Open decisions to confirm before/while implementing

1. **Threshold auto-binding** (Task 3) — do it now and with what ratio, or keep
   `max_tokens_before_compact` explicit?
2. **Anti-thrash tuning** — keep `_ANTI_THRASH_NEW_MSGS = 8` for the tighter
   in-run cadence (default: keep, verify via tests), or retune?
