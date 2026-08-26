# Trailing Synthetic Controls Compaction Repair — Implementation Plan

> **For agentic workers:** Execute with strict RED → GREEN. Do not write production
> code until the focused production-shaped regression has failed for the expected
> reason.

**Goal:** Prevent trailing synthetic user controls from aging and pruning current-turn
tool evidence while preserving threshold accounting, message order, state refs, and
all existing compaction behavior.

**Spec:** [`dev/specs/2026-08-26-compaction-synthetic-controls.md`](../specs/2026-08-26-compaction-synthetic-controls.md)

**Architecture:** Add one private split helper in the compaction module. Run
state/tail/boundary/pruning/summarization over the ordinary history prefix, measure the
actual model-send view with controls attached, then reattach the unchanged controls on
every return path. No public API or state schema changes.

**Tech stack:** Python 3.11+, Pydantic v2 message models, pytest asyncio tests, Ruff,
mypy.

---

## File structure

- **Modify** `cubepi/middleware/compaction/__init__.py`
  - import `UserMessage` and `is_synthetic_message`;
  - add a private maximal-suffix split helper;
  - apply compaction decisions to history and reattach controls.
- **Modify** `tests/middleware/test_compaction.py`
  - add the production-shaped RED and focused controls.
- **Modify** `website/docs/guides/middleware/compaction.md`
  - document trailing synthetic-control behavior and ordering.
- **Create** this spec and plan.

## Task 1: Establish the real RED

- [ ] Add a test helper that creates two large, non-chip tool results with unique
  sentinels and one trailing `synthetic_user_message`.
- [ ] Use the production-relevant settings:
  - `max_tokens_before_compact=55_000`
  - `keep_tail_tokens=8_000`
  - `min_compact_messages=4`
- [ ] Message shape must be one assistant turn with two tool calls followed by their
  two results and the synthetic control.
- [ ] Assert the returned model view retains both full sentinels and the exact trailing
  control, and that no summary call/state is produced when history alone has no legal
  boundary.
- [ ] Run only that test and observe the expected RED: current CubePi compacts at the
  synthetic control and removes both result sentinels.

Command:

```bash
uv run pytest \
  tests/middleware/test_compaction.py::test_trailing_synthetic_control_does_not_age_current_tool_results \
  -vv
```

## Task 2: Implement the smallest generic repair

- [ ] Add `_split_trailing_synthetic_user_controls(messages)` returning a copied
  history prefix and ordered controls suffix.
- [ ] Match only `UserMessage` + `is_synthetic_message`; do not inspect source names or
  payload text.
- [ ] In `transform_context`, use history for:
  - state validation and refs;
  - tail computation and safe boundary;
  - pruning;
  - summarizer/ref slices;
  - preserved-result state.
- [ ] Build the threshold view as compressed history + controls so controls still count.
- [ ] Return history/compressed history + controls from every exit path.
- [ ] Keep raw/savings accounting over full input/output views.
- [ ] Run the focused RED again and observe GREEN.

## Task 3: Add boundary and compatibility controls

Write the remaining tests before expanding production code beyond Task 2. The Task 2
implementation should satisfy them without source-specific branches.

- [ ] Multiple trailing synthetic user controls survive unchanged and in exact order.
- [ ] A large trailing control contributes to threshold crossing; an otherwise
  under-threshold compactable history triggers compaction.
- [ ] Older history compacts while a current multi-tool turn keeps all raw results.
- [ ] A synthetic tail `ToolResultMessage` is not detached from its tool call.
- [ ] No-suffix behavior stays unchanged (existing tests plus an explicit equality
  control if needed).
- [ ] State refs remain valid across a repeated transform and when a formerly trailing
  synthetic user control becomes internal after an assistant message is appended.

Run:

```bash
uv run pytest tests/middleware/test_compaction.py -v
uv run pytest tests/middleware/compaction/ -v
uv run pytest tests/test_synthetic_messages.py -v
```

## Task 4: Document the behavior

Update `website/docs/guides/middleware/compaction.md`:

- [ ] explain that trailing synthetic `UserMessage` controls count toward total context
  but do not consume the historical tail or create a boundary;
- [ ] explain that they are reattached after the compressed history in original order;
- [ ] state that only a trailing user-role synthetic suffix is treated this way and
  synthetic tool results remain in history to preserve pairing;
- [ ] keep docs generic; do not mention Bot tool names or application payloads.

## Task 5: Verify CubePi comprehensively

- [ ] Focused tests:

```bash
uv run pytest tests/middleware/test_compaction.py -v
uv run pytest tests/middleware/compaction/ -v
uv run pytest tests/test_synthetic_messages.py -v
```

- [ ] Full test suite:

```bash
uv run pytest tests/
```

- [ ] Static gates:

```bash
uv run ruff check cubepi/ tests/
uv run ruff format --check cubepi/ tests/
uv run mypy cubepi
```

- [ ] Inspect `git diff --check` and the exact diff.
- [ ] Obtain independent spec-compliance review, then independent code-quality review.
- [ ] Resolve and re-review every confirmed finding.

Do not start CubePi's optional local Codex loop without separate permission under its
project instructions. Opening a PR later will still receive the repository's normal PR
Codex review.

## Task 6: Upstream delivery and Cubemanus adoption

Only after Task 5 is clean:

- [ ] Commit the CubePi branch; push and open a PR against `main`.
- [ ] Wait for exact-SHA CI and required PR review; do not merge before both are clean.
- [ ] Establish a consumable upstream commit/release.
- [ ] In the isolated Cubemanus worktree, update only the dependency pin/lock plus a
  real Bot lifecycle regression using actual Temporal Freshness → Compaction ordering.
- [ ] Run focused and full Bot tests plus independent reviews.
- [ ] Deliver through feature MR → `dgts` → exact-SHA build → `cuecue-test` deploy.
- [ ] Rerun the large report synthesis E2E and verify full evidence reaches synthesis,
  with no duplicate tool calls and stable SSE/replay/transcript/browser behavior.

Hard stop before `master`, PROD, ConfigMaps, Secrets, or unrelated MRs.
