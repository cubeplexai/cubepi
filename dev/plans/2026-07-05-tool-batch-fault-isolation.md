# Plan — Parallel Tool-Batch Fault Isolation & Checkpoint Load Hardening

- Date: 2026-07-05
- Spec: `dev/specs/2026-07-05-tool-batch-fault-isolation.md`
- Branch: `2026-07-05-tool-batch-fault-isolation`

## Step 1 — broaden the narrow catches (`cubepi/agent/tools.py`)

1. `_execute_prepared` (`:301-304`): replace the
   `except HitlControlException: raise / except Exception` pair with a
   control-flow re-raise (`HitlControlException`, `asyncio.CancelledError`,
   `KeyboardInterrupt`, `SystemExit`) followed by `except BaseException` →
   `_error_result(str(exc)), True`.
2. `_finalize`'s `after_tool_call` guard (`:353`): same shape. Sequential
   executor inherits both fixes (shared helpers).

## Step 2 — rewrite `_execute_parallel` collect stage (`tools.py:630-651`)

Keep the two-phase prepare and `_run` untouched. Replace the bare-await loop:

1. Build `scheduled` as today (aligned index-wise with `entries`, so a task
   slot's `_PreparedToolCall` — hence `tool_call` id/name and `hitl_trace` —
   is recoverable for synthesis).
2. `await asyncio.gather(*tasks, return_exceptions=True)`.
   - `except asyncio.CancelledError` (outer cancel): re-gather to settle the
     now-cancelled children; best-effort (`except Exception: pass`) emit the
     `ToolResultMessage`s of slots that completed before the cancel (immediate
     outcomes included); re-raise. Agent-layer
     `_complete_cancelled_tool_calls` backfills only the truly-unanswered ids.
3. Classify settled slots in batch order:
   - result present → normal `_FinalizedOutcome`.
   - `HitlControlException` / `KeyboardInterrupt` / `SystemExit` → record
     first as `control_exc`; slot stays unanswered, **no** synthesized End
     event (matches sequential detach shape).
   - `CancelledError` without outer cancel (tool self-cancel) → synthesized
     error outcome, text `[Tool execution cancelled]`.
   - any other exception → synthesized error outcome, text `str(exc)`.
   - Synthesized outcomes emit a `ToolExecutionEndEvent` to pair with the
     Start emitted inside `_run` (keeps `pending_tool_calls`/trace balanced).
4. If `control_exc` is set: best-effort emit the `ToolResultMessage`s for all
   finalized/synthesized slots, then `raise control_exc`.
5. Otherwise: existing tail (build messages, emit, return `ToolCallBatch`).

## Step 3 — `CheckpointCorruptionError` (`cubepi/checkpointer/`)

1. `exceptions.py`: add `CheckpointCorruptionError(CheckpointerError)` with
   `thread_id`, `backend`, `row_ref`, `cause` (spec Part 3 shape).
2. `sqlite.py`:
   - `load()` (`:154-173`): `SELECT id, message_json`; per-row
     `except Exception` → `CheckpointCorruptionError(backend="sqlite",
     row_ref=f"messages.id={id}")`.
   - completed-prefix read (`:415-425`): same (`SELECT id, message_json`).
   - `_deserialize_message` (`:529-537`): unknown role now raises
     `ValueError` instead of returning the raw dict (wrapped by the callers'
     guards; only the two call sites above exist).
3. `postgres/checkpointer.py`: `load()` row loop (`:172-183`) and
   completed-prefix loop (`:392-403`): per-row guard,
   `row_ref=f"seq={r['seq']}"`; the existing unknown-role `ValueError` moves
   inside the guarded block so it wraps uniformly.
4. `mysql/checkpointer.py`: same at `:212-219` and `:475-483`,
   `row_ref=f"seq={seq}"`.
5. Guards catch `Exception` only (never wrap `CancelledError`).

## Step 4 — tests

1. `tests/agent/test_tools.py`, new `TestParallelFaultIsolation`:
   - HITL raise mid-batch → siblings' results emitted (ordered), exception
     re-raised, HITL slot unanswered w/o End event, no still-pending tasks.
   - `after_tool_call` raising a bare `BaseException` subclass → error result
     for that call only (parallel *and* sequential variants).
   - tool-body `CancelledError` (no outer cancel) → `[Tool execution
     cancelled]` error result, siblings unaffected.
   - outer cancellation mid-batch → completed slots' messages emitted before
     `CancelledError` re-raises; unfinished slots unanswered.
   - Start/End pairing for synthesized outcomes.
2. Checkpointer corruption tests (sqlite in `test_sqlite.py`; postgres/mysql
   in their suites behind the existing skip-if-no-DB fixtures): corrupt
   payload row → `CheckpointCorruptionError` with thread_id/backend/row_ref;
   unknown role → same (including sqlite's former silent fallthrough); both
   `load()` and the fork-source read.

## Step 5 — docs (required by workflow)

- `website/docs/guides/agents/tool-use.md`: short "Fault isolation in
  parallel batches" note — one failing tool never drops sibling results;
  HITL/cancel semantics preserved.
- `website/docs/guides/checkpointing/{sqlite,postgres,mysql}.md`:
  troubleshooting entry for `CheckpointCorruptionError` (what it means, that
  `row_ref` locates the bad row).

## Step 6 — gates & PR

`uv run pytest tests/` + `ruff check` + `ruff format --check` + `mypy cubepi`;
commit, push, `gh pr create`; then drive the PR codex review loop
(poll ~2 min → fix → `@codex`) until clean.
