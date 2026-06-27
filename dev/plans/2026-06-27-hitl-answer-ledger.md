# HITL Answer Ledger Implementation Plan

- Date: 2026-06-27
- Spec: `dev/specs/2026-06-27-hitl-answer-ledger.md`
- Status: Draft for review

## Objective

Fix the ROOT-A deadlock where parallel two-phase tool preparation and the
single-slot HITL resume channel can ping-pong between two approval-gated tool
calls forever. The fix is a framework-level durable answer ledger keyed by
thread, run, and question id.

## Constraints

- Do not force approval-gated tools into `execution_mode="sequential"`.
- Do not weaken the two-phase parallel executor's side-effect safety.
- Do not require hosts to classify HITL-gated tools.
- Keep the public visible HITL model as one pending request at a time.
- Run all repo commands through `uv`.

## Phase 1: Add the Checkpointer Ledger Contract

Files:

- `cubepi/checkpointer/base.py`
- `cubepi/checkpointer/memory.py`
- `tests/checkpointer/test_memory_checkpointer.py` or the nearest existing
  memory checkpointer test module

Steps:

1. Import `Iterable` and `StructuredValue` where needed.
2. Add `save_hitl_answer`, `load_hitl_answer`, and `clear_hitl_answers` to the
   `Checkpointer` protocol.
3. Implement the methods in `MemoryCheckpointer` with a private dict keyed by
   `(thread_id, run_id or "", question_id)`.
4. Copy answers on save/load if the existing message/pending code uses deep
   copies for comparable mutable state.
5. Add focused unit tests:
   - saving then loading returns the answer
   - run id scopes answers
   - clearing a single question id preserves other answers
   - clearing all answers for a run removes only that run

Acceptance:

- Memory checkpointer tests pass.
- The protocol names match the spec exactly.

## Phase 2: Add Durable Storage to SQL Checkpointers

Files:

- `cubepi/checkpointer/sqlite.py`
- `cubepi/checkpointer/postgres/checkpointer.py`
- `cubepi/checkpointer/postgres/models.py`
- `cubepi/checkpointer/mysql/checkpointer.py`
- `cubepi/checkpointer/mysql/models.py`
- existing SQL checkpointer tests under `tests/checkpointer/`

Steps:

1. Add a `cubepi_hitl_answers` table to each first-party SQL backend.
2. Use columns:
   - `thread_id`
   - `run_id`, stored as non-null text with `""` representing `None`
   - `question_id`
   - `answer_json`
   - `answered_at`
3. Make `(thread_id, run_id, question_id)` the primary key.
4. Implement upsert semantics for `save_hitl_answer`.
5. Implement exact-key lookup for `load_hitl_answer`.
6. Implement `clear_hitl_answers` with two modes:
   - `question_ids is None`: delete all answers for `(thread_id, run_id)`
   - `question_ids` supplied: delete only those ids for `(thread_id, run_id)`
7. Add CRUD tests for SQLite. Extend Postgres/MySQL tests where the repo already
   runs backend-specific checkpointer HITL tests.

Acceptance:

- SQLite tests pass locally.
- Backend-specific tests are updated without adding hard dependencies to core.

## Phase 3: Teach HITL Channels to Replay Ledger Answers

Files:

- `cubepi/hitl/channel.py`
- `tests/hitl/test_in_memory_channel.py`
- `tests/hitl/test_channel.py` or nearest existing checkpointed channel tests

Steps:

1. Extend `_BaseChannel` with optional ledger hooks:
   - `_load_answer(question_id)`
   - `_save_answer(question_id, answer)`
   - `_clear_answers(...)` only if channel-level abort cleanup is already
     centralized there
2. Implement default no-op behavior for plain in-process channels, then override
   as needed in `InMemoryChannel` and `CheckpointedChannel`.
3. Add an in-memory answer dict to `InMemoryChannel` so same-process behavior
   follows the same replay semantics.
4. Update `CheckpointedChannel.__init__` validation to require the new ledger
   methods when used for detached HITL.
5. Change `_await_answer()` resolution order:
   - load durable/in-memory ledger answer for `question_id`
   - fall back to matching `_resume_slot`
   - otherwise publish pending and wait/detach
6. When `answer()` resolves the currently pending request in process, save the
   answer before waking the waiter.
7. Do not clear a ledger answer when `_await_answer()` returns it.
8. Add tests proving:
   - replayed ledger answers do not create a new pending request
   - stale process-local resume slots still do not satisfy the wrong question id
   - in-process `answer()` also records a replayable answer

Acceptance:

- Existing HITL channel tests still pass.
- New replay tests fail without the ledger lookup and pass with it.

## Phase 4: Persist Answers in Agent.respond and Clear Them Safely

Files:

- `cubepi/agent/agent.py`
- `cubepi/agent/loop.py`
- `cubepi/hitl/channel.py` if abort cleanup belongs there
- `tests/hitl/test_loop_hitl_passthrough.py`
- new or existing agent HITL resume tests

Steps:

1. In `Agent.respond()`, after validating pending `question_id` and recovered
   run id, call `save_hitl_answer(thread_id, question_id, answer, run_id=...)`
   before `_run_hitl_resume()`.
2. Keep `attach_resume_answer()` as a same-process fast path, but make the
   ledger the durable source of truth.
3. Ensure resume detaching on the next pending request does not clear saved
   answers.
4. Clear ledger answers for `(thread_id, active_run_id)` only after the suspended
   tool cycle has completed and tool results have been appended.
5. Clear ledger answers when pending HITL is aborted/cancelled or when the run
   reaches a terminal cleanup path.
6. Prefer clearing all answers for the run over clearing only tool call ids, so
   `ask_user` question ids are covered.
7. Add tests for cleanup:
   - after successful completion
   - after abort
   - no cleanup after detach on another request

Acceptance:

- A saved answer survives a detach on a later tool call.
- Saved answers are not left behind after successful completion or abort.

## Phase 5: Add the ROOT-A Regression Test

Files:

- `tests/hitl/test_loop_hitl_passthrough.py` or a new
  `tests/hitl/test_parallel_approval_ledger.py`

Test shape:

1. Define two tools that require approval and record executions.
2. Use a provider response that requests both tools in the same assistant turn
   with parallel execution enabled.
3. Start `Agent.prompt()` and observe pending `A`.
4. Call `Agent.respond(question_id=A, answer=approve)`.
5. Assert:
   - no tool body has executed yet
   - no tool results have been appended yet
   - pending is now `B`
6. Call `Agent.respond(question_id=B, answer=approve)`.
7. Assert:
   - both tool bodies executed exactly once
   - two tool results were appended
   - pending is cleared
   - ledger answers for the run are cleared
8. Repeat the same flow with a fresh `Agent` instance between the two approvals
   to prove durable replay works after process restart.

Acceptance:

- The test reproduces the current deadlock before the implementation.
- The test passes after the answer ledger implementation.

## Phase 6: Cover Decision Variants

Files:

- `tests/hitl/test_loop_hitl_passthrough.py`
- approval middleware tests under `tests/hitl/`

Steps:

1. Add a deny/reject variant where `A` is denied, resume detaches on `B`, then
   `B` is approved. Verify the replayed denial emits the expected tool result
   and no denied tool body executes.
2. Add an edit variant if the existing approval middleware supports edited tool
   arguments in tests. Verify the edited decision survives the second detach and
   the tool receives edited args.
3. Add an `ask_user` replay case if the existing `ask_user` tests can express a
   multi-question resume flow without building a large fixture.

Acceptance:

- The ledger stores and replays structured answers, not just boolean approvals.

## Phase 7: Update Documentation

Files:

- `website/docs/` HITL guide or recipe page
- Chinese mirror if this docs tree keeps one for the changed page

Steps:

1. Document the new guarantee: approval-gated tools can remain parallel.
2. Explain that approving one request in a parallel batch records the decision,
   but execution waits until the remaining gates are answered.
3. Remove or avoid guidance that tells users to mark gated tools sequential only
   to avoid HITL deadlocks.
4. Add a short example flow with two parallel approval-gated tool calls.

Acceptance:

- User-facing docs match the runtime behavior.
- The feature is not considered complete until docs ship in the same PR.

## Verification Commands

Run focused tests first:

```bash
uv run pytest tests/hitl/test_in_memory_channel.py -v
uv run pytest tests/hitl/test_loop_hitl_passthrough.py -v
uv run pytest tests/checkpointer/ -k "memory or sqlite or hitl" -v
```

Then run broader checks:

```bash
uv run pytest tests/
uv run ruff check cubepi/ tests/
uv run ruff format --check cubepi/ tests/
uv run mypy cubepi
```

## Review Checkpoints

1. Review the spec and this plan before coding.
2. After code and docs are ready, ask before entering the local codex review
   loop required by `AGENTS.md`.
3. Open a PR from the worktree branch after implementation and documentation.
4. Drive the PR codex review loop until review and CI are clean.
