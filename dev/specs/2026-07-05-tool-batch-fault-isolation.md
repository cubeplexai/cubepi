# Parallel Tool-Batch Fault Isolation & Checkpoint Load Hardening

- Date: 2026-07-05
- Status: Draft for review
- Related issue: downstream report — one escaping exception in a parallel tool
  batch drops every `ToolResultMessage` in the batch, persisting dangling
  `tool_calls`; combined with no row-level fault handling in checkpointer
  `load()`, the thread 400s on every subsequent turn (permanently bricked).
- Related spec: `2026-06-27-hitl-answer-ledger.md` (two-phase parallel prepare;
  this spec covers the *execute* phase the two-phase split does not protect)
- Companion plan: `dev/plans/2026-07-05-tool-batch-fault-isolation.md`

## Problem

### 1. One escaping exception destroys the whole parallel batch

`_execute_parallel` collects its fan-out tasks with a bare `await` loop
(`cubepi/agent/tools.py:638-642`). Every `ToolResultMessage` is built only
*after* the loop completes (`:644-649`). If any single task raises an exception
that escapes `_execute_prepared`, the loop re-raises at the first failing
`await` and the message-construction block never runs — **zero** tool results
are emitted, including those of sibling tools that already succeeded.

Three exception classes escape today:

- **`HitlControlException`** — deliberately re-raised by the selective handler
  in `_execute_prepared` (`tools.py:301-302`). It subclasses `BaseException`
  (`cubepi/hitl/exceptions.py:4`) precisely so generic `except Exception`
  guards don't swallow a suspend. A tool body that prompts mid-execution
  (`HitlDetached`, `HitlAborted`, …) therefore detonates the batch.
- **`asyncio.CancelledError`** — `BaseException` since Python 3.8; not caught
  by `_execute_prepared`'s `except Exception` (`:303`).
- **Any `BaseException` raised by an `after_tool_call` hook** — `_finalize`
  guards the hook with `except Exception` only (`tools.py:353`), so a
  `BaseException`-shaped hook failure escapes with no handler anywhere in the
  stack.

Reproduced (3-tool parallel batch, one raising `HitlDetached`):

```
execute_tool_calls raised: HitlDetached
ToolResultMessages built: 0        # including the already-succeeded sibling
leaked still-running tasks: 1
side effects after grace: ['fast_ok done',
                           'slow_ok side effect COMMITTED after batch already failed']
```

The repro shows a second defect beyond the dropped results: the collection
loop never cancels or drains sibling tasks, so **still-running siblings leak**
— they keep executing detached, commit their side effects, and their results
are discarded. On resume the batch replays and duplicates those side effects.
This is exactly the hazard the two-phase prepare comment documents and
prevents for the *prepare* stage (`tools.py:539-547`); the *execute* stage has
no equivalent protection.

Downstream consequence: the assistant message carrying the `tool_calls` was
already checkpointed via `MessageEndEvent` (`agent.py:1194-1209`), but no
`tool_result` ever follows it. Every provider rejects such a transcript, so
the next turn — and every turn after — fails with a 400.

#### Existing partial defenses (and their gaps)

- **Two-phase prepare** (`tools.py:539-547`): prevents prepare-stage HITL from
  leaking started tasks. Does not cover a raise from inside a tool body or
  from `after_tool_call`.
- **Agent-layer cancel backfill** (`agent.py:1094-1101`,
  `_complete_cancelled_tool_calls`): on `CancelledError` synthesizes
  tool_results for every unanswered tool_call and re-raises. Covers the cancel
  path only, only for the stateful `Agent` (not `run_loop` driven directly),
  and does nothing about leaked sibling tasks or their lost results (the
  synthetic text claims cancellation even for tools that actually completed).
- **Sequential executor**: emits each `ToolResultMessage` as it goes
  (`tools.py:521-523`), so a mid-sequence raise persists earlier siblings.
  Later calls stay unanswered but the existing HITL resume/abort backfill
  paths answer them. Parallel has no such incremental persistence.

### 2. Checkpointer `load()` has no row-level fault handling

All three SQL backends deserialize message rows in a loop with no per-row
guard:

- sqlite: `cubepi/checkpointer/sqlite.py:171-173` (and the fork-source read at
  `:425`)
- postgres: `cubepi/checkpointer/postgres/checkpointer.py:172-183` (and
  `:396`)
- mysql: `cubepi/checkpointer/mysql/checkpointer.py:214-219` (and `:477-482`)

One corrupt payload (bad msgpack/JSON, failed `model_validate`, unknown role)
raises a raw `msgpack`/`ValueError`/`ValidationError` out of `load()` and the
entire thread becomes unloadable — the caller can't tell corruption from an
operational error, and can't tell *which* row is bad.

A related inconsistency: on an unknown role, postgres and mysql raise
`ValueError`, while sqlite's `_deserialize_message` silently returns the raw
dict (`sqlite.py:537`) — corruption flows straight into the message list as a
non-`Message` object and fails later, far from the cause.

## Goals

1. A failure in one tool call of a parallel batch must never destroy the
   results of its siblings, leak running sibling tasks, or persist dangling
   `tool_calls` — while preserving the HITL suspend/abort and cancellation
   contracts, which *require* certain exceptions to propagate.
2. Checkpoint corruption must surface as a typed, row-locatable error — never
   as a raw decoder exception, and never as silently-wrong data.

Non-goals: send-time transcript repair at the provider seam (hosts may keep
their own belt-and-suspenders); a skip-corrupt-rows load mode (see
Alternatives); multi-slot HITL within one batch (single-slot channel semantics
are owned by the answer-ledger spec).

## Design

### Part 1 — per-task fault isolation in `_execute_parallel`

The two-phase prepare stays exactly as is. Only the fan-out/collect stage
changes.

**Collect with `asyncio.gather(*tasks, return_exceptions=True)`** instead of
the bare `await` loop. Every task settles before any outcome is processed;
results stay in tool_call order. Then classify each slot:

1. **`_FinalizedOutcome`** — success path, unchanged.
2. **Framework failure** (any exception that is not `HitlControlException`,
   `CancelledError`, `KeyboardInterrupt`, or `SystemExit`) — with Part 2 in
   place, tool-body and hook failures are converted to error results *at the
   source*, inside the task. So an exception landing in the classification
   stage is framework-side — in practice `emit_fn` raising while processing
   a Start/End event. Emitter failures propagate at every other
   `emit_event` call site, and synthesizing a bogus `tool_result` for one
   would let the run continue with event processing broken; so it is
   recorded and **re-raised after every task has settled** (no leaks, no
   conversion). A framework failure takes precedence over a pending
   suspend: durably suspending a run whose event pipeline is failing is not
   safe.
3. **`HitlControlException`** — the suspend contract requires propagation, so
   it cannot become an error result. Order of operations: first emit the
   `ToolResultMessage`s for every *other* settled slot (each `MessageEndEvent`
   checkpoints immediately via `Agent._process_event`, `agent.py:1207-1209`),
   then re-raise the HITL exception. This emission is deliberately **not**
   best-effort: if a sibling's persistence fails, suspending anyway would
   durably record a batch whose completed work is missing (resume would
   re-run side-effecting tools), so the persistence failure propagates and
   the run fails rather than suspends. The emitted sibling results are also
   attached to the exception as
   `HitlControlException.partial_tool_results`; the stateless loop entry
   points (`_run_loop` / `run_agent_loop_resume` HITL handlers) append them
   to the message lists they return, so callers persisting the loop's
   return value — who never see the raised-through batch — keep the
   completed siblings' results. (The sequential executor attaches the same
   attribute for its already-emitted prefix.) The HITL call's own tool_call
   deliberately stays unanswered and gets no End event — identical to today's
   sequential detach shape — and the existing detach/resume/abort paths
   answer it (synthetic-deny backfill scans unanswered ids of the last
   assistant message). If several slots raise HITL exceptions, the first in
   batch order wins; the rest also stay unanswered and are covered by the
   same backfill.
4. **`CancelledError` in a slot, without outer cancellation** — a tool body
   raised it spuriously (tool bug). Treat as an ordinary failure (case 2)
   with text `[Tool execution cancelled]`. Escalating one buggy tool's
   self-cancel into run-level cancellation would kill sibling work for no
   reason. *(Flagged as a decision point — the conservative alternative is to
   re-raise after draining, like case 3.)* Note the inherent
   sequential/parallel asymmetry: in sequential mode tools run in the
   executor's own task, so a self-raised `CancelledError` is
   indistinguishable from a genuine external cancel and must keep
   propagating (the Agent-layer backfill covers it); only the parallel
   executor can tell the two apart (`task.cancelled()` vs an exception with
   no outer cancel).
5. **Outer cancellation** (the executor itself is cancelled while awaiting
   the gather): `gather` propagates the cancel to all children. Catch the
   `CancelledError`, await the tasks once more with
   `return_exceptions=True` so they fully settle, emit the
   `ToolResultMessage`s of slots that completed *before* the cancel landed,
   and re-raise. The Agent-layer `_complete_cancelled_tool_calls` backfill
   then covers only the genuinely-unanswered ids — fixing today's secondary
   bug where completed tools get a synthetic "[cancelled]" result and their
   real results (and side effects) are lost, causing duplicate execution on
   resume.
6. **`KeyboardInterrupt` / `SystemExit`** — settle children, re-raise. Never
   converted to results.

The `terminate` aggregation (`_should_terminate`) runs over real plus
synthesized outcomes, unchanged in shape.

### Part 2 — broaden the narrow catches at the source

- `_execute_prepared` (`tools.py:303`): `except Exception` becomes a
  `BaseException` guard that re-raises `HitlControlException`,
  `CancelledError`, `KeyboardInterrupt`, `SystemExit` and converts everything
  else to an error result.
- `_finalize`'s `after_tool_call` guard (`tools.py:353`): same treatment. A
  hook raising a `BaseException` subclass currently escapes *both* executors
  and every Agent-layer handler (it is neither `CancelledError` nor
  `Exception`, `agent.py:1094/1102`) — after this change it degrades to an
  error result on that one call.

The sequential executor gets the same per-call robustness for free, since it
shares both helpers.

### Part 3 — typed, row-locatable checkpoint corruption

New exception in `cubepi/checkpointer/exceptions.py`, following the
`CompletionMarkerFailedError` shape:

```python
class CheckpointCorruptionError(CheckpointerError):
    """A persisted message row failed to deserialize during load."""

    def __init__(self, *, thread_id: str, backend: str, row_ref: str,
                 cause: BaseException) -> None:
        super().__init__(
            f"corrupt checkpoint row for thread {thread_id!r} "
            f"({backend}, {row_ref}): {cause}")
        self.thread_id = thread_id
        self.backend = backend    # "sqlite" | "postgres" | "mysql"
        self.row_ref = row_ref    # e.g. "messages.id=42" — locates the row
        self.__cause__ = cause
        self.__suppress_context__ = True
```

Every row-deserialization site (the six listed in Problem §2) wraps its
per-row work — decode, role lookup, `model_validate` — in a guard that raises
`CheckpointCorruptionError` with the row's primary key. The SELECTs gain the
pk column where they don't already fetch it.

Unknown role becomes corruption too, in all three backends — including
sqlite, whose current silent raw-dict fallthrough (`sqlite.py:537`) is
replaced. **This is a deliberate behavior change**: any host that depended on
unknown-role dicts passing through sqlite load was already broken downstream;
failing loudly at the source with the row pk is strictly more debuggable.

`load()` still fails the whole thread on a corrupt row — but now with a typed
error that names the row, so hosts can catch `CheckpointCorruptionError`
distinctly from operational `CheckpointerError`s and repair or quarantine the
row surgically.

## Alternatives considered

- **`asyncio.TaskGroup` instead of `gather`** — TaskGroup cancels all siblings
  on first failure. That is the opposite of the requirement (siblings must
  finish and their results must persist), and ExceptionGroup unwrapping adds
  noise. `gather(return_exceptions=True)` has exactly the settle-everything
  semantics needed.
- **Convert HITL exceptions to error results** — breaks the suspend contract:
  detach/abort rely on propagation to reach `run_loop`'s outer handlers
  (`loop.py:445-458`) and the channel's pending-state cleanup.
- **Skip corrupt rows on load** (per-row tolerance in the recovery sense) —
  rejected for v1: silently dropping an assistant message that carries
  `tool_calls`, or a `tool_result`, manufactures the very dangling-tool_call
  transcript this spec exists to prevent. If a lossy mode is ever wanted it
  must synthesize structural repairs, not just skip — deferred until a
  concrete need appears; the typed error with `row_ref` is the enabling step
  either way.
- **Emit each parallel result as its task completes** (as-completed
  streaming, matching the sequential executor) — would also fix the loss but
  changes observable event ordering for hosts and interleaves checkpointer
  appends with running tools; settle-then-emit preserves today's ordering
  with the same durability outcome.

## Testing

- `tests/agent/test_tools.py` (parallel executor, using the existing
  `AgentTool` fixtures):
  - HITL raise mid-batch: siblings' `ToolResultMessage`s emitted and in
    order, `HitlDetached` re-raised, HITL call has no result and no End
    event, no still-running tasks after the raise.
  - `after_tool_call` raising a bare `BaseException`: converted to an error
    result for that call only; batch completes.
  - Tool body raising `CancelledError` without outer cancel: error result,
    siblings unaffected.
  - Outer cancellation mid-batch: completed slots' results emitted before the
    re-raise; unfinished slots have none (Agent backfill covers them —
    asserted at the Agent layer with the memory checkpointer).
  - Start/End event pairing holds for every synthesized outcome.
- Checkpointer suites (sqlite, postgres, mysql — reusing each backend's
  existing test harness): corrupt payload row → `CheckpointCorruptionError`
  carrying thread_id/backend/row_ref; unknown role → same, including the
  sqlite fallthrough replacement; fork-source read paths covered as well as
  `load()`.
- Regression: full existing suite (the repro script's scenario becomes a
  permanent test).
