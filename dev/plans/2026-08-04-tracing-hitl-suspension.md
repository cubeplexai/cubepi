# HITL Suspension Trace Semantics Implementation Plan

**Goal:** Export durable HITL pauses as committed `suspended` activations,
never as aborted, while preserving cancellation behavior and tracing cleanup.

**Architecture:** Snapshot the pending request in `Agent.detach()`, commit the
`HitlDetached` channel transition, and publish `AgentSuspendedEvent` from the
owning run task after outcome dispatch and active-run cleanup. The recorder then
finalizes the activation with explicit suspended attributes.

**Tech stack:** Python 3.13, asyncio, FauxProvider, CheckpointedChannel,
OpenTelemetry SDK, pytest, Ruff, mypy, uv.

## Task 1: Pin the production-shaped regression

- Add `tests/tracing/test_hitl_suspension.py` using a real Agent, HITL channel,
  ask-user tool, Tracer, and in-memory exporter.
- Record RED for the original false-abort behavior.
- Add adversarial RED assertions for:
  - event observers seeing pre-commit runtime state;
  - cross-task `_active_run` retention;
  - assistant output count incorrectly remaining zero;
  - an earlier regular/cancelled listener hiding suspension from tracing;
  - detach followed by immediate owner-task cancellation retaining the pending
    payload and emitting an unhandled-future warning;
  - same-tick `reset()` deleting a committed suspension snapshot before the
    owner task can publish it.

## Task 2: Commit suspension at the owning task boundary

- Snapshot the pending request in `Agent.detach()` before the channel clears its
  in-memory slot.
- Set `HitlDetached` without publishing the terminal event from the host task.
- After `prompt()`, `resume()`, or `respond()` records `outcome=suspended`,
  performs outcome dispatch, and clears `active_run_id`, publish the event from
  that same owning task.
- Fan out agent events across regular exceptions and `CancelledError`; after
  fan-out, give cancellation priority or re-raise the first regular exception.
- Clear the pending snapshot on every failed owning activation; do not let
  `reset()` clear a snapshot before that activation publishes its terminal.
- Mark the detach control exception retrieved via a future done callback without
  changing what channel awaiters receive.

## Task 3: Add explicit trace semantics

- Add the recorder-owned schema attribute `cubepi.run.outcome`.
- Handle committed `AgentSuspendedEvent` in `Recorder`.
- End open tool/chat/turn/root spans with outcome `suspended`.
- Sweep tool-span registrations, close stream state, and clear the active run.
- Move assistant output accumulation to `MessageEndEvent` so suspension before
  `TurnEndEvent` still records the partial output.
- Do not change `_close_open_spans()` cancellation semantics; make one-shot
  cancellation match the same aborted-without-exception-event contract.
- When `record_content=False`, preserve typed ERROR classification while using
  generic status descriptions and omitting exception messages/stack traces from
  provider, turn, one-shot, and MCP spans; keep stream telemetry structural by
  omitting raw argument/error previews.

## Task 4: Document the lifecycle contract

- Update tracing guidance to distinguish cancellation from durable suspension.
- Update HITL reference/durable examples to describe post-commit event timing.
- State that resume creates a new activation trace.

## Task 5: Verify

Run:

```bash
uv run pytest tests/tracing/test_hitl_suspension.py -q
uv run pytest tests/tracing tests/hitl -q
uv run pytest tests/
uv run ruff check cubepi/ tests/
uv run ruff format --check cubepi/ tests/
uv run mypy cubepi
```

Inspect `git diff --check` and confirm the branch contains only the spec, plan,
focused tests, lifecycle fix, recorder/schema fix, and user-facing docs.
