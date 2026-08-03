# HITL Suspension Trace Semantics

- Date: 2026-08-04
- Status: Accepted
- Companion plan: `dev/plans/2026-08-04-tracing-hitl-suspension.md`

## Problem

A durable HITL pause is a successful activation outcome, not a cancellation.
The agent loop records `RunOutcome="suspended"`, but the tracing recorder did
not have a committed suspension terminal. It therefore kept the root, turn,
and active tool spans open until tracing detach, whose cancellation cleanup
marked them `cubepi.aborted=true`.

A first implementation consumed the existing `AgentSuspendedEvent` directly,
but adversarial review found that event was emitted before `Agent.detach()` set
`HitlDetached` on the channel future. A later listener could raise, prevent the
future transition, and leave the prompt running while tracing had already
exported a false suspended terminal. The handler also ran from the host detach
task, so resetting a `ContextVar` token created by the prompt task failed and
left the old `_RunState` retained in that task's context.

## Goals

- Finalize every open span only after suspension is committed.
- Mark the activation with `cubepi.run.outcome="suspended"`.
- Never add `cubepi.aborted=true` or `error.type=cubepi.aborted` to a suspended
  activation.
- Emit the terminal event from the owning prompt/resume task so recorder
  context cleanup occurs in the token's original context.
- Ensure one failing or cancelled observer cannot hide a committed event from
  later observers; preserve exception propagation after fan-out and give
  cancellation priority.
- Clear the pending-request snapshot on every owning activation failure/exit and
  avoid unhandled `HitlDetached` future warnings.
- Preserve run cancellation cleanup and classification; align one-shot
  cancellation with Agent/MCP semantics by recording aborted control flow
  without an exception event.
- Remove tool-span and MCP-provider registrations during normal tracing detach.
- Count the assistant tool-call message as partial activation output.
- Keep tracing observational: recorder failures must not affect the agent.
- Keep raw provider, turn, one-shot, MCP, and stream-log diagnostics behind
  `record_content=True`; privacy-default traces retain only typed error
  classification and structural stream timing/size evidence.

## Non-goals

- Changing HITL persistence, answer, or resume behavior.
- Continuing one OTel trace across pause and resume. A resumed activation gets a
  new `invoke_agent` root and can be correlated by host-supplied metadata.
- Recording pending questions, answers, tool arguments, or other content.
- Reclassifying provider errors or explicit cancellation.

## Design

### Commit suspension before publishing it

`Agent.detach()` snapshots the real pending request, then sets
`HitlDetached` on the channel future. It does not publish the terminal event.
The owning `prompt()`, `resume()`, or `respond()` task lets the loop record
`RunOutcome="suspended"`, performs outcome dispatch, clears `active_run_id`, and
only then emits `AgentSuspendedEvent` using the snapshot.

Because the event now runs in the task that received `AgentStartEvent`, the
recorder can reset its `_active_run` token in the owning context. If an event
listener raises, the committed runtime state remains suspended rather than
pending.

Agent event fan-out invokes every registered listener before propagating
listener failures. If any listener raises `asyncio.CancelledError`, cancellation
wins after fan-out; otherwise the first regular exception is re-raised. A user
listener registered before tracing can no longer hide the committed suspension
from the recorder.

The pending snapshot is cleared from every `prompt()`, `resume()`, and
`respond()` failure path. `reset()` deliberately does not touch a snapshot owned
by an activation that has not yet published its committed terminal. The detach future registers a
done callback that marks its control exception retrieved without changing what
awaiters receive, preventing an immediate owner-task cancellation from leaving
an unhandled-future warning.

### Finalize the trace as suspended

On the committed `AgentSuspendedEvent`, the recorder:

1. ends open tool, chat, and turn spans with
   `cubepi.run.outcome="suspended"`;
2. ends the root span with the same outcome and the partial output-message count;
3. sweeps any open tool-span registrations;
4. closes an active stream file;
5. releases recorder state so tracing detach is idempotent.

No span receives error status or aborted attributes. The existing generic
`_close_open_spans()` path remains cancellation-only.

Assistant messages are added to `run.output_messages` at `MessageEndEvent`, not
at `TurnEndEvent`. That keeps normal output unchanged while making the assistant
tool-call visible when HITL suspension prevents `TurnEndEvent`.

This aligns trace semantics with Cubepi's existing runtime outcome model.
Durable agent runtimes similarly treat an interrupt/pause as resumable control
flow rather than an execution error; Cubepi preserves that distinction without
adopting a graph runtime or changing its public event payload.

## Acceptance

A real `Agent` using `FauxProvider`, `CheckpointedChannel`, `ask_user_tool`, and
`Tracer` must demonstrate:

- one exported `invoke_agent` root;
- `cubepi.run.outcome="suspended"` on the root and open child spans;
- `cubepi.output.messages.count=1` for the assistant tool-call;
- zero exported spans with `cubepi.aborted=true`;
- no leaked MCP provider or active tool-span registrations after tracing detach;
- `_active_run.get()` is clear when the owning prompt task returns;
- a listener sees `last_outcome="suspended"` and `active_run_id=None`, never the
  pre-commit state;
- tracing still receives the event when an earlier listener raises a regular
  exception or `CancelledError`;
- listener cancellation remains the propagated outcome after terminal fan-out;
- detach followed by immediate owner-task cancellation clears the request
  snapshot, exports a real aborted trace, and emits no unhandled-future warning;
- `reset()` in the same tick after detach cannot steal the owning activation's
  snapshot; the persisted pending remains and the trace is suspended;
- unchanged passing cancellation, tracing, and HITL suites.
