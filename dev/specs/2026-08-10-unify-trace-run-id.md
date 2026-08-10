# Unify Trace `cubepi.run_id` with Business Run ID

- Date: 2026-08-10
- Status: Implemented (pending release)
- Repos: cubepi (primary), cubeplex (consumer; almost no code change)
- Related: `dev/specs/2026-05-18-cubepi-tracing-design.md` §10.1,
  `dev/specs/2026-08-04-tracing-hitl-suspension.md` (HITL: one business run,
  multiple `invoke_agent` activations)

## Goal

One **run** concept end-to-end.

The string that hosts put on `agent.prompt(run_id=...)`, that lands on
`Message.run_id` / `cubepi_messages.run_id`, and that products use for SSE,
cancel, steer, and billing **is the same string** stamped on every span as
`cubepi.run_id`.

There is no second, tracer-private "trace run id".

## Context

Today two UUIDs coexist for one agent activation:

| Layer | Who mints | Example |
|---|---|---|
| Business / agent | host (`uuid7`) or `prompt` fallback (`uuid4().hex`) | cubeplex stream path, message ledger |
| Tracing | recorder always `str(uuid.uuid4())` at `AgentStart` | span attr `cubepi.run_id`, stream log name |

That split was never a deliberate product rule that "business run ≠ trace run".
History:

1. **2026-05-18** tracing Phase 1 defined `cubepi.run_id` as "uuid generated at
   root open" so JSONL could shard by run.
2. **2026-05-19** PR #92 protected that attribute from being *clobbered by
   user metadata* (namespace → `cubepi.metadata.*`) — not a ban on adopting
   the agent run id.
3. **2026-06-06** agent gained `prompt(run_id=...)` / `active_run_id` for the
   message ledger, fork, HITL. Tracing was never updated to read it.
4. JSONL later **re-sharded by OTel `trace_id`**, so the original reason for a
   tracer-minted run id (file path) is already gone. Comments on the exporter
   still describe parent + subagent as same `trace_id`, different
   `cubepi.run_id`.

Symptom hosts hit in the wild: UI/API `run_id` cannot find the matching
`cubepi trace` file; investigators rediscover the run via `conversation_id`.

## Concepts (normative)

```
Business run_id  ──►  messages / SSE / billing / cubepi.run_id (span attr)
OTel trace_id    ──►  span tree storage & CLI grouping (may hold multiple runs)
OTel span_id     ──►  one span node
```

- **Run** = one agent activation identity (`prompt` / `resume` / `respond`
  effective id, or oneshot session id). Product and tracing share it.
- **Trace** = OTel distributed tree. Not a run. Nesting may put several runs
  under one `trace_id`.
- **Span** = one node. Root `invoke_agent` starts a subtree; filtering "all
  spans of this run" uses the denormalized `cubepi.run_id` attribute (or a
  tree walk from the root `span_id`). The attribute must be the **business**
  id, not a second random UUID.

## Non-goals

- Continuing one OTel `trace_id` across HITL pause and resume (already out of
  scope per HITL suspension spec). Correlation across activations is the
  **shared business `run_id`** (and host metadata).
- Changing JSONL sharding (stays `trace_id`).
- Changing how hosts mint run ids (cubeplex keeps `uuid7()`).
- Building a UI; only the id contract.
- Removing the `cubepi.run_id` attribute name (keep it; change who fills it).

## Design

### 1. Recorder: prefer `agent.active_run_id`

On `AgentStartEvent` → `_on_agent_start`:

```text
if attached agent has active_run_id set:
    run_id = that value
else:
    run_id = str(uuid.uuid4())   # fallback only
stamp CUBEPI_RUN_ID = run_id on root and all children (unchanged)
```

Ordering already allows this: `Agent.prompt` / `resume` / `respond` set
`active_run_id` **before** the loop emits `AgentStartEvent`. Recorder holds
`self._agent` from `attach`.

No change to the PR #92 rule: user `tracing_context(metadata=...)` still lands
under `cubepi.metadata.*` and **must not** overwrite `cubepi.run_id`. Hosts
pass the business id through `agent.prompt(run_id=...)`, not through metadata.

### 2. Oneshot: keep minting, same semantic

`Tracer.oneshot` has no agent. It continues to mint a run id for that session.
That id **is** the oneshot's run id (only activation id that exists), not a
parallel "trace run" concept. Optional later: accept `run_id=` on oneshot for
host correlation.

### 3. Nested subagents

Unchanged structure:

- Parent and child may share one OTel `trace_id` (existing nesting).
- Each agent's activation has its own business `run_id` (child `prompt` with a
  fresh id, or host policy).
- Spans of each activation carry **that** activation's id.

Do not force child spans to inherit the parent's business run id.

### 4. HITL pause / resume

One business `run_id` across pause and resume (already true on the agent).
Each activation still opens a new `invoke_agent` root / may be a new
`trace_id`. After this change both roots' spans carry the **same**
`cubepi.run_id`, so `cubepi trace` filter-by-run-id finds both halves.

### 5. Stream recording path

`record_stream` writes `<run_id>.stream.jsonl`. After the change the filename
uses the business id. Sanitize as today (`_safe_filename`). Collision risk is
the same as "two concurrent activations with the same run_id", which hosts
already must not do for the ledger.

### 6. Spec text update

In the tracing design §10.1, change:

| Attribute | Source (old) | Source (new) |
|---|---|---|
| `cubepi.run_id` | uuid generated at root open | agent `active_run_id` when set; else generated fallback |

### 7. cubeplex

Already passes `run_id` into `agent.prompt`. No required change for alignment.

Optional follow-ups (separate PRs, not required for the contract):

- Message action **Info** chip: document that the id is the cubepi/trace run id
  (already true once this lands).
- Admin traces filter: accept the same id.

## Change list (implementation)

### cubepi

| Area | Change |
|---|---|
| `cubepi/tracing/recorder.py` | `_on_agent_start`: resolve run id from `self._agent` active run; fallback uuid |
| `cubepi/tracing/tracer.py` | oneshot: document that minted id *is* the run id; optional `run_id=` later |
| `dev/specs/2026-05-18-cubepi-tracing-design.md` | §10.1 source line for `cubepi.run_id` |
| Tests | See below |
| Changelog | breaking-ish note for anyone who stored old tracer-minted ids |

Public API surface: no new functions required. Behavior change only.

### Tests (invariants)

1. **`prompt(run_id="host-1")` → all spans `cubepi.run_id == "host-1"`**  
   (root, turn, chat, tool if any).
2. **`prompt()` without run_id → spans share one non-empty id; equals returned
   run id from `prompt`.**  
   (Today `prompt` returns `effective_run_id`; tracing must match that, not a
   third uuid — so fallback should use the same source as the agent, not a
   second `uuid4()` in the recorder.)

   **Important refinement:** when the agent already chose `effective_run_id`
   (including self-minted `uuid.uuid4().hex`), the recorder must **read that**,
   not mint again. Fallback self-mint only if `active_run_id` is unexpectedly
   unset (defensive).
3. **Metadata cannot clobber** (existing PR #92 test still passes).
4. **Subagent:** parent and child different business ids → different
   `cubepi.run_id` on their respective span subtrees; same `trace_id` when
   nested.
5. **Sequential runs** on one attached agent: run A then run B → no id leak.
6. **Oneshot:** still produces a non-empty `cubepi.run_id` on root + chat.

### Migration / compatibility

- Old JSONL files keep historical tracer-minted ids; no migration.
- Operators who bookmarked old `cubepi.run_id` values for in-flight runs will
  not match new spans; only new activations align.
- CLI filters (`cubepi trace ls` by run id attribute) start matching host ids
  after upgrade — that is the intended win.

## Success criteria

1. Given cubeplex (or any host) `run_id=R` on `prompt`, every span of that
   activation has `cubepi.run_id=R`.
2. `cubepi trace` lookup by the id shown in the product UI finds that run's
   spans (for agent activations; oneshot still uses its own id).
3. No second uuid is minted by the recorder when `active_run_id` is set.
4. JSONL layout, OTel parentage, and metadata namespacing remain unchanged.
5. Existing "metadata cannot clobber reserved attrs" tests stay green.

## Open points (non-blocking)

- Expose `run_id=` on `Tracer.oneshot` for host-supplied correlation of
  memory/background jobs.
- Whether `AgentStartEvent` should carry `run_id` explicitly (nice for pure
  observers; not required if recorder reads `active_run_id`).
- Admin UI deep-link from message Info chip → traces filtered by run id
  (cubeplex product; after this lands).

## Decision needed before implement

None for the core path: **one run concept; recorder adopts `active_run_id`.**

Approve this draft → implement in cubepi (tests first) → bump cubeplex's
cubepi pin when released.
