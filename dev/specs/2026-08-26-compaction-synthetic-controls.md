# Trailing Synthetic Controls Must Not Age Current-Turn Evidence

- **Date**: 2026-08-26
- **Status**: Approved for implementation
- **Branch / worktree**: `2026-08-26-compaction-synthetic-controls` → `.worktrees/2026-08-26-compaction-synthetic-controls`
- **Related**:
  - [`2026-06-26-run-scoped-compaction.md`](2026-06-26-run-scoped-compaction.md)
  - [`../../website/docs/guides/middleware/compaction.md`](../../website/docs/guides/middleware/compaction.md)

## 1. Motivation

`CompactionMiddleware` receives the message view produced by middleware earlier in
the transform chain. Those earlier transforms may append synthetic `UserMessage`
controls for the next model call: policy envelopes, reminders, continuation nudges,
or other framework instructions built with `synthetic_user_message(...)`.

Today compaction treats those call-local controls as ordinary conversation history
when it chooses the protected tail and a safe summary boundary. That can make tool
evidence produced immediately before the controls look old.

A production-equivalent Bot run exposed the failure with this shape:

```text
User(request)
Assistant(tool_call A, tool_call B)
ToolResult A                          # 106,277 chars
ToolResult B                          #  38,796 chars
Synthetic User(freshness control)    # small, appended by transform_context
```

With `max_tokens_before_compact=55_000`, `keep_tail_tokens=8_000`, and
`min_compact_messages=4`:

1. the synthetic control alone becomes the protected tail;
2. `safe_boundary` accepts its index as a self-contained boundary;
3. both current-turn `ToolResultMessage`s fall before `tail_start`;
4. the default pruner replaces them with `[tool] N chars` markers;
5. the synthesis model sees that tools ran, but not the rows it must synthesize.

The model then correctly reports that row-level evidence is unavailable even though
the durable run transcript still contains the full results. This is not a tool-specific
or application-specific failure: any transform middleware that appends a synthetic
user-role control after current evidence can create the same false aging.

## 2. Required invariant

A **maximal trailing suffix of synthetic `UserMessage`s** is call-local control
context for compaction purposes:

- it remains visible to the main model;
- it remains in exact input order;
- it contributes to the total context-size trigger;
- it does **not** consume the protected historical tail;
- it does **not** create a compaction boundary;
- it is not summarized or pruned;
- compaction state refs and boundaries continue to describe the ordinary message
  prefix, not transient transform additions.

Only a trailing suffix is special. Synthetic user messages inside ordinary history
remain ordinary history. Only `UserMessage` controls are detached; synthetic
`ToolResultMessage`s stay with the history so no tool call/result pair can be
orphaned.

## 3. Design

### 3.1 Split the transformed view into history and trailing controls

At the start of `CompactionMiddleware.transform_context`, split the input at the
first message of the maximal trailing suffix satisfying both:

```python
isinstance(message, UserMessage) and is_synthetic_message(message)
```

The two views are:

- `history`: every message before that suffix;
- `trailing_controls`: the suffix, unchanged and ordered.

When no such suffix exists, `history` is the full input and
`trailing_controls` is empty, preserving current behavior.

### 3.2 State and boundary decisions use history only

The following operations use `history`, not the combined transformed view:

- persisted-state boundary range validation;
- `_state_matches_history`;
- `tail_start_by_tokens`;
- `safe_boundary`;
- `prune_tool_results`;
- summarizer slices and reference messages;
- preserved-result indexing and state refs;
- `_compressed_view` construction.

Because the detached controls form only a trailing suffix, every index in `history`
is also the same index in the original input. Existing `compaction_until_msg_index`
values and message refs therefore keep their meaning; no state-schema migration is
needed.

A synthetic user control that is durably persisted may be trailing for one call and
later become internal history after an assistant response follows it. This remains
safe: while trailing it cannot define the boundary; once internal it participates in
history normally, and every earlier boundary index remains stable.

### 3.3 Threshold decisions include controls

The main model receives the compressed/uncompressed history plus the controls, so
the threshold must measure that exact view:

```text
real_context_estimate(_compressed_view(history, state, boundary) + controls)
```

This prevents a large control suffix from bypassing compaction. The anti-thrash
emergency check and savings accounting continue to use the full input/output views,
so controls contribute to actual send size without affecting tail placement.

### 3.4 Reattach controls after every exit path

Every returned model view has this order:

```text
compressed_or_original_history + trailing_controls
```

This applies to:

- below-threshold returns;
- no-safe-boundary returns;
- anti-thrash skips;
- successful LLM summaries;
- deterministic fallback summaries.

Reattachment does not mutate the original messages or the control objects.

## 4. Why this boundary is generic and low-maintenance

The repair consumes CubePi's existing semantic marker,
`metadata["synthetic"] is True`; it introduces no source-name allowlist, middleware
registry, tool-family table, payload inspection, or application-specific exception.
Any current or future middleware using `synthetic_user_message(...)` receives the
same behavior automatically.

The restriction to a **maximal trailing suffix** is deliberate:

- trailing controls are the additions that can falsely displace current evidence;
- internal synthetic messages may be durable control history and can be summarized
  once later turns follow them;
- detaching arbitrary synthetic messages would require index remapping and would
  destabilize compaction refs;
- detaching synthetic tool results could orphan their tool calls.

## 5. Prior-art check and CubePi divergence

- **LangGraph** distinguishes a temporary `llm_input_messages` view from updates to
  persistent `messages` state. That supports the same architectural separation:
  model-call context transforms should not automatically redefine durable-history
  semantics.
- **pi agent context construction** converts compaction and branch-summary entries
  into synthetic context messages during context building, rather than treating all
  generated context as equivalent to persisted dialogue.
- **Claude Code** reconstructs post-compaction context in a fixed order — summary,
  explicitly retained messages, attachments, and hook results — instead of letting
  post-processing additions determine the retained conversation boundary.

CubePi differs because its middleware API intentionally uses one ordinary
`list[Message]` rather than introducing a second context-entry hierarchy. The
smallest compatible repair is therefore to consume the existing synthetic marker at
the compaction seam, not add a new message type or public API.

References:

- https://reference.langchain.com/python/langgraph.prebuilt/chat_agent_executor/create_react_agent
- https://github.com/earendil-works/pi/blob/main/packages/agent/src/harness/session/context.ts
- `/Users/test/work/claude-code/src/services/compact/compact.ts:325-337`

## 6. Non-goals

- No Bot-, finance-, freshness-, report-, or tool-specific preservation rule.
- No change to `tool_result_compressor` semantics.
- No automatic preservation of every large tool result.
- No new public middleware configuration or message metadata field.
- No change to provider serialization.
- No change to `safe_boundary`'s tool-call self-containment rules.
- No detachment of synthetic assistant or tool-result messages.
- No compaction-state schema/version migration.

## 7. Compatibility

This is a behavioral bug fix with no public API change.

- Inputs without a trailing synthetic-user suffix follow the same code path and must
  remain byte-for-byte/equality equivalent at the message-model level.
- Inputs with such a suffix may choose a different tail/boundary, preserving newer
  ordinary evidence that was previously pruned.
- The controls still count toward the threshold, so the repair cannot silently send
  an over-limit context merely by labeling content synthetic.
- Existing persisted boundaries remain valid because suffix removal never changes
  prefix indices.

## 8. Test plan

### Required RED

Drive the real `CompactionMiddleware.transform_context` with the production-shaped
five-message input above and the production threshold/tail/minimum values. Before the
fix, both large result sentinels disappear into size-only markers. After the fix, no
safe history-only boundary exists, so the original current-turn results and trailing
control all remain intact.

### Regression matrix

1. No trailing synthetic suffix: existing compaction behavior is unchanged.
2. One trailing synthetic user control: current tool evidence remains protected.
3. Multiple trailing synthetic user controls: all survive in exact order.
4. Controls count toward the threshold: history compacts when the combined send view,
   but not history alone, crosses the threshold.
5. A synthetic `ToolResultMessage` at the tail is not detached and remains paired with
   its tool call.
6. With older history before the current tool turn, the old prefix can compact while
   all current-turn tool results remain raw.
7. Persisted state refs/boundary remain valid across repeated calls and when a formerly
   trailing synthetic user message later becomes internal history.
8. Existing multi-round, pruner, circuit-breaker, anti-thrash, and run-scoped tests
   remain green.

## 9. Rollout

1. Land the generic fix and docs in CubePi after focused/full tests, lint, format, and
   type checks plus independent spec and code-quality review.
2. Establish the upstream commit/release path.
3. Update the Cubemanus dependency pin only after the upstream artifact is available.
4. Add a Bot lifecycle regression using its real Temporal Freshness transform before
   compaction.
5. Rebuild and deploy the exact Cubemanus `dgts` SHA to TEST, then rerun the demanding
   report synthesis E2E.

Rollback is a dependency-pin revert. No persisted state migration or data repair is
required.
