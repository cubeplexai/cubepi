# Run-scoped Compaction, Real-token Triggering, and an Env-agnostic Tool-result Seam

- **Date**: 2026-06-26
- **Status**: Draft
- **Branch / worktree**: `2026-06-26-run-scoped-compaction` → `.worktrees/2026-06-26-run-scoped-compaction`
- **Related**: [`2026-06-08-compaction-improvements.md`](2026-06-08-compaction-improvements.md) (the
  pruner / circuit-breaker / anti-thrash machinery this spec builds on)

## 1. Motivation

`CompactionMiddleware` already summarises old history to keep long agents within
context. Its `transform_context` hook fires before **every** model call
(`loop.py:711`, inside the inner `while has_more_tool_calls` loop at
`loop.py:497`), so the *hook position* is already in the right place to compact
mid-run. Three real gaps remain:

| # | Gap | Impact |
|---|-----|--------|
| 1 | `safe_boundary` only cuts at a `UserMessage` (`boundary.py:64`) | A single long agentic **run** has no user messages in its tool-call chain, so the summary boundary is pinned to the run's opening prompt. The unbounded tool_use/tool_result chain inside the run is never compressed — compaction is effectively *between-run only*. |
| 2 | Trigger uses a char estimate calibrated with `usage.input_tokens` only (`tokens.py:40,43`) | Across **all** cubepi providers `input_tokens` is the *uncached* prompt portion; under prompt caching (the agent norm) most of the prompt is `cache_read` and never appears in `input_tokens`. The calibration silently degrades to a flat 1.25× multiplier and the trigger undercounts real context fill. |
| 3 | Oversized tool results have no bound, and cubepi has nowhere to put spilled content | A single huge tool result (the most-recent message, which the model must read this turn) cannot be compacted away — by definition it has to be sent. cubepi is **environment-agnostic**: it must not assume a filesystem / session dir / blob store, so it cannot implement claude-code's persist-to-disk strategy in core. |

This spec **implements (1) and (2)** — small, surgical changes inside the
compaction package. **(3) ships no code**: the seam already exists
(`after_tool_call`); this spec only documents that it is the supported mechanism
and records *why* cubepi can't do more in core.

---

## 2. Design

### 2.1 Run-scoped compaction — let the boundary advance within a run

**Current behaviour** (`boundary.py:63-72`):

```python
while candidate > 0:
    if not isinstance(messages[candidate], UserMessage):   # ← the restriction
        candidate -= 1
        continue
    if not _suffix_is_self_contained(messages[candidate:]):
        candidate -= 1
        continue
    if candidate < min_compact:
        return None
    return candidate
```

A run's message shape is `User(open) → Assistant(tool_use) → ToolResult →
Assistant(tool_use) → ToolResult → …` with **no `UserMessage` in between**. So
the only candidate is the run's opening prompt; the growing tail is never cut.

**Fix**: drop the `isinstance(..., UserMessage)` gate entirely. The boundary then
lands on the first self-contained suffix start walking back from `tail_start` —
which is a turn-boundary `AssistantMessage` when no user message is nearby.

```python
while candidate > 0:
    if not _suffix_is_self_contained(messages[candidate:]):
        candidate -= 1
        continue
    if candidate < min_compact:
        return None
    return candidate
```

**Why this is correct (and why we don't need the UserMessage anchor).**
The only invariant the provider requires after compaction is that
`[summary_user, *messages[boundary:]]` has no orphaned tool_use/tool_result.
`_suffix_is_self_contained` (`boundary.py:77`) already guarantees this:

- Candidate is a `ToolResultMessage` → suffix starts with a tool_result whose id
  is not yet in `available_call_ids` → rejected. ✅ (so ToolResultMessage can
  never be a cut point — the "add tool-result as candidate" idea is subsumed and
  correctly excluded.)
- Candidate splits a multi-tool turn between `ToolResult A | ToolResult B` →
  suffix starts with the orphan `ToolResult B` → rejected. ✅
- Candidate is a turn-start `AssistantMessage` (all its tool_use results follow
  it) → self-contained → accepted. ✅
- "tool_use in suffix, its result in prefix" is impossible: a result always
  follows its tool_use in message order, so `tool_use_index ≥ boundary ⇒
  result_index ≥ boundary`. ✅

Therefore the `UserMessage` check was a *semantic* anchor (summaries end at a
conversational turn), never a correctness requirement. Removing it also gives a
boundary **closer to the tail** (the first self-contained point, not the latest
user turn) → larger summarised region → more aggressive in-run compaction, which
is exactly the goal.

**Side effects / required follow-ups:**

- **Docstring** (`boundary.py:42-53`) currently says "searches the prefix for the
  latest `UserMessage`" — reword to "latest self-contained turn boundary".
- **Semantic shift**: the summary may now end mid-run (after a tool turn) rather
  than at a user turn. For agentic runs this reads naturally (`[summary_user,
  assistant_continues_acting, …]`). Confirm `summarize`'s handoff prompt
  (`summarizer.py`) produces coherent text for a half-run tool chain.
- **Residual case (out of scope here, see §2.3)**: if a *single* turn is itself
  larger than the context budget (100 parallel tool calls, or one giant tool
  result), it is atomic and cannot be split by any boundary. That is a
  tool-output-bounding problem, not a boundary problem.

### 2.2 Trigger on real tokens; remove the calibration

**Current behaviour.** The trigger compares `approx_tokens(unpruned_compressed)`
(char-based, `chars/2`) against the configured threshold (`__init__.py:210-211`).
`usage` is consulted only to compute a calibration `scale_factor`, and only from
`usage.input_tokens` (`tokens.py:37-46`), clamped to `[1.0, 1.25]`.

**Problem.** cubepi normalises `Usage.input_tokens` to the **uncached** prompt
portion on every provider:

- Anthropic (`anthropic.py:790-795`): `input_tokens` is natively cache-excluded;
  `cache_read_tokens = cache_read_input_tokens`, `cache_write_tokens =
  cache_creation_input_tokens`.
- OpenAI (`openai.py:258-265`): `input_tokens = max(prompt_tokens -
  cached_tokens, 0)`.
- OpenAI Responses (`openai_responses.py:539`), Faux (`faux.py:253`): same.

So the real prompt fill is `input_tokens + cache_read_tokens +
cache_write_tokens`. Using `input_tokens` alone undercounts massively whenever
caching is on, and the `[1.0, 1.25]` clamp masks the bug by pinning the factor at
1.25.

**Decision: switch the trigger to the real token count and delete the calibration
entirely** (path *b* from the design discussion, not "fix the calibration to use
all three").

Rationale — calibration's *only* job is to make an **absolute** char estimate
match reality for comparison against an **absolute** threshold. Once the trigger
uses the real number, every other use of `approx_tokens` is a **relative**
decision (where to cut, what to prune, the summary budget) where a consistent
estimator bias cancels. A bounded "nudge up 0–25%" heuristic with a known
scope-mismatch (numerator counts the assistant's own output; the real
`input_tokens` does not, but does include system prompt + tool defs) adds nothing
there and is a liability to maintain.

**New trigger:**

```
real_last = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
            # from the most recent AssistantMessage that carries usage
estimate  = real_last + approx_tokens(messages_appended_since_that_assistant)
compact if estimate >= threshold
```

- `messages_appended_since` is a small delta (the assistant + its tool results);
  over-estimating it only triggers compaction slightly early — safe.
- **Cold start**: the first `transform_context` has no assistant/usage yet → fall
  back to `approx_tokens` over the whole view. So `approx_tokens` stays, but
  *uncalibrated*: delete the `scale_factor` block (`tokens.py:24,37-46,53-54`),
  keep the plain `chars/2` path for relative sizing and cold start.

**Threshold sourcing (smaller, optional).** `max_tokens_before_compact` remains a
constructor arg, but we can default it from the bound model when omitted:
`threshold = int(ratio * model.context_window)` (`Model.context_window`,
`providers/base.py:91`; default ratio TBD, e.g. 0.75). This removes the "swap
model, forget to retune the threshold" footgun. Marked optional — confirm in
review whether to bind it now or keep it explicit.

### 2.3 Env-agnostic tool-result seam

**Why this can't live in cubepi core.** claude-code bounds tool output with a
persist-to-disk + 2000-byte-preview strategy (`toolResultStorage.ts`), and
Read throws-and-paginates. Both assume an environment: a session directory, a
readable filesystem, a `Read` tool to fetch the spilled file. **cubepi is
environment-agnostic** — core must not assume a filesystem, a session dir, or a
blob store. *Where* an oversized result goes (disk, object storage, a database
row, dropped) is an upper-layer decision.

**Divergence call-out (per CLAUDE.md):** claude-code does X (persist to a session
dir, hand back a path the model can `Read`); cubepi does Y (detect oversize in a
middleware, delegate the sink to the application) because Z (cubepi has no
environment to persist into and must stay dependency-/env-lean).

**The seam already exists.** `Middleware.after_tool_call` (`base.py:59`) receives
`AfterToolCallContext` (`types.py:91-97`: `tool_call`, `args`, `result`,
`is_error`, `context`) and may return `AfterToolCallResult` (`types.py:75-79`:
`content`, `details`, `is_error`, `terminate`). An application can already
implement an `after_tool_call` middleware that:

1. measures the result's text size,
2. ships the full bytes wherever it wants (its environment's choice),
3. returns an `AfterToolCallResult` whose `content` is a preview + a reference
   the app understands.

cubepi never sees the storage. This satisfies "reserve a mechanism for the upper
layer to do the handling itself."

**Scope decision: ship no code for this.** We do *not* add a `ToolResultGuard`
middleware or a `ToolResultSink` protocol. The existing `after_tool_call` seam is
sufficient and any helper would have to bake in a preview/size policy that is
itself application-specific. This spec's only deliverable here is **documentation**:
a short note in the compaction / middleware docs stating that bounding oversized
tool results is an `after_tool_call` middleware responsibility owned by the
application, with the env-agnostic rationale above. No core change, no new types.

This is orthogonal to §2.1/§2.2: §2.1 handles the common case (many
normal-sized turns); the pathological single giant result that compaction
provably cannot fix is the application's job via the documented seam.

---

## 3. Non-goals

- Mid-tool-pair cutting (splitting inside a tool_use/tool_result pair). Rejected
  as risky; §2.1's turn-boundary cutting is sufficient and provably safe.
- Any new tool-result middleware/type (`ToolResultGuard` / `ToolResultSink`).
  Core stays env-agnostic; the existing `after_tool_call` seam is the mechanism
  and we only document it.
- Post-compaction context re-injection (re-reading touched files) — already ruled
  out in `2026-06-08-compaction-improvements.md` §1 as application-specific.
- Dynamic per-tool caps computed from context-window size. claude-code uses fixed
  per-tool constants + env/flag overrides; we follow that — the cap is the app's
  policy, not context-relative math.

## 4. Open questions (spec stage — to confirm with the user)

1. **Threshold auto-binding (§2.2)**: default `max_tokens_before_compact` from
   `model.context_window * ratio` now, or keep it explicit? If now, what ratio?
2. **Anti-thrash tuning (§2.1)**: with the boundary advancing ~2 messages/turn,
   `_ANTI_THRASH_NEW_MSGS = 8` (`__init__.py:50`) gates how often in-run
   compaction fires. Keep 8, or retune for the tighter in-run cadence? (Leaning
   keep — verify via tests rather than guess.)

## 5. Test plan (FauxProvider)

- **2.1** — A run with a pure tool-call chain (no intermediate UserMessage)
  compacts at a turn boundary; the cut never orphans a tool_use/tool_result;
  multi-tool-per-turn is never split between results; `ctx.extra["compaction"]`
  state accumulates correctly across in-run turns; `_state_matches_history`
  (`__init__.py:118`) still validates after an in-run cut.
- **2.2** — With `FauxProvider` reporting cache_read/cache_write usage
  (`faux.py:262-265`), the trigger fires at the real combined token count, not the
  uncached `input_tokens`; cold-start (no usage yet) falls back to the estimate;
  `scale_factor` is gone and relative decisions (tail, prune, budget) are
  unaffected.
- **2.3** — No code, no test. Documentation only (the `after_tool_call` seam).

## 6. Rollout / compatibility

- §2.1 changes default boundary placement — existing callers get *more* in-run
  compaction automatically. Update any test asserting a UserMessage boundary.
- §2.2 removes `scale_factor`; behaviour-visible only when caching is active
  (compaction now triggers nearer the true limit). No API change.
- §2.3 ships no code — docs only; no behavioural or API change.
