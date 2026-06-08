# Compaction Improvements

- **Date**: 2026-06-08
- **Status**: Draft
- **Branch / worktree**: TBD

## 1. Motivation

`CompactionMiddleware` summarises older turns to keep long-running agents within
their context window. It works, but a comparison against claude-code and
hermes-agent surfaced seven gaps that degrade quality or reliability:

| # | Gap | Impact |
|---|-----|--------|
| 1 | Tool call arguments dropped in transcript | Summariser can't retain file paths, commands, query strings |
| 2 | No tool-result pre-pruning pass | Large tool outputs sent verbatim to summariser LLM; wasted cost |
| 3 | Tail protection is count-based (`keep_recent=8`) | 8 messages may be 500 or 80 000 tokens — unpredictable |
| 4 | `max_summary_tokens` is a fixed 1024 | Too small for long conversations; summary truncates critical facts |
| 5 | No circuit breaker on summariser failures | Failing summariser retries every turn indefinitely |
| 6 | No anti-thrashing guard | Near-threshold agents compact every turn, saving almost nothing |
| 7 | No fallback when LLM summariser is unavailable | Degradation leaves context uncompressed; next turn still over-limit |

Post-compact context re-injection (re-reading recently touched files after
compaction, as claude-code does) is **out of scope** for this spec — it requires
knowledge of which files an agent touched, which is application-specific.

---

## 2. Design

### 2.1 Tool call arguments in the summariser transcript

**Current behaviour** (`summarizer.py:_format_message_for_summary`):

```python
elif isinstance(block, ToolCall):
    parts.append(f"[tool_call:{block.name}]")   # arguments silently dropped
```

`read_file(path="/home/user/config.py")` becomes `[tool_call:read_file]`.
The summariser loses every file path, command, and query string — the exact
details that need to survive compaction.

**Fix**: include arguments, but per-field truncate long string values so a
`write_file` with 50 KB content doesn't dominate the transcript.

```python
# target representation:
[tool_call:read_file] {"path": "/home/user/config.py"}
[tool_call:bash] {"command": "npm test"}
[tool_call:write_file] {"path": "out.py", "content": "def f():\n    ...[truncated]"}
```

Per-field truncation: parse arguments JSON; for each string value, keep the
first `_ARG_VALUE_CHARS = 200` characters if longer; re-serialise. Non-string
leaves (int, bool, null) are preserved intact. If arguments is not valid JSON
(some backends send raw strings), fall back to a plain-string head truncation.
Result must NOT exceed `_ARG_REPR_MAX = 500` characters total.

This is the same approach used by hermes-agent's
`_truncate_tool_call_args_json`, adapted for CubePi's typed message model.

### 2.2 Tool-result pre-pruning pass (cheap, no LLM call)

Before calling the summariser, replace the text content of old
`ToolResultMessage` instances with a single-line summary. This is a read-only
scan that never touches the last `keep_tail_messages` messages (same tail
protected from summarisation). It runs on the raw history, not the compressed
view, so very large tool outputs are collapsed before the transcript is built.

**Replacement format:**

```
[{tool_name}] {short description of what happened}
```

Examples:
- `[bash] exit 0, 142 lines`
- `[read_file] 3 400 chars`
- `[web_search] 5 results`

Rules:
- Preserve the last `keep_tail_messages` results intact (same guard as
  compaction boundary).
- If a result's content is already ≤ `_PRUNE_KEEP_CHARS = 120` characters,
  keep it as-is.
- The replacement is a `TextContent` with the one-liner; all other content
  blocks in the result (e.g. images) are dropped.
- Tool name is recovered from the `ToolResultMessage.tool_name` field if
  present, otherwise falls back to `"tool"`.
- No deduplication in this iteration (different from hermes-agent) — keep it
  simple.

The pre-pruning pass runs **before** boundary finding. Boundary finding and
the summariser transcript both operate on the pruned content.

**Critical invariant — refs must come from original messages.**
`CompactionState.summarized_message_refs` is computed by `message_refs()` which
SHA256-hashes each message for stale-state detection on the next turn.
`_state_matches_history()` compares those refs against the *raw* `messages`
list (never pruned). Therefore refs must always be computed from the original
unpruned slice `messages[boundary:new_boundary]`, not from `pruned_messages`.
The pruner output is used **only** for token counting and transcript generation.

In `transform_context`, the call sequence is:

```
pruned = prune_tool_results(messages, keep_tail=tail_start)
...
new_state = summarize(
    messages_to_summarize=pruned[boundary:new_boundary],  # transcript only
    ref_messages=messages[boundary:new_boundary],          # refs from originals
    ...
)
```

`summarize()` gains an optional `ref_messages` parameter. When provided, refs
and IDs are extracted from `ref_messages` instead of `messages_to_summarize`.

### 2.3 Token-based tail protection

**Current**: `keep_recent_messages: int = 8` — an arbitrary message count.

**Fix**: replace with `keep_tail_tokens: int` (default `8_000`). The boundary
finder walks backward from the end of the message list, accumulating
`approx_tokens()` per message, and stops when the accumulated count exceeds
`keep_tail_tokens`. The resulting message index becomes the candidate tail
start.

`safe_boundary()` gains a `keep_tail_tokens: int` parameter. The existing
`keep_recent: int` parameter is kept temporarily for backwards-compatibility
but is deprecated.

`CompactionMiddleware.__init__` replaces `keep_recent_messages` with
`keep_tail_tokens` (default `8_000`).

**Tail start is computed once and shared.** `transform_context` calls
`_tail_start_by_tokens(messages, keep_tail_tokens)` once to get `tail_start`
(an integer index). That same index is passed to both the pruner and to
`safe_boundary` as the initial candidate. This guarantees both see the same
protection boundary — no token-to-count conversion.

**Overflow behaviour of `_tail_start_by_tokens`:** if the last message alone
exceeds `budget`, return `len(messages) - 1` so the tail always contains at
least one message. If *all* messages together are under budget, return `0`
(entire history in tail, nothing to compact). The returned index is always in
`[0, len(messages) - 1]` for non-empty lists; for empty input return `0`.

### 2.4 Dynamic `max_summary_tokens`

**Current**: fixed `max_summary_tokens: int = 1024` passed to the summariser.

**Fix**: compute a budget at summarisation time:

```python
content_tokens = approx_tokens(messages_to_summarize)
budget = max(512, min(int(content_tokens * _SUMMARY_RATIO), _SUMMARY_MAX))
# _SUMMARY_RATIO = 0.15, _SUMMARY_MAX = 4096
```

The `max_summary_tokens` constructor parameter becomes an **override** (when
provided, use it verbatim; when `None`, use the dynamic formula). Default
changes to `None`.

### 2.5 Circuit breaker

Track consecutive **LLM summariser** failures in
`AgentContext.extra["compaction_failures"]` (int, default 0).

- On LLM success: reset to 0.
- On LLM failure: increment; then attempt fallback summary (§2.7). The fallback
  is a successful compaction (context shrinks) — it does NOT count as an LLM
  failure for circuit-breaker purposes.
- When `compaction_failures >= _MAX_FAILURES = 3`: skip the LLM call but
  **still run the fallback**. The breaker gates only the expensive LLM call.
  Log a warning once when the breaker first trips.
- The LLM failure counter resets only when the LLM summariser succeeds, not
  when the fallback succeeds.

This separation ensures the agent continues to compact (via fallback) even
after three LLM failures, so context never grows unbounded due to a temporarily
broken summariser model.

### 2.6 Anti-thrashing guard

After each compaction attempt (LLM or fallback), record savings:

```python
savings_pct = (tokens_before - tokens_after) / tokens_before * 100
ctx.extra["compaction_low_savings_count"] = (
    low_savings + 1 if savings_pct < _MIN_SAVINGS_PCT else 0
)
```

When `compaction_low_savings_count >= _MAX_LOW_SAVINGS = 2`, skip the next
compaction trigger. Log a debug message.

**Reset conditions** (any one clears the guard):
1. A subsequent compaction saves ≥ `_MIN_SAVINGS_PCT = 10.0 %` — tracked by
   resetting the counter to 0 in the savings recording step above.
2. The candidate `new_boundary` advances by `_ANTI_THRASH_NEW_MSGS = 8` or
   more messages beyond the current boundary — enough new content has
   accumulated that compaction is likely worthwhile again. Check this before
   the guard fires: if `new_boundary - boundary >= _ANTI_THRASH_NEW_MSGS`,
   skip the guard and proceed.
3. Context exceeds `max_tokens_before_compact * _ANTI_THRASH_FORCE_RATIO = 1.5`
   — treat as an emergency override regardless of prior savings.

These reset conditions prevent the guard from permanently disabling compaction
for a long-running agent.

### 2.7 Static fallback summary

When the LLM summariser raises an exception **and** the circuit breaker has not
yet tripped (i.e. this is the first or second failure), generate a deterministic
fallback summary from the message list structure and store it.

Fallback format:

```
[Compaction fallback — LLM summariser unavailable]
User requests: {list of user message first lines, max 5}
Tool calls: {distinct tool names seen, sorted}
```

This is intentionally low-fidelity. Its purpose is to allow compaction to
proceed (reducing context size) even when the summariser is unavailable, so the
agent is not stuck over-limit on every subsequent turn.

The `CompactionState` gains a boolean field `is_fallback: bool = False` to allow
callers to distinguish fallback from real summaries.

---

## 3. What does NOT change

- The `CompactionState` schema (except adding `is_fallback`) — checkpointed
  state must remain compatible.
- The `safe_boundary()` invariant: boundary is always at a `UserMessage`, never
  splits a tool-call/result pair.
- The `extra_llm_calls()` hook for tracing.
- The cumulative merge approach (`<previous_summary>` passed back to
  summariser).
- The stale-state validation (SHA256 refs).

---

## 4. File map

| File | Change |
|------|--------|
| `cubepi/middleware/compaction/pruner.py` | **New.** `prune_tool_results(messages, keep_tail) -> list[Message]`. |
| `cubepi/middleware/compaction/summarizer.py` | `_format_message_for_summary`: add per-field-truncated arguments. Dynamic token budget. |
| `cubepi/middleware/compaction/boundary.py` | `safe_boundary`: add `keep_tail_tokens` param, deprecate `keep_recent`. |
| `cubepi/middleware/compaction/state.py` | Add `is_fallback: bool = False` to `CompactionState`. |
| `cubepi/middleware/compaction/__init__.py` | Orchestrate pre-pruning, circuit breaker, anti-thrashing, fallback. Replace `keep_recent_messages` with `keep_tail_tokens`. |
| `tests/middleware/compaction/test_pruner.py` | **New.** Unit tests for pre-pruning pass. |
| `tests/middleware/compaction/test_summarizer.py` | Add tests for argument formatting, dynamic budget. |
| `tests/middleware/compaction/test_boundary.py` | Add tests for token-based boundary. |
| `tests/middleware/test_compaction.py` | Add tests for circuit breaker, anti-thrashing, fallback. |

---

## 5. Implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve `CompactionMiddleware` with pre-pruning, argument-aware transcripts, token-based tail protection, dynamic summary budgets, a circuit breaker, anti-thrashing guard, and a static fallback summary.

**Architecture:** Seven focused changes spread across four existing files and one new file (`pruner.py`). Each task is independently testable. Tasks 1–3 are groundwork; Tasks 4–7 layer on top.

**Tech Stack:** Python 3.11+, pytest (asyncio_mode=auto), FauxProvider for LLM stubbing, pydantic.

---

### Task 1: Tool-result pre-pruning pass

**Files:**
- Create: `cubepi/middleware/compaction/pruner.py`
- Create: `tests/middleware/compaction/test_pruner.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/middleware/compaction/test_pruner.py
import pytest
from cubepi.middleware.compaction.pruner import prune_tool_results
from cubepi.providers.base import (
    AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage,
)

def _user(text="hi"):
    return UserMessage(content=[TextContent(text=text)])

def _assistant_with_call(tool_name, call_id, args=None):
    return AssistantMessage(content=[ToolCall(id=call_id, name=tool_name, arguments=args or {})])

def _result(call_id, text, tool_name="tool"):
    return ToolResultMessage(
        tool_call_id=call_id,
        tool_name=tool_name,
        content=[TextContent(text=text)],
    )

def test_short_result_kept_intact():
    msgs = [
        _user(), _assistant_with_call("bash", "c1"), _result("c1", "ok", "bash"),
        _user(), _assistant_with_call("bash", "c2"), _result("c2", "ok2", "bash"),
    ]
    pruned = prune_tool_results(msgs, keep_tail=2)
    # last 2 messages untouched; first result replaced
    assert "bash" in pruned[2].content[0].text  # one-liner
    assert pruned[5].content[0].text == "ok2"   # tail kept

def test_large_result_replaced_with_one_liner():
    big = "x" * 5000
    msgs = [
        _user(), _assistant_with_call("read_file", "c1"), _result("c1", big, "read_file"),
        _user(),
    ]
    pruned = prune_tool_results(msgs, keep_tail=1)
    result_text = pruned[2].content[0].text
    assert len(result_text) < 200
    assert "read_file" in result_text
    assert "5000" in result_text or "5 000" in result_text or "chars" in result_text

def test_tail_messages_kept_intact():
    big = "x" * 5000
    msgs = [
        _user(), _assistant_with_call("bash", "c1"), _result("c1", big, "bash"),
    ]
    pruned = prune_tool_results(msgs, keep_tail=3)
    # all 3 in tail — none pruned
    assert pruned[2].content[0].text == big

def test_result_already_short_kept_intact():
    msgs = [
        _user(), _assistant_with_call("bash", "c1"), _result("c1", "exit 0", "bash"),
        _user(),
    ]
    pruned = prune_tool_results(msgs, keep_tail=1)
    assert pruned[2].content[0].text == "exit 0"

def test_non_tool_result_messages_untouched():
    msgs = [_user("hello"), _user("world")]
    assert prune_tool_results(msgs, keep_tail=0) == msgs
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/middleware/compaction/test_pruner.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — `pruner` does not exist yet.

- [ ] **Step 3: Implement `pruner.py`**

```python
# cubepi/middleware/compaction/pruner.py
from __future__ import annotations

from cubepi.providers.base import Message, TextContent, ToolResultMessage

_PRUNE_KEEP_CHARS = 120


def prune_tool_results(messages: list[Message], *, keep_tail: int) -> list[Message]:
    """Replace old ToolResultMessage content with a compact one-liner.

    Messages within the last ``keep_tail`` positions are left intact.
    Results whose text is already <= _PRUNE_KEEP_CHARS chars are also kept.
    """
    if keep_tail >= len(messages):
        return list(messages)

    boundary = len(messages) - keep_tail
    result: list[Message] = []

    for i, msg in enumerate(messages):
        if i >= boundary or not isinstance(msg, ToolResultMessage):
            result.append(msg)
            continue

        text = _extract_text(msg)
        if len(text) <= _PRUNE_KEEP_CHARS:
            result.append(msg)
            continue

        tool_name = getattr(msg, "tool_name", None) or "tool"
        summary = f"[{tool_name}] {len(text)} chars"
        pruned = msg.model_copy(
            update={"content": [TextContent(text=summary)]}
        )
        result.append(pruned)

    return result


def _extract_text(msg: ToolResultMessage) -> str:
    parts = []
    for block in msg.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
    return "\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/middleware/compaction/test_pruner.py -v
```

Expected: all 5 pass.

- [ ] **Step 5: Commit**

```bash
git add cubepi/middleware/compaction/pruner.py tests/middleware/compaction/test_pruner.py
git commit -m "feat(compaction): add tool-result pre-pruning pass"
```

---

### Task 2: Tool call arguments in summariser transcript

**Files:**
- Modify: `cubepi/middleware/compaction/summarizer.py`
- Modify: `tests/middleware/compaction/test_summarizer.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/middleware/compaction/test_summarizer.py`:

```python
from cubepi.middleware.compaction.summarizer import _format_message_for_summary
from cubepi.providers.base import AssistantMessage, TextContent, ToolCall

def test_tool_call_arguments_included():
    msg = AssistantMessage(content=[
        ToolCall(id="c1", name="read_file", arguments={"path": "/home/user/config.py"}),
    ])
    result = _format_message_for_summary(msg)
    assert "read_file" in result
    assert "/home/user/config.py" in result

def test_tool_call_long_string_value_truncated():
    big_content = "x" * 1000
    msg = AssistantMessage(content=[
        ToolCall(id="c1", name="write_file", arguments={"path": "out.py", "content": big_content}),
    ])
    result = _format_message_for_summary(msg)
    assert "out.py" in result          # short field kept
    assert big_content not in result   # long field truncated
    assert "truncated" in result

def test_tool_call_non_json_arguments_graceful():
    msg = AssistantMessage(content=[
        ToolCall(id="c1", name="bash", arguments={"command": "ls -la"}),
    ])
    result = _format_message_for_summary(msg)
    assert "bash" in result

def test_tool_call_repr_max_chars():
    msg = AssistantMessage(content=[
        ToolCall(id="c1", name="search", arguments={"q": "a" * 2000}),
    ])
    result = _format_message_for_summary(msg)
    # entire formatted tool call portion must not blow up
    assert len(result) < 1000
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/middleware/compaction/test_summarizer.py -v -k "tool_call"
```

Expected: `AssertionError` — arguments not in output.

- [ ] **Step 3: Implement per-field argument truncation in `summarizer.py`**

Add at module top:

```python
import json

_ARG_VALUE_CHARS = 200
_ARG_REPR_MAX = 500
```

Add helper function:

```python
def _format_arguments(arguments: dict | None) -> str:
    """Serialise tool call arguments with per-field string truncation."""
    if not arguments:
        return ""
    try:
        shrunk = _shrink_strings(arguments)
        serialised = json.dumps(shrunk, ensure_ascii=False)
    except (TypeError, ValueError):
        serialised = str(arguments)
    if len(serialised) > _ARG_REPR_MAX:
        serialised = serialised[:_ARG_REPR_MAX] + "…"
    return " " + serialised


def _shrink_strings(obj: object) -> object:
    if isinstance(obj, str):
        return obj if len(obj) <= _ARG_VALUE_CHARS else obj[:_ARG_VALUE_CHARS] + "...[truncated]"
    if isinstance(obj, dict):
        return {k: _shrink_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_shrink_strings(v) for v in obj]
    return obj
```

Replace the `ToolCall` branch in `_format_message_for_summary`:

```python
elif isinstance(block, ToolCall):
    args_repr = _format_arguments(block.arguments)
    parts.append(f"[tool_call:{block.name}]{args_repr}")
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/middleware/compaction/test_summarizer.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add cubepi/middleware/compaction/summarizer.py tests/middleware/compaction/test_summarizer.py
git commit -m "feat(compaction): include tool call arguments in summariser transcript"
```

---

### Task 3: Token-based tail protection

**Files:**
- Modify: `cubepi/middleware/compaction/boundary.py`
- Modify: `tests/middleware/compaction/test_boundary.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/middleware/compaction/test_boundary.py`:

```python
from cubepi.middleware.compaction.boundary import safe_boundary
from cubepi.providers.base import (
    AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage,
)

def _big_user(chars: int) -> UserMessage:
    return UserMessage(content=[TextContent(text="x" * chars)])

def test_token_based_tail_protects_by_budget():
    # 4 messages, each ~2000 chars ≈ 1000 tokens
    msgs = [_big_user(2000), _big_user(2000), _big_user(2000), _big_user(2000)]
    # tail budget = 1500 tokens → protects last 2 messages (first one that fits
    # brings accumulated to ~1000, second brings it to ~2000 > 1500 so tail
    # starts at index 2, meaning messages[2] and [3] are protected)
    boundary = safe_boundary(msgs, keep_tail_tokens=1500, min_compact=1)
    assert boundary is not None
    assert boundary == 2  # exactly 2 messages protected

def test_token_based_tail_all_fit_returns_none():
    # all 3 messages together are under budget → nothing to compact
    msgs = [_big_user(10), _big_user(10), _big_user(10)]
    boundary = safe_boundary(msgs, keep_tail_tokens=100_000, min_compact=1)
    assert boundary is None

def test_token_based_tail_oversized_last_message():
    # last message alone exceeds budget — it must still be in the tail
    msgs = [_big_user(100), _big_user(100), _big_user(100_000)]
    boundary = safe_boundary(msgs, keep_tail_tokens=1000, min_compact=1)
    # tail_start = len-1 = 2 (clamped), candidate walks back to find UserMessage
    # message[2] is UserMessage → safe_boundary returns 2
    assert boundary == 2

def test_token_based_exact_budget():
    # each message exactly fits budget — tail has exactly 1 message
    msgs = [_big_user(2000), _big_user(2000), _big_user(2000)]
    # budget = 1000 tokens; each msg ≈ 1000 tokens; walking back: first msg
    # brings accumulated to 1000 = budget (not strictly over), so include it.
    # second msg would push to 2000 > 1000, so tail_start = 2
    boundary = safe_boundary(msgs, keep_tail_tokens=1000, min_compact=1)
    assert boundary is not None
    assert boundary == 2
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/middleware/compaction/test_boundary.py -v -k "token_based"
```

Expected: `TypeError` — `keep_tail_tokens` not a valid parameter.

- [ ] **Step 3: Implement token-based tail in `boundary.py`**

```python
# cubepi/middleware/compaction/boundary.py
from __future__ import annotations

from cubepi.middleware.compaction.tokens import approx_tokens
from cubepi.providers.base import (
    AssistantMessage, Message, ToolCall, ToolResultMessage, UserMessage,
)


def safe_boundary(
    messages: list[Message],
    *,
    keep_tail_tokens: int | None = None,
    keep_recent: int | None = None,   # deprecated, use keep_tail_tokens
    min_compact: int = 1,
) -> int | None:
    """Return a message index that can be summarised safely.

    Tail size is determined by ``keep_tail_tokens`` (preferred) or the legacy
    ``keep_recent`` message count.  At least one of the two must be provided.
    """
    if keep_tail_tokens is not None:
        tail_start = _tail_start_by_tokens(messages, keep_tail_tokens)
    elif keep_recent is not None:
        tail_start = max(0, len(messages) - keep_recent)
    else:
        raise ValueError("Provide keep_tail_tokens or keep_recent")

    candidate = tail_start
    while candidate > 0:
        if not isinstance(messages[candidate], UserMessage):
            candidate -= 1
            continue
        if not _suffix_is_self_contained(messages[candidate:]):
            candidate -= 1
            continue
        if candidate < min_compact:
            return None
        return candidate

    return None


def _tail_start_by_tokens(messages: list[Message], budget: int) -> int:
    """Walk backward accumulating token estimates; return where the tail starts.

    Overflow rule: if a single message exceeds ``budget``, it is still included
    in the tail (tail always contains at least one message for non-empty input).
    Returns an index in [0, len(messages)-1] for non-empty input; 0 for empty.
    """
    if not messages:
        return 0
    accumulated = 0
    for i in range(len(messages) - 1, -1, -1):
        msg_tokens = approx_tokens([messages[i]])
        if accumulated + msg_tokens > budget and accumulated > 0:
            # adding this message would exceed budget and tail already has
            # at least one message — stop here
            return i + 1
        accumulated += msg_tokens
    # all messages fit in budget (or only one message total)
    return 0


def _suffix_is_self_contained(suffix: list[Message]) -> bool:
    available_call_ids: set[str] = set()
    for message in suffix:
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolCall) and block.id:
                    available_call_ids.add(block.id)
        elif isinstance(message, ToolResultMessage):
            if message.tool_call_id and message.tool_call_id not in available_call_ids:
                return False
    return True
```

- [ ] **Step 4: Run all boundary tests**

```bash
uv run pytest tests/middleware/compaction/test_boundary.py -v
```

Expected: all pass (including pre-existing tests with `keep_recent`).

- [ ] **Step 5: Commit**

```bash
git add cubepi/middleware/compaction/boundary.py tests/middleware/compaction/test_boundary.py
git commit -m "feat(compaction): token-based tail protection in safe_boundary"
```

---

### Task 4: Dynamic summary token budget

**Files:**
- Modify: `cubepi/middleware/compaction/summarizer.py`
- Modify: `tests/middleware/compaction/test_summarizer.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/middleware/compaction/test_summarizer.py`:

```python
from cubepi.middleware.compaction.summarizer import _dynamic_summary_budget
from cubepi.providers.base import UserMessage, TextContent

def test_dynamic_budget_scales_with_content():
    small = [UserMessage(content=[TextContent(text="hi")])]
    large = [UserMessage(content=[TextContent(text="x" * 20_000)])]
    assert _dynamic_summary_budget(small) == 512           # floor
    assert _dynamic_summary_budget(large) > 512
    assert _dynamic_summary_budget(large) <= 4096          # ceiling

def test_dynamic_budget_floor():
    assert _dynamic_summary_budget([]) == 512

def test_dynamic_budget_ceiling():
    huge = [UserMessage(content=[TextContent(text="x" * 200_000)])]
    assert _dynamic_summary_budget(huge) == 4096
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/middleware/compaction/test_summarizer.py -v -k "dynamic_budget"
```

Expected: `ImportError` — `_dynamic_summary_budget` does not exist.

- [ ] **Step 3: Add `_dynamic_summary_budget` and update `summarize()`**

Add to `summarizer.py`:

```python
_SUMMARY_RATIO = 0.15
_SUMMARY_MAX = 4096
_SUMMARY_MIN = 512


def _dynamic_summary_budget(messages: list[Message]) -> int:
    from cubepi.middleware.compaction.tokens import approx_tokens
    content_tokens = approx_tokens(messages)
    return max(_SUMMARY_MIN, min(int(content_tokens * _SUMMARY_RATIO), _SUMMARY_MAX))
```

Update `summarize()` signature — add `ref_messages` and make `max_summary_tokens` optional:

```python
async def summarize(
    *,
    model: BoundModel,
    messages_to_summarize: list[Message],
    existing: CompactionState | None,
    ref_messages: list[Message] | None = None,  # overrides source for ID/refs
    max_summary_tokens: int | None = None,       # None → dynamic
    abort_signal: asyncio.Event | None = None,
) -> CompactionState:
    ref_source = ref_messages if ref_messages is not None else messages_to_summarize
    budget = max_summary_tokens if max_summary_tokens is not None else _dynamic_summary_budget(messages_to_summarize)

    # ... existing generate() call unchanged ...

    # Replace ref extraction to use ref_source instead of messages_to_summarize:
    new_ids = [str(getattr(m, "id", "") or "") for m in ref_source]
    new_ids = [mid for mid in new_ids if mid]
    prior_ids = list(existing.summarized_message_ids) if existing else []
    prior_refs = list(existing.summarized_message_refs) if existing else []
    last_id = (
        new_ids[-1] if new_ids
        else (existing.last_summarized_message_id if existing else None)
    )
    return CompactionState(
        summary=text.strip(),
        summarized_message_ids=prior_ids + new_ids,
        summarized_message_refs=prior_refs + message_refs(ref_source),
        last_summarized_message_id=last_id,
    )
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/middleware/compaction/test_summarizer.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add cubepi/middleware/compaction/summarizer.py tests/middleware/compaction/test_summarizer.py
git commit -m "feat(compaction): dynamic summary token budget"
```

---

### Task 5: Static fallback summary

**Files:**
- Modify: `cubepi/middleware/compaction/state.py`
- Modify: `cubepi/middleware/compaction/summarizer.py`
- Modify: `tests/middleware/compaction/test_summarizer.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/middleware/compaction/test_summarizer.py`:

```python
from cubepi.middleware.compaction.summarizer import build_fallback_summary
from cubepi.providers.base import (
    AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage,
)

def test_fallback_summary_includes_user_requests():
    msgs = [
        UserMessage(content=[TextContent(text="Please write a hello world script")]),
        AssistantMessage(content=[TextContent(text="Sure")]),
    ]
    state = build_fallback_summary(msgs, existing=None)
    assert state.is_fallback is True
    assert "hello world" in state.summary.lower() or "Please write" in state.summary

def test_fallback_summary_includes_tool_names():
    msgs = [
        UserMessage(content=[TextContent(text="run the tests")]),
        AssistantMessage(content=[ToolCall(id="c1", name="bash", arguments={"command": "pytest"})]),
        ToolResultMessage(tool_call_id="c1", content=[TextContent(text="3 passed")]),
    ]
    state = build_fallback_summary(msgs, existing=None)
    assert "bash" in state.summary

def test_fallback_summary_merges_existing():
    from cubepi.middleware.compaction.state import CompactionState
    existing = CompactionState(summary="prior context", is_fallback=False)
    msgs = [UserMessage(content=[TextContent(text="new task")])]
    state = build_fallback_summary(msgs, existing=existing)
    assert "prior context" in state.summary
    assert state.is_fallback is True
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/middleware/compaction/test_summarizer.py -v -k "fallback"
```

Expected: `ImportError` — `build_fallback_summary` does not exist.

- [ ] **Step 3: Add `is_fallback` to `CompactionState`**

```python
# cubepi/middleware/compaction/state.py  (add field)
class CompactionState(BaseModel):
    summary: str
    summarized_message_ids: list[str] = Field(default_factory=list)
    summarized_message_refs: list[str] = Field(default_factory=list)
    last_summarized_message_id: str | None = None
    is_fallback: bool = False
```

- [ ] **Step 4: Add `build_fallback_summary` to `summarizer.py`**

```python
def build_fallback_summary(
    messages_to_summarize: list[Message],
    *,
    existing: CompactionState | None,
    ref_messages: list[Message] | None = None,
) -> CompactionState:
    """Deterministic fallback when the LLM summariser is unavailable.

    ``ref_messages`` overrides which messages are used for ID/ref extraction.
    Pass the original (unpruned) slice when the transcript was built from pruned
    content so SHA256 refs stay consistent with ``_state_matches_history``.
    """
    ref_source = ref_messages if ref_messages is not None else messages_to_summarize

    user_lines: list[str] = []
    tool_names: list[str] = []

    for msg in messages_to_summarize:
        if isinstance(msg, UserMessage):
            for block in msg.content:
                if isinstance(block, TextContent) and block.text.strip():
                    first_line = block.text.strip().splitlines()[0][:120]
                    user_lines.append(first_line)
                    if len(user_lines) >= 5:
                        break
        elif isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, ToolCall) and block.name not in tool_names:
                    tool_names.append(block.name)

    parts: list[str] = ["[Compaction fallback — LLM summariser unavailable]"]
    if existing and existing.summary:
        parts.append(f"Prior context: {existing.summary}")
    if user_lines:
        parts.append("User requests: " + "; ".join(user_lines))
    if tool_names:
        parts.append("Tool calls: " + ", ".join(sorted(tool_names)))

    summary = "\n".join(parts)

    prior_ids = list(existing.summarized_message_ids) if existing else []
    prior_refs = list(existing.summarized_message_refs) if existing else []
    new_ids = [str(getattr(m, "id", "") or "") for m in ref_source]
    new_ids = [i for i in new_ids if i]

    return CompactionState(
        summary=summary,
        summarized_message_ids=prior_ids + new_ids,
        summarized_message_refs=prior_refs + message_refs(ref_source),
        last_summarized_message_id=new_ids[-1] if new_ids else (existing.last_summarized_message_id if existing else None),
        is_fallback=True,
    )
```

- [ ] **Step 5: Run tests**

```bash
uv run pytest tests/middleware/compaction/test_summarizer.py -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add cubepi/middleware/compaction/state.py cubepi/middleware/compaction/summarizer.py tests/middleware/compaction/test_summarizer.py
git commit -m "feat(compaction): static fallback summary when LLM unavailable"
```

---

### Task 6: Circuit breaker + anti-thrashing + wire everything up

**Files:**
- Modify: `cubepi/middleware/compaction/__init__.py`
- Modify: `tests/middleware/test_compaction.py`

This task updates `CompactionMiddleware` to:
1. Call `prune_tool_results` before boundary finding.
2. Pass `keep_tail_tokens` to `safe_boundary`.
3. Pass `max_summary_tokens=None` (dynamic) unless overridden.
4. Use `build_fallback_summary` on failure instead of returning compressed unchanged.
5. Track failure count (circuit breaker).
6. Track consecutive low-savings rounds (anti-thrashing).

- [ ] **Step 1: Write the failing tests**

Add to `tests/middleware/test_compaction.py`. The file already has `FauxProvider`
fixtures and helpers — adapt the pattern. A "failing model" is a `BoundModel`
whose `generate()` always raises `RuntimeError`. Look at the existing fixture
setup in the file; create `failing_model` similarly by making `FauxProvider`
raise on every call, or by using a minimal stub.

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from cubepi.middleware.compaction import CompactionMiddleware
from cubepi.agent.types import AgentContext
from cubepi.providers.base import (
    AssistantMessage, BoundModel, TextContent, ToolCall,
    ToolResultMessage, UserMessage,
)

# --- helpers ---

def _big_msgs(n: int, chars: int = 2000) -> list:
    """Alternating user/assistant messages large enough to trigger compaction."""
    msgs = []
    for i in range(n):
        if i % 2 == 0:
            msgs.append(UserMessage(content=[TextContent(text="q" * chars)]))
        else:
            msgs.append(AssistantMessage(content=[TextContent(text="a" * chars)]))
    return msgs

def _failing_bound_model() -> BoundModel:
    """BoundModel whose generate() always raises RuntimeError."""
    m = MagicMock(spec=BoundModel)
    m.generate = AsyncMock(side_effect=RuntimeError("summariser down"))
    return m

def _counting_bound_model(summary: str = "ok") -> tuple:
    """BoundModel that records call count and returns a fixed summary."""
    from cubepi.providers.base import ModelResponse
    call_count = {"n": 0}
    resp = MagicMock(spec=ModelResponse)
    resp.content = [TextContent(text=summary)]
    resp.error_message = None
    resp.usage = None
    async def _generate(*args, **kwargs):
        call_count["n"] += 1
        return resp
    m = MagicMock(spec=BoundModel)
    m.generate = _generate
    return m, call_count

# --- circuit breaker ---

async def test_circuit_breaker_stops_after_three_failures():
    """After 3 LLM failures the breaker opens; 4th call uses fallback, not LLM."""
    failing = _failing_bound_model()
    mw = CompactionMiddleware(
        summary_model=failing,
        max_tokens_before_compact=100,
        keep_tail_tokens=200,
    )
    ctx = AgentContext(thread_id="t1")
    msgs = _big_msgs(10, chars=200)  # well over 100-token threshold

    # Turns 1-3: LLM fails, fallback writes state, failure counter climbs
    for i in range(3):
        result = await mw.transform_context(msgs, ctx=ctx)
        assert ctx.extra["compaction_failures"] == i + 1
        # fallback state still written — result is compressed
        assert "compaction" in ctx.extra

    # Turn 4: breaker is open — LLM must NOT be called again
    call_count_before = failing.generate.call_count
    await mw.transform_context(msgs, ctx=ctx)
    assert failing.generate.call_count == call_count_before  # no new LLM call
    # failure counter frozen at 3 (not incremented beyond MAX_FAILURES)
    assert ctx.extra["compaction_failures"] == 3

async def test_circuit_breaker_resets_on_llm_success():
    """Failure counter resets to 0 after LLM summariser succeeds once."""
    model, calls = _counting_bound_model("summary text")
    mw = CompactionMiddleware(
        summary_model=model,
        max_tokens_before_compact=100,
        keep_tail_tokens=200,
    )
    ctx = AgentContext(thread_id="t1")
    ctx.extra["compaction_failures"] = 2  # pre-seed: 2 prior failures
    msgs = _big_msgs(10, chars=200)

    await mw.transform_context(msgs, ctx=ctx)
    assert ctx.extra["compaction_failures"] == 0  # reset after success

async def test_fallback_written_when_breaker_open():
    """When breaker is open, fallback still runs and compresses context."""
    failing = _failing_bound_model()
    mw = CompactionMiddleware(
        summary_model=failing,
        max_tokens_before_compact=100,
        keep_tail_tokens=200,
    )
    ctx = AgentContext(thread_id="t1")
    ctx.extra["compaction_failures"] = 3  # breaker already open
    msgs = _big_msgs(10, chars=200)

    result = await mw.transform_context(msgs, ctx=ctx)
    # fallback must have written state (context was compressed)
    assert "compaction" in ctx.extra
    assert len(result) < len(msgs)

# --- anti-thrashing ---

async def test_anti_thrashing_skips_after_two_low_savings():
    """After two low-savings compactions, the third trigger is skipped."""
    model, calls = _counting_bound_model("x")  # very short summary → low savings
    mw = CompactionMiddleware(
        summary_model=model,
        max_tokens_before_compact=100,
        keep_tail_tokens=200,
    )
    ctx = AgentContext(thread_id="t1")
    ctx.extra["compaction_low_savings_count"] = 2  # guard already tripped
    msgs = _big_msgs(10, chars=200)

    call_count_before = calls["n"]
    await mw.transform_context(msgs, ctx=ctx)
    assert calls["n"] == call_count_before  # no LLM call made

async def test_anti_thrashing_resets_after_good_savings():
    """Low-savings counter resets to 0 when a compaction saves >= 10%."""
    # Use a summary that is significantly shorter than the messages
    model, calls = _counting_bound_model("short summary")
    mw = CompactionMiddleware(
        summary_model=model,
        max_tokens_before_compact=100,
        keep_tail_tokens=200,
    )
    ctx = AgentContext(thread_id="t1")
    ctx.extra["compaction_low_savings_count"] = 1  # 1 prior low-savings run
    msgs = _big_msgs(10, chars=2000)  # very large messages → high savings

    await mw.transform_context(msgs, ctx=ctx)
    assert ctx.extra.get("compaction_low_savings_count", 0) == 0  # reset

async def test_anti_thrashing_resets_when_enough_new_messages():
    """Guard is bypassed when new_boundary advances by >= _ANTI_THRASH_NEW_MSGS."""
    model, calls = _counting_bound_model("summary")
    mw = CompactionMiddleware(
        summary_model=model,
        max_tokens_before_compact=100,
        keep_tail_tokens=200,
    )
    ctx = AgentContext(thread_id="t1")
    ctx.extra["compaction_low_savings_count"] = 2  # guard tripped
    # pre-seed a low boundary so new_boundary - boundary will be >= 8
    ctx.extra["compaction_until_msg_index"] = 0

    msgs = _big_msgs(20, chars=200)  # enough new messages beyond boundary
    call_count_before = calls["n"]
    await mw.transform_context(msgs, ctx=ctx)
    # LLM was called despite guard because enough new msgs accumulated
    assert calls["n"] > call_count_before

# --- pruned refs do not corrupt state ---

async def test_pruned_messages_do_not_corrupt_state_refs():
    """State refs must survive even when tool result content is pruned."""
    model, _ = _counting_bound_model("summary of work done")
    mw = CompactionMiddleware(
        summary_model=model,
        max_tokens_before_compact=50,
        keep_tail_tokens=100,
    )
    ctx = AgentContext(thread_id="t1")

    # Conversation with a large tool result that will be pruned
    big_result_text = "output " * 500  # >> 120 chars → will be pruned
    msgs = [
        UserMessage(content=[TextContent(text="run tests")]),
        AssistantMessage(content=[ToolCall(id="c1", name="bash", arguments={"cmd": "pytest"})]),
        ToolResultMessage(tool_call_id="c1", tool_name="bash", content=[TextContent(text=big_result_text)]),
        UserMessage(content=[TextContent(text="what next?")]),
        AssistantMessage(content=[TextContent(text="fix the failures")]),
        UserMessage(content=[TextContent(text="ok fix them")]),
    ]

    # First turn: compaction runs, state is written
    await mw.transform_context(msgs, ctx=ctx)
    assert "compaction" in ctx.extra
    boundary_after_first = ctx.extra["compaction_until_msg_index"]

    # Second turn with same messages: state must validate (not cleared)
    await mw.transform_context(msgs, ctx=ctx)
    # boundary must not have been reset to 0 (stale state detection fired)
    assert ctx.extra.get("compaction_until_msg_index", 0) >= boundary_after_first

# --- keep_tail_tokens replaces keep_recent_messages ---

def test_keep_tail_tokens_constructor():
    """CompactionMiddleware accepts keep_tail_tokens."""
    model, _ = _counting_bound_model()
    mw = CompactionMiddleware(
        summary_model=model,
        max_tokens_before_compact=100,
        keep_tail_tokens=2000,
    )
    assert mw._keep_tail_tokens == 2000

def test_keep_recent_messages_no_longer_accepted():
    """keep_recent_messages is no longer a valid constructor argument."""
    model, _ = _counting_bound_model()
    with pytest.raises(TypeError):
        CompactionMiddleware(
            summary_model=model,
            max_tokens_before_compact=100,
            keep_recent_messages=8,  # removed parameter
        )
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/middleware/test_compaction.py -v -k "circuit_breaker or anti_thrash or keep_tail"
```

Expected: `TypeError` or `AssertionError`.

- [ ] **Step 3: Rewrite `CompactionMiddleware.__init__` and `transform_context`**

```python
_MAX_FAILURES = 3
_MIN_SAVINGS_PCT = 10.0
_MAX_LOW_SAVINGS = 2
_ANTI_THRASH_NEW_MSGS = 8
_ANTI_THRASH_FORCE_RATIO = 1.5


class CompactionMiddleware(Middleware):
    def __init__(
        self,
        *,
        summary_model: BoundModel,
        max_tokens_before_compact: int,
        keep_tail_tokens: int = 8_000,
        max_summary_tokens: int | None = None,   # None → dynamic
        min_compact_messages: int = 4,
    ) -> None:
        self._summary_model = summary_model
        self._max_tokens_before = max_tokens_before_compact
        self._keep_tail_tokens = keep_tail_tokens
        self._max_summary_tokens = max_summary_tokens
        self._min_compact = min_compact_messages

    async def transform_context(
        self,
        messages: list[Message],
        *,
        ctx: AgentContext,
        signal: asyncio.Event | None = None,
    ) -> list[Message]:
        state = _load_state(ctx.extra.get("compaction"))
        raw_boundary = ctx.extra.get("compaction_until_msg_index")
        boundary = int(raw_boundary) if isinstance(raw_boundary, (int, float, str)) else 0

        if state is None and ("compaction" in ctx.extra or boundary > 0):
            boundary = 0
            _clear_state(ctx)
        if boundary >= len(messages) or not _state_matches_history(messages, state, boundary):
            boundary = 0
            state = None
            _clear_state(ctx)

        # Compute shared tail start once — used by both pruner and safe_boundary
        tail_start = _tail_start_by_tokens(messages, self._keep_tail_tokens)

        # Phase 1: pre-prune old tool results (cheap, no LLM call)
        # Pruner uses the same tail_start so it never touches protected messages.
        pruned_messages = prune_tool_results(messages, keep_tail=len(messages) - tail_start)

        compressed = _compressed_view(pruned_messages, state, boundary)

        tokens_now = approx_tokens(compressed)
        if tokens_now < self._max_tokens_before:
            return compressed

        # Find boundary before running guards (needed for anti-thrash new-msgs check)
        new_boundary = safe_boundary(
            pruned_messages,
            keep_tail_tokens=self._keep_tail_tokens,
            min_compact=max(self._min_compact, boundary + 1),
        )
        if new_boundary is None or new_boundary <= boundary:
            return compressed

        # Circuit breaker — gates LLM only; fallback always runs
        failures = ctx.extra.get("compaction_failures", 0)
        llm_allowed = failures < _MAX_FAILURES
        if not llm_allowed:
            logger.warning(
                "CompactionMiddleware: LLM circuit breaker open (%d failures), using fallback",
                failures,
            )

        # Anti-thrashing guard (skips both LLM and fallback)
        low_savings = ctx.extra.get("compaction_low_savings_count", 0)
        force_emergency = tokens_now >= self._max_tokens_before * _ANTI_THRASH_FORCE_RATIO
        enough_new = (new_boundary - boundary) >= _ANTI_THRASH_NEW_MSGS
        if low_savings >= _MAX_LOW_SAVINGS and not force_emergency and not enough_new:
            logger.debug("CompactionMiddleware: skipping — low savings guard active")
            return compressed

        tokens_before = approx_tokens(compressed)

        if llm_allowed:
            try:
                new_state = await summarize(
                    model=self._summary_model,
                    messages_to_summarize=pruned_messages[boundary:new_boundary],
                    ref_messages=messages[boundary:new_boundary],   # refs from originals
                    existing=state,
                    max_summary_tokens=self._max_summary_tokens,
                    abort_signal=signal,
                )
                ctx.extra["compaction_failures"] = 0
            except Exception as exc:  # noqa: BLE001
                logger.warning("CompactionMiddleware LLM summariser failed: %s", exc)
                ctx.extra["compaction_failures"] = failures + 1
                new_state = build_fallback_summary(
                    pruned_messages[boundary:new_boundary],
                    ref_messages=messages[boundary:new_boundary],
                    existing=state,
                )
        else:
            new_state = build_fallback_summary(
                pruned_messages[boundary:new_boundary],
                ref_messages=messages[boundary:new_boundary],
                existing=state,
            )

        ctx.extra["compaction"] = new_state.model_dump()
        ctx.extra["compaction_until_msg_index"] = new_boundary
        result = _compressed_view(pruned_messages, new_state, new_boundary)

        # Anti-thrashing tracking
        tokens_after = approx_tokens(result)
        if tokens_before > 0:
            savings_pct = (tokens_before - tokens_after) / tokens_before * 100
            ctx.extra["compaction_low_savings_count"] = (
                low_savings + 1 if savings_pct < _MIN_SAVINGS_PCT else 0
            )

        return result
```

Add to imports at the top of `__init__.py`:

```python
from cubepi.middleware.compaction.boundary import _tail_start_by_tokens
from cubepi.middleware.compaction.pruner import prune_tool_results
from cubepi.middleware.compaction.summarizer import build_fallback_summary
```

Also update `summarize()` and `build_fallback_summary()` in `summarizer.py` to
accept `ref_messages: list[Message] | None = None`. When provided, use
`ref_messages` instead of `messages_to_summarize` for all ref/ID extraction.
When `None`, fall back to `messages_to_summarize` (existing behaviour, no
breaking change for direct callers).

- [ ] **Step 4: Run full compaction test suite**

```bash
uv run pytest tests/middleware/ -v
```

Expected: all pass.

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all pass. Fix any regressions before proceeding.

- [ ] **Step 6: Run type check and linter**

```bash
uv run mypy cubepi/middleware/compaction/
uv run ruff check cubepi/middleware/compaction/ tests/middleware/
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add cubepi/middleware/compaction/__init__.py tests/middleware/test_compaction.py
git commit -m "feat(compaction): circuit breaker, anti-thrashing, fallback, pre-pruning wire-up"
```

---

## 6. Non-goals (explicitly deferred)

- **Post-compact context re-injection** (re-reading active files): requires
  application-level knowledge of which files the agent touched. Out of scope.
- **Tool-result deduplication by content hash**: hermes-agent does MD5 dedup;
  skipped here to keep pruner.py simple on first pass.
- **Microcompaction / cache-edit pruning**: claude-code uses the Anthropic
  `cache_edits` API to delete tool results without rewriting prefix. Requires
  provider-level support. Out of scope.
