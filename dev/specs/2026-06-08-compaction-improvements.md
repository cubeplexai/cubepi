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

The pre-pruning pass runs **before** boundary finding. This means boundary
finding and the summariser both operate on already-pruned content.

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

Track consecutive summariser failures in `AgentContext.extra["compaction_failures"]`
(int, default 0). After each failure, increment. On success, reset to 0.

When the count reaches `_MAX_FAILURES = 3`, `transform_context` skips the
summarisation attempt entirely and returns the current compressed view — same
as the "under threshold" fast path. Log a warning once when the breaker trips.

The breaker resets automatically the first time the agent successfully completes
a compaction.

### 2.6 Anti-thrashing guard

After each successful compaction, record:

```python
ctx.extra["compaction_savings_pct"] = savings_pct   # float 0–100
```

Where `savings_pct = (tokens_before - tokens_after) / tokens_before * 100`.

On the **next** compaction trigger, check: if the previous `savings_pct < 10.0`
**and** the current `savings_pct` (computed speculatively by comparing
`approx_tokens(compressed)` before and after a dry-run boundary search) would
also be < 10.0, skip compaction. Log a debug message.

Simpler implementation: skip the speculative dry-run; instead, after two
consecutive savings_pct < 10.0, skip until the breaker resets (next successful
compaction).

Track consecutive low-savings count in
`ctx.extra["compaction_low_savings_count"]` (int).

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

def _result(call_id, text, tool_name=None):
    r = ToolResultMessage(tool_call_id=call_id, content=[TextContent(text=text)])
    if tool_name:
        r.tool_name = tool_name
    return r

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
    # tail budget = 1500 tokens → should protect last 2 messages (≈ 2000 tokens)
    boundary = safe_boundary(msgs, keep_tail_tokens=1500, min_compact=1)
    # boundary must be ≤ 2 (at most 2 messages in tail)
    assert boundary is not None
    assert boundary <= 2

def test_token_based_tail_falls_back_to_first_message_if_all_fit():
    msgs = [_big_user(10), _big_user(10), _big_user(10)]
    boundary = safe_boundary(msgs, keep_tail_tokens=100_000, min_compact=1)
    # everything fits in tail budget, nothing to compact
    assert boundary is None
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
    """Walk backward accumulating token estimates; return where the tail starts."""
    accumulated = 0
    for i in range(len(messages) - 1, -1, -1):
        msg_tokens = approx_tokens([messages[i]])
        if accumulated + msg_tokens > budget:
            return i + 1
        accumulated += msg_tokens
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

Update `summarize()` signature — `max_summary_tokens` becomes optional:

```python
async def summarize(
    *,
    model: BoundModel,
    messages_to_summarize: list[Message],
    existing: CompactionState | None,
    max_summary_tokens: int | None = None,   # None → dynamic
    abort_signal: asyncio.Event | None = None,
) -> CompactionState:
    budget = max_summary_tokens if max_summary_tokens is not None else _dynamic_summary_budget(messages_to_summarize)

    response = await model.generate(
        ...
        max_output_tokens=budget,
        ...
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
) -> CompactionState:
    """Deterministic fallback when the LLM summariser is unavailable."""
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
    new_ids = [str(getattr(m, "id", "") or "") for m in messages_to_summarize]
    new_ids = [i for i in new_ids if i]

    return CompactionState(
        summary=summary,
        summarized_message_ids=prior_ids + new_ids,
        summarized_message_refs=prior_refs + message_refs(messages_to_summarize),
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

Add to `tests/middleware/test_compaction.py`:

```python
# Helpers already exist in the file — adapt as needed.

async def test_circuit_breaker_stops_after_three_failures(faux_provider, failing_model):
    """After 3 consecutive LLM failures, no more summariser calls."""
    # failing_model: a BoundModel whose provider always raises RuntimeError
    mw = CompactionMiddleware(
        summary_model=failing_model,
        max_tokens_before_compact=10,
        keep_tail_tokens=500,
    )
    ctx = AgentContext(thread_id="t1")
    big_messages = [...]  # enough messages to trigger compaction

    for _ in range(4):
        await mw.transform_context(big_messages, ctx=ctx)

    assert ctx.extra.get("compaction_failures", 0) >= 3
    # 4th call must NOT invoke the model (check call count on failing_model)

async def test_anti_thrashing_skips_when_savings_below_threshold(faux_provider):
    """If compaction saves < 10% twice in a row, skip on the third trigger."""
    ...

async def test_keep_tail_tokens_replaces_keep_recent_messages(faux_provider):
    """CompactionMiddleware accepts keep_tail_tokens, not keep_recent_messages."""
    mw = CompactionMiddleware(
        summary_model=...,
        max_tokens_before_compact=100,
        keep_tail_tokens=2000,
    )
    assert mw._keep_tail_tokens == 2000
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

        # Phase 1: pre-prune old tool results (cheap, no LLM call)
        pruned_messages = prune_tool_results(messages, keep_tail=self._keep_tail_tokens // 500 or 8)

        compressed = _compressed_view(pruned_messages, state, boundary)

        if approx_tokens(compressed) < self._max_tokens_before:
            return compressed

        # Circuit breaker
        failures = ctx.extra.get("compaction_failures", 0)
        if failures >= _MAX_FAILURES:
            logger.warning("CompactionMiddleware: circuit breaker open (%d failures)", failures)
            return compressed

        # Anti-thrashing guard
        low_savings = ctx.extra.get("compaction_low_savings_count", 0)
        if low_savings >= _MAX_LOW_SAVINGS:
            logger.debug("CompactionMiddleware: skipping — low savings in last %d runs", low_savings)
            return compressed

        new_boundary = safe_boundary(
            pruned_messages,
            keep_tail_tokens=self._keep_tail_tokens,
            min_compact=max(self._min_compact, boundary + 1),
        )
        if new_boundary is None or new_boundary <= boundary:
            return compressed

        tokens_before = approx_tokens(compressed)

        try:
            new_state = await summarize(
                model=self._summary_model,
                messages_to_summarize=pruned_messages[boundary:new_boundary],
                existing=state,
                max_summary_tokens=self._max_summary_tokens,
                abort_signal=signal,
            )
            ctx.extra["compaction_failures"] = 0
        except Exception as exc:  # noqa: BLE001
            logger.warning("CompactionMiddleware summariser failed: %s", exc)
            ctx.extra["compaction_failures"] = failures + 1
            # Use fallback summary so context shrinks even without LLM
            new_state = build_fallback_summary(
                pruned_messages[boundary:new_boundary],
                existing=state,
            )

        ctx.extra["compaction"] = new_state.model_dump()
        ctx.extra["compaction_until_msg_index"] = new_boundary
        result = _compressed_view(pruned_messages, new_state, new_boundary)

        # Anti-thrashing tracking
        tokens_after = approx_tokens(result)
        if tokens_before > 0:
            savings_pct = (tokens_before - tokens_after) / tokens_before * 100
            if savings_pct < _MIN_SAVINGS_PCT:
                ctx.extra["compaction_low_savings_count"] = low_savings + 1
            else:
                ctx.extra["compaction_low_savings_count"] = 0

        return result
```

Also add to the imports at the top:

```python
from cubepi.middleware.compaction.pruner import prune_tool_results
from cubepi.middleware.compaction.summarizer import build_fallback_summary
```

Update `safe_boundary` call to use `keep_tail_tokens`.

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
