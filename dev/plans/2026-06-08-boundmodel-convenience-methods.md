# BoundModel Convenience Methods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `BoundModel` directly drive a provider call (`await bound.generate(...)` / `await bound.stream(...)`), and propagate that ergonomics into middleware so we stop splitting `BoundModel` back into `(Provider, Model)` everywhere.

**Architecture:** Add two forwarding methods to the existing `@dataclass(frozen=True) BoundModel` that delegate to `self.provider.stream` / `self.provider.generate` with `model=self.spec`. Then migrate the two middleware seams that still expose the unwrapped pair:
1. `cubepi/middleware/compaction/summarizer.py::summarize` takes `model: BoundModel` and uses `model.generate(...)`.
2. `Middleware.extra_llm_calls()` returns `Iterable[BoundModel]` instead of `Iterable[tuple[Provider, Model]]`; `cubepi/tracing/recorder.py` is the only consumer and unwraps it.

**Non-goals:** Touching `Agent.__init__` / `cubepi/agent/loop.py` call sites is out of scope — they still pass `provider=` and `model=` separately and the loop module's signature is wide. Leave a follow-up note.

**Tech Stack:** Python 3.11+, Pydantic v2 (`Model` spec only — `BoundModel` is a stdlib `dataclass`), pytest with `asyncio_mode=auto`, `FauxProvider` for deterministic provider tests, ruff, mypy.

---

## File Structure

- **Modify** `cubepi/providers/base.py` — add `BoundModel.stream` and `BoundModel.generate` methods (lines 93–96 today).
- **Modify** `cubepi/middleware/compaction/summarizer.py` — `summarize()` signature: `provider: Provider, model: Model` → `model: BoundModel`.
- **Modify** `cubepi/middleware/compaction/__init__.py` — drop split fields; keep one `self._summary_model: BoundModel`; `extra_llm_calls()` returns `(self._summary_model,)`.
- **Modify** `cubepi/middleware/base.py` — `extra_llm_calls` return type → `Iterable[BoundModel]`; refresh docstring; drop now-unused `Model` import if applicable.
- **Modify** `cubepi/tracing/recorder.py` (around line 289) — adapt consumer loop to iterate `BoundModel`.
- **Create** `tests/providers/test_bound_model_calls.py` — focused forwarding tests for the two new methods on `BoundModel` (uses a small recording fake provider for `generate`, `FauxProvider` + queued response for `stream`).
- **Modify** `tests/middleware/compaction/test_summarizer.py` — three `summarize(provider=..., model=...)` call sites migrate to `summarize(model=BoundModel(provider=..., spec=...))`.
- **Modify** `tests/tracing/test_recorder.py` — three middleware mocks (`extra_llm_calls` at lines ~666, ~706, ~735) return `BoundModel` instead of tuples.
- **Modify** `website/docs/agents/providers.md` (or whichever current providers page documents `provider.model(...)`) — add a one-paragraph note that the returned `BoundModel` is itself callable via `.stream()` / `.generate()`.

`tests/middleware/test_compaction.py` already constructs `CompactionMiddleware(summary_model=BoundModel(...), ...)` (line 75), so the public-API surface needs no test changes there. Recorder tests at `tests/tracing/test_recorder.py:515` / `:591` likewise already construct `BoundModel(...)` directly and stay green after the Task 3 atomic flip.

---

## Task 1: `BoundModel.generate` forwards to the bound provider

**Files:**
- Modify: `cubepi/providers/base.py:93-96`
- Test: `tests/providers/test_bound_model_calls.py` (new)

- [ ] **Step 1: Write the failing test**

We use a small recording provider (not `FauxProvider`) because the goal here is to verify *forwarding semantics* — that `BoundModel.generate` calls `self.provider.generate(model=self.spec, **kwargs)` exactly. `FauxProvider` without a queued response just emits an error message, which would mask whether forwarding worked.

```python
# tests/providers/test_bound_model_calls.py
from __future__ import annotations

from typing import Any

import pytest

from cubepi.providers.base import (
    AssistantMessage,
    BaseProvider,
    Message,
    Model,
    StreamOptions,
    TextContent,
    ToolDefinition,
    UserMessage,
)


class _RecordingProvider(BaseProvider):
    def __init__(self) -> None:
        super().__init__(provider_id="rec")
        self.generate_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def generate(  # type: ignore[override]
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        thinking: Any = None,
        thinking_budgets: Any = None,
    ) -> AssistantMessage:
        self.generate_calls.append(
            {
                "model": model,
                "messages": messages,
                "system_prompt": system_prompt,
                "tools": tools,
                "options": options,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "thinking": thinking,
                "thinking_budgets": thinking_budgets,
            }
        )
        return AssistantMessage(
            content=[TextContent(text="ok")],
            provider_id=model.provider_id,
            model_id=model.id,
        )


@pytest.mark.asyncio
async def test_bound_model_generate_forwards_to_provider() -> None:
    provider = _RecordingProvider()
    bound = provider.model("model-x", temperature=0.5)
    messages = [UserMessage(content=[TextContent(text="hi")])]

    response = await bound.generate(
        messages=messages,
        system_prompt="be brief",
        max_output_tokens=64,
        temperature=0.0,
        thinking="off",
    )

    assert isinstance(response, AssistantMessage)
    assert response.provider_id == "rec"
    assert response.model_id == "model-x"

    assert len(provider.generate_calls) == 1
    call = provider.generate_calls[0]
    assert call["model"] is bound.spec
    assert call["messages"] is messages
    assert call["system_prompt"] == "be brief"
    assert call["max_output_tokens"] == 64
    assert call["temperature"] == 0.0
    assert call["thinking"] == "off"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/chris/cubepi/.worktrees/2026-06-08-boundmodel-methods
uv run pytest tests/providers/test_bound_model_calls.py::test_bound_model_generate_forwards_to_provider -v
```

Expected: FAIL with `AttributeError: 'BoundModel' object has no attribute 'generate'`.

- [ ] **Step 3: Add `generate` to `BoundModel`**

In `cubepi/providers/base.py`, replace the existing class body (currently lines 93–96):

```python
@dataclass(frozen=True)
class BoundModel:
    provider: "Provider"
    spec: Model

    async def generate(
        self,
        messages: list["Message"],
        *,
        system_prompt: str = "",
        tools: list["ToolDefinition"] | None = None,
        options: "StreamOptions | None" = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        thinking: "ThinkingLevel | None" = None,
        thinking_budgets: "ThinkingBudgets | None" = None,
    ) -> "AssistantMessage":
        return await self.provider.generate(
            model=self.spec,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            options=options,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            thinking=thinking,
            thinking_budgets=thinking_budgets,
        )
```

The string annotations avoid forward-reference issues — `Provider`, `Message`, `ToolDefinition`, `StreamOptions`, `ThinkingLevel`, `ThinkingBudgets`, and `AssistantMessage` are all defined *below* `BoundModel` in this file. Keep `from __future__ import annotations` if it's already at the top (check line 1); if it is, plain (unstringed) annotations are also fine — prefer that for readability.

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/providers/test_bound_model_calls.py::test_bound_model_generate_forwards_to_provider -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add cubepi/providers/base.py tests/providers/test_bound_model_calls.py
git commit -m "feat(providers): add BoundModel.generate convenience method"
```

---

## Task 2: `BoundModel.stream` forwards to the bound provider

**Files:**
- Modify: `cubepi/providers/base.py` (the `BoundModel` class added in Task 1)
- Test: `tests/providers/test_bound_model_calls.py`

- [ ] **Step 1: Append the failing test**

Stream needs a real `MessageStream`, so use `FauxProvider` with a queued response (cheaper than reimplementing the stream protocol in the recording fake). Add `FauxProvider` + `faux_assistant_message` to the imports at the top of the test file.

```python
from cubepi.providers.faux import FauxProvider, faux_assistant_message


@pytest.mark.asyncio
async def test_bound_model_stream_forwards_to_provider() -> None:
    provider = FauxProvider(provider_id="faux")
    provider.set_responses([faux_assistant_message("hello")])
    bound = provider.model("faux-1")

    stream = await bound.stream(
        messages=[UserMessage(content=[TextContent(text="hi")])],
        system_prompt="be brief",
    )

    events: list[str] = []
    async for event in stream:
        events.append(event.type)
        if event.type in ("done", "error"):
            break
    result = await stream.result()

    assert "start" in events
    assert "done" in events
    assert result.model_id == "faux-1"
    assert result.provider_id == "faux"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/providers/test_bound_model_calls.py::test_bound_model_stream_forwards_to_provider -v
```

Expected: FAIL with `AttributeError: 'BoundModel' object has no attribute 'stream'`.

- [ ] **Step 3: Add `stream` to `BoundModel`**

Append to the class body added in Task 1 (above `generate`, to match the order in `Provider` protocol):

```python
    async def stream(
        self,
        messages: list["Message"],
        *,
        system_prompt: str = "",
        tools: list["ToolDefinition"] | None = None,
        options: "StreamOptions | None" = None,
    ) -> "MessageStream":
        return await self.provider.stream(
            model=self.spec,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            options=options,
        )
```

`MessageStream` is also defined below — use the same string-annotation strategy as in Task 1.

- [ ] **Step 4: Run both BoundModel tests + the existing bound-model test file**

```bash
uv run pytest tests/providers/test_bound_model_calls.py tests/providers/test_bound_model.py -v
```

Expected: PASS for all (`test_bound_model.py` exercises construction, the new file exercises calls).

- [ ] **Step 5: Run type check on providers/**

```bash
uv run mypy cubepi/providers/base.py
```

Expected: no new errors. If mypy complains about `MessageStream` forward refs and `from __future__ import annotations` is *not* at the top, add it.

- [ ] **Step 6: Commit**

```bash
git add cubepi/providers/base.py tests/providers/test_bound_model_calls.py
git commit -m "feat(providers): add BoundModel.stream convenience method"
```

---

## Task 3: Migrate compaction + `extra_llm_calls()` to `BoundModel` (atomic)

**Files (all flip in one commit):**
- Modify: `cubepi/middleware/compaction/summarizer.py` (function signature + body, imports)
- Modify: `cubepi/middleware/compaction/__init__.py` (`__init__`, `transform_context` call site, `extra_llm_calls`, imports)
- Modify: `cubepi/middleware/base.py:96-116` (`extra_llm_calls` signature + docstring, imports)
- Modify: `cubepi/tracing/recorder.py:286-299` (the unpacking loop, imports)
- Modify: `tests/middleware/compaction/test_summarizer.py` (three call sites at lines ~60, ~85, ~119)
- Modify: `tests/tracing/test_recorder.py` (three mocks at ~666, ~706, ~735 — per-test replacements, not one template)

**Why atomic:** The producer side (`Middleware.extra_llm_calls` return shape) and the consumer side (`cubepi/tracing/recorder.py` + the test mocks) are coupled. Splitting this into two commits leaves an intermediate state where either the recorder unpacks the wrong shape or `CompactionMiddleware` re-pairs split fields that have already been collapsed. One commit, one atomic flip, larger diff — accepted trade-off for skipping the awkward `BoundModel(provider=self._summary_provider, spec=self._summary_model)` transition.

- [ ] **Step 0: Confirm no third-party `extra_llm_calls` overrides exist**

```bash
grep -rn "extra_llm_calls" cubepi/ tests/ --include="*.py"
```

Expected hits: the base definition (`cubepi/middleware/base.py`), the compaction override (`cubepi/middleware/compaction/__init__.py`), the recorder consumer (`cubepi/tracing/recorder.py`), and four test references (`tests/tracing/test_recorder.py`). If anything else shows up, fold it into the migration before continuing.

- [ ] **Step 1: Baseline run**

```bash
uv run pytest tests/middleware/ tests/tracing/test_recorder.py tests/middleware/compaction/test_summarizer.py -v
```

Expected: PASS. Record the totals — these must all still pass after the refactor.

- [ ] **Step 2: Update `summarize()` to take a `BoundModel`**

In `cubepi/middleware/compaction/summarizer.py`, replace the import block (lines 6-14) and the function (lines 55-80):

```python
from cubepi.middleware.compaction.state import CompactionState, message_refs
from cubepi.providers.base import (
    BoundModel,
    Message,
    StreamOptions,
    TextContent,
    ToolCall,
    UserMessage,
)
# ...

async def summarize(
    *,
    model: BoundModel,
    messages_to_summarize: list[Message],
    existing: CompactionState | None,
    max_summary_tokens: int = 1024,
    abort_signal: asyncio.Event | None = None,
) -> CompactionState:
    system_prompt = SUMMARIZER_SYSTEM_PROMPT
    if existing and existing.summary:
        system_prompt += "\n\n" + EXISTING_SUMMARY_SUFFIX.format(prev=existing.summary)

    response = await model.generate(
        messages=[
            UserMessage(
                content=[TextContent(text=_format_transcript(messages_to_summarize))]
            )
        ],
        system_prompt=system_prompt,
        options=StreamOptions(signal=abort_signal),
        max_output_tokens=max_summary_tokens,
        temperature=0.0,
        thinking="off",
    )
```

Drop `Model` and `Provider` from the imports (now unused).

- [ ] **Step 3: Collapse `CompactionMiddleware` to one `BoundModel` field**

In `cubepi/middleware/compaction/__init__.py`, three changes in one pass.

**(a)** Replace `__init__`:

```python
    def __init__(
        self,
        *,
        summary_model: BoundModel,
        max_tokens_before_compact: int,
        keep_recent_messages: int = 8,
        max_summary_tokens: int = 1024,
        min_compact_messages: int = 4,
    ) -> None:
        self._summary_model = summary_model
        self._max_tokens_before = max_tokens_before_compact
        self._keep_recent = keep_recent_messages
        self._max_summary_tokens = max_summary_tokens
        self._min_compact = min_compact_messages
```

**(b)** Simplify the `summarize` call inside `transform_context`:

```python
        try:
            new_state = await summarize(
                model=self._summary_model,
                messages_to_summarize=messages[boundary:new_boundary],
                existing=state,
                max_summary_tokens=self._max_summary_tokens,
                abort_signal=signal,
            )
```

**(c)** Replace `extra_llm_calls`:

```python
    def extra_llm_calls(self) -> tuple[BoundModel, ...]:
        # Surface the bound summary model so ``cubepi.tracing.Recorder`` can
        # both subscribe its listeners (the summarizer's chat span lands in
        # the trace) AND identify the summary call by spec — important when
        # the summary model's provider is the same instance as the agent's
        # main provider, the common "reuse the client, swap the model"
        # pattern.
        return (self._summary_model,)
```

Drop `Model` and `Provider` from the imports at the top of this file.

- [ ] **Step 4: Update the `Middleware.extra_llm_calls` base signature + docstring**

In `cubepi/middleware/base.py`, adjust imports and replace lines 96–116:

```python
from cubepi.providers.base import AssistantMessage, BoundModel, Message  # drop Model, Provider
```

(Verify with `grep -n "Provider\b\|Model\b" cubepi/middleware/base.py` after editing — neither should remain in the file.)

```python
    def extra_llm_calls(self) -> Iterable[BoundModel]:
        """Declare LLM calls this middleware drives outside the agent's main
        bound model.

        Each entry is a ``BoundModel`` — the same handle the user gets from
        ``provider.model(...)``. ``cubepi.tracing.Recorder`` uses these to:

        * Subscribe listeners on any provider the recorder isn't already
          watching, so the resulting calls show up in the trace tree
          alongside the agent's own chat spans.
        * Identify middleware-owned calls by ``(spec.provider_id, spec.id)``
          so they don't overwrite the root ``invoke_agent`` span's
          attribution (provider name, system prompt hash, tool list). This
          model-based gate is what handles the common "reuse one provider
          client, swap the model" pattern — listener identity alone would
          attribute the middleware's first call to the agent.

        Default is empty — middlewares that do not call any LLM directly
        need not override.
        """
        return ()
```

- [ ] **Step 5: Update the recorder consumer**

In `cubepi/tracing/recorder.py`, around lines 286–299, the current unpacking loop is:

```python
for p, m in extra:
    key = (m.provider_id, m.id)
    if key != agent_key:
        self._extra_call_models.add(key)
    if id(p) in seen:
        continue
    seen.add(id(p))
    _subscribe(p)
```

Replace with:

```python
for bound in extra:
    spec = bound.spec
    key = (spec.provider_id, spec.id)
    if key != agent_key:
        self._extra_call_models.add(key)
    provider = bound.provider
    if id(provider) in seen:
        continue
    seen.add(id(provider))
    _subscribe(provider)
```

**Do not** add `BoundModel` to this file's imports. The loop body only uses attribute access (`bound.spec`, `bound.provider`) and no other code in `recorder.py` references the `BoundModel` type, so an import would be flagged by ruff as `F401`. (Verify with `grep -n "from cubepi.providers.base" cubepi/tracing/recorder.py` — only `StreamEvent`, `ToolCall`, `ToolResultMessage`, `UserMessage` are imported today; leave that list unchanged.)

- [ ] **Step 6: Migrate `tests/middleware/compaction/test_summarizer.py` to the new signature**

This file has three `summarize(provider=..., model=...)` call sites (around lines 60, 85, 119). All three wrap the provider + model in a `BoundModel`. Extend the imports:

```python
from cubepi.providers.base import (
    AssistantMessage,
    BoundModel,
    Message,
    Model,
    StreamOptions,
    TextContent,
    ToolCall,
    ToolDefinition,
    UserMessage,
)
```

Replace each `summarize(provider=provider, model=model, ...)` with:

```python
result = await summarize(
    model=BoundModel(provider=provider, spec=model),
    messages_to_summarize=[...],
    existing=None,
    max_summary_tokens=512,
    abort_signal=signal,
)
```

`_FakeProvider` (defined in the same file) only implements `generate` — that's fine. `BoundModel.generate` delegates to `self.provider.generate(model=self.spec, ...)`, so `_FakeProvider.generate` receives the same kwargs it does today, and the `provider.calls[0]["max_output_tokens"]` style assertions stay green. Don't change `_FakeProvider` or the assertions.

Do all three call sites in one edit.

- [ ] **Step 7: Update the three recorder test mocks**

In `tests/tracing/test_recorder.py`, three middleware classes return from `extra_llm_calls`. They are NOT interchangeable — each exercises a different recorder branch, so spell out the per-test replacement. `BoundModel` is already imported at line 22 of the test file.

**(a)** `_SameModelMiddleware` (around line 666): exercises the "same `(provider_id, id)` as the agent's bound model — must be excluded from `_extra_call_models`" branch. The mock stores `provider` and `model` on `self._p` / `self._m`.

```python
            def extra_llm_calls(self):
                return [BoundModel(provider=self._p, spec=self._m)]
```

**(b)** `_DuckMiddleware` (around line 706): exercises the "provider not derived from `BaseProvider` is skipped in `_subscribe`" branch. The provider MUST stay `_DuckProvider()` — using the agent's BaseProvider would silently kill the branch this test exists to cover. The model field is `self._m` from the middleware's constructor.

```python
            def extra_llm_calls(self):
                return [BoundModel(provider=_DuckProvider(), spec=self._m)]
```

`BoundModel` is a `@dataclass(frozen=True)`; even though `provider: Provider` is a `runtime_checkable` `Protocol`, dataclasses don't enforce protocol membership at construction, so `_DuckProvider` is accepted just like the current tuple form. The recorder's `_subscribe` check is still what skips it downstream.

**(c)** `_BoomMiddleware` (around line 735): raises before returning, so the return shape doesn't matter. **Do not change this class.**

- [ ] **Step 8: Run the touched suites, then the full sweep**

```bash
uv run pytest tests/middleware/ tests/middleware/compaction/test_summarizer.py tests/tracing/test_recorder.py -v
```

Expected: same PASS totals as Step 1. If anything fails, the producer/consumer didn't flip together — re-check Steps 2-7.

Specifically verify the branch-coverage tests still hit their intended branches:

```bash
uv run pytest tests/tracing/test_recorder.py -k "extra_llm_calls or extra_provider_not_baseprovider" -v
```

Then the full sweep + lint + types:

```bash
uv run pytest tests/ -x
uv run ruff check cubepi/ tests/
uv run ruff format --check cubepi/ tests/
uv run mypy cubepi
```

Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add cubepi/middleware/compaction/summarizer.py cubepi/middleware/compaction/__init__.py cubepi/middleware/base.py cubepi/tracing/recorder.py tests/middleware/compaction/test_summarizer.py tests/tracing/test_recorder.py
git commit -m "refactor(middleware): take and surface BoundModel through compaction + extra_llm_calls"
```

---

## Task 4: Document the new ergonomics

**Files:**
- Modify: `website/docs/agents/providers.md` (or the closest existing page; verify with `ls website/docs/agents/ website/docs/getting-started/`)

CLAUDE.md requires user-facing docs in the same PR. `BoundModel.generate` / `.stream` are now part of the public surface.

- [ ] **Step 1: Locate the providers doc**

```bash
grep -rln "provider.model(" website/docs/ | head -5
```

Open the page that introduces `provider.model("...")` (most likely `website/docs/agents/providers.md` or the getting-started page).

- [ ] **Step 2: Add a short subsection right after the `provider.model(...)` introduction**

Insert (adjust heading level to match neighbours):

```markdown
### Calling a bound model directly

`provider.model(...)` returns a `BoundModel` that you can invoke without
fishing the provider out again:

```python
bound = provider.model("claude-sonnet-4-6")

# Single-shot call.
reply = await bound.generate(
    messages=[UserMessage(content=[TextContent(text="hi")])],
    system_prompt="Be brief.",
)

# Streaming.
stream = await bound.stream(messages=[...])
async for event in stream:
    ...
```

Both methods forward to the bound provider with `model=bound.spec` — useful
for utilities (e.g. summarizers, classifiers) where you already hold a
`BoundModel` and want to skip the agent loop.
```

- [ ] **Step 3: Verify docs build**

```bash
cd website && pnpm typecheck && cd -
```

Expected: PASS. (If `pnpm` isn't set up locally, skip; CI will catch.)

- [ ] **Step 4: Commit**

```bash
git add website/docs/
git commit -m "docs(providers): document BoundModel.generate/stream"
```

---

## Task 5 (gated): Local codex review loop

**Per CLAUDE.md, ask the user before starting this loop.** Do not enter it autonomously.

- [ ] **Step 1: Ask the user**

> "Code is done and tests green. Want me to run the `codex:rescue` local review loop now?"

- [ ] **Step 2: If yes, run the loop**

Use the `codex:rescue` subagent to review the diff. Iterate until codex reports no remaining issues. Push fixups as separate commits — do not amend.

- [ ] **Step 3: Open the PR**

```bash
git push -u origin 2026-06-08-boundmodel-methods
gh pr create --title "feat(providers): BoundModel.generate/stream + middleware migration" --body "$(cat <<'EOF'
## Summary
- Add `BoundModel.generate` / `BoundModel.stream` convenience methods
- `CompactionMiddleware` summarizer takes `BoundModel` instead of `(Provider, Model)`
- `Middleware.extra_llm_calls()` returns `Iterable[BoundModel]`; tracing recorder + tests adapted
- Doc note added on the providers page

## Test plan
- [x] `uv run pytest tests/`
- [x] `uv run ruff check cubepi/ tests/`
- [x] `uv run ruff format --check cubepi/ tests/`
- [x] `uv run mypy cubepi`
EOF
)"
```

- [ ] **Step 4: Enter the PR codex review loop**

Poll `gh api repos/cubeplexai/cubepi/pulls/<#>/comments` every ~2 minutes, resolve feedback, reply `@codex review again` on the PR after pushing fixes. Repeat until clean and CI is green. Then merge.

---

## Self-Review (run before handing off)

1. **Spec coverage** — every item from the conversation is mapped to a task:
   - "Add `BoundModel.generate/stream`" → Task 1, Task 2.
   - "`summarize()` takes `BoundModel`" → Task 3 Step 2.
   - "`extra_llm_calls()` returns `Iterable[BoundModel]`" → Task 3 Step 4.
   - Recorder consumer + test mocks → Task 3 Steps 5, 7.
   - Compaction field collapse + summarizer test migration → Task 3 Steps 3, 6.
   - Docs requirement (CLAUDE.md) → Task 4.
2. **Placeholders** — none. Every code step has the actual code; commands are concrete.
3. **Type consistency** — `BoundModel.generate` / `BoundModel.stream` signatures mirror `Provider.generate` / `Provider.stream` exactly (verified against `cubepi/providers/base.py:579-603`). `self._summary_model: BoundModel` is the single field name used throughout Task 3.
4. **Atomicity** — Task 3 producer-side (compaction `__init__` / `extra_llm_calls` return shape) and consumer-side (recorder loop + test mocks) flip in one commit. No intermediate broken state.

## Follow-ups (out of scope)

- `Agent.__init__` still does `self._provider = model.provider` then passes them separately into `cubepi/agent/loop.py`. The loop module's `provider=` / `model=` signature is wide enough that converting it deserves its own plan. Track separately.
