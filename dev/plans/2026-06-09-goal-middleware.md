# Goal Middleware Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tool_choice` to providers, `BoundModel.generate_structured()` for tool-based structured output, then build `GoalMiddleware` that lets an agent work autonomously until an evaluator model confirms a `/goal` condition is met.

**Architecture:** Four layers of changes: (1) `ToolChoice` type and `tool_choice` parameter threaded through `Provider` protocol → all four providers. (2) `generate_structured()` on `BoundModel` — injects a synthetic tool with `tool_choice` forcing, validates via Pydantic, retries on validation failure. (3) Remove `_reflection_fired` guard in `_run_loop` so `on_run_end` can fire multiple times. (4) `GoalMiddleware` — detects `/goal` prefix in `transform_context`, evaluates via `on_run_end` using the evaluator model.

**Tech Stack:** Python, Pydantic, cubepi providers (FauxProvider for tests), asyncio.

---

### Task 1: `tool_choice` on Providers — Tests

**Files:**
- Create: `tests/providers/test_tool_choice.py`

- [ ] **Step 1: Write tests for tool_choice wire format**

```python
"""Tests for tool_choice support across providers."""

from __future__ import annotations

import pytest

from cubepi.providers.base import ToolChoice, ToolDefinition


# -- Anthropic mapping --

def test_anthropic_tool_choice_auto() -> None:
    from cubepi.providers.anthropic import AnthropicProvider

    result = AnthropicProvider._map_tool_choice("auto")
    assert result == {"type": "auto"}


def test_anthropic_tool_choice_required() -> None:
    from cubepi.providers.anthropic import AnthropicProvider

    result = AnthropicProvider._map_tool_choice("required")
    assert result == {"type": "any"}


def test_anthropic_tool_choice_none() -> None:
    from cubepi.providers.anthropic import AnthropicProvider

    result = AnthropicProvider._map_tool_choice("none")
    assert result is None


def test_anthropic_tool_choice_specific_name() -> None:
    from cubepi.providers.anthropic import AnthropicProvider

    result = AnthropicProvider._map_tool_choice("structured_output")
    assert result == {"type": "tool", "name": "structured_output"}


# -- OpenAI mapping --

def test_openai_tool_choice_auto() -> None:
    from cubepi.providers.openai import OpenAIProvider

    result = OpenAIProvider._map_tool_choice("auto")
    assert result == "auto"


def test_openai_tool_choice_required() -> None:
    from cubepi.providers.openai import OpenAIProvider

    result = OpenAIProvider._map_tool_choice("required")
    assert result == "required"


def test_openai_tool_choice_none() -> None:
    from cubepi.providers.openai import OpenAIProvider

    result = OpenAIProvider._map_tool_choice("none")
    assert result == "none"


def test_openai_tool_choice_specific_name() -> None:
    from cubepi.providers.openai import OpenAIProvider

    result = OpenAIProvider._map_tool_choice("structured_output")
    assert result == {
        "type": "function",
        "function": {"name": "structured_output"},
    }


# -- OpenAI Responses mapping --

def test_openai_responses_tool_choice_required() -> None:
    from cubepi.providers.openai_responses import OpenAIResponsesProvider

    result = OpenAIResponsesProvider._map_tool_choice("required")
    assert result == "required"


def test_openai_responses_tool_choice_specific_name() -> None:
    from cubepi.providers.openai_responses import OpenAIResponsesProvider

    result = OpenAIResponsesProvider._map_tool_choice("structured_output")
    assert result == {
        "type": "function",
        "name": "structured_output",
    }


# -- FauxProvider accepts but ignores --

def test_faux_accepts_tool_choice() -> None:
    from cubepi.providers.faux import FauxProvider, faux_assistant_message

    provider = FauxProvider(provider_id="faux")
    provider.set_responses([faux_assistant_message("ok")])

    # Should not raise — tool_choice is accepted and ignored
    import asyncio
    from cubepi.providers.base import TextContent, UserMessage

    async def _run() -> None:
        model = provider.model("test")
        await model.generate(
            [UserMessage(content=[TextContent(text="hi")])],
            tool_choice="required",
        )

    asyncio.run(_run())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware && uv run pytest tests/providers/test_tool_choice.py -v`
Expected: FAIL — `ToolChoice`, `_map_tool_choice` not defined, `tool_choice` parameter not accepted.

- [ ] **Step 3: Commit test file**

```bash
cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware
git add tests/providers/test_tool_choice.py
git commit -m "test: add failing tests for tool_choice across providers"
```

---

### Task 2: `tool_choice` on Providers — Implementation

**Files:**
- Modify: `cubepi/providers/base.py` (add `ToolChoice` type, update `Provider` protocol, `BoundModel`, `BaseProvider`)
- Modify: `cubepi/providers/anthropic.py` (add `_map_tool_choice`, wire into `stream()`)
- Modify: `cubepi/providers/openai.py` (add `_map_tool_choice`, wire into `stream()`)
- Modify: `cubepi/providers/openai_responses.py` (add `_map_tool_choice`, wire into `stream()`)
- Modify: `cubepi/providers/faux.py` (accept `tool_choice` param in `stream()`)

- [ ] **Step 1: Add `ToolChoice` type to `base.py`**

After the `ThinkingLevel` line (line 21), add:

```python
ToolChoice = Literal["auto", "required", "none"] | str
```

- [ ] **Step 2: Add `tool_choice` to `BoundModel.stream()` and `BoundModel.generate()`**

In `BoundModel.stream()` (line 98-118), add `tool_choice` parameter and forward it:

```python
    async def stream(
        self,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        return await self.provider.stream(
            self.spec,
            messages,
            system_prompt=system_prompt,
            tools=tools,
            tool_choice=tool_choice,
            options=options,
        )
```

In `BoundModel.generate()` (line 120-143), add `tool_choice` parameter and forward it:

```python
    async def generate(
        self,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: StreamOptions | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        thinking: ThinkingLevel | None = None,
        thinking_budgets: ThinkingBudgets | None = None,
    ) -> AssistantMessage:
        return await self.provider.generate(
            self.spec,
            messages,
            system_prompt=system_prompt,
            tools=tools,
            tool_choice=tool_choice,
            options=options,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            thinking=thinking,
            thinking_budgets=thinking_budgets,
        )
```

- [ ] **Step 3: Add `tool_choice` to `Provider` protocol and `BaseProvider`**

In the `Provider` protocol `stream()` (line 628-636):

```python
    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream: ...
```

In the `Provider` protocol `generate()` (line 638-650):

```python
    async def generate(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: StreamOptions | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        thinking: ThinkingLevel | None = None,
        thinking_budgets: ThinkingBudgets | None = None,
    ) -> AssistantMessage: ...
```

In `BaseProvider.stream()` (line 706-715):

```python
    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        raise NotImplementedError
```

In `BaseProvider.generate()` (line 717+), add `tool_choice` parameter and forward it to `self.stream()`:

```python
    async def generate(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: StreamOptions | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        thinking: ThinkingLevel | None = None,
        thinking_budgets: ThinkingBudgets | None = None,
    ) -> AssistantMessage:
```

And in the body where it calls `self.stream(...)`, add `tool_choice=tool_choice`.

- [ ] **Step 4: Wire `tool_choice` into `AnthropicProvider`**

Add static method and update `stream()` in `cubepi/providers/anthropic.py`:

```python
    @staticmethod
    def _map_tool_choice(choice: str) -> dict[str, str] | None:
        if choice == "auto":
            return {"type": "auto"}
        if choice == "required":
            return {"type": "any"}
        if choice == "none":
            return None
        return {"type": "tool", "name": choice}
```

In `stream()` signature, add `tool_choice: ToolChoice | None = None` after `tools`.

In the body, after the `if tools:` block (after line 187), add:

```python
        if tool_choice is not None and tool_choice != "none":
            mapped = self._map_tool_choice(tool_choice)
            if mapped is not None:
                kwargs["tool_choice"] = mapped
```

Import `ToolChoice` at top of file.

- [ ] **Step 5: Wire `tool_choice` into `OpenAIProvider`**

Add static method and update `stream()` in `cubepi/providers/openai.py`:

```python
    @staticmethod
    def _map_tool_choice(choice: str) -> str | dict:
        if choice in ("auto", "required", "none"):
            return choice
        return {"type": "function", "function": {"name": choice}}
```

In `stream()` signature, add `tool_choice: ToolChoice | None = None` after `tools`.

In the body, after the `if tools:` block (after line 107), add:

```python
        if tool_choice is not None:
            kwargs["tool_choice"] = self._map_tool_choice(tool_choice)
```

Import `ToolChoice` at top of file.

- [ ] **Step 6: Wire `tool_choice` into `OpenAIResponsesProvider`**

Add static method and update `stream()` in `cubepi/providers/openai_responses.py`:

```python
    @staticmethod
    def _map_tool_choice(choice: str) -> str | dict:
        if choice in ("auto", "required", "none"):
            return choice
        return {"type": "function", "name": choice}
```

Note: OpenAI Responses uses `{"type": "function", "name": ...}` (no nested `"function"` key), unlike the Chat Completions API.

In `stream()` signature, add `tool_choice: ToolChoice | None = None` after `tools`.

In the body, after the `if tools:` block (after line 128), add:

```python
        if tool_choice is not None:
            kwargs["tool_choice"] = self._map_tool_choice(tool_choice)
```

Import `ToolChoice` at top of file.

- [ ] **Step 7: Accept `tool_choice` in `FauxProvider`**

In `cubepi/providers/faux.py`, update `stream()` signature to accept `tool_choice`:

```python
    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
```

No body changes — FauxProvider returns scripted responses regardless of tool_choice. Import `ToolChoice` at top.

- [ ] **Step 8: Run tool_choice tests**

Run: `cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware && uv run pytest tests/providers/test_tool_choice.py -v`
Expected: All tests pass.

- [ ] **Step 9: Run full test suite for regressions**

Run: `cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware && uv run pytest tests/ -x -q`
Expected: All tests pass.

- [ ] **Step 10: Commit**

```bash
cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware
git add cubepi/providers/base.py cubepi/providers/anthropic.py cubepi/providers/openai.py cubepi/providers/openai_responses.py cubepi/providers/faux.py
git commit -m "feat(providers): add tool_choice parameter to Provider protocol and all implementations"
```

---

### Task 3: `BoundModel.generate_structured()` — Tests

**Files:**
- Create: `tests/test_structured_output.py`

- [ ] **Step 1: Write test — happy path returns validated Pydantic model**

```python
"""Tests for BoundModel.generate_structured()."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from cubepi.providers.base import (
    StructuredOutputError,
    TextContent,
    UserMessage,
)
from cubepi.providers.faux import FauxProvider, faux_assistant_message, faux_tool_call


class MovieReview(BaseModel):
    title: str
    rating: int
    summary: str


@pytest.mark.asyncio
async def test_generate_structured_happy_path() -> None:
    provider = FauxProvider(provider_id="faux")
    provider.set_responses([
        faux_assistant_message(
            faux_tool_call(
                "structured_output",
                {"title": "Inception", "rating": 9, "summary": "Great film"},
            )
        ),
    ])
    model = provider.model("test")

    result = await model.generate_structured(
        MovieReview,
        messages=[UserMessage(content=[TextContent(text="review Inception")])],
    )

    assert isinstance(result, MovieReview)
    assert result.title == "Inception"
    assert result.rating == 9
```

- [ ] **Step 2: Write test — raises when model returns text instead of tool call**

```python
@pytest.mark.asyncio
async def test_generate_structured_no_tool_call_raises() -> None:
    provider = FauxProvider(provider_id="faux")
    provider.set_responses([faux_assistant_message("just text, no tool call")])
    model = provider.model("test")

    with pytest.raises(StructuredOutputError, match="no tool call"):
        await model.generate_structured(
            MovieReview,
            messages=[UserMessage(content=[TextContent(text="review")])],
        )
```

- [ ] **Step 3: Write test — raises on Pydantic validation failure (after retries exhausted)**

```python
@pytest.mark.asyncio
async def test_generate_structured_validation_error_raises() -> None:
    provider = FauxProvider(provider_id="faux")
    # First call: bad data. Retry: bad data again. Both fail validation.
    provider.set_responses([
        faux_assistant_message(
            faux_tool_call(
                "structured_output",
                {"title": "X", "rating": "not-a-number", "summary": "bad"},
            )
        ),
        faux_assistant_message(
            faux_tool_call(
                "structured_output",
                {"title": "X", "rating": "still-bad", "summary": "bad"},
            )
        ),
    ])
    model = provider.model("test")

    with pytest.raises(StructuredOutputError, match="validation"):
        await model.generate_structured(
            MovieReview,
            messages=[UserMessage(content=[TextContent(text="review")])],
            max_retries=1,
        )
```

- [ ] **Step 4: Write test — custom tool_name is forwarded**

```python
@pytest.mark.asyncio
async def test_generate_structured_custom_tool_name() -> None:
    provider = FauxProvider(provider_id="faux")
    provider.set_responses([
        faux_assistant_message(
            faux_tool_call(
                "my_output",
                {"title": "X", "rating": 5, "summary": "ok"},
            )
        ),
    ])
    model = provider.model("test")

    result = await model.generate_structured(
        MovieReview,
        messages=[UserMessage(content=[TextContent(text="review")])],
        tool_name="my_output",
    )

    assert result.title == "X"
```

- [ ] **Step 5: Write test — retry succeeds on second attempt**

```python
@pytest.mark.asyncio
async def test_generate_structured_retry_succeeds() -> None:
    """First attempt has invalid data, retry produces valid data."""
    provider = FauxProvider(provider_id="faux")
    provider.set_responses([
        faux_assistant_message(
            faux_tool_call(
                "structured_output",
                {"title": "X", "rating": "bad", "summary": "bad"},
            )
        ),
        faux_assistant_message(
            faux_tool_call(
                "structured_output",
                {"title": "X", "rating": 5, "summary": "ok"},
            )
        ),
    ])
    model = provider.model("test")

    result = await model.generate_structured(
        MovieReview,
        messages=[UserMessage(content=[TextContent(text="review")])],
        max_retries=1,
    )

    assert result.rating == 5
    assert provider.call_count == 2
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware && uv run pytest tests/test_structured_output.py -v`
Expected: FAIL — `StructuredOutputError` and `generate_structured` not defined.

- [ ] **Step 7: Commit test file**

```bash
cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware
git add tests/test_structured_output.py
git commit -m "test: add failing tests for BoundModel.generate_structured()"
```

---

### Task 4: `BoundModel.generate_structured()` — Implementation

**Files:**
- Modify: `cubepi/providers/base.py` (add `StructuredOutputError` class, `BaseModelT` TypeVar, and `generate_structured` method to `BoundModel`)

- [ ] **Step 1: Add `BaseModelT` TypeVar and `StructuredOutputError`**

Near the top of `base.py`, add to the `typing` import:

```python
from typing import TypeVar
```

Below the imports, before the first class definition:

```python
BaseModelT = TypeVar("BaseModelT", bound=BaseModel)
```

Right before the `BoundModel` dataclass (around line 92), add:

```python
class StructuredOutputError(Exception):
    """Raised when generate_structured() cannot extract or validate the output."""
```

- [ ] **Step 2: Add `generate_structured` method to `BoundModel`**

Add this method after the existing `generate()` method on `BoundModel`:

```python
    async def generate_structured(
        self,
        output_type: type[BaseModelT],
        messages: list[Message],
        *,
        system_prompt: str = "",
        tool_name: str = "structured_output",
        tool_description: str = "Return the structured output",
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        max_retries: int = 1,
    ) -> BaseModelT:
        schema = output_type.model_json_schema()
        tool = ToolDefinition(
            name=tool_name,
            description=tool_description,
            parameters=schema,
        )
        default_hint = (
            f"You MUST respond by calling the '{tool_name}' tool. "
            "Do NOT respond with plain text."
        )
        full_system = f"{system_prompt}\n\n{default_hint}".strip() if system_prompt else default_hint

        attempt_messages = list(messages)
        last_error: Exception | None = None

        for _ in range(1 + max_retries):
            response = await self.generate(
                attempt_messages,
                system_prompt=full_system,
                tools=[tool],
                tool_choice=tool_name,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            )
            for block in response.content:
                if isinstance(block, ToolCall) and block.name == tool_name:
                    try:
                        return output_type.model_validate(block.arguments)
                    except Exception as exc:
                        last_error = exc
                        attempt_messages = [
                            *attempt_messages,
                            response,
                            UserMessage(
                                content=[
                                    TextContent(
                                        text=f"Validation error: {exc}. Fix the data and try again."
                                    )
                                ]
                            ),
                        ]
                        break
            else:
                raise StructuredOutputError(
                    f"Model returned no tool call for '{tool_name}'; "
                    f"got stop_reason={response.stop_reason!r}"
                )

        raise StructuredOutputError(
            f"Structured output validation failed after retries: {last_error}"
        ) from last_error
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware && uv run pytest tests/test_structured_output.py -v`
Expected: 5 passed.

- [ ] **Step 4: Run full test suite for regressions**

Run: `cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware && uv run pytest tests/ -x -q`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware
git add cubepi/providers/base.py
git commit -m "feat(providers): add BoundModel.generate_structured() with tool_choice forcing and retry"
```

---

### Task 5: `on_run_end` Multi-Fire — Tests

**Files:**
- Modify: `tests/middleware/test_on_run_end.py`

The existing test `test_on_run_end_fires_exactly_once` asserts `fire_count == 1`. After the loop change, `on_run_end` should fire on every outer-loop iteration. A middleware that returns messages N times then returns `None` should fire N+1 times total (N injections + 1 final None).

- [ ] **Step 1: Replace `test_on_run_end_fires_exactly_once` with a multi-fire test**

Replace lines 113-139 of `tests/middleware/test_on_run_end.py`:

```python
@pytest.mark.asyncio
async def test_on_run_end_fires_multiple_times() -> None:
    """on_run_end fires on each outer-loop iteration, not just once."""
    provider = FauxProvider(provider_id="faux")
    # 1 main + 2 continuation + 1 final = 4 model calls
    provider.set_responses(
        [
            faux_assistant_message("main"),
            faux_assistant_message("cont-1"),
            faux_assistant_message("cont-2"),
            faux_assistant_message("final"),
        ]
    )

    fire_count = 0

    class _FireThrice(Middleware):
        async def on_run_end(self, ctx, *, signal=None):
            nonlocal fire_count
            fire_count += 1
            if fire_count <= 3:
                return [UserMessage(content=[TextContent(text="continue")])]
            return None

    agent = Agent(
        model=provider.model("test"),
        middleware=[_FireThrice()],
    )
    await agent.prompt("hi")

    assert fire_count == 4  # 3 injections + 1 final check returning None
    assert provider.call_count == 4
```

- [ ] **Step 2: Run the new test to verify it fails (pre-change)**

Run: `cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware && uv run pytest tests/middleware/test_on_run_end.py::test_on_run_end_fires_multiple_times -v`
Expected: FAIL — currently `on_run_end` fires only once due to `_reflection_fired` guard.

- [ ] **Step 3: Commit test change**

```bash
cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware
git add tests/middleware/test_on_run_end.py
git commit -m "test: replace single-fire test with multi-fire on_run_end test"
```

---

### Task 6: `on_run_end` Multi-Fire — Implementation

**Files:**
- Modify: `cubepi/agent/loop.py:464-679` (the `_run_loop` function)

- [ ] **Step 1: Remove `_reflection_fired` variable and guard**

In `cubepi/agent/loop.py`, inside `_run_loop`:

1. Delete line 467: `_reflection_fired = False`
2. Replace lines 664-678:

Before:
```python
        # on_run_end fires exactly once per prompt() call, after all normal
        # turns and follow-ups are drained. _reflection_fired prevents the
        # reflection pass itself from triggering another reflection.
        # Skipped for error/aborted runs (those return early before reaching here).
        if on_run_end and not _reflection_fired:
            _reflection_fired = True
            inject = await on_run_end(current_context, signal=opts.signal)
            if inject:
                for msg in inject:
                    await emit_event(emit, MessageStartEvent(message=msg))
                    await emit_event(emit, MessageEndEvent(message=msg))
                    current_context.messages.append(msg)
                    new_messages.append(msg)
                first_turn = False
                continue
```

After:
```python
        if on_run_end:
            inject = await on_run_end(current_context, signal=opts.signal)
            if inject:
                for msg in inject:
                    await emit_event(emit, MessageStartEvent(message=msg))
                    await emit_event(emit, MessageEndEvent(message=msg))
                    current_context.messages.append(msg)
                    new_messages.append(msg)
                first_turn = False
                continue
```

- [ ] **Step 2: Run the multi-fire test**

Run: `cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware && uv run pytest tests/middleware/test_on_run_end.py -v`
Expected: All tests pass, including `test_on_run_end_fires_multiple_times`.

- [ ] **Step 3: Run full test suite for regressions**

Run: `cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware && uv run pytest tests/ -x -q`
Expected: All tests pass. Existing `on_run_end` tests that return messages once still work because the middleware returns `None` on subsequent calls (they only inject once by design).

- [ ] **Step 4: Commit**

```bash
cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware
git add cubepi/agent/loop.py
git commit -m "feat(loop): allow on_run_end to fire on every outer-loop iteration"
```

---

### Task 7: `GoalMiddleware` — Tests

**Files:**
- Create: `tests/test_goal.py`

- [ ] **Step 1: Write test — no /goal prefix, middleware is transparent**

```python
"""Tests for GoalMiddleware."""

from __future__ import annotations

import pytest

from cubepi import Agent
from cubepi.middleware.goal import GoalMiddleware
from cubepi.providers.base import TextContent, UserMessage
from cubepi.providers.faux import FauxProvider, faux_assistant_message, faux_tool_call


@pytest.mark.asyncio
async def test_no_goal_prefix_transparent() -> None:
    """Without /goal prefix, middleware does nothing."""
    worker = FauxProvider(provider_id="worker")
    worker.set_responses([faux_assistant_message("done")])

    evaluator_provider = FauxProvider(provider_id="evaluator")
    evaluator_provider.set_responses([])

    goal = GoalMiddleware(
        evaluator=evaluator_provider.model("eval"),
        max_evaluations=10,
    )
    agent = Agent(
        model=worker.model("work"),
        middleware=[goal],
    )
    await agent.prompt("fix the bug")

    assert worker.call_count == 1
    assert evaluator_provider.call_count == 0
```

- [ ] **Step 2: Write test — goal achieved on first evaluation**

```python
@pytest.mark.asyncio
async def test_goal_achieved_first_eval() -> None:
    """Worker finishes, evaluator says achieved=True, loop stops."""
    worker = FauxProvider(provider_id="worker")
    worker.set_responses([faux_assistant_message("all tests pass now")])

    evaluator_provider = FauxProvider(provider_id="evaluator")
    evaluator_provider.set_responses([
        faux_assistant_message(
            faux_tool_call(
                "structured_output",
                {"achieved": True, "reason": "All tests passing"},
            )
        ),
    ])

    goal = GoalMiddleware(
        evaluator=evaluator_provider.model("eval"),
        max_evaluations=10,
    )
    agent = Agent(
        model=worker.model("work"),
        middleware=[goal],
    )
    await agent.prompt("/goal all tests pass")

    assert worker.call_count == 1
    assert evaluator_provider.call_count == 1
    assert agent.state.extra["goal"]["status"] == "achieved"
    assert agent.state.extra["goal"]["evaluations"] == 1
```

- [ ] **Step 3: Write test — goal not achieved, worker retries, then achieved**

```python
@pytest.mark.asyncio
async def test_goal_retry_then_achieved() -> None:
    """Evaluator says no, worker gets feedback and retries, then achieved."""
    worker = FauxProvider(provider_id="worker")
    worker.set_responses([
        faux_assistant_message("first attempt"),
        faux_assistant_message("fixed it"),
    ])

    evaluator_provider = FauxProvider(provider_id="evaluator")
    evaluator_provider.set_responses([
        faux_assistant_message(
            faux_tool_call(
                "structured_output",
                {"achieved": False, "reason": "2 tests still failing"},
            )
        ),
        faux_assistant_message(
            faux_tool_call(
                "structured_output",
                {"achieved": True, "reason": "All tests pass"},
            )
        ),
    ])

    goal = GoalMiddleware(
        evaluator=evaluator_provider.model("eval"),
        max_evaluations=10,
    )
    agent = Agent(
        model=worker.model("work"),
        middleware=[goal],
    )
    await agent.prompt("/goal all tests pass")

    assert worker.call_count == 2
    assert evaluator_provider.call_count == 2
    assert agent.state.extra["goal"]["status"] == "achieved"
    assert agent.state.extra["goal"]["evaluations"] == 2
```

- [ ] **Step 4: Write test — max_evaluations exhausted**

```python
@pytest.mark.asyncio
async def test_goal_max_evaluations_exhausted() -> None:
    """Evaluator keeps saying no until max_evaluations hit."""
    worker = FauxProvider(provider_id="worker")
    worker.set_responses([
        faux_assistant_message(f"attempt {i}") for i in range(3)
    ])

    evaluator_provider = FauxProvider(provider_id="evaluator")
    evaluator_provider.set_responses([
        faux_assistant_message(
            faux_tool_call(
                "structured_output",
                {"achieved": False, "reason": "still broken"},
            )
        )
        for _ in range(2)
    ])

    goal = GoalMiddleware(
        evaluator=evaluator_provider.model("eval"),
        max_evaluations=2,
    )
    agent = Agent(
        model=worker.model("work"),
        middleware=[goal],
    )
    await agent.prompt("/goal all tests pass")

    assert agent.state.extra["goal"]["status"] == "exhausted"
    assert agent.state.extra["goal"]["evaluations"] == 2
```

- [ ] **Step 5: Write test — /goal prefix is stripped from worker's message**

```python
@pytest.mark.asyncio
async def test_goal_prefix_stripped_from_message() -> None:
    """/goal prefix is removed; worker sees the condition as a work directive."""
    worker = FauxProvider(provider_id="worker")
    worker.set_responses([faux_assistant_message("done")])

    evaluator_provider = FauxProvider(provider_id="evaluator")
    evaluator_provider.set_responses([
        faux_assistant_message(
            faux_tool_call(
                "structured_output",
                {"achieved": True, "reason": "done"},
            )
        ),
    ])

    goal = GoalMiddleware(
        evaluator=evaluator_provider.model("eval"),
        max_evaluations=10,
    )
    agent = Agent(
        model=worker.model("work"),
        middleware=[goal],
    )
    await agent.prompt("/goal make all tests green")

    first_user_msg = agent.state.messages[0]
    assert isinstance(first_user_msg, UserMessage)
    text = first_user_msg.content[0].text
    assert not text.startswith("/goal")
    assert "make all tests green" in text
```

- [ ] **Step 6: Write test — extra_llm_calls declares evaluator**

```python
def test_extra_llm_calls_declares_evaluator() -> None:
    provider = FauxProvider(provider_id="eval")
    evaluator = provider.model("eval-model")
    goal = GoalMiddleware(evaluator=evaluator, max_evaluations=5)

    extras = list(goal.extra_llm_calls())
    assert len(extras) == 1
    assert extras[0] is evaluator
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware && uv run pytest tests/test_goal.py -v`
Expected: FAIL — `GoalMiddleware` does not exist yet.

- [ ] **Step 8: Commit test file**

```bash
cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware
git add tests/test_goal.py
git commit -m "test: add failing tests for GoalMiddleware"
```

---

### Task 8: `GoalMiddleware` — Implementation

**Files:**
- Create: `cubepi/middleware/goal.py`
- Modify: `cubepi/middleware/__init__.py`

- [ ] **Step 1: Create `cubepi/middleware/goal.py`**

```python
"""GoalMiddleware — autonomous goal-driven agent runs.

Hooks used:
    transform_context — detect /goal prefix, extract condition, rewrite message
    on_run_end        — evaluate condition via evaluator model, inject feedback
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from cubepi.agent.types import AgentContext
from cubepi.middleware.base import Middleware
from cubepi.providers.base import (
    BoundModel,
    Message,
    TextContent,
    UserMessage,
)

_GOAL_PREFIX = "/goal "

_MESSAGE_WINDOW = 20


class _GoalResult(BaseModel):
    achieved: bool
    reason: str


_EVALUATOR_PROMPT = """\
You are evaluating whether an agent has achieved its goal.

Goal condition:
{condition}

Recent conversation (last {window} messages):
{messages}

Has the goal been achieved? Use the structured_output tool to respond."""


def _format_messages_for_eval(messages: list[Message], window: int) -> str:
    tail = messages[-window:]
    parts: list[str] = []
    for msg in tail:
        role = getattr(msg, "role", "unknown")
        texts: list[str] = []
        for block in getattr(msg, "content", []):
            if isinstance(block, TextContent):
                texts.append(block.text)
        if texts:
            parts.append(f"[{role}] {' '.join(texts)}")
    return "\n".join(parts)


class GoalMiddleware(Middleware):
    def __init__(
        self,
        evaluator: BoundModel,
        max_evaluations: int = 10,
    ) -> None:
        self._evaluator = evaluator
        self._max_evaluations = max_evaluations
        self._condition: str | None = None
        self._evaluations = 0

    def extra_llm_calls(self) -> Iterable[BoundModel]:
        return (self._evaluator,)

    async def transform_context(
        self,
        messages: list[Message],
        *,
        ctx: AgentContext,
        signal: asyncio.Event | None = None,
    ) -> list[Message]:
        if not messages:
            return messages
        last = messages[-1]
        if not isinstance(last, UserMessage):
            return messages
        if not last.content:
            return messages
        first_block = last.content[0]
        if not isinstance(first_block, TextContent):
            return messages
        if not first_block.text.startswith(_GOAL_PREFIX):
            return messages

        condition = first_block.text[len(_GOAL_PREFIX) :].strip()
        self._condition = condition
        self._evaluations = 0

        rewritten = last.model_copy(
            update={"content": [TextContent(text=condition)]}
        )
        return [*messages[:-1], rewritten]

    async def on_run_end(
        self,
        ctx: AgentContext,
        *,
        signal: asyncio.Event | None = None,
    ) -> list[Message] | None:
        if self._condition is None:
            return None

        self._evaluations += 1

        transcript = _format_messages_for_eval(ctx.messages, _MESSAGE_WINDOW)
        prompt = _EVALUATOR_PROMPT.format(
            condition=self._condition,
            window=_MESSAGE_WINDOW,
            messages=transcript,
        )

        result = await self._evaluator.generate_structured(
            _GoalResult,
            messages=[UserMessage(content=[TextContent(text=prompt)])],
            system_prompt="",
            temperature=0.0,
        )

        goal_state: dict[str, Any] = {
            "condition": self._condition,
            "evaluations": self._evaluations,
            "max_evaluations": self._max_evaluations,
            "last_reason": result.reason,
        }

        if result.achieved:
            goal_state["status"] = "achieved"
            ctx.extra["goal"] = goal_state
            return None

        if self._evaluations >= self._max_evaluations:
            goal_state["status"] = "exhausted"
            ctx.extra["goal"] = goal_state
            return None

        goal_state["status"] = "active"
        ctx.extra["goal"] = goal_state
        return [
            UserMessage(
                content=[
                    TextContent(
                        text=f"Goal not yet met: {result.reason}. Continue working."
                    )
                ]
            )
        ]
```

- [ ] **Step 2: Add `GoalMiddleware` to `cubepi/middleware/__init__.py`**

Add `"GoalMiddleware"` to the `__all__` list and add the lazy import entry to `_LAZY`:

In `__all__`, add:
```python
    "GoalMiddleware",
```

In `_LAZY`, add:
```python
    "GoalMiddleware": ("cubepi.middleware.goal", "GoalMiddleware"),
```

- [ ] **Step 3: Run goal tests**

Run: `cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware && uv run pytest tests/test_goal.py -v`
Expected: All 6 tests pass.

- [ ] **Step 4: Run full test suite**

Run: `cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware && uv run pytest tests/ -x -q`
Expected: All tests pass.

- [ ] **Step 5: Run type checks and linter**

Run: `cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware && uv run ruff check cubepi/middleware/goal.py cubepi/providers/base.py && uv run mypy cubepi/middleware/goal.py cubepi/providers/base.py`
Expected: Clean.

- [ ] **Step 6: Commit**

```bash
cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware
git add cubepi/middleware/goal.py cubepi/middleware/__init__.py
git commit -m "feat(middleware): add GoalMiddleware for autonomous goal-driven runs"
```

---

### Task 9: Final Verification

**Files:** (none — verification only)

- [ ] **Step 1: Run full test suite**

Run: `cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware && uv run pytest tests/ -x -q`
Expected: All tests pass (1613+ original + new tests).

- [ ] **Step 2: Run linter and type checker on all changed files**

Run:
```bash
cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware
uv run ruff check cubepi/ tests/
uv run ruff format --check cubepi/ tests/
uv run mypy cubepi
```
Expected: All clean.

- [ ] **Step 3: Review git log**

Run: `cd /home/chris/cubepi/.worktrees/2026-06-09-goal-middleware && git log --oneline main..HEAD`
Expected commits:
```
feat(middleware): add GoalMiddleware for autonomous goal-driven runs
test: add failing tests for GoalMiddleware
feat(loop): allow on_run_end to fire on every outer-loop iteration
test: replace single-fire test with multi-fire on_run_end test
feat(providers): add BoundModel.generate_structured() with tool_choice forcing and retry
test: add failing tests for BoundModel.generate_structured()
feat(providers): add tool_choice parameter to Provider protocol and all implementations
test: add failing tests for tool_choice across providers
```
