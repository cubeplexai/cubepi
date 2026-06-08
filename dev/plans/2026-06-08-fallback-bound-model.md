# FallbackBoundModel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `FallbackBoundModel` to cubepi — a standalone class that wraps an ordered chain of `BoundModel` instances and transparently tries the next model when the primary fails with a retriable error.

**Architecture:** `FallbackBoundModel` is a frozen dataclass in `cubepi/providers/fallback.py`. It is NOT a `BoundModel` subclass — it holds `chain: tuple[BoundModel, ...]` and exposes `provider`/`spec` properties pointing to `chain[0]`. `stream()` peeks at the first `StreamEvent`; if it is `error`, or if `stream()` raises a typed error in `trigger_errors`, the next model in the chain is tried. `generate()` uses plain exception-based retry. `Agent.__init__` type hint is widened to `BoundModel | FallbackBoundModel`.

**Tech Stack:** Python 3.11+, dataclasses, asyncio, loguru (with stdlib logging fallback), pytest-asyncio, cubepi `FauxProvider` + custom `_RaisingProvider` test helper.

**Worktree:** `/home/chris/cubepi/.worktrees/feat/fallback-bound-model`
**All commands run from:** `/home/chris/cubepi/.worktrees/feat/fallback-bound-model`

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `cubepi/providers/fallback.py` | `FallbackBoundModel`, `DEFAULT_TRIGGER_ERRORS` |
| Modify | `cubepi/providers/__init__.py` | Export `FallbackBoundModel`, `DEFAULT_TRIGGER_ERRORS` |
| Modify | `cubepi/__init__.py` | Export `FallbackBoundModel`, `DEFAULT_TRIGGER_ERRORS` |
| Modify | `cubepi/agent/agent.py` | Widen `model` type hint |
| Create | `tests/providers/test_fallback.py` | All 8 test scenarios |

---

## Task 1: Write failing tests

**Files:**
- Create: `tests/providers/test_fallback.py`

- [ ] **Step 1: Create the test file**

```python
# tests/providers/test_fallback.py
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from cubepi.errors import (
    ContextLengthExceeded,
    ProviderAuthFailed,
    ProviderBadRequest,
    ProviderError,
    ProviderUnavailable,
    RateLimited,
)
from cubepi.providers.base import (
    AssistantMessage,
    BaseProvider,
    BoundModel,
    Message,
    MessageStream,
    Model,
    StreamOptions,
    TextContent,
    ToolDefinition,
    Usage,
    UserMessage,
)
from cubepi.providers.faux import FauxProvider, faux_assistant_message
from cubepi.providers.fallback import DEFAULT_TRIGGER_ERRORS, FallbackBoundModel


class _RaisingProvider(BaseProvider):
    """Provider that raises a given exception unconditionally from stream() and generate()."""

    def __init__(self, error: ProviderError) -> None:
        super().__init__(provider_id=error.provider or "raising")
        self._error = error

    async def stream(
        self,
        model: Model,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        raise self._error

    async def generate(
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
        raise self._error


def _faux(provider_id: str = "faux", response: str | None = None) -> BoundModel:
    p = FauxProvider(provider_id=provider_id)
    if response is not None:
        p.set_responses([faux_assistant_message(response)])
    return p.model("model-1")


def _raising(error: ProviderError, model_id: str = "model-1") -> BoundModel:
    p = _RaisingProvider(error)
    return BoundModel(provider=p, spec=Model(id=model_id, provider_id=p.provider_id))


def _messages() -> list[Message]:
    return [UserMessage(content=[TextContent(text="hi")])]


# ---------------------------------------------------------------------------
# stream() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_primary_succeeds() -> None:
    """Primary succeeds — returns its stream, no failover."""
    primary = _faux("primary", "hello")
    fallback = _faux("fallback", "world")
    fbm = FallbackBoundModel(chain=(primary, fallback))

    stream = await fbm.stream(_messages())
    events = [ev.type async for ev in stream]
    result = await stream.result()

    assert "done" in events
    assert result.provider_id == "primary"
    # fallback provider was never used
    assert fallback.provider.call_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_stream_primary_raises_trigger_error_fallback_succeeds() -> None:
    """Primary raises RateLimited → failover to second model, on_failover called."""
    rate_err = RateLimited("429", provider="primary", model="model-1")
    primary = _raising(rate_err)
    fallback = _faux("fallback", "ok")

    failover_calls: list[tuple[BoundModel, BoundModel, Any]] = []

    async def _cb(failed: BoundModel, nxt: BoundModel, err: Any) -> None:
        failover_calls.append((failed, nxt, err))

    fbm = FallbackBoundModel(chain=(primary, fallback), on_failover=_cb)

    stream = await fbm.stream(_messages())
    result = await stream.result()

    assert result.provider_id == "fallback"
    assert len(failover_calls) == 1
    assert failover_calls[0][0] is primary
    assert failover_calls[0][1] is fallback
    assert isinstance(failover_calls[0][2], RateLimited)


@pytest.mark.asyncio
async def test_stream_primary_raises_non_trigger_error_reraises() -> None:
    """Primary raises ProviderBadRequest (not in trigger_errors) → re-raised, fallback not tried."""
    bad_req = ProviderBadRequest("400", provider="primary", model="model-1")
    primary = _raising(bad_req)
    fallback = _faux("fallback", "ok")

    fbm = FallbackBoundModel(chain=(primary, fallback))

    with pytest.raises(ProviderBadRequest):
        await fbm.stream(_messages())

    assert fallback.provider.call_count == 0  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_stream_primary_first_event_error_fallback_succeeds() -> None:
    """Primary emits error as first StreamEvent → fallback to second model."""
    # FauxProvider with no responses queued emits StreamEvent(type="error") as first event.
    primary_prov = FauxProvider(provider_id="primary")
    primary = primary_prov.model("model-1")
    fallback = _faux("fallback", "rescued")

    fbm = FallbackBoundModel(chain=(primary, fallback))

    stream = await fbm.stream(_messages())
    result = await stream.result()

    assert result.provider_id == "fallback"


@pytest.mark.asyncio
async def test_stream_all_exhausted_raises_provider_unavailable() -> None:
    """All models in chain fail → raises ProviderUnavailable."""
    err = RateLimited("429", provider="p", model="m")
    fbm = FallbackBoundModel(
        chain=(_raising(err, "m1"), _raising(err, "m2"), _raising(err, "m3"))
    )

    with pytest.raises(ProviderUnavailable, match="all providers exhausted"):
        await fbm.stream(_messages())


@pytest.mark.asyncio
async def test_stream_on_failover_callback_raises_is_swallowed() -> None:
    """on_failover callback that raises must not abort the failover."""
    rate_err = RateLimited("429", provider="primary", model="model-1")
    primary = _raising(rate_err)
    fallback = _faux("fallback", "ok")

    async def _bad_cb(failed: BoundModel, nxt: BoundModel, err: Any) -> None:
        raise RuntimeError("callback is broken")

    fbm = FallbackBoundModel(chain=(primary, fallback), on_failover=_bad_cb)

    stream = await fbm.stream(_messages())
    result = await stream.result()

    assert result.provider_id == "fallback"


# ---------------------------------------------------------------------------
# generate() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_primary_raises_trigger_error_fallback_succeeds() -> None:
    """generate() — primary raises RateLimited, fallback returns AssistantMessage."""
    rate_err = RateLimited("429", provider="primary", model="model-1")
    primary = _raising(rate_err)
    fallback = _faux("fallback", "generated")

    fbm = FallbackBoundModel(chain=(primary, fallback))

    result = await fbm.generate(_messages())

    assert result.provider_id == "fallback"


# ---------------------------------------------------------------------------
# Custom trigger_errors tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_trigger_errors_includes_auth_failed() -> None:
    """Custom trigger_errors that includes ProviderAuthFailed → auth failure triggers failover."""
    auth_err = ProviderAuthFailed("401", provider="primary", model="model-1")
    primary = _raising(auth_err)
    fallback = _faux("fallback", "ok")

    fbm = FallbackBoundModel(
        chain=(primary, fallback),
        trigger_errors=frozenset({ProviderAuthFailed}),
    )

    stream = await fbm.stream(_messages())
    result = await stream.result()

    assert result.provider_id == "fallback"
```

- [ ] **Step 2: Run tests — expect ImportError (module doesn't exist yet)**

```bash
uv run pytest tests/providers/test_fallback.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name 'FallbackBoundModel' from 'cubepi.providers.fallback'`
(or `ModuleNotFoundError: No module named 'cubepi.providers.fallback'`)

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/providers/test_fallback.py
git commit -m "test(fallback): failing tests for FallbackBoundModel"
```

---

## Task 2: Implement `FallbackBoundModel`

**Files:**
- Create: `cubepi/providers/fallback.py`

- [ ] **Step 1: Create the implementation file**

```python
# cubepi/providers/fallback.py
from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from cubepi.errors import ContextLengthExceeded, ProviderError, ProviderUnavailable, RateLimited
from cubepi.providers.base import (
    AssistantMessage,
    BoundModel,
    Message,
    MessageStream,
    Model,
    Provider,
    StreamEvent,
    StreamOptions,
    ThinkingBudgets,
    ThinkingLevel,
    ToolDefinition,
    Usage,
)

try:
    from loguru import logger as _log
except ImportError:  # pragma: no cover
    import logging as _logging

    _log = _logging.getLogger("cubepi.providers.fallback")  # type: ignore[assignment]


DEFAULT_TRIGGER_ERRORS: frozenset[type[ProviderError]] = frozenset(
    {RateLimited, ProviderUnavailable, ContextLengthExceeded}
)


@dataclass(frozen=True)
class FallbackBoundModel:
    """Ordered chain of BoundModels — tries each in turn on retriable errors.

    chain[0] is the primary model. On a trigger_errors exception or a first-event
    error from stream(), the next model in the chain is tried transparently.
    Mid-stream errors (after the first non-error event) are forwarded as-is.

    provider and spec proxy chain[0] so tracing/billing code that reads
    agent._model.provider / agent._model.spec continues to work unchanged.
    """

    chain: tuple[BoundModel, ...]
    trigger_errors: frozenset[type[ProviderError]] = DEFAULT_TRIGGER_ERRORS
    on_failover: (
        Callable[[BoundModel, BoundModel | None, BaseException | str], Awaitable[None] | None]
        | None
    ) = None

    @property
    def provider(self) -> Provider:
        return self.chain[0].provider

    @property
    def spec(self) -> Model:
        return self.chain[0].spec

    async def _notify(
        self,
        failed: BoundModel,
        next_model: BoundModel | None,
        error: BaseException | str,
        attempt: int,
    ) -> None:
        failed_label = f"{failed.spec.provider_id}/{failed.spec.id}"
        next_label = (
            f"{next_model.spec.provider_id}/{next_model.spec.id}"
            if next_model
            else "none (exhausted)"
        )
        _log.warning(
            "cubepi.providers.fallback: failover triggered  "
            "failed={}  →  next={}  reason={}  attempt={}/{}",
            failed_label,
            next_label,
            error,
            attempt,
            len(self.chain),
        )
        if self.on_failover is not None:
            try:
                result = self.on_failover(failed, next_model, error)
                if inspect.isawaitable(result):
                    await result
            except Exception as cb_exc:  # noqa: BLE001
                _log.warning(
                    "cubepi.providers.fallback: on_failover callback raised; swallowed: {}",
                    cb_exc,
                )

    async def stream(
        self,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
    ) -> MessageStream:
        last_error: BaseException | str = "no providers in chain"
        trigger = tuple(self.trigger_errors)

        for attempt, bound in enumerate(self.chain, start=1):
            next_bound = self.chain[attempt] if attempt < len(self.chain) else None

            try:
                inner = await bound.stream(
                    messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    options=options,
                )
            except trigger as exc:  # type: ignore[misc]
                last_error = exc
                await self._notify(bound, next_bound, exc, attempt)
                continue
            except Exception:
                raise

            iterator = inner.__aiter__()
            try:
                first = await iterator.__anext__()
            except StopAsyncIteration:
                last_error = "stream ended before producing any events"
                await self._notify(bound, next_bound, last_error, attempt)
                continue

            if first.type == "error":
                last_error = first.error_message or "stream error"
                await self._notify(bound, next_bound, last_error, attempt)
                continue

            outer = MessageStream()

            async def _forward(
                first_ev: StreamEvent = first,
                src: asyncio.coroutines = iterator,  # type: ignore[assignment]
                src_stream: MessageStream = inner,
                out: MessageStream = outer,
            ) -> None:
                try:
                    out.push(first_ev)
                    async for ev in src:  # type: ignore[union-attr]
                        out.push(ev)
                    out.set_result(await src_stream.result())
                except Exception as exc:  # noqa: BLE001
                    err_msg = AssistantMessage(
                        content=[],
                        stop_reason="error",
                        error_message=str(exc),
                        usage=Usage(),
                        timestamp=time.time(),
                    )
                    out.push(StreamEvent(type="error", error_message=str(exc)))
                    out.set_result(err_msg)

            outer.attach_task(asyncio.create_task(_forward()))
            return outer

        raise ProviderUnavailable(
            f"all providers exhausted; last error: {last_error!r}"
        )

    async def generate(
        self,
        messages: list[Message],
        *,
        system_prompt: str = "",
        tools: list[ToolDefinition] | None = None,
        options: StreamOptions | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        thinking: ThinkingLevel | None = None,
        thinking_budgets: ThinkingBudgets | None = None,
    ) -> AssistantMessage:
        last_error: BaseException | str = "no providers in chain"
        trigger = tuple(self.trigger_errors)

        for attempt, bound in enumerate(self.chain, start=1):
            next_bound = self.chain[attempt] if attempt < len(self.chain) else None

            try:
                return await bound.generate(
                    messages,
                    system_prompt=system_prompt,
                    tools=tools,
                    options=options,
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                    thinking=thinking,
                    thinking_budgets=thinking_budgets,
                )
            except trigger as exc:  # type: ignore[misc]
                last_error = exc
                await self._notify(bound, next_bound, exc, attempt)
                continue
            except Exception:
                raise

        raise ProviderUnavailable(
            f"all providers exhausted; last error: {last_error!r}"
        )
```

- [ ] **Step 2: Run tests — expect them to pass**

```bash
uv run pytest tests/providers/test_fallback.py -v
```

Expected: 8 passed

- [ ] **Step 3: Run mypy**

```bash
uv run mypy cubepi/providers/fallback.py
```

Expected: `Success: no issues found in 1 source file`

- [ ] **Step 4: Commit implementation**

```bash
git add cubepi/providers/fallback.py
git commit -m "feat(providers): add FallbackBoundModel"
```

---

## Task 3: Export from `cubepi/providers/__init__.py` and `cubepi/__init__.py`

**Files:**
- Modify: `cubepi/providers/__init__.py`
- Modify: `cubepi/__init__.py`

- [ ] **Step 1: Add import + `__all__` entries to `cubepi/providers/__init__.py`**

After the existing `from cubepi.providers.faux import (...)` block, add:

```python
from cubepi.providers.fallback import (
    DEFAULT_TRIGGER_ERRORS,
    FallbackBoundModel,
)
```

Add to `__all__`:

```python
    "DEFAULT_TRIGGER_ERRORS",
    "FallbackBoundModel",
```

(Insert alphabetically: `DEFAULT_TRIGGER_ERRORS` after `"Content"`, `FallbackBoundModel` after `"FauxProvider"`.)

- [ ] **Step 2: Add import + `__all__` entries to `cubepi/__init__.py`**

Add to the `from cubepi.providers import (...)` block:

```python
    DEFAULT_TRIGGER_ERRORS,
    FallbackBoundModel,
```

Add to `__all__`:

```python
    "DEFAULT_TRIGGER_ERRORS",
    "FallbackBoundModel",
```

(Insert `DEFAULT_TRIGGER_ERRORS` after `"ContextLengthExceeded"`, `FallbackBoundModel` after `"FauxProvider"` — but `FauxProvider` is not in `cubepi/__init__.py` currently, so place `FallbackBoundModel` after `"JsonValue"`.)

- [ ] **Step 3: Verify imports work**

```bash
uv run python -c "from cubepi import FallbackBoundModel, DEFAULT_TRIGGER_ERRORS; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Run full test suite — must still be clean**

```bash
uv run pytest tests/ -q --tb=short 2>&1 | tail -5
```

Expected: `1535 passed, 45 skipped` (or more if new tests added)

- [ ] **Step 5: Run mypy on both files**

```bash
uv run mypy cubepi/providers/__init__.py cubepi/__init__.py
```

Expected: `Success: no issues found in 2 source files`

- [ ] **Step 6: Commit**

```bash
git add cubepi/providers/__init__.py cubepi/__init__.py
git commit -m "feat(providers): export FallbackBoundModel from cubepi"
```

---

## Task 4: Widen `Agent` model type hint

**Files:**
- Modify: `cubepi/agent/agent.py`

- [ ] **Step 1: Add TYPE_CHECKING import and widen type hint**

`cubepi/agent/agent.py` has `from __future__ import annotations` at line 1 (lazy annotation evaluation) and `from typing import Callable, Generic, TypeVar` around line 8. Make two edits:

**Edit A** — add `TYPE_CHECKING` to the existing typing import:

```python
# before
from typing import Callable, Generic, TypeVar

# after
from typing import TYPE_CHECKING, Callable, Generic, TypeVar
```

**Edit B** — add a `TYPE_CHECKING` block after all existing imports (just before the `class AgentState` or `class Agent` definition):

```python
if TYPE_CHECKING:
    from cubepi.providers.fallback import FallbackBoundModel
```

**Edit C** — in `Agent.__init__`, change:

```python
model: BoundModel,
```

to:

```python
model: BoundModel | FallbackBoundModel,
```

The `self._model = model` assignment and all other usages stay unchanged — `FallbackBoundModel` exposes the same `provider`, `spec`, `stream()`, and `generate()` interface.

- [ ] **Step 2: Run mypy on the agent module**

```bash
uv run mypy cubepi/agent/agent.py
```

Expected: `Success: no issues found in 1 source file`

- [ ] **Step 3: Run full mypy**

```bash
uv run mypy cubepi/
```

Expected: `Success: no issues found in N source files`

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest tests/ -q --tb=short 2>&1 | tail -5
```

Expected: all passing, no regressions

- [ ] **Step 5: Commit**

```bash
git add cubepi/agent/agent.py
git commit -m "feat(agent): widen model type to accept FallbackBoundModel"
```

---

## Final: Lint check

- [ ] **Step 1: Run ruff**

```bash
uv run ruff check cubepi/ tests/providers/test_fallback.py
uv run ruff format --check cubepi/ tests/providers/test_fallback.py
```

Expected: all checks passed / already formatted.
If format check fails, run `uv run ruff format cubepi/ tests/providers/test_fallback.py` and amend the last relevant commit or add a fixup commit.

- [ ] **Step 2: Full suite one last time**

```bash
uv run pytest tests/ -q --tb=short 2>&1 | tail -3
```

Expected: all passed, 0 failures.
