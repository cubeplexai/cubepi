# Bound Provider Model API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CubePi's public agent construction API use one bound model object, so callers write `model = provider.model("...")` and `Agent(model=model, ...)` instead of passing both `provider` and `Model(provider=...)`.

**Architecture:** Keep the serializable model configuration as a pure Pydantic object, but rename its provider metadata to `provider_id`. Add a lightweight `BoundModel` dataclass that pairs a runtime `Provider` with that pure model spec. Providers own `provider_id` and expose `provider.model(...)` as the main constructor for bound models; Agent and middleware unwrap the bound model internally.

**Tech Stack:** Python 3.11+, Pydantic v2, dataclasses, pytest, ruff, mypy, Docusaurus docs.

---

## File Structure

- Modify `cubepi/providers/base.py`: add `BoundModel`, rename `Model.provider` to `Model.provider_id`, add `provider_id` and `model()` to `BaseProvider`, and update provider-error formatting.
- Modify `cubepi/providers/{anthropic.py,openai.py,openai_responses.py,faux.py}`: accept optional `provider_id` and pass it to `BaseProvider`.
- Modify `cubepi/agent/agent.py`: make `Agent.__init__` accept only a bound model object; store the runtime provider and pure model spec internally.
- Modify `cubepi/agent/loop.py`: no behavior change expected, but type imports should keep using the pure model spec and provider protocol.
- Modify `cubepi/middleware/compaction/__init__.py` and `cubepi/middleware/subagents.py`: replace `*_provider + *_model` constructor pairs with one bound model.
- Modify `cubepi/tracing/{meter.py,recorder.py}` and `cubepi/errors.py`: read `model.provider_id`.
- Modify provider implementations and image provider code that currently write `provider_id=model.provider`.
- Modify exports in `cubepi/providers/__init__.py` and `cubepi/__init__.py`: export `BoundModel`.
- Modify tests under `tests/agent`, `tests/providers`, `tests/middleware`, `tests/tracing`, `tests/hitl`, and `tests/mcp`: update construction call sites.
- Modify docs and examples in `README.md`, `AGENTS.md`, `skills/cubepi/SKILL.md`, `website/docs`, `website/i18n/zh-Hans/...`, and `website/versioned_docs/version-0.7`.

## API Shape

Target public usage:

```python
from cubepi import Agent
from cubepi.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(provider_id="anthropic")
model = provider.model(
    "claude-sonnet-4-5-20250929",
    reasoning=True,
    max_tokens=8192,
)

agent = Agent(
    model=model,
    system_prompt="You are helpful.",
)
```

Target Cubebox mapping:

```python
provider = factory.build_cubepi_provider(
    provider_config,
    provider_id=provider_slug,
)
model = provider.model(
    model_id,
    reasoning=reasoning_enabled,
    max_tokens=max_tokens,
    temperature=temperature,
)
agent = Agent(model=model, ...)
```

The underlying pure model spec remains serializable:

```python
Model(
    id="claude-sonnet-4-5-20250929",
    provider_id="anthropic",
    reasoning=True,
)
```

No backwards-compatibility shim is included in this plan. This is a breaking 0.7 API change.

---

### Task 1: Add BoundModel and Provider-Owned provider_id

**Files:**
- Modify: `cubepi/providers/base.py`
- Modify: `cubepi/providers/__init__.py`
- Modify: `cubepi/__init__.py`
- Test: `tests/providers/test_bound_model.py`

- [ ] **Step 1: Write the failing bound-model tests**

Create `tests/providers/test_bound_model.py`:

```python
from cubepi.providers.base import BaseProvider, BoundModel, Model


def test_base_provider_model_binds_runtime_provider_and_model_spec() -> None:
    provider = BaseProvider(provider_id="catalog-anthropic")

    bound = provider.model(
        "claude-sonnet-4-5-20250929",
        api="anthropic-messages",
        reasoning=True,
        context_window=200_000,
        max_tokens=8192,
        temperature=0.2,
    )

    assert isinstance(bound, BoundModel)
    assert bound.provider is provider
    assert bound.spec == Model(
        id="claude-sonnet-4-5-20250929",
        provider_id="catalog-anthropic",
        api="anthropic-messages",
        reasoning=True,
        context_window=200_000,
        max_tokens=8192,
        temperature=0.2,
    )


def test_base_provider_model_leaves_provider_id_empty_by_default() -> None:
    provider = BaseProvider()

    bound = provider.model("local-model")

    assert bound.provider is provider
    assert bound.spec.provider_id == ""
    assert bound.spec.id == "local-model"
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
uv run pytest tests/providers/test_bound_model.py -q
```

Expected: FAIL because `BoundModel`, `BaseProvider.provider_id`, and `BaseProvider.model()` do not exist yet.

- [ ] **Step 3: Implement the minimal base API**

In `cubepi/providers/base.py`, change `Model` and `BaseProvider` to this shape:

```python
@dataclass(frozen=True)
class BoundModel:
    provider: Provider
    spec: Model


class Model(BaseModel):
    id: str
    provider_id: str = ""
    api: str = ""
    reasoning: bool = False
    context_window: int = 200_000
    max_tokens: int = 8192
    temperature: float = 0.7
    cost: ModelCost | None = None
    thinking_level_map: dict[str, str | None] | None = None
```

Update `BaseProvider.__init__` and add `model()`:

```python
class BaseProvider:
    def __init__(self, *, provider_id: str = "") -> None:
        self.provider_id = provider_id
        self._request_listeners: list[OnRequestCallback] = []
        self._chunk_listeners: list[OnChunkCallback] = []
        self._response_listeners: list[OnResponseBodyCallback] = []

    def model(
        self,
        id: str,
        *,
        api: str = "",
        reasoning: bool = False,
        context_window: int = 200_000,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        cost: ModelCost | None = None,
        thinking_level_map: dict[str, str | None] | None = None,
    ) -> BoundModel:
        return BoundModel(
            provider=self,
            spec=Model(
                id=id,
                provider_id=self.provider_id,
                api=api,
                reasoning=reasoning,
                context_window=context_window,
                max_tokens=max_tokens,
                temperature=temperature,
                cost=cost,
                thinking_level_map=thinking_level_map,
            ),
        )
```

Keep `Provider` as the runtime protocol with `stream()` and `generate()`. Do not require custom protocol implementors to expose `model()`; users with custom providers can either inherit `BaseProvider` or construct `BoundModel(provider=custom_provider, spec=Model(...))`.

- [ ] **Step 4: Export the new type**

In `cubepi/providers/__init__.py` and `cubepi/__init__.py`, import and add `BoundModel` to `__all__`.

- [ ] **Step 5: Run the bound-model tests**

Run:

```bash
uv run pytest tests/providers/test_bound_model.py -q
```

Expected: PASS.

---

### Task 2: Move Built-In Providers onto BaseProvider(provider_id=...)

**Files:**
- Modify: `cubepi/providers/anthropic.py`
- Modify: `cubepi/providers/openai.py`
- Modify: `cubepi/providers/openai_responses.py`
- Modify: `cubepi/providers/faux.py`
- Test: `tests/providers/test_bound_model.py`

- [ ] **Step 1: Extend the failing test for concrete providers**

Append to `tests/providers/test_bound_model.py`:

```python
from cubepi.providers.faux import FauxProvider


def test_concrete_provider_accepts_provider_id() -> None:
    provider = FauxProvider(provider_id="faux-catalog")

    bound = provider.model("faux-1")

    assert bound.provider is provider
    assert bound.spec.provider_id == "faux-catalog"
    assert bound.spec.id == "faux-1"
```

- [ ] **Step 2: Run the concrete-provider test and verify it fails**

Run:

```bash
uv run pytest tests/providers/test_bound_model.py::test_concrete_provider_accepts_provider_id -q
```

Expected: FAIL because `FauxProvider.__init__` does not accept `provider_id`.

- [ ] **Step 3: Add provider_id to concrete provider constructors**

Update each built-in text provider constructor to accept `provider_id: str = ""` and call `super().__init__(provider_id=provider_id)`.

Example for `cubepi/providers/openai.py`:

```python
class OpenAIProvider(BaseProvider):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
        capability: CapabilityDescriptor | None = None,
        model_capability_overrides: dict[str, CapabilityDescriptor] | None = None,
        provider_id: str = "",
    ) -> None:
        super().__init__(provider_id=provider_id)
```

Apply the same pattern to:

```python
AnthropicProvider(..., provider_id: str = "")
OpenAIResponsesProvider(..., provider_id: str = "")
FauxProvider(..., provider_id: str = "")
```

- [ ] **Step 4: Run provider constructor tests**

Run:

```bash
uv run pytest tests/providers/test_bound_model.py -q
```

Expected: PASS.

---

### Task 3: Rename Model.provider to Model.provider_id Internally

**Files:**
- Modify: `cubepi/providers/base.py`
- Modify: `cubepi/providers/{anthropic.py,openai.py,openai_responses.py,faux.py}`
- Modify: `cubepi/providers/images/{types.py,openai_images.py,faux.py,registry.py}`
- Modify: `cubepi/tracing/{meter.py,recorder.py}`
- Modify: `cubepi/errors.py`
- Modify: tests that assert provider metadata
- Test: `tests/providers/test_bound_model.py`
- Test: `tests/tracing/test_meter.py`
- Test: `tests/tracing/test_recorder.py`
- Test: `tests/test_errors.py`

- [ ] **Step 1: Add focused tests for provider_id metadata**

Append to `tests/providers/test_bound_model.py`:

```python
from cubepi.providers.base import AssistantMessage, TextContent


def test_assistant_message_uses_model_provider_id_metadata() -> None:
    model = Model(id="faux-1", provider_id="faux-provider")

    message = AssistantMessage(
        content=[TextContent(text="ok")],
        provider_id=model.provider_id,
        model_id=model.id,
    )

    assert message.provider_id == "faux-provider"
    assert message.model_id == "faux-1"
```

- [ ] **Step 2: Run focused tests and verify old references still need cleanup**

Run:

```bash
uv run pytest tests/providers/test_bound_model.py -q
uv run ruff check cubepi/providers/base.py cubepi/providers cubepi/tracing cubepi/errors.py
```

Expected: the new test passes after Task 1, but ruff or pytest still finds runtime references to `model.provider` once `Model.provider` has been removed.

- [ ] **Step 3: Replace `model.provider` reads with `model.provider_id`**

Run a search:

```bash
rg "model\\.provider" cubepi tests website/docs README.md
```

Update code references in `cubepi/`:

```python
provider_id=model.provider_id
```

Update error formatting in `cubepi/providers/base.py`:

```python
target = f"{model.provider_id}/{model.id}" if model.provider_id else model.id
```

Update `cubepi/errors.py`:

```python
provider = model.provider_id
```

Do not update docs in this task; docs are handled after the code API settles.

- [ ] **Step 4: Update tests to instantiate pure model specs with provider_id**

Replace test construction like:

```python
Model(id="faux-1", provider="faux")
```

with:

```python
Model(id="faux-1", provider_id="faux")
```

Use `provider.model("...")` only in tests that construct an `Agent` or middleware requiring a bound runtime model.

- [ ] **Step 5: Run metadata and provider tests**

Run:

```bash
uv run pytest tests/providers tests/tracing tests/test_errors.py -q
```

Expected: PASS for provider metadata, tracing, and error tests.

---

### Task 4: Make Agent Accept Only a Bound Model

**Files:**
- Modify: `cubepi/agent/agent.py`
- Modify: `cubepi/agent/loop.py` only if type imports need cleanup
- Test: `tests/agent/test_agent.py`
- Test: `tests/agent/test_loop.py`
- Test: `tests/agent/test_e2e.py`

- [ ] **Step 1: Add a focused Agent API test**

Add to `tests/agent/test_agent.py`:

```python
from cubepi import Agent
from cubepi.providers.faux import FauxProvider


def test_agent_unwraps_bound_model() -> None:
    provider = FauxProvider(provider_id="faux")
    model = provider.model("faux-1")

    agent: Agent = Agent(model=model)

    assert agent.state.model.id == "faux-1"
    assert agent.state.model.provider_id == "faux"
```

- [ ] **Step 2: Run the focused Agent API test and verify it fails**

Run:

```bash
uv run pytest tests/agent/test_agent.py::test_agent_unwraps_bound_model -q
```

Expected: FAIL because `Agent.__init__` still requires `provider=`.

- [ ] **Step 3: Change Agent.__init__ signature and internal unwrapping**

In `cubepi/agent/agent.py`, change imports:

```python
from cubepi.providers.base import BoundModel, Model, Provider, ...
```

Change the constructor signature from:

```python
def __init__(
    self,
    *,
    provider: Provider,
    model: Model,
    ...
) -> None:
```

to:

```python
def __init__(
    self,
    *,
    model: BoundModel,
    ...
) -> None:
```

At the top of the constructor, unwrap:

```python
self._provider: Provider = model.provider
self._state = AgentState(
    system_prompt=system_prompt,
    model=model.spec,
    thinking=thinking,
)
```

Keep `AgentState.model` typed as the pure `Model` spec.

- [ ] **Step 4: Update Agent construction tests**

Replace:

```python
provider = FauxProvider()
agent = Agent(provider=provider, model=Model(id="faux-1", provider_id="faux"))
```

with:

```python
provider = FauxProvider(provider_id="faux")
agent = Agent(model=provider.model("faux-1"))
```

When a test needs non-default spec fields:

```python
agent = Agent(
    model=provider.model("faux-reasoning", reasoning=True),
    thinking="medium",
)
```

- [ ] **Step 5: Run Agent tests**

Run:

```bash
uv run pytest tests/agent -q
```

Expected: PASS.

---

### Task 5: Update Compaction and Subagent Middleware to Use Bound Models

**Files:**
- Modify: `cubepi/middleware/compaction/__init__.py`
- Modify: `cubepi/middleware/compaction/summarizer.py` if type imports require cleanup
- Modify: `cubepi/middleware/subagents.py`
- Test: `tests/middleware/test_compaction.py`
- Test: `tests/middleware/test_subagents.py`
- Test: `tests/middleware/compaction/test_summarizer.py`

- [ ] **Step 1: Add/update middleware API tests**

Update `tests/middleware/test_compaction.py` setup to use:

```python
provider = FauxProvider(provider_id="faux")
middleware = CompactionMiddleware(
    summary_model=provider.model("summary-model"),
    max_tokens_before_compact=1,
)
```

Update `tests/middleware/test_subagents.py` setup to use:

```python
provider = FauxProvider(provider_id="faux")
middleware = SubagentMiddleware(
    subagents={},
    default_model=provider.model("faux-1"),
)
```

For per-subagent overrides, use:

```python
SubagentSpec(
    name="researcher",
    description="Research one topic",
    system_prompt="You research.",
    model=other_provider.model("faux-researcher"),
)
```

- [ ] **Step 2: Run middleware tests and verify they fail**

Run:

```bash
uv run pytest tests/middleware/test_compaction.py tests/middleware/test_subagents.py -q
```

Expected: FAIL because constructors still require separate provider arguments and `Agent(...)` calls still pass `provider=`.

- [ ] **Step 3: Update CompactionMiddleware**

Change constructor imports and signature:

```python
from cubepi.providers.base import BoundModel, Message, Model, TextContent, UserMessage


class CompactionMiddleware(Middleware):
    def __init__(
        self,
        *,
        summary_model: BoundModel,
        max_tokens_before_compact: int,
        keep_recent_messages: int = 8,
        max_summary_tokens: int = 1024,
        min_compact_messages: int = 4,
    ) -> None:
        self._summary_provider = summary_model.provider
        self._summary_model = summary_model.spec
```

Keep the `summarize(provider=..., model=...)` helper unchanged because it is an internal low-level call.

- [ ] **Step 4: Update SubagentMiddleware**

Change `SubagentSpec`:

```python
@dataclass(frozen=True)
class SubagentSpec:
    name: str
    description: str
    system_prompt: str
    model: BoundModel | None = None
    tools: Sequence[AgentTool[BaseModel]] = field(default_factory=tuple)
    middleware: Sequence[Middleware] = field(default_factory=tuple)
```

Change `SubagentMiddleware.__init__`:

```python
def __init__(
    self,
    *,
    subagents: dict[str, SubagentSpec],
    default_model: BoundModel,
    shared_tools: Sequence[AgentTool[BaseModel]] = (),
    inherited_middleware: Sequence[Middleware] = (),
    ...
) -> None:
    self._default_model = default_model
```

Change `_run_subagent`:

```python
model = spec.model or self._default_model
child: Agent[BaseModel] = Agent(
    model=model,
    system_prompt=spec.system_prompt,
    tools=tools,
    middleware=middleware,
)
```

- [ ] **Step 5: Run middleware tests**

Run:

```bash
uv run pytest tests/middleware -q
```

Expected: PASS.

---

### Task 6: Update All Remaining Test Call Sites

**Files:**
- Modify: `tests/**/*.py`

- [ ] **Step 1: Search all old public API call sites**

Run:

```bash
rg "Agent\\([^\\n]*provider=|provider=.*model=|Model\\([^\\n]*provider=" tests cubepi
rg "default_provider|summary_provider|spec\\.provider|model\\.provider" tests cubepi
```

Expected: output lists remaining old API usage.

- [ ] **Step 2: Replace Agent constructions**

Use this mechanical pattern:

```python
provider = FauxProvider(provider_id="faux")
agent = Agent(model=provider.model("faux-1"), ...)
```

For tests that already have `provider` and `model_id` variables:

```python
agent = Agent(model=provider.model(model_id), ...)
```

For tests that need a pure model spec without an Agent:

```python
model = Model(id="faux-1", provider_id="faux")
```

- [ ] **Step 3: Replace compaction/subagent constructions**

Use:

```python
CompactionMiddleware(summary_model=provider.model("summary-model"), ...)
SubagentMiddleware(default_model=provider.model("faux-1"), ...)
```

- [ ] **Step 4: Run the full Python test suite**

Run:

```bash
uv run pytest tests/ -q
```

Expected: PASS, with the existing skipped tests unchanged.

---

### Task 7: Update Docs, Examples, and Release Notes

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `skills/cubepi/SKILL.md`
- Modify: `website/docs/**/*.md`
- Modify: `website/docs/**/*.mdx`
- Modify: `website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/**/*.md`
- Modify: `website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/**/*.mdx`
- Modify: `website/versioned_docs/version-0.7/**/*.mdx`
- Modify: `website/i18n/zh-Hans/docusaurus-plugin-content-docs/version-0.7/**/*.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Search docs for old API**

Run:

```bash
rg "Agent\\(|Model\\(|provider=|default_provider|summary_provider|model\\.provider" README.md AGENTS.md skills/cubepi/SKILL.md website/docs website/i18n/zh-Hans/docusaurus-plugin-content-docs/current website/versioned_docs/version-0.7 website/i18n/zh-Hans/docusaurus-plugin-content-docs/version-0.7
```

Expected: output lists all examples and prose that need updates.

- [ ] **Step 2: Update basic examples**

Replace examples like:

```python
provider = AnthropicProvider()
agent = Agent(
    provider=provider,
    model=Model(id="claude-sonnet-4-5-20250929", provider="anthropic"),
)
```

with:

```python
provider = AnthropicProvider(provider_id="anthropic")
model = provider.model("claude-sonnet-4-5-20250929")
agent = Agent(model=model)
```

- [ ] **Step 3: Update compaction and subagent examples**

Replace:

```python
CompactionMiddleware(
    summary_provider=cheap_provider,
    summary_model=Model(id="claude-haiku-4-5", provider="anthropic"),
    max_tokens_before_compact=80_000,
)
```

with:

```python
CompactionMiddleware(
    summary_model=cheap_provider.model("claude-haiku-4-5"),
    max_tokens_before_compact=80_000,
)
```

Replace:

```python
SubagentMiddleware(
    subagents=subagents,
    default_provider=provider,
    default_model=model,
)
```

with:

```python
SubagentMiddleware(
    subagents=subagents,
    default_model=model,
)
```

- [ ] **Step 4: Update conceptual docs**

Add a short explanation in provider and quick-start docs:

```text
`provider.model(...)` returns a bound model. The provider holds credentials,
base URL, wire API configuration, and optional `provider_id` metadata. The
model spec stays serializable; CubePi stores only the spec in agent state and
uses the bound provider object for runtime calls.
```

Remove prose that says `Model(id=..., provider=...)` is required.

- [ ] **Step 5: Update CHANGELOG breaking-change entry**

Add to the 0.7.0 breaking changes:

```markdown
- Agent construction now takes a bound model from `provider.model(...)`.
  Replace `Agent(provider=provider, model=Model(...))` with
  `Agent(model=provider.model("model-id", ...))`. `provider_id` now lives on
  the provider constructor and is copied into the model metadata used for
  tracing and error messages.
```

- [ ] **Step 6: Re-run docs search**

Run:

```bash
rg "Agent\\([^\\n]*provider=|Model\\([^\\n]*provider=|default_provider|summary_provider|model\\.provider" README.md AGENTS.md skills/cubepi/SKILL.md website/docs website/i18n/zh-Hans/docusaurus-plugin-content-docs/current website/versioned_docs/version-0.7 website/i18n/zh-Hans/docusaurus-plugin-content-docs/version-0.7
```

Expected: no old public API examples remain. Internal explanations may mention the old API only in migration text.

---

### Task 8: Update API Reference Generation and Versioned Docs

**Files:**
- Modify generated files under `website/versioned_docs/version-0.7/api/*.mdx`
- Test: `website/scripts/tests/test_build_api_reference.py`

- [ ] **Step 1: Regenerate API reference**

Run:

```bash
cd website
pnpm build:api
```

Expected: API docs include `BoundModel`, `Model.provider_id`, and updated `Agent` signature.

- [ ] **Step 2: Inspect generated API snippets**

Run:

```bash
rg "BoundModel|provider_id|Agent\\(" website/docs/api website/versioned_docs/version-0.7/api
```

Expected: generated API docs expose `BoundModel`, show `provider_id`, and no longer show `Agent(..., provider=...)` as the constructor signature.

- [ ] **Step 3: Run API reference tests**

Run:

```bash
cd website
pnpm test -- VersionAwareDocLink
cd ..
uv run pytest website/scripts/tests/test_build_api_reference.py -q
```

Expected: PASS.

---

### Task 9: Static Checks and Full Validation

**Files:**
- No new files expected

- [ ] **Step 1: Run Python lint and formatting checks**

Run:

```bash
uv run ruff check cubepi/ tests/
uv run ruff format --check cubepi/ tests/
```

Expected: PASS.

- [ ] **Step 2: Run mypy**

Run:

```bash
uv run mypy cubepi
```

Expected: PASS. Existing mypy informational notes are acceptable.

- [ ] **Step 3: Run full Python tests**

Run:

```bash
uv run pytest tests/ -q
```

Expected: PASS, with the existing skipped tests unchanged.

- [ ] **Step 4: Run website checks**

Run:

```bash
cd website
pnpm typecheck
pnpm test
pnpm build
```

Expected: PASS. Existing Docusaurus v4 deprecation warning is acceptable.

- [ ] **Step 5: Build package**

Run:

```bash
uv build
```

Expected: PASS and produces `dist/cubepi-0.7.0.tar.gz` and `dist/cubepi-0.7.0-py3-none-any.whl`.

- [ ] **Step 6: Check release diff cleanliness**

Run:

```bash
git diff --check
git status --short
```

Expected: `git diff --check` emits no whitespace errors. `git status --short` shows only intentional release-prep/API-change files.

---

## Self-Review

- Spec coverage: The plan covers provider-owned `provider_id`, bound model construction, Agent single-argument model construction, compaction/subagent middleware, tracing/error metadata, docs, generated API reference, and full release validation.
- Placeholder scan: No implementation step uses unresolved placeholders; every code-changing task includes concrete signatures or replacement examples.
- Type consistency: `Model` is the pure Pydantic spec, `BoundModel` is the runtime pair, `provider.model(...)` returns `BoundModel`, and `AgentState.model` remains pure `Model`.
