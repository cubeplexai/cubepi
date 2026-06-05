# Image Provider Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `cubepi/providers/images/` surface with a redesign that mirrors chat-provider 0.7 conventions: `provider_id` on the provider constructor, `provider.model("id", ...)` factory, typed `ProviderError` failures, request/response observer registry, and an `ImagesCapabilityDescriptor` that lets one `OpenAIImagesProvider` reach OpenAI-shape backends (OpenAI / Doubao Seedream / SiliconFlow / Together) without subclassing.

**Architecture:** Demolish the old image surface first (delete `registry.py`, `openai_images.py`, `faux.py`, all 4 old test files, and gut `__init__.py` to a stub), then rebuild bottom-up: types → capability descriptor → `BaseImagesProvider` (factory + listeners) → `_build_payload` (capability application) → `classify_and_raise` widening → concrete providers → public exports → docs. Each task is a single coherent commit that leaves the test suite green.

**Tech Stack:** Python 3.11+, Pydantic v2, dataclasses, pytest (`asyncio_mode=auto`), ruff, mypy, OpenAI SDK, Docusaurus docs.

**Spec:** `dev/specs/2026-06-05-images-provider-redesign.md`

---

## File Structure

**Files to CREATE:**

- `cubepi/providers/images/base.py` — `ImagesProvider` Protocol + `BaseImagesProvider` (provider_id, `.model()` factory, listener registry, `_build_payload`, `_capability_for`, `_error_message`).
- `cubepi/providers/images/capability.py` — `SizeSpec`, `ImagesCapabilityDescriptor`.
- `tests/providers/images/test_types.py` — REPLACES old file; pins fields on the redesigned types.
- `tests/providers/images/test_capability.py` — pins capability descriptor structure and defaults.
- `tests/providers/images/test_base.py` — provider_id propagation, `.model()` factory, listener register/detach.
- `tests/providers/images/test_capability_payload_mapping.py` — parametrized `_build_payload` tests for each `SizeSpec.kind`, `count_field`, `supports_*` gates, `extra_payload` + `ctx.extra` merge.
- `tests/providers/images/test_faux_provider.py` — happy path, `raise_on_call` injection, listener wiring.
- `tests/providers/images/test_openai_images.py` — REPLACES old file; capability paths, error classification, cancellation, listeners.
- `tests/errors/test_classify_images.py` — `classify_and_raise(model=ImagesModel(...))` paths.

**Files to MODIFY:**

- `cubepi/providers/images/types.py` — full rewrite: drop `provider` field rename to `provider_id`, add model-level defaults, add `ImagesCost`, add `ImagesContext` typed fields + `extra`, add `ImagesOptions`, tighten `AssistantImages.stop_reason` to `Literal["stop", "aborted"]`, remove `error_message`.
- `cubepi/providers/images/openai_images.py` — full rewrite: inherit `BaseImagesProvider`, apply capability descriptor, classify errors, honor `options.signal`, fire request/response listeners.
- `cubepi/providers/images/faux.py` — full rewrite: inherit `BaseImagesProvider`, add `raise_on_call` parameter.
- `cubepi/providers/images/__init__.py` — wipe then rebuild: export new types, descriptor, base class, two concrete providers.
- `cubepi/__init__.py` — add image-side public surface (types, descriptor, base — NOT concrete providers, parallel to chat's pattern of not exporting `AnthropicProvider`/`OpenAIProvider` at top level).
- `cubepi/errors.py` — widen `classify_and_raise(model=...)` to accept a structural type covering both `Model` and `ImagesModel` (use `getattr` for `context_window`).
- `website/docs/guides/providers/image-generation.md` — full rewrite (English current).
- `website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/guides/providers/image-generation.md` — full rewrite (Chinese current).
- `website/versioned_docs/version-0.7/guides/providers/image-generation.md` — mirror EN current.
- `website/i18n/zh-Hans/docusaurus-plugin-content-docs/version-0.7/guides/providers/image-generation.md` — mirror zh current.
- `website/docs/guides/providers/overview.md` + zh-Hans + versioned mirrors — add a one-paragraph image-providers entry pointer.
- `CHANGELOG.md` — add a Breaking bullet under `[0.7.0]`.

**Files to DELETE:**

- `cubepi/providers/images/registry.py`
- `tests/providers/images/test_registry.py`
- `tests/providers/images/test_generate.py`
- *(`tests/providers/images/test_types.py` and `tests/providers/images/test_openai_images.py` are deleted then re-created with new content.)*

---

## API Shape Reference

For Tasks 4–8, the target shape from the spec is:

```python
from cubepi.providers.images import (
    OpenAIImagesProvider, ImagesContext, ImagesOptions,
)
from cubepi.providers.images.capability import (
    ImagesCapabilityDescriptor, SizeSpec,
)

provider = OpenAIImagesProvider(
    provider_id="doubao",
    api_key=os.environ["ARK_API_KEY"],
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    capability=ImagesCapabilityDescriptor(
        supports_seed=True, extra_payload={"watermark": False},
    ),
)
model = provider.model("doubao-seedream-4-5-251128", default_size="2K", default_n=1)
result = await provider.generate_images(
    model,
    ImagesContext(prompt="A robot at sunrise"),
    options=ImagesOptions(signal=cancel_event),
)
```

Failures raise `cubepi.errors.ProviderError` subclasses; the `stop_reason` field on `AssistantImages` is only `"stop"` or `"aborted"`.

---

## Task 1: Demolish old image surface

**Files:**
- Delete: `cubepi/providers/images/registry.py`
- Delete: `cubepi/providers/images/openai_images.py`
- Delete: `cubepi/providers/images/faux.py`
- Delete: `tests/providers/images/test_types.py`
- Delete: `tests/providers/images/test_registry.py`
- Delete: `tests/providers/images/test_generate.py`
- Delete: `tests/providers/images/test_openai_images.py`
- Modify: `cubepi/providers/images/__init__.py` (gut to empty)
- Modify: `cubepi/providers/images/types.py` (gut to empty)

- [ ] **Step 1: Delete the old files**

```bash
rm cubepi/providers/images/registry.py
rm cubepi/providers/images/openai_images.py
rm cubepi/providers/images/faux.py
rm tests/providers/images/test_types.py
rm tests/providers/images/test_registry.py
rm tests/providers/images/test_generate.py
rm tests/providers/images/test_openai_images.py
```

- [ ] **Step 2: Stub `cubepi/providers/images/__init__.py` to a minimal package marker**

Replace contents with:

```python
"""Image generation providers (redesigned in 0.7).

This package is in the process of being rebuilt. See
``dev/specs/2026-06-05-images-provider-redesign.md`` and the matching plan.
"""

__all__: list[str] = []
```

- [ ] **Step 3: Stub `cubepi/providers/images/types.py` to a minimal placeholder**

Replace contents with:

```python
"""Image generation types — placeholder during the 0.7 redesign.

The real types are added back in Task 2 of the implementation plan.
"""
```

- [ ] **Step 4: Verify the package still imports and the test suite collects with no image tests**

Run:
```bash
uv run python -c "import cubepi.providers.images"
uv run pytest tests/providers/images/ -v
```

Expected:
- The `python -c` import succeeds (no traceback).
- The pytest run reports `no tests ran` from `tests/providers/images/` (other test directories still run normally).

- [ ] **Step 5: Run the full test suite and confirm nothing else broke**

Run: `uv run pytest tests/ -q`
Expected: PASS (image tests are gone; nothing outside `tests/providers/images/` depended on the old image API — verified by the grep in §4.1 of the spec).

- [ ] **Step 6: Commit**

```bash
git add cubepi/providers/images/ tests/providers/images/
git commit -m "chore(images): wipe old image-provider surface for 0.7 redesign

Demolish before rebuild: registry, openai_images, faux, and all four old
image test files are removed; __init__.py and types.py are reduced to
package-marker stubs. The new surface is rebuilt bottom-up in the next
tasks per dev/specs/2026-06-05-images-provider-redesign.md."
```

---

## Task 2: Rebuild types (`types.py`)

**Files:**
- Modify: `cubepi/providers/images/types.py`
- Create: `tests/providers/images/test_types.py`

- [ ] **Step 1: Write the failing test**

Create `tests/providers/images/test_types.py`:

```python
import asyncio
from typing import get_args

import pytest
from pydantic import ValidationError

from cubepi.providers.base import ImageContent
from cubepi.providers.images.types import (
    AssistantImages,
    ImagesContext,
    ImagesCost,
    ImagesModel,
    ImagesOptions,
)


def test_images_model_required_field_id():
    with pytest.raises(ValidationError):
        ImagesModel()  # type: ignore[call-arg]


def test_images_model_provider_id_renamed():
    m = ImagesModel(id="gpt-image-1", provider_id="openai", api="openai-images")
    assert m.provider_id == "openai"
    assert m.api == "openai-images"
    assert not hasattr(m, "provider"), "old `provider` field must be gone"


def test_images_model_default_fields():
    m = ImagesModel(
        id="gpt-image-1",
        provider_id="openai",
        default_size="1024x1024",
        default_n=2,
        default_quality="high",
        default_output_format="png",
    )
    assert m.default_size == "1024x1024"
    assert m.default_n == 2
    assert m.default_quality == "high"
    assert m.default_output_format == "png"


def test_images_model_defaults_are_optional_none():
    m = ImagesModel(id="gpt-image-1", provider_id="openai")
    assert m.default_size is None
    assert m.default_n is None
    assert m.default_quality is None
    assert m.default_output_format is None
    assert m.cost is None
    assert m.max_input_images is None


def test_images_cost_fields():
    cost = ImagesCost(per_image=0.04, per_megapixel=0.0)
    assert cost.per_image == 0.04
    assert cost.per_megapixel == 0.0


def test_images_context_typed_fields():
    ctx = ImagesContext(
        prompt="A robot",
        size="1024x1024",
        n=2,
        quality="high",
        output_format="png",
        seed=42,
        negative_prompt="blurry",
        steps=20,
        guidance=7.5,
        extra={"watermark": False},
    )
    assert ctx.prompt == "A robot"
    assert ctx.size == "1024x1024"
    assert ctx.n == 2
    assert ctx.quality == "high"
    assert ctx.output_format == "png"
    assert ctx.seed == 42
    assert ctx.negative_prompt == "blurry"
    assert ctx.steps == 20
    assert ctx.guidance == 7.5
    assert ctx.extra == {"watermark": False}


def test_images_context_input_images_accepts_image_content():
    ctx = ImagesContext(
        prompt="edit",
        input_images=[ImageContent(source="b64", media_type="image/png")],
    )
    assert len(ctx.input_images) == 1


def test_images_context_quality_literal_rejects_unknown():
    with pytest.raises(ValidationError):
        ImagesContext(prompt="x", quality="ultra")  # type: ignore[arg-type]


def test_images_options_default_construction():
    opts = ImagesOptions()
    assert opts.signal is None
    assert opts.on_payload is None
    assert opts.on_response is None


def test_images_options_accepts_event_and_callbacks():
    ev = asyncio.Event()
    opts = ImagesOptions(signal=ev, on_payload=lambda p, m: None, on_response=lambda r, m: None)
    assert opts.signal is ev
    assert opts.on_payload is not None
    assert opts.on_response is not None


def test_assistant_images_stop_reason_literal():
    allowed = set(get_args(AssistantImages.model_fields["stop_reason"].annotation))
    assert allowed == {"stop", "aborted"}, "error stop_reason is removed in 0.7"


def test_assistant_images_no_error_message_field():
    assert "error_message" not in AssistantImages.model_fields


def test_assistant_images_provider_id_renamed():
    out = AssistantImages(
        api="openai-images", provider_id="openai", model="gpt-image-1", output=[],
    )
    assert out.provider_id == "openai"
    assert not hasattr(out, "provider"), "old `provider` field must be gone"
```

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest tests/providers/images/test_types.py -v`
Expected: FAIL with ImportError or AttributeError (types module is still the stub from Task 1).

- [ ] **Step 3: Rewrite `cubepi/providers/images/types.py`**

Replace the entire file with:

```python
from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cubepi.providers.base import (
    ImageContent,
    OnPayloadCallback,
    OnResponseCallback,
    TextContent,
)
from cubepi.types import JsonObject


class ImagesCost(BaseModel):
    """Image-generation pricing (per-image is dominant; per-megapixel for Imagen-like models)."""

    per_image: float = 0
    per_megapixel: float = 0


class ImagesModel(BaseModel):
    """Image-generation model spec, with model-level defaults applied when the
    matching ``ImagesContext`` field is not set."""

    id: str
    provider_id: str = ""
    api: str = ""

    default_size: str | None = None
    default_n: int | None = None
    default_quality: Literal["low", "medium", "high"] | None = None
    default_output_format: Literal["png", "jpeg", "webp"] | None = None

    cost: ImagesCost | None = None
    max_input_images: int | None = None


class ImagesContext(BaseModel):
    """Per-call request payload.

    ``size`` / ``n`` / ``quality`` / ``output_format`` override the matching
    ``ImagesModel.default_*``. ``seed`` / ``negative_prompt`` / ``steps`` /
    ``guidance`` are only written to the wire payload when the provider's
    ``ImagesCapabilityDescriptor`` declares support; otherwise they are
    dropped with a one-time warning. ``extra`` carries truly backend-specific
    fields that the descriptor does not model (e.g. Doubao's ``watermark``).
    """

    prompt: str
    input_images: list[ImageContent] = Field(default_factory=list)

    size: str | None = None
    n: int | None = None
    quality: Literal["low", "medium", "high"] | None = None
    output_format: Literal["png", "jpeg", "webp"] | None = None

    seed: int | None = None
    negative_prompt: str | None = None
    steps: int | None = None
    guidance: float | None = None

    extra: dict[str, Any] = Field(default_factory=dict)


class ImagesOptions(BaseModel):
    """Per-call cross-cutting options (analog of chat's ``StreamOptions``)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    signal: asyncio.Event | None = None
    on_payload: OnPayloadCallback | None = None
    on_response: OnResponseCallback | None = None


class AssistantImages(BaseModel):
    """Response from a successful or aborted image generation call.

    Failures raise ``cubepi.errors.ProviderError`` subclasses; they never
    appear as a ``stop_reason``. ``stop_reason="aborted"`` is produced when
    ``ImagesOptions.signal`` fires mid-call.
    """

    api: str
    provider_id: str
    model: str
    output: list[ImageContent | TextContent] = Field(default_factory=list)
    stop_reason: Literal["stop", "aborted"] = "stop"
    response_id: str | None = None
    metadata: JsonObject = Field(default_factory=dict)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/images/test_types.py -v`
Expected: PASS (all 12 tests pass).

- [ ] **Step 5: Run ruff and the rest of the suite**

Run:
```bash
uv run ruff check cubepi/providers/images/types.py tests/providers/images/test_types.py
uv run ruff format --check cubepi/providers/images/types.py tests/providers/images/test_types.py
uv run pytest tests/ -q
```
Expected: PASS for all three.

- [ ] **Step 6: Commit**

```bash
git add cubepi/providers/images/types.py tests/providers/images/test_types.py
git commit -m "feat(images): rebuild types with provider_id + model-level defaults + typed context

ImagesModel renames provider→provider_id and gains default_size / default_n /
default_quality / default_output_format / cost / max_input_images. ImagesContext
gets typed size/n/quality/output_format/seed/negative_prompt/steps/guidance plus
an extra dict for backend-only fields. AssistantImages drops error_message and
the 'error' stop_reason — failures will raise typed ProviderError in 0.7.
ImagesOptions is the new per-call cross-cutting bag (signal + on_payload +
on_response), parallel to chat's StreamOptions."
```

---

## Task 3: Capability descriptor (`capability.py`)

**Files:**
- Create: `cubepi/providers/images/capability.py`
- Create: `tests/providers/images/test_capability.py`

- [ ] **Step 1: Write the failing test**

Create `tests/providers/images/test_capability.py`:

```python
from cubepi.providers.images.capability import (
    ImagesCapabilityDescriptor,
    SizeSpec,
)


def test_default_descriptor_is_openai_shape():
    d = ImagesCapabilityDescriptor()
    assert d.size_spec.kind == "size_string"
    assert d.count_field == "n"
    assert d.supports_seed is False
    assert d.supports_negative_prompt is False
    assert d.supports_steps is False
    assert d.supports_guidance is False
    assert d.output_format_field == "output_format"
    assert d.response_format_field == "response_format"
    assert d.response_format_value == "b64_json"
    assert d.supports_edit is True
    assert d.input_images_field == "image"
    assert d.extra_payload == {}


def test_size_spec_kinds_enum():
    for kind in ("size_string", "image_size_string", "width_height", "aspect_ratio"):
        s = SizeSpec(kind=kind)
        assert s.kind == kind


def test_descriptor_for_doubao_shape():
    d = ImagesCapabilityDescriptor(
        supports_seed=True, extra_payload={"watermark": False},
    )
    assert d.supports_seed is True
    assert d.extra_payload == {"watermark": False}
    assert d.size_spec.kind == "size_string"


def test_descriptor_for_siliconflow_shape():
    d = ImagesCapabilityDescriptor(
        size_spec=SizeSpec(kind="image_size_string"),
        count_field="batch_size",
        supports_seed=True,
        supports_steps=True, steps_field="num_inference_steps",
        supports_guidance=True, guidance_field="guidance_scale",
        supports_negative_prompt=True,
        output_format_field=None,
    )
    assert d.size_spec.kind == "image_size_string"
    assert d.count_field == "batch_size"
    assert d.steps_field == "num_inference_steps"
    assert d.guidance_field == "guidance_scale"
    assert d.output_format_field is None


def test_descriptor_for_together_flux_schnell_shape():
    d = ImagesCapabilityDescriptor(
        size_spec=SizeSpec(kind="aspect_ratio"),
        supports_seed=True,
        supports_steps=True, steps_field="steps",
    )
    assert d.size_spec.kind == "aspect_ratio"
    assert d.steps_field == "steps"
```

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest tests/providers/images/test_capability.py -v`
Expected: FAIL with `ModuleNotFoundError: cubepi.providers.images.capability`.

- [ ] **Step 3: Create `cubepi/providers/images/capability.py`**

```python
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SizeSpec(BaseModel):
    """How the canonical ``ctx.size`` value is serialized to the wire.

    - ``size_string``       → ``{"size": ctx.size}`` (OpenAI, Doubao)
    - ``image_size_string`` → ``{"image_size": ctx.size}`` (SiliconFlow)
    - ``width_height``      → split ``"<W>x<H>"`` into ``{"width": W, "height": H}``
    - ``aspect_ratio``      → ``{"aspect_ratio": ctx.size}`` (Together FLUX schnell, Imagen)
    """

    kind: Literal["size_string", "image_size_string", "width_height", "aspect_ratio"]


class ImagesCapabilityDescriptor(BaseModel):
    """Data-level description of an OpenAI-shape image backend's wire quirks.

    The descriptor is consumed by ``BaseImagesProvider._build_payload`` to
    rename canonical CubePi fields onto whatever wire keys the backend
    expects, and to gate ``ImagesContext`` fields the backend does not
    support. Backends whose shape is fundamentally different (async-task
    models like Aliyun Wanxiang, Imagen on Vertex, Stability, Replicate)
    need a separate provider subclass; this descriptor does not try to
    cover them.
    """

    size_spec: SizeSpec = Field(default_factory=lambda: SizeSpec(kind="size_string"))
    count_field: str = "n"

    supports_seed: bool = False
    seed_field: str = "seed"

    supports_negative_prompt: bool = False
    negative_prompt_field: str = "negative_prompt"

    supports_steps: bool = False
    steps_field: str = "num_inference_steps"

    supports_guidance: bool = False
    guidance_field: str = "guidance_scale"

    output_format_field: str | None = "output_format"
    response_format_field: str = "response_format"
    response_format_value: Literal["b64_json", "url"] = "b64_json"

    supports_edit: bool = True
    input_images_field: str = "image"

    extra_payload: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/images/test_capability.py -v`
Expected: PASS (all 5 tests pass).

- [ ] **Step 5: Run ruff**

Run: `uv run ruff check cubepi/providers/images/capability.py tests/providers/images/test_capability.py && uv run ruff format --check cubepi/providers/images/capability.py tests/providers/images/test_capability.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add cubepi/providers/images/capability.py tests/providers/images/test_capability.py
git commit -m "feat(images): add ImagesCapabilityDescriptor + SizeSpec

Data-level description of OpenAI-shape image backend quirks: size
serialization (4 kinds), count field name, supports_seed/steps/guidance/
negative_prompt gating with renameable field names, output_format and
response_format fields, edit-path field, and provider-level extra_payload
for always-injected fields. Async-task backends (Wanxiang, Imagen, etc.)
are not modeled here — they need a separate provider subclass."
```

---

## Task 4: `BaseImagesProvider` scaffolding (no `_build_payload` yet)

**Files:**
- Create: `cubepi/providers/images/base.py`
- Create: `tests/providers/images/test_base.py`

- [ ] **Step 1: Write the failing test**

Create `tests/providers/images/test_base.py`:

```python
import pytest

from cubepi.providers.images.base import BaseImagesProvider, ImagesProvider
from cubepi.providers.images.capability import (
    ImagesCapabilityDescriptor,
    SizeSpec,
)
from cubepi.providers.images.types import (
    AssistantImages,
    ImagesContext,
    ImagesCost,
    ImagesModel,
)


class _StubBase(BaseImagesProvider):
    """Minimal concrete subclass exposing helpers + a no-op generate_images."""

    async def generate_images(self, model, context, *, options=None):
        return AssistantImages(
            api=model.api, provider_id=model.provider_id, model=model.id, output=[],
        )


def test_provider_id_stored_on_instance():
    p = _StubBase(provider_id="openai")
    assert p.provider_id == "openai"


def test_default_capability_is_openai_shape():
    p = _StubBase(provider_id="openai")
    # _capability_for falls back to the default descriptor when no override matches.
    cap = p._capability_for(
        ImagesModel(id="gpt-image-1", provider_id="openai")
    )
    assert cap.size_spec.kind == "size_string"


def test_capability_override_by_model_id():
    base_cap = ImagesCapabilityDescriptor()
    flux_cap = ImagesCapabilityDescriptor(size_spec=SizeSpec(kind="aspect_ratio"))
    p = _StubBase(
        provider_id="together",
        capability=base_cap,
        model_capability_overrides={"flux-schnell": flux_cap},
    )
    assert p._capability_for(
        ImagesModel(id="flux-schnell", provider_id="together")
    ).size_spec.kind == "aspect_ratio"
    assert p._capability_for(
        ImagesModel(id="flux-pro", provider_id="together")
    ).size_spec.kind == "size_string"


def test_model_factory_propagates_provider_id():
    p = _StubBase(provider_id="doubao")
    model = p.model("doubao-seedream-4-5-251128", api="doubao-images")
    assert isinstance(model, ImagesModel)
    assert model.id == "doubao-seedream-4-5-251128"
    assert model.provider_id == "doubao"
    assert model.api == "doubao-images"
    assert model.default_size is None  # nothing passed → None


def test_model_factory_passes_defaults_through():
    p = _StubBase(provider_id="openai")
    model = p.model(
        "gpt-image-1",
        default_size="1024x1024",
        default_n=2,
        default_quality="high",
        default_output_format="png",
        cost=ImagesCost(per_image=0.04),
        max_input_images=4,
    )
    assert model.default_size == "1024x1024"
    assert model.default_n == 2
    assert model.default_quality == "high"
    assert model.default_output_format == "png"
    assert model.cost is not None
    assert model.cost.per_image == 0.04
    assert model.max_input_images == 4


def test_subscribe_request_and_detach():
    p = _StubBase(provider_id="openai")
    events: list[dict] = []

    def cb(payload, model):
        events.append(payload)

    detach = p.subscribe_request(cb)
    assert cb in p._request_listeners
    detach()
    assert cb not in p._request_listeners


def test_subscribe_response_and_detach():
    p = _StubBase(provider_id="openai")
    events: list[BaseException | None] = []

    def cb(body, model, exc):
        events.append(exc)

    detach = p.subscribe_response(cb)
    assert cb in p._response_listeners
    detach()
    assert cb not in p._response_listeners


def test_no_subscribe_chunk_method():
    # Image is one-shot: there are no chunks, so no subscribe_chunk.
    p = _StubBase(provider_id="openai")
    assert not hasattr(p, "subscribe_chunk")
    assert not hasattr(p, "_chunk_listeners")


def test_base_generate_images_raises_not_implemented():
    base = BaseImagesProvider(provider_id="x")
    with pytest.raises(NotImplementedError):
        # Run as coroutine; we expect NotImplementedError at call time.
        import asyncio
        asyncio.run(base.generate_images(
            ImagesModel(id="x", provider_id="x"), ImagesContext(prompt="x"),
        ))


def test_protocol_runtime_check_accepts_stub():
    p = _StubBase(provider_id="openai")
    assert isinstance(p, ImagesProvider)
```

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest tests/providers/images/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: cubepi.providers.images.base`.

- [ ] **Step 3: Create `cubepi/providers/images/base.py`**

```python
from __future__ import annotations

from typing import Any, Callable, Literal, Protocol, runtime_checkable

from cubepi.providers.base import (
    OnRequestCallback,
    OnResponseBodyCallback,
    _detach,
)
from cubepi.providers.images.capability import ImagesCapabilityDescriptor
from cubepi.providers.images.types import (
    AssistantImages,
    ImagesContext,
    ImagesCost,
    ImagesModel,
    ImagesOptions,
)


@runtime_checkable
class ImagesProvider(Protocol):
    """Protocol for image-generation providers.

    Provider classes implement ``generate_images(model, context, options=...)``
    and expose ``provider_id``. They do NOT need to subclass
    :class:`BaseImagesProvider`, but built-in providers and most user
    implementations should — the base class supplies the ``.model()``
    factory, listener registry, and capability-application helper.
    """

    provider_id: str

    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        *,
        options: ImagesOptions | None = None,
    ) -> AssistantImages: ...


class BaseImagesProvider:
    """Concrete base class for built-in and user-defined image providers.

    Mirrors the role of :class:`cubepi.providers.base.BaseProvider` in chat:
    holds ``provider_id``, exposes a ``.model(...)`` factory that propagates
    it onto :class:`ImagesModel`, and runs request/response observer
    registries. Image is one-shot (no streamed chunks), so there is no
    ``subscribe_chunk``.
    """

    def __init__(
        self,
        *,
        provider_id: str = "",
        capability: ImagesCapabilityDescriptor | None = None,
        model_capability_overrides: dict[str, ImagesCapabilityDescriptor] | None = None,
    ) -> None:
        self.provider_id = provider_id
        self._capability = capability or ImagesCapabilityDescriptor()
        self._model_capability_overrides: dict[str, ImagesCapabilityDescriptor] = (
            dict(model_capability_overrides) if model_capability_overrides else {}
        )
        self._request_listeners: list[OnRequestCallback] = []
        self._response_listeners: list[OnResponseBodyCallback] = []

    # ──── Factory ────────────────────────────────────────────────
    def model(
        self,
        id: str,
        *,
        api: str = "",
        default_size: str | None = None,
        default_n: int | None = None,
        default_quality: Literal["low", "medium", "high"] | None = None,
        default_output_format: Literal["png", "jpeg", "webp"] | None = None,
        cost: ImagesCost | None = None,
        max_input_images: int | None = None,
    ) -> ImagesModel:
        return ImagesModel(
            id=id,
            provider_id=self.provider_id,
            api=api,
            default_size=default_size,
            default_n=default_n,
            default_quality=default_quality,
            default_output_format=default_output_format,
            cost=cost,
            max_input_images=max_input_images,
        )

    # ──── Protocol method — subclass implements ─────────────────
    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        *,
        options: ImagesOptions | None = None,
    ) -> AssistantImages:
        raise NotImplementedError

    # ──── Listener registry ──────────────────────────────────────
    def subscribe_request(self, cb: OnRequestCallback) -> Callable[[], None]:
        self._request_listeners.append(cb)
        return lambda: _detach(self._request_listeners, cb)

    def subscribe_response(self, cb: OnResponseBodyCallback) -> Callable[[], None]:
        self._response_listeners.append(cb)
        return lambda: _detach(self._response_listeners, cb)

    # ──── Helpers for subclasses ─────────────────────────────────
    def _capability_for(self, model: ImagesModel) -> ImagesCapabilityDescriptor:
        """Resolve the descriptor that applies to ``model`` (per-model override > base)."""
        return self._model_capability_overrides.get(model.id, self._capability)

    def _build_payload(
        self, model: ImagesModel, context: ImagesContext
    ) -> dict[str, Any]:  # pragma: no cover — implemented in Task 5
        raise NotImplementedError("_build_payload arrives in Task 5")
```

Note: `_build_payload` is intentionally a `NotImplementedError` placeholder; Task 5 fills it in. The `# pragma: no cover` keeps codecov honest about this being a stub.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/images/test_base.py -v`
Expected: PASS (all 10 tests pass).

- [ ] **Step 5: Run ruff and the rest of the suite**

Run:
```bash
uv run ruff check cubepi/providers/images/base.py tests/providers/images/test_base.py
uv run ruff format --check cubepi/providers/images/base.py tests/providers/images/test_base.py
uv run pytest tests/ -q
```
Expected: PASS for all three.

- [ ] **Step 6: Commit**

```bash
git add cubepi/providers/images/base.py tests/providers/images/test_base.py
git commit -m "feat(images): add ImagesProvider Protocol + BaseImagesProvider scaffolding

Mirrors BaseProvider on the chat side: provider_id field, .model(...) factory
that propagates provider_id onto ImagesModel, and request/response observer
registries. No subscribe_chunk (image is one-shot). _build_payload is
left as a NotImplementedError placeholder for Task 5."
```

---

## Task 5: Capability application — `_build_payload`

**Files:**
- Modify: `cubepi/providers/images/base.py` (implement `_build_payload`)
- Create: `tests/providers/images/test_capability_payload_mapping.py`

- [ ] **Step 1: Write the failing test**

Create `tests/providers/images/test_capability_payload_mapping.py`:

```python
import pytest

from cubepi.providers.images.base import BaseImagesProvider
from cubepi.providers.images.capability import (
    ImagesCapabilityDescriptor,
    SizeSpec,
)
from cubepi.providers.images.types import (
    AssistantImages,
    ImagesContext,
    ImagesModel,
)


class _Stub(BaseImagesProvider):
    async def generate_images(self, model, context, *, options=None):
        return AssistantImages(api="", provider_id="", model="", output=[])


def _model(model_id: str = "m", **defaults) -> ImagesModel:
    return ImagesModel(id=model_id, provider_id="p", **defaults)


def test_baseline_payload_has_model_and_prompt():
    p = _Stub(provider_id="p")
    payload = p._build_payload(_model(), ImagesContext(prompt="A robot"))
    assert payload["model"] == "m"
    assert payload["prompt"] == "A robot"
    assert payload["response_format"] == "b64_json"


def test_size_spec_size_string():
    p = _Stub(provider_id="p")
    payload = p._build_payload(_model(), ImagesContext(prompt="x", size="1024x1024"))
    assert payload["size"] == "1024x1024"
    assert "image_size" not in payload
    assert "width" not in payload


def test_size_spec_image_size_string():
    p = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(size_spec=SizeSpec(kind="image_size_string")),
    )
    payload = p._build_payload(_model(), ImagesContext(prompt="x", size="1024x1024"))
    assert payload["image_size"] == "1024x1024"
    assert "size" not in payload


def test_size_spec_width_height():
    p = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(size_spec=SizeSpec(kind="width_height")),
    )
    payload = p._build_payload(_model(), ImagesContext(prompt="x", size="1024x768"))
    assert payload["width"] == 1024
    assert payload["height"] == 768
    assert "size" not in payload


def test_size_spec_aspect_ratio():
    p = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(size_spec=SizeSpec(kind="aspect_ratio")),
    )
    payload = p._build_payload(_model(), ImagesContext(prompt="x", size="1:1"))
    assert payload["aspect_ratio"] == "1:1"


def test_size_falls_back_to_model_default():
    p = _Stub(provider_id="p")
    payload = p._build_payload(
        _model(default_size="2K"), ImagesContext(prompt="x"),
    )
    assert payload["size"] == "2K"


def test_ctx_size_overrides_model_default():
    p = _Stub(provider_id="p")
    payload = p._build_payload(
        _model(default_size="2K"), ImagesContext(prompt="x", size="1K"),
    )
    assert payload["size"] == "1K"


def test_size_omitted_when_both_none():
    p = _Stub(provider_id="p")
    payload = p._build_payload(_model(), ImagesContext(prompt="x"))
    assert "size" not in payload


def test_count_field_default_n():
    p = _Stub(provider_id="p")
    payload = p._build_payload(_model(), ImagesContext(prompt="x", n=2))
    assert payload["n"] == 2


def test_count_field_renamed_to_batch_size():
    p = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(count_field="batch_size"),
    )
    payload = p._build_payload(_model(), ImagesContext(prompt="x", n=3))
    assert payload["batch_size"] == 3
    assert "n" not in payload


def test_quality_written_when_set():
    p = _Stub(provider_id="p")
    payload = p._build_payload(_model(), ImagesContext(prompt="x", quality="high"))
    assert payload["quality"] == "high"


def test_output_format_renamed_or_dropped():
    # Default name preserved
    p1 = _Stub(provider_id="p")
    payload1 = p1._build_payload(_model(), ImagesContext(prompt="x", output_format="webp"))
    assert payload1["output_format"] == "webp"

    # output_format_field=None → silently dropped
    p2 = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(output_format_field=None),
    )
    payload2 = p2._build_payload(_model(), ImagesContext(prompt="x", output_format="webp"))
    assert "output_format" not in payload2


def test_supports_seed_gating_on():
    p = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(supports_seed=True),
    )
    payload = p._build_payload(_model(), ImagesContext(prompt="x", seed=42))
    assert payload["seed"] == 42


def test_supports_seed_gating_off_drops_silently():
    p = _Stub(provider_id="p")  # default supports_seed=False
    payload = p._build_payload(_model(), ImagesContext(prompt="x", seed=42))
    assert "seed" not in payload


def test_seed_field_rename():
    p = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(supports_seed=True, seed_field="rng_seed"),
    )
    payload = p._build_payload(_model(), ImagesContext(prompt="x", seed=42))
    assert payload["rng_seed"] == 42


@pytest.mark.parametrize(
    "flag,field_default,ctx_field,ctx_value",
    [
        ("supports_negative_prompt", "negative_prompt", "negative_prompt", "blurry"),
        ("supports_steps",           "num_inference_steps", "steps", 20),
        ("supports_guidance",        "guidance_scale", "guidance", 7.5),
    ],
)
def test_optional_field_gates(flag, field_default, ctx_field, ctx_value):
    p_off = _Stub(provider_id="p")
    p_on = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(**{flag: True}),
    )
    ctx = ImagesContext(prompt="x", **{ctx_field: ctx_value})
    payload_off = p_off._build_payload(_model(), ctx)
    payload_on = p_on._build_payload(_model(), ctx)
    assert field_default not in payload_off
    assert payload_on[field_default] == ctx_value


def test_extra_payload_deep_merge():
    p = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(
            extra_payload={"watermark": False, "nested": {"a": 1}},
        ),
    )
    payload = p._build_payload(
        _model(),
        ImagesContext(prompt="x", extra={"nested": {"b": 2}, "seed_override": "ignored"}),
    )
    assert payload["watermark"] is False
    # ctx.extra deep-merges over capability.extra_payload
    assert payload["nested"] == {"a": 1, "b": 2}
    assert payload["seed_override"] == "ignored"


def test_ctx_extra_wins_over_capability_extra_on_collision():
    p = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(extra_payload={"watermark": False}),
    )
    payload = p._build_payload(
        _model(),
        ImagesContext(prompt="x", extra={"watermark": True}),
    )
    assert payload["watermark"] is True


def test_model_capability_overrides_per_model():
    cap_default = ImagesCapabilityDescriptor()
    cap_flux_schnell = ImagesCapabilityDescriptor(size_spec=SizeSpec(kind="aspect_ratio"))
    p = _Stub(
        provider_id="together",
        capability=cap_default,
        model_capability_overrides={"flux-schnell": cap_flux_schnell},
    )
    payload_pro = p._build_payload(
        _model("flux-pro"), ImagesContext(prompt="x", size="1024x1024"),
    )
    payload_schnell = p._build_payload(
        _model("flux-schnell"), ImagesContext(prompt="x", size="1:1"),
    )
    assert "size" in payload_pro
    assert payload_schnell["aspect_ratio"] == "1:1"
```

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest tests/providers/images/test_capability_payload_mapping.py -v`
Expected: FAIL with `NotImplementedError: _build_payload arrives in Task 5`.

- [ ] **Step 3: Implement `_build_payload` (and an internal `_deep_merge` helper)**

Replace the placeholder `_build_payload` in `cubepi/providers/images/base.py` with:

```python
    def _build_payload(
        self, model: ImagesModel, context: ImagesContext
    ) -> dict[str, Any]:
        cap = self._capability_for(model)
        payload: dict[str, Any] = {"model": model.id, "prompt": context.prompt}

        # size — four wire shapes
        size = context.size if context.size is not None else model.default_size
        if size is not None:
            kind = cap.size_spec.kind
            if kind == "size_string":
                payload["size"] = size
            elif kind == "image_size_string":
                payload["image_size"] = size
            elif kind == "width_height":
                w_str, h_str = size.lower().split("x")
                payload["width"] = int(w_str)
                payload["height"] = int(h_str)
            elif kind == "aspect_ratio":
                payload["aspect_ratio"] = size

        # n
        n = context.n if context.n is not None else model.default_n
        if n is not None:
            payload[cap.count_field] = n

        # quality
        quality = context.quality if context.quality is not None else model.default_quality
        if quality is not None:
            payload["quality"] = quality

        # output_format
        of = (
            context.output_format
            if context.output_format is not None
            else model.default_output_format
        )
        if of is not None and cap.output_format_field is not None:
            payload[cap.output_format_field] = of

        # response_format — always written, controlled entirely by capability
        payload[cap.response_format_field] = cap.response_format_value

        # supports_* gating
        if context.seed is not None and cap.supports_seed:
            payload[cap.seed_field] = context.seed
        if context.negative_prompt is not None and cap.supports_negative_prompt:
            payload[cap.negative_prompt_field] = context.negative_prompt
        if context.steps is not None and cap.supports_steps:
            payload[cap.steps_field] = context.steps
        if context.guidance is not None and cap.supports_guidance:
            payload[cap.guidance_field] = context.guidance

        # Provider-level extra_payload (always injected) and per-call ctx.extra,
        # deep-merged in order: payload <- cap.extra_payload <- ctx.extra.
        payload = _deep_merge(payload, cap.extra_payload)
        payload = _deep_merge(payload, context.extra)
        return payload
```

Add the helper at module top-level (above the class):

```python
def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursive merge that copies ``base`` and applies ``overlay`` on top.

    Dict keys recurse; everything else (lists, scalars) is overwritten by
    the overlay value. Used for ``capability.extra_payload`` and
    ``context.extra`` application.
    """
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/images/test_capability_payload_mapping.py -v`
Expected: PASS (all ~20 tests pass; the three parametrized cases count as three each).

- [ ] **Step 5: Run ruff and the rest of the suite**

Run:
```bash
uv run ruff check cubepi/providers/images/base.py tests/providers/images/test_capability_payload_mapping.py
uv run ruff format --check cubepi/providers/images/base.py tests/providers/images/test_capability_payload_mapping.py
uv run pytest tests/ -q
```
Expected: PASS for all three.

- [ ] **Step 6: Commit**

```bash
git add cubepi/providers/images/base.py tests/providers/images/test_capability_payload_mapping.py
git commit -m "feat(images): implement _build_payload with full capability application

Translates ImagesContext + ImagesModel defaults into a wire payload dict
according to the capability descriptor: four size_spec shapes, count_field
rename, output_format rename/drop, supports_* gating for seed/negative_prompt/
steps/guidance with renameable field names, deep-merged provider-level
extra_payload, and per-call ctx.extra overlay (ctx wins on collision)."
```

---

## Task 6: Widen `classify_and_raise` to accept `ImagesModel`

**Files:**
- Modify: `cubepi/errors.py`
- Create: `tests/errors/test_classify_images.py`

- [ ] **Step 1: Inspect the current signature and dependencies**

Open `cubepi/errors.py` and locate `classify_and_raise` (around line 176). Confirm it currently reads `model.provider_id`, `model.id`, and `model.context_window`. The widening must:
- Read `context_window` via `getattr(model, "context_window", None)` so types without that attribute (image) work.
- Replace the `Model` annotation with a `Protocol` that requires `provider_id: str` and `id: str` only.

- [ ] **Step 2: Write the failing test**

Create `tests/errors/test_classify_images.py` (the `tests/errors/` directory may need an `__init__.py` if it doesn't exist; check `ls tests/errors/` first and `touch tests/errors/__init__.py` if missing):

```python
import pytest

from cubepi.errors import (
    ContextLengthExceeded,
    ProviderAuthFailed,
    ProviderBadRequest,
    ProviderError,
    ProviderUnavailable,
    RateLimited,
    classify_and_raise,
)
from cubepi.providers.images.types import ImagesModel


def _img_model() -> ImagesModel:
    return ImagesModel(id="gpt-image-1", provider_id="openai", api="openai-images")


class _StatusErr(Exception):
    def __init__(self, msg: str, status: int) -> None:
        super().__init__(msg)
        self.status_code = status


def test_rate_limit_via_status_429():
    with pytest.raises(RateLimited):
        classify_and_raise(_StatusErr("too many", 429), model=_img_model())


def test_auth_via_status_401():
    with pytest.raises(ProviderAuthFailed):
        classify_and_raise(_StatusErr("nope", 401), model=_img_model())


def test_auth_via_status_403():
    with pytest.raises(ProviderAuthFailed):
        classify_and_raise(_StatusErr("forbidden", 403), model=_img_model())


def test_unavailable_via_status_503():
    with pytest.raises(ProviderUnavailable):
        classify_and_raise(_StatusErr("down", 503), model=_img_model())


def test_bad_request_via_status_400():
    with pytest.raises(ProviderBadRequest):
        classify_and_raise(_StatusErr("bad", 400), model=_img_model())


def test_context_length_pattern_match():
    # Pattern wording wins regardless of status; just ensures images-side
    # model lookup doesn't blow up reading context_window.
    with pytest.raises(ContextLengthExceeded):
        classify_and_raise(
            _StatusErr("Request exceeds maximum context length", 400),
            model=_img_model(),
        )


def test_already_typed_passthrough():
    err = RateLimited("manual", provider="openai", model="gpt-image-1")
    with pytest.raises(RateLimited) as exc:
        classify_and_raise(err, model=_img_model())
    assert exc.value is err


def test_unknown_exception_falls_through_reraise():
    class _Weird(Exception): ...
    weird = _Weird("unknown")
    with pytest.raises(_Weird):
        classify_and_raise(weird, model=_img_model())


def test_chat_model_still_works():
    """Widening must not break the existing chat-side call path."""
    from cubepi.providers.base import Model

    chat_model = Model(id="claude-sonnet-4-6", provider_id="anthropic", context_window=200_000)
    with pytest.raises(RateLimited):
        classify_and_raise(_StatusErr("limit", 429), model=chat_model)


def test_uses_default_context_window_when_attribute_absent():
    """ImagesModel has no context_window; the function must not raise AttributeError."""
    assert not hasattr(_img_model(), "context_window")
    # Status 400 without context-length wording → ProviderBadRequest (not Context).
    with pytest.raises(ProviderBadRequest):
        classify_and_raise(_StatusErr("plain bad request", 400), model=_img_model())
```

- [ ] **Step 3: Run the failing test**

Run: `uv run pytest tests/errors/test_classify_images.py -v`
Expected: FAIL with `AttributeError: 'ImagesModel' object has no attribute 'context_window'` (because the old code accesses `model.context_window` directly).

- [ ] **Step 4: Widen `classify_and_raise` in `cubepi/errors.py`**

Locate the existing signature (around line 176) and edit:

```python
# Before
from cubepi.providers.base import Message, Model
...
def classify_and_raise(
    exc: BaseException,
    *,
    model: Model,
    messages: list[Message] | None = None,
) -> NoReturn:
    ...
    provider = model.provider_id
    model_id = model.id
    tokens_in = _estimate_input_tokens(messages)
    context_window = model.context_window if model.context_window else None
```

becomes:

```python
# After — top of file, add the structural protocol
from typing import Protocol


class _ClassifyTarget(Protocol):
    """Structural type for the `model` parameter of `classify_and_raise`.

    Both `cubepi.providers.base.Model` and
    `cubepi.providers.images.types.ImagesModel` satisfy it; `context_window`
    is read with a getattr fallback so image models (which don't have it)
    skip the token-budget heuristic cleanly.
    """

    id: str
    provider_id: str


# ...later in the function signature:
def classify_and_raise(
    exc: BaseException,
    *,
    model: _ClassifyTarget,
    messages: list[Message] | None = None,
) -> NoReturn:
    ...
    provider = model.provider_id
    model_id = model.id
    tokens_in = _estimate_input_tokens(messages)
    cw_val = getattr(model, "context_window", None)
    context_window = cw_val if cw_val else None
```

Leave the rest of the function body unchanged.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/errors/ -v`
Expected: PASS (new image tests + any existing tests in `tests/errors/`).

- [ ] **Step 6: Run ruff, mypy on `cubepi`, and the full suite**

Run:
```bash
uv run ruff check cubepi/errors.py tests/errors/test_classify_images.py
uv run ruff format --check cubepi/errors.py tests/errors/test_classify_images.py
uv run mypy cubepi
uv run pytest tests/ -q
```
Expected: PASS for all four.

- [ ] **Step 7: Commit**

```bash
git add cubepi/errors.py tests/errors/
git commit -m "feat(errors): widen classify_and_raise to accept ImagesModel

classify_and_raise(model=...) is now typed against a structural protocol
satisfied by both Model (chat) and ImagesModel (image). context_window is
read via getattr with a None fallback so image models — which have no
context_window field — skip the token-budget heuristic cleanly. Chat call
sites are unaffected; image providers can now reuse the same typed
ProviderError taxonomy."
```

---

## Task 7: Rewrite `FauxImagesProvider`

**Files:**
- Create: `cubepi/providers/images/faux.py`
- Create: `tests/providers/images/test_faux_provider.py`

- [ ] **Step 1: Write the failing test**

Create `tests/providers/images/test_faux_provider.py`:

```python
import base64

import pytest

from cubepi.errors import ProviderError, RateLimited
from cubepi.providers.images.faux import FauxImagesProvider
from cubepi.providers.images.types import ImagesContext


def _png_b64() -> str:
    return base64.b64encode(b"\x89PNG-stub").decode()


@pytest.mark.asyncio
async def test_happy_path_returns_image():
    p = FauxImagesProvider(png_b64=_png_b64())
    model = p.model("faux-1", api="faux-images")
    out = await p.generate_images(model, ImagesContext(prompt="x"))
    assert out.stop_reason == "stop"
    assert len(out.output) == 1
    assert out.output[0].type == "image"
    assert out.output[0].media_type == "image/png"
    assert out.provider_id == "faux"


@pytest.mark.asyncio
async def test_custom_provider_id_propagates():
    p = FauxImagesProvider(provider_id="custom-faux", png_b64=_png_b64())
    model = p.model("faux-1")
    assert model.provider_id == "custom-faux"
    out = await p.generate_images(model, ImagesContext(prompt="x"))
    assert out.provider_id == "custom-faux"


@pytest.mark.asyncio
async def test_raise_on_call_injects_typed_error():
    p = FauxImagesProvider(png_b64=_png_b64(), raise_on_call=RateLimited)
    model = p.model("faux-1")
    with pytest.raises(RateLimited) as exc:
        await p.generate_images(model, ImagesContext(prompt="x"))
    assert "faux-1" in str(exc.value)


@pytest.mark.asyncio
async def test_raise_on_call_can_be_base_provider_error():
    p = FauxImagesProvider(png_b64=_png_b64(), raise_on_call=ProviderError)
    model = p.model("faux-1")
    with pytest.raises(ProviderError):
        await p.generate_images(model, ImagesContext(prompt="x"))


def test_inherits_from_base_images_provider():
    from cubepi.providers.images.base import BaseImagesProvider
    p = FauxImagesProvider(png_b64=_png_b64())
    assert isinstance(p, BaseImagesProvider)
    # listener registry is inherited
    assert hasattr(p, "subscribe_request")
    assert hasattr(p, "subscribe_response")
    assert not hasattr(p, "subscribe_chunk")
```

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest tests/providers/images/test_faux_provider.py -v`
Expected: FAIL with `ModuleNotFoundError: cubepi.providers.images.faux`.

- [ ] **Step 3: Create `cubepi/providers/images/faux.py`**

```python
from __future__ import annotations

from cubepi.errors import ProviderError
from cubepi.providers.base import ImageContent
from cubepi.providers.images.base import BaseImagesProvider
from cubepi.providers.images.types import (
    AssistantImages,
    ImagesContext,
    ImagesModel,
    ImagesOptions,
)


class FauxImagesProvider(BaseImagesProvider):
    """Deterministic image-generation stub for tests.

    Returns a single image whose b64 body is the value passed at construction
    time. ``raise_on_call`` lets tests inject a typed ``ProviderError``
    subclass (e.g. ``RateLimited``) so retry middleware can be exercised
    against the image path.
    """

    def __init__(
        self,
        *,
        provider_id: str = "faux",
        png_b64: str,
        raise_on_call: type[ProviderError] | None = None,
    ) -> None:
        super().__init__(provider_id=provider_id)
        self._png_b64 = png_b64
        self._raise = raise_on_call

    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        *,
        options: ImagesOptions | None = None,
    ) -> AssistantImages:
        if self._raise is not None:
            raise self._raise(
                f"injected by FauxImagesProvider for {model.provider_id}/{model.id}"
            )
        return AssistantImages(
            api=model.api,
            provider_id=model.provider_id,
            model=model.id,
            output=[ImageContent(source=self._png_b64, media_type="image/png")],
            stop_reason="stop",
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/images/test_faux_provider.py -v`
Expected: PASS (all 5 tests pass).

- [ ] **Step 5: Run ruff and the rest of the suite**

Run:
```bash
uv run ruff check cubepi/providers/images/faux.py tests/providers/images/test_faux_provider.py
uv run ruff format --check cubepi/providers/images/faux.py tests/providers/images/test_faux_provider.py
uv run pytest tests/ -q
```
Expected: PASS for all three.

- [ ] **Step 6: Commit**

```bash
git add cubepi/providers/images/faux.py tests/providers/images/test_faux_provider.py
git commit -m "feat(images): rebuild FauxImagesProvider on BaseImagesProvider

Inherits provider_id propagation, .model() factory, and listener registry
from BaseImagesProvider. Adds a raise_on_call parameter so tests can inject
a typed ProviderError subclass (e.g. RateLimited) — useful for exercising
retry middleware against the image path without standing up real backends."
```

---

## Task 8: Rewrite `OpenAIImagesProvider`

**Files:**
- Create: `cubepi/providers/images/openai_images.py`
- Create: `tests/providers/images/test_openai_images.py`

- [ ] **Step 1: Write the failing test**

Create `tests/providers/images/test_openai_images.py`:

```python
import asyncio
import base64
from types import SimpleNamespace

import pytest

from cubepi.errors import (
    ProviderAuthFailed,
    ProviderUnavailable,
    RateLimited,
)
from cubepi.providers.base import ImageContent
from cubepi.providers.images.capability import (
    ImagesCapabilityDescriptor,
    SizeSpec,
)
from cubepi.providers.images.openai_images import OpenAIImagesProvider
from cubepi.providers.images.types import (
    ImagesContext,
    ImagesOptions,
)


class _StatusErr(Exception):
    def __init__(self, msg: str, status: int) -> None:
        super().__init__(msg)
        self.status_code = status


class _FakeImages:
    def __init__(self, exc: Exception | None = None, sleep: float = 0.0):
        self.generate_kwargs: dict | None = None
        self.edit_kwargs: dict | None = None
        self._exc = exc
        self._sleep = sleep

    async def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        if self._sleep:
            await asyncio.sleep(self._sleep)
        if self._exc:
            raise self._exc
        return SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(b"GEN").decode())]
        )

    async def edit(self, **kwargs):
        self.edit_kwargs = kwargs
        return SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(b"EDIT").decode())]
        )


class _FakeClient:
    def __init__(self, exc: Exception | None = None, sleep: float = 0.0):
        self.images = _FakeImages(exc=exc, sleep=sleep)


def _provider(
    *,
    capability: ImagesCapabilityDescriptor | None = None,
    exc: Exception | None = None,
    sleep: float = 0.0,
) -> OpenAIImagesProvider:
    p = OpenAIImagesProvider(provider_id="openai", api_key="sk-test", capability=capability)
    p._client = _FakeClient(exc=exc, sleep=sleep)
    return p


@pytest.mark.asyncio
async def test_provider_id_propagated_through_factory():
    p = _provider()
    model = p.model("gpt-image-1", api="openai-images")
    assert model.provider_id == "openai"


@pytest.mark.asyncio
async def test_happy_path_writes_canonical_openai_fields():
    p = _provider()
    model = p.model("gpt-image-1", api="openai-images")
    out = await p.generate_images(
        model,
        ImagesContext(prompt="A cat", size="1024x1024", n=2, quality="high"),
    )
    kw = p._client.images.generate_kwargs
    assert kw["model"] == "gpt-image-1"
    assert kw["prompt"] == "A cat"
    assert kw["size"] == "1024x1024"
    assert kw["n"] == 2
    assert kw["quality"] == "high"
    assert kw["response_format"] == "b64_json"
    assert out.stop_reason == "stop"
    assert out.output[0].type == "image"


@pytest.mark.asyncio
async def test_doubao_extra_payload_injection():
    p = _provider(
        capability=ImagesCapabilityDescriptor(
            supports_seed=True, extra_payload={"watermark": False},
        ),
    )
    model = p.model("doubao-seedream-4-5-251128")
    await p.generate_images(model, ImagesContext(prompt="x", seed=42))
    kw = p._client.images.generate_kwargs
    assert kw["watermark"] is False
    assert kw["seed"] == 42


@pytest.mark.asyncio
async def test_siliconflow_field_remap():
    p = _provider(
        capability=ImagesCapabilityDescriptor(
            size_spec=SizeSpec(kind="image_size_string"),
            count_field="batch_size",
            output_format_field=None,
        ),
    )
    model = p.model("Kwai-Kolors/Kolors")
    await p.generate_images(
        model,
        ImagesContext(prompt="x", size="1024x1024", n=2, output_format="png"),
    )
    kw = p._client.images.generate_kwargs
    assert kw["image_size"] == "1024x1024"
    assert kw["batch_size"] == 2
    assert "n" not in kw
    assert "output_format" not in kw  # dropped because output_format_field=None


@pytest.mark.asyncio
async def test_edit_path_when_input_images_provided():
    p = _provider()
    model = p.model("gpt-image-1")
    ctx = ImagesContext(
        prompt="make blue",
        input_images=[
            ImageContent(source=base64.b64encode(b"SRC").decode(), media_type="image/png"),
        ],
    )
    await p.generate_images(model, ctx)
    assert p._client.images.edit_kwargs is not None
    assert p._client.images.generate_kwargs is None


@pytest.mark.asyncio
async def test_edit_path_disabled_by_capability():
    p = _provider(
        capability=ImagesCapabilityDescriptor(supports_edit=False),
    )
    model = p.model("gpt-image-1")
    ctx = ImagesContext(
        prompt="x",
        input_images=[
            ImageContent(source=base64.b64encode(b"SRC").decode(), media_type="image/png"),
        ],
    )
    await p.generate_images(model, ctx)
    # Falls back to generate path even with input_images
    assert p._client.images.generate_kwargs is not None
    assert p._client.images.edit_kwargs is None


@pytest.mark.asyncio
async def test_rate_limit_raises_typed_error():
    p = _provider(exc=_StatusErr("limit", 429))
    model = p.model("gpt-image-1")
    with pytest.raises(RateLimited):
        await p.generate_images(model, ImagesContext(prompt="x"))


@pytest.mark.asyncio
async def test_auth_failure_raises_typed_error():
    p = _provider(exc=_StatusErr("nope", 401))
    model = p.model("gpt-image-1")
    with pytest.raises(ProviderAuthFailed):
        await p.generate_images(model, ImagesContext(prompt="x"))


@pytest.mark.asyncio
async def test_unavailable_raises_typed_error():
    p = _provider(exc=_StatusErr("down", 503))
    model = p.model("gpt-image-1")
    with pytest.raises(ProviderUnavailable):
        await p.generate_images(model, ImagesContext(prompt="x"))


@pytest.mark.asyncio
async def test_signal_cancels_and_returns_aborted():
    """Setting the signal mid-call returns AssistantImages(stop_reason='aborted')."""
    p = _provider(sleep=0.5)  # SDK call sleeps long enough to cancel
    model = p.model("gpt-image-1")
    signal = asyncio.Event()

    async def trigger():
        await asyncio.sleep(0.05)
        signal.set()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(trigger())
        task = tg.create_task(
            p.generate_images(model, ImagesContext(prompt="x"), options=ImagesOptions(signal=signal))
        )

    result = task.result()
    assert result.stop_reason == "aborted"
    assert result.output == []


@pytest.mark.asyncio
async def test_per_call_on_payload_mutates_outgoing():
    p = _provider()
    model = p.model("gpt-image-1")

    def mutate(payload, _model):
        payload["custom_tag"] = "tracing-id-123"
        return payload

    await p.generate_images(
        model,
        ImagesContext(prompt="x"),
        options=ImagesOptions(on_payload=mutate),
    )
    assert p._client.images.generate_kwargs["custom_tag"] == "tracing-id-123"


@pytest.mark.asyncio
async def test_subscribe_request_fires_with_final_payload():
    p = _provider()
    model = p.model("gpt-image-1")
    seen: list[dict] = []
    p.subscribe_request(lambda payload, m: seen.append(payload))
    await p.generate_images(model, ImagesContext(prompt="x", size="1024x1024"))
    assert len(seen) == 1
    assert seen[0]["model"] == "gpt-image-1"
    assert seen[0]["size"] == "1024x1024"


@pytest.mark.asyncio
async def test_subscribe_response_fires_on_success():
    p = _provider()
    model = p.model("gpt-image-1")
    seen: list[BaseException | None] = []
    p.subscribe_response(lambda body, m, exc: seen.append(exc))
    await p.generate_images(model, ImagesContext(prompt="x"))
    assert seen == [None]


@pytest.mark.asyncio
async def test_subscribe_response_fires_on_error_with_exception():
    p = _provider(exc=_StatusErr("limit", 429))
    model = p.model("gpt-image-1")
    seen: list[BaseException | None] = []
    p.subscribe_response(lambda body, m, exc: seen.append(exc))
    with pytest.raises(RateLimited):
        await p.generate_images(model, ImagesContext(prompt="x"))
    assert len(seen) == 1
    assert seen[0] is not None
    assert "limit" in str(seen[0])


def test_base_url_accepted_and_constructor_works():
    p = OpenAIImagesProvider(
        provider_id="doubao",
        api_key="sk-test",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    assert p.provider_id == "doubao"
```

- [ ] **Step 2: Run the failing test**

Run: `uv run pytest tests/providers/images/test_openai_images.py -v`
Expected: FAIL with `ModuleNotFoundError: cubepi.providers.images.openai_images`.

- [ ] **Step 3: Create `cubepi/providers/images/openai_images.py`**

```python
from __future__ import annotations

import asyncio
import base64
import io
from typing import Any

from cubepi.errors import classify_and_raise
from cubepi.providers.base import (
    ImageContent,
    OnPayloadCallback,
    ProviderResponse,
    TextContent,
    _fire_request_listeners,
    _fire_response_listeners,
    invoke_on_payload,
    invoke_on_response,
)
from cubepi.providers.images.base import BaseImagesProvider
from cubepi.providers.images.capability import ImagesCapabilityDescriptor
from cubepi.providers.images.types import (
    AssistantImages,
    ImagesContext,
    ImagesModel,
    ImagesOptions,
)


_MEDIA_TYPE_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
}

_OUTPUT_FORMAT_MEDIA_TYPE = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


class OpenAIImagesProvider(BaseImagesProvider):
    """OpenAI-shape image provider.

    With the default ``ImagesCapabilityDescriptor`` this targets OpenAI's
    own ``/v1/images/generations`` endpoint. By supplying a different
    capability descriptor (and a matching ``base_url``) the same class
    targets other OpenAI-compatible backends: Volcengine Ark / Doubao
    Seedream, SiliconFlow, Together AI, and similar.
    """

    def __init__(
        self,
        *,
        provider_id: str = "",
        api_key: str | None = None,
        base_url: str | None = None,
        capability: ImagesCapabilityDescriptor | None = None,
        model_capability_overrides: dict[str, ImagesCapabilityDescriptor] | None = None,
    ) -> None:
        super().__init__(
            provider_id=provider_id,
            capability=capability,
            model_capability_overrides=model_capability_overrides,
        )
        import openai

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client: Any = openai.AsyncOpenAI(**kwargs)

    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        *,
        options: ImagesOptions | None = None,
    ) -> AssistantImages:
        cap = self._capability_for(model)
        payload = self._build_payload(model, context)

        # input_images go in via the capability-declared field name; not
        # merged by _build_payload because they're not a wire dict slot.
        if context.input_images and cap.supports_edit:
            payload[cap.input_images_field] = [
                self._to_file(img) for img in context.input_images
            ]
            sdk_call = self._client.images.edit
        else:
            sdk_call = self._client.images.generate

        # per-call on_payload + persistent subscribe_request
        on_payload: OnPayloadCallback | None = options.on_payload if options else None
        payload = await invoke_on_payload(on_payload, payload, model)
        if self._request_listeners:
            await _fire_request_listeners(self._request_listeners, payload, model)

        body: dict | None = None
        exc: BaseException | None = None
        try:
            sdk_resp = await self._run_with_signal(
                sdk_call, payload, options.signal if options else None,
            )
            body = self._resp_to_dict(sdk_resp)

            if options and options.on_response:
                await invoke_on_response(
                    options.on_response, ProviderResponse(status=200), model,
                )

            return self._parse_response(sdk_resp, model, cap)

        except asyncio.CancelledError:
            # Signal-triggered abort: return as AssistantImages, do not re-raise.
            return AssistantImages(
                api=model.api,
                provider_id=model.provider_id,
                model=model.id,
                output=[],
                stop_reason="aborted",
            )
        except Exception as raw:  # noqa: BLE001
            exc = raw
            classify_and_raise(raw, model=model)
        finally:
            if self._response_listeners:
                await _fire_response_listeners(
                    self._response_listeners, body, model, exc,
                )

    # ──── internals ──────────────────────────────────────────────
    async def _run_with_signal(
        self,
        sdk_call: Any,
        payload: dict[str, Any],
        signal: asyncio.Event | None,
    ) -> Any:
        """Run ``sdk_call(**payload)`` while listening to ``signal``.

        If ``signal`` fires first, the SDK task is cancelled and
        ``asyncio.CancelledError`` propagates upward.
        """
        if signal is None:
            return await sdk_call(**payload)

        sdk_task = asyncio.ensure_future(sdk_call(**payload))
        signal_task = asyncio.ensure_future(signal.wait())
        try:
            done, _ = await asyncio.wait(
                {sdk_task, signal_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if signal_task in done and not sdk_task.done():
                sdk_task.cancel()
                raise asyncio.CancelledError
            signal_task.cancel()
            return sdk_task.result()
        finally:
            if not signal_task.done():
                signal_task.cancel()
            if not sdk_task.done():
                sdk_task.cancel()

    def _parse_response(
        self,
        resp: Any,
        model: ImagesModel,
        cap: ImagesCapabilityDescriptor,
    ) -> AssistantImages:
        # Determine output media type from output_format if present.
        out_format = "png"
        data = getattr(resp, "data", None) or []
        images: list[ImageContent | TextContent] = [
            ImageContent(
                source=item.b64_json,
                media_type=_OUTPUT_FORMAT_MEDIA_TYPE.get(out_format, "image/png"),
            )
            for item in data
            if getattr(item, "b64_json", None)
        ]
        if not images:
            # Empty response: still a "stop" — but no images. Surface via empty output.
            return AssistantImages(
                api=model.api,
                provider_id=model.provider_id,
                model=model.id,
                output=[],
                stop_reason="stop",
            )
        return AssistantImages(
            api=model.api,
            provider_id=model.provider_id,
            model=model.id,
            output=images,
            stop_reason="stop",
        )

    @staticmethod
    def _resp_to_dict(resp: Any) -> dict[str, Any]:
        if hasattr(resp, "model_dump"):
            try:
                return resp.model_dump()
            except Exception:  # noqa: BLE001
                pass
        return {"data": getattr(resp, "data", [])}

    @staticmethod
    def _to_file(img: ImageContent) -> io.BytesIO:
        ext = _MEDIA_TYPE_EXT.get(img.media_type, "png")
        buf = io.BytesIO(base64.b64decode(img.source))
        buf.name = f"source.{ext}"
        return buf
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/providers/images/test_openai_images.py -v`
Expected: PASS (all 15 tests pass).

- [ ] **Step 5: Run ruff, mypy on `cubepi`, and the full suite**

Run:
```bash
uv run ruff check cubepi/providers/images/openai_images.py tests/providers/images/test_openai_images.py
uv run ruff format --check cubepi/providers/images/openai_images.py tests/providers/images/test_openai_images.py
uv run mypy cubepi
uv run pytest tests/ -q
```
Expected: PASS for all four.

- [ ] **Step 6: Commit**

```bash
git add cubepi/providers/images/openai_images.py tests/providers/images/test_openai_images.py
git commit -m "feat(images): rebuild OpenAIImagesProvider with capability + cancel + listeners

Inherits BaseImagesProvider for provider_id + .model() + listener registry.
Applies _build_payload (capability-aware payload assembly), honors
options.signal via asyncio.wait race (returns AssistantImages with
stop_reason='aborted' on cancellation), classifies SDK exceptions via
classify_and_raise into typed ProviderError subclasses, fires
subscribe_request before send and subscribe_response in finally."
```

---

## Task 9: Public exports — `__init__.py` files

**Files:**
- Modify: `cubepi/providers/images/__init__.py`
- Modify: `cubepi/__init__.py`

- [ ] **Step 1: Rewrite `cubepi/providers/images/__init__.py`**

Replace the stub with:

```python
"""Image generation providers — public exports."""

from cubepi.providers.images.base import (
    BaseImagesProvider,
    ImagesProvider,
)
from cubepi.providers.images.capability import (
    ImagesCapabilityDescriptor,
    SizeSpec,
)
from cubepi.providers.images.faux import FauxImagesProvider
from cubepi.providers.images.openai_images import OpenAIImagesProvider
from cubepi.providers.images.types import (
    AssistantImages,
    ImagesContext,
    ImagesCost,
    ImagesModel,
    ImagesOptions,
)

__all__ = [
    "AssistantImages",
    "BaseImagesProvider",
    "FauxImagesProvider",
    "ImagesCapabilityDescriptor",
    "ImagesContext",
    "ImagesCost",
    "ImagesModel",
    "ImagesOptions",
    "ImagesProvider",
    "OpenAIImagesProvider",
    "SizeSpec",
]
```

- [ ] **Step 2: Update `cubepi/__init__.py` to re-export the image types**

Insert imports below the chat-side `from cubepi.providers.capability` block:

```python
from cubepi.providers.images import (
    AssistantImages,
    BaseImagesProvider,
    ImagesCapabilityDescriptor,
    ImagesContext,
    ImagesCost,
    ImagesModel,
    ImagesOptions,
    ImagesProvider,
    SizeSpec,
)
```

Then add the same nine names to `__all__` in alphabetical position. Concrete providers (`OpenAIImagesProvider`, `FauxImagesProvider`) are **not** exported at the top level — chat does the same (`AnthropicProvider`/`OpenAIProvider` are reachable only via `cubepi.providers.anthropic` / `cubepi.providers.openai`).

- [ ] **Step 3: Add a smoke test to confirm exports work**

Append to `tests/providers/images/test_types.py`:

```python
def test_top_level_re_exports():
    """The image types are reachable via `from cubepi import ...`."""
    import cubepi
    assert cubepi.ImagesModel is __import__(
        "cubepi.providers.images", fromlist=["ImagesModel"]
    ).ImagesModel
    for name in (
        "AssistantImages", "BaseImagesProvider", "ImagesCapabilityDescriptor",
        "ImagesContext", "ImagesCost", "ImagesModel", "ImagesOptions",
        "ImagesProvider", "SizeSpec",
    ):
        assert hasattr(cubepi, name), f"cubepi.{name} not exported"
```

- [ ] **Step 4: Run all tests, ruff, and mypy**

Run:
```bash
uv run pytest tests/ -q
uv run ruff check cubepi/__init__.py cubepi/providers/images/__init__.py
uv run ruff format --check cubepi/__init__.py cubepi/providers/images/__init__.py
uv run mypy cubepi
```
Expected: PASS for all four.

- [ ] **Step 5: Commit**

```bash
git add cubepi/__init__.py cubepi/providers/images/__init__.py tests/providers/images/test_types.py
git commit -m "feat(images): wire public exports for the redesigned image surface

cubepi.providers.images.__init__ exports the new types, descriptor, base
class, Protocol, and two concrete providers. cubepi top-level adds image
types/descriptor/base/Protocol (concrete provider classes stay reachable
only via the submodule, matching chat's pattern)."
```

---

## Task 10: Rewrite English current docs (`image-generation.md`)

**Files:**
- Modify: `website/docs/guides/providers/image-generation.md`

- [ ] **Step 1: Replace the entire file**

The new structure follows the spec §10.5: conceptual model → factory → context → options → descriptor (with four worked backends) → errors → listeners → cancellation → Roadmap. Replace `website/docs/guides/providers/image-generation.md` with:

```markdown
---
title: Image Generation
description: "Generate images with CubePi image providers — OpenAI, Doubao Seedream, SiliconFlow, Together AI, and other OpenAI-compatible backends."
---

# Image Generation

CubePi's image-generation path mirrors the chat-provider conventions: a
provider holds the connection (`provider_id`, `api_key`, `base_url`,
`capability`), a model spec holds model-level defaults, and per-call work
goes through a typed `ImagesContext` plus an optional `ImagesOptions`
cross-cutting bag. Failures raise typed `ProviderError` subclasses; UI
hosts catch them like they would any chat error.

The single concrete provider class — `OpenAIImagesProvider` — reaches
multiple OpenAI-shape backends through an `ImagesCapabilityDescriptor`
that declares wire differences as data (size field name, count field
name, supports_seed/steps/guidance gates, …).

> **Async-task backends** (Aliyun Wanxiang, Google Imagen, Stability,
> Replicate, fal, FLUX official) follow a submit→poll→fetch pattern that
> is not modeled by the capability descriptor. Custom subclasses can
> implement them today; first-class async-task scaffolding is a Roadmap
> item — see the [bottom of this page](#roadmap).

## Quickstart — OpenAI

```python
import os
from cubepi.providers.images import OpenAIImagesProvider, ImagesContext

provider = OpenAIImagesProvider(
    provider_id="openai",
    api_key=os.environ["OPENAI_API_KEY"],
)
model = provider.model(
    "gpt-image-1",
    default_size="1024x1024",
    default_quality="high",
)

result = await provider.generate_images(
    model,
    ImagesContext(prompt="A cute robot at sunrise"),
)

# result.stop_reason in {"stop", "aborted"}; failures raise ProviderError.
for block in result.output:
    print(block.type, block.media_type, len(block.source))
```

## `provider.model("id", ...)` — model factory

`provider.model(...)` builds an `ImagesModel`. The provider's
`provider_id` is propagated automatically so the model knows where it
came from (used for tracing, error messages, and response metadata).

| Parameter | Type | Effect |
|---|---|---|
| `id` (positional) | `str` | The wire model id (e.g. `"gpt-image-1"`) |
| `api` | `str` | Routing tag (e.g. `"openai-images"`) |
| `default_size` | `str \| None` | Used when `ImagesContext.size` is `None` |
| `default_n` | `int \| None` | Used when `ImagesContext.n` is `None` |
| `default_quality` | `Literal["low","medium","high"] \| None` | Default when context omits it |
| `default_output_format` | `Literal["png","jpeg","webp"] \| None` | Default output format |
| `cost` | `ImagesCost \| None` | Per-image / per-megapixel pricing metadata |
| `max_input_images` | `int \| None` | Edit-path cap; meaningful only when the capability supports edit |

## `ImagesContext` — per-call payload

```python
ctx = ImagesContext(
    prompt="A robot",
    size="1024x1024",
    n=2,
    quality="high",
    output_format="png",
    seed=42,                # only written if capability.supports_seed
    negative_prompt="...",  # only written if capability.supports_negative_prompt
    steps=20,               # only written if capability.supports_steps
    guidance=7.5,           # only written if capability.supports_guidance
    extra={"watermark": False},  # always written verbatim
    input_images=[...],     # ImageContent list; triggers edit path
)
```

Field merge rule: `ctx.<field>` wins over `model.default_<field>`; if both
are `None`, the field is omitted from the wire payload (the backend uses
its own default). Value semantics — `"1024x1024"` vs `"1K"` vs `"1:1"` —
remain the user's responsibility; the capability descriptor only renames
the wire **key**.

## `ImagesOptions` — per-call cross-cutting

```python
from cubepi.providers.images import ImagesOptions

opts = ImagesOptions(
    signal=cancel_event,         # asyncio.Event; set to abort
    on_payload=lambda p, m: p,   # pre-send payload mutator (per-call)
    on_response=lambda r, m: None,  # response observer (per-call)
)
```

When `signal` is set mid-call, the SDK request is cancelled and the
provider returns `AssistantImages(stop_reason="aborted", output=[])` —
the `CancelledError` does not escape.

`on_payload` and `on_response` are per-call hooks; for persistent
observers (tracing, audit), use `provider.subscribe_request()` /
`provider.subscribe_response()` — see [Observability](#observability).

## `ImagesCapabilityDescriptor` — reach other OpenAI-shape backends

Different OpenAI-shape backends use different field names. The
descriptor lets one `OpenAIImagesProvider` reach all of them:

### Volcengine Ark / Doubao Seedream

Mostly OpenAI-compatible with a `watermark` extra and `seed` support:

```python
OpenAIImagesProvider(
    provider_id="doubao",
    api_key=os.environ["ARK_API_KEY"],
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    capability=ImagesCapabilityDescriptor(
        supports_seed=True,
        extra_payload={"watermark": False},
    ),
)
```

### SiliconFlow

OpenAI-shape URL but field names differ:

```python
from cubepi.providers.images.capability import ImagesCapabilityDescriptor, SizeSpec

OpenAIImagesProvider(
    provider_id="siliconflow",
    api_key=os.environ["SILICONFLOW_API_KEY"],
    base_url="https://api.siliconflow.cn/v1",
    capability=ImagesCapabilityDescriptor(
        size_spec=SizeSpec(kind="image_size_string"),
        count_field="batch_size",
        supports_seed=True,
        supports_steps=True, steps_field="num_inference_steps",
        supports_guidance=True, guidance_field="guidance_scale",
        supports_negative_prompt=True,
        output_format_field=None,    # not supported
    ),
)
```

### Together AI — FLUX schnell

FLUX schnell uses `aspect_ratio` instead of `size`:

```python
OpenAIImagesProvider(
    provider_id="together",
    api_key=os.environ["TOGETHER_API_KEY"],
    base_url="https://api.together.xyz/v1",
    capability=ImagesCapabilityDescriptor(
        size_spec=SizeSpec(kind="aspect_ratio"),
        supports_seed=True,
        supports_steps=True, steps_field="steps",
    ),
)
```

### Mixed-model gateways

When one gateway serves multiple models with different shapes, use
`model_capability_overrides`:

```python
provider = OpenAIImagesProvider(
    provider_id="together",
    api_key="...",
    base_url="https://api.together.xyz/v1",
    capability=together_pro_cap,       # default
    model_capability_overrides={
        "black-forest-labs/FLUX.1-schnell": together_schnell_cap,
    },
)
```

Resolution is exact-match on `model.id`; unmatched models fall back to
the base `capability`.

## Error handling

All built-in image providers raise typed `cubepi.errors.ProviderError`
subclasses on failure — never in-band error strings:

```python
from cubepi.errors import RateLimited, ProviderAuthFailed, ProviderUnavailable

try:
    result = await provider.generate_images(model, ctx)
except RateLimited as exc:
    # exc.retry_after may be populated
    ...
except ProviderAuthFailed:
    ...
except ProviderUnavailable:
    # 5xx, timeout, network — typically retryable
    ...
```

`AssistantImages.stop_reason` is now only `"stop"` (success) or
`"aborted"` (signal-triggered cancel). There is no `"error"` value and
no `error_message` field.

## Observability

Persistent observers register on the provider:

```python
provider.subscribe_request(lambda payload, model: log_payload(payload))
provider.subscribe_response(lambda body, model, exc: log_response(body, exc))
```

- `subscribe_request` fires once per call, just before the SDK send, with
  the final assembled payload dict (after `on_payload` mutators).
- `subscribe_response` fires once per call in the provider's `finally`
  block, with the assembled response body (or `None` on failure) and the
  exception (or `None` on success).

There is **no** `subscribe_chunk` — image generation is one-shot.

## Edit path

Passing `input_images` triggers the edit path when the capability
supports it:

```python
import base64
from cubepi.providers.base import ImageContent

with open("source.png", "rb") as fh:
    source_b64 = base64.b64encode(fh.read()).decode("ascii")

ctx = ImagesContext(
    prompt="Make it brighter and warmer.",
    input_images=[ImageContent(source=source_b64, media_type="image/png")],
)
result = await provider.generate_images(model, ctx)
```

Setting `capability=ImagesCapabilityDescriptor(supports_edit=False)`
falls back to the generate path even when `input_images` is non-empty —
useful when targeting a backend whose model can't edit.

## Faux provider for tests

```python
from cubepi.providers.images import FauxImagesProvider
from cubepi.errors import RateLimited

# Happy path:
provider = FauxImagesProvider(png_b64="iVBORw0KGgo...")

# Inject an error to exercise retry middleware:
provider = FauxImagesProvider(
    png_b64="iVBORw0KGgo...",
    raise_on_call=RateLimited,
)
```

`FauxImagesProvider` inherits the listener registry, `.model()` factory,
and `provider_id` propagation from `BaseImagesProvider`, so tests that
exercise observability against the image path can use it interchangeably
with `OpenAIImagesProvider`.

## Roadmap

- **Async-task backends.** Aliyun Wanxiang, Google Imagen on Vertex,
  Stability, Replicate, fal, FLUX official all follow a submit→poll→fetch
  pattern that this version does not model first-class. Subclasses of
  `BaseImagesProvider` can implement them by hand today; a future release
  will likely add an `AsyncTaskImagesProvider` base with shared polling
  scaffolding.
- **Tracing wiring.** This release adds the listener registry on image
  providers, but `cubepi.tracing` does not yet auto-subscribe to image
  calls. Hosts that want image-call spans should subscribe manually for
  now.

## See also

- [Providers Overview](./overview) — chat-provider configuration; image
  providers follow the same `provider_id` / `.model()` / capability
  conventions.
- [OpenAI Provider](./openai) — shared OpenAI-shape patterns on the chat
  side.
- [API Reference → `cubepi.providers.images`](../../api/cubepi-providers).
```

- [ ] **Step 2: Verify the doc builds in dev mode**

Run (in a separate terminal if convenient):
```bash
cd website && pnpm install --frozen-lockfile && pnpm run start
```
Then open `http://localhost:3000/docs/guides/providers/image-generation` and confirm the page renders without MDX errors.

(If pnpm/install is slow or unavailable, skip the dev-server step and rely on the build step in a later task.)

- [ ] **Step 3: Commit**

```bash
git add website/docs/guides/providers/image-generation.md
git commit -m "docs(images): rewrite image-generation guide for redesigned surface

Restructured to follow the new provider/model/context split: quickstart,
provider.model() factory, ImagesContext, ImagesOptions, capability
descriptor with four worked backends (OpenAI, Doubao, SiliconFlow,
Together FLUX schnell), error handling, observability, edit path, faux
provider, Roadmap (async-task backends, tracing wiring)."
```

---

## Task 11: Rewrite Chinese current docs

**Files:**
- Modify: `website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/guides/providers/image-generation.md`

- [ ] **Step 1: Verify the zh-Hans file exists**

Run: `ls website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/guides/providers/image-generation.md`
Expected: file exists (it was added in commit 47bca97 alongside the EN file).

- [ ] **Step 2: Replace the entire zh-Hans file**

Replace with this content (same structure as the EN file in Task 10; translation must keep terms-of-art consistent with established CubePi Chinese docs — "provider" stays "Provider"; "model" / "模型"; "capability" / "兼容配置"; "context" / "上下文"; "options" / "选项"; "prompt" / "提示词"):

```markdown
---
title: 图片生成
description: "用 CubePi 的 image provider 生成图片 —— OpenAI、豆包 Seedream、SiliconFlow、Together AI 等 OpenAI 兼容后端。"
---

# 图片生成

CubePi 的图片生成路径与 chat provider 范式同构：Provider 装连接信息
（`provider_id`、`api_key`、`base_url`、`capability`），model spec 装模型级
默认值，per-call 走类型化的 `ImagesContext` 加可选的 `ImagesOptions`
跨切面选项。调用失败抛出类型化的 `ProviderError` 子类——UI 端跟 chat
错误一样 catch 即可。

唯一一个具体 provider 类 `OpenAIImagesProvider` 通过
`ImagesCapabilityDescriptor` 把后端的字段差异**声明为数据**，从而能打多个
OpenAI 形态的后端（OpenAI 官方、豆包 Seedream、SiliconFlow、Together AI 等）。

> **异步任务后端**（阿里万相、Google Imagen、Stability、Replicate、
> fal、FLUX 官方）走的是 submit→poll→fetch 模式，不在本版 capability
> descriptor 的覆盖范围内。今天可以通过继承 `BaseImagesProvider` 自定义实现；
> 一线异步后端的统一脚手架是 [Roadmap 项](#roadmap)。

## Quickstart — OpenAI

```python
import os
from cubepi.providers.images import OpenAIImagesProvider, ImagesContext

provider = OpenAIImagesProvider(
    provider_id="openai",
    api_key=os.environ["OPENAI_API_KEY"],
)
model = provider.model(
    "gpt-image-1",
    default_size="1024x1024",
    default_quality="high",
)

result = await provider.generate_images(
    model,
    ImagesContext(prompt="一只黎明时分的小机器人"),
)

# result.stop_reason ∈ {"stop", "aborted"}；失败会抛 ProviderError。
for block in result.output:
    print(block.type, block.media_type, len(block.source))
```

## `provider.model("id", ...)` — 模型工厂

`provider.model(...)` 返回 `ImagesModel`。provider 的 `provider_id` 会自动
拷进 model（用于 tracing、错误信息、response metadata）。

| 参数 | 类型 | 作用 |
|---|---|---|
| `id`（位置参数） | `str` | wire 上的模型 ID（如 `"gpt-image-1"`） |
| `api` | `str` | 路由 tag（如 `"openai-images"`） |
| `default_size` | `str \| None` | `ImagesContext.size` 为 `None` 时使用 |
| `default_n` | `int \| None` | `ImagesContext.n` 为 `None` 时使用 |
| `default_quality` | `Literal["low","medium","high"] \| None` | context 未指定时的默认值 |
| `default_output_format` | `Literal["png","jpeg","webp"] \| None` | 默认输出格式 |
| `cost` | `ImagesCost \| None` | 单图/百万像素计价元数据 |
| `max_input_images` | `int \| None` | 编辑路径输入图上限；仅 capability 支持 edit 时有意义 |

## `ImagesContext` — per-call 请求负载

```python
ctx = ImagesContext(
    prompt="一只机器人",
    size="1024x1024",
    n=2,
    quality="high",
    output_format="png",
    seed=42,                # 仅 capability.supports_seed=True 时写入
    negative_prompt="...",  # 仅 capability.supports_negative_prompt=True 时写入
    steps=20,               # 仅 capability.supports_steps=True 时写入
    guidance=7.5,           # 仅 capability.supports_guidance=True 时写入
    extra={"watermark": False},  # 始终透传
    input_images=[...],     # ImageContent 列表，触发 edit 路径
)
```

字段合并规则：`ctx.<字段>` 优先于 `model.default_<字段>`；都为 `None` 时
该字段不写进 payload（后端用自家默认值）。值的语义——`"1024x1024"` /
`"1K"` / `"1:1"` 怎么写——仍是用户自己的责任；capability 只换 wire 上的
**字段名**。

## `ImagesOptions` — per-call 跨切面选项

```python
from cubepi.providers.images import ImagesOptions

opts = ImagesOptions(
    signal=cancel_event,         # asyncio.Event；set 后中止当前调用
    on_payload=lambda p, m: p,   # 发包前的 payload mutator（per-call）
    on_response=lambda r, m: None,  # response observer（per-call）
)
```

中途 `signal.set()` 时，SDK 请求会被取消，provider 返回
`AssistantImages(stop_reason="aborted", output=[])`，`CancelledError` 不
冒出来。

`on_payload` / `on_response` 是 per-call hook；对于持久观察者（tracing、
审计），用 `provider.subscribe_request()` / `provider.subscribe_response()`
—— 见 [可观察性](#可观察性)。

## `ImagesCapabilityDescriptor` —— 对接其它 OpenAI 形态后端

不同的 OpenAI 形态后端字段名不同，descriptor 让同一个
`OpenAIImagesProvider` 都能打：

### 火山方舟豆包 Seedream

基本 OpenAI 兼容，多了 `watermark` 扩展和 `seed` 支持：

```python
OpenAIImagesProvider(
    provider_id="doubao",
    api_key=os.environ["ARK_API_KEY"],
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    capability=ImagesCapabilityDescriptor(
        supports_seed=True,
        extra_payload={"watermark": False},
    ),
)
```

### SiliconFlow

URL 长得像 OpenAI，但字段名要换：

```python
from cubepi.providers.images.capability import ImagesCapabilityDescriptor, SizeSpec

OpenAIImagesProvider(
    provider_id="siliconflow",
    api_key=os.environ["SILICONFLOW_API_KEY"],
    base_url="https://api.siliconflow.cn/v1",
    capability=ImagesCapabilityDescriptor(
        size_spec=SizeSpec(kind="image_size_string"),
        count_field="batch_size",
        supports_seed=True,
        supports_steps=True, steps_field="num_inference_steps",
        supports_guidance=True, guidance_field="guidance_scale",
        supports_negative_prompt=True,
        output_format_field=None,    # 该后端不支持
    ),
)
```

### Together AI — FLUX schnell

FLUX schnell 用 `aspect_ratio` 不是 `size`：

```python
OpenAIImagesProvider(
    provider_id="together",
    api_key=os.environ["TOGETHER_API_KEY"],
    base_url="https://api.together.xyz/v1",
    capability=ImagesCapabilityDescriptor(
        size_spec=SizeSpec(kind="aspect_ratio"),
        supports_seed=True,
        supports_steps=True, steps_field="steps",
    ),
)
```

### 多模型混合网关

一个网关服务多个不同形态的模型时，用 `model_capability_overrides`：

```python
provider = OpenAIImagesProvider(
    provider_id="together",
    api_key="...",
    base_url="https://api.together.xyz/v1",
    capability=together_pro_cap,       # 默认
    model_capability_overrides={
        "black-forest-labs/FLUX.1-schnell": together_schnell_cap,
    },
)
```

按 `model.id` 精确匹配；未匹配的退回到基础 `capability`。

## 错误处理

所有内置 image provider 在失败时抛出类型化的 `cubepi.errors.ProviderError`
子类——不再用 in-band 错误字符串：

```python
from cubepi.errors import RateLimited, ProviderAuthFailed, ProviderUnavailable

try:
    result = await provider.generate_images(model, ctx)
except RateLimited as exc:
    # exc.retry_after 可能有值
    ...
except ProviderAuthFailed:
    ...
except ProviderUnavailable:
    # 5xx / timeout / network——通常可重试
    ...
```

`AssistantImages.stop_reason` 现在只有 `"stop"`（成功）和 `"aborted"`
（信号触发的中止）。没有 `"error"` 值，也没有 `error_message` 字段。

## 可观察性

持久观察者注册在 provider 上：

```python
provider.subscribe_request(lambda payload, model: log_payload(payload))
provider.subscribe_response(lambda body, model, exc: log_response(body, exc))
```

- `subscribe_request` 每次调用 SDK 发包前触发一次，拿到最终拼好的
  payload dict（`on_payload` mutator 之后的版本）。
- `subscribe_response` 每次调用结束在 `finally` 块触发一次，拿到 response
  body（失败时为 `None`）和异常（成功时为 `None`）。

**没有** `subscribe_chunk`——图片生成是 one-shot。

## 编辑路径

传入 `input_images` 触发 edit 路径（前提是 capability 声明支持）：

```python
import base64
from cubepi.providers.base import ImageContent

with open("source.png", "rb") as fh:
    source_b64 = base64.b64encode(fh.read()).decode("ascii")

ctx = ImagesContext(
    prompt="调亮、调暖一点。",
    input_images=[ImageContent(source=source_b64, media_type="image/png")],
)
result = await provider.generate_images(model, ctx)
```

设 `capability=ImagesCapabilityDescriptor(supports_edit=False)` 即使
`input_images` 非空也回到 generate 路径——目标模型不支持编辑时用。

## 测试 stub `FauxImagesProvider`

```python
from cubepi.providers.images import FauxImagesProvider
from cubepi.errors import RateLimited

# 正常路径：
provider = FauxImagesProvider(png_b64="iVBORw0KGgo...")

# 注入错误（测重试中间件）：
provider = FauxImagesProvider(
    png_b64="iVBORw0KGgo...",
    raise_on_call=RateLimited,
)
```

`FauxImagesProvider` 继承 `BaseImagesProvider`，自带 listener 注册表、
`.model()` 工厂和 `provider_id` 传播，所以涉及可观察性的测试可以跟
`OpenAIImagesProvider` 互换使用。

## Roadmap

- **异步任务后端**：阿里万相、Google Imagen on Vertex、Stability、Replicate、
  fal、FLUX 官方——这些走的是 submit→poll→fetch 模式，本版没做一级支持。
  今天可以继承 `BaseImagesProvider` 自己实现；未来版本会加 `AsyncTaskImagesProvider`
  这种共享 polling 脚手架。
- **Tracing 接入**：本版加了 image provider 的 listener 注册表，但
  `cubepi.tracing` 还没自动订阅 image 调用。需要 image 调用 span 的
  Host 暂时手动 `subscribe_*`。

## 另见

- [Providers Overview](./overview) —— chat-provider 的配置方式；image
  provider 跟它共享 `provider_id` / `.model()` / capability 范式。
- [OpenAI Provider](./openai) —— chat 那边 OpenAI 形态的共通配置。
- [API Reference → `cubepi.providers.images`](../../api/cubepi-providers)。
```

- [ ] **Step 3: Commit**

```bash
git add website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/guides/providers/image-generation.md
git commit -m "docs(images): zh-Hans rewrite of image-generation guide

Mirrors the EN current rewrite from the previous commit; same section
structure (quickstart → factory → context → options → descriptor with
four worked backends → errors → observability → edit → faux → Roadmap)."
```

---

## Task 12: Sync version-0.7 docs mirrors

**Files:**
- Modify: `website/versioned_docs/version-0.7/guides/providers/image-generation.md`
- Modify: `website/i18n/zh-Hans/docusaurus-plugin-content-docs/version-0.7/guides/providers/image-generation.md`

- [ ] **Step 1: Verify both versioned files exist**

Run:
```bash
ls website/versioned_docs/version-0.7/guides/providers/image-generation.md
ls website/i18n/zh-Hans/docusaurus-plugin-content-docs/version-0.7/guides/providers/image-generation.md
```
Expected: both files exist (added in commit 47bca97).

- [ ] **Step 2: Copy current EN content into the versioned EN mirror**

Run:
```bash
cp website/docs/guides/providers/image-generation.md \
   website/versioned_docs/version-0.7/guides/providers/image-generation.md
```

- [ ] **Step 3: Copy current zh-Hans content into the versioned zh mirror**

Run:
```bash
cp website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/guides/providers/image-generation.md \
   website/i18n/zh-Hans/docusaurus-plugin-content-docs/version-0.7/guides/providers/image-generation.md
```

- [ ] **Step 4: Verify both mirrors are byte-identical to their current counterparts**

Run:
```bash
diff website/docs/guides/providers/image-generation.md \
     website/versioned_docs/version-0.7/guides/providers/image-generation.md
diff website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/guides/providers/image-generation.md \
     website/i18n/zh-Hans/docusaurus-plugin-content-docs/version-0.7/guides/providers/image-generation.md
```
Expected: no diff output for either pair.

- [ ] **Step 5: Commit**

```bash
git add website/versioned_docs/version-0.7/guides/providers/image-generation.md \
        website/i18n/zh-Hans/docusaurus-plugin-content-docs/version-0.7/guides/providers/image-generation.md
git commit -m "docs(images): sync version-0.7 mirrors of image-generation guide

0.7 has not yet shipped, so versioned docs should reflect the redesigned
surface. EN and zh-Hans version-0.7 copies are byte-identical to their
current counterparts."
```

---

## Task 13: Overview page additions + CHANGELOG entry

**Files:**
- Modify: `website/docs/guides/providers/overview.md`
- Modify: `website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/guides/providers/overview.md`
- Modify: `website/versioned_docs/version-0.7/guides/providers/overview.md`
- Modify: `website/i18n/zh-Hans/docusaurus-plugin-content-docs/version-0.7/guides/providers/overview.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add an image-providers paragraph to `website/docs/guides/providers/overview.md`**

Locate the existing "See also" section near the bottom (lines ~291–297). Just above it, insert:

```markdown
## Image providers

Image generation has its own provider surface (`cubepi.providers.images`)
that follows the same conventions described above: `provider_id` on the
provider, `provider.model("id", ...)` factory, typed `ProviderError`
failures, and a capability descriptor for backend wire differences. See
[Image Generation](./image-generation) for the full guide.
```

- [ ] **Step 2: Add the same paragraph (translated) to the zh-Hans current `overview.md`**

In `website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/guides/providers/overview.md`, locate the equivalent "另见" section and insert above it:

```markdown
## 图片生成 provider

图片生成有独立的 provider 表面（`cubepi.providers.images`），范式与上文
描述完全一致：provider 上的 `provider_id`、`provider.model("id", ...)`
工厂、类型化的 `ProviderError` 错误，以及处理后端字段差异的 capability
descriptor。完整指南见 [图片生成](./image-generation)。
```

- [ ] **Step 3: Copy the changes to the versioned mirrors**

After editing both current files, copy them over the versioned mirrors:

```bash
cp website/docs/guides/providers/overview.md \
   website/versioned_docs/version-0.7/guides/providers/overview.md
cp website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/guides/providers/overview.md \
   website/i18n/zh-Hans/docusaurus-plugin-content-docs/version-0.7/guides/providers/overview.md
```

Verify with `diff` that all four are aligned.

- [ ] **Step 4: Add a Breaking bullet to `CHANGELOG.md` under `[0.7.0]` → `Changed`**

Locate the `### Changed` subsection in the `[0.7.0]` block (around the `**Breaking:** agent construction` bullet). Append:

```markdown
- **Breaking:** the `cubepi.providers.images` surface has been redesigned
  to align with the chat-provider 0.7 conventions: providers now take
  `provider_id` and an optional `ImagesCapabilityDescriptor`; models are
  built via `provider.model("id", ...)` (renamed `provider` field to
  `provider_id`, added `default_size/n/quality/output_format` and `cost`
  metadata); `ImagesContext` is typed (`size/n/quality/output_format/seed/
  negative_prompt/steps/guidance/extra`); per-call options live on a new
  `ImagesOptions` bag (`signal`, `on_payload`, `on_response`); failures
  raise `cubepi.errors.ProviderError` subclasses instead of in-band
  `AssistantImages.error_message`; the `create_images_provider` /
  `register_images_provider_class` registry is removed. The new shape
  reaches OpenAI, Doubao Seedream, SiliconFlow, and Together AI through
  a single `OpenAIImagesProvider` configured with the right
  `ImagesCapabilityDescriptor`.
```

- [ ] **Step 5: Verify ruff + full test suite (no code changed but pre-commit hooks may touch docs)**

Run:
```bash
uv run ruff check cubepi/ tests/
uv run ruff format --check cubepi/ tests/
uv run pytest tests/ -q
uv run mypy cubepi
```
Expected: PASS for all four.

- [ ] **Step 6: Commit**

```bash
git add website/docs/guides/providers/overview.md \
        website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/guides/providers/overview.md \
        website/versioned_docs/version-0.7/guides/providers/overview.md \
        website/i18n/zh-Hans/docusaurus-plugin-content-docs/version-0.7/guides/providers/overview.md \
        CHANGELOG.md
git commit -m "docs(images): wire image-providers into overview + CHANGELOG breaking note

overview.md (EN + zh-Hans, current + version-0.7) gains a one-paragraph
pointer to the image-generation guide so the providers landing page
surfaces it. CHANGELOG documents the breaking redesign under 0.7.0."
```

---

## Final Verification

After Task 13 commits, run the complete sanity sweep once more:

- [ ] **Step 1: Full local CI parity**

```bash
uv run ruff check cubepi/ tests/
uv run ruff format --check cubepi/ tests/
uv run mypy cubepi
uv run pytest tests/ -q
```
Expected: PASS for all four.

- [ ] **Step 2: Confirm no remaining references to the old image surface**

```bash
git grep -nE "create_images_provider|register_images_provider_class|ImagesModel\(.*\bprovider=" -- cubepi/ tests/
```
Expected: no output (every reference to the removed symbols / old shape is gone).

- [ ] **Step 3: Confirm the doc site has no broken internal links**

```bash
cd website && pnpm run build && cd ..
```
Expected: build completes without "Broken links" warnings touching image pages. (If pnpm is unavailable in the environment, skip this step — the PR's CI will run it.)

- [ ] **Step 4: Summarize the worktree state**

```bash
git log --oneline 2026-06-05-release-0.7-review ^main | head -20
```
Expected: 13 new commits on top of the prior branch state, all touching `cubepi/providers/images/`, `tests/providers/images/`, `cubepi/errors.py`, docs, or `CHANGELOG.md`.

---

## Self-Review Checklist

(For the engineer executing this plan — verify before declaring "done".)

- [ ] Every spec section §1–§13 has corresponding tasks (types ↔ Task 2; capability ↔ Tasks 3+5; base ↔ Task 4; errors ↔ Task 6; faux ↔ Task 7; openai ↔ Task 8; exports ↔ Task 9; docs ↔ Tasks 10–13).
- [ ] All four "Verified Backend" worked examples from spec §7.1 appear verbatim in the docs (OpenAI / Doubao / SiliconFlow / Together).
- [ ] `provider_id` propagation is tested at three levels: factory (Task 4), model rename test (Task 2), and end-to-end via FauxImagesProvider (Task 7).
- [ ] The `subscribe_chunk` absence is asserted explicitly in Task 4 and Task 7 (not just omitted by accident).
- [ ] The `stop_reason="aborted"` cancellation path is tested in Task 8.
- [ ] The error taxonomy widening in Task 6 has tests for ImagesModel **and** confirms chat Model still works.
- [ ] `ImagesContext.extra` deep-merge precedence is pinned in Task 5.
- [ ] CHANGELOG bullet covers all five breaking points (provider_id ctor / .model() factory / typed errors / removed registry / removed in-band error_message).
- [ ] No backwards-compat shims or deprecation wrappers exist anywhere (per spec §10 and CLAUDE.md "no half-finished implementations").
