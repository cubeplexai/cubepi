# Image Provider Redesign — Align with Chat Provider 0.7 Conventions

- **Status**: Draft, awaiting review
- **Date**: 2026-06-05
- **Branch / worktree**: `2026-06-05-release-0.7-review` / `.worktrees/2026-06-05-release-0.7-review`
- **Author**: brainstormed with the user, drafted by Claude

## 1. Motivation

CubePi 0.7 unified the chat-provider surface around four ideas:

1. **`provider_id` lives on the provider constructor**, and propagates into the model spec automatically (`BaseProvider.model(...)`).
2. **`provider.model("id", ...)` is the canonical model-building factory** — it bundles spec construction with provider-id propagation, and (for chat) yields a `BoundModel` that the agent loop consumes as a single argument.
3. **A `CapabilityDescriptor` describes wire-level differences as data** — letting one `OpenAIProvider` reach any OpenAI-shaped endpoint by declaring the field renames / payload injections that backend needs.
4. **Errors are typed** — every built-in provider wraps SDK exceptions into `cubepi.errors.ProviderError` subclasses (`RateLimited`, `ContextLengthExceeded`, `ProviderAuthFailed`, `ProviderUnavailable`, `ProviderBadRequest`).

The image-generation path (`cubepi/providers/images/`) was added before this consolidation and follows the **0.6 idiom**:

- `OpenAIImagesProvider(api_key=...)` — no `provider_id` on the constructor.
- `ImagesModel(id="gpt-image-1", provider="openai", api="openai-images")` — three fields the user has to hand-fill.
- `provider.generate_images(model, ctx, options=dict)` — `options` is a free-form dict, no type checking.
- Failures are returned in-band as `AssistantImages(stop_reason="error", error_message=str)`.
- A separate `create_images_provider("openai-images", **kwargs)` registry duplicates direct class import.

Two things follow from this gap:

- **Users who learn the chat-provider idiom and then reach for image generation encounter a different shape** — different parameter location, different error convention, different observability story. That's exactly the friction the 0.7 release set out to remove.
- **Backend fragmentation in image generation is worse than chat.** Verification (see §11) confirms that even among "OpenAI-compatible" image endpoints, the wire dialects differ in field names: `size` vs `image_size` vs `width+height` vs `aspect_ratio`; `n` vs `batch_size` vs `number_of_images`. Without a descriptor mechanism, every new backend forces a fork of `OpenAIImagesProvider`.

This spec replaces the image-provider surface with a redesign that mirrors the chat-provider conventions one-for-one, while staying honest about what's genuinely different (image is one-shot, not streamed; image has no `Agent` loop on top of it).

## 2. Design Philosophy

> CubePi's image provider should look like its chat provider for the same reasons, and look different for the same reasons. Structural conventions (`provider_id`, `.model()` factory, typed errors, capability descriptor, listener registry, options bag) are mirrored. Concepts that don't exist in image (streaming events, agent loops, bound-model containers for agent consumption) are not invented for the sake of symmetry.

Three concrete rules:

- **Symmetry where chat solved a real problem.** `provider_id` propagation, model-defaults-with-per-call-overrides, capability-descriptor field remapping, typed `ProviderError` taxonomy, request/response observer registry — all of these solve problems image users also have, and the cost of mirroring is small.
- **No empty symmetry.** Chat's `BoundModel(provider, spec)` is a container that exists **only** to let `Agent` accept a single `model=` argument instead of `provider= + model=`. Image has no agent-loop consumer, so `BoundImagesModel` would be a class with no caller — we don't introduce it. `provider.model("id", ...)` returns `ImagesModel` (the spec) directly.
- **One genuinely different shape: `generate_images` is one-shot.** Image generation does not stream tokens, has no per-chunk events, and has no `MessageStream` analog. The protocol method name stays `generate_images` (not `generate`) to keep call sites unambiguous when both a chat and an image provider are in scope. `subscribe_chunk` does not exist on `BaseImagesProvider`.

## 3. Goals and Non-Goals

**Goals**

- Replace the image-provider surface so it mirrors the chat-provider 0.7 conventions.
- Support OpenAI-shape backends (OpenAI, Volcengine Ark / Doubao Seedream, SiliconFlow, Together AI) without subclassing `OpenAIImagesProvider`, via `ImagesCapabilityDescriptor` data.
- Provide typed errors, abort signals, request/response observers, and model-level defaults.
- Ship a complete docs page (Chinese + English, current + version-0.7) explaining the new shape with worked examples for each backend.
- Integrate at the source — no deprecation shim, since 0.7 has not yet shipped and the old image surface has no public consumers.

**Non-goals**

- **Async-task backends (Aliyun Wanxiang, Google Imagen on Vertex, Stability, Replicate, fal, FLUX official).** These follow a submit→poll→fetch pattern that needs a separate scaffolding (`AsyncTaskImagesProvider` or similar). Designing that scaffolding without a concrete implementation candidate would be speculative; it is deferred to a future release. The new docs explicitly call this out as a Roadmap item.
- **Streaming image events.** OpenAI's image API and its dialects are all one-shot in 2026; introducing an `ImagesStream` for future-proofing would add an abstraction with no consumer today.
- **Restructuring `cubepi.errors`.** The existing `ProviderError` taxonomy already fits image failures; image just reuses it.
- **Cost calculation, cache wiring, multi-provider routing.** Out of scope; future work.

## 4. Architecture

### 4.1 Module layout (after this change)

```
cubepi/providers/images/
├── __init__.py             # public exports
├── base.py                 # NEW — ImagesProvider Protocol + BaseImagesProvider
├── capability.py           # NEW — ImagesCapabilityDescriptor + SizeSpec
├── types.py                # CHANGED — ImagesModel + ImagesContext + ImagesOptions + AssistantImages + ImagesCost
├── openai_images.py        # CHANGED — inherits BaseImagesProvider, applies capability
├── faux.py                 # CHANGED — inherits BaseImagesProvider, supports error injection
└── (DELETED) registry.py   # removed
```

### 4.2 Layer correspondence with chat

| Layer | Chat (`cubepi.providers`) | Image (`cubepi.providers.images`) |
|---|---|---|
| Model spec (pydantic) | `Model` | `ImagesModel` |
| Provider+spec container | `BoundModel(provider, spec)` | — *(not introduced — no agent consumer)* |
| Protocol | `Provider` | `ImagesProvider` |
| Concrete base class | `BaseProvider` | `BaseImagesProvider` |
| Per-call cross-cutting options | `StreamOptions` | `ImagesOptions` |
| Typed error family | `cubepi.errors.ProviderError` | *same — reused* |
| Wire-quirks-as-data | `CapabilityDescriptor` | `ImagesCapabilityDescriptor` |
| Request payload | `Message` / `list[Message]` | `ImagesContext` |
| Response | `AssistantMessage` / `MessageStream` | `AssistantImages` |
| Concrete OpenAI-shape provider | `OpenAIProvider` | `OpenAIImagesProvider` |
| Test stub | `FauxProvider` | `FauxImagesProvider` |
| Discovery registry | — *(none — direct import)* | — *(none — `create_images_provider` deleted)* |

Three lines are deliberately absent: `BoundImagesModel`, `ImagesStream`, image registry. Each absence is justified by the philosophy above (no empty symmetry, no streaming, parity with chat's direct-import approach).

### 4.3 Call shape

The redesigned call shape:

```python
from cubepi.providers.images import OpenAIImagesProvider, ImagesContext, ImagesOptions
from cubepi.providers.images.capability import ImagesCapabilityDescriptor

provider = OpenAIImagesProvider(
    provider_id="doubao",
    api_key=os.environ["ARK_API_KEY"],
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    capability=ImagesCapabilityDescriptor(
        supports_seed=True,
        extra_payload={"watermark": False},
    ),
)

model = provider.model("doubao-seedream-4-5-251128", default_size="2K", default_n=1)

result = await provider.generate_images(
    model,
    ImagesContext(prompt="A cute robot at sunrise"),
    options=ImagesOptions(signal=cancel_event),
)

# result.stop_reason in {"stop", "aborted"}; failures raised, not returned in-band.
for block in result.output:
    ...
```

Note that the user never constructs `ImagesModel` directly; they go through `provider.model("id", ...)`, which automatically propagates `provider.provider_id` into `ImagesModel.provider_id`. Direct construction is allowed but discouraged.

## 5. Type Definitions

### 5.1 `ImagesModel`

```python
class ImagesModel(BaseModel):
    # Identity
    id: str
    provider_id: str = ""          # renamed from old `provider` field
    api: str = ""                  # routing tag, e.g. "openai-images" / "doubao-images"

    # Model-level defaults (used when ImagesContext field is None; both None → not written to wire)
    default_size: str | None = None
    default_n: int | None = None
    default_quality: Literal["low", "medium", "high"] | None = None
    default_output_format: Literal["png", "jpeg", "webp"] | None = None

    # Metadata
    cost: ImagesCost | None = None
    max_input_images: int | None = None    # only meaningful when capability.supports_edit=True
```

### 5.2 `ImagesCost`

```python
class ImagesCost(BaseModel):
    """Image pricing is per-image (most backends) or per-megapixel (Imagen et al.)."""
    per_image: float = 0
    per_megapixel: float = 0
```

`ImagesCost` is independent of `ModelCost` because the pricing model differs (token-based vs image-based); blending would force confusing optional fields on both sides.

### 5.3 `ImagesContext`

```python
class ImagesContext(BaseModel):
    # Required
    prompt: str

    # Edit path (used when capability.supports_edit and input_images is non-empty)
    input_images: list[ImageContent] = Field(default_factory=list)

    # Common fields (CubePi canonical names; capability descriptor maps to wire names)
    size: str | None = None
    n: int | None = None
    quality: Literal["low", "medium", "high"] | None = None
    output_format: Literal["png", "jpeg", "webp"] | None = None

    # Backend-optional fields (capability.supports_* gates them; unsupported → dropped + one-time warn)
    seed: int | None = None
    negative_prompt: str | None = None
    steps: int | None = None
    guidance: float | None = None

    # Backend-specific per-call extras (deep-merged into wire payload)
    extra: dict[str, Any] = Field(default_factory=dict)
```

**Merge rule:** for each declared field, the effective value is `ctx.field if ctx.field is not None else model.default_field`. If both are `None`, the field is not written to the wire payload (the backend uses its own default).

**Value semantics:** the capability descriptor only renames the wire field; **value formats are the user's responsibility.** `ctx.size="1024x1024"` works for OpenAI and Doubao; `ctx.size="1:1"` works for Together FLUX schnell when `capability.size_spec.kind="aspect_ratio"`. The user picks values that match the backend; CubePi routes the value to the right wire key.

### 5.4 `ImagesOptions`

```python
class ImagesOptions(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    signal: asyncio.Event | None = None
    on_payload: OnPayloadCallback | None = None    # reused from cubepi.providers.base
    on_response: OnResponseCallback | None = None  # reused from cubepi.providers.base
```

`OnPayloadCallback / OnResponseCallback / ProviderResponse` are reused as-is from `cubepi.providers.base`; their signatures `(dict, Model) -> dict | None` and `(ProviderResponse, Model) -> None` work for image too (the `Model` parameter is duck-typed against `ImagesModel` — both have `id` and `provider_id`).

### 5.5 `AssistantImages`

```python
class AssistantImages(BaseModel):
    api: str
    provider_id: str               # renamed from old `provider`
    model: str
    output: list[ImageContent | TextContent] = Field(default_factory=list)
    stop_reason: Literal["stop", "aborted"] = "stop"
    response_id: str | None = None
    metadata: JsonObject = Field(default_factory=dict)
```

**Removed:** `error_message`, `stop_reason="error"`. Failures raise `ProviderError` subclasses — there is no in-band error path. The `"aborted"` stop reason exists only for abort-signal cancellation, where the call did not fail but was deliberately stopped.

### 5.6 `ImagesCapabilityDescriptor` and `SizeSpec`

```python
class SizeSpec(BaseModel):
    """How the canonical `ctx.size` value is serialized to the wire."""
    kind: Literal["size_string", "image_size_string", "width_height", "aspect_ratio"]
    # size_string        → {"size": ctx.size}                (OpenAI / Doubao)
    # image_size_string  → {"image_size": ctx.size}          (SiliconFlow)
    # width_height       → split "<W>x<H>" → {"width": W, "height": H}
    # aspect_ratio       → {"aspect_ratio": ctx.size}        (Together FLUX schnell / Imagen)


class ImagesCapabilityDescriptor(BaseModel):
    # Size / count (the two biggest fragmentation points)
    size_spec: SizeSpec = Field(default_factory=lambda: SizeSpec(kind="size_string"))
    count_field: str = "n"

    # Optional fields that some backends support
    supports_seed: bool = False
    seed_field: str = "seed"

    supports_negative_prompt: bool = False
    negative_prompt_field: str = "negative_prompt"

    supports_steps: bool = False
    steps_field: str = "num_inference_steps"

    supports_guidance: bool = False
    guidance_field: str = "guidance_scale"

    # Output controls
    output_format_field: str | None = "output_format"   # None → silently drop ctx.output_format
    response_format_field: str = "response_format"
    response_format_value: Literal["b64_json", "url"] = "b64_json"

    # Edit path
    supports_edit: bool = True
    input_images_field: str = "image"

    # Provider-level always-injected payload, deep-merged into every request
    extra_payload: dict[str, Any] = Field(default_factory=dict)
```

**Why these fields specifically:** each one corresponds to a verified divergence among OpenAI-shape backends (see §11). Fields not on this list either don't exist anywhere (we're not pre-designing for hypothetical needs) or live in `extra_payload` / `ctx.extra` (true backend-specific extras like Doubao's `watermark`, `sequential_image_generation`).

## 6. Protocol and Base Class

### 6.1 `ImagesProvider` Protocol

```python
@runtime_checkable
class ImagesProvider(Protocol):
    provider_id: str

    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        *,
        options: ImagesOptions | None = None,
    ) -> AssistantImages: ...
```

`api` is no longer on the Protocol — it lives on `ImagesModel.api` only. `provider_id` is the new identity field.

### 6.2 `BaseImagesProvider` — recommended base for all built-in and user-defined image providers

```python
class BaseImagesProvider:
    def __init__(
        self,
        *,
        provider_id: str = "",
        capability: ImagesCapabilityDescriptor | None = None,
        model_capability_overrides: dict[str, ImagesCapabilityDescriptor] | None = None,
    ) -> None:
        self.provider_id = provider_id
        self._capability = capability or ImagesCapabilityDescriptor()
        self._model_capability_overrides = model_capability_overrides or {}
        self._request_listeners: list[OnRequestCallback] = []
        self._response_listeners: list[OnResponseBodyCallback] = []

    # Factory
    def model(self, id: str, *, api: str = "",
              default_size: str | None = None,
              default_n: int | None = None,
              default_quality: Literal["low", "medium", "high"] | None = None,
              default_output_format: Literal["png", "jpeg", "webp"] | None = None,
              cost: ImagesCost | None = None,
              max_input_images: int | None = None) -> ImagesModel:
        return ImagesModel(
            id=id, provider_id=self.provider_id, api=api,
            default_size=default_size, default_n=default_n,
            default_quality=default_quality, default_output_format=default_output_format,
            cost=cost, max_input_images=max_input_images,
        )

    # Protocol method — subclass implements
    async def generate_images(self, model, context, *, options=None) -> AssistantImages:
        raise NotImplementedError

    # Listener registry (note: subscribe_chunk does not exist — image is one-shot)
    def subscribe_request(self, cb: OnRequestCallback) -> Callable[[], None]: ...
    def subscribe_response(self, cb: OnResponseBodyCallback) -> Callable[[], None]: ...

    # Helpers for subclasses
    def _capability_for(self, model: ImagesModel) -> ImagesCapabilityDescriptor:
        return self._model_capability_overrides.get(model.id, self._capability)

    def _build_payload(self, model: ImagesModel, context: ImagesContext) -> dict[str, Any]:
        """Apply capability descriptor + model defaults to ctx → wire payload dict."""

    def _error_message(self, exc: BaseException, model: ImagesModel) -> str:
        """Format provider/model/base_url + cause chain (same shape as chat)."""
```

### 6.3 `_build_payload` semantics

The payload assembly is the heart of capability application:

```python
def _build_payload(self, model, ctx):
    cap = self._capability_for(model)
    payload: dict[str, Any] = {"model": model.id, "prompt": ctx.prompt}

    # size — four possible wire shapes
    size = ctx.size if ctx.size is not None else model.default_size
    if size is not None:
        if cap.size_spec.kind == "size_string":
            payload["size"] = size
        elif cap.size_spec.kind == "image_size_string":
            payload["image_size"] = size
        elif cap.size_spec.kind == "width_height":
            w, h = (int(x) for x in size.lower().split("x"))
            payload["width"], payload["height"] = w, h
        elif cap.size_spec.kind == "aspect_ratio":
            payload["aspect_ratio"] = size

    # n
    n = ctx.n if ctx.n is not None else model.default_n
    if n is not None:
        payload[cap.count_field] = n

    # quality
    quality = ctx.quality if ctx.quality is not None else model.default_quality
    if quality is not None:
        payload["quality"] = quality   # OpenAI canonical name; backends with different names use extra_payload

    # output_format
    of = ctx.output_format if ctx.output_format is not None else model.default_output_format
    if of is not None and cap.output_format_field is not None:
        payload[cap.output_format_field] = of

    # response_format — always set by capability default (b64_json) for now
    payload[cap.response_format_field] = cap.response_format_value

    # supports_* gating
    if ctx.seed is not None:
        if cap.supports_seed:
            payload[cap.seed_field] = ctx.seed
        else:
            _warn_once(f"{model.provider_id}/{model.id} does not declare supports_seed; dropping ctx.seed")
    # ... same shape for negative_prompt, steps, guidance ...

    # Provider-level extra_payload (capability) + per-call extra (ctx), deep-merged
    payload = deep_merge(payload, cap.extra_payload, ctx.extra)
    return payload
```

The `_warn_once` helper is keyed on `(provider_id, model_id, field_name)` to avoid log spam.

### 6.4 Generate-images implementation template (what concrete providers do)

```python
async def generate_images(self, model, context, *, options=None) -> AssistantImages:
    cap = self._capability_for(model)
    payload = self._build_payload(model, context)

    # 1. per-call on_payload mutator + persistent request listeners
    if options and options.on_payload:
        payload = await invoke_on_payload(options.on_payload, payload, model)
    if self._request_listeners:
        await _fire_request_listeners(self._request_listeners, payload, model)

    body: dict | None = None
    exc: BaseException | None = None
    try:
        # 2. abort-signal wiring — see §6.5
        sdk_resp = await self._call_sdk_with_signal(payload, context, cap, options)
        body = _sdk_resp_to_dict(sdk_resp)

        # 3. on_response observer (per-call) — receives ProviderResponse(status, headers)
        if options and options.on_response:
            await invoke_on_response(options.on_response, _to_provider_response(sdk_resp), model)

        return _parse_response(sdk_resp, model, cap)

    except asyncio.CancelledError:
        # signal-triggered → return aborted, do not raise
        return AssistantImages(api=model.api, provider_id=model.provider_id,
                               model=model.id, output=[], stop_reason="aborted")
    except Exception as raw:    # noqa: BLE001
        exc = raw
        classify_and_raise(raw, model)   # raises a typed ProviderError subclass

    finally:
        if self._response_listeners:
            await _fire_response_listeners(self._response_listeners, body, model, exc)
```

The two listener call points are **exactly two** — pre-send for `subscribe_request`, in `finally` for `subscribe_response`. The chat side has the same two-point structure plus a third per-chunk hook (`subscribe_chunk`) that image does not need.

### 6.5 Cancellation semantics

`ImagesOptions.signal: asyncio.Event` is honored as follows:

- The provider call is wrapped in an `asyncio.wait` race between the actual SDK call and `signal.wait()`.
- If the signal fires first, the SDK call is cancelled and `asyncio.CancelledError` propagates up.
- The provider catches `CancelledError` and **returns `AssistantImages(stop_reason="aborted", output=[])`** — does not re-raise. Rationale: `"aborted"` is the dedicated enum value for this case; making every caller write `try / except CancelledError` to translate it would be uniform noise.
- The response observer still fires in `finally`, with `exc=CancelledError` and `body=None`, so tracing can record the abort.

Chat's cancellation path differs (it cancels mid-stream and the `MessageStream` surfaces a cancel event), so this is not a parity break — it's the right shape for one-shot calls.

## 7. Concrete Providers

### 7.1 `OpenAIImagesProvider`

```python
class OpenAIImagesProvider(BaseImagesProvider):
    def __init__(self, *, provider_id: str = "",
                 api_key: str | None = None,
                 base_url: str | None = None,
                 capability: ImagesCapabilityDescriptor | None = None,
                 model_capability_overrides: dict[str, ImagesCapabilityDescriptor] | None = None) -> None:
        super().__init__(provider_id=provider_id, capability=capability,
                         model_capability_overrides=model_capability_overrides)
        import openai
        kwargs: dict[str, Any] = {}
        if api_key:  kwargs["api_key"] = api_key
        if base_url: kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**kwargs)

    async def generate_images(self, model, context, *, options=None) -> AssistantImages:
        # see §6.4 template
```

Worked instantiation samples (each verified against the backend's real wire shape — see §11):

```python
# OpenAI official
OpenAIImagesProvider(provider_id="openai", api_key=os.environ["OPENAI_API_KEY"])

# Volcengine Ark / Doubao Seedream
OpenAIImagesProvider(
    provider_id="doubao",
    api_key=os.environ["ARK_API_KEY"],
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    capability=ImagesCapabilityDescriptor(
        supports_seed=True,
        extra_payload={"watermark": False},
    ),
)

# SiliconFlow (URL looks OpenAI-shaped, fields are not)
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

# Together FLUX schnell (uses aspect_ratio)
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

For a single OpenRouter-style gateway that routes to multiple models with different wire shapes, `model_capability_overrides` lets one provider instance hold per-model descriptors — identical mechanism to chat's `model_capability_overrides`.

### 7.2 `FauxImagesProvider`

```python
class FauxImagesProvider(BaseImagesProvider):
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

    async def generate_images(self, model, context, *, options=None) -> AssistantImages:
        if self._raise is not None:
            raise self._raise(f"injected by FauxImagesProvider for {model.id}")
        return AssistantImages(
            api=model.api,
            provider_id=model.provider_id,
            model=model.id,
            output=[ImageContent(source=self._png_b64, media_type="image/png")],
            stop_reason="stop",
        )
```

`raise_on_call` is new — the old faux had no error path to inject, which made testing retry middleware against image providers awkward. The new shape lets tests assert "retry middleware sees `RateLimited` and back-offs."

`FauxImagesProvider` inherits from `BaseImagesProvider`, so it gets `provider_id`, `.model()`, `.subscribe_request()`, `.subscribe_response()` for free — the same parity `FauxProvider` has on the chat side.

## 8. Error Semantics

All built-in image providers wrap SDK exceptions via the existing `cubepi.errors.classify_and_raise(exc, model)`. The taxonomy reused unchanged:

- `ProviderAuthFailed` — 401/403, API key issues
- `RateLimited` — 429, includes `retry_after` when the SDK exposes one
- `ProviderUnavailable` — 5xx, network failures, timeouts
- `ProviderBadRequest` — 400 with a model-attributable cause (banned prompt, invalid size)
- `ContextLengthExceeded` — image generation does have a prompt-length limit on some backends; this is the right bucket
- `ProviderError` — generic fallback

`classify_and_raise` already accepts a `model` argument typed as the chat `Model`. To accommodate `ImagesModel`, either:

1. Adjust `classify_and_raise`'s signature to take a `Protocol` with `id` + `provider_id` attributes — works for both `Model` and `ImagesModel`. **Preferred.**
2. Build a temporary `Model`-shaped object in image providers before calling. Wasteful.

Plan: tighten `classify_and_raise`'s annotation to a structural type that both `Model` and `ImagesModel` satisfy. No behavioral change for chat callers.

## 9. Listener / Observability Parity

`BaseImagesProvider` exposes:

- `subscribe_request(cb: OnRequestCallback)` — pre-send observer, sees final payload dict + model.
- `subscribe_response(cb: OnResponseBodyCallback)` — fires exactly once per `generate_images` call in `finally`, sees assembled response body (or `None` on failure) + model + exception (or `None` on success).

**Not present:** `subscribe_chunk`. Image is one-shot; there are no chunks.

Per-call `options.on_payload` and `options.on_response` retain their existing single-slot semantics — they are independent of the persistent listener registry, fire alongside it, and reuse the chat-side `OnPayloadCallback / OnResponseCallback` types verbatim.

Hosts that use `cubepi.tracing` to observe chat calls will get image calls "for free" once tracing is updated to also wire `subscribe_request / subscribe_response` on image providers. That wiring is a small follow-up (out of scope for this spec — tracking issue to be filed alongside the implementation PR).

## 10. Migration

0.7 has not yet shipped to PyPI. The old image surface (added earlier in this same release branch — commit `47bca97`) has no public users, so there is no deprecation window to honor.

### 10.1 Removed (no replacement, full delete)

- `cubepi/providers/images/registry.py` — the entire `create_images_provider` / `register_images_provider_class` registry.
- `ImagesProvider.api: str` Protocol attribute (replaced by `ImagesModel.api`).
- `ImagesModel.provider` field (renamed to `provider_id`).
- `AssistantImages.error_message`, `stop_reason="error"`.
- Old `OpenAIImagesProvider.__init__` signature (no `provider_id`, no `capability`).

### 10.2 Added

- `cubepi/providers/images/base.py` — `ImagesProvider` Protocol, `BaseImagesProvider`.
- `cubepi/providers/images/capability.py` — `ImagesCapabilityDescriptor`, `SizeSpec`.
- `ImagesCost` in `cubepi/providers/images/types.py`.
- `ImagesOptions` in `cubepi/providers/images/types.py`.
- `raise_on_call` parameter on `FauxImagesProvider`.

### 10.3 Renamed / restructured

- `OpenAIImagesProvider` and `FauxImagesProvider` now inherit `BaseImagesProvider`.
- Field rename: `ImagesModel.provider` → `provider_id`; `AssistantImages.provider` → `provider_id`.
- `ImagesContext` gains typed fields: `size / n / quality / output_format / seed / negative_prompt / steps / guidance / extra`.

### 10.4 Tests

New test files (under `tests/providers/images/`):

- `test_capability_payload_mapping.py` — for each `SizeSpec.kind`, assert the payload key/value; assert each `supports_*` gate behavior; assert `extra_payload` + `ctx.extra` deep-merge precedence; assert per-model overrides.
- `test_faux_provider.py` — happy path; `raise_on_call=RateLimited` raises typed; subscribe-listener wiring fires; `subscribe_chunk` is intentionally absent.
- `test_openai_images_classify.py` — using a stub `openai.AsyncOpenAI`, simulate SDK exceptions and assert each maps to the right `ProviderError` subclass via `classify_and_raise`.
- `test_options_signal.py` — set the signal mid-call (via `FauxImagesProvider` with an `asyncio.sleep` inside), assert result is `AssistantImages(stop_reason="aborted", output=[])` and `subscribe_response` saw `exc=CancelledError`.
- `test_listener_registry.py` — `subscribe_request` is called with deep-copied payload (mutating in one listener doesn't affect others); `subscribe_response` fires once per call in `finally`; detach callable works.

### 10.5 Documentation

- **Rewrite** `website/docs/guides/providers/image-generation.md` (English) following the new structure: conceptual model → `provider.model()` → `ImagesContext` → `ImagesOptions` → `ImagesCapabilityDescriptor` (with four worked backends) → errors → listeners → cancellation → Roadmap.
- **Rewrite** `website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/guides/providers/image-generation.md` (Chinese, same structure).
- **Sync** the version-0.7 mirrors at `website/versioned_docs/version-0.7/guides/providers/image-generation.md` and the `i18n` zh-Hans mirror — 0.7 hasn't shipped, so versioned docs should reflect the new shape.
- **Add a paragraph** to `website/docs/guides/providers/overview.md` (and the zh-Hans mirror) introducing the image-provider entry point and linking out to `image-generation.md`.
- **CHANGELOG**: under the existing `[0.7.0]` section, add a "Breaking" bullet describing the image-provider redesign.

## 11. Verified Backend Compatibility (research notes)

Verified June 2026; see references in §13.

| Backend | URL shape | Field compatibility | Notes |
|---|---|---|---|
| OpenAI official | `/v1/images/generations` | baseline | — |
| Volcengine Ark / Doubao Seedream | `/api/v3/images/generations` | High | extra fields `watermark`, `seed`, `sequential_image_generation`; `size` accepts `"1K"/"2K"/"4K"` strings as well as pixel strings |
| SiliconFlow | `/v1/images/generations` | **Low** (URL is OpenAI-shaped, fields are not) | `image_size` not `size`; `batch_size` not `n`; `num_inference_steps`; `guidance_scale` |
| Together AI | `/v1/images/generations` | Medium | FLUX schnell / Kontext: `aspect_ratio`; FLUX pro: `width/height`; extras: `steps`, `seed`, `negative_prompt`, `reference_images` |
| Aliyun DashScope Wanxiang | `/api/v1/services/aigc/text2image/image-synthesis` | **Not compatible — async task model** | submit → poll `task_id` → fetch. Out of scope for this spec; future `AsyncTaskImagesProvider` work. |
| Google Imagen on Vertex | Vertex native | **Not compatible** | `aspectRatio`, `numberOfImages`. Same future work as above. |
| Stability / Replicate / fal native / FLUX official | Each native | **Not compatible** | Various per-backend shapes. Same future work. |

The capability descriptor's value rests on the "Low + Medium + High" rows above; the "Not compatible" rows are deferred to future async-task scaffolding.

## 12. Open Questions and Risks

- **Tracing wiring.** `cubepi.tracing` currently subscribes to chat providers' request/response observers. After this spec lands, we want it to subscribe to image providers too. This is a small change but lives outside this spec — it should be a follow-up PR that depends on this one.
- **`classify_and_raise` signature widening.** §8 proposes widening its `model` parameter to a structural type. If reviewers prefer a narrower change (e.g. a sibling `classify_and_raise_images`), the implementation will adapt — neither is hard.
- **`ImagesCapabilityDescriptor` field maximalism risk.** We've kept the field list to what we verified backends actually need. If the implementation review surfaces a fifth verified divergence we missed, we add a field; we are not pre-designing for hypothetical backends.
- **`response_format` default.** The descriptor defaults `response_format_value` to `"b64_json"` to match the current `OpenAIImagesProvider` behavior. Some backends only support `"url"` and the URL is short-lived (SiliconFlow says 1 hour). When a backend returns URLs, the provider should download or surface the URLs verbatim — to be settled during implementation; the descriptor allows either.

## 13. References

- CubePi chat-side baseline:
  - `cubepi/providers/base.py` (Provider/BaseProvider, BoundModel, StreamOptions, listener registry)
  - `cubepi/providers/capability.py` (CapabilityDescriptor)
  - `cubepi/errors.py` (ProviderError taxonomy)
  - `cubepi/agent/agent.py` (BoundModel consumption pattern)
  - `CHANGELOG.md` `[0.7.0]` section
  - `website/docs/guides/providers/overview.md`
- Verified backend documentation (June 2026):
  - [Volcengine Ark / BytePlus ModelArk Image Generation API](https://docs.byteplus.com/en/docs/ModelArk/1541523)
  - [SiliconFlow Image Generations API reference](https://docs.siliconflow.cn/en/api-reference/images/images-generations)
  - [Together AI image generation overview](https://docs.together.ai/docs/images-overview)
  - [Alibaba Cloud DashScope vs OpenAI compatibility](https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope)
