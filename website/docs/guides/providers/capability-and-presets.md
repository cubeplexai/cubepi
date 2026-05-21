---
title: Capabilities & Preset Catalog
---

# Capabilities & Preset Catalog

_Added in cubepi `0.5`._

This page is about reaching **any** model — the big SaaS APIs, a regional
endpoint, a coding-plan tier, or your own vLLM box — without writing
per-vendor glue. The progression is deliberate:

1. **The default is zero config.** For Anthropic and OpenAI you write a
   provider and a model. Nothing on this page is required.
2. **For a specific model/provider, grab a preset.** One lookup gives you
   the right base URL and wire shape.
3. **To go off-catalog, describe the quirks as data.** A
   `CapabilityDescriptor` captures the differences declaratively — no
   subclassing, no forking.

## 1. The simple case — no config at all

Most users never touch capabilities or presets. The built-in providers
ship with sensible defaults:

```python
import cubepi
from cubepi import Agent, Model
from cubepi.providers.anthropic import AnthropicProvider

agent = Agent(
    provider=AnthropicProvider(),                 # reads ANTHROPIC_API_KEY
    model=Model(id="claude-sonnet-4-6", provider="anthropic"),
)
await agent.prompt("Hello!")
```

That's the whole setup. A provider built without `capability=` produces
byte-identical output to cubepi `0.4` — the machinery below only kicks in
when you ask for it.

## 2. Targeting a specific model — use a preset

When you want a model that isn't OpenAI or Anthropic — DeepSeek, Qwen,
Doubao, an OpenRouter route, a local server — the awkward part is
remembering each one's base URL, auth header, and wire dialect (does it
want `max_tokens` or `max_completion_tokens`? how is reasoning toggled?).

The **preset catalog** answers that for you. Look one up by slug:

```python
import cubepi

cubepi.list_provider_presets()         # every preset, in catalog order
preset = cubepi.get_provider_preset("deepseek-openai")
```

A preset is plain data. Pick the provider class by its `api` field and
hand over the preset's settings — base URL, the right
`CapabilityDescriptor`, and any per-model overrides — in one shot:

```python
import os
import cubepi
from cubepi.providers.openai import OpenAIProvider
from cubepi.providers.openai_responses import OpenAIResponsesProvider
from cubepi.providers.anthropic import AnthropicProvider

PROVIDER_FOR_API = {
    "anthropic-messages": AnthropicProvider,
    "openai-completions": OpenAIProvider,
    "openai-responses": OpenAIResponsesProvider,
}

preset = cubepi.get_provider_preset("deepseek-openai")
provider = PROVIDER_FOR_API[preset.api](
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=preset.base_url,
    capability=preset.capability,
    model_capability_overrides=preset.model_capability_overrides,
)

model = cubepi.Model(id=preset.default_models[0].model_id, provider=preset.slug)
```

You didn't have to know that DeepSeek's OpenAI-compatible endpoint renames
the token field or how it flips reasoning on — the preset already encodes
it.

### What's in the catalog

Anthropic · OpenAI (Responses) · OpenAI (Chat Completions) · Qwen /
DashScope · Doubao / Volcengine · DeepSeek (both Anthropic and OpenAI
shapes) · Moonshot · Zhipu · MiniMax · xAI · Mistral · OpenRouter ·
Together AI · Groq · Fireworks · vLLM · Ollama · LM Studio · HuggingFace
TGI — plus coding-plan tiers, CN-region companions, and
`custom-openai` / `custom-anthropic` blanks to start from.

`cubepi.list_provider_presets()` is the authoritative, current list.

### What a preset carries

Each `ProviderPreset` bundles everything you need to reach an endpoint:

| Field | What it gives you |
| --- | --- |
| `slug` | Stable lookup key (`"anthropic"`, `"deepseek-openai"`, …). |
| `display_name` / `short_name` | Human labels for a UI. |
| `category` | `"saas"`, `"oss-framework"`, or `"custom"`. |
| `logo` | `@lobehub/icons` provider id for rendering an icon (`None` → generic fallback). |
| `api` | Wire dialect — picks the provider class. |
| `base_url` | Default endpoint. |
| `auth` | An `AuthSpec` (`mode`, `header_name`, `header_prefix`). |
| `capability` | The pre-built `CapabilityDescriptor` for this endpoint. |
| `model_capability_overrides` | Per-model descriptor overrides (see §3). |
| `default_models` | `ModelPreset` list: id, context window, max tokens, modalities, reasoning flag. |

## 3. Going off-catalog — the CapabilityDescriptor

If your endpoint isn't in the catalog (a brand-new gateway, an internal
proxy, a tweaked deployment), you don't subclass a provider. You describe
its quirks as a [`CapabilityDescriptor`](pathname:///pydoc/cubepi/providers/capability.html)
and pass it in:

```python
from cubepi import CapabilityDescriptor
from cubepi.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    api_key="…",
    base_url="https://my-gateway.example/v1",
    capability=CapabilityDescriptor(max_tokens_field="max_completion_tokens"),
)
```

Each field maps to one wire behavior, and an unset field does nothing —
so you only declare what's actually different.

### `max_tokens_field`

`"max_tokens"` (default) or `"max_completion_tokens"`. Some
OpenAI-compatible servers accept only one spelling; this renames the key
on the way out. **Effect:** wrong choice → the server ignores your output
cap or 400s.

### `temperature`

A `TemperatureSpec` controlling how the caller's temperature is treated:

```python
from cubepi import TemperatureSpec

TemperatureSpec(mode="free", min=0.0, max=2.0, default=1.0)  # clamp into [min, max]
TemperatureSpec(mode="fixed", fixed_value=1.0)               # always overwrite
TemperatureSpec(mode="ignored")                              # drop the key
```

- **`free`** — the caller's value is clamped into `[min, max]`; if none was
  sent, nothing is written. **Effect:** protects against an out-of-range
  value the backend would reject.
- **`fixed`** — `fixed_value` always wins. **Effect:** use for models that
  permit only one temperature (e.g. some o-series reasoning models).
- **`ignored`** — the key is stripped entirely. **Effect:** for backends
  that 400 on any `temperature` while reasoning.

### Reasoning toggle: `reasoning_off_payload` / `reasoning_on_payload`

When thinking is off, `reasoning_off_payload` is deep-merged into the
request; when it's on, `reasoning_on_payload` is. **Effect:** this is how
"turn reasoning on/off" becomes whatever field the vendor expects:

```python
CapabilityDescriptor(
    reasoning_off_payload={"extra_body": {"enable_thinking": False}},
    reasoning_on_payload={"extra_body": {"enable_thinking": True}},
)
```

The merge recurses into nested dicts; arrays are atomic; on a collision
the capability value wins.

### Reasoning level: `reasoning_level` (three shapes)

Beyond on/off, cubepi maps a `ThinkingLevel`
(`off`/`minimal`/`low`/`medium`/`high`/`xhigh`) onto a concrete wire value
written at a dotted `path`. `kind` picks the shape:

```python
from cubepi import ReasoningLevelSpec

# int_budget — a token budget (Anthropic).
ReasoningLevelSpec(
    path="thinking.budget_tokens", kind="int_budget",
    level_budgets={"off": 0, "minimal": 1024, "low": 2048,
                   "medium": 8192, "high": 16384, "xhigh": 16384},
)

# effort — an effort string (OpenAI Responses).
ReasoningLevelSpec(
    path="reasoning.effort", kind="effort",
    level_to_effort={"minimal": "minimal", "low": "low",
                     "medium": "medium", "high": "high", "xhigh": "high"},
)

# enum — a vendor-specific state (Doubao's 3-state thinking).
ReasoningLevelSpec(
    path="thinking.type", kind="enum",
    level_to_enum={"off": "disabled", "low": "enabled", "high": "enabled"},
)
```

**Effect:** a level missing from the map is simply not written, so the
endpoint keeps its own default for that level.

### `supports_tools` / `supports_images` / `supports_streaming`

Declarative flags read by the catalog and frontends (for example, to grey
out image upload). The providers themselves don't gate on them.

### Per-model overrides on a shared endpoint

One gateway (OpenRouter, LiteLLM, an internal proxy) often serves both
reasoning and non-reasoning models. `model_capability_overrides` maps a
`model_id` to a descriptor that **replaces** the base one for that model:

```python
provider = OpenAIProvider(
    api_key="…",
    base_url="https://openrouter.ai/api/v1",
    capability=base_cap,                        # default for unlisted models
    model_capability_overrides={
        "deepseek/deepseek-r1": reasoning_cap,  # this model only
    },
)
```

Resolution is exact-match on `model_id`; anything not listed falls back to
`capability`. This is exactly how the `openrouter` preset ships.

## See also

- [OpenAI Provider](./openai) — concrete OpenAI / OpenAI-compatible setup.
- [Anthropic Provider](./anthropic) — the `int_budget` reasoning shape in practice.
- [Writing a Custom Provider](./custom) — when the endpoint isn't even OpenAI/Anthropic-shaped.
- [API Reference → `cubepi.providers`](../../api/cubepi-providers).
