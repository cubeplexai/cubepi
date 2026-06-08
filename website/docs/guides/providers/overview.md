---
title: Providers Overview
description: "Provider setup, capabilities, and presets in CubePi."
---

# Providers Overview

_Start here for provider setup in CubePi._

This page is the entry point for provider configuration. It covers the
default path for Anthropic and OpenAI, then explains how to describe
wire-level differences and preset overrides when you need to reach a
different backend without writing per-vendor glue. The progression is
deliberate:

1. **The default is zero config.** For Anthropic and OpenAI you write a
   provider and a model. Nothing on this page is required.
2. **For an off-default endpoint, describe the quirks as data.** A
   `CapabilityDescriptor` captures the differences declaratively — no
   subclassing, no forking.

## `provider.model(...)` parameters

Use `provider.model(model_id, ...)` to create a bound model for an agent.
`model_id` is required and positional; everything else is optional keyword
arguments:

- `api: str` — alternate API name/route tag for downstream integrations.
- `reasoning: bool` — enable reasoning mode and thinking-level negotiation.
- `context_window: int` — context-capacity hint used for validation and prompt planning.
- `max_tokens: int` — default max generation cap for this model.
- `temperature: float` — default sampling temperature for this model.
- `cost: ModelCost | None` — optional cost metadata object.
- `thinking_level_map: dict[str, str | None] | None` — optional map for level
  overrides and unsupported levels (`None` disables a level).

### Calling a bound model directly

`provider.model(...)` returns a `BoundModel` that you can invoke without
fishing the provider back out:

```python
from cubepi.providers.base import TextContent, UserMessage

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
for utilities (summarizers, classifiers) where you already hold a
`BoundModel` and want to skip the agent loop.

## `CapabilityDescriptor` is what to use when behavior differs by backend

`CapabilityDescriptor` is passed to a provider to express wire differences for
all models served by that provider:

```python
from cubepi import CapabilityDescriptor
from cubepi.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    api_key="...",
    base_url="https://api.deepseek.com",
    capability=CapabilityDescriptor(
        reasoning_on_payload={"extra_body": {"thinking": True}},
        reasoning_off_payload={"extra_body": {"thinking": False}},
        max_tokens_field="max_completion_tokens",
    ),
)
```

If only one model needs an override, use
`model_capability_overrides`:

```python
from cubepi import CapabilityDescriptor
from cubepi.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    api_key="...",
    base_url="https://openrouter.ai/api/v1",
    capability=CapabilityDescriptor(
        reasoning_on_payload={"extra_body": {"thinking": True}},
    ),
    model_capability_overrides={
        "deepseek-r1": CapabilityDescriptor(
            reasoning_on_payload={"extra_body": {"thinking": "enabled"}},
        ),
    },
)
```

`model_capability_overrides` is matched by exact `model_id`.

`CapabilityDescriptor` supports these fields:

- `reasoning_on_payload / reasoning_off_payload` — payload merged when
  reasoning is on/off.
- `reasoning_level` (`ReasoningLevelSpec`) — map `off`/`minimal`/... to backend
  payload paths.
- `temperature` (`TemperatureSpec`) — clip, force, or strip temperature.
- `max_tokens_field` — pick `max_tokens` or `max_completion_tokens`.
- `supports_tools` / `supports_images` / `supports_streaming` — metadata consumed
  by host UI and product code.

:::note Preset catalogs live in the host application
CubePi ships the **mechanism** (the `CapabilityDescriptor` and the wire
runtime that applies it), not a catalog of vendors. A ready-made list of
providers — base URLs, auth, regional/coding-plan endpoints, model lists —
is product data and belongs to the application embedding CubePi (for
example, cubebox maintains its own provider catalog). To reach a specific
vendor, build the provider with the right `base_url` + `CapabilityDescriptor`
as shown below.
:::

## 1. The simple case — no config at all

Most users never touch capabilities. The built-in providers ship with
sensible defaults:

```python
import cubepi
from cubepi import Agent
from cubepi.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(provider_id="anthropic")  # reads ANTHROPIC_API_KEY
agent = Agent(model=provider.model("claude-sonnet-4-6"))
await agent.prompt("Hello!")
```

That's the whole setup. A provider built without `capability=` produces
byte-identical output to CubePi `0.4` — the machinery below only kicks in
when you ask for it.

## 2. Off-default endpoints — the CapabilityDescriptor

When you want a model that isn't OpenAI or Anthropic — DeepSeek, Qwen,
Doubao, an OpenRouter route, a local server — the awkward part is each
one's wire dialect (does it want `max_tokens` or `max_completion_tokens`?
how is reasoning toggled?). You don't subclass a provider; you describe the
quirks as a [`CapabilityDescriptor`](pathname:///pydoc/cubepi/providers/capability.html)
and pass it in, along with the right `base_url` and provider class for the
endpoint's wire shape:

```python
import os
from cubepi import CapabilityDescriptor
from cubepi.providers.openai import OpenAIProvider

provider = OpenAIProvider(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    capability=CapabilityDescriptor(
        reasoning_off_payload={"extra_body": {"reasoning": {"exclude": True}}},
        reasoning_on_payload={"extra_body": {"reasoning": {"exclude": False}}},
    ),
)
```

Pick the provider class by the endpoint's wire dialect:

| Wire dialect | Provider class |
| --- | --- |
| `anthropic-messages` | `AnthropicProvider` |
| `openai-completions` | `OpenAIProvider` |
| `openai-responses` | `OpenAIResponsesProvider` |

Each field maps to one wire behavior, and an unset field does nothing — so
you only declare what's actually different.

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

Beyond on/off, CubePi maps a `ThinkingLevel`
(`off`/`minimal`/`low`/`medium`/`high`/`xhigh`) onto a concrete wire value
written at a dotted `path`. `kind` picks the shape:

`ReasoningLevelSpec` only changes how that level is serialized. You still need
two call-site controls:

- set `reasoning=True` when binding the model (enable reasoning for that model)
- set the Agent's `thinking` argument to one of `off|minimal|low|medium|high|xhigh`
  (defaults to `off`).

```python
from cubepi import CapabilityDescriptor, ReasoningLevelSpec
from cubepi.providers.openai import OpenAIProvider
from cubepi import Agent

provider = OpenAIProvider(
    api_key="...",
    capability=CapabilityDescriptor(
        reasoning_on_payload={"extra_body": {"reasoning": {"enabled": True}}},
        reasoning_level=ReasoningLevelSpec(
            path="reasoning.effort",
            kind="effort",
            level_to_effort={
                "off": "low",
                "minimal": "low",
                "low": "low",
                "medium": "medium",
                "high": "high",
                "xhigh": "high",
            },
        ),
    ),
)

agent = Agent(model=provider.model("deepseek-r1", reasoning=True), thinking="high")
```

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

Declarative flags read by host applications and frontends (for example, to
grey out image upload). The providers themselves don't gate on them.

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
`capability`.

## Image providers

Image generation has its own provider surface (`cubepi.providers.images`)
that follows the same conventions described above: `provider_id` on the
provider, `provider.model("id", ...)` factory, typed `ProviderError`
failures, and a capability descriptor for backend wire differences. See
[Image Generation](./image-generation) for the full guide.

## See also

- [OpenAI Provider](./openai) — concrete OpenAI / OpenAI-compatible setup.
- [Anthropic Provider](./anthropic) — the `int_budget` reasoning shape in practice.
- [Writing a Custom Provider](./custom) — when the endpoint isn't even OpenAI/Anthropic-shaped.
- [API Reference → `cubepi.providers`](../../api/cubepi-providers).
