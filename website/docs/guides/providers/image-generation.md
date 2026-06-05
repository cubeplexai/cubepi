---
title: Image Generation
description: "Generate images with CubePi image providers."
---

# Image Generation

This page documents CubePi's separate image generation path. It is intentionally
independent from `Agent`: chat models use the `Model` path, while image models
use `ImagesModel` and run through an image provider instance directly.

## `ImagesModel` types

Image generation uses three small types from `cubepi.providers.images.types`:

```python
from cubepi.providers.images.types import AssistantImages, ImagesContext, ImagesModel
```

- `ImagesModel`: minimal model binding for image APIs (`id`, `provider`, `api`).
- `ImagesContext`: request body for generation (`prompt`, optional `input_images`).
- `AssistantImages`: provider response with `output`, `stop_reason`, and
  optional `error_message`.

`ImageContent` (from `cubepi.providers.base`) is used in `input_images` and
`AssistantImages.output`.

## Create image models

For OpenAI image models, use `create_images_provider("openai-images", ...)` and
an `ImagesModel`:

```python
import os

from cubepi.providers.images import create_images_provider
from cubepi.providers.images.types import ImagesContext, ImagesModel

image_provider = create_images_provider(
    "openai-images",
    api_key=os.environ["OPENAI_API_KEY"],
)
model = ImagesModel(id="gpt-image-1", provider="openai", api="openai-images")
```

`api` is an identifier used by downstream systems for routing and reporting.

## Generate text-to-image

```python
import os

from cubepi.providers.images import create_images_provider
from cubepi.providers.images.types import ImagesContext, ImagesModel

provider = create_images_provider(
    "openai-images",
    api_key=os.environ["OPENAI_API_KEY"],
)
model = ImagesModel(id="gpt-image-1", provider="openai", api="openai-images")

resp = await provider.generate_images(
    model=model,
    context=ImagesContext(prompt="A cute robot at sunrise"),
    options={"size": "1024x1024", "quality": "high", "n": 2},
)

if resp.stop_reason == "error":
    raise RuntimeError(resp.error_message or "image generation failed")

for block in resp.output:
    print(block.type, block.source[:30], block.media_type)
```

If `options` is omitted, the provider sends defaults (`n=1`) and no optional
image-only params.

## Edit with input images

Passing `input_images` switches to the edit path automatically:

```python
import base64
from cubepi.providers.base import ImageContent
from cubepi.providers.images.types import ImagesContext

with open("source.png", "rb") as fh:
    source_b64 = base64.b64encode(fh.read()).decode("ascii")

ctx = ImagesContext(
    prompt="Make it brighter and warmer.",
    input_images=[ImageContent(source=source_b64, media_type="image/png")],
)
```

`generate_images(model, context=ctx)` calls `edit` instead of `generate`.

## `options` forwarding

`generate_images(..., options)` is a free-form dict passed straight into the
backend call (after adding the required `model`, `prompt`, and default `n`).
The OpenAI image provider currently supports common keys like:

- `size`
- `quality`
- `n`
- `output_format` (`png`, `jpeg`, `webp`)
- `background`
- any backend extension fields accepted by your gateway

The provider only filters output media type for `output_format`:
`png` -> `image/png`, `jpeg` -> `image/jpeg`, `webp` -> `image/webp`.

## Error handling

On backend failures, `generate_images(...)` returns `AssistantImages` with:

- `stop_reason="error"`
- `output=[]`
- `error_message=<provider error>`

This keeps the API failure-in-band and lets callers render a single
success/error branch.

## Testing and stubs

For local tests, `FauxImagesProvider` provides a deterministic image output:

```python
from cubepi.providers.images.faux import FauxImagesProvider

provider = FauxImagesProvider(png_b64="iVBORw0KGgoAAAANSUh...")
```

It is useful for end-to-end UI tests where you only need a stable `ImageContent`.

## See also

- [Providers Overview](./overview) — when model-level wiring differs from defaults.
- [OpenAI Provider](./openai) — shared OpenAI config patterns.
- [API Reference → Image types in providers](../../api/cubepi-providers).
