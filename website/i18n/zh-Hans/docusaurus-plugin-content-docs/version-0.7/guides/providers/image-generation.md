---
title: 图片生成
description: "使用 CubePi 的图片生成能力。"
---

# 图片生成

这一页介绍 CubePi 的独立图片生成链路。它与 `Agent` 有意分离：对话场景仍然走
`Model`，图片场景走 `ImagesModel`，并直接调用图片 provider。

## `ImagesModel` 类型

图片生成使用 `cubepi.providers.images.types` 中的三个基础类型：

```python
from cubepi.providers.images.types import AssistantImages, ImagesContext, ImagesModel
```

- `ImagesModel`：图片模型绑定（`id`、`provider`、`api`）。
- `ImagesContext`：生成请求体（`prompt`、可选 `input_images`）。
- `AssistantImages`：图片 provider 的返回值，包含 `output`、`stop_reason`，
  以及可选的 `error_message`。

`input_images` 和 `AssistantImages.output` 中用到的是
`ImageContent`（定义在 `cubepi.providers.base`）。

## 创建图片模型

OpenAI 图片模型一般通过 `create_images_provider("openai-images", ...)` 创建
provider，并用 `ImagesModel` 绑定：

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

`api` 是给上游系统做路由与归因的标识，可按需自定义。

## 文生图

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
    context=ImagesContext(prompt="一只戴帽子的猫在日落中"),
    options={"size": "1024x1024", "quality": "high", "n": 2},
)

if resp.stop_reason == "error":
    raise RuntimeError(resp.error_message or "图片生成失败")

for block in resp.output:
    print(block.type, block.source[:30], block.media_type)
```

如果不传 `options`，provider 会按默认参数发送（`n=1`），且不带可选的图片参数。

## 图像编辑

`input_images` 非空时自动走编辑链路：

```python
import base64
from cubepi.providers.base import ImageContent
from cubepi.providers.images.types import ImagesContext

with open("source.png", "rb") as fh:
    source_b64 = base64.b64encode(fh.read()).decode("ascii")

ctx = ImagesContext(
    prompt="让它更亮一些，色温更暖。",
    input_images=[ImageContent(source=source_b64, media_type="image/png")],
)
```

`generate_images(model, context=ctx)` 会调用 `edit`，而不是 `generate`。

## `options` 下发行为

`generate_images(..., options)` 是一个透传字典（在补齐 `model` / `prompt` /
`n` 后）直接传给后端。OpenAI 图片 provider 常见参数：

- `size`
- `quality`
- `n`
- `output_format`（`png` / `jpeg` / `webp`）
- `background`
- 以及网关允许的其他扩展参数

provider 只对 `output_format` 做输出类型映射：
`png` -> `image/png`，`jpeg` -> `image/jpeg`，`webp` -> `image/webp`。

## 错误处理

后端失败时 `generate_images(...)` 会返回 `AssistantImages`，并包含：

- `stop_reason="error"`
- `output=[]`
- `error_message=<后端报错>`

这样可以把调用端的成功 / 失败逻辑统一到一个返回分支中处理。

## 测试与模拟

本地测试可以直接使用 `FauxImagesProvider` 产出稳定图像结果：

```python
from cubepi.providers.images.faux import FauxImagesProvider

provider = FauxImagesProvider(png_b64="iVBORw0KGgoAAAANSUh...")
```

适合不依赖真实图片 API 的端到端或 UI 测试。

## 参见

- [Providers Overview](./overview) —— 默认行为和能力描述符。
- [OpenAI Provider](./openai) —— OpenAI 与 OpenAI 兼容网关的配置方式。
- [API Reference → providers（图片类型）](../../api/cubepi-providers)。
