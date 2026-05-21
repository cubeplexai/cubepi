from __future__ import annotations

from typing import Any

from cubepi.providers.images.registry import (
    ImagesProvider,
    get_images_provider,
    register_images_provider,
)
from cubepi.providers.images.types import (
    AssistantImages,
    ImagesContext,
    ImagesModel,
    ImagesQuality,
    ImagesSize,
)


async def generate_images(
    model: ImagesModel,
    context: ImagesContext,
    options: dict[str, Any] | None = None,
) -> AssistantImages:
    provider = get_images_provider(model.api)
    if provider is None:
        raise ValueError(f"No images provider registered for api: {model.api}")
    return await provider.generate_images(model, context, options)


__all__ = [
    "AssistantImages",
    "ImagesContext",
    "ImagesModel",
    "ImagesProvider",
    "ImagesQuality",
    "ImagesSize",
    "generate_images",
    "get_images_provider",
    "register_images_provider",
]
