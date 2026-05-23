from __future__ import annotations

# Import openai_images so its module-level register_images_provider_class call runs,
# making create_images_provider("openai-images", ...) work out of the box.
from cubepi.providers.images import openai_images as _openai_images_mod  # noqa: F401

from cubepi.providers.images.faux import FauxImagesProvider
from cubepi.providers.images.openai_images import OpenAIImagesProvider
from cubepi.providers.images.registry import (
    ImagesProvider,
    create_images_provider,
    register_images_provider_class,
)
from cubepi.providers.images.types import (
    AssistantImages,
    ImagesContext,
    ImagesModel,
)

__all__ = [
    "AssistantImages",
    "ImagesContext",
    "ImagesModel",
    "ImagesProvider",
    "FauxImagesProvider",
    "OpenAIImagesProvider",
    "create_images_provider",
    "register_images_provider_class",
]
