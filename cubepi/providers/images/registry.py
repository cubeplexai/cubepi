from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from cubepi.providers.images.types import AssistantImages, ImagesContext, ImagesModel


@runtime_checkable
class ImagesProvider(Protocol):
    api: str

    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        options: dict[str, Any] | None = None,
    ) -> AssistantImages: ...


_REGISTRY: dict[str, ImagesProvider] = {}


def register_images_provider(provider: ImagesProvider) -> None:
    _REGISTRY[provider.api] = provider


def get_images_provider(api: str) -> ImagesProvider | None:
    return _REGISTRY.get(api)
