from __future__ import annotations

from typing import Protocol, runtime_checkable

from cubepi.providers.images.types import AssistantImages, ImagesContext, ImagesModel
from cubepi.types import JsonObject, JsonValue


@runtime_checkable
class ImagesProvider(Protocol):
    api: str

    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        options: JsonObject | None = None,
    ) -> AssistantImages: ...


_PROVIDER_CLASSES: dict[str, type[ImagesProvider]] = {}


def register_images_provider_class(api: str, cls: type[ImagesProvider]) -> None:
    _PROVIDER_CLASSES[api] = cls


def create_images_provider(api: str, **kwargs: JsonValue) -> ImagesProvider:
    cls = _PROVIDER_CLASSES.get(api)
    if cls is None:
        raise ValueError(f"No images provider registered for api: {api}")
    return cls(**kwargs)
