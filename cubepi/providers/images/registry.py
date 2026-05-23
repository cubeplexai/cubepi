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


_PROVIDER_CLASSES: dict[str, type[ImagesProvider]] = {}


def register_images_provider_class(api: str, cls: type[ImagesProvider]) -> None:
    _PROVIDER_CLASSES[api] = cls


def create_images_provider(api: str, **kwargs: Any) -> ImagesProvider:
    cls = _PROVIDER_CLASSES.get(api)
    if cls is None:
        raise ValueError(f"No images provider registered for api: {api}")
    return cls(**kwargs)  # type: ignore[call-arg]
