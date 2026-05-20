from __future__ import annotations

from typing import Any

from cubepi.providers.base import ImageContent
from cubepi.providers.images.registry import register_images_provider
from cubepi.providers.images.types import AssistantImages, ImagesContext, ImagesModel


class FauxImagesProvider:
    api = "faux-images"

    def __init__(self, png_b64: str) -> None:
        self._png_b64 = png_b64

    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        options: dict[str, Any] | None = None,
    ) -> AssistantImages:
        return AssistantImages(
            api=model.api,
            provider=model.provider,
            model=model.id,
            output=[ImageContent(source=self._png_b64, media_type="image/png")],
            stop_reason="stop",
        )


def register_faux_images(png_b64: str) -> None:
    register_images_provider(FauxImagesProvider(png_b64))
