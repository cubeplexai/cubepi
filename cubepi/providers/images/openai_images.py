from __future__ import annotations

import base64
import io
from typing import Any

from cubepi.providers.base import ImageContent
from cubepi.providers.images.registry import register_images_provider
from cubepi.providers.images.types import AssistantImages, ImagesContext, ImagesModel


class OpenAIImagesProvider:
    api = "openai-images"

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None) -> None:
        import openai

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client: Any = openai.AsyncOpenAI(**kwargs)

    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        options: dict[str, Any] | None = None,
    ) -> AssistantImages:
        params: dict[str, Any] = {"model": model.id, "prompt": context.prompt, "n": 1}
        if model.size != "auto":
            params["size"] = model.size
        if model.quality != "auto":
            params["quality"] = model.quality

        def _err(message: str) -> AssistantImages:
            return AssistantImages(
                api=model.api,
                provider=model.provider,
                model=model.id,
                output=[],
                stop_reason="error",
                error_message=message,
            )

        try:
            if context.input_images:
                files = [self._to_file(img) for img in context.input_images]
                resp = await self._client.images.edit(image=files, **params)
            else:
                resp = await self._client.images.generate(**params)
        except Exception as exc:  # noqa: BLE001
            return _err(str(exc))

        data = getattr(resp, "data", None) or []
        b64 = getattr(data[0], "b64_json", None) if data else None
        if not b64:
            return _err("image provider returned no image data")

        return AssistantImages(
            api=model.api,
            provider=model.provider,
            model=model.id,
            output=[ImageContent(source=b64, media_type="image/png")],
            stop_reason="stop",
        )

    @staticmethod
    def _to_file(img: ImageContent) -> io.BytesIO:
        buf = io.BytesIO(base64.b64decode(img.source))
        buf.name = "source.png"
        return buf


def register_openai_images(*, api_key: str | None = None, base_url: str | None = None) -> None:
    register_images_provider(OpenAIImagesProvider(api_key=api_key, base_url=base_url))
