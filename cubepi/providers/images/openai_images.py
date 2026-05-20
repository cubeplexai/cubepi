from __future__ import annotations

import base64
import io
from typing import Any

from cubepi.providers.base import ImageContent, TextContent
from cubepi.providers.images.registry import register_images_provider
from cubepi.providers.images.types import AssistantImages, ImagesContext, ImagesModel


_MEDIA_TYPE_EXT = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
}

_OUTPUT_FORMAT_MEDIA_TYPE = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


class OpenAIImagesProvider:
    api = "openai-images"

    def __init__(
        self, *, api_key: str | None = None, base_url: str | None = None
    ) -> None:
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
        # Caller passthrough: output_format, output_compression, background, n, …
        # Merged last so callers can override the defaults above.
        if options:
            params.update(options)

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

        out_format = params.get("output_format", "png")
        media_type = _OUTPUT_FORMAT_MEDIA_TYPE.get(out_format, "image/png")

        data = getattr(resp, "data", None) or []
        images: list[ImageContent | TextContent] = [
            ImageContent(source=item.b64_json, media_type=media_type)
            for item in data
            if getattr(item, "b64_json", None)
        ]
        if not images:
            return _err("image provider returned no image data")

        return AssistantImages(
            api=model.api,
            provider=model.provider,
            model=model.id,
            output=images,
            stop_reason="stop",
        )

    @staticmethod
    def _to_file(img: ImageContent) -> io.BytesIO:
        ext = _MEDIA_TYPE_EXT.get(img.media_type, "png")
        buf = io.BytesIO(base64.b64decode(img.source))
        buf.name = f"source.{ext}"
        return buf


def register_openai_images(
    *, api_key: str | None = None, base_url: str | None = None
) -> None:
    register_images_provider(OpenAIImagesProvider(api_key=api_key, base_url=base_url))
