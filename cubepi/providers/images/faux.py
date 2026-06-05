from __future__ import annotations

from cubepi.errors import ProviderError
from cubepi.providers.base import ImageContent
from cubepi.providers.images.base import BaseImagesProvider
from cubepi.providers.images.types import (
    AssistantImages,
    ImagesContext,
    ImagesModel,
    ImagesOptions,
)


class FauxImagesProvider(BaseImagesProvider):
    """Deterministic image-generation stub for tests.

    Returns a single image whose b64 body is the value passed at construction
    time. ``raise_on_call`` lets tests inject a typed ``ProviderError``
    subclass (e.g. ``RateLimited``) so retry middleware can be exercised
    against the image path.
    """

    def __init__(
        self,
        *,
        provider_id: str = "faux",
        png_b64: str,
        raise_on_call: type[ProviderError] | None = None,
    ) -> None:
        super().__init__(provider_id=provider_id)
        self._png_b64 = png_b64
        self._raise = raise_on_call

    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        *,
        options: ImagesOptions | None = None,
    ) -> AssistantImages:
        # Honor a pre-set abort signal the same way the real provider does
        # — return ``stop_reason="aborted"`` without the happy-path image
        # so tests / local flows that exercise cancellation behave the same
        # against Faux as they would against ``OpenAIImagesProvider``.
        if options and options.signal and options.signal.is_set():
            return AssistantImages(
                api=model.api,
                provider_id=model.provider_id,
                model=model.id,
                output=[],
                stop_reason="aborted",
            )
        if self._raise is not None:
            raise self._raise(
                f"injected by FauxImagesProvider for {model.provider_id}/{model.id}"
            )
        return AssistantImages(
            api=model.api,
            provider_id=model.provider_id,
            model=model.id,
            output=[ImageContent(source=self._png_b64, media_type="image/png")],
            stop_reason="stop",
        )
