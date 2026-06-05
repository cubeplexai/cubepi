from __future__ import annotations

from cubepi.errors import ProviderError
from cubepi.providers.base import (
    ImageContent,
    _fire_request_listeners,
    _fire_response_listeners,
)
from cubepi.providers.images.base import BaseImagesProvider
from cubepi.providers.images.types import (
    AssistantImages,
    ImagesAborted,
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
        # Faux mirrors OpenAIImagesProvider's observability semantics so
        # tests that exercise listener wiring against Faux see the same
        # contract as production: subscribe_request fires before the
        # "send"; subscribe_response fires in finally with body + exc.
        body: dict | None = None
        exc: BaseException | None = None
        try:
            # Pre-set abort short-circuit (no listeners observe a payload
            # for a call that won't run, matching the real provider).
            if options and options.signal and options.signal.is_set():
                raise ImagesAborted("signal was already set before generate_images()")

            # Synthetic payload for observers: same canonical shape the
            # real provider would have assembled at minimum.
            payload: dict = {"model": model.id, "prompt": context.prompt}
            if self._request_listeners:
                await _fire_request_listeners(self._request_listeners, payload, model)  # type: ignore[arg-type]

            if self._raise is not None:
                raise self._raise(
                    f"injected by FauxImagesProvider for {model.provider_id}/{model.id}"
                )

            body = {"data": [{"b64_json": self._png_b64, "media_type": "image/png"}]}
            return AssistantImages(
                api=model.api,
                provider_id=model.provider_id,
                model=model.id,
                output=[ImageContent(source=self._png_b64, media_type="image/png")],
                stop_reason="stop",
            )
        except ImagesAborted as aborted:
            exc = aborted
            return AssistantImages(
                api=model.api,
                provider_id=model.provider_id,
                model=model.id,
                output=[],
                stop_reason="aborted",
            )
        except BaseException as raw:
            # Includes the injected raise_on_call ProviderError. Record
            # for observers, then re-raise unchanged.
            exc = raw
            raise
        finally:
            if self._response_listeners:
                await _fire_response_listeners(
                    self._response_listeners,
                    body,
                    model,  # type: ignore[arg-type]
                    exc,
                )
