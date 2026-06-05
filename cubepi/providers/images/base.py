from __future__ import annotations

from typing import Any, Callable, Literal, Protocol, runtime_checkable

from cubepi.providers.base import (
    OnRequestCallback,
    OnResponseBodyCallback,
    _detach,
)
from cubepi.providers.images.capability import ImagesCapabilityDescriptor
from cubepi.providers.images.types import (
    AssistantImages,
    ImagesContext,
    ImagesCost,
    ImagesModel,
    ImagesOptions,
)


@runtime_checkable
class ImagesProvider(Protocol):
    """Protocol for image-generation providers.

    Provider classes implement ``generate_images(model, context, options=...)``
    and expose ``provider_id``. They do NOT need to subclass
    :class:`BaseImagesProvider`, but built-in providers and most user
    implementations should — the base class supplies the ``.model()``
    factory, listener registry, and capability-application helper.
    """

    provider_id: str

    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        *,
        options: ImagesOptions | None = None,
    ) -> AssistantImages: ...


class BaseImagesProvider:
    """Concrete base class for built-in and user-defined image providers.

    Mirrors the role of :class:`cubepi.providers.base.BaseProvider` in chat:
    holds ``provider_id``, exposes a ``.model(...)`` factory that propagates
    it onto :class:`ImagesModel`, and runs request/response observer
    registries. Image is one-shot (no streamed chunks), so there is no
    ``subscribe_chunk``.
    """

    def __init__(
        self,
        *,
        provider_id: str = "",
        capability: ImagesCapabilityDescriptor | None = None,
        model_capability_overrides: dict[str, ImagesCapabilityDescriptor] | None = None,
    ) -> None:
        self.provider_id = provider_id
        self._capability = capability or ImagesCapabilityDescriptor()
        self._model_capability_overrides: dict[str, ImagesCapabilityDescriptor] = (
            dict(model_capability_overrides) if model_capability_overrides else {}
        )
        self._request_listeners: list[OnRequestCallback] = []
        self._response_listeners: list[OnResponseBodyCallback] = []

    # ──── Factory ────────────────────────────────────────────────
    def model(
        self,
        id: str,
        *,
        api: str = "",
        default_size: str | None = None,
        default_n: int | None = None,
        default_quality: Literal["low", "medium", "high"] | None = None,
        default_output_format: Literal["png", "jpeg", "webp"] | None = None,
        cost: ImagesCost | None = None,
        max_input_images: int | None = None,
    ) -> ImagesModel:
        return ImagesModel(
            id=id,
            provider_id=self.provider_id,
            api=api,
            default_size=default_size,
            default_n=default_n,
            default_quality=default_quality,
            default_output_format=default_output_format,
            cost=cost,
            max_input_images=max_input_images,
        )

    # ──── Protocol method — subclass implements ─────────────────
    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        *,
        options: ImagesOptions | None = None,
    ) -> AssistantImages:
        raise NotImplementedError

    # ──── Listener registry ──────────────────────────────────────
    def subscribe_request(self, cb: OnRequestCallback) -> Callable[[], None]:
        self._request_listeners.append(cb)
        return lambda: _detach(self._request_listeners, cb)

    def subscribe_response(self, cb: OnResponseBodyCallback) -> Callable[[], None]:
        self._response_listeners.append(cb)
        return lambda: _detach(self._response_listeners, cb)

    # ──── Helpers for subclasses ─────────────────────────────────
    def _capability_for(self, model: ImagesModel) -> ImagesCapabilityDescriptor:
        """Resolve the descriptor that applies to ``model`` (per-model override > base)."""
        return self._model_capability_overrides.get(model.id, self._capability)

    def _build_payload(
        self, model: ImagesModel, context: ImagesContext
    ) -> dict[str, Any]:  # pragma: no cover — implemented in Task 5
        raise NotImplementedError("_build_payload arrives in Task 5")
