from __future__ import annotations

import copy
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


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursive merge that returns a fully-isolated copy of ``base`` with
    ``overlay`` applied on top.

    Dict keys recurse; everything else (lists, scalars, custom objects)
    is overwritten by a deep-copied overlay value so the merged result
    shares NO references with the input dicts. This matters because the
    merged payload is then passed to an ``on_payload`` mutator (which can
    do in-place writes) and to the SDK (which may also mutate); without
    isolation, a mutator like ``payload["extra_body"]["watermark"] = True``
    would permanently change ``cap.extra_payload`` or ``context.extra``
    and leak into later requests.

    Used for ``capability.extra_payload`` and ``context.extra`` application.
    """
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


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
    ) -> dict[str, Any]:
        cap = self._capability_for(model)
        payload: dict[str, Any] = {"model": model.id, "prompt": context.prompt}

        # size — four wire shapes
        size = context.size if context.size is not None else model.default_size
        if size is not None:
            kind = cap.size_spec.kind
            if kind == "size_string":
                payload["size"] = size
            elif kind == "image_size_string":
                payload["image_size"] = size
            elif kind == "width_height":
                w_str, h_str = size.lower().split("x")
                payload["width"] = int(w_str)
                payload["height"] = int(h_str)
            elif kind == "aspect_ratio":
                payload["aspect_ratio"] = size

        # n
        n = context.n if context.n is not None else model.default_n
        if n is not None:
            payload[cap.count_field] = n

        # quality
        quality = (
            context.quality if context.quality is not None else model.default_quality
        )
        if quality is not None:
            payload["quality"] = quality

        # output_format
        of = (
            context.output_format
            if context.output_format is not None
            else model.default_output_format
        )
        if of is not None and cap.output_format_field is not None:
            payload[cap.output_format_field] = of

        # response_format — written only when the capability opts in
        # (None default avoids 400s on OpenAI GPT image models, which
        # reject the field and return base64 by default).
        if cap.response_format_field is not None:
            payload[cap.response_format_field] = cap.response_format_value

        # supports_* gating
        if context.seed is not None and cap.supports_seed:
            payload[cap.seed_field] = context.seed
        if context.negative_prompt is not None and cap.supports_negative_prompt:
            payload[cap.negative_prompt_field] = context.negative_prompt
        if context.steps is not None and cap.supports_steps:
            payload[cap.steps_field] = context.steps
        if context.guidance is not None and cap.supports_guidance:
            payload[cap.guidance_field] = context.guidance

        # Provider-level extra_payload (always injected) and per-call ctx.extra,
        # deep-merged in order: payload <- cap.extra_payload <- ctx.extra.
        payload = _deep_merge(payload, cap.extra_payload)
        payload = _deep_merge(payload, context.extra)
        return payload
