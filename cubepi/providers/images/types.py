from __future__ import annotations

import asyncio
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cubepi.providers.base import (
    ImageContent,
    OnPayloadCallback,
    OnResponseCallback,
    ProviderResponse,
    TextContent,
)
from cubepi.types import JsonObject

# Re-exported so the ``OnResponseCallback`` forward reference to
# ``ProviderResponse`` can be resolved when pydantic builds ``ImagesOptions``
# under ``from __future__ import annotations``.
__all__ = [
    "AssistantImages",
    "ImagesAborted",
    "ImagesContext",
    "ImagesCost",
    "ImagesModel",
    "ImagesOptions",
    "ProviderResponse",
]


class ImagesAborted(Exception):
    """Signals that a :meth:`generate_images` call was aborted via
    :attr:`ImagesOptions.signal`.

    Surfaced to ``subscribe_response`` observers as the ``exc`` argument so
    tracing / audit listeners can distinguish a deliberate abort from a
    normal completion (``exc=None``) or a real failure (``ProviderError``
    subclass). Deliberately **not** an :class:`asyncio.CancelledError` —
    the response-listener fanout takes a synchronous fast-path when ``exc``
    is a cancellation, which schedules async listeners as detached tasks
    that can be cancelled by ``asyncio.run()`` teardown before they run.
    Using a distinct, non-cancellation exception keeps observers on the
    awaited path so they finish before the call returns.
    """


class ImagesCost(BaseModel):
    """Image-generation pricing (per-image is dominant; per-megapixel for Imagen-like models)."""

    per_image: float = 0
    per_megapixel: float = 0


class ImagesModel(BaseModel):
    """Image-generation model spec, with model-level defaults applied when the
    matching ``ImagesContext`` field is not set."""

    id: str
    provider_id: str = ""
    api: str = ""

    default_size: str | None = None
    default_n: int | None = None
    default_quality: Literal["low", "medium", "high"] | None = None
    default_output_format: Literal["png", "jpeg", "webp"] | None = None

    cost: ImagesCost | None = None
    max_input_images: int | None = None


class ImagesContext(BaseModel):
    """Per-call request payload.

    ``size`` / ``n`` / ``quality`` / ``output_format`` override the matching
    ``ImagesModel.default_*``. ``seed`` / ``negative_prompt`` / ``steps`` /
    ``guidance`` are only written to the wire payload when the provider's
    ``ImagesCapabilityDescriptor`` declares support; otherwise they are
    dropped with a one-time warning. ``extra`` carries truly backend-specific
    fields that the descriptor does not model (e.g. Doubao's ``watermark``).
    """

    prompt: str
    input_images: list[ImageContent] = Field(default_factory=list)

    size: str | None = None
    n: int | None = None
    quality: Literal["low", "medium", "high"] | None = None
    output_format: Literal["png", "jpeg", "webp"] | None = None

    seed: int | None = None
    negative_prompt: str | None = None
    steps: int | None = None
    guidance: float | None = None

    extra: dict[str, Any] = Field(default_factory=dict)


class ImagesOptions(BaseModel):
    """Per-call cross-cutting options (analog of chat's ``StreamOptions``)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    signal: asyncio.Event | None = None
    on_payload: OnPayloadCallback | None = None
    on_response: OnResponseCallback | None = None


class AssistantImages(BaseModel):
    """Response from a successful or aborted image generation call.

    Failures raise ``cubepi.errors.ProviderError`` subclasses; they never
    appear as a ``stop_reason``. ``stop_reason="aborted"`` is produced when
    ``ImagesOptions.signal`` fires mid-call.
    """

    api: str
    provider_id: str
    model: str
    output: list[ImageContent | TextContent] = Field(default_factory=list)
    stop_reason: Literal["stop", "aborted"] = "stop"
    response_id: str | None = None
    metadata: JsonObject = Field(default_factory=dict)
