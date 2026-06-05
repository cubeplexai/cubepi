from __future__ import annotations

import asyncio
import base64
import io
from typing import Any

from cubepi.errors import classify_and_raise
from cubepi.providers.base import (
    ImageContent,
    OnPayloadCallback,
    ProviderResponse,
    TextContent,
    _fire_request_listeners,
    _fire_response_listeners,
    invoke_on_payload,
    invoke_on_response,
)
from cubepi.providers.images.base import BaseImagesProvider
from cubepi.providers.images.capability import ImagesCapabilityDescriptor
from cubepi.providers.images.types import (
    AssistantImages,
    ImagesContext,
    ImagesModel,
    ImagesOptions,
)

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


class OpenAIImagesProvider(BaseImagesProvider):
    """OpenAI-shape image provider.

    With the default ``ImagesCapabilityDescriptor`` this targets OpenAI's
    own ``/v1/images/generations`` endpoint. By supplying a different
    capability descriptor (and a matching ``base_url``) the same class
    targets other OpenAI-compatible backends: Volcengine Ark / Doubao
    Seedream, SiliconFlow, Together AI, and similar.
    """

    def __init__(
        self,
        *,
        provider_id: str = "",
        api_key: str | None = None,
        base_url: str | None = None,
        capability: ImagesCapabilityDescriptor | None = None,
        model_capability_overrides: dict[str, ImagesCapabilityDescriptor] | None = None,
    ) -> None:
        super().__init__(
            provider_id=provider_id,
            capability=capability,
            model_capability_overrides=model_capability_overrides,
        )
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
        *,
        options: ImagesOptions | None = None,
    ) -> AssistantImages:
        cap = self._capability_for(model)
        payload = self._build_payload(model, context)

        # input_images go in via the capability-declared field name; not
        # merged by _build_payload because they're not a wire dict slot.
        if context.input_images and cap.supports_edit:
            payload[cap.input_images_field] = [
                self._to_file(img) for img in context.input_images
            ]
            sdk_call = self._client.images.edit
        else:
            sdk_call = self._client.images.generate

        # per-call on_payload + persistent subscribe_request.
        # ``invoke_on_payload`` / ``_fire_request_listeners`` /
        # ``invoke_on_response`` / ``_fire_response_listeners`` were typed for
        # chat ``Model``; image models are structurally compatible (they only
        # read ``.id`` / ``.provider_id``), but mypy can't see that — so the
        # listener calls below carry a localized ``arg-type`` ignore.
        on_payload: OnPayloadCallback | None = options.on_payload if options else None
        payload = await invoke_on_payload(on_payload, payload, model)  # type: ignore[arg-type]
        if self._request_listeners:
            await _fire_request_listeners(self._request_listeners, payload, model)  # type: ignore[arg-type]

        body: dict | None = None
        exc: BaseException | None = None
        try:
            sdk_resp = await self._run_with_signal(
                sdk_call,
                payload,
                options.signal if options else None,
            )
            body = self._resp_to_dict(sdk_resp)

            if options and options.on_response:
                await invoke_on_response(
                    options.on_response,
                    ProviderResponse(status=200),
                    model,  # type: ignore[arg-type]
                )

            return self._parse_response(sdk_resp, model, context, cap)

        except asyncio.CancelledError as cancel:
            # Signal-triggered abort: return as AssistantImages, do not re-raise.
            # The CancelledError is recorded so subscribe_response observers
            # can distinguish abort from successful completion.
            exc = cancel
            return AssistantImages(
                api=model.api,
                provider_id=model.provider_id,
                model=model.id,
                output=[],
                stop_reason="aborted",
            )
        except Exception as raw:  # noqa: BLE001
            exc = raw
            classify_and_raise(raw, model=model)
        finally:
            if self._response_listeners:
                await _fire_response_listeners(
                    self._response_listeners,
                    body,
                    model,  # type: ignore[arg-type]
                    exc,
                )

    # ──── internals ──────────────────────────────────────────────
    async def _run_with_signal(
        self,
        sdk_call: Any,
        payload: dict[str, Any],
        signal: asyncio.Event | None,
    ) -> Any:
        """Run ``sdk_call(**payload)`` while listening to ``signal``.

        If ``signal`` fires first, the SDK task is cancelled and
        ``asyncio.CancelledError`` propagates upward.
        """
        if signal is None:
            return await sdk_call(**payload)

        sdk_task = asyncio.ensure_future(sdk_call(**payload))
        signal_task = asyncio.ensure_future(signal.wait())
        try:
            done, _ = await asyncio.wait(
                {sdk_task, signal_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if signal_task in done and not sdk_task.done():
                sdk_task.cancel()
                raise asyncio.CancelledError
            signal_task.cancel()
            return sdk_task.result()
        finally:
            if not signal_task.done():
                signal_task.cancel()
            if not sdk_task.done():
                sdk_task.cancel()

    def _parse_response(
        self,
        resp: Any,
        model: ImagesModel,
        context: ImagesContext,
        cap: ImagesCapabilityDescriptor,
    ) -> AssistantImages:
        # Resolve output_format the same way _build_payload does, so the
        # returned ImageContent.media_type matches what the user requested.
        # When the capability dropped output_format from the wire payload
        # (output_format_field=None), fall back to png labelling.
        out_format = (
            context.output_format
            if context.output_format is not None
            else model.default_output_format
        ) or "png"
        media_type = _OUTPUT_FORMAT_MEDIA_TYPE.get(out_format, "image/png")
        data = getattr(resp, "data", None) or []
        images: list[ImageContent | TextContent] = [
            ImageContent(source=item.b64_json, media_type=media_type)
            for item in data
            if getattr(item, "b64_json", None)
        ]
        if not images:
            # Empty response: still a "stop" — but no images. Surface via empty output.
            return AssistantImages(
                api=model.api,
                provider_id=model.provider_id,
                model=model.id,
                output=[],
                stop_reason="stop",
            )
        return AssistantImages(
            api=model.api,
            provider_id=model.provider_id,
            model=model.id,
            output=images,
            stop_reason="stop",
        )

    @staticmethod
    def _resp_to_dict(resp: Any) -> dict[str, Any]:
        if hasattr(resp, "model_dump"):
            try:
                return resp.model_dump()
            except Exception:  # noqa: BLE001
                pass
        return {"data": getattr(resp, "data", [])}

    @staticmethod
    def _to_file(img: ImageContent) -> io.BytesIO:
        ext = _MEDIA_TYPE_EXT.get(img.media_type, "png")
        buf = io.BytesIO(base64.b64decode(img.source))
        buf.name = f"source.{ext}"
        return buf
