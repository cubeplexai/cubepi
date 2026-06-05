import pytest

from cubepi.providers.images.base import BaseImagesProvider, ImagesProvider
from cubepi.providers.images.capability import (
    ImagesCapabilityDescriptor,
    SizeSpec,
)
from cubepi.providers.images.types import (
    AssistantImages,
    ImagesContext,
    ImagesCost,
    ImagesModel,
)


class _StubBase(BaseImagesProvider):
    """Minimal concrete subclass exposing helpers + a no-op generate_images."""

    async def generate_images(self, model, context, *, options=None):
        return AssistantImages(
            api=model.api,
            provider_id=model.provider_id,
            model=model.id,
            output=[],
        )


def test_provider_id_stored_on_instance():
    p = _StubBase(provider_id="openai")
    assert p.provider_id == "openai"


def test_default_capability_is_openai_shape():
    p = _StubBase(provider_id="openai")
    # _capability_for falls back to the default descriptor when no override matches.
    cap = p._capability_for(ImagesModel(id="gpt-image-1", provider_id="openai"))
    assert cap.size_spec.kind == "size_string"


def test_capability_override_by_model_id():
    base_cap = ImagesCapabilityDescriptor()
    flux_cap = ImagesCapabilityDescriptor(size_spec=SizeSpec(kind="aspect_ratio"))
    p = _StubBase(
        provider_id="together",
        capability=base_cap,
        model_capability_overrides={"flux-schnell": flux_cap},
    )
    assert (
        p._capability_for(
            ImagesModel(id="flux-schnell", provider_id="together")
        ).size_spec.kind
        == "aspect_ratio"
    )
    assert (
        p._capability_for(
            ImagesModel(id="flux-pro", provider_id="together")
        ).size_spec.kind
        == "size_string"
    )


def test_model_factory_propagates_provider_id():
    p = _StubBase(provider_id="doubao")
    model = p.model("doubao-seedream-4-5-251128", api="doubao-images")
    assert isinstance(model, ImagesModel)
    assert model.id == "doubao-seedream-4-5-251128"
    assert model.provider_id == "doubao"
    assert model.api == "doubao-images"
    assert model.default_size is None  # nothing passed → None


def test_model_factory_passes_defaults_through():
    p = _StubBase(provider_id="openai")
    model = p.model(
        "gpt-image-1",
        default_size="1024x1024",
        default_n=2,
        default_quality="high",
        default_output_format="png",
        cost=ImagesCost(per_image=0.04),
        max_input_images=4,
    )
    assert model.default_size == "1024x1024"
    assert model.default_n == 2
    assert model.default_quality == "high"
    assert model.default_output_format == "png"
    assert model.cost is not None
    assert model.cost.per_image == 0.04
    assert model.max_input_images == 4


def test_subscribe_request_and_detach():
    p = _StubBase(provider_id="openai")
    events: list[dict] = []

    def cb(payload, model):
        events.append(payload)

    detach = p.subscribe_request(cb)
    assert cb in p._request_listeners
    detach()
    assert cb not in p._request_listeners


def test_subscribe_response_and_detach():
    p = _StubBase(provider_id="openai")
    events: list[BaseException | None] = []

    def cb(body, model, exc):
        events.append(exc)

    detach = p.subscribe_response(cb)
    assert cb in p._response_listeners
    detach()
    assert cb not in p._response_listeners


def test_no_subscribe_chunk_method():
    # Image is one-shot: there are no chunks, so no subscribe_chunk.
    p = _StubBase(provider_id="openai")
    assert not hasattr(p, "subscribe_chunk")
    assert not hasattr(p, "_chunk_listeners")


def test_base_generate_images_raises_not_implemented():
    base = BaseImagesProvider(provider_id="x")
    with pytest.raises(NotImplementedError):
        # Run as coroutine; we expect NotImplementedError at call time.
        import asyncio

        asyncio.run(
            base.generate_images(
                ImagesModel(id="x", provider_id="x"),
                ImagesContext(prompt="x"),
            )
        )


def test_protocol_runtime_check_accepts_stub():
    p = _StubBase(provider_id="openai")
    assert isinstance(p, ImagesProvider)
