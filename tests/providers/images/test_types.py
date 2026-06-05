import asyncio
from typing import get_args

import pytest
from pydantic import ValidationError

from cubepi.providers.base import ImageContent
from cubepi.providers.images.types import (
    AssistantImages,
    ImagesContext,
    ImagesCost,
    ImagesModel,
    ImagesOptions,
)


def test_images_model_required_field_id():
    with pytest.raises(ValidationError):
        ImagesModel()  # type: ignore[call-arg]


def test_images_model_provider_id_renamed():
    m = ImagesModel(id="gpt-image-1", provider_id="openai", api="openai-images")
    assert m.provider_id == "openai"
    assert m.api == "openai-images"
    assert not hasattr(m, "provider"), "old `provider` field must be gone"


def test_images_model_default_fields():
    m = ImagesModel(
        id="gpt-image-1",
        provider_id="openai",
        default_size="1024x1024",
        default_n=2,
        default_quality="high",
        default_output_format="png",
    )
    assert m.default_size == "1024x1024"
    assert m.default_n == 2
    assert m.default_quality == "high"
    assert m.default_output_format == "png"


def test_images_model_defaults_are_optional_none():
    m = ImagesModel(id="gpt-image-1", provider_id="openai")
    assert m.default_size is None
    assert m.default_n is None
    assert m.default_quality is None
    assert m.default_output_format is None
    assert m.cost is None
    assert m.max_input_images is None


def test_images_cost_fields():
    cost = ImagesCost(per_image=0.04, per_megapixel=0.0)
    assert cost.per_image == 0.04
    assert cost.per_megapixel == 0.0


def test_images_context_typed_fields():
    ctx = ImagesContext(
        prompt="A robot",
        size="1024x1024",
        n=2,
        quality="high",
        output_format="png",
        seed=42,
        negative_prompt="blurry",
        steps=20,
        guidance=7.5,
        extra={"watermark": False},
    )
    assert ctx.prompt == "A robot"
    assert ctx.size == "1024x1024"
    assert ctx.n == 2
    assert ctx.quality == "high"
    assert ctx.output_format == "png"
    assert ctx.seed == 42
    assert ctx.negative_prompt == "blurry"
    assert ctx.steps == 20
    assert ctx.guidance == 7.5
    assert ctx.extra == {"watermark": False}


def test_images_context_input_images_accepts_image_content():
    ctx = ImagesContext(
        prompt="edit",
        input_images=[ImageContent(source="b64", media_type="image/png")],
    )
    assert len(ctx.input_images) == 1


def test_images_context_quality_literal_rejects_unknown():
    with pytest.raises(ValidationError):
        ImagesContext(prompt="x", quality="ultra")  # type: ignore[arg-type]


def test_images_options_default_construction():
    opts = ImagesOptions()
    assert opts.signal is None
    assert opts.on_payload is None
    assert opts.on_response is None


def test_images_options_accepts_event_and_callbacks():
    ev = asyncio.Event()
    opts = ImagesOptions(
        signal=ev, on_payload=lambda p, m: None, on_response=lambda r, m: None
    )
    assert opts.signal is ev
    assert opts.on_payload is not None
    assert opts.on_response is not None


def test_assistant_images_stop_reason_literal():
    allowed = set(get_args(AssistantImages.model_fields["stop_reason"].annotation))
    assert allowed == {"stop", "aborted"}, "error stop_reason is removed in 0.7"


def test_assistant_images_no_error_message_field():
    assert "error_message" not in AssistantImages.model_fields


def test_assistant_images_provider_id_renamed():
    out = AssistantImages(
        api="openai-images",
        provider_id="openai",
        model="gpt-image-1",
        output=[],
    )
    assert out.provider_id == "openai"
    assert not hasattr(out, "provider"), "old `provider` field must be gone"


def test_top_level_re_exports():
    """The image types are reachable via `from cubepi import ...`."""
    import cubepi

    assert (
        cubepi.ImagesModel
        is __import__("cubepi.providers.images", fromlist=["ImagesModel"]).ImagesModel
    )
    for name in (
        "AssistantImages",
        "BaseImagesProvider",
        "ImagesCapabilityDescriptor",
        "ImagesContext",
        "ImagesCost",
        "ImagesModel",
        "ImagesOptions",
        "ImagesProvider",
        "SizeSpec",
    ):
        assert hasattr(cubepi, name), f"cubepi.{name} not exported"
