from cubepi.providers.base import ImageContent, TextContent
from cubepi.providers.images.types import (
    AssistantImages,
    ImagesContext,
    ImagesModel,
)


def test_images_model_fields():
    m = ImagesModel(id="gpt-image-1", provider="openai", api="openai-images")
    assert m.id == "gpt-image-1"
    assert m.provider == "openai"
    assert m.api == "openai-images"


def test_images_model_api_defaults_to_empty():
    m = ImagesModel(id="gpt-image-1", provider="openai")
    assert m.api == ""


def test_images_model_no_size_or_quality():
    # size and quality are not fields on ImagesModel; only id/provider/api exist.
    m = ImagesModel(id="gpt-image-1", provider="openai", api="openai-images")
    assert not hasattr(m, "size")
    assert not hasattr(m, "quality")
    assert set(ImagesModel.model_fields) == {"id", "provider", "api"}


def test_images_context_input_content():
    ctx = ImagesContext(
        prompt="a cat",
        input_images=[ImageContent(source="b64", media_type="image/png")],
    )
    assert ctx.prompt == "a cat"
    assert len(ctx.input_images) == 1


def test_assistant_images_output():
    out = AssistantImages(
        api="openai-images",
        provider="openai",
        model="gpt-image-1",
        output=[ImageContent(source="b64", media_type="image/png")],
        stop_reason="stop",
    )
    assert out.stop_reason == "stop"
    assert isinstance(out.output[0], (ImageContent, TextContent))
