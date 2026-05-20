from cubepi.providers.base import ImageContent, TextContent
from cubepi.providers.images.types import (
    AssistantImages,
    ImagesContext,
    ImagesModel,
)


def test_images_model_defaults():
    m = ImagesModel(id="gpt-image-1", provider="openai", api="openai-images")
    assert m.api == "openai-images"
    assert m.size == "auto"
    assert m.quality == "auto"


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
