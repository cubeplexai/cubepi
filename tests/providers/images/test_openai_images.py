import base64
from types import SimpleNamespace

import pytest

from cubepi.providers.base import ImageContent
from cubepi.providers.images.openai_images import OpenAIImagesProvider
from cubepi.providers.images.types import ImagesContext, ImagesModel


class _FakeImages:
    def __init__(self):
        self.generate_kwargs = None
        self.edit_kwargs = None

    async def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(b"GEN").decode())]
        )

    async def edit(self, **kwargs):
        self.edit_kwargs = kwargs
        return SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(b"EDIT").decode())]
        )


class _FakeClient:
    def __init__(self):
        self.images = _FakeImages()


def _provider_with_fake():
    p = OpenAIImagesProvider(api_key="sk-test")
    p._client = _FakeClient()
    return p


@pytest.mark.asyncio
async def test_generate_text_to_image():
    p = _provider_with_fake()
    model = ImagesModel(
        id="gpt-image-1",
        provider="openai",
        api="openai-images",
        size="1024x1024",
        quality="high",
    )
    out = await p.generate_images(model, ImagesContext(prompt="a cat"))
    assert out.stop_reason == "stop"
    assert out.output[0].type == "image"
    assert p._client.images.generate_kwargs["model"] == "gpt-image-1"
    assert p._client.images.generate_kwargs["size"] == "1024x1024"
    assert p._client.images.generate_kwargs["quality"] == "high"
    assert p._client.images.edit_kwargs is None


@pytest.mark.asyncio
async def test_auto_size_quality_omitted():
    p = _provider_with_fake()
    model = ImagesModel(id="gpt-image-1", provider="openai", api="openai-images")
    await p.generate_images(model, ImagesContext(prompt="x"))
    assert "size" not in p._client.images.generate_kwargs
    assert "quality" not in p._client.images.generate_kwargs


@pytest.mark.asyncio
async def test_edit_branch_uses_input_images():
    p = _provider_with_fake()
    model = ImagesModel(id="gpt-image-1", provider="openai", api="openai-images")
    ctx = ImagesContext(
        prompt="make it blue",
        input_images=[
            ImageContent(
                source=base64.b64encode(b"SRC").decode(), media_type="image/png"
            )
        ],
    )
    out = await p.generate_images(model, ctx)
    assert p._client.images.edit_kwargs is not None
    assert p._client.images.generate_kwargs is None
    assert out.output[0].type == "image"


@pytest.mark.asyncio
async def test_empty_data_returns_error():
    p = _provider_with_fake()

    async def _empty(**kwargs):
        return SimpleNamespace(data=[])

    p._client.images.generate = _empty
    model = ImagesModel(id="gpt-image-1", provider="openai", api="openai-images")
    out = await p.generate_images(model, ImagesContext(prompt="x"))
    assert out.stop_reason == "error"
    assert out.error_message


@pytest.mark.asyncio
async def test_sdk_exception_returns_error():
    p = _provider_with_fake()

    async def _boom(**kwargs):
        raise RuntimeError("rate limited")

    p._client.images.generate = _boom
    model = ImagesModel(id="gpt-image-1", provider="openai", api="openai-images")
    out = await p.generate_images(model, ImagesContext(prompt="x"))
    assert out.stop_reason == "error"
    assert "rate limited" in out.error_message


@pytest.mark.parametrize(
    "media_type,expected_name",
    [
        ("image/png", "source.png"),
        ("image/jpeg", "source.jpg"),
        ("image/webp", "source.webp"),
        ("image/gif", "source.png"),  # unknown → safe png default
    ],
)
def test_to_file_preserves_input_format(media_type, expected_name):
    img = ImageContent(source=base64.b64encode(b"SRC").decode(), media_type=media_type)
    f = OpenAIImagesProvider._to_file(img)
    assert f.name == expected_name


def test_register_openai_images_registers_provider():
    from cubepi.providers.images.openai_images import register_openai_images
    from cubepi.providers.images.registry import get_images_provider

    register_openai_images(api_key="sk-test")
    assert get_images_provider("openai-images") is not None


def test_base_url_is_accepted():
    p = OpenAIImagesProvider(api_key="sk-test", base_url="https://example.test/v1")
    assert p.api == "openai-images"


@pytest.mark.asyncio
async def test_options_are_forwarded_to_request():
    p = _provider_with_fake()
    model = ImagesModel(id="gpt-image-1", provider="openai", api="openai-images")
    await p.generate_images(
        model,
        ImagesContext(prompt="x"),
        options={"n": 2, "output_format": "jpeg", "background": "transparent"},
    )
    kw = p._client.images.generate_kwargs
    assert kw["n"] == 2
    assert kw["output_format"] == "jpeg"
    assert kw["background"] == "transparent"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output_format,expected_media_type",
    [
        ("jpeg", "image/jpeg"),
        ("webp", "image/webp"),
        ("png", "image/png"),
    ],
)
async def test_output_media_type_follows_output_format(
    output_format, expected_media_type
):
    p = _provider_with_fake()
    model = ImagesModel(id="gpt-image-1", provider="openai", api="openai-images")
    out = await p.generate_images(
        model, ImagesContext(prompt="x"), options={"output_format": output_format}
    )
    assert out.output[0].media_type == expected_media_type


@pytest.mark.asyncio
async def test_output_media_type_defaults_to_png():
    p = _provider_with_fake()
    model = ImagesModel(id="gpt-image-1", provider="openai", api="openai-images")
    out = await p.generate_images(model, ImagesContext(prompt="x"))
    assert out.output[0].media_type == "image/png"
