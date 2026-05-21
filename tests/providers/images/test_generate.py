import base64

import pytest

from cubepi.providers.images import create_images_provider
from cubepi.providers.images.faux import FauxImagesProvider
from cubepi.providers.images.openai_images import OpenAIImagesProvider
from cubepi.providers.images.types import ImagesContext, ImagesModel


@pytest.mark.asyncio
async def test_faux_provider_direct():
    """FauxImagesProvider can be instantiated directly and returns an image."""
    provider = FauxImagesProvider(png_b64=base64.b64encode(b"\x89PNG-stub").decode())
    model = ImagesModel(id="faux-image", provider="faux", api="faux-images")
    out = await provider.generate_images(model, ImagesContext(prompt="a cat"))
    assert out.stop_reason == "stop"
    assert out.output and out.output[0].type == "image"


@pytest.mark.asyncio
async def test_create_openai_provider_returns_instance():
    """create_images_provider('openai-images', ...) returns an OpenAIImagesProvider."""
    # Importing cubepi.providers.images registers the class; just verify the factory works.
    p = create_images_provider("openai-images", api_key="sk-test")
    assert isinstance(p, OpenAIImagesProvider)
    assert p.api == "openai-images"


def test_create_unknown_api_raises():
    with pytest.raises(ValueError):
        create_images_provider("missing-images")
