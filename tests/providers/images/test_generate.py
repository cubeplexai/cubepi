import base64

import pytest

from cubepi.providers.images import generate_images
from cubepi.providers.images.faux import register_faux_images
from cubepi.providers.images.types import ImagesContext, ImagesModel


@pytest.mark.asyncio
async def test_generate_images_via_faux():
    register_faux_images(png_b64=base64.b64encode(b"\x89PNG-stub").decode())
    model = ImagesModel(id="faux-image", provider="faux", api="faux-images")
    out = await generate_images(model, ImagesContext(prompt="a cat"))
    assert out.stop_reason == "stop"
    assert out.output and out.output[0].type == "image"


@pytest.mark.asyncio
async def test_generate_images_unknown_api_raises():
    model = ImagesModel(id="x", provider="x", api="missing-images")
    with pytest.raises(ValueError):
        await generate_images(model, ImagesContext(prompt="x"))
