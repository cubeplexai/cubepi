import pytest

from cubepi.providers.images.registry import (
    ImagesProvider,
    create_images_provider,
    register_images_provider_class,
)
from cubepi.providers.images.types import AssistantImages


class _StubProvider:
    api = "stub-images"

    def __init__(self, *, token: str = "") -> None:
        self.token = token

    async def generate_images(self, model, context, options=None):
        return AssistantImages(
            api=model.api, provider=model.provider, model=model.id, output=[]
        )


def test_register_and_create():
    register_images_provider_class("stub-images", _StubProvider)  # type: ignore[arg-type]
    p = create_images_provider("stub-images")
    assert isinstance(p, ImagesProvider)
    assert p.api == "stub-images"


def test_create_passes_kwargs():
    register_images_provider_class("stub-images", _StubProvider)  # type: ignore[arg-type]
    p = create_images_provider("stub-images", token="abc")
    assert isinstance(p, _StubProvider)
    assert p.token == "abc"


def test_create_unknown_api_raises():
    with pytest.raises(
        ValueError, match="No images provider registered for api: nope-images"
    ):
        create_images_provider("nope-images")
