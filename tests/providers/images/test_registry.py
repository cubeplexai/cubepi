from cubepi.providers.images.registry import (
    ImagesProvider,
    get_images_provider,
    register_images_provider,
)
from cubepi.providers.images.types import AssistantImages


class _Stub:
    api = "stub-images"

    async def generate_images(self, model, context, options=None):
        return AssistantImages(
            api=model.api, provider=model.provider, model=model.id, output=[]
        )


def test_register_and_get():
    register_images_provider(_Stub())
    p = get_images_provider("stub-images")
    assert p is not None
    assert isinstance(p, ImagesProvider)


def test_get_unknown_returns_none():
    assert get_images_provider("nope-images") is None
