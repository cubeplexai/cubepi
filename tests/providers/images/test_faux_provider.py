import base64

import pytest

from cubepi.errors import ProviderError, RateLimited
from cubepi.providers.images.faux import FauxImagesProvider
from cubepi.providers.images.types import ImagesContext


def _png_b64() -> str:
    return base64.b64encode(b"\x89PNG-stub").decode()


@pytest.mark.asyncio
async def test_happy_path_returns_image():
    p = FauxImagesProvider(png_b64=_png_b64())
    model = p.model("faux-1", api="faux-images")
    out = await p.generate_images(model, ImagesContext(prompt="x"))
    assert out.stop_reason == "stop"
    assert len(out.output) == 1
    assert out.output[0].type == "image"
    assert out.output[0].media_type == "image/png"
    assert out.provider_id == "faux"


@pytest.mark.asyncio
async def test_custom_provider_id_propagates():
    p = FauxImagesProvider(provider_id="custom-faux", png_b64=_png_b64())
    model = p.model("faux-1")
    assert model.provider_id == "custom-faux"
    out = await p.generate_images(model, ImagesContext(prompt="x"))
    assert out.provider_id == "custom-faux"


@pytest.mark.asyncio
async def test_raise_on_call_injects_typed_error():
    p = FauxImagesProvider(png_b64=_png_b64(), raise_on_call=RateLimited)
    model = p.model("faux-1")
    with pytest.raises(RateLimited) as exc:
        await p.generate_images(model, ImagesContext(prompt="x"))
    assert "faux-1" in str(exc.value)


@pytest.mark.asyncio
async def test_raise_on_call_can_be_base_provider_error():
    p = FauxImagesProvider(png_b64=_png_b64(), raise_on_call=ProviderError)
    model = p.model("faux-1")
    with pytest.raises(ProviderError):
        await p.generate_images(model, ImagesContext(prompt="x"))


def test_inherits_from_base_images_provider():
    from cubepi.providers.images.base import BaseImagesProvider

    p = FauxImagesProvider(png_b64=_png_b64())
    assert isinstance(p, BaseImagesProvider)
    # listener registry is inherited
    assert hasattr(p, "subscribe_request")
    assert hasattr(p, "subscribe_response")
    assert not hasattr(p, "subscribe_chunk")


@pytest.mark.asyncio
async def test_pre_set_signal_returns_aborted_matching_real_provider():
    """Faux must short-circuit a pre-set signal the same way the real
    OpenAIImagesProvider does — otherwise tests that exercise abort
    behavior with Faux would silently pass while the production code
    path actually aborts."""
    import asyncio

    from cubepi.providers.images.types import ImagesOptions

    p = FauxImagesProvider(png_b64=_png_b64())
    model = p.model("faux-1")
    signal = asyncio.Event()
    signal.set()  # pre-set

    out = await p.generate_images(
        model,
        ImagesContext(prompt="x"),
        options=ImagesOptions(signal=signal),
    )
    assert out.stop_reason == "aborted"
    assert out.output == []


@pytest.mark.asyncio
async def test_unset_signal_still_returns_happy_path_image():
    """Sanity guard: signal present but never set must NOT abort —
    Faux returns the deterministic image as usual."""
    import asyncio

    from cubepi.providers.images.types import ImagesOptions

    p = FauxImagesProvider(png_b64=_png_b64())
    model = p.model("faux-1")
    signal = asyncio.Event()  # never set
    out = await p.generate_images(
        model,
        ImagesContext(prompt="x"),
        options=ImagesOptions(signal=signal),
    )
    assert out.stop_reason == "stop"
    assert len(out.output) == 1
    assert out.output[0].media_type == "image/png"


@pytest.mark.asyncio
async def test_subscribe_request_fires_on_happy_path():
    """Faux fires subscribe_request with a synthetic payload so observability
    tests work the same way against Faux and OpenAIImagesProvider."""
    p = FauxImagesProvider(png_b64=_png_b64())
    model = p.model("faux-1")
    seen: list = []
    p.subscribe_request(lambda payload, m: seen.append(payload))

    await p.generate_images(model, ImagesContext(prompt="hello"))
    assert len(seen) == 1
    assert seen[0]["model"] == "faux-1"
    assert seen[0]["prompt"] == "hello"


@pytest.mark.asyncio
async def test_subscribe_response_fires_on_happy_path_with_no_exc():
    p = FauxImagesProvider(png_b64=_png_b64())
    model = p.model("faux-1")
    seen: list = []
    p.subscribe_response(lambda body, m, exc: seen.append((body, exc)))

    await p.generate_images(model, ImagesContext(prompt="x"))
    assert len(seen) == 1
    body, exc = seen[0]
    assert body is not None and "data" in body
    assert exc is None


@pytest.mark.asyncio
async def test_subscribe_response_fires_on_raise_on_call_with_typed_exc():
    """When raise_on_call injects RateLimited, the observer must see
    the same exception class the caller catches."""
    p = FauxImagesProvider(png_b64=_png_b64(), raise_on_call=RateLimited)
    model = p.model("faux-1")
    seen: list = []
    p.subscribe_response(lambda body, m, exc: seen.append((body, exc)))

    with pytest.raises(RateLimited):
        await p.generate_images(model, ImagesContext(prompt="x"))

    assert len(seen) == 1
    body, exc = seen[0]
    assert body is None
    assert isinstance(exc, RateLimited)


@pytest.mark.asyncio
async def test_subscribe_response_fires_on_pre_set_abort_with_images_aborted():
    """Faux abort path must surface ImagesAborted to response observers
    the same way OpenAIImagesProvider does."""
    import asyncio

    from cubepi.providers.images import ImagesAborted
    from cubepi.providers.images.types import ImagesOptions

    p = FauxImagesProvider(png_b64=_png_b64())
    model = p.model("faux-1")
    signal = asyncio.Event()
    signal.set()

    seen: list = []
    p.subscribe_response(lambda body, m, exc: seen.append((body, exc)))

    await p.generate_images(
        model,
        ImagesContext(prompt="x"),
        options=ImagesOptions(signal=signal),
    )
    assert len(seen) == 1
    body, exc = seen[0]
    assert body is None
    assert isinstance(exc, ImagesAborted)


@pytest.mark.asyncio
async def test_pre_set_abort_skips_request_listener_in_faux():
    """Pre-set signal aborts before subscribe_request fires — same
    contract as OpenAIImagesProvider."""
    import asyncio

    from cubepi.providers.images.types import ImagesOptions

    p = FauxImagesProvider(png_b64=_png_b64())
    model = p.model("faux-1")
    signal = asyncio.Event()
    signal.set()

    seen_requests: list = []
    p.subscribe_request(lambda payload, m: seen_requests.append(payload))

    await p.generate_images(
        model,
        ImagesContext(prompt="x"),
        options=ImagesOptions(signal=signal),
    )
    assert seen_requests == []
