import asyncio
import base64
from types import SimpleNamespace

import pytest

from cubepi.errors import (
    ProviderAuthFailed,
    ProviderUnavailable,
    RateLimited,
)
from cubepi.providers.base import ImageContent
from cubepi.providers.images.capability import (
    ImagesCapabilityDescriptor,
    SizeSpec,
)
from cubepi.providers.images.openai_images import OpenAIImagesProvider
from cubepi.providers.images.types import (
    ImagesContext,
    ImagesOptions,
)


class _StatusErr(Exception):
    def __init__(self, msg: str, status: int) -> None:
        super().__init__(msg)
        self.status_code = status


class _FakeImages:
    def __init__(self, exc: Exception | None = None, sleep: float = 0.0):
        self.generate_kwargs: dict | None = None
        self.edit_kwargs: dict | None = None
        self._exc = exc
        self._sleep = sleep

    async def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        if self._sleep:
            await asyncio.sleep(self._sleep)
        if self._exc:
            raise self._exc
        return SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(b"GEN").decode())]
        )

    async def edit(self, **kwargs):
        self.edit_kwargs = kwargs
        return SimpleNamespace(
            data=[SimpleNamespace(b64_json=base64.b64encode(b"EDIT").decode())]
        )


class _FakeClient:
    def __init__(self, exc: Exception | None = None, sleep: float = 0.0):
        self.images = _FakeImages(exc=exc, sleep=sleep)


def _provider(
    *,
    capability: ImagesCapabilityDescriptor | None = None,
    exc: Exception | None = None,
    sleep: float = 0.0,
) -> OpenAIImagesProvider:
    p = OpenAIImagesProvider(
        provider_id="openai", api_key="sk-test", capability=capability
    )
    p._client = _FakeClient(exc=exc, sleep=sleep)
    return p


@pytest.mark.asyncio
async def test_provider_id_propagated_through_factory():
    p = _provider()
    model = p.model("gpt-image-1", api="openai-images")
    assert model.provider_id == "openai"


@pytest.mark.asyncio
async def test_happy_path_writes_canonical_openai_fields():
    p = _provider()
    model = p.model("gpt-image-1", api="openai-images")
    out = await p.generate_images(
        model,
        ImagesContext(prompt="A cat", size="1024x1024", n=2, quality="high"),
    )
    kw = p._client.images.generate_kwargs
    assert kw["model"] == "gpt-image-1"
    assert kw["prompt"] == "A cat"
    assert kw["size"] == "1024x1024"
    assert kw["n"] == 2
    assert kw["quality"] == "high"
    assert kw["response_format"] == "b64_json"
    assert out.stop_reason == "stop"
    assert out.output[0].type == "image"


@pytest.mark.asyncio
async def test_doubao_extra_payload_injection():
    p = _provider(
        capability=ImagesCapabilityDescriptor(
            supports_seed=True,
            extra_payload={"watermark": False},
        ),
    )
    model = p.model("doubao-seedream-4-5-251128")
    await p.generate_images(model, ImagesContext(prompt="x", seed=42))
    kw = p._client.images.generate_kwargs
    assert kw["watermark"] is False
    assert kw["seed"] == 42


@pytest.mark.asyncio
async def test_siliconflow_field_remap():
    p = _provider(
        capability=ImagesCapabilityDescriptor(
            size_spec=SizeSpec(kind="image_size_string"),
            count_field="batch_size",
            output_format_field=None,
        ),
    )
    model = p.model("Kwai-Kolors/Kolors")
    await p.generate_images(
        model,
        ImagesContext(prompt="x", size="1024x1024", n=2, output_format="png"),
    )
    kw = p._client.images.generate_kwargs
    assert kw["image_size"] == "1024x1024"
    assert kw["batch_size"] == 2
    assert "n" not in kw
    assert "output_format" not in kw  # dropped because output_format_field=None


@pytest.mark.asyncio
async def test_edit_path_when_input_images_provided():
    p = _provider()
    model = p.model("gpt-image-1")
    ctx = ImagesContext(
        prompt="make blue",
        input_images=[
            ImageContent(
                source=base64.b64encode(b"SRC").decode(), media_type="image/png"
            ),
        ],
    )
    await p.generate_images(model, ctx)
    assert p._client.images.edit_kwargs is not None
    assert p._client.images.generate_kwargs is None


@pytest.mark.asyncio
async def test_edit_path_disabled_by_capability():
    p = _provider(
        capability=ImagesCapabilityDescriptor(supports_edit=False),
    )
    model = p.model("gpt-image-1")
    ctx = ImagesContext(
        prompt="x",
        input_images=[
            ImageContent(
                source=base64.b64encode(b"SRC").decode(), media_type="image/png"
            ),
        ],
    )
    await p.generate_images(model, ctx)
    # Falls back to generate path even with input_images
    assert p._client.images.generate_kwargs is not None
    assert p._client.images.edit_kwargs is None


@pytest.mark.asyncio
async def test_rate_limit_raises_typed_error():
    p = _provider(exc=_StatusErr("limit", 429))
    model = p.model("gpt-image-1")
    with pytest.raises(RateLimited):
        await p.generate_images(model, ImagesContext(prompt="x"))


@pytest.mark.asyncio
async def test_auth_failure_raises_typed_error():
    p = _provider(exc=_StatusErr("nope", 401))
    model = p.model("gpt-image-1")
    with pytest.raises(ProviderAuthFailed):
        await p.generate_images(model, ImagesContext(prompt="x"))


@pytest.mark.asyncio
async def test_unavailable_raises_typed_error():
    p = _provider(exc=_StatusErr("down", 503))
    model = p.model("gpt-image-1")
    with pytest.raises(ProviderUnavailable):
        await p.generate_images(model, ImagesContext(prompt="x"))


@pytest.mark.asyncio
async def test_signal_cancels_and_returns_aborted():
    """Setting the signal mid-call returns AssistantImages(stop_reason='aborted')."""
    p = _provider(sleep=0.5)  # SDK call sleeps long enough to cancel
    model = p.model("gpt-image-1")
    signal = asyncio.Event()

    async def trigger():
        await asyncio.sleep(0.05)
        signal.set()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(trigger())
        task = tg.create_task(
            p.generate_images(
                model, ImagesContext(prompt="x"), options=ImagesOptions(signal=signal)
            )
        )

    result = task.result()
    assert result.stop_reason == "aborted"
    assert result.output == []


@pytest.mark.asyncio
async def test_per_call_on_payload_mutates_outgoing():
    p = _provider()
    model = p.model("gpt-image-1")

    def mutate(payload, _model):
        payload["custom_tag"] = "tracing-id-123"
        return payload

    await p.generate_images(
        model,
        ImagesContext(prompt="x"),
        options=ImagesOptions(on_payload=mutate),
    )
    assert p._client.images.generate_kwargs["custom_tag"] == "tracing-id-123"


@pytest.mark.asyncio
async def test_subscribe_request_fires_with_final_payload():
    p = _provider()
    model = p.model("gpt-image-1")
    seen: list[dict] = []
    p.subscribe_request(lambda payload, m: seen.append(payload))
    await p.generate_images(model, ImagesContext(prompt="x", size="1024x1024"))
    assert len(seen) == 1
    assert seen[0]["model"] == "gpt-image-1"
    assert seen[0]["size"] == "1024x1024"


@pytest.mark.asyncio
async def test_subscribe_response_fires_on_success():
    p = _provider()
    model = p.model("gpt-image-1")
    seen: list[BaseException | None] = []
    p.subscribe_response(lambda body, m, exc: seen.append(exc))
    await p.generate_images(model, ImagesContext(prompt="x"))
    assert seen == [None]


@pytest.mark.asyncio
async def test_subscribe_response_fires_on_error_with_exception():
    p = _provider(exc=_StatusErr("limit", 429))
    model = p.model("gpt-image-1")
    seen: list[BaseException | None] = []
    p.subscribe_response(lambda body, m, exc: seen.append(exc))
    with pytest.raises(RateLimited):
        await p.generate_images(model, ImagesContext(prompt="x"))
    assert len(seen) == 1
    assert seen[0] is not None
    assert "limit" in str(seen[0])


def test_base_url_accepted_and_constructor_works():
    p = OpenAIImagesProvider(
        provider_id="doubao",
        api_key="sk-test",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
    assert p.provider_id == "doubao"
