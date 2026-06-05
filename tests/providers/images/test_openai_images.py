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
    # response_format is NOT sent by default — OpenAI's GPT image models
    # reject the field and return base64 anyway.
    assert "response_format" not in kw
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


@pytest.mark.asyncio
async def test_subscribe_response_fires_on_abort_with_cancelled_error():
    """Abort path must surface CancelledError to response listeners so tracing
    can distinguish abort from successful completion (spec §6.5)."""
    p = _provider(sleep=0.5)
    model = p.model("gpt-image-1")
    signal = asyncio.Event()
    seen: list[tuple[dict | None, BaseException | None]] = []
    p.subscribe_response(lambda body, m, exc: seen.append((body, exc)))

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
    assert len(seen) == 1
    body, exc = seen[0]
    assert body is None
    assert isinstance(exc, asyncio.CancelledError)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output_format,expected_media_type",
    [
        ("png", "image/png"),
        ("jpeg", "image/jpeg"),
        ("webp", "image/webp"),
    ],
)
async def test_media_type_follows_requested_output_format(
    output_format, expected_media_type
):
    """media_type on returned ImageContent must reflect the requested
    output_format, not a hardcoded default (final-review fix)."""
    p = _provider()
    model = p.model("gpt-image-1")
    out = await p.generate_images(
        model, ImagesContext(prompt="x", output_format=output_format)
    )
    assert out.output[0].media_type == expected_media_type


@pytest.mark.asyncio
async def test_media_type_uses_model_default_output_format():
    """When ctx.output_format is None, fall back to model.default_output_format."""
    p = _provider()
    model = p.model("gpt-image-1", default_output_format="webp")
    out = await p.generate_images(model, ImagesContext(prompt="x"))
    assert out.output[0].media_type == "image/webp"


@pytest.mark.asyncio
async def test_external_cancel_propagates_not_swallowed():
    """task.cancel() on the caller must propagate as CancelledError, not
    be silently converted into AssistantImages(stop_reason='aborted').
    Only ImagesOptions.signal triggers the aborted path."""
    p = _provider(sleep=0.5)
    model = p.model("gpt-image-1")

    task = asyncio.create_task(p.generate_images(model, ImagesContext(prompt="x")))
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_wait_for_timeout_propagates_cancellation():
    """asyncio.wait_for cancellation on the caller must propagate as a
    timeout, not be swallowed into stop_reason='aborted'."""
    p = _provider(sleep=0.5)
    model = p.model("gpt-image-1")

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            p.generate_images(model, ImagesContext(prompt="x")), timeout=0.05
        )


@pytest.mark.asyncio
async def test_per_call_on_response_observer_fires():
    """ImagesOptions.on_response observer is invoked with ProviderResponse."""
    p = _provider()
    model = p.model("gpt-image-1")
    seen: list = []

    async def on_response(resp, m):
        seen.append((resp.status, m.id))

    await p.generate_images(
        model,
        ImagesContext(prompt="x"),
        options=ImagesOptions(on_response=on_response),
    )
    assert seen == [(200, "gpt-image-1")]


@pytest.mark.asyncio
async def test_signal_present_but_unset_lets_sdk_complete():
    """When ImagesOptions.signal is provided but never fires, the SDK
    completion branch in _run_with_signal must clean up the signal task
    and return the result normally."""
    p = _provider()
    model = p.model("gpt-image-1")
    signal = asyncio.Event()  # never set
    out = await p.generate_images(
        model, ImagesContext(prompt="x"), options=ImagesOptions(signal=signal)
    )
    assert out.stop_reason == "stop"
    assert out.output[0].type == "image"


@pytest.mark.asyncio
async def test_empty_data_returns_stop_with_empty_output():
    """SDK returns data items without b64_json — provider returns
    AssistantImages(stop_reason='stop', output=[]) rather than erroring."""
    p = _provider()

    async def _empty_data(**kwargs):
        return SimpleNamespace(data=[SimpleNamespace(b64_json=None)])

    p._client.images.generate = _empty_data
    model = p.model("gpt-image-1")
    out = await p.generate_images(model, ImagesContext(prompt="x"))
    assert out.stop_reason == "stop"
    assert out.output == []


@pytest.mark.asyncio
async def test_resp_to_dict_falls_back_when_model_dump_raises():
    """When ``model_dump()`` raises (pydantic edge cases), the body
    recorded for subscribe_response falls back to the SimpleNamespace
    ``{"data": ...}`` shape rather than failing the whole call."""
    p = _provider()
    seen_bodies: list = []
    p.subscribe_response(lambda body, m, exc: seen_bodies.append(body))

    class _BadDumpResp:
        data = [SimpleNamespace(b64_json=base64.b64encode(b"GEN").decode())]

        def model_dump(self):
            raise RuntimeError("dump exploded")

    async def _resp(**kwargs):
        return _BadDumpResp()

    p._client.images.generate = _resp
    model = p.model("gpt-image-1")
    out = await p.generate_images(model, ImagesContext(prompt="x"))
    assert out.stop_reason == "stop"  # call still succeeded
    assert len(seen_bodies) == 1
    # Fallback returns {"data": <list>} from the SimpleNamespace path
    assert "data" in seen_bodies[0]
    assert isinstance(seen_bodies[0]["data"], list)


@pytest.mark.asyncio
async def test_resp_to_dict_uses_model_dump_when_available():
    """When the SDK response exposes model_dump (pydantic-style), the
    response body recorded for subscribe_response is the dumped dict —
    not the SimpleNamespace fallback."""
    p = _provider()
    seen_bodies: list = []
    p.subscribe_response(lambda body, m, exc: seen_bodies.append(body))

    class _PydanticLikeResp:
        data = [SimpleNamespace(b64_json=base64.b64encode(b"GEN").decode())]

        def model_dump(self):
            return {"data": [{"b64_json": "DUMPED"}], "id": "resp_123"}

    async def _resp(**kwargs):
        return _PydanticLikeResp()

    p._client.images.generate = _resp
    model = p.model("gpt-image-1")
    await p.generate_images(model, ImagesContext(prompt="x"))
    assert len(seen_bodies) == 1
    assert seen_bodies[0] == {"data": [{"b64_json": "DUMPED"}], "id": "resp_123"}


@pytest.mark.asyncio
async def test_url_response_items_surface_as_imagecontent():
    """When a capability requests response_format='url', SDK items expose
    `url` instead of `b64_json`. Parse path must surface those rather than
    silently drop them (P2 codex finding)."""
    p = _provider(
        capability=ImagesCapabilityDescriptor(
            response_format_field="response_format",
            response_format_value="url",
        ),
    )

    async def _url_resp(**kwargs):
        return SimpleNamespace(
            data=[
                SimpleNamespace(b64_json=None, url="https://cdn.example/img1.png"),
                SimpleNamespace(b64_json=None, url="https://cdn.example/img2.png"),
            ]
        )

    p._client.images.generate = _url_resp
    model = p.model("gpt-image-1")
    out = await p.generate_images(model, ImagesContext(prompt="x"))
    assert out.stop_reason == "stop"
    assert len(out.output) == 2
    assert out.output[0].source == "https://cdn.example/img1.png"
    assert out.output[1].source == "https://cdn.example/img2.png"
    assert all(o.source.startswith("https://") for o in out.output)


@pytest.mark.asyncio
async def test_media_type_defaults_to_png_when_output_format_dropped():
    """When the capability sets output_format_field=None (SiliconFlow shape),
    the backend chooses its own format. The parse path must NOT trust the
    user's requested output_format for media_type labelling — it defaults
    to png instead of mislabelling backend-default bytes (P2 codex finding)."""
    p = _provider(
        capability=ImagesCapabilityDescriptor(
            size_spec=SizeSpec(kind="image_size_string"),
            count_field="batch_size",
            output_format_field=None,
        ),
    )
    model = p.model("Kwai-Kolors/Kolors")
    # User asks for webp but the backend ignores that field and returns its
    # own default — media_type should NOT lie and claim webp.
    out = await p.generate_images(
        model, ImagesContext(prompt="x", output_format="webp")
    )
    assert out.output[0].media_type == "image/png"


@pytest.mark.asyncio
async def test_media_type_still_uses_output_format_when_field_active():
    """Sanity guard for the dual-condition fix: when output_format_field IS
    active (default OpenAI shape), the media_type follows the request."""
    p = _provider()  # default descriptor → output_format_field="output_format"
    model = p.model("gpt-image-1")
    out = await p.generate_images(
        model, ImagesContext(prompt="x", output_format="webp")
    )
    assert out.output[0].media_type == "image/webp"
