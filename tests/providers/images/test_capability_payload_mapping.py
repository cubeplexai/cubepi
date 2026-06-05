import pytest

from cubepi.providers.images.base import BaseImagesProvider
from cubepi.providers.images.capability import (
    ImagesCapabilityDescriptor,
    SizeSpec,
)
from cubepi.providers.images.types import (
    AssistantImages,
    ImagesContext,
    ImagesModel,
)


class _Stub(BaseImagesProvider):
    async def generate_images(self, model, context, *, options=None):
        return AssistantImages(api="", provider_id="", model="", output=[])


def _model(model_id: str = "m", **defaults) -> ImagesModel:
    return ImagesModel(id=model_id, provider_id="p", **defaults)


def test_baseline_payload_has_model_and_prompt():
    p = _Stub(provider_id="p")
    payload = p._build_payload(_model(), ImagesContext(prompt="A robot"))
    assert payload["model"] == "m"
    assert payload["prompt"] == "A robot"
    # response_format is opt-in via the capability descriptor; the default
    # OpenAI-shape descriptor leaves it off so gpt-image-1 doesn't 400.
    assert "response_format" not in payload


def test_response_format_opt_in_when_field_set():
    p = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(
            response_format_field="response_format",
        ),
    )
    payload = p._build_payload(_model(), ImagesContext(prompt="x"))
    assert payload["response_format"] == "b64_json"


def test_response_format_renamed_when_field_overridden():
    p = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(
            response_format_field="resp_fmt",
            response_format_value="url",
        ),
    )
    payload = p._build_payload(_model(), ImagesContext(prompt="x"))
    assert payload["resp_fmt"] == "url"
    assert "response_format" not in payload


def test_size_spec_size_string():
    p = _Stub(provider_id="p")
    payload = p._build_payload(_model(), ImagesContext(prompt="x", size="1024x1024"))
    assert payload["size"] == "1024x1024"
    assert "image_size" not in payload
    assert "width" not in payload


def test_size_spec_image_size_string():
    p = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(
            size_spec=SizeSpec(kind="image_size_string")
        ),
    )
    payload = p._build_payload(_model(), ImagesContext(prompt="x", size="1024x1024"))
    assert payload["image_size"] == "1024x1024"
    assert "size" not in payload


def test_size_spec_width_height():
    p = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(size_spec=SizeSpec(kind="width_height")),
    )
    payload = p._build_payload(_model(), ImagesContext(prompt="x", size="1024x768"))
    assert payload["width"] == 1024
    assert payload["height"] == 768
    assert "size" not in payload


def test_size_spec_aspect_ratio():
    p = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(size_spec=SizeSpec(kind="aspect_ratio")),
    )
    payload = p._build_payload(_model(), ImagesContext(prompt="x", size="1:1"))
    assert payload["aspect_ratio"] == "1:1"


def test_size_falls_back_to_model_default():
    p = _Stub(provider_id="p")
    payload = p._build_payload(
        _model(default_size="2K"),
        ImagesContext(prompt="x"),
    )
    assert payload["size"] == "2K"


def test_ctx_size_overrides_model_default():
    p = _Stub(provider_id="p")
    payload = p._build_payload(
        _model(default_size="2K"),
        ImagesContext(prompt="x", size="1K"),
    )
    assert payload["size"] == "1K"


def test_size_omitted_when_both_none():
    p = _Stub(provider_id="p")
    payload = p._build_payload(_model(), ImagesContext(prompt="x"))
    assert "size" not in payload


def test_count_field_default_n():
    p = _Stub(provider_id="p")
    payload = p._build_payload(_model(), ImagesContext(prompt="x", n=2))
    assert payload["n"] == 2


def test_count_field_renamed_to_batch_size():
    p = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(count_field="batch_size"),
    )
    payload = p._build_payload(_model(), ImagesContext(prompt="x", n=3))
    assert payload["batch_size"] == 3
    assert "n" not in payload


def test_quality_written_when_set():
    p = _Stub(provider_id="p")
    payload = p._build_payload(_model(), ImagesContext(prompt="x", quality="high"))
    assert payload["quality"] == "high"


def test_output_format_renamed_or_dropped():
    # Default name preserved
    p1 = _Stub(provider_id="p")
    payload1 = p1._build_payload(
        _model(), ImagesContext(prompt="x", output_format="webp")
    )
    assert payload1["output_format"] == "webp"

    # output_format_field=None → silently dropped
    p2 = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(output_format_field=None),
    )
    payload2 = p2._build_payload(
        _model(), ImagesContext(prompt="x", output_format="webp")
    )
    assert "output_format" not in payload2


def test_supports_seed_gating_on():
    p = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(supports_seed=True),
    )
    payload = p._build_payload(_model(), ImagesContext(prompt="x", seed=42))
    assert payload["seed"] == 42


def test_supports_seed_gating_off_drops_silently():
    p = _Stub(provider_id="p")  # default supports_seed=False
    payload = p._build_payload(_model(), ImagesContext(prompt="x", seed=42))
    assert "seed" not in payload


def test_seed_field_rename():
    p = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(
            supports_seed=True, seed_field="rng_seed"
        ),
    )
    payload = p._build_payload(_model(), ImagesContext(prompt="x", seed=42))
    assert payload["rng_seed"] == 42


@pytest.mark.parametrize(
    "flag,field_default,ctx_field,ctx_value",
    [
        ("supports_negative_prompt", "negative_prompt", "negative_prompt", "blurry"),
        ("supports_steps", "num_inference_steps", "steps", 20),
        ("supports_guidance", "guidance_scale", "guidance", 7.5),
    ],
)
def test_optional_field_gates(flag, field_default, ctx_field, ctx_value):
    p_off = _Stub(provider_id="p")
    p_on = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(**{flag: True}),
    )
    ctx = ImagesContext(prompt="x", **{ctx_field: ctx_value})
    payload_off = p_off._build_payload(_model(), ctx)
    payload_on = p_on._build_payload(_model(), ctx)
    assert field_default not in payload_off
    assert payload_on[field_default] == ctx_value


def test_extra_payload_deep_merge():
    p = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(
            extra_payload={"watermark": False, "nested": {"a": 1}},
        ),
    )
    payload = p._build_payload(
        _model(),
        ImagesContext(
            prompt="x", extra={"nested": {"b": 2}, "seed_override": "ignored"}
        ),
    )
    assert payload["watermark"] is False
    # ctx.extra deep-merges over capability.extra_payload
    assert payload["nested"] == {"a": 1, "b": 2}
    assert payload["seed_override"] == "ignored"


def test_ctx_extra_wins_over_capability_extra_on_collision():
    p = _Stub(
        provider_id="p",
        capability=ImagesCapabilityDescriptor(extra_payload={"watermark": False}),
    )
    payload = p._build_payload(
        _model(),
        ImagesContext(prompt="x", extra={"watermark": True}),
    )
    assert payload["watermark"] is True


def test_model_capability_overrides_per_model():
    cap_default = ImagesCapabilityDescriptor()
    cap_flux_schnell = ImagesCapabilityDescriptor(
        size_spec=SizeSpec(kind="aspect_ratio")
    )
    p = _Stub(
        provider_id="together",
        capability=cap_default,
        model_capability_overrides={"flux-schnell": cap_flux_schnell},
    )
    payload_pro = p._build_payload(
        _model("flux-pro"),
        ImagesContext(prompt="x", size="1024x1024"),
    )
    payload_schnell = p._build_payload(
        _model("flux-schnell"),
        ImagesContext(prompt="x", size="1:1"),
    )
    assert "size" in payload_pro
    assert payload_schnell["aspect_ratio"] == "1:1"
