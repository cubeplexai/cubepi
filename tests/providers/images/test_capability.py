from cubepi.providers.images.capability import (
    ImagesCapabilityDescriptor,
    SizeSpec,
)


def test_default_descriptor_is_openai_shape():
    d = ImagesCapabilityDescriptor()
    assert d.size_spec.kind == "size_string"
    assert d.count_field == "n"
    assert d.supports_seed is False
    assert d.supports_negative_prompt is False
    assert d.supports_steps is False
    assert d.supports_guidance is False
    assert d.output_format_field == "output_format"
    # response_format defaults to OFF so OpenAI gpt-image-1 — which rejects
    # the field — works without an explicit override.
    assert d.response_format_field is None
    assert d.response_format_value == "b64_json"
    assert d.supports_edit is True
    assert d.input_images_field == "image"
    assert d.extra_payload == {}


def test_size_spec_kinds_enum():
    for kind in ("size_string", "image_size_string", "width_height", "aspect_ratio"):
        s = SizeSpec(kind=kind)
        assert s.kind == kind


def test_descriptor_for_doubao_shape():
    d = ImagesCapabilityDescriptor(
        supports_seed=True,
        extra_payload={"watermark": False},
    )
    assert d.supports_seed is True
    assert d.extra_payload == {"watermark": False}
    assert d.size_spec.kind == "size_string"


def test_descriptor_for_siliconflow_shape():
    d = ImagesCapabilityDescriptor(
        size_spec=SizeSpec(kind="image_size_string"),
        count_field="batch_size",
        supports_seed=True,
        supports_steps=True,
        steps_field="num_inference_steps",
        supports_guidance=True,
        guidance_field="guidance_scale",
        supports_negative_prompt=True,
        output_format_field=None,
    )
    assert d.size_spec.kind == "image_size_string"
    assert d.count_field == "batch_size"
    assert d.steps_field == "num_inference_steps"
    assert d.guidance_field == "guidance_scale"
    assert d.output_format_field is None


def test_descriptor_for_together_flux_schnell_shape():
    d = ImagesCapabilityDescriptor(
        size_spec=SizeSpec(kind="aspect_ratio"),
        supports_seed=True,
        supports_steps=True,
        steps_field="steps",
    )
    assert d.size_spec.kind == "aspect_ratio"
    assert d.steps_field == "steps"
