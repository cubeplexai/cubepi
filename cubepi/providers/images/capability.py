from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SizeSpec(BaseModel):
    """How the canonical ``ctx.size`` value is serialized to the wire.

    - ``size_string``       → ``{"size": ctx.size}`` (OpenAI, Doubao)
    - ``image_size_string`` → ``{"image_size": ctx.size}`` (SiliconFlow)
    - ``width_height``      → split ``"<W>x<H>"`` into ``{"width": W, "height": H}``
    - ``aspect_ratio``      → ``{"aspect_ratio": ctx.size}`` (Together FLUX schnell, Imagen)
    """

    kind: Literal["size_string", "image_size_string", "width_height", "aspect_ratio"]


class ImagesCapabilityDescriptor(BaseModel):
    """Data-level description of an OpenAI-shape image backend's wire quirks.

    The descriptor is consumed by ``BaseImagesProvider._build_payload`` to
    rename canonical CubePi fields onto whatever wire keys the backend
    expects, and to gate ``ImagesContext`` fields the backend does not
    support. Backends whose shape is fundamentally different (async-task
    models like Aliyun Wanxiang, Imagen on Vertex, Stability, Replicate)
    need a separate provider subclass; this descriptor does not try to
    cover them.
    """

    size_spec: SizeSpec = Field(default_factory=lambda: SizeSpec(kind="size_string"))
    count_field: str = "n"

    supports_seed: bool = False
    seed_field: str = "seed"

    supports_negative_prompt: bool = False
    negative_prompt_field: str = "negative_prompt"

    supports_steps: bool = False
    steps_field: str = "num_inference_steps"

    supports_guidance: bool = False
    guidance_field: str = "guidance_scale"

    output_format_field: str | None = "output_format"
    response_format_field: str = "response_format"
    response_format_value: Literal["b64_json", "url"] = "b64_json"

    supports_edit: bool = True
    input_images_field: str = "image"

    extra_payload: dict[str, Any] = Field(default_factory=dict)
