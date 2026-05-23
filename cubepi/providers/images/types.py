from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from cubepi.providers.base import ImageContent, TextContent


class ImagesModel(BaseModel):
    id: str
    provider: str
    api: str = ""


class ImagesContext(BaseModel):
    prompt: str
    input_images: list[ImageContent] = Field(default_factory=list)


class AssistantImages(BaseModel):
    api: str
    provider: str
    model: str
    output: list[ImageContent | TextContent] = Field(default_factory=list)
    stop_reason: Literal["stop", "error", "aborted"] = "stop"
    error_message: str | None = None
