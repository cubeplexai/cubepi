from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field

from cubepi.types import JsonObject


class Option(BaseModel):
    label: str
    value: str
    description: str | None = None
    allow_input: bool = False


class Question(BaseModel):
    key: str
    prompt: str
    options: list[Option] | None = None
    multi_select: bool = False
    required: bool = True


class ConfirmRequest(BaseModel):
    kind: Literal["confirm"] = "confirm"
    prompt: str
    details: JsonObject | None = None


class ApproveRequest(BaseModel):
    kind: Literal["approve"] = "approve"
    tool_name: str
    tool_call_id: str
    args: JsonObject
    details: JsonObject | None = None


class AskRequest(BaseModel):
    kind: Literal["ask"] = "ask"
    questions: list[Question]


HitlPayload = Union[ConfirmRequest, ApproveRequest, AskRequest]


class HitlRequest(BaseModel):
    question_id: str
    thread_id: str | None
    payload: HitlPayload = Field(discriminator="kind")
    created_at: float
    timeout_seconds: float | None = None


class ApproveAnswer(BaseModel):
    decision: Literal["approve", "deny", "edit"]
    edited_args: JsonObject | None = None
    reason: str | None = None
