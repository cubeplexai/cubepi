from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from cubepi.types import JsonObject


@dataclass(frozen=True)
class Approve:
    pass


@dataclass(frozen=True)
class Deny:
    reason: str


@dataclass(frozen=True)
class AskUser:
    prompt: str | None = None
    timeout_seconds: float | None = None
    details: JsonObject | None = None


ApprovalDecision = Union[Approve, Deny, AskUser]
