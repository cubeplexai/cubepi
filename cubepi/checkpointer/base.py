from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class CheckpointData:
    messages: list[Any] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Checkpointer(Protocol):
    async def load(self, thread_id: str) -> CheckpointData | None: ...
    async def append(self, thread_id: str, messages: list[Any]) -> None: ...
    async def save_extra(self, thread_id: str, extra: dict[str, Any]) -> None: ...
