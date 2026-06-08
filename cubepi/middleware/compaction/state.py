from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, Field

from cubepi.providers.base import Message


class CompactionState(BaseModel):
    """JSON-safe summary state stored in ``AgentContext.extra``."""

    summary: str
    summarized_message_ids: list[str] = Field(default_factory=list)
    summarized_message_refs: list[str] = Field(default_factory=list)
    last_summarized_message_id: str | None = None
    is_fallback: bool = False


def message_ref(message: Message) -> str:
    message_id = str(getattr(message, "id", "") or "")
    if message_id:
        return f"id:{message_id}"
    payload = message.model_dump(mode="json", exclude_none=True)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def message_refs(messages: list[Message]) -> list[str]:
    return [message_ref(message) for message in messages]
