from __future__ import annotations

from pydantic import BaseModel, Field


class CompactionState(BaseModel):
    """JSON-safe summary state stored in ``AgentContext.extra``."""

    summary: str
    summarized_message_ids: list[str] = Field(default_factory=list)
    last_summarized_message_id: str | None = None
