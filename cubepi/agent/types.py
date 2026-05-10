from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, Literal, TypeVar

from pydantic import BaseModel

from cubepi.providers.base import (
    AssistantMessage,
    Content,
    Message,
    StreamEvent,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
)

TParams = TypeVar("TParams", bound=BaseModel)
TMessage = TypeVar("TMessage")


class AgentToolResult(BaseModel):
    content: list[Content]
    details: Any = None
    terminate: bool | None = None


@dataclass
class AgentTool(Generic[TParams]):
    name: str
    description: str
    parameters: type[TParams]
    execute: Callable[..., Awaitable[AgentToolResult]]
    label: str = ""
    execution_mode: Literal["sequential", "parallel"] | None = None

    def to_definition(self) -> ToolDefinition:
        schema = self.parameters.model_json_schema()
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=schema,
        )


@dataclass
class AgentContext:
    system_prompt: str
    messages: list[Message]
    tools: list[AgentTool] | None = None


# --- Hook context types ---


class BeforeToolCallResult(BaseModel):
    block: bool = False
    reason: str | None = None


class AfterToolCallResult(BaseModel):
    content: list[Content] | None = None
    details: Any = None
    is_error: bool | None = None
    terminate: bool | None = None


@dataclass
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: Any
    context: AgentContext


@dataclass
class AfterToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: Any
    result: AgentToolResult
    is_error: bool
    context: AgentContext


@dataclass
class ShouldStopAfterTurnContext:
    message: AssistantMessage
    tool_results: list[ToolResultMessage]
    context: AgentContext
    new_messages: list[Message]


# --- Event types (11 total, matching pi) ---


class AgentStartEvent(BaseModel):
    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(BaseModel):
    type: Literal["agent_end"] = "agent_end"
    messages: list[Message]


class TurnStartEvent(BaseModel):
    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(BaseModel):
    type: Literal["turn_end"] = "turn_end"
    message: AssistantMessage
    tool_results: list[ToolResultMessage]


class MessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: Message


class MessageUpdateEvent(BaseModel):
    type: Literal["message_update"] = "message_update"
    message: AssistantMessage
    stream_event: StreamEvent


class MessageEndEvent(BaseModel):
    type: Literal["message_end"] = "message_end"
    message: Message


class ToolExecutionStartEvent(BaseModel):
    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: Any


class ToolExecutionUpdateEvent(BaseModel):
    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    args: Any = None
    partial_result: Any = None


class ToolExecutionEndEvent(BaseModel):
    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: Any = None
    is_error: bool = False


AgentEvent = (
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
)

AgentEventSink = Callable[[AgentEvent], Awaitable[None]]

AgentListener = Callable[[AgentEvent, asyncio.Event | None], Awaitable[None] | None]
