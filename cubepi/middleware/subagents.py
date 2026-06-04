from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentEvent, AgentTool, AgentToolResult
from cubepi.middleware.base import Middleware
from cubepi.providers.base import AssistantMessage, Model, Provider, TextContent

logger = logging.getLogger(__name__)

EventMapper = Callable[[AgentEvent], Sequence[Any] | Any | None]
EventHandler = Callable[[str, Any], Awaitable[None] | None]


@dataclass(frozen=True)
class SubagentSpec:
    name: str
    description: str
    system_prompt: str
    provider: Provider | None = None
    model: Model | None = None
    tools: Sequence[AgentTool[Any]] = field(default_factory=tuple)
    middleware: Sequence[Middleware] = field(default_factory=tuple)


class SubagentRequest(BaseModel):
    name: str
    role: str
    task: str
    prompt: str
    subagent_type: str = "general-purpose"


@dataclass(frozen=True)
class SubagentResult:
    agent_id: str
    text: str
    events: list[Any]
    error: str | None = None


class SubagentMiddleware(Middleware):
    """Inject a tool that delegates one task to an ephemeral child agent."""

    def __init__(
        self,
        *,
        subagents: dict[str, SubagentSpec],
        default_provider: Provider,
        default_model: Model,
        shared_tools: Sequence[AgentTool[Any]] = (),
        inherited_middleware: Sequence[Middleware] = (),
        excluded_tool_names: set[str] | None = None,
        event_mapper: EventMapper | None = None,
        event_handler: EventHandler | None = None,
        tracer: Any = None,
    ) -> None:
        if "general-purpose" not in subagents:
            subagents = {
                **subagents,
                "general-purpose": SubagentSpec(
                    name="general-purpose",
                    description="A general-purpose AI assistant",
                    system_prompt="You are a helpful AI assistant.",
                ),
            }

        excluded = excluded_tool_names or {"subagent"}
        self._subagents = dict(subagents)
        self._default_provider = default_provider
        self._default_model = default_model
        self._shared_tools = tuple(
            tool for tool in shared_tools if tool.name not in excluded
        )
        self._inherited_middleware = tuple(inherited_middleware)
        self._event_mapper = event_mapper
        self._event_handler = event_handler
        self._tracer = tracer
        self.tools: list[AgentTool[Any]] = [self._make_tool()]

    @property
    def subagents(self) -> dict[str, SubagentSpec]:
        return dict(self._subagents)

    @property
    def shared_tools(self) -> tuple[AgentTool[Any], ...]:
        return self._shared_tools

    def _make_tool(self) -> AgentTool[SubagentRequest]:
        available = ", ".join(f'"{name}"' for name in self._subagents)

        async def execute(
            tool_call_id: str,
            args: SubagentRequest,
            *,
            signal: Any = None,
            on_update: Any = None,
        ) -> AgentToolResult:
            del signal, on_update
            try:
                result = await self._run_subagent(tool_call_id, args)
            except Exception as exc:
                logger.error("Subagent %s failed: %s", args.subagent_type, exc)
                return AgentToolResult(
                    content=[TextContent(text=f"[error: {exc}]")],
                    is_error=True,
                )
            if result.error is not None:
                return AgentToolResult(
                    content=[TextContent(text=f"[error: {result.error}]")],
                    details={"subagent_events": result.events},
                    is_error=True,
                )
            return AgentToolResult(
                content=[TextContent(text=result.text)],
                details={"subagent_events": result.events},
            )

        return AgentTool(
            name="subagent",
            description=(
                f"Delegate a task to a subagent. Available subagent types: {available}. "
                "Provide a name, role, task, and self-contained prompt."
            ),
            parameters=SubagentRequest,
            execute=execute,
        )

    async def _run_subagent(
        self,
        tool_call_id: str,
        request: SubagentRequest,
    ) -> SubagentResult:
        spec = self._subagents.get(
            request.subagent_type,
            self._subagents["general-purpose"],
        )
        provider = spec.provider or self._default_provider
        model = spec.model or self._default_model
        tools = [*self._shared_tools, *spec.tools]
        middleware = [*self._inherited_middleware, *spec.middleware]
        child = Agent(
            provider=provider,
            model=model,
            system_prompt=spec.system_prompt,
            tools=tools,
            middleware=middleware,
        )

        agent_id = f"subagent:{tool_call_id}"
        events: list[Any] = []
        text_parts: list[str] = []

        async def listener(event: AgentEvent, signal: Any = None) -> None:
            del signal
            self._collect_text(event, text_parts)
            await self._handle_event(agent_id, event, events)

        child.subscribe(listener)

        detach = self._attach_tracer(child)
        try:
            await child.prompt(request.prompt)
        finally:
            await self._detach_tracer(detach)

        text = "".join(text_parts) or "[subagent produced no output]"
        error = self._child_error(child)
        return SubagentResult(agent_id=agent_id, text=text, events=events, error=error)

    @staticmethod
    def _child_error(child: Agent[Any]) -> str | None:
        if child.state.error_message:
            return child.state.error_message
        if not child.state.messages:
            return None
        last = child.state.messages[-1]
        if not isinstance(last, AssistantMessage):
            return None
        if last.error_message is not None:
            return last.error_message
        if last.stop_reason == "error":
            return "subagent failed"
        return None

    @staticmethod
    def _collect_text(event: AgentEvent, text_parts: list[str]) -> None:
        if event.type != "message_end":
            return
        message = event.message
        if getattr(message, "role", None) != "assistant":
            return
        for block in getattr(message, "content", []):
            if isinstance(block, TextContent):
                text_parts.append(block.text)

    async def _handle_event(
        self,
        agent_id: str,
        event: AgentEvent,
        events: list[Any],
    ) -> None:
        if self._event_mapper is None:
            return
        mapped = self._event_mapper(event)
        if mapped is None:
            return
        payloads = list(mapped) if isinstance(mapped, Sequence) else [mapped]
        for payload in payloads:
            events.append(payload)
            if self._event_handler is not None:
                result = self._event_handler(agent_id, payload)
                if inspect.isawaitable(result):
                    await result

    def _attach_tracer(self, child: Agent[Any]) -> Any:
        if self._tracer is None:
            return None
        try:
            return self._tracer.attach(child)
        except Exception as exc:  # noqa: BLE001
            logger.debug("subagent tracer attach failed: %s", exc)
            return None

    async def _detach_tracer(self, detach: Any) -> None:
        if detach is None:
            return
        try:
            result = detach()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # noqa: BLE001
            logger.debug("subagent tracer detach failed: %s", exc)


__all__ = [
    "SubagentMiddleware",
    "SubagentRequest",
    "SubagentResult",
    "SubagentSpec",
]
