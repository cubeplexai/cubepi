from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast

from pydantic import BaseModel

from cubepi.agent.agent import Agent
from cubepi.agent.types import AgentEvent, AgentTool, AgentToolResult
from cubepi.middleware.base import Middleware
from cubepi.providers.base import AssistantMessage, BoundModel, TextContent
from cubepi.types import StructuredValue

logger = logging.getLogger(__name__)

EventMapper = Callable[[AgentEvent], Sequence[StructuredValue] | StructuredValue | None]
EventHandler = Callable[[str, StructuredValue], Awaitable[None] | None]


def _is_checkpointed_hitl(x: object) -> bool:
    """True if ``x.hitl`` is a checkpointed ``HitlBinding``."""
    binding = getattr(x, "hitl", None)
    return binding is not None and bool(getattr(binding, "checkpointed", False))


def _drop_checkpointed_hitl(items: Sequence) -> list:
    """Remove elements that carry — or expose — a checkpointed HITL binding.

    Used by ``SubagentMiddleware`` to keep parent-bound HITL state out of an
    ephemeral child agent. The child has its own run_id, but a checkpointed
    ``HitlBinding`` carries the parent's run_id, which
    ``Agent._validate_hitl_bindings`` rejects at ``prompt()`` entry.

    Two ways an element can introduce parent-bound HITL into the child:

    * The element itself has ``.hitl`` set (an ``AgentTool`` like
      ``ask_user_tool(...)`` or a middleware like
      ``ApprovalPolicyMiddleware``). Dropped directly.
    * A middleware that doesn't carry ``.hitl`` itself but exposes
      ``.tools`` containing an HITL-checkpointed tool — ``Agent.__init__``
      appends ``getattr(mw, "tools", [])`` to ``state.tools`` and the
      validator would still see the parent-bound tool there. The whole
      middleware is dropped in this case; constructing a tools-filtered
      proxy would break ``compose_middleware._has_method`` (it does
      ``type(mw)`` lookups), so dropping the unit is the simpler and
      type-safe option. Trade-off: any non-HITL hooks on that middleware
      are also lost in the subagent — by packaging HITL tools inside a
      middleware the host has effectively declared the middleware to be
      an HITL unit.

    Elements without ``.hitl`` (or with a non-checkpointed binding) and
    with no checkpointed-HITL tools inside their ``.tools`` attribute
    pass through untouched.
    """
    out: list = []
    for x in items:
        if _is_checkpointed_hitl(x):
            continue
        nested_tools = getattr(x, "tools", None)
        if nested_tools and any(_is_checkpointed_hitl(t) for t in nested_tools):
            continue
        out.append(x)
    return out


class SubagentTracer(Protocol):
    def attach(
        self, agent: Agent[BaseModel]
    ) -> Callable[[], Awaitable[None] | None] | None: ...


@dataclass(frozen=True)
class SubagentSpec:
    name: str
    description: str
    system_prompt: str
    model: BoundModel | None = None
    tools: Sequence[AgentTool[BaseModel]] = field(default_factory=tuple)
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
    events: list[StructuredValue]
    error: str | None = None


class SubagentMiddleware(Middleware):
    """Inject a tool that delegates one task to an ephemeral child agent."""

    def __init__(
        self,
        *,
        subagents: dict[str, SubagentSpec],
        default_model: BoundModel,
        shared_tools: Sequence[AgentTool[BaseModel]] = (),
        inherited_middleware: Sequence[Middleware] = (),
        excluded_tool_names: set[str] | None = None,
        event_mapper: EventMapper | None = None,
        event_handler: EventHandler | None = None,
        tracer: SubagentTracer | None = None,
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
        self._default_model = default_model
        self._shared_tools = tuple(
            tool for tool in shared_tools if tool.name not in excluded
        )
        self._inherited_middleware = tuple(inherited_middleware)
        self._event_mapper = event_mapper
        self._event_handler = event_handler
        self._tracer = tracer
        self.tools: list[AgentTool[BaseModel]] = [self._make_tool()]

    @property
    def subagents(self) -> dict[str, SubagentSpec]:
        return dict(self._subagents)

    @property
    def shared_tools(self) -> tuple[AgentTool[BaseModel], ...]:
        return self._shared_tools

    def _make_tool(self) -> AgentTool[SubagentRequest]:
        available = ", ".join(f'"{name}"' for name in self._subagents)

        async def execute(
            tool_call_id: str,
            args: SubagentRequest,
            *,
            signal: asyncio.Event | None = None,
            on_update: Callable[[StructuredValue], None] | None = None,
        ) -> AgentToolResult:
            del on_update
            try:
                result = await self._run_subagent(tool_call_id, args, signal=signal)
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
        *,
        signal: asyncio.Event | None = None,
    ) -> SubagentResult:
        spec = self._subagents.get(
            request.subagent_type,
            self._subagents["general-purpose"],
        )
        model = spec.model or self._default_model
        # Subagents are ephemeral and autonomous: they run under their own
        # run_id, so a tool/middleware whose HITL channel was checkpointed
        # against the parent's run_id can't legally pause from the child —
        # ``Agent._validate_hitl_bindings`` would reject the child's
        # ``prompt()`` call with "Agent has checkpointed HITL elements bound
        # to run_ids ...". Drop those bindings before constructing the child;
        # non-HITL behaviour on the same element (e.g. tool execution path,
        # non-HITL middleware logic) is unaffected because we only strip the
        # whole element when its ``.hitl`` is checkpointed.
        tools = _drop_checkpointed_hitl([*self._shared_tools, *spec.tools])
        middleware = _drop_checkpointed_hitl(
            [*self._inherited_middleware, *spec.middleware]
        )
        child: Agent[BaseModel] = Agent(
            model=model,
            system_prompt=spec.system_prompt,
            tools=tools,
            middleware=middleware,
        )

        agent_id = f"subagent:{tool_call_id}"
        events: list[StructuredValue] = []

        async def listener(
            event: AgentEvent, signal: asyncio.Event | None = None
        ) -> None:
            del signal
            await self._handle_event(agent_id, event, events)

        child.subscribe(listener)

        abort_task = self._start_abort_forwarder(child, signal)
        detach = self._attach_tracer(child)
        try:
            await child.prompt(request.prompt)
        finally:
            await self._stop_abort_forwarder(abort_task)
            await self._detach_tracer(detach)

        text = self._final_assistant_text(child) or "[subagent produced no output]"
        error = self._child_error(child)
        return SubagentResult(agent_id=agent_id, text=text, events=events, error=error)

    @staticmethod
    def _child_error(child: Agent[BaseModel]) -> str | None:
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
    def _final_assistant_text(child: Agent[BaseModel]) -> str:
        for message in reversed(child.state.messages):
            if not isinstance(message, AssistantMessage):
                continue
            return "".join(
                block.text
                for block in message.content
                if isinstance(block, TextContent)
            )
        return ""

    async def _handle_event(
        self,
        agent_id: str,
        event: AgentEvent,
        events: list[StructuredValue],
    ) -> None:
        if self._event_mapper is None:
            return
        mapped = self._event_mapper(event)
        if mapped is None:
            return
        if isinstance(mapped, Sequence) and not isinstance(mapped, (str, bytes)):
            payloads = list(mapped)
        else:
            payloads = [cast(StructuredValue, mapped)]
        for payload in payloads:
            events.append(payload)
            if self._event_handler is not None:
                result = self._event_handler(agent_id, payload)
                if inspect.isawaitable(result):
                    await result

    def _attach_tracer(
        self, child: Agent[BaseModel]
    ) -> Callable[[], Awaitable[None] | None] | None:
        if self._tracer is None:
            return None
        try:
            return self._tracer.attach(child)
        except Exception as exc:  # noqa: BLE001
            logger.debug("subagent tracer attach failed: %s", exc)
            return None

    def _start_abort_forwarder(
        self, child: Agent[BaseModel], signal: asyncio.Event | None
    ) -> asyncio.Task | None:
        if signal is None:
            return None

        async def forward_abort() -> None:
            await signal.wait()
            while not child.state.is_streaming:
                await asyncio.sleep(0)
            child.abort()

        return asyncio.create_task(forward_abort())

    async def _stop_abort_forwarder(self, task: asyncio.Task | None) -> None:
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _detach_tracer(
        self, detach: Callable[[], Awaitable[None] | None] | None
    ) -> None:
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
