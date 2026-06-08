from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel

from cubepi import Agent, AgentTool, AgentToolResult, TextContent, UserMessage
from cubepi.middleware.subagents import SubagentMiddleware, SubagentSpec
from cubepi.providers.base import AssistantMessage, MessageStream, StreamEvent
from cubepi.providers.faux import FauxProvider, faux_assistant_message, faux_tool_call


def _make_middleware(
    *,
    provider: FauxProvider | None = None,
    subagents: dict[str, SubagentSpec] | None = None,
    **kwargs: Any,
) -> SubagentMiddleware:
    provider = provider or FauxProvider(provider_id="faux")
    subagents = subagents or {
        "general-purpose": SubagentSpec(
            name="general-purpose",
            description="general",
            system_prompt="You are a subagent.",
        )
    }
    return SubagentMiddleware(
        subagents=subagents,
        default_model=provider.model("faux-1"),
        **kwargs,
    )


def test_middleware_exposes_subagent_tool() -> None:
    middleware = _make_middleware()

    assert len(middleware.tools) == 1
    assert middleware.tools[0].name == "subagent"
    schema = middleware.tools[0].parameters.model_json_schema()
    assert "prompt" in schema["properties"]
    assert "subagent_type" in schema["properties"]


def test_general_purpose_fallback_is_registered() -> None:
    middleware = _make_middleware(
        subagents={
            "specialist": SubagentSpec(
                name="specialist",
                description="niche",
                system_prompt="You are special.",
            )
        }
    )

    assert "general-purpose" in middleware.subagents
    assert "specialist" in middleware.subagents


def test_shared_tools_filter_is_configurable() -> None:
    class _NoArgs(BaseModel):
        pass

    async def _noop(
        tool_call_id: str,
        args: _NoArgs,
        *,
        signal: Any = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        del tool_call_id, args, signal, on_update
        return AgentToolResult(content=[TextContent(text="ok")])

    safe = AgentTool(name="safe", description="safe", parameters=_NoArgs, execute=_noop)
    subagent = AgentTool(
        name="subagent", description="recursive", parameters=_NoArgs, execute=_noop
    )
    load_skill = AgentTool(
        name="load_skill",
        description="host-specific",
        parameters=_NoArgs,
        execute=_noop,
    )

    middleware = _make_middleware(
        shared_tools=[safe, subagent, load_skill],
        excluded_tool_names={"subagent", "load_skill"},
    )

    assert middleware.shared_tools == (safe,)


async def test_subagent_tool_dispatches_to_child_agent() -> None:
    provider = FauxProvider(provider_id="faux")
    provider.set_responses([faux_assistant_message("subagent reply")])
    middleware = _make_middleware(provider=provider)
    [tool] = middleware.tools

    args = tool.parameters(
        name="worker",
        role="researcher",
        task="answer",
        prompt="please reply",
        subagent_type="general-purpose",
    )
    result = await tool.execute("tc-1", args, signal=None, on_update=None)

    assert result.is_error is not True
    assert result.content[0].text == "subagent reply"
    assert isinstance(result.details, dict)
    assert "subagent_events" in result.details


async def test_unknown_subagent_type_falls_back_to_general_purpose() -> None:
    provider = FauxProvider(provider_id="faux")
    provider.set_responses([faux_assistant_message("fallback reply")])
    middleware = _make_middleware(provider=provider)
    [tool] = middleware.tools

    args = tool.parameters(
        name="worker",
        role="researcher",
        task="answer",
        prompt="please reply",
        subagent_type="missing",
    )
    result = await tool.execute("tc-2", args, signal=None, on_update=None)

    assert result.is_error is not True
    assert result.content[0].text == "fallback reply"


async def test_subagent_tool_returns_error_when_dispatch_raises() -> None:
    middleware = _make_middleware()
    [tool] = middleware.tools

    async def fail_run(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("dispatch boom")

    middleware._run_subagent = fail_run  # type: ignore[method-assign]
    args = tool.parameters(
        name="worker",
        role="researcher",
        task="answer",
        prompt="please reply",
        subagent_type="general-purpose",
    )

    result = await tool.execute("tc-dispatch-error", args, signal=None, on_update=None)

    assert result.is_error is True
    assert "dispatch boom" in result.content[0].text


async def test_subagent_tool_returns_only_child_final_answer() -> None:
    class _EchoParams(BaseModel):
        value: str

    async def _echo(
        tool_call_id: str,
        args: _EchoParams,
        *,
        signal: Any = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        del tool_call_id, signal, on_update
        return AgentToolResult(content=[TextContent(text=f"echoed {args.value}")])

    echo_tool = AgentTool(
        name="echo",
        description="echo",
        parameters=_EchoParams,
        execute=_echo,
    )
    provider = FauxProvider(provider_id="faux")
    provider.set_responses(
        [
            faux_assistant_message(
                [
                    TextContent(text="I'll check."),
                    faux_tool_call("echo", {"value": "x"}, id="echo-1"),
                ],
                stop_reason="tool_use",
            ),
            faux_assistant_message("final answer"),
        ]
    )
    middleware = _make_middleware(provider=provider, shared_tools=[echo_tool])
    [tool] = middleware.tools

    args = tool.parameters(
        name="worker",
        role="researcher",
        task="answer",
        prompt="please use a tool",
        subagent_type="general-purpose",
    )
    result = await tool.execute("tc-final", args, signal=None, on_update=None)

    assert result.is_error is not True
    assert result.content[0].text == "final answer"


async def test_event_mapper_and_handler_receive_child_events() -> None:
    provider = FauxProvider(provider_id="faux")
    provider.set_responses([faux_assistant_message("mapped reply")])
    handled: list[tuple[str, dict[str, Any]]] = []

    def mapper(event: Any) -> list[dict[str, Any]]:
        return [{"type": event.type}]

    async def handler(agent_id: str, payload: dict[str, Any]) -> None:
        handled.append((agent_id, payload))

    middleware = _make_middleware(
        provider=provider,
        event_mapper=mapper,
        event_handler=handler,
    )
    [tool] = middleware.tools

    args = tool.parameters(
        name="worker",
        role="researcher",
        task="answer",
        prompt="please reply",
        subagent_type="general-purpose",
    )
    result = await tool.execute("tc-3", args, signal=None, on_update=None)

    assert handled
    assert all(agent_id == "subagent:tc-3" for agent_id, _ in handled)
    assert result.details["subagent_events"] == [payload for _, payload in handled]


async def test_string_event_mapper_payload_is_treated_as_one_payload() -> None:
    provider = FauxProvider(provider_id="faux")
    provider.set_responses([faux_assistant_message("mapped reply")])
    handled: list[tuple[str, str]] = []

    def mapper(event: Any) -> str | None:
        if event.type == "agent_start":
            return "started"
        return None

    def handler(agent_id: str, payload: str) -> None:
        handled.append((agent_id, payload))

    middleware = _make_middleware(
        provider=provider,
        event_mapper=mapper,
        event_handler=handler,
    )
    [tool] = middleware.tools

    args = tool.parameters(
        name="worker",
        role="researcher",
        task="answer",
        prompt="please reply",
        subagent_type="general-purpose",
    )
    result = await tool.execute("tc-string", args, signal=None, on_update=None)

    assert handled == [("subagent:tc-string", "started")]
    assert result.details["subagent_events"] == ["started"]


def test_child_error_handles_empty_non_assistant_and_error_assistant_states() -> None:
    provider = FauxProvider(provider_id="faux")
    child = Agent(
        model=provider.model("faux-1"),
    )

    assert SubagentMiddleware._child_error(child) is None

    child.state.messages = [UserMessage(content=[TextContent(text="user only")])]
    assert SubagentMiddleware._child_error(child) is None

    child.state.messages = [
        AssistantMessage(content=[], error_message="assistant failed")
    ]
    assert SubagentMiddleware._child_error(child) == "assistant failed"

    child.state.messages = [AssistantMessage(content=[], stop_reason="error")]
    assert SubagentMiddleware._child_error(child) == "subagent failed"


def test_final_assistant_text_returns_empty_without_assistant_message() -> None:
    provider = FauxProvider(provider_id="faux")
    child = Agent(
        model=provider.model("faux-1"),
    )
    child.state.messages = [UserMessage(content=[TextContent(text="user only")])]

    assert SubagentMiddleware._final_assistant_text(child) == ""


async def test_inner_agent_failure_returns_tool_error() -> None:
    class _FailingProvider(FauxProvider):
        async def stream(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise RuntimeError("inner boom")

    middleware = _make_middleware(provider=_FailingProvider())
    [tool] = middleware.tools

    args = tool.parameters(
        name="worker",
        role="researcher",
        task="answer",
        prompt="please reply",
        subagent_type="general-purpose",
    )
    result = await tool.execute("tc-4", args, signal=None, on_update=None)

    assert result.is_error is True
    assert "inner boom" in result.content[0].text


async def test_tracer_detaches_after_success_and_cancel() -> None:
    tracer = _FakeTracer(awaitable_detach=True)
    provider = FauxProvider(provider_id="faux")
    provider.set_responses([faux_assistant_message("ok")])
    middleware = _make_middleware(provider=provider, tracer=tracer)
    [tool] = middleware.tools

    args = tool.parameters(
        name="worker",
        role="researcher",
        task="answer",
        prompt="please reply",
        subagent_type="general-purpose",
    )
    result = await tool.execute("tc-5", args, signal=None, on_update=None)

    assert result.is_error is not True
    assert len(tracer.attached) == 1
    assert tracer.detached == 1
    assert tracer.awaited is True


async def test_cancelled_subagent_run_propagates_and_detaches_tracer() -> None:
    class _CancelledProvider(FauxProvider):
        async def stream(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise asyncio.CancelledError()

    tracer = _FakeTracer()
    middleware = _make_middleware(provider=_CancelledProvider(), tracer=tracer)
    [tool] = middleware.tools

    args = tool.parameters(
        name="worker",
        role="researcher",
        task="answer",
        prompt="please reply",
        subagent_type="general-purpose",
    )
    try:
        await tool.execute("tc-6", args, signal=None, on_update=None)
    except asyncio.CancelledError:
        pass
    else:  # pragma: no cover
        raise AssertionError("CancelledError was swallowed")

    assert len(tracer.attached) == 1
    assert tracer.detached == 1


async def test_parent_signal_aborts_running_child_agent() -> None:
    class _SlowProvider(FauxProvider):
        async def stream(self, *args: Any, **kwargs: Any) -> MessageStream:
            del args
            options = kwargs["options"]
            stream = MessageStream()

            async def produce() -> None:
                await options.signal.wait()
                stream.push(StreamEvent(type="error", error_message="aborted"))
                stream.set_result(
                    AssistantMessage(
                        content=[],
                        stop_reason="aborted",
                        error_message="aborted",
                    )
                )

            stream.attach_task(asyncio.create_task(produce()))
            return stream

    signal = asyncio.Event()
    middleware = _make_middleware(provider=_SlowProvider())
    [tool] = middleware.tools
    args = tool.parameters(
        name="worker",
        role="researcher",
        task="answer",
        prompt="please reply",
        subagent_type="general-purpose",
    )

    task = asyncio.create_task(tool.execute("tc-abort", args, signal=signal))
    await asyncio.sleep(0.05)
    signal.set()
    result = await asyncio.wait_for(task, timeout=2)

    assert result.is_error is True
    assert "aborted" in result.content[0].text


async def test_abort_forwarder_waits_for_child_streaming_before_abort() -> None:
    class _State:
        is_streaming = False

    class _Child:
        state = _State()
        aborted = False

        def abort(self) -> None:
            self.aborted = True

    child = _Child()
    signal = asyncio.Event()
    middleware = _make_middleware()
    task = middleware._start_abort_forwarder(child, signal)  # type: ignore[arg-type]
    assert task is not None

    signal.set()
    await asyncio.sleep(0)
    assert child.aborted is False

    child.state.is_streaming = True
    await asyncio.wait_for(task, timeout=1)
    assert child.aborted is True


async def test_stop_abort_forwarder_cancels_active_task() -> None:
    middleware = _make_middleware()

    async def never() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(never())

    await middleware._stop_abort_forwarder(task)

    assert task.cancelled()


def test_tracer_attach_failure_is_ignored() -> None:
    class _BadTracer:
        def attach(self, agent: Any) -> Any:
            del agent
            raise RuntimeError("attach failed")

    middleware = _make_middleware(tracer=_BadTracer())
    provider = FauxProvider(provider_id="faux")
    child = Agent(
        model=provider.model("faux-1"),
    )

    assert middleware._attach_tracer(child) is None


async def test_tracer_detach_failure_is_ignored() -> None:
    middleware = _make_middleware()

    def detach() -> None:
        raise RuntimeError("detach failed")

    await middleware._detach_tracer(detach)


class _FakeTracer:
    def __init__(self, *, awaitable_detach: bool = False) -> None:
        self.awaitable_detach = awaitable_detach
        self.attached: list[Any] = []
        self.detached = 0
        self.awaited = False

    def attach(self, agent: Any) -> Any:
        self.attached.append(agent)

        def detach() -> Any:
            self.detached += 1
            if not self.awaitable_detach:
                return None

            async def flush() -> None:
                await asyncio.sleep(0)
                self.awaited = True

            return flush()

        return detach


async def test_subagent_strips_checkpointed_hitl_bound_tools() -> None:
    """A shared_tool whose ``.hitl`` is a checkpointed binding (parent's
    ``ask_user_tool`` is the typical case) must NOT reach the child agent.

    Otherwise ``Agent._validate_hitl_bindings`` at the child's ``prompt()``
    entry would reject the run with ``"Agent has checkpointed HITL elements
    bound to run_ids ..."`` because the child runs under its own run_id.
    """
    from cubepi.hitl.binding import HitlBinding

    class _NoOpParams(BaseModel):
        pass

    async def _noop_exec(
        call_id: str, args: _NoOpParams, *, signal=None, on_update=None
    ) -> AgentToolResult:
        del call_id, args, signal, on_update
        return AgentToolResult(content=[TextContent(text="noop")])

    parent_bound_tool = AgentTool(
        name="parent_hitl_tool",
        description="A tool whose HITL channel is checkpointed against a parent run_id",
        parameters=_NoOpParams,
        execute=_noop_exec,
        hitl=HitlBinding(checkpointed=True, run_id="parent-run-abc"),
    )

    plain_tool = AgentTool(
        name="plain_tool",
        description="Plain tool with no HITL binding",
        parameters=_NoOpParams,
        execute=_noop_exec,
    )

    provider = FauxProvider(provider_id="faux")
    provider.set_responses([faux_assistant_message("subagent done")])

    middleware = _make_middleware(
        provider=provider,
        shared_tools=[parent_bound_tool, plain_tool],
    )

    parent_provider = FauxProvider(provider_id="faux")
    parent_provider.set_responses(
        [
            faux_assistant_message(
                [
                    faux_tool_call(
                        "subagent",
                        {"name": "n", "role": "r", "task": "t", "prompt": "p"},
                    )
                ],
                stop_reason="tool_use",
            ),
            faux_assistant_message("done"),
        ]
    )
    parent = Agent(
        model=parent_provider.model("faux-1"),
        system_prompt="parent",
        tools=middleware.tools,
        middleware=[middleware],
    )

    # Without the strip this raises:
    #   ValueError: Agent has checkpointed HITL elements bound to run_ids ['parent-run-abc']; ...
    await parent.prompt("dispatch the subagent")

    assert parent.state.error_message is None
