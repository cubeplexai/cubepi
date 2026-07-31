import pytest

from cubepi.agent.agent import Agent
from cubepi.providers.base import AssistantMessage, TextContent
from cubepi.providers.faux import FauxProvider


def _ok_faux() -> FauxProvider:
    p = FauxProvider()
    p.set_responses(
        [AssistantMessage(content=[TextContent(text="ok")], stop_reason="end_turn")]
    )
    return p


def _agent(**kw):
    return Agent(model=_ok_faux().model("faux-model"), **kw)


@pytest.mark.asyncio
async def test_prompt_returns_supplied_run_id():
    a = _agent()
    got = await a.prompt("hello", run_id="R1")
    assert got == "R1"


@pytest.mark.asyncio
async def test_prompt_generates_run_id_when_none():
    a = _agent()
    got = await a.prompt("hello")
    assert isinstance(got, str) and len(got) >= 8


@pytest.mark.asyncio
async def test_prompt_sets_then_clears_active_run_id_on_clean_return():
    a = _agent()
    assert a.state.active_run_id is None
    await a.prompt("hello", run_id="R1")
    assert a.state.active_run_id is None  # cleared on clean return


@pytest.mark.asyncio
async def test_prompt_leaves_active_run_id_set_on_raise(monkeypatch):
    """Spec §3.7 + Task 22: active_run_id must be LEFT SET on any
    propagating failure after claim."""
    a = Agent(model=_ok_faux().model("faux-model"))

    async def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(a, "_run_prompt", _boom)
    with pytest.raises(RuntimeError, match="boom"):
        await a.prompt("hello", run_id="R1")
    assert a.state.active_run_id == "R1"


@pytest.mark.asyncio
async def test_appended_messages_carry_run_id():
    from cubepi.checkpointer.memory import MemoryCheckpointer

    cp = MemoryCheckpointer()
    a = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="t",
    )
    await a.prompt("hello", run_id="R1")
    data = await cp.load("t")
    for m in data.messages:
        assert m.run_id == "R1"


@pytest.mark.asyncio
async def test_live_tool_results_carry_owning_run_id():
    from pydantic import BaseModel

    from cubepi.agent.types import AgentTool, AgentToolResult
    from cubepi.checkpointer.memory import MemoryCheckpointer
    from cubepi.providers.base import ToolResultMessage
    from cubepi.providers.faux import faux_assistant_message, faux_tool_call

    class EchoParams(BaseModel):
        value: str

    async def execute(tool_call_id, params, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=params.value)])

    live_provider_run_ids = []
    hook_snapshots = []

    def finish(messages, model):
        assistant = next(
            m
            for m in reversed(messages)
            if isinstance(m, AssistantMessage) and m.stop_reason == "tool_use"
        )
        tool_result = next(
            m for m in reversed(messages) if isinstance(m, ToolResultMessage)
        )
        live_provider_run_ids.append((assistant.run_id, tool_result.run_id))
        return faux_assistant_message("done")

    async def should_stop_after_turn(ctx):
        if ctx.tool_results:
            hook_snapshots.append(
                (
                    [m.run_id for m in ctx.tool_results],
                    [
                        m.run_id
                        for m in ctx.context.messages
                        if isinstance(m, ToolResultMessage)
                    ],
                    [
                        m.run_id
                        for m in ctx.new_messages
                        if isinstance(m, ToolResultMessage)
                    ],
                )
            )
        return False

    provider = FauxProvider()
    provider.set_responses(
        [
            faux_assistant_message(
                faux_tool_call("echo", {"value": "ok"}, id="call-1"),
                stop_reason="tool_use",
            ),
            finish,
        ]
    )
    checkpointer = MemoryCheckpointer()
    agent = Agent(
        model=provider.model("faux-model"),
        tools=[
            AgentTool(
                name="echo",
                description="Echo a value",
                parameters=EchoParams,
                execute=execute,
            )
        ],
        checkpointer=checkpointer,
        thread_id="run-id-tool-cycle",
        should_stop_after_turn=should_stop_after_turn,
    )
    events = []
    agent.subscribe(lambda event, signal=None: events.append(event))

    await agent.prompt("use the tool", run_id="R1")

    state_tool_result = next(
        message
        for message in agent.state.messages
        if isinstance(message, ToolResultMessage)
    )
    checkpoint = await checkpointer.load("run-id-tool-cycle")
    checkpoint_tool_result = next(
        message
        for message in checkpoint.messages
        if isinstance(message, ToolResultMessage)
    )
    assert state_tool_result.run_id == checkpoint_tool_result.run_id == "R1"

    assert live_provider_run_ids == [("R1", "R1")]
    assert hook_snapshots == [(["R1"], ["R1"], ["R1"])]

    tool_message_events = [
        event.message
        for event in events
        if event.type in ("message_start", "message_end")
        and isinstance(event.message, ToolResultMessage)
    ]
    assert [message.run_id for message in tool_message_events] == ["R1", "R1"]


@pytest.mark.asyncio
async def test_prompt_rejects_mismatched_run_id_before_claim():
    """Caller pre-stamps a Message with a different run_id than the
    one supplied to prompt(). Reject BEFORE claim_run so no row is
    written and the run_id is still reusable."""
    from cubepi.checkpointer.memory import MemoryCheckpointer
    from cubepi.providers.base import TextContent, UserMessage

    cp = MemoryCheckpointer()
    a = Agent(
        model=_ok_faux().model("faux-model"),
        checkpointer=cp,
        thread_id="t",
    )
    bad_msg = UserMessage(content=[TextContent(text="hi")], run_id="WRONG")
    with pytest.raises(ValueError, match="does not match"):
        await a.prompt(bad_msg, run_id="R1")
    # No claim row written — "R1" still freely claimable.
    # NOTE: As of Task 23 there is no claim_run yet (Task 25 adds it),
    # so cp._runs should be empty regardless. Once Task 25 lands,
    # this assertion will guarantee no claim row was written.
    assert "R1" not in cp._runs.get("t", {})
    # ... and a second prompt with the same run_id succeeds:
    await a.prompt("hi", run_id="R1")


@pytest.mark.asyncio
async def test_process_event_rejects_mismatched_message_run_id():
    """`_process_event` for MessageEndEvent stamps msg.run_id with
    state.active_run_id when msg.run_id is None, but raises if the
    message already carries a DIFFERENT run_id (defensive — a provider
    or middleware bug that produced the wrong stamp must not silently
    write to the persisted history)."""
    from cubepi.agent.types import MessageEndEvent
    from cubepi.providers.base import (
        AssistantMessage,
        TextContent,
    )

    a = Agent(model=_ok_faux().model("faux-model"))
    a._state.active_run_id = "R_active"
    bad_msg = AssistantMessage(
        content=[TextContent(text="bad")],
        stop_reason="end_turn",
        run_id="R_OTHER",
    )
    with pytest.raises(ValueError, match="does not match"):
        await a._process_event(MessageEndEvent(message=bad_msg))
