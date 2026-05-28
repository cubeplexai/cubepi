import asyncio

from cubepi.agent.types import AgentContext, BeforeToolCallContext
from cubepi.hitl import ApproveAnswer
from cubepi.hitl.channel import InMemoryChannel
from cubepi.hitl.middleware import ConfirmToolCallMiddleware
from cubepi.providers.base import AssistantMessage, TextContent, ToolCall


def _ctx(name="bash"):
    return BeforeToolCallContext(
        assistant_message=AssistantMessage(
            content=[TextContent(text=""), ToolCall(id="tc", name=name, arguments={})],
            stop_reason="tool_use",
        ),
        tool_call=ToolCall(id="tc", name=name, arguments={}),
        args={},
        context=AgentContext(system_prompt="", messages=[], tools=[]),
    )


async def test_set_based_require_confirm_only_asks_for_listed():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="approve"))

    asyncio.create_task(host())
    mw = ConfirmToolCallMiddleware(ch, require_confirm={"bash"})
    # bash: prompts
    assert (await mw.before_tool_call(_ctx("bash"))) is None
    # read_file: not in set — passes through silently
    assert (await mw.before_tool_call(_ctx("read_file"))) is None
    # bash prompted exactly once; read_file did not engage channel
    assert ch.pending is None


async def test_predicate_require_confirm():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="approve"))

    asyncio.create_task(host())

    def needs_confirm(ctx):
        return ctx.tool_call.name.startswith("dangerous_")

    mw = ConfirmToolCallMiddleware(ch, require_confirm=needs_confirm)
    assert (await mw.before_tool_call(_ctx("dangerous_op"))) is None


async def test_default_none_asks_for_every_tool():
    ch = InMemoryChannel()

    async def host():
        while ch.pending is None:
            await asyncio.sleep(0)
        await ch.answer(ch.pending.question_id, ApproveAnswer(decision="approve"))

    asyncio.create_task(host())
    mw = ConfirmToolCallMiddleware(ch)  # no require_confirm
    assert (await mw.before_tool_call(_ctx("anything"))) is None
