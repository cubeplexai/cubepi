from cubepi.agent.types import (
    AgentContext,
    BeforeToolCallContext,
    BeforeToolCallResult,
)
from cubepi.middleware.base import Middleware, compose_middleware
from cubepi.providers.base import AssistantMessage, TextContent, ToolCall


def _ctx(args: dict | None = None) -> BeforeToolCallContext:
    args = args or {}
    return BeforeToolCallContext(
        assistant_message=AssistantMessage(
            content=[
                TextContent(text="t"),
                ToolCall(id="tc-1", name="bash", arguments={"cmd": "ls"}),
            ],
            stop_reason="end_turn",
        ),
        tool_call=ToolCall(id="tc-1", name="bash", arguments={"cmd": "ls"}),
        args=args,
        context=AgentContext(system_prompt="", messages=[], tools=[]),
    )


class _MWEdit(Middleware):
    def __init__(self, edited):
        self._edited = edited

    async def before_tool_call(self, ctx, *, signal=None):
        return BeforeToolCallResult(
            edited_args=self._edited,
            hitl_trace={"decision": "edit", "by": "first"},
        )


class _MWBlock(Middleware):
    async def before_tool_call(self, ctx, *, signal=None):
        return BeforeToolCallResult(
            block=True,
            deny_reason="bad",
            hitl_trace={"decision": "policy_deny", "by": "second"},
        )


class _MWInspect(Middleware):
    """Records what args it sees from upstream edits."""

    def __init__(self):
        self.seen_args: list = []

    async def before_tool_call(self, ctx, *, signal=None):
        self.seen_args.append(ctx.args)
        return None


async def test_compose_before_edit_chain_passes_edited_args_downstream():
    inspect = _MWInspect()
    hooks = compose_middleware([_MWEdit({"cmd": "ls -l"}), inspect])
    result = await hooks["before_tool_call"](_ctx({"cmd": "ls"}))
    assert result is not None
    assert result.edited_args == {"cmd": "ls -l"}
    # Inspect MW should have seen the edited args, not the original
    assert inspect.seen_args == [{"cmd": "ls -l"}]


async def test_compose_before_block_after_edit_discards_edit_but_keeps_hitl_trace():
    hooks = compose_middleware([_MWEdit({"cmd": "ls -l"}), _MWBlock()])
    result = await hooks["before_tool_call"](_ctx())
    assert result.block is True
    assert result.deny_reason == "bad"
    # hitl_trace should contain the most-recent (the block) primary keys,
    # with the edit step archived under _chain
    assert result.hitl_trace["decision"] == "policy_deny"
    assert "_chain" in result.hitl_trace


async def test_compose_before_hitl_trace_merge_keeps_history():
    class _MWTrace1(Middleware):
        async def before_tool_call(self, ctx, *, signal=None):
            return BeforeToolCallResult(hitl_trace={"by": "one", "extra": 1})

    class _MWTrace2(Middleware):
        async def before_tool_call(self, ctx, *, signal=None):
            return BeforeToolCallResult(hitl_trace={"by": "two", "more": 2})

    hooks = compose_middleware([_MWTrace1(), _MWTrace2()])
    result = await hooks["before_tool_call"](_ctx())
    assert result.hitl_trace["by"] == "two"  # last writer wins
    assert result.hitl_trace["more"] == 2
    assert "_chain" in result.hitl_trace
    assert any(c.get("by") == "one" for c in result.hitl_trace["_chain"])


async def test_compose_before_returns_none_when_no_middleware_speaks():
    class _MWSilent(Middleware):
        async def before_tool_call(self, ctx, *, signal=None):
            return None

    hooks = compose_middleware([_MWSilent(), _MWSilent()])
    result = await hooks["before_tool_call"](_ctx())
    assert result is None
