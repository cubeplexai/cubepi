import asyncio

from pydantic import BaseModel

from cubepi.agent.tools import execute_tool_calls
from cubepi.agent.types import (
    AfterToolCallResult,
    AgentContext,
    AgentTool,
    AgentToolResult,
    BeforeToolCallResult,
)
from cubepi.providers.base import AssistantMessage, TextContent, ToolCall


class EchoParams(BaseModel):
    value: str


def make_echo_tool(
    *,
    name: str = "echo",
    execution_mode=None,
    execute_fn=None,
) -> AgentTool:
    async def default_execute(tool_call_id, params, *, signal=None, on_update=None):
        return AgentToolResult(content=[TextContent(text=f"echoed: {params.value}")])

    return AgentTool(
        name=name,
        description="Echo tool",
        parameters=EchoParams,
        execute=execute_fn or default_execute,
        execution_mode=execution_mode,
    )


def make_assistant_msg(tool_calls: list[ToolCall]) -> AssistantMessage:
    return AssistantMessage(content=tool_calls, stop_reason="tool_use")


def make_context(tools: list[AgentTool]) -> AgentContext:
    return AgentContext(system_prompt="", messages=[], tools=tools)


class TestSequentialExecution:
    async def test_single_tool_call(self):
        tool = make_echo_tool()
        ctx = make_context([tool])
        msg = make_assistant_msg(
            [ToolCall(id="t1", name="echo", arguments={"value": "hi"})]
        )

        events = []
        batch = await execute_tool_calls(
            ctx,
            msg,
            tool_execution="sequential",
            emit=lambda e: events.append(e),
        )

        assert len(batch.messages) == 1
        assert batch.messages[0].tool_call_id == "t1"
        assert not batch.messages[0].is_error
        assert batch.terminate is False

    async def test_multiple_tool_calls_run_sequentially(self):
        order = []

        async def tracked_execute(tool_call_id, params, *, signal=None, on_update=None):
            order.append(f"start:{params.value}")
            await asyncio.sleep(0.01)
            order.append(f"end:{params.value}")
            return AgentToolResult(
                content=[TextContent(text=f"echoed: {params.value}")]
            )

        tool = make_echo_tool(execute_fn=tracked_execute)
        ctx = make_context([tool])
        msg = make_assistant_msg(
            [
                ToolCall(id="t1", name="echo", arguments={"value": "first"}),
                ToolCall(id="t2", name="echo", arguments={"value": "second"}),
            ]
        )

        await execute_tool_calls(
            ctx, msg, tool_execution="sequential", emit=lambda e: None
        )

        assert order == ["start:first", "end:first", "start:second", "end:second"]

    async def test_unknown_tool_returns_error(self):
        ctx = make_context([])
        msg = make_assistant_msg([ToolCall(id="t1", name="unknown", arguments={})])

        batch = await execute_tool_calls(
            ctx, msg, tool_execution="sequential", emit=lambda e: None
        )

        assert len(batch.messages) == 1
        assert batch.messages[0].is_error
        assert "not found" in batch.messages[0].content[0].text.lower()


class TestParallelExecution:
    async def test_tools_run_concurrently(self):
        first_resolved = False
        parallel_observed = False
        release = asyncio.Event()

        async def slow_execute(tool_call_id, params, *, signal=None, on_update=None):
            nonlocal first_resolved, parallel_observed
            if params.value == "first":
                await release.wait()
                first_resolved = True
            if params.value == "second" and not first_resolved:
                parallel_observed = True
            return AgentToolResult(
                content=[TextContent(text=f"echoed: {params.value}")]
            )

        tool = make_echo_tool(execute_fn=slow_execute)
        ctx = make_context([tool])
        msg = make_assistant_msg(
            [
                ToolCall(id="t1", name="echo", arguments={"value": "first"}),
                ToolCall(id="t2", name="echo", arguments={"value": "second"}),
            ]
        )

        async def run():
            await asyncio.sleep(0.02)
            release.set()

        asyncio.create_task(run())
        batch = await execute_tool_calls(
            ctx, msg, tool_execution="parallel", emit=lambda e: None
        )

        assert parallel_observed
        assert len(batch.messages) == 2
        assert batch.messages[0].tool_call_id == "t1"
        assert batch.messages[1].tool_call_id == "t2"

    async def test_parallel_produces_results_for_multiple_calls(self):
        tool = make_echo_tool()
        ctx = make_context([tool])
        msg = make_assistant_msg(
            [
                ToolCall(id="t1", name="echo", arguments={"value": "alpha"}),
                ToolCall(id="t2", name="echo", arguments={"value": "beta"}),
            ]
        )

        batch = await execute_tool_calls(
            ctx, msg, tool_execution="parallel", emit=lambda e: None
        )

        assert len(batch.messages) == 2
        assert batch.messages[0].tool_call_id == "t1"
        assert not batch.messages[0].is_error
        assert "alpha" in batch.messages[0].content[0].text
        assert batch.messages[1].tool_call_id == "t2"
        assert not batch.messages[1].is_error
        assert "beta" in batch.messages[1].content[0].text

    async def test_parallel_with_blocked_and_normal_tool(self):
        """One tool is blocked by before_tool_call, the other executes normally."""
        tool = make_echo_tool()
        ctx = make_context([tool])
        msg = make_assistant_msg(
            [
                ToolCall(id="t1", name="echo", arguments={"value": "blocked_one"}),
                ToolCall(id="t2", name="echo", arguments={"value": "allowed"}),
            ]
        )

        async def before(ctx_arg, *, signal=None):
            if ctx_arg.tool_call.id == "t1":
                return BeforeToolCallResult(block=True, reason="not allowed")
            return None

        batch = await execute_tool_calls(
            ctx,
            msg,
            tool_execution="parallel",
            before_tool_call=before,
            emit=lambda e: None,
        )

        assert len(batch.messages) == 2
        # First tool was blocked
        assert batch.messages[0].is_error
        assert "not allowed" in batch.messages[0].content[0].text
        # Second tool executed normally
        assert not batch.messages[1].is_error
        assert "allowed" in batch.messages[1].content[0].text

    async def test_parallel_tool_execute_exception(self):
        """Tool execution exception in parallel mode produces error result."""

        async def failing_execute(tool_call_id, params, *, signal=None, on_update=None):
            raise RuntimeError("parallel boom")

        tool = make_echo_tool(execute_fn=failing_execute)
        ctx = make_context([tool])
        msg = make_assistant_msg(
            [ToolCall(id="t1", name="echo", arguments={"value": "hi"})]
        )

        batch = await execute_tool_calls(
            ctx, msg, tool_execution="parallel", emit=lambda e: None
        )

        assert len(batch.messages) == 1
        assert batch.messages[0].is_error
        assert "parallel boom" in batch.messages[0].content[0].text

    async def test_sequential_tool_forces_sequential_mode(self):
        order = []

        async def tracked(tool_call_id, params, *, signal=None, on_update=None):
            order.append(f"start:{params.value}")
            await asyncio.sleep(0.01)
            order.append(f"end:{params.value}")
            return AgentToolResult(content=[TextContent(text="ok")])

        tool = make_echo_tool(execute_fn=tracked, execution_mode="sequential")
        ctx = make_context([tool])
        msg = make_assistant_msg(
            [
                ToolCall(id="t1", name="echo", arguments={"value": "a"}),
                ToolCall(id="t2", name="echo", arguments={"value": "b"}),
            ]
        )

        await execute_tool_calls(
            ctx, msg, tool_execution="parallel", emit=lambda e: None
        )

        assert order[0] == "start:a"
        assert order[1] == "end:a"


class TestBeforeToolCall:
    async def test_block_prevents_execution(self):
        tool = make_echo_tool()
        ctx = make_context([tool])
        msg = make_assistant_msg(
            [ToolCall(id="t1", name="echo", arguments={"value": "hi"})]
        )

        async def before(ctx_arg, *, signal=None):
            return BeforeToolCallResult(block=True, reason="Blocked by test")

        batch = await execute_tool_calls(
            ctx,
            msg,
            tool_execution="sequential",
            before_tool_call=before,
            emit=lambda e: None,
        )

        assert len(batch.messages) == 1
        assert batch.messages[0].is_error
        assert "Blocked by test" in batch.messages[0].content[0].text

    async def test_before_tool_call_exception_returns_error(self):
        tool = make_echo_tool()
        ctx = make_context([tool])
        msg = make_assistant_msg(
            [ToolCall(id="t1", name="echo", arguments={"value": "hi"})]
        )

        async def before(ctx_arg, *, signal=None):
            raise RuntimeError("hook exploded")

        batch = await execute_tool_calls(
            ctx,
            msg,
            tool_execution="sequential",
            before_tool_call=before,
            emit=lambda e: None,
        )

        assert len(batch.messages) == 1
        assert batch.messages[0].is_error
        assert "hook exploded" in batch.messages[0].content[0].text


class TestAfterToolCall:
    async def test_override_result(self):
        tool = make_echo_tool()
        ctx = make_context([tool])
        msg = make_assistant_msg(
            [ToolCall(id="t1", name="echo", arguments={"value": "hi"})]
        )

        async def after(ctx_arg, *, signal=None):
            return AfterToolCallResult(
                content=[TextContent(text="overridden")],
                terminate=True,
            )

        batch = await execute_tool_calls(
            ctx,
            msg,
            tool_execution="sequential",
            after_tool_call=after,
            emit=lambda e: None,
        )

        assert batch.messages[0].content[0].text == "overridden"
        assert batch.terminate is True

    async def test_after_tool_call_exception_returns_error(self):
        tool = make_echo_tool()
        ctx = make_context([tool])
        msg = make_assistant_msg(
            [ToolCall(id="t1", name="echo", arguments={"value": "hi"})]
        )

        async def after(ctx_arg, *, signal=None):
            raise RuntimeError("after hook failed")

        batch = await execute_tool_calls(
            ctx,
            msg,
            tool_execution="sequential",
            after_tool_call=after,
            emit=lambda e: None,
        )

        assert len(batch.messages) == 1
        assert batch.messages[0].is_error
        assert "after hook failed" in batch.messages[0].content[0].text


class TestToolExecutionError:
    async def test_tool_execute_exception_returns_error(self):
        async def failing_execute(tool_call_id, params, *, signal=None, on_update=None):
            raise RuntimeError("execution boom")

        tool = make_echo_tool(execute_fn=failing_execute)
        ctx = make_context([tool])
        msg = make_assistant_msg(
            [ToolCall(id="t1", name="echo", arguments={"value": "hi"})]
        )

        batch = await execute_tool_calls(
            ctx, msg, tool_execution="sequential", emit=lambda e: None
        )

        assert len(batch.messages) == 1
        assert batch.messages[0].is_error
        assert "execution boom" in batch.messages[0].content[0].text

    async def test_invalid_parameters_returns_error(self):
        tool = make_echo_tool()
        ctx = make_context([tool])
        # EchoParams requires a 'value' string; passing wrong type triggers validation error
        msg = make_assistant_msg(
            [ToolCall(id="t1", name="echo", arguments={"wrong_key": 123})]
        )

        batch = await execute_tool_calls(
            ctx, msg, tool_execution="sequential", emit=lambda e: None
        )

        assert len(batch.messages) == 1
        assert batch.messages[0].is_error


class TestValidationErrorFormatting:
    """Pin the contract that ValidationError text is model-friendly.

    A raw str(ValidationError) names pydantic's internal model class and
    points at the pydantic docs site, which the LLM cannot act on. The
    formatter must produce field-path-anchored lines the model can use to
    self-correct without a second round-trip.
    """

    async def test_missing_required_field_names_the_field(self):
        tool = make_echo_tool()
        ctx = make_context([tool])
        msg = make_assistant_msg([ToolCall(id="t1", name="echo", arguments={})])

        batch = await execute_tool_calls(
            ctx, msg, tool_execution="sequential", emit=lambda e: None
        )

        text = batch.messages[0].content[0].text
        assert "Invalid arguments for tool 'echo'" in text
        assert "value" in text
        assert "field required" in text
        # Raw pydantic decoration must NOT leak to the model.
        assert "errors.pydantic.dev" not in text
        assert "validation error" not in text.lower()[: len("validation error")] or (
            "validation error" not in text.lower().split("\n")[0]
        )

    async def test_discriminator_failure_lists_allowed_tags(self):
        from typing import Annotated, Literal, Union

        from pydantic import Field, RootModel

        class ListOp(BaseModel):
            operation: Literal["list"]

        class CreateOp(BaseModel):
            operation: Literal["create"]
            name: str

        union_type = Annotated[
            Union[ListOp, CreateOp],
            Field(discriminator="operation"),
        ]
        union_root = RootModel[union_type]

        async def execute_fn(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="ok")])

        tool = AgentTool(
            name="opt",
            description="discriminated tool",
            parameters=union_root,
            execute=execute_fn,
        )
        ctx = make_context([tool])

        msg = make_assistant_msg(
            [ToolCall(id="t1", name="opt", arguments={"operation": "delete"})]
        )

        batch = await execute_tool_calls(
            ctx, msg, tool_execution="sequential", emit=lambda e: None
        )

        text = batch.messages[0].content[0].text
        assert "discriminator" in text
        # Allowed tags must be enumerated so the model can self-correct.
        assert "list" in text
        assert "create" in text

    async def test_missing_discriminator_key_is_named(self):
        from typing import Annotated, Literal, Union

        from pydantic import Field, RootModel

        class ListOp(BaseModel):
            operation: Literal["list"]

        class CreateOp(BaseModel):
            operation: Literal["create"]
            name: str

        union_type = Annotated[
            Union[ListOp, CreateOp],
            Field(discriminator="operation"),
        ]
        union_root = RootModel[union_type]

        async def execute_fn(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="ok")])

        tool = AgentTool(
            name="opt",
            description="discriminated tool",
            parameters=union_root,
            execute=execute_fn,
        )
        ctx = make_context([tool])

        # Wrong top-level key — pydantic raises union_tag_not_found.
        # That error type does not carry expected_tags in its ctx, but the
        # error must still name the discriminator key so the model can find
        # it in the tool's JSON Schema and self-correct.
        msg = make_assistant_msg(
            [ToolCall(id="t1", name="opt", arguments={"action": "list"})]
        )

        batch = await execute_tool_calls(
            ctx, msg, tool_execution="sequential", emit=lambda e: None
        )

        text = batch.messages[0].content[0].text
        assert "missing required discriminator key 'operation'" in text
        # Surrounding quotes from pydantic's ctx must be stripped.
        assert "''operation''" not in text


class TestTermination:
    async def test_all_terminate_stops_loop(self):
        async def term_execute(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(content=[TextContent(text="done")], terminate=True)

        tool = make_echo_tool(execute_fn=term_execute)
        ctx = make_context([tool])
        msg = make_assistant_msg(
            [ToolCall(id="t1", name="echo", arguments={"value": "a"})]
        )

        batch = await execute_tool_calls(
            ctx, msg, tool_execution="sequential", emit=lambda e: None
        )
        assert batch.terminate is True

    async def test_partial_terminate_continues(self):
        async def maybe_term(tool_call_id, params, *, signal=None, on_update=None):
            return AgentToolResult(
                content=[TextContent(text="done")],
                terminate=(params.value == "first"),
            )

        tool = make_echo_tool(execute_fn=maybe_term)
        ctx = make_context([tool])
        msg = make_assistant_msg(
            [
                ToolCall(id="t1", name="echo", arguments={"value": "first"}),
                ToolCall(id="t2", name="echo", arguments={"value": "second"}),
            ]
        )

        batch = await execute_tool_calls(
            ctx, msg, tool_execution="parallel", emit=lambda e: None
        )
        assert batch.terminate is False


class TestToolEvents:
    async def test_emits_execution_lifecycle_events(self):
        tool = make_echo_tool()
        ctx = make_context([tool])
        msg = make_assistant_msg(
            [ToolCall(id="t1", name="echo", arguments={"value": "hi"})]
        )

        events = []
        await execute_tool_calls(
            ctx,
            msg,
            tool_execution="sequential",
            emit=lambda e: events.append(e),
        )

        types = [e.type for e in events]
        assert "tool_execution_start" in types
        assert "tool_execution_end" in types
        start_idx = types.index("tool_execution_start")
        end_idx = types.index("tool_execution_end")
        assert start_idx < end_idx


class TestToolResultDetails:
    async def test_details_propagated_to_tool_result_message(self):
        async def execute_with_details(
            tool_call_id, params, *, signal=None, on_update=None
        ):
            return AgentToolResult(
                content=[TextContent(text="result")],
                details={"execution_time": 42},
            )

        tool = make_echo_tool(execute_fn=execute_with_details)
        ctx = make_context([tool])
        msg = make_assistant_msg(
            [ToolCall(id="t1", name="echo", arguments={"value": "hi"})]
        )

        batch = await execute_tool_calls(
            ctx, msg, tool_execution="sequential", emit=lambda e: None
        )

        assert len(batch.messages) == 1
        assert batch.messages[0].details == {"execution_time": 42}
