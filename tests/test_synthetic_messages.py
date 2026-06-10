"""Tests for the unified synthetic-message marker (issue #171).

Framework-injected user-role messages (todo guard, goal continuation,
compaction summary, …) must carry ``metadata.synthetic == True`` so
downstream consumers can distinguish them from messages a human typed.
"""

from __future__ import annotations

from typing import Any

from cubepi import Agent
from cubepi.agent.types import AgentContext
from cubepi.middleware.goal import GoalMiddleware
from cubepi.middleware.todo import TodoListMiddleware
from cubepi.providers.base import (
    SYNTHETIC_METADATA_KEY,
    SYNTHETIC_SOURCE_METADATA_KEY,
    AssistantMessage,
    Message,
    TextContent,
    UserMessage,
    is_synthetic_message,
    synthetic_user_message,
)
from cubepi.providers.faux import FauxProvider, faux_assistant_message, faux_tool_call

# ---------------------------------------------------------------------------
# Factory + predicate unit tests
# ---------------------------------------------------------------------------


def test_factory_sets_marker_and_source() -> None:
    msg = synthetic_user_message("nudge text", source="todo_guard")
    assert isinstance(msg, UserMessage)
    assert msg.content == [TextContent(text="nudge text")]
    assert msg.metadata[SYNTHETIC_METADATA_KEY] is True
    assert msg.metadata[SYNTHETIC_SOURCE_METADATA_KEY] == "todo_guard"


def test_predicate_true_for_synthetic() -> None:
    assert is_synthetic_message(synthetic_user_message("x", source="test"))


def test_predicate_false_for_plain_user_message() -> None:
    assert not is_synthetic_message(UserMessage(content=[TextContent(text="hi")]))


def test_predicate_false_for_assistant_message() -> None:
    assert not is_synthetic_message(AssistantMessage(content=[TextContent(text="hi")]))


def test_predicate_false_for_non_boolean_marker() -> None:
    msg = UserMessage(
        content=[TextContent(text="hi")],
        metadata={SYNTHETIC_METADATA_KEY: "yes"},
    )
    assert not is_synthetic_message(msg)


def test_marker_survives_serialization_round_trip() -> None:
    """The marker must be data, not class identity: persistence (checkpointer,
    downstream APIs) round-trips through plain dicts."""
    original = synthetic_user_message("nudge", source="goal_continuation")
    restored = UserMessage.model_validate(original.model_dump())
    assert is_synthetic_message(restored)
    assert restored.metadata[SYNTHETIC_SOURCE_METADATA_KEY] == "goal_continuation"


# ---------------------------------------------------------------------------
# Real user input must NOT be marked
# ---------------------------------------------------------------------------


async def test_real_user_prompt_is_not_synthetic() -> None:
    provider = FauxProvider(provider_id="worker")
    provider.set_responses([faux_assistant_message("hello")])
    agent = Agent(model=provider.model("test"))

    await agent.prompt("real human input")

    user_messages = [m for m in agent.state.messages if isinstance(m, UserMessage)]
    assert user_messages
    assert not any(is_synthetic_message(m) for m in user_messages)


# ---------------------------------------------------------------------------
# Goal middleware: continuation nudge is synthetic
# ---------------------------------------------------------------------------


async def test_goal_continuation_message_is_synthetic() -> None:
    worker = FauxProvider(provider_id="worker")
    worker.set_responses(
        [
            faux_assistant_message("first attempt"),
            faux_assistant_message("second attempt"),
        ]
    )
    evaluator_provider = FauxProvider(provider_id="evaluator")
    evaluator_provider.set_responses(
        [
            faux_assistant_message(
                faux_tool_call(
                    "structured_output",
                    {"achieved": False, "reason": "not yet"},
                )
            ),
            faux_assistant_message(
                faux_tool_call(
                    "structured_output",
                    {"achieved": True, "reason": "done"},
                )
            ),
        ]
    )

    goal = GoalMiddleware(evaluator=evaluator_provider.model("eval"))
    agent = Agent(model=worker.model("test"), middleware=[goal])
    await agent.prompt("/goal all tests pass")

    continuations = [
        m
        for m in agent.state.messages
        if isinstance(m, UserMessage)
        and m.content
        and isinstance(m.content[0], TextContent)
        and m.content[0].text.startswith("Goal not yet met")
    ]
    assert continuations, "expected an injected goal continuation message"
    for msg in continuations:
        assert is_synthetic_message(msg)
        assert msg.metadata[SYNTHETIC_SOURCE_METADATA_KEY] == "goal_continuation"


# ---------------------------------------------------------------------------
# Todo middleware: injected error/guard messages are synthetic
# ---------------------------------------------------------------------------


def _todo_middleware(extra: dict[str, Any]) -> TodoListMiddleware:
    return TodoListMiddleware(extra_ref=lambda: extra)


async def test_todo_parallel_write_todos_errors_are_synthetic() -> None:
    extra: dict[str, Any] = {}
    middleware = _todo_middleware(extra)

    response = faux_assistant_message(
        [
            faux_tool_call("write_todos", {"todos": []}),
            faux_tool_call("write_todos", {"todos": []}),
        ]
    )
    ctx = AgentContext(system_prompt="", messages=[], extra=extra)

    action = await middleware.after_model_response(response, ctx)

    assert action is not None and action.inject_messages
    for msg in action.inject_messages:
        assert is_synthetic_message(msg)


async def test_todo_validation_errors_are_synthetic() -> None:
    extra: dict[str, Any] = {}
    middleware = _todo_middleware(extra)

    # Invalid payload: unfinished todo without any in_progress item.
    response = faux_assistant_message(
        faux_tool_call(
            "write_todos",
            {"todos": [{"content": "task", "status": "pending"}]},
        )
    )
    ctx = AgentContext(system_prompt="", messages=[], extra=extra)

    action = await middleware.after_model_response(response, ctx)

    assert action is not None and action.inject_messages
    for msg in action.inject_messages:
        assert is_synthetic_message(msg)


async def test_todo_context_reminder_is_synthetic() -> None:
    extra: dict[str, Any] = {
        "todos": [{"content": "task", "status": "in_progress"}],
    }
    middleware = _todo_middleware(extra)
    history: list[Message] = [UserMessage(content=[TextContent(text="hi")])]
    ctx = AgentContext(system_prompt="", messages=history, extra=extra)

    transformed = await middleware.transform_context(list(history), ctx=ctx)

    appended = transformed[len(history) :]
    assert appended, "expected the todo reminder to be appended"
    for msg in appended:
        assert isinstance(msg, UserMessage)
        assert is_synthetic_message(msg)
