"""Tests for GoalMiddleware."""

from __future__ import annotations


from cubepi import Agent
from cubepi.agent.types import AgentContext
from cubepi.middleware.goal import GoalMiddleware
from cubepi.providers.base import (
    AssistantMessage,
    ImageContent,
    Message,
    TextContent,
    UserMessage,
)
from cubepi.providers.faux import FauxProvider, faux_assistant_message, faux_tool_call


def _ctx(messages: list[Message] | None = None) -> AgentContext:
    msgs = messages or []
    return AgentContext(system_prompt="", messages=msgs)


# ---------------------------------------------------------------------------
# transform_context guard branches (unit tests)
# ---------------------------------------------------------------------------


async def test_transform_context_empty_messages() -> None:
    evaluator = FauxProvider(provider_id="eval")
    goal = GoalMiddleware(evaluator=evaluator.model("eval"))
    result = await goal.transform_context([], ctx=_ctx())
    assert result == []


async def test_transform_context_last_not_user_message() -> None:
    evaluator = FauxProvider(provider_id="eval")
    goal = GoalMiddleware(evaluator=evaluator.model("eval"))
    msgs: list[Message] = [AssistantMessage(content=[TextContent(text="hi")])]
    result = await goal.transform_context(msgs, ctx=_ctx(msgs))
    assert result is msgs


async def test_transform_context_empty_content() -> None:
    evaluator = FauxProvider(provider_id="eval")
    goal = GoalMiddleware(evaluator=evaluator.model("eval"))
    msgs: list[Message] = [UserMessage(content=[])]
    result = await goal.transform_context(msgs, ctx=_ctx(msgs))
    assert result is msgs


async def test_transform_context_first_block_not_text() -> None:
    evaluator = FauxProvider(provider_id="eval")
    goal = GoalMiddleware(evaluator=evaluator.model("eval"))
    msgs: list[Message] = [UserMessage(content=[ImageContent(source="data:image/png")])]
    result = await goal.transform_context(msgs, ctx=_ctx(msgs))
    assert result is msgs


# ---------------------------------------------------------------------------
# Test 1: Without /goal prefix, middleware is transparent
# ---------------------------------------------------------------------------


async def test_no_goal_prefix_transparent() -> None:
    """Without /goal prefix, middleware does nothing — worker responds once, evaluator never called."""
    worker = FauxProvider(provider_id="worker")
    worker.set_responses([faux_assistant_message("done")])

    evaluator_provider = FauxProvider(provider_id="evaluator")
    evaluator_provider.set_responses([])  # should never be called

    goal = GoalMiddleware(evaluator=evaluator_provider.model("eval"))
    agent = Agent(model=worker.model("test"), middleware=[goal])

    await agent.prompt("fix the bug")

    assert worker.call_count == 1
    assert evaluator_provider.call_count == 0


# ---------------------------------------------------------------------------
# Test 2: Goal achieved on first evaluation
# ---------------------------------------------------------------------------


async def test_goal_achieved_first_eval() -> None:
    """Worker finishes, evaluator says achieved=True, loop stops."""
    worker = FauxProvider(provider_id="worker")
    worker.set_responses([faux_assistant_message("all done")])

    evaluator_provider = FauxProvider(provider_id="evaluator")
    evaluator_provider.set_responses(
        [
            faux_assistant_message(
                faux_tool_call(
                    "structured_output",
                    {"achieved": True, "reason": "All tests passing"},
                )
            )
        ]
    )

    goal = GoalMiddleware(evaluator=evaluator_provider.model("eval"))
    agent = Agent(model=worker.model("test"), middleware=[goal])

    await agent.prompt("/goal all tests pass")

    assert worker.call_count == 1
    assert evaluator_provider.call_count == 1
    assert agent.state.extra["goal"]["status"] == "achieved"
    assert agent.state.extra["goal"]["evaluations"] == 1


# ---------------------------------------------------------------------------
# Test 3: Goal not met on first eval, achieved on second
# ---------------------------------------------------------------------------


async def test_goal_retry_then_achieved() -> None:
    """Evaluator says no first, then yes on second try."""
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
                    {"achieved": False, "reason": "2 tests still failing"},
                )
            ),
            faux_assistant_message(
                faux_tool_call(
                    "structured_output",
                    {"achieved": True, "reason": "All tests pass"},
                )
            ),
        ]
    )

    goal = GoalMiddleware(evaluator=evaluator_provider.model("eval"))
    agent = Agent(model=worker.model("test"), middleware=[goal])

    await agent.prompt("/goal all tests pass")

    assert worker.call_count == 2
    assert evaluator_provider.call_count == 2
    assert agent.state.extra["goal"]["status"] == "achieved"
    assert agent.state.extra["goal"]["evaluations"] == 2


# ---------------------------------------------------------------------------
# Test 4: Goal never achieved — hits max_evaluations
# ---------------------------------------------------------------------------


async def test_goal_max_evaluations_exhausted() -> None:
    """Evaluator always says no, hits max_evaluations cap."""
    worker = FauxProvider(provider_id="worker")
    worker.set_responses(
        [
            faux_assistant_message("attempt 1"),
            faux_assistant_message("attempt 2"),
            faux_assistant_message("attempt 3"),
        ]
    )

    evaluator_provider = FauxProvider(provider_id="evaluator")
    evaluator_provider.set_responses(
        [
            faux_assistant_message(
                faux_tool_call(
                    "structured_output",
                    {"achieved": False, "reason": "still broken"},
                )
            ),
            faux_assistant_message(
                faux_tool_call(
                    "structured_output",
                    {"achieved": False, "reason": "still broken"},
                )
            ),
        ]
    )

    goal = GoalMiddleware(evaluator=evaluator_provider.model("eval"), max_evaluations=2)
    agent = Agent(model=worker.model("test"), middleware=[goal])

    await agent.prompt("/goal fix everything")

    assert agent.state.extra["goal"]["status"] == "exhausted"
    assert agent.state.extra["goal"]["evaluations"] == 2


# ---------------------------------------------------------------------------
# Test 5: /goal prefix stripped from the worker's message
# ---------------------------------------------------------------------------


async def test_goal_prefix_stripped_from_message() -> None:
    """/goal prefix is removed from the message the worker sees."""
    worker = FauxProvider(provider_id="worker")

    # Capture the messages the worker actually receives
    captured_messages: list = []

    def capture_and_respond(messages, model):
        captured_messages.extend(messages)
        return faux_assistant_message("done")

    worker.set_responses([capture_and_respond])

    evaluator_provider = FauxProvider(provider_id="evaluator")
    evaluator_provider.set_responses(
        [
            faux_assistant_message(
                faux_tool_call(
                    "structured_output",
                    {"achieved": True, "reason": "all green"},
                )
            )
        ]
    )

    goal = GoalMiddleware(evaluator=evaluator_provider.model("eval"))
    agent = Agent(model=worker.model("test"), middleware=[goal])

    await agent.prompt("/goal make all tests green")

    # The worker should receive the stripped message (no /goal prefix)
    first_user_msg = next(m for m in captured_messages if isinstance(m, UserMessage))
    first_text = first_user_msg.content[0].text
    assert not first_text.startswith("/goal")
    assert "make all tests green" in first_text


# ---------------------------------------------------------------------------
# Test 6: extra_llm_calls declares the evaluator BoundModel
# ---------------------------------------------------------------------------


def test_extra_llm_calls_declares_evaluator() -> None:
    """extra_llm_calls() yields exactly 1 element which is the evaluator BoundModel."""
    evaluator_provider = FauxProvider(provider_id="evaluator")
    evaluator_model = evaluator_provider.model("eval")

    goal = GoalMiddleware(evaluator=evaluator_model)
    calls = list(goal.extra_llm_calls())

    assert len(calls) == 1
    assert calls[0] is evaluator_model


# ---------------------------------------------------------------------------
# Test 7: Empty goal condition is treated as no-op
# ---------------------------------------------------------------------------


async def test_empty_goal_condition_transparent() -> None:
    """/goal with only whitespace after prefix is treated as no goal — transparent pass-through."""
    worker = FauxProvider(provider_id="worker")
    worker.set_responses([faux_assistant_message("done")])

    evaluator_provider = FauxProvider(provider_id="evaluator")
    evaluator_provider.set_responses([])

    goal = GoalMiddleware(evaluator=evaluator_provider.model("eval"))
    agent = Agent(model=worker.model("test"), middleware=[goal])

    await agent.prompt("/goal ")

    assert worker.call_count == 1
    assert evaluator_provider.call_count == 0
    assert "goal" not in agent.state.extra


# ---------------------------------------------------------------------------
# Test 8: Goal state restored from ctx.extra after checkpoint
# ---------------------------------------------------------------------------


async def test_goal_state_restored_from_extra() -> None:
    """If middleware instance lost state but ctx.extra has active goal, restore and continue."""
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
                    {"achieved": False, "reason": "not done yet"},
                )
            ),
            faux_assistant_message(
                faux_tool_call(
                    "structured_output",
                    {"achieved": True, "reason": "all done"},
                )
            ),
        ]
    )

    goal = GoalMiddleware(evaluator=evaluator_provider.model("eval"))
    agent = Agent(model=worker.model("test"), middleware=[goal])

    await agent.prompt("/goal all tests pass")

    assert agent.state.extra["goal"]["status"] == "achieved"
    assert agent.state.extra["goal"]["evaluations"] == 2

    # Simulate checkpoint restore: new middleware instance, same agent state
    goal2 = GoalMiddleware(evaluator=evaluator_provider.model("eval"))
    assert goal2._condition is None  # fresh instance has no state

    # Seed extra with an "active" goal as if mid-evaluation was checkpointed
    agent2 = Agent(model=worker.model("test"), middleware=[goal2])
    agent2._extra["goal"] = {
        "status": "active",
        "condition": "all tests pass",
        "evaluations": 1,
        "max_evaluations": 10,
        "last_reason": "2 tests still failing",
    }

    worker.set_responses([faux_assistant_message("third attempt")])
    evaluator_provider.set_responses(
        [
            faux_assistant_message(
                faux_tool_call(
                    "structured_output",
                    {"achieved": True, "reason": "all pass now"},
                )
            )
        ]
    )

    # Normal prompt (no /goal prefix) — middleware should restore from extra
    await agent2.prompt("continue working")

    assert agent2.state.extra["goal"]["status"] == "achieved"
    assert agent2.state.extra["goal"]["evaluations"] == 2
