from cubepi.agent.agent import Agent
from cubepi.providers.base import Model, TextContent, UserMessage
from cubepi.providers.faux import FauxProvider, faux_assistant_message


def make_model() -> Model:
    return Model(id="faux-1", provider="faux")


async def test_steer_drained_at_turn_boundary_without_tool_calls():
    provider = FauxProvider()
    # Turn 1: plain text answer (no tool calls). Turn 2: plain text after steer.
    provider.set_responses(
        [
            faux_assistant_message("first answer"),
            faux_assistant_message("acknowledged the steer"),
        ]
    )
    agent = Agent(provider=provider, model=make_model())

    # Enqueue a steer before running; with no tool calls, the only chance to
    # drain it is the turn boundary.
    agent.steer(UserMessage(content=[TextContent(text="actually do X instead")]))
    await agent.prompt("start")

    roles = [m.role for m in agent.state.messages]
    # user(start) -> assistant(first) -> user(steer) -> assistant(ack)
    assert roles == ["user", "assistant", "user", "assistant"]
    assert any(
        getattr(b, "text", "") == "actually do X instead"
        for m in agent.state.messages
        if m.role == "user"
        for b in m.content
    )
