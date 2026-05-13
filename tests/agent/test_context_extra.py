"""AgentContext.extra field tests (D6)."""

from cubepi.agent.types import AgentContext


def test_agent_context_default_extra_is_empty_dict() -> None:
    ctx = AgentContext(system_prompt="", messages=[])
    assert ctx.extra == {}


def test_agent_context_accepts_extra() -> None:
    ctx = AgentContext(
        system_prompt="",
        messages=[],
        extra={"todos": ["a", "b"]},
    )
    assert ctx.extra["todos"] == ["a", "b"]


def test_extra_is_mutable() -> None:
    ctx = AgentContext(system_prompt="", messages=[])
    ctx.extra["k"] = "v"
    assert ctx.extra == {"k": "v"}


def test_extra_independent_between_instances() -> None:
    a = AgentContext(system_prompt="", messages=[])
    b = AgentContext(system_prompt="", messages=[])
    a.extra["x"] = 1
    assert "x" not in b.extra
