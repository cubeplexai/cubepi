from cubepi.agent.types import AgentTool
from cubepi.hitl.binding import HitlBinding
from cubepi.middleware.base import Middleware
from pydantic import BaseModel


class _NoArgs(BaseModel):
    pass


async def _noop(tool_call_id, args, *, signal=None, on_update=None):
    raise NotImplementedError


def test_agent_tool_hitl_default_none():
    t = AgentTool(
        name="t",
        description="d",
        parameters=_NoArgs,
        execute=_noop,
    )
    assert t.hitl is None


def test_agent_tool_hitl_can_be_set():
    binding = HitlBinding(checkpointed=True, run_id="r-1")
    t = AgentTool(
        name="t",
        description="d",
        parameters=_NoArgs,
        execute=_noop,
        hitl=binding,
    )
    assert t.hitl is binding
    assert t.hitl.checkpointed is True
    assert t.hitl.run_id == "r-1"


def test_middleware_hitl_default_none():
    class _Mw(Middleware):
        pass

    assert _Mw().hitl is None


def test_middleware_hitl_can_be_set_in_subclass():
    class _Mw(Middleware):
        def __init__(self) -> None:
            self.hitl = HitlBinding(checkpointed=False, run_id=None)

    mw = _Mw()
    assert mw.hitl is not None
    assert mw.hitl.checkpointed is False
    assert mw.hitl.run_id is None


def test_hitl_binding_is_frozen():
    b = HitlBinding(checkpointed=True, run_id="r-1")
    try:
        b.checkpointed = False  # type: ignore[misc]
    except Exception:
        return
    assert False, "HitlBinding should be frozen"
