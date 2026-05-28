from cubepi.agent.types import (
    AgentAbortedEvent,
    AgentSuspendedEvent,
    HitlAnswerEvent,
    HitlRequestEvent,
)
from cubepi.hitl.types import ConfirmRequest, HitlRequest


def _req() -> HitlRequest:
    return HitlRequest(
        question_id="q-1",
        thread_id="t-1",
        payload=ConfirmRequest(prompt="ok?"),
        created_at=0.0,
    )


def test_hitl_request_event_construct():
    e = HitlRequestEvent(request=_req())
    assert e.type == "hitl_request"
    assert e.request.question_id == "q-1"


def test_hitl_answer_event_construct():
    e = HitlAnswerEvent(question_id="q-1", answer=True)
    assert e.type == "hitl_answer"
    assert e.cancelled is False
    assert e.timed_out is False


def test_agent_suspended_event_construct():
    e = AgentSuspendedEvent(pending_request=_req())
    assert e.type == "agent_suspended"
    assert e.pending_request.question_id == "q-1"


def test_agent_aborted_event_construct():
    e = AgentAbortedEvent(reason="user closed")
    assert e.type == "agent_aborted"
