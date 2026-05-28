import dataclasses

import pytest
from cubepi.hitl.types import (
    Option,
    Question,
    ConfirmRequest,
    ApproveRequest,
    AskRequest,
    HitlRequest,
    ApproveAnswer,
)


def test_option_default_allow_input_false():
    o = Option(label="A", value="a")
    assert o.allow_input is False
    assert o.description is None


def test_question_defaults():
    q = Question(key="color", prompt="Pick:")
    assert q.options is None
    assert q.multi_select is False
    assert q.required is True


def test_confirm_request_kind_literal():
    r = ConfirmRequest(prompt="ok?")
    assert r.kind == "confirm"


def test_approve_request_kind_literal():
    r = ApproveRequest(tool_name="bash", tool_call_id="tc-1", args={"cmd": "ls"})
    assert r.kind == "approve"


def test_ask_request_kind_literal():
    r = AskRequest(questions=[Question(key="x", prompt="?")])
    assert r.kind == "ask"


def test_hitl_request_envelope_round_trip():
    req = HitlRequest(
        question_id="tc-1",
        thread_id="t-7",
        payload=ApproveRequest(tool_name="bash", tool_call_id="tc-1", args={}),
        created_at=1.0,
        timeout_seconds=42.0,
    )
    raw = req.model_dump_json()
    back = HitlRequest.model_validate_json(raw)
    assert back == req
    assert back.payload.kind == "approve"


def test_hitl_request_discriminated_union_round_trip_for_each_kind():
    payloads = [
        ConfirmRequest(prompt="ok?"),
        ApproveRequest(tool_name="t", tool_call_id="c", args={"a": 1}),
        AskRequest(questions=[Question(key="k", prompt="p")]),
    ]
    for p in payloads:
        req = HitlRequest(question_id="q", thread_id=None, payload=p, created_at=0.0)
        back = HitlRequest.model_validate_json(req.model_dump_json())
        assert type(back.payload) is type(p)


def test_approve_answer_decisions():
    assert ApproveAnswer(decision="approve").decision == "approve"
    assert ApproveAnswer(decision="deny", reason="no").reason == "no"
    assert ApproveAnswer(decision="edit", edited_args={"x": 1}).edited_args == {"x": 1}


def test_approval_decision_dataclasses_frozen():
    from cubepi.hitl.policy import Approve, Deny, AskUser

    a = Approve()
    d = Deny(reason="forbidden")
    u = AskUser(timeout_seconds=10.0)
    assert d.reason == "forbidden"
    assert u.timeout_seconds == 10.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.foo = "bar"  # frozen dataclass
