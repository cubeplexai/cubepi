from __future__ import annotations

from cubepi.cli.trace.model import Span, build_forest


def _raw(span_id, parent_id, name, start, end=None, status="UNSET", attrs=None):
    return {
        "name": name,
        "context": {"trace_id": "0xtrace", "span_id": span_id},
        "parent_id": parent_id,
        "start_time": start,
        "end_time": end,
        "status": {"status_code": status},
        "attributes": attrs or {},
    }


def test_span_fields_and_duration():
    sp = Span(_raw("0x1", None, "invoke_agent",
                   "2026-05-20T00:00:00.000000Z", "2026-05-20T00:00:01.500000Z",
                   attrs={"cubepi.run_id": "run-1"}))
    assert sp.span_id == "0x1"
    assert sp.parent_id is None
    assert sp.name == "invoke_agent"
    assert sp.run_id == "run-1"
    assert sp.duration_ms == 1500.0
    assert sp.is_error is False


def test_error_and_abort_distinct():
    err = Span(_raw("0x2", "0x1", "chat gpt-x", "2026-05-20T00:00:00Z",
                    "2026-05-20T00:00:00.1Z", status="ERROR",
                    attrs={"gen_ai.operation.name": "chat"}))
    aborted = Span(_raw("0x3", "0x1", "chat gpt-x", "2026-05-20T00:00:00Z",
                        "2026-05-20T00:00:00.1Z",
                        attrs={"gen_ai.operation.name": "chat",
                               "cubepi.aborted": True,
                               "error.type": "cubepi.aborted"}))
    assert err.is_error is True and err.is_aborted is False
    assert aborted.is_error is False and aborted.is_aborted is True


def test_operation_classification():
    chat = Span(_raw("0x4", "0x1", "chat gpt-x", "2026-05-20T00:00:00Z",
                     attrs={"gen_ai.operation.name": "chat"}))
    tool = Span(_raw("0x5", "0x1", "execute_tool read", "2026-05-20T00:00:00Z",
                     attrs={"gen_ai.operation.name": "execute_tool"}))
    agent = Span(_raw("0x6", None, "invoke_agent", "2026-05-20T00:00:00Z",
                      attrs={"gen_ai.operation.name": "invoke_agent"}))
    assert chat.is_chat and not chat.is_tool
    assert tool.is_tool and not tool.is_chat
    assert agent.is_invoke_agent


def test_build_forest_orders_children_and_marks_orphans():
    # Children written before parent (real exporter order).
    spans = [
        Span(_raw("0xb", "0xa", "chat", "2026-05-20T00:00:00.2Z")),
        Span(_raw("0xc", "0xa", "execute_tool", "2026-05-20T00:00:00.1Z")),
        Span(_raw("0xa", None, "invoke_agent", "2026-05-20T00:00:00.0Z")),
        Span(_raw("0xz", "0xMISSING", "chat", "2026-05-20T00:00:00.3Z")),
    ]
    roots = build_forest(spans)
    # invoke_agent root + one orphan whose parent is absent.
    names = sorted(r.span.name for r in roots)
    assert names == ["chat", "invoke_agent"]
    root = next(r for r in roots if r.span.name == "invoke_agent")
    assert [c.span.name for c in root.children] == ["execute_tool", "chat"]
    orphan = next(r for r in roots if r.span.name == "chat")
    assert orphan.orphan is True
