import pytest

from cubepi.agent.types import ForkOnceResult


def test_fork_once_result_constructs_and_is_frozen():
    r = ForkOnceResult(text="hi", messages=[], stop_reason="end_turn")
    assert r.text == "hi"
    assert r.messages == []
    assert r.stop_reason == "end_turn"
    with pytest.raises((AttributeError, Exception)):
        r.text = "mut"  # type: ignore[misc]
