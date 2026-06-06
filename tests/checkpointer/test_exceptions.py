import pytest

from cubepi.checkpointer.exceptions import (
    CheckpointerError,
    CheckpointerLockTimeoutError,
    CompletionMarkerFailedError,
    CubepiSchemaError,
    RunAlreadyClaimedError,
    RunAlreadyCompletedError,
    RunNotClaimedError,
    RunNotCompletedError,
    ThreadAlreadyExistsError,
    ThreadNotFoundError,
)


@pytest.mark.parametrize(
    "exc_cls",
    [
        ThreadNotFoundError,
        ThreadAlreadyExistsError,
        RunNotCompletedError,
        RunNotClaimedError,
        RunAlreadyClaimedError,
        RunAlreadyCompletedError,
        CheckpointerLockTimeoutError,
    ],
)
def test_runtime_errors_inherit_checkpointer_error(exc_cls):
    assert issubclass(exc_cls, CheckpointerError)
    assert issubclass(exc_cls, Exception)


def test_checkpointer_error_separate_from_schema_error():
    assert not issubclass(CheckpointerError, CubepiSchemaError)
    assert not issubclass(CubepiSchemaError, CheckpointerError)


def test_completion_marker_failed_error_carries_run_id():
    cause = RuntimeError("db timeout")
    exc = CompletionMarkerFailedError(
        thread_id="t1", run_id="r1", cause=cause
    )
    assert exc.thread_id == "t1"
    assert exc.run_id == "r1"
    assert exc.__cause__ is cause
    assert "t1" in str(exc) and "r1" in str(exc)


def test_runtime_errors_constructable_with_kwargs():
    e = ThreadNotFoundError("missing thread t1")
    assert "t1" in str(e)
    e = RunNotCompletedError("thread=t1 run=r1")
    assert "r1" in str(e)
