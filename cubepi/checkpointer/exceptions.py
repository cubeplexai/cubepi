"""Backend-agnostic schema exceptions for SQL checkpointers.

Shared by the Postgres and MySQL checkpointers so that ``except
CubepiSchemaError`` works across backends and neither backend's package
(and its driver) needs to be imported to obtain them.
"""


class CubepiSchemaError(Exception):
    """Base class for cubepi SQL checkpointer schema errors."""


class CubepiSchemaUninitialized(CubepiSchemaError):
    """The cubepi_schema_version table is empty or missing.

    Typically means the host application's alembic upgrade hasn't been
    run yet against this database.
    """


class CubepiSchemaMismatch(CubepiSchemaError):
    """The DB schema version doesn't match cubepi's expected version.

    Typically means the cubepi library was upgraded but the host
    application's alembic is behind. Run a new alembic revision.
    """

    def __init__(self, *, expected: int, actual: int, hint: str = "") -> None:
        msg = f"cubepi schema mismatch: expected={expected} actual={actual}."
        if hint:
            msg += f" {hint}"
        super().__init__(msg)
        self.expected = expected
        self.actual = actual


class CheckpointerError(Exception):
    """Base class for cubepi checkpointer runtime errors.

    Distinct from ``CubepiSchemaError`` (schema-vs-library incompatibility).
    ``CheckpointerError`` covers runtime operation outcomes — missing
    thread, lock timeout, run state, etc.
    """


class ThreadNotFoundError(CheckpointerError):
    pass


class ThreadAlreadyExistsError(CheckpointerError):
    pass


class RunNotCompletedError(CheckpointerError):
    """The cubepi_runs row for (thread_id, run_id) does not exist, or
    exists with completed_at IS NULL (paused, abandoned, or in flight)."""


class RunNotClaimedError(CheckpointerError):
    """mark_run_complete() called but no cubepi_runs row exists for
    (thread_id, run_id). Indicates an agent-loop logic bug."""


class RunAlreadyClaimedError(CheckpointerError):
    """claim_run() found an existing row with completed_at IS NULL.
    Another process is currently running this run_id; retry with a
    different run_id."""


class RunAlreadyCompletedError(CheckpointerError):
    """claim_run() found an existing row with completed_at IS NOT NULL.
    Runs are append-only; start a new run with a different run_id.

    NOT raised by mark_run_complete() — that path is idempotent on
    already-completed rows (spec §3.6.2).
    """


class CheckpointerLockTimeoutError(CheckpointerError):
    """Backend writer lock not acquired within the configured timeout
    (SQLite busy_timeout, etc.)."""


class CompletionMarkerFailedError(CheckpointerError):
    """mark_run_complete() failed AFTER the run's final append succeeded.
    Carries `run_id` so callers using prompt(run_id=None) can recover
    the cubepi-generated value (spec §3.6.2)."""

    def __init__(
        self,
        *,
        thread_id: str,
        run_id: str,
        cause: BaseException,
    ) -> None:
        super().__init__(
            f"mark_run_complete failed for ({thread_id}, {run_id}): {cause}"
        )
        self.thread_id = thread_id
        self.run_id = run_id
        self.__cause__ = cause
