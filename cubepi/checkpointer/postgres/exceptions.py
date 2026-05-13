"""Exceptions raised by PostgresCheckpointer schema verification."""


class CubepiSchemaError(Exception):
    """Base class for cubepi Postgres schema errors."""


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
