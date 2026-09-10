"""Schema exceptions for MySQLCheckpointer.

Re-exported from the backend-agnostic ``cubeloop.checkpointer.exceptions``.
Importing from there (rather than the Postgres package) keeps a single source
of truth without pulling in ``asyncpg``, so the ``cubepi[mysql]`` extra works
without the Postgres extra installed.
"""

from cubeloop.checkpointer.exceptions import (
    CubeloopSchemaError,
    CubeloopSchemaMismatch,
    CubeloopSchemaUninitialized,
)

__all__ = [
    "CubeloopSchemaError",
    "CubeloopSchemaMismatch",
    "CubeloopSchemaUninitialized",
]
