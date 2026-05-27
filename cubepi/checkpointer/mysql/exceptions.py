"""Schema exceptions for MySQLCheckpointer.

Re-exported from the backend-agnostic ``cubepi.checkpointer.exceptions``.
Importing from there (rather than the Postgres package) keeps a single source
of truth without pulling in ``asyncpg``, so the ``cubepi[mysql]`` extra works
without the Postgres extra installed.
"""

from cubepi.checkpointer.exceptions import (
    CubepiSchemaError,
    CubepiSchemaMismatch,
    CubepiSchemaUninitialized,
)

__all__ = [
    "CubepiSchemaError",
    "CubepiSchemaMismatch",
    "CubepiSchemaUninitialized",
]
