"""Schema exceptions for MySQLCheckpointer.

These are backend-agnostic schema errors; re-exported from the Postgres
module so there is a single source of truth.
"""

from cubepi.checkpointer.postgres.exceptions import (
    CubepiSchemaError,
    CubepiSchemaMismatch,
    CubepiSchemaUninitialized,
)

__all__ = [
    "CubepiSchemaError",
    "CubepiSchemaMismatch",
    "CubepiSchemaUninitialized",
]
