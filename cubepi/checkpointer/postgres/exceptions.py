"""Exceptions raised by PostgresCheckpointer schema verification.

Re-exported from the backend-agnostic ``cubepi.checkpointer.exceptions`` so
the same classes are shared with the MySQL checkpointer.
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
