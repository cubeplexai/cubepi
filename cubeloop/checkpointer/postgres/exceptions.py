"""Exceptions raised by PostgresCheckpointer schema verification.

Re-exported from the backend-agnostic ``cubeloop.checkpointer.exceptions`` so
the same classes are shared with the MySQL checkpointer.
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
