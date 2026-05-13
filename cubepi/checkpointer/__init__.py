from cubepi.checkpointer.base import Checkpointer, CheckpointData
from cubepi.checkpointer.memory import MemoryCheckpointer


def __getattr__(name: str) -> object:
    if name == "PostgresCheckpointer":
        from cubepi.checkpointer.postgres.checkpointer import PostgresCheckpointer

        return PostgresCheckpointer
    if name == "SQLiteCheckpointer":
        from cubepi.checkpointer.sqlite import SQLiteCheckpointer

        return SQLiteCheckpointer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Checkpointer",
    "CheckpointData",
    "MemoryCheckpointer",
    "PostgresCheckpointer",
    "SQLiteCheckpointer",
]
