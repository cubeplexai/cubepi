from cubepi.checkpointer.base import Checkpointer, CheckpointData
from cubepi.checkpointer.memory import MemoryCheckpointer
from cubepi.checkpointer.sqlite import SQLiteCheckpointer


def __getattr__(name: str) -> object:
    if name == "PostgresCheckpointer":
        from cubepi.checkpointer.postgres.checkpointer import PostgresCheckpointer

        return PostgresCheckpointer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Checkpointer",
    "CheckpointData",
    "MemoryCheckpointer",
    "PostgresCheckpointer",
    "SQLiteCheckpointer",
]
