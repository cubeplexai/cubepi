from cubeloop.checkpointer.base import Checkpointer, CheckpointData
from cubeloop.checkpointer.memory import MemoryCheckpointer


def __getattr__(name: str) -> object:
    if name == "PostgresCheckpointer":
        from cubeloop.checkpointer.postgres.checkpointer import PostgresCheckpointer

        return PostgresCheckpointer
    if name == "SQLiteCheckpointer":
        from cubeloop.checkpointer.sqlite import SQLiteCheckpointer

        return SQLiteCheckpointer
    if name == "MySQLCheckpointer":
        from cubeloop.checkpointer.mysql.checkpointer import MySQLCheckpointer

        return MySQLCheckpointer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Checkpointer",
    "CheckpointData",
    "MemoryCheckpointer",
    "PostgresCheckpointer",
    "SQLiteCheckpointer",
    "MySQLCheckpointer",
]
