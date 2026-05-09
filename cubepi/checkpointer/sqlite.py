from __future__ import annotations

import json
from typing import Any

import aiosqlite

from cubepi.checkpointer.base import CheckpointData


class SQLiteCheckpointer:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def __aenter__(self) -> SQLiteCheckpointer:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  thread_id TEXT NOT NULL,"
            "  message_json TEXT NOT NULL,"
            "  created_at REAL NOT NULL DEFAULT (julianday('now'))"
            ")"
        )
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS thread_extra ("
            "  thread_id TEXT PRIMARY KEY,"
            "  extra_json TEXT NOT NULL DEFAULT '{}'"
            ")"
        )
        await self._db.commit()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def load(self, thread_id: str) -> CheckpointData | None:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT message_json FROM messages WHERE thread_id = ? ORDER BY id",
            (thread_id,),
        )
        rows = await cursor.fetchall()
        if not rows:
            extra_cursor = await self._db.execute(
                "SELECT extra_json FROM thread_extra WHERE thread_id = ?",
                (thread_id,),
            )
            extra_row = await extra_cursor.fetchone()
            if not extra_row:
                return None

        messages = []
        for row in rows:
            msg_data = json.loads(row[0])
            messages.append(_deserialize_message(msg_data))

        extra_cursor = await self._db.execute(
            "SELECT extra_json FROM thread_extra WHERE thread_id = ?",
            (thread_id,),
        )
        extra_row = await extra_cursor.fetchone()
        extra = json.loads(extra_row[0]) if extra_row else {}

        return CheckpointData(messages=messages, extra=extra)

    async def append(self, thread_id: str, messages: list[Any]) -> None:
        assert self._db is not None
        for msg in messages:
            msg_json = _serialize_message(msg)
            await self._db.execute(
                "INSERT INTO messages (thread_id, message_json) VALUES (?, ?)",
                (thread_id, msg_json),
            )
        await self._db.commit()

    async def save_extra(self, thread_id: str, extra: dict[str, Any]) -> None:
        assert self._db is not None
        existing_cursor = await self._db.execute(
            "SELECT extra_json FROM thread_extra WHERE thread_id = ?",
            (thread_id,),
        )
        existing_row = await existing_cursor.fetchone()
        if existing_row:
            existing_extra = json.loads(existing_row[0])
            existing_extra.update(extra)
            await self._db.execute(
                "UPDATE thread_extra SET extra_json = ? WHERE thread_id = ?",
                (json.dumps(existing_extra), thread_id),
            )
        else:
            await self._db.execute(
                "INSERT INTO thread_extra (thread_id, extra_json) VALUES (?, ?)",
                (thread_id, json.dumps(extra)),
            )
        await self._db.commit()


def _serialize_message(msg: Any) -> str:
    if hasattr(msg, "model_dump"):
        return json.dumps(msg.model_dump())
    return json.dumps(msg)


def _deserialize_message(data: dict) -> Any:
    from cubepi.providers.base import AssistantMessage, ToolResultMessage, UserMessage

    role = data.get("role")
    if role == "user":
        return UserMessage.model_validate(data)
    elif role == "assistant":
        return AssistantMessage.model_validate(data)
    elif role == "tool_result":
        return ToolResultMessage.model_validate(data)
    return data
