from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import aiosqlite

from cubepi.checkpointer.base import CheckpointData
from cubepi.hitl.types import HitlRequest
from cubepi.providers.base import (
    AssistantMessage,
    Message,
    ToolResultMessage,
    UserMessage,
)
from cubepi.types import JsonObject


class SQLiteCheckpointer:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

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
        await self._db.execute(
            "CREATE TABLE IF NOT EXISTS thread_pending_request ("
            "  thread_id TEXT PRIMARY KEY,"
            "  request_json TEXT NOT NULL,"
            "  run_id TEXT,"
            "  created_at REAL NOT NULL DEFAULT (julianday('now'))"
            ")"
        )
        # One-shot migration: older DBs (created before run_id existed) have
        # the table without the run_id column. SQLite has no schema_version
        # gate, so we ALTER inline when it's missing.
        cur = await self._db.execute("PRAGMA table_info(thread_pending_request)")
        cols = {row[1] for row in await cur.fetchall()}
        if "run_id" not in cols:
            await self._db.execute(
                "ALTER TABLE thread_pending_request ADD COLUMN run_id TEXT"
            )
        await self._db.commit()
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def load(self, thread_id: str) -> CheckpointData | None:
        assert self._db is not None
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT message_json FROM messages WHERE thread_id = ? ORDER BY id",
                (thread_id,),
            )
            rows = await cursor.fetchall()

            extra_cursor = await self._db.execute(
                "SELECT extra_json FROM thread_extra WHERE thread_id = ?",
                (thread_id,),
            )
            extra_row = await extra_cursor.fetchone()

            if not rows and not extra_row:
                return None

            messages = []
            for row in rows:
                msg_data = json.loads(row[0])
                messages.append(_deserialize_message(msg_data))

            extra = json.loads(extra_row[0]) if extra_row else {}
            return CheckpointData(messages=messages, extra=extra)

    async def append(self, thread_id: str, messages: list[Message]) -> None:
        assert self._db is not None
        async with self._lock:
            for msg in messages:
                msg_json = _serialize_message(msg)
                await self._db.execute(
                    "INSERT INTO messages (thread_id, message_json) VALUES (?, ?)",
                    (thread_id, msg_json),
                )
            await self._db.commit()

    async def save_extra(self, thread_id: str, extra: JsonObject) -> None:
        assert self._db is not None
        async with self._lock:
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

    async def save_pending_request(
        self,
        thread_id: str,
        request: HitlRequest | None,
        *,
        run_id: str | None = None,
    ) -> None:
        assert self._db is not None
        async with self._lock:
            if request is None:
                # Clearing pending implicitly clears the associated run_id.
                await self._db.execute(
                    "DELETE FROM thread_pending_request WHERE thread_id = ?",
                    (thread_id,),
                )
            else:
                payload = request.model_dump_json()
                # Single statement writes pending + run_id atomically.
                await self._db.execute(
                    "INSERT OR REPLACE INTO thread_pending_request "
                    "(thread_id, request_json, run_id) VALUES (?, ?, ?)",
                    (thread_id, payload, run_id),
                )
            await self._db.commit()

    async def load_pending_request(self, thread_id: str) -> HitlRequest | None:
        assert self._db is not None
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT request_json FROM thread_pending_request WHERE thread_id = ?",
                (thread_id,),
            )
            row = await cursor.fetchone()
            return HitlRequest.model_validate_json(row[0]) if row else None

    async def load_pending_run_id(self, thread_id: str) -> str | None:
        assert self._db is not None
        async with self._lock:
            cursor = await self._db.execute(
                "SELECT run_id FROM thread_pending_request WHERE thread_id = ?",
                (thread_id,),
            )
            row = await cursor.fetchone()
        return row[0] if row else None


def _serialize_message(msg: Any) -> str:
    if hasattr(msg, "model_dump"):
        return json.dumps(msg.model_dump())
    return json.dumps(msg)


def _deserialize_message(data: dict[str, Any]) -> Message:
    role = data.get("role")
    if role == "user":
        return UserMessage.model_validate(data)
    elif role == "assistant":
        return AssistantMessage.model_validate(data)
    elif role == "tool_result":
        return ToolResultMessage.model_validate(data)
    return cast(Message, data)
