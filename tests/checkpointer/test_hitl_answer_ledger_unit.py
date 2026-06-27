from __future__ import annotations

import json

import pytest

from cubepi.checkpointer.mysql.checkpointer import MySQLCheckpointer
from cubepi.checkpointer.postgres.checkpointer import PostgresCheckpointer


class _AsyncCtx:
    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _PgConn:
    def __init__(self, row=None):
        self.row = row
        self.exec_calls: list[tuple[str, tuple]] = []

    async def execute(self, sql, *args):
        self.exec_calls.append((sql, args))

    async def fetchrow(self, sql, *args):
        self.exec_calls.append((sql, args))
        return self.row

    def transaction(self):
        return _AsyncCtx(None)


class _PgPool:
    def __init__(self, conn: _PgConn):
        self.conn = conn

    def acquire(self):
        return _AsyncCtx(self.conn)


class _MyCursor:
    def __init__(self, conn: "_MyConn", row=None):
        self.conn = conn
        self.row = row

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, sql, params):
        self.conn.exec_calls.append((sql, params))

    async def fetchone(self):
        return self.row


class _MyConn:
    def __init__(self, row=None):
        self.row = row
        self.exec_calls: list[tuple[str, tuple]] = []
        self.begin_calls = 0

    async def begin(self):
        self.begin_calls += 1

    def cursor(self):
        return _AsyncCtx(_MyCursor(self, self.row))

    async def commit(self):
        return None

    async def rollback(self):
        return None


class _MyPool:
    def __init__(self, conn: _MyConn):
        self.conn = conn

    def acquire(self):
        return _AsyncCtx(self.conn)


@pytest.mark.asyncio
async def test_postgres_hitl_answer_ledger_crud() -> None:
    cp = PostgresCheckpointer("postgresql://unused")
    conn = _PgConn(row={"answer": {"decision": "approve"}})
    cp._pool = _PgPool(conn)

    await cp.save_hitl_answer("t-1", "q-1", {"decision": "approve"}, run_id="r-1")
    assert await cp.load_hitl_answer("t-1", "q-1", run_id="r-1") == {
        "decision": "approve"
    }
    await cp.clear_hitl_answers("t-1", ["q-1"], run_id="r-1")
    await cp.clear_hitl_answers("t-1", run_id="r-1")

    sqls = [sql for sql, _args in conn.exec_calls]
    assert any("cubepi_hitl_answers" in sql for sql in sqls)
    assert any("ON CONFLICT (thread_id, run_id, question_id)" in sql for sql in sqls)
    assert any("DELETE FROM cubepi_hitl_answers" in sql for sql in sqls)


@pytest.mark.asyncio
async def test_postgres_hitl_answer_ledger_load_from_json_string() -> None:
    cp = PostgresCheckpointer("postgresql://unused")
    conn = _PgConn(row={"answer": json.dumps({"decision": "deny"})})
    cp._pool = _PgPool(conn)

    assert await cp.load_hitl_answer("t-1", "q-1", run_id="r-1") == {"decision": "deny"}
    assert await cp.load_hitl_answer("t-1", "q-missing", run_id="r-1") == {
        "decision": "deny"
    }


@pytest.mark.asyncio
async def test_postgres_hitl_answer_ledger_returns_none_when_missing() -> None:
    cp = PostgresCheckpointer("postgresql://unused")
    conn = _PgConn(row=None)
    cp._pool = _PgPool(conn)

    assert await cp.load_hitl_answer("t-1", "q-1", run_id="r-1") is None


@pytest.mark.asyncio
async def test_mysql_hitl_answer_ledger_crud() -> None:
    cp = MySQLCheckpointer("mysql://unused")
    conn = _MyConn(row=({"decision": "approve"},))
    cp._pool = _MyPool(conn)

    await cp.save_hitl_answer("t-1", "q-1", {"decision": "approve"}, run_id="r-1")
    assert await cp.load_hitl_answer("t-1", "q-1", run_id="r-1") == {
        "decision": "approve"
    }
    await cp.clear_hitl_answers("t-1", ["q-1"], run_id="r-1")
    await cp.clear_hitl_answers("t-1", run_id="r-1")

    sqls = [sql for sql, _params in conn.exec_calls]
    assert any("cubepi_hitl_answers" in sql for sql in sqls)
    assert any("ON DUPLICATE KEY UPDATE" in sql for sql in sqls)
    assert any("DELETE FROM cubepi_hitl_answers" in sql for sql in sqls)


@pytest.mark.asyncio
async def test_mysql_hitl_answer_ledger_load_from_json_string() -> None:
    cp = MySQLCheckpointer("mysql://unused")
    conn = _MyConn(row=("{}".format(json.dumps({"decision": "deny"})),))
    cp._pool = _MyPool(conn)

    assert await cp.load_hitl_answer("t-1", "q-1", run_id="r-1") == {"decision": "deny"}


@pytest.mark.asyncio
async def test_mysql_hitl_answer_ledger_returns_none_when_missing() -> None:
    cp = MySQLCheckpointer("mysql://unused")
    conn = _MyConn(row=None)
    cp._pool = _MyPool(conn)

    assert await cp.load_hitl_answer("t-1", "q-1", run_id="r-1") is None
