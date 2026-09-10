import asyncpg
import pytest

from cubeloop.checkpointer.postgres.checkpointer import _schema_mismatch_hint
from cubeloop.checkpointer.postgres.models import EXPECTED_SCHEMA_VERSION


def test_expected_schema_version_is_6():
    assert EXPECTED_SCHEMA_VERSION == 6


def test_schema_mismatch_hint_names_the_v5_to_v6_helper():
    hint = _schema_mismatch_hint(actual=5, expected=6)
    assert "upgrade_v5_to_v6_op()" in hint
    assert "add_run_id_column_op" not in hint


def test_schema_mismatch_hint_names_every_step_across_multiple_versions():
    hint = _schema_mismatch_hint(actual=3, expected=6)
    assert "upgrade_v3_to_v4_op()" in hint
    assert "upgrade_v4_to_v5_op()" in hint
    assert "upgrade_v5_to_v6_op()" in hint


@pytest.mark.asyncio
async def test_cubepi_runs_table_present(pg_v4_dsn):
    conn = await asyncpg.connect(pg_v4_dsn)
    try:
        rows = await conn.fetch(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'cubeloop_runs'"
        )
        cols = {r["column_name"] for r in rows}
        assert {
            "thread_id",
            "run_id",
            "claimed_at",
            "completed_at",
            "completion_seq",
        } <= cols
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_cubepi_messages_has_run_id_column(pg_v4_dsn):
    conn = await asyncpg.connect(pg_v4_dsn)
    try:
        row = await conn.fetchrow(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'cubeloop_messages' AND column_name = 'run_id'"
        )
        assert row is not None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_cubepi_hitl_answers_table_present(pg_v4_dsn):
    conn = await asyncpg.connect(pg_v4_dsn)
    try:
        rows = await conn.fetch(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'cubeloop_hitl_answers'"
        )
        cols = {r["column_name"] for r in rows}
        assert {
            "thread_id",
            "run_id",
            "question_id",
            "answer",
            "answered_at",
        } <= cols
    finally:
        await conn.close()
