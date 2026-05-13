"""Postgres test fixtures for D1.3+ tests."""

import os
import secrets

import asyncpg
import pytest
import pytest_asyncio


@pytest.fixture(scope="session")
def pg_dsn() -> str:
    """Postgres DSN for tests. Override via CUBEPI_TEST_PG_DSN env."""
    return os.environ.get(
        "CUBEPI_TEST_PG_DSN",
        "postgresql://postgres:postgres@localhost:5432/postgres",
    )


@pytest_asyncio.fixture(scope="session")
async def _pg_available(pg_dsn: str) -> bool:
    """Quick probe — connection refused → skip Postgres tests."""
    try:
        conn = await asyncpg.connect(pg_dsn, timeout=2.0)
        await conn.close()
        return True
    except (asyncpg.PostgresError, OSError, ConnectionError):
        return False


@pytest_asyncio.fixture
async def clean_db(pg_dsn: str, _pg_available: bool):
    """Create a fresh database for each test; drop after."""
    if not _pg_available:
        pytest.skip(
            "Postgres not available for E2E tests; set CUBEPI_TEST_PG_DSN "
            "to a working DSN to enable."
        )

    db_name = f"cubepi_test_{secrets.token_hex(6)}"
    admin = await asyncpg.connect(pg_dsn)
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()

    # DSN for the new DB
    base = pg_dsn.rsplit("/", 1)[0]
    test_dsn = f"{base}/{db_name}"
    yield test_dsn

    admin = await asyncpg.connect(pg_dsn)
    try:
        await admin.execute(
            f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            f"WHERE datname = '{db_name}'"
        )
        await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}"')
    finally:
        await admin.close()
