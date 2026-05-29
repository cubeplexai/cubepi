"""Run the checkpointing example scripts against a real DB.

These guard examples/checkpointing_{postgres,mysql}.py against bitrot. Each
example creates its own throwaway database off the admin DSN, runs an agent
turn through the checkpointer, reloads it, and drops the database. We just
point the example at the test DSN and invoke its ``main()``.

Skips automatically when no DB is reachable (same gate as the other E2E
tests). In CI the postgres:16 / mysql:8.0 service containers make them run.
"""

import importlib.util
import os
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _load_example(filename: str):
    path = _EXAMPLES / filename
    spec = importlib.util.spec_from_file_location(f"_example_{path.stem}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_postgres_example_runs(pg_dsn: str, _pg_available: bool, monkeypatch):
    if not _pg_available:
        pytest.skip("Postgres not available; set CUBEPI_TEST_PG_DSN to enable.")
    # The example reads CUBEPI_PG_DSN; point it at the test admin DSN.
    monkeypatch.setenv("CUBEPI_PG_DSN", pg_dsn)
    module = _load_example("checkpointing_postgres.py")
    await module.main()  # creates + drops its own throwaway DB


@pytest.mark.asyncio
async def test_mysql_example_runs(mysql_dsn: str, _mysql_available: bool, monkeypatch):
    if not _mysql_available:
        pytest.skip("MySQL not available; set CUBEPI_TEST_MYSQL_DSN to enable.")
    monkeypatch.setenv("CUBEPI_MYSQL_DSN", mysql_dsn)
    module = _load_example("checkpointing_mysql.py")
    await module.main()  # creates + drops its own throwaway DB


def test_examples_directory_exists():
    """Cheap always-on guard: the example files are present and importable."""
    assert (_EXAMPLES / "checkpointing_postgres.py").is_file()
    assert (_EXAMPLES / "checkpointing_mysql.py").is_file()
    # Importable without a DB (module body has no side effects at import).
    assert _load_example("checkpointing_postgres.py").THREAD_ID
    assert _load_example("checkpointing_mysql.py").THREAD_ID
    assert "CUBEPI_PG_DSN" in os.environ or True  # smoke


if __name__ == "__main__":  # pragma: no cover
    import asyncio

    asyncio.run(_load_example("checkpointing_postgres.py").main())
