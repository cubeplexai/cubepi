"""Tests for cubepi.checkpointer public module __getattr__ lazy imports."""

import pytest


def test_postgres_checkpointer_lazy_import_via_public_module() -> None:
    """`from cubeloop.checkpointer import PostgresCheckpointer` hits the lazy path."""
    import cubeloop.checkpointer as pkg

    PG = pkg.PostgresCheckpointer  # triggers __getattr__
    from cubeloop.checkpointer.postgres.checkpointer import (
        PostgresCheckpointer as Direct,
    )

    assert PG is Direct


def test_sqlite_checkpointer_lazy_import_via_public_module() -> None:
    import cubeloop.checkpointer as pkg

    SQ = pkg.SQLiteCheckpointer
    from cubeloop.checkpointer.sqlite import SQLiteCheckpointer as Direct

    assert SQ is Direct


def test_unknown_attribute_raises_attribute_error() -> None:
    import cubeloop.checkpointer as pkg

    with pytest.raises(AttributeError, match="has no attribute 'NotAThing'"):
        _ = pkg.NotAThing
