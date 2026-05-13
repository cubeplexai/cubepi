"""Tests for cubepi.checkpointer public module __getattr__ lazy imports."""

import pytest


def test_postgres_checkpointer_lazy_import_via_public_module() -> None:
    """`from cubepi.checkpointer import PostgresCheckpointer` hits the lazy path."""
    import cubepi.checkpointer as pkg

    PG = pkg.PostgresCheckpointer  # triggers __getattr__
    from cubepi.checkpointer.postgres.checkpointer import (
        PostgresCheckpointer as Direct,
    )

    assert PG is Direct


def test_sqlite_checkpointer_lazy_import_via_public_module() -> None:
    import cubepi.checkpointer as pkg

    SQ = pkg.SQLiteCheckpointer
    from cubepi.checkpointer.sqlite import SQLiteCheckpointer as Direct

    assert SQ is Direct


def test_unknown_attribute_raises_attribute_error() -> None:
    import cubepi.checkpointer as pkg

    with pytest.raises(AttributeError, match="has no attribute 'NotAThing'"):
        _ = pkg.NotAThing
