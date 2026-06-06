"""End-to-end test fixtures.

Re-exports the Postgres + MySQL DSN fixtures from
``tests/checkpointer/conftest.py`` so the cross-backend e2e suite can use
them without duplicating fixture setup.

We avoid ``pytest_plugins`` here because pytest deprecates declaring it in
non-top-level conftest files (it would affect the whole suite, not just
``tests/e2e``). Direct imports are scoped to this package only.
"""

from __future__ import annotations

from tests.checkpointer.conftest import (  # noqa: F401 — fixtures must be re-exported
    _mysql_available,
    _pg_available,
    clean_db,
    clean_mysql_db,
    mysql_dsn,
    mysql_v4_dsn,
    pg_dsn,
    pg_v4_dsn,
)
