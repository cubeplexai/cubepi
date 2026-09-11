"""The removed ``cubepi`` name fails with actionable migration guidance."""

from __future__ import annotations

import subprocess
import shutil
import sys

import pytest


@pytest.mark.parametrize(
    "statement",
    [
        "import cubepi",
        "import cubepi.agent",
        "from cubepi.providers.anthropic import AnthropicProvider",
        "from cubepi.checkpointer.postgres import PostgresCheckpointer",
        "from cubepi.tracing import schema",
    ],
)
def test_imports_fail_with_migration_guidance(statement: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", statement], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "ImportError" in result.stderr
    assert "Install 'cubeloop'" in result.stderr
    assert "replace 'cubepi' imports with 'cubeloop'" in result.stderr
    assert "https://cubeloop.dev/docs/migration/from-cubepi" in result.stderr


@pytest.mark.parametrize(
    "arguments",
    [
        [sys.executable, "-m", "cubepi"],
        [sys.executable, "-m", "cubepi", "trace", "--help"],
        [sys.executable, "-m", "cubepi.cli", "--help"],
    ],
)
def test_module_commands_fail_with_migration_guidance(
    arguments: list[str],
) -> None:
    result = subprocess.run(arguments, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Install 'cubeloop'" in result.stderr
    assert "https://cubeloop.dev/docs/migration/from-cubepi" in result.stderr


def test_console_command_fails_with_migration_guidance() -> None:
    command = shutil.which("cubepi")
    assert command is not None
    result = subprocess.run([command], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Install 'cubeloop'" in result.stderr
    assert "https://cubeloop.dev/docs/migration/from-cubepi" in result.stderr
