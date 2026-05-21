from __future__ import annotations

import subprocess
import sys


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cubepi.cli", *args],
        capture_output=True,
        text=True,
    )


def test_top_level_help_lists_trace():
    result = _run("--help")
    assert result.returncode == 0
    assert "trace" in result.stdout


def test_trace_help_lists_subcommands():
    result = _run("trace", "--help")
    assert result.returncode == 0
    for sub in ("ls", "view", "follow", "stats"):
        assert sub in result.stdout
