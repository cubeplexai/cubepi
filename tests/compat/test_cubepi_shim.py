"""Transitional cubepi shim — import alias, warning, lazy tracing."""

from __future__ import annotations

import subprocess
import sys
import warnings

import pytest


def test_import_cubepi_warns_and_reexports() -> None:
    import cubeloop

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import cubepi
    assert cubepi.Agent is cubeloop.Agent
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_deep_import_aliases_provider() -> None:
    from cubepi.providers.anthropic import AnthropicProvider
    from cubeloop.providers.anthropic import AnthropicProvider as Direct

    assert AnthropicProvider is Direct


def test_deep_import_checkpointer() -> None:
    from cubepi.checkpointer.postgres import PostgresCheckpointer
    from cubeloop.checkpointer.postgres import PostgresCheckpointer as Direct

    assert PostgresCheckpointer is Direct


def test_tracing_schema_lazy_without_otel() -> None:
    code = (
        "import builtins, sys\n"
        "real_import = builtins.__import__\n"
        "def fake(name, *a, **k):\n"
        "    if name == 'opentelemetry' or name.startswith('opentelemetry.'):\n"
        "        raise ImportError('hidden')\n"
        "    return real_import(name, *a, **k)\n"
        "builtins.__import__ = fake\n"
        "from cubepi.tracing import schema\n"
        "assert schema.CUBELOOP_RUN_ID == 'cubeloop.run_id'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_python_m_cubepi_trace_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cubepi", "trace", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "renamed to 'cubeloop'" in result.stderr
    assert "trace" in result.stdout


def test_python_m_cubepi_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cubepi.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "trace" in result.stdout


def test_second_import_does_not_reprint(capsys: pytest.CaptureFixture[str]) -> None:
    import cubepi  # noqa: F401

    capsys.readouterr()
    import importlib

    importlib.reload(cubepi)
    captured = capsys.readouterr()
    assert "renamed to 'cubeloop'" not in captured.err
