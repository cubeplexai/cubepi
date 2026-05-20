"""schema constants must be importable without the opentelemetry SDK."""

from __future__ import annotations

import subprocess
import sys

import pytest

import cubepi.tracing


def test_schema_importable_without_opentelemetry():
    # Run in a subprocess with opentelemetry hidden, so the lazy __init__
    # is exercised exactly as a trace-cli-only install would see it.
    code = (
        "import builtins, sys\n"
        "real_import = builtins.__import__\n"
        "def fake(name, *a, **k):\n"
        "    if name == 'opentelemetry' or name.startswith('opentelemetry.'):\n"
        "        raise ImportError('hidden')\n"
        "    return real_import(name, *a, **k)\n"
        "builtins.__import__ = fake\n"
        "from cubepi.tracing import schema\n"
        "assert schema.CUBEPI_RUN_ID == 'cubepi.run_id'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        cubepi.tracing.does_not_exist


def test_dir_lists_public_names():
    names = dir(cubepi.tracing)
    assert "Tracer" in names
    assert "JsonlSpanExporter" in names
