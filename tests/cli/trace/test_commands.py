from __future__ import annotations

import json
from pathlib import Path

from cubepi.cli.__main__ import main


def _write_run(directory: Path):
    f = directory / "2026-05-20" / "r1.jsonl"
    f.parent.mkdir(parents=True)
    rows = [
        {
            "name": "invoke_agent",
            "context": {"trace_id": "0xt", "span_id": "0x1"},
            "parent_id": None,
            "start_time": "2026-05-20T00:00:00Z",
            "end_time": "2026-05-20T00:00:02Z",
            "status": {"status_code": "UNSET"},
            "attributes": {
                "cubepi.run_id": "r1",
                "gen_ai.operation.name": "invoke_agent",
            },
        },
        {
            "name": "chat gpt-x",
            "context": {"trace_id": "0xt", "span_id": "0x2"},
            "parent_id": "0x1",
            "start_time": "2026-05-20T00:00:00Z",
            "end_time": "2026-05-20T00:00:01Z",
            "status": {"status_code": "UNSET"},
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "gpt-x",
                "gen_ai.usage.input_tokens": 10,
                "gen_ai.usage.output_tokens": 5,
            },
        },
    ]
    with f.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_ls_lists_run(tmp_path, capsys):
    _write_run(tmp_path)
    rc = main(["trace", "ls", "--dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "r1" in out


def test_view_renders_tree(tmp_path, capsys):
    _write_run(tmp_path)
    rc = main(["trace", "view", "r1", "--dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "invoke_agent" in out
    assert "chat" in out


def test_stats_by_model(tmp_path, capsys):
    _write_run(tmp_path)
    rc = main(["trace", "stats", "--dir", str(tmp_path), "--by", "model"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "gpt-x" in out


def _write_run_with_meta(directory: Path, stem: str, conversation_id: str):
    f = directory / "2026-05-20" / f"{stem}.jsonl"
    f.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "name": "invoke_agent",
            "context": {"trace_id": f"0x{stem}", "span_id": "0x1"},
            "parent_id": None,
            "start_time": "2026-05-20T00:00:00Z",
            "end_time": "2026-05-20T00:00:02Z",
            "status": {"status_code": "UNSET"},
            "attributes": {
                "cubepi.run_id": stem,
                "gen_ai.operation.name": "invoke_agent",
                "cubepi.metadata.conversation_id": conversation_id,
            },
        },
        {
            "name": "chat gpt-x",
            "context": {"trace_id": f"0x{stem}", "span_id": "0x2"},
            "parent_id": "0x1",
            "start_time": "2026-05-20T00:00:00Z",
            "end_time": "2026-05-20T00:00:01Z",
            "status": {"status_code": "UNSET"},
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "gpt-x",
                "gen_ai.usage.input_tokens": 10,
                "gen_ai.usage.output_tokens": 5,
            },
        },
    ]
    with f.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_ls_meta_bad_format(tmp_path, capsys):
    _write_run(tmp_path)
    rc = main(["trace", "ls", "--dir", str(tmp_path), "--meta", "nokey"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "KEY=VALUE" in out


def test_ls_meta_no_match(tmp_path, capsys):
    _write_run(tmp_path)  # has no cubepi.metadata.*
    rc = main(
        ["trace", "ls", "--dir", str(tmp_path), "--meta", "conversation_id=conv_X"]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "no runs found" in out


def test_stats_meta_filters(tmp_path, capsys):
    _write_run_with_meta(tmp_path, "tA", "conv_A")
    # matching meta -> the model shows up
    rc = main(
        ["trace", "stats", "--dir", str(tmp_path), "--meta", "conversation_id=conv_A"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "gpt-x" in out
    # non-matching meta -> filtered out, model absent
    rc = main(
        ["trace", "stats", "--dir", str(tmp_path), "--meta", "conversation_id=zzz"]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "gpt-x" not in out
