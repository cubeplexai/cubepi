"""argparse wiring for `cubepi trace` (stub — completed in Task 6)."""
from __future__ import annotations

import argparse


def register(subparsers: "argparse._SubParsersAction") -> None:
    trace = subparsers.add_parser("trace", help="inspect cubepi JSONL traces")
    trace_sub = trace.add_subparsers(dest="trace_cmd", required=True)
    for name, help_text in (
        ("ls", "list recent runs"),
        ("view", "render a run as a tree"),
        ("follow", "stream a run's spans as they complete"),
        ("stats", "aggregate stats across runs"),
    ):
        trace_sub.add_parser(name, help=help_text)
