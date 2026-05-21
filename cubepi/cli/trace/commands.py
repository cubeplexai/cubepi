"""argparse wiring for `cubepi trace`."""

from __future__ import annotations

import argparse
from pathlib import Path

from cubepi.cli.trace import loader, render, stats
from cubepi.cli.trace.follow import follow_run
from cubepi.cli.trace.model import build_forest


def register(subparsers: "argparse._SubParsersAction") -> None:
    trace = subparsers.add_parser("trace", help="inspect cubepi JSONL traces")
    trace_sub = trace.add_subparsers(dest="trace_cmd", required=True)

    p_ls = trace_sub.add_parser("ls", help="list recent runs")
    _add_dir(p_ls)
    p_ls.add_argument("-n", type=int, default=20, help="max runs to show")
    p_ls.set_defaults(handler=cmd_ls)

    p_view = trace_sub.add_parser("view", help="render a run as a tree")
    p_view.add_argument("run", help="run id or path to a .jsonl file")
    _add_dir(p_view)
    p_view.add_argument(
        "-v", "--verbose", action="store_true", help="expand all span attributes"
    )
    p_view.add_argument(
        "--content", action="store_true", help="expand gen_ai content messages"
    )
    p_view.set_defaults(handler=cmd_view)

    p_follow = trace_sub.add_parser("follow", help="stream spans as they complete")
    p_follow.add_argument("run", help="run id or path to a .jsonl file")
    _add_dir(p_follow)
    p_follow.add_argument(
        "--interval", type=float, default=0.5, help="poll interval seconds"
    )
    p_follow.add_argument(
        "--timeout", type=float, default=None, help="exit after this many idle seconds"
    )
    p_follow.set_defaults(handler=cmd_follow)

    p_stats = trace_sub.add_parser("stats", help="aggregate stats across runs")
    p_stats.add_argument("runs", nargs="*", help="run ids (default: whole dir)")
    _add_dir(p_stats)
    p_stats.add_argument("--by", choices=("model", "tool"), default="model")
    p_stats.add_argument("--since", default=None, help="YYYY-MM-DD lower bound")
    p_stats.set_defaults(handler=cmd_stats)


def _add_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dir",
        default=str(loader.DEFAULT_DIR),
        help="traces directory (default: ./cubepi-traces)",
    )


def _emit_skipped(skipped: int) -> None:
    if skipped:
        print(f"({skipped} lines skipped (malformed))")


def cmd_ls(args: argparse.Namespace) -> int:
    directory = Path(args.dir)
    if not directory.exists():
        print(f"no traces directory at {directory}")
        return 1
    runs = loader.list_runs(directory, limit=args.n)
    if not runs:
        print(f"no runs found under {directory}")
        return 1
    render.render_runs(runs)
    return 0


def cmd_view(args: argparse.Namespace) -> int:
    directory = Path(args.dir)
    try:
        files = loader.resolve_run(args.run, directory)
    except loader.RunResolutionError as exc:
        print(str(exc))
        return 1
    spans, skipped = loader.load_run(files)
    forest = build_forest(spans)
    render.render_tree(forest, verbose=args.verbose, content=args.content)
    _emit_skipped(skipped)
    return 0


def cmd_follow(args: argparse.Namespace) -> int:
    directory = Path(args.dir)
    try:
        loader.resolve_run(args.run, directory)  # validate up front
    except loader.RunResolutionError as exc:
        print(str(exc))
        return 1

    def resolver() -> list[Path]:
        # Re-glob each poll so a cross-midnight run's next-day file is picked up.
        try:
            return loader.resolve_run(args.run, directory)
        except loader.RunResolutionError:
            return []

    follow_run(resolver, interval=args.interval, timeout=args.timeout)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    directory = Path(args.dir)
    if args.runs:
        files: list[Path] = []
        for run in args.runs:
            try:
                files.extend(loader.resolve_run(run, directory))
            except loader.RunResolutionError as exc:
                print(str(exc))
                return 1
    else:
        files = sorted(directory.glob("*/*.jsonl"))
        if args.since:
            files = [f for f in files if f.parent.name >= args.since]
    if not files:
        print(f"no runs found under {directory}")
        return 1
    spans, skipped = loader.load_run(files)
    rows = stats.aggregate(spans, by=args.by)
    render.render_stats(rows, by=args.by)
    _emit_skipped(skipped)
    return 0
