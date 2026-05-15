"""Generate Docusaurus-compatible MDX from cubepi public API via griffe."""
from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import griffe
from griffe import AliasResolutionError

MODULES = [
    ("cubepi.agent",        "Agents",        1),
    ("cubepi.providers",    "Providers",     2),
    ("cubepi.checkpointer", "Checkpointing", 3),
    ("cubepi.middleware",   "Middleware",    4),
    ("cubepi.mcp",          "MCP",           5),
    ("cubepi.utils",        "Utils",         6),
]


def _is_public(name: str, parent_all: list[str] | None) -> bool:
    if parent_all is not None:
        return name in parent_all
    return not name.startswith("_")


def collect_public_symbols(module) -> list:
    """Return resolved Class/Function objects for the module's public surface.

    Walks `module.members` first (eagerly imported symbols), then fills in any
    names in `__all__` that griffe missed — typically lazy exports surfaced via
    `__getattr__`. The lazy ones are resolved at runtime via importlib to find
    their canonical dotted path, then loaded back through griffe so the rest of
    the pipeline (signatures, docstrings, source links) works identically.
    """
    parent_all: list[str] | None = None
    if hasattr(module, "exports") and module.exports is not None:
        try:
            parent_all = list(module.exports)
        except TypeError:
            parent_all = None

    out = []
    seen: set[str] = set()
    for member_name, member in module.members.items():
        if not _is_public(member_name, parent_all):
            continue
        if getattr(member, "is_alias", False):
            try:
                member = member.final_target
            except AliasResolutionError:
                continue
        out.append(member)
        seen.add(member_name)

    if parent_all:
        missing = [n for n in parent_all if n not in seen]
        if missing:
            try:
                runtime_mod = importlib.import_module(module.path)
            except Exception as e:
                print(f"[warn] cannot import {module.path} to resolve lazy exports: {e}",
                      file=sys.stderr)
                runtime_mod = None
            if runtime_mod is not None:
                for name in missing:
                    try:
                        obj = getattr(runtime_mod, name)
                    except (AttributeError, ImportError) as e:
                        print(f"[warn] {module.path}.{name} unresolvable ({e}); skipping",
                              file=sys.stderr)
                        continue
                    real_module = getattr(obj, "__module__", None)
                    qualname = getattr(obj, "__qualname__", None) or name
                    if not real_module:
                        continue
                    try:
                        target = griffe.load(f"{real_module}.{qualname}")
                    except Exception as e:
                        print(f"[warn] griffe.load({real_module}.{qualname}) failed: {e}",
                              file=sys.stderr)
                        continue
                    out.append(target)
    return out


def render_signature(name: str, parameters: list, returns: str | None) -> str:
    parts = []
    for pname, ptype, pdefault in parameters:
        s = pname
        if ptype:
            s += f": {ptype}"
        if pdefault:
            s += f" = {pdefault}"
        parts.append(s)
    sig = f"{name}({', '.join(parts)})"
    if returns:
        sig += f" -> {returns}"
    return f"```python\n{sig}\n```"


_GOOGLE_SECTIONS = ("Args", "Arguments", "Returns", "Raises", "Yields",
                    "Example", "Examples", "Note", "Notes", "Usage")
_CODE_SECTIONS = {"Example", "Examples", "Usage"}


def _escape_mdx_braces(text: str) -> str:
    """Escape `{` and `}` so MDX does not try to parse them as JSX expressions.

    MDX treats unescaped curly braces in prose as inline JS expressions and
    passes them to acorn; Python dict literals in docstrings (e.g. {"k": "v"})
    are NOT valid expressions and crash the build. We escape every `{`/`}` that
    falls outside fenced code blocks (those are handled by the caller).
    """
    return text.replace("{", r"\{").replace("}", r"\}")


def render_docstring(text: str | None) -> str:
    if not text:
        return ""
    lines = text.strip("\n").splitlines()
    out: list[str] = []
    in_section: str | None = None
    code_buffer: list[str] | None = None  # active when collecting an Example/Usage block

    def _flush_code() -> None:
        nonlocal code_buffer
        if code_buffer is None:
            return
        if any(s.strip() for s in code_buffer):
            out.append("```python")
            out.extend(code_buffer)
            out.append("```")
            out.append("")
        code_buffer = None

    for line in lines:
        m = re.match(r"^([A-Z][a-zA-Z]+):\s*$", line.strip())
        if m and m.group(1) in _GOOGLE_SECTIONS:
            _flush_code()
            in_section = m.group(1)
            out.append("")
            out.append(f"**{in_section}**")
            out.append("")
            if in_section in _CODE_SECTIONS:
                code_buffer = []
            continue
        if in_section in _CODE_SECTIONS:
            if line.strip() == "" or line.startswith(("    ", "\t")):
                # part of the indented code block
                if code_buffer is None:
                    code_buffer = []
                # de-indent four-space prefix so the python block reads naturally
                if line.startswith("    "):
                    code_buffer.append(line[4:])
                elif line.startswith("\t"):
                    code_buffer.append(line[1:])
                else:
                    code_buffer.append(line)
                continue
            # non-indented, non-empty line ends the code block
            _flush_code()
            in_section = None
            # fall through to render as prose
        if in_section in {"Args", "Arguments"} and line.startswith(("    ", "\t")):
            stripped = line.strip()
            arg_m = re.match(r"^(\w+)\s*(\([^)]+\))?:\s*(.*)$", stripped)
            if arg_m:
                arg, _ty, desc = arg_m.groups()
                out.append(f"- `{arg}` — {_escape_mdx_braces(desc)}")
                continue
        if in_section in {"Returns", "Yields"} and line.startswith(("    ", "\t")):
            out.append(f"- {_escape_mdx_braces(line.strip())}")
            continue
        if in_section == "Raises" and line.startswith(("    ", "\t")):
            stripped = line.strip()
            raise_m = re.match(r"^(\w+):\s*(.*)$", stripped)
            if raise_m:
                exc, desc = raise_m.groups()
                out.append(f"- `{exc}` — {_escape_mdx_braces(desc)}")
                continue
        out.append(_escape_mdx_braces(line))
    _flush_code()
    return "\n".join(out).strip() + "\n"


def _params_of(symbol) -> list:
    out = []
    params = getattr(symbol, "parameters", None) or []
    for p in params:
        pname = p.name
        ptype = str(p.annotation) if getattr(p, "annotation", None) else None
        # griffe's default is a sentinel or None; treat None as "no default"
        default = getattr(p, "default", None)
        pdefault = str(default) if default is not None else None
        out.append((pname, ptype, pdefault))
    return out


def render_symbol(symbol, github_blob_root: str) -> str:
    name = symbol.name
    kind = symbol.kind.value if hasattr(symbol.kind, "value") else str(symbol.kind)
    block: list[str] = [f"### {name}", "", f"_{kind}_", ""]

    params = _params_of(symbol)
    if params or getattr(symbol, "returns", None):
        returns = str(symbol.returns) if getattr(symbol, "returns", None) else None
        block.append(render_signature(name, params, returns))
        block.append("")

    doc = symbol.docstring.value if getattr(symbol, "docstring", None) else None
    block.append(render_docstring(doc))

    fp = getattr(symbol, "filepath", None)
    ln = getattr(symbol, "lineno", None)
    if fp and ln:
        rel = Path(str(fp)).as_posix()
        # turn /abs/.../cubepi/agent/agent.py into cubepi/agent/agent.py
        if "/cubepi/" in rel:
            rel = "cubepi/" + rel.split("/cubepi/", 1)[1]
        link = f"{github_blob_root}/{rel}#L{ln}"
        block.append(f"[source]({link})")
        block.append("")

    return "\n".join(block)


def emit_module(out_path: Path, module_name: str, sidebar_position: int,
                symbols: Iterable, source_ref: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = (
        "---\n"
        f"id: {module_name.replace('.', '-')}\n"
        f"title: {module_name}\n"
        f"sidebar_position: {sidebar_position}\n"
        "hide_table_of_contents: false\n"
        "---\n\n"
    )
    body = [f"# `{module_name}`", ""]
    for sym in symbols:
        body.append(render_symbol(sym, github_blob_root=f"https://github.com/cubeplexai/cubepi/blob/{source_ref}"))
        body.append("")
    body.append("")
    body.append("<!-- GENERATED by build-api-reference.py — DO NOT EDIT -->")
    out_path.write_text(frontmatter + "\n".join(body), encoding="utf-8")


def resolve_source_ref(cli_ref: str | None) -> str:
    """Decide which git ref the [source] links should target.

    Priority: --ref CLI flag → CUBEPI_DOCS_SOURCE_REF env → "main".
    Avoids commit SHAs because they get stale (especially after squash-merge)
    and break the link for snapshotted versions.
    """
    if cli_ref:
        return cli_ref
    env = os.environ.get("CUBEPI_DOCS_SOURCE_REF")
    if env:
        return env
    return "main"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path,
                        help="Output directory, typically website/docs/api/")
    parser.add_argument("--ref", default=None,
                        help="Git ref for source links (default: main; override per snapshotted version with e.g. v0.3.0).")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    source_ref = resolve_source_ref(args.ref)
    # griffe 2.x: load returns the Module directly
    top: griffe.Module = griffe.load("cubepi")  # type: ignore[assignment]

    for mod_name, _label, position in MODULES:
        # mod_name like "cubepi.agent" — get the submodule
        short = mod_name.split(".")[-1]
        submod = top.members.get(short)
        if submod is None:
            # Fallback: load directly
            try:
                submod = griffe.load(mod_name)
            except Exception as e:
                print(f"[warn] {mod_name} not importable; skipping ({e})", file=sys.stderr)
                continue
        symbols = collect_public_symbols(submod)
        out_path = args.out / f"{mod_name.replace('.', '-')}.mdx"
        emit_module(out_path, mod_name, position, symbols, source_ref)
        print(f"[ok] wrote {out_path} ({len(symbols)} symbols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
