from __future__ import annotations

import json

from cubepi.deferred.types import DeferredToolGroup

DEFAULT_CATALOG_HEADER = (
    "# Deferred tool groups\n"
    "\n"
    "These tool groups are available but not yet loaded. Call `load_tools(group_id)`\n"
    "to load a group's tools for the rest of this conversation.\n"
    "You can also call `load_tools(group_id, tool_names=[...])` to load specific tools only."
)


DEFAULT_DISPATCH_CATALOG_HEADER = (
    "# Deferred tool groups\n"
    "\n"
    "These tool groups are available but not yet loaded. Call `load_tools(group_id)`\n"
    "to get their full schemas, then invoke them via\n"
    "`deferred_tool_call(tool_name=..., arguments=...)`.\n"
    "If you already know the right arguments from the names below, you may call\n"
    "`deferred_tool_call` directly — the tool loads on demand."
)


ToolSchema = dict[str, object]


def render_static_catalog(
    *,
    groups: list[DeferredToolGroup],
    header: str = DEFAULT_DISPATCH_CATALOG_HEADER,
) -> str:
    """Dispatch-mode catalog: byte-stable, independent of expansion state."""
    lines: list[str] = []
    for group in sorted(groups, key=lambda g: g.group_id):
        count = len(group.tool_names)
        lines.append(
            f"- `{group.group_id}` — {group.display_name}: "
            f"{group.description} ({count} tools)"
        )
        lines.append(f"  {', '.join(group.tool_names)}")
    if not lines:
        return ""
    return header + "\n\n" + "\n".join(lines)


def render_catalog(
    *,
    groups: list[DeferredToolGroup],
    expanded: dict[str, list[str] | None],
    header: str = DEFAULT_CATALOG_HEADER,
) -> str:
    lines: list[str] = []

    for group in sorted(groups, key=lambda g: g.group_id):
        expanded_names = expanded.get(group.group_id)

        if expanded_names is None and group.group_id in expanded:
            # Fully expanded (None sentinel) — omit from catalog
            continue

        if expanded_names is not None:
            expanded_set = set(expanded_names)
            remaining = [n for n in group.tool_names if n not in expanded_set]
        else:
            remaining = list(group.tool_names)

        if not remaining:
            continue

        count = len(remaining)
        count_label = (
            f"{count} remaining tools"
            if group.group_id in expanded
            else f"{count} tools"
        )
        lines.append(
            f"- `{group.group_id}` — {group.display_name}: {group.description} ({count_label})"
        )
        lines.append(f"  {', '.join(remaining)}")

    if not lines:
        return ""

    return header + "\n\n" + "\n".join(lines)


def render_expanded_schemas(
    *,
    expanded_schemas: list[tuple[str, list[ToolSchema]]],
) -> str:
    if not expanded_schemas:
        return ""

    sections: list[str] = []
    for group_id, tool_defs in expanded_schemas:
        tool_lines: list[str] = []
        for td in tool_defs:
            name = td.get("name", "")
            desc = td.get("description", "")
            params = td.get("parameters", {})
            params_json = json.dumps(params, sort_keys=True, ensure_ascii=False)
            tool_lines.append(f"- **{name}**: {desc}")
            tool_lines.append(f"  Parameters: {params_json}")
        sections.append(f"## {group_id}\n\n" + "\n".join(tool_lines))

    return "# Expanded tool groups\n\n" + "\n\n".join(sections)
