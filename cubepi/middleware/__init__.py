from __future__ import annotations

from typing import Any

from cubepi.middleware.base import Middleware, TurnAction, compose_middleware

__all__ = [
    "CompactionMiddleware",
    "CompactionState",
    "Middleware",
    "SubagentMiddleware",
    "SubagentRequest",
    "SubagentResult",
    "SubagentSpec",
    "Todo",
    "TodoGuardBlocked",
    "TodoGuardType",
    "TodoListMiddleware",
    "TurnAction",
    "WriteTodosInput",
    "WRITE_TODOS_SYSTEM_PROMPT",
    "WRITE_TODOS_TOOL_DESCRIPTION",
    "compose_middleware",
]

_LAZY = {
    "CompactionMiddleware": ("cubepi.middleware.compaction", "CompactionMiddleware"),
    "CompactionState": ("cubepi.middleware.compaction", "CompactionState"),
    "SubagentMiddleware": ("cubepi.middleware.subagents", "SubagentMiddleware"),
    "SubagentRequest": ("cubepi.middleware.subagents", "SubagentRequest"),
    "SubagentResult": ("cubepi.middleware.subagents", "SubagentResult"),
    "SubagentSpec": ("cubepi.middleware.subagents", "SubagentSpec"),
    "Todo": ("cubepi.middleware.todo", "Todo"),
    "TodoGuardBlocked": ("cubepi.middleware.todo", "TodoGuardBlocked"),
    "TodoGuardType": ("cubepi.middleware.todo", "TodoGuardType"),
    "TodoListMiddleware": ("cubepi.middleware.todo", "TodoListMiddleware"),
    "WriteTodosInput": ("cubepi.middleware.todo", "WriteTodosInput"),
    "WRITE_TODOS_SYSTEM_PROMPT": ("cubepi.middleware.todo", "WRITE_TODOS_SYSTEM_PROMPT"),
    "WRITE_TODOS_TOOL_DESCRIPTION": ("cubepi.middleware.todo", "WRITE_TODOS_TOOL_DESCRIPTION"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, attr)
