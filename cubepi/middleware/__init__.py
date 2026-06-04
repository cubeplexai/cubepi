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
    "TurnAction",
    "compose_middleware",
]

_LAZY = {
    "CompactionMiddleware": ("cubepi.middleware.compaction", "CompactionMiddleware"),
    "CompactionState": ("cubepi.middleware.compaction", "CompactionState"),
    "SubagentMiddleware": ("cubepi.middleware.subagents", "SubagentMiddleware"),
    "SubagentRequest": ("cubepi.middleware.subagents", "SubagentRequest"),
    "SubagentResult": ("cubepi.middleware.subagents", "SubagentResult"),
    "SubagentSpec": ("cubepi.middleware.subagents", "SubagentSpec"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, attr)
