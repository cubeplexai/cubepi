"""Event emission helper shared by the agent loop and tool executor."""

import asyncio
from collections.abc import Callable


async def emit_event(emit_fn: Callable, event: object) -> None:
    result = emit_fn(event)
    if asyncio.iscoroutine(result):
        await result
