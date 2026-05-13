from __future__ import annotations

from typing import Any, Callable


class Middleware:
    async def transform_context(self, messages: list, *, signal=None) -> list:
        raise NotImplementedError

    async def convert_to_llm(self, messages: list) -> list:
        raise NotImplementedError

    async def before_tool_call(self, ctx: Any, *, signal=None) -> Any:
        raise NotImplementedError

    async def after_tool_call(self, ctx: Any, *, signal=None) -> Any:
        raise NotImplementedError

    async def transform_system_prompt(
        self,
        system_prompt: str,
        *,
        signal=None,
    ) -> str:
        raise NotImplementedError

    async def should_stop_after_turn(self, ctx: Any) -> bool:
        raise NotImplementedError


def _has_method(middleware: Middleware, name: str) -> bool:
    method = getattr(type(middleware), name, None)
    base_method = getattr(Middleware, name, None)
    return method is not None and method is not base_method


def compose_middleware(middlewares: list[Middleware]) -> dict[str, Callable]:
    hooks: dict[str, Callable] = {}

    transform_chain = [m for m in middlewares if _has_method(m, "transform_context")]
    if transform_chain:

        async def composed_transform(messages, *, signal=None):
            result = messages
            for mw in transform_chain:
                result = await mw.transform_context(result, signal=signal)
            return result

        hooks["transform_context"] = composed_transform

    convert_impls = [m for m in middlewares if _has_method(m, "convert_to_llm")]
    if convert_impls:
        last = convert_impls[-1]

        async def composed_convert(messages):
            return await last.convert_to_llm(messages)

        hooks["convert_to_llm"] = composed_convert

    before_chain = [m for m in middlewares if _has_method(m, "before_tool_call")]
    if before_chain:

        async def composed_before(ctx, *, signal=None):
            for mw in before_chain:
                result = await mw.before_tool_call(ctx, signal=signal)
                if result and getattr(result, "block", False):
                    return result
            return None

        hooks["before_tool_call"] = composed_before

    after_chain = [m for m in middlewares if _has_method(m, "after_tool_call")]
    if after_chain:

        async def composed_after(ctx, *, signal=None):
            last_result = None
            for mw in after_chain:
                result = await mw.after_tool_call(ctx, signal=signal)
                if result is not None:
                    last_result = result
            return last_result

        hooks["after_tool_call"] = composed_after

    sp_chain = [m for m in middlewares if _has_method(m, "transform_system_prompt")]
    if sp_chain:

        async def composed_sp(system_prompt, *, signal=None):
            result = system_prompt
            for mw in sp_chain:
                result = await mw.transform_system_prompt(result, signal=signal)
            return result

        hooks["transform_system_prompt"] = composed_sp

    stop_chain = [m for m in middlewares if _has_method(m, "should_stop_after_turn")]
    if stop_chain:

        async def composed_stop(ctx):
            for mw in stop_chain:
                if await mw.should_stop_after_turn(ctx):
                    return True
            return False

        hooks["should_stop_after_turn"] = composed_stop

    return hooks
