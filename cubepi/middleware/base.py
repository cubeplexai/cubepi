from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal

import asyncio

from cubepi.agent.types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentContext,
    BeforeToolCallContext,
    BeforeToolCallResult,
)
from cubepi.providers.base import AssistantMessage, Message, Model, Provider
from cubepi.types import JsonObject, StructuredObject


@dataclass
class TurnAction:
    """Directs the agent loop's next step after a model response.

    Composition (chain): each middleware sees previous middleware's
    TurnAction. Last middleware's value wins for response and decision.
    inject_messages concatenates across the chain.
    """

    response: AssistantMessage | None = None
    inject_messages: list[Message] = field(default_factory=list)
    decision: Literal["natural", "stop", "loop_to_model"] = "natural"


class Middleware:
    async def transform_context(
        self,
        messages: list[Message],
        *,
        ctx: AgentContext,
        signal: asyncio.Event | None = None,
    ) -> list[Message]:
        raise NotImplementedError

    async def convert_to_llm(
        self, messages: list[Message], *, ctx: AgentContext
    ) -> list[Message]:
        raise NotImplementedError

    async def before_tool_call(
        self,
        ctx: BeforeToolCallContext,
        *,
        signal: asyncio.Event | None = None,
    ) -> BeforeToolCallResult | None:
        raise NotImplementedError

    async def after_tool_call(
        self,
        ctx: AfterToolCallContext,
        *,
        signal: asyncio.Event | None = None,
    ) -> AfterToolCallResult | None:
        raise NotImplementedError

    async def transform_system_prompt(
        self,
        system_prompt: str,
        *,
        ctx: AgentContext,
        signal: asyncio.Event | None = None,
    ) -> str:
        raise NotImplementedError

    async def should_stop_after_turn(self, ctx: AgentContext) -> bool:
        raise NotImplementedError

    async def after_model_response(
        self,
        response: AssistantMessage,
        ctx: AgentContext,
        *,
        signal: asyncio.Event | None = None,
    ) -> TurnAction | None:
        raise NotImplementedError

    async def on_run_end(
        self,
        ctx: AgentContext,
        *,
        signal: asyncio.Event | None = None,
    ) -> list[Message] | None:
        raise NotImplementedError

    def extra_llm_calls(self) -> Iterable[tuple[Provider, Model]]:
        """Declare LLM calls this middleware drives outside the agent's main
        provider/model.

        Each pair is ``(provider, model)``. ``cubepi.tracing.Recorder`` uses
        these to:

        * Subscribe listeners on any provider the recorder isn't already
          watching, so the resulting calls show up in the trace tree
          alongside the agent's own chat spans.
        * Identify middleware-owned calls by ``(model.provider, model.id)``
          so they don't overwrite the root ``invoke_agent`` span's
          attribution (provider name, system prompt hash, tool list). This
          model-based gate is what handles the common "reuse one provider
          client, swap the model" pattern — listener identity alone would
          attribute the middleware's first call to the agent.

        Default is empty — middlewares that do not call any LLM directly
        need not override.
        """
        return ()


def _has_method(middleware: Middleware, name: str) -> bool:
    method = getattr(type(middleware), name, None)
    base_method = getattr(Middleware, name, None)
    return method is not None and method is not base_method


def compose_middleware(middlewares: list[Middleware]) -> dict[str, Callable]:
    hooks: dict[str, Callable] = {}

    transform_chain = [m for m in middlewares if _has_method(m, "transform_context")]
    if transform_chain:

        async def composed_transform(messages, *, ctx, signal=None):
            result = messages
            for mw in transform_chain:
                result = await mw.transform_context(result, ctx=ctx, signal=signal)
            return result

        hooks["transform_context"] = composed_transform

    convert_impls = [m for m in middlewares if _has_method(m, "convert_to_llm")]
    if convert_impls:
        last = convert_impls[-1]

        async def composed_convert(messages, *, ctx):
            return await last.convert_to_llm(messages, ctx=ctx)

        hooks["convert_to_llm"] = composed_convert

    before_chain = [m for m in middlewares if _has_method(m, "before_tool_call")]
    if before_chain:

        def _rebuild_ctx_with_args(
            ctx: BeforeToolCallContext, new_args: JsonObject
        ) -> BeforeToolCallContext:
            from dataclasses import replace

            return replace(ctx, args=new_args)

        async def composed_before(ctx, *, signal=None):
            accumulated_hitl: StructuredObject = {}
            edited_args: JsonObject | None = None
            deny_reason: str | None = None
            block_reason: str | None = None
            blocked = False

            cur_ctx = ctx
            for mw in before_chain:
                if edited_args is not None:
                    cur_ctx = _rebuild_ctx_with_args(cur_ctx, edited_args)
                result = await mw.before_tool_call(cur_ctx, signal=signal)
                if result is None:
                    continue
                if result.hitl_trace:
                    if accumulated_hitl:
                        accumulated_hitl.setdefault("_chain", []).append(
                            {k: v for k, v in accumulated_hitl.items() if k != "_chain"}
                        )
                        # remove already-archived keys before updating with new
                        for k in list(accumulated_hitl.keys()):
                            if k != "_chain":
                                accumulated_hitl.pop(k)
                    accumulated_hitl.update(result.hitl_trace)
                if result.edited_args is not None:
                    edited_args = result.edited_args
                if result.block:
                    blocked = True
                    block_reason = result.reason or block_reason
                    deny_reason = result.deny_reason or deny_reason
                    break

            if not blocked and edited_args is None and not accumulated_hitl:
                return None
            return BeforeToolCallResult(
                block=blocked,
                reason=block_reason,
                deny_reason=deny_reason,
                edited_args=edited_args,
                hitl_trace=accumulated_hitl or None,
            )

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

        async def composed_sp(system_prompt, *, ctx, signal=None):
            result = system_prompt
            for mw in sp_chain:
                result = await mw.transform_system_prompt(
                    result, ctx=ctx, signal=signal
                )
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

    amr_chain = [m for m in middlewares if _has_method(m, "after_model_response")]
    if amr_chain:

        async def composed_amr(response, ctx, *, signal=None):
            current_response = response
            all_inject: list[Message] = []
            last_decision: Literal["natural", "stop", "loop_to_model"] = "natural"
            for mw in amr_chain:
                result = await mw.after_model_response(
                    current_response, ctx, signal=signal
                )
                if result is None:
                    continue
                if result.response is not None:
                    current_response = result.response
                if result.inject_messages:
                    all_inject.extend(result.inject_messages)
                last_decision = result.decision
            return TurnAction(
                response=current_response,
                inject_messages=all_inject,
                decision=last_decision,
            )

        hooks["after_model_response"] = composed_amr

    ore_chain = [m for m in middlewares if _has_method(m, "on_run_end")]
    if ore_chain:

        async def composed_ore(ctx, *, signal=None):
            all_inject: list[Message] = []
            for mw in ore_chain:
                result = await mw.on_run_end(ctx, signal=signal)
                if result:
                    all_inject.extend(result)
            return all_inject or None

        hooks["on_run_end"] = composed_ore

    return hooks
