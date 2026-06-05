"""The ``@tool`` decorator — build an :class:`AgentTool` from a plain function.

Declaring a tool by hand means writing a Pydantic params model, an ``execute``
callable with the engine's exact ``(tool_call_id, params, *, signal, on_update)``
signature, and an ``AgentTool(...)`` wrapper. ``@tool`` collapses all three: the
input schema is generated from the function signature, the engine-supplied
arguments are injected only if the function asks for them, and the return value
is normalised so a tool can simply ``return "some text"``.

```python
from cubepi import tool


@tool
async def get_weather(city: str) -> str:
    "Get the current weather for a city."
    return f"72F and sunny in {city}"
```

is equivalent to the longhand ``AgentTool(name=..., parameters=..., execute=...)``.
"""

from __future__ import annotations

import inspect
from typing import (
    Any,
    Awaitable,
    Callable,
    Literal,
    get_type_hints,
    overload,
)

from pydantic import BaseModel, create_model

from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.providers.base import Content, TextContent

# Arguments the agent loop supplies when it calls a tool (see
# cubepi/agent/tools.py). They are injected into the decorated function only if
# it declares a parameter by that name, and never appear in the input schema.
_RESERVED = ("tool_call_id", "signal", "on_update")

ToolFunc = Callable[..., Awaitable[Any]]


def _build_params_model(
    fn: Callable[..., Any],
    schema_params: list[inspect.Parameter],
    hints: dict[str, Any],
) -> type[BaseModel]:
    """Generate a Pydantic model from the tool function's schema parameters.

    Structural validation (no *args/**kwargs or positional-only parameters) is
    done by the caller over the full signature; here we only need each schema
    parameter to carry a type annotation.
    """
    fields: dict[str, Any] = {}
    for p in schema_params:
        if p.name not in hints and p.annotation is inspect.Parameter.empty:
            raise TypeError(
                f"@tool parameter {p.name!r} of {fn.__name__!r} needs a type "
                "annotation to generate its schema."
            )
        annotation = hints.get(p.name, p.annotation)
        default = ... if p.default is inspect.Parameter.empty else p.default
        fields[p.name] = (annotation, default)

    model_name = "".join(part.capitalize() for part in fn.__name__.split("_")) + "Args"
    return create_model(model_name, **fields)


def _normalize_result(result: Any, fn: Callable[..., Any]) -> AgentToolResult:
    """Accept the ergonomic return shapes and coerce to an AgentToolResult."""
    if isinstance(result, AgentToolResult):
        return result
    if isinstance(result, str):
        return AgentToolResult(content=[TextContent(text=result)])
    if isinstance(result, Content):
        return AgentToolResult(content=[result])
    if isinstance(result, list):
        return AgentToolResult(content=result)
    raise TypeError(
        f"@tool function {fn.__name__!r} returned {type(result).__name__}; "
        "return an AgentToolResult, a str, a Content, or a list of Content."
    )


def _make_agent_tool(
    fn: ToolFunc,
    *,
    name: str | None,
    description: str | None,
    execution_mode: Literal["sequential", "parallel"] | None,
) -> AgentTool[Any]:
    if not inspect.iscoroutinefunction(fn):
        raise TypeError(
            f"@tool requires an async function; {fn.__name__!r} is not declared "
            "with 'async def'. Wrap blocking work with asyncio.to_thread()."
        )

    sig = inspect.signature(fn)

    # Structural validation runs over the FULL signature, before reserved names
    # are filtered out below. The generated executor always calls the function
    # with keyword arguments (schema fields plus any injected reserved args), so
    # a *args/**kwargs or positional-only parameter — even one using a reserved
    # name like *signal — would fail at call time. Reject it at definition time.
    for p in sig.parameters.values():
        if p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f"@tool function {fn.__name__!r} cannot use *args/**kwargs "
                f"(parameter {p.name!r}); declare explicit parameters so an "
                "input schema can be generated."
            )
        if p.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"@tool parameter {p.name!r} of {fn.__name__!r} is "
                "positional-only; @tool calls the function with keyword "
                "arguments, so declare it as a normal parameter (remove the '/')."
            )

    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception as exc:  # pragma: no cover — unresolved annotations
        raise TypeError(
            f"@tool could not resolve type hints for {fn.__name__!r}: {exc}"
        ) from exc

    schema_params = [p for n, p in sig.parameters.items() if n not in _RESERVED]
    injected = [n for n in _RESERVED if n in sig.parameters]
    params_model = _build_params_model(fn, schema_params, hints)
    field_names = list(params_model.model_fields.keys())

    tool_name = name or fn.__name__
    tool_desc = description if description is not None else (inspect.getdoc(fn) or "")

    async def execute(
        tool_call_id: str,
        params: BaseModel,
        *,
        signal: Any = None,
        on_update: Any = None,
    ) -> AgentToolResult:
        kwargs: dict[str, Any] = {n: getattr(params, n) for n in field_names}
        if "tool_call_id" in injected:
            kwargs["tool_call_id"] = tool_call_id
        if "signal" in injected:
            kwargs["signal"] = signal
        if "on_update" in injected:
            kwargs["on_update"] = on_update
        return _normalize_result(await fn(**kwargs), fn)

    return AgentTool(
        name=tool_name,
        description=tool_desc,
        parameters=params_model,
        execute=execute,
        execution_mode=execution_mode,
    )


@overload
def tool(fn: ToolFunc) -> AgentTool[Any]: ...


@overload
def tool(
    *,
    name: str | None = ...,
    description: str | None = ...,
    execution_mode: Literal["sequential", "parallel"] | None = ...,
) -> Callable[[ToolFunc], AgentTool[Any]]: ...


def tool(
    fn: ToolFunc | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    execution_mode: Literal["sequential", "parallel"] | None = None,
) -> AgentTool[Any] | Callable[[ToolFunc], AgentTool[Any]]:
    """Turn an async function into an :class:`AgentTool`.

    Usable bare (``@tool``) or with arguments (``@tool(name=..., ...)``).

    - The tool's input schema is generated from the function's parameters; each
      needs a type annotation. ``Field(...)`` defaults/metadata are honoured.
    - If the function declares ``tool_call_id``, ``signal``, or ``on_update``,
      the loop's values are passed through; otherwise they are omitted from the
      schema and not passed.
    - The return value may be an ``AgentToolResult``, a ``str``, a ``Content``,
      or a ``list[Content]``.
    """

    def decorator(f: ToolFunc) -> AgentTool[Any]:
        return _make_agent_tool(
            f, name=name, description=description, execution_mode=execution_mode
        )

    return decorator(fn) if fn is not None else decorator
