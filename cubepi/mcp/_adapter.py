"""MCP tool descriptor → cubepi.AgentTool adapter."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, Field, create_model

from cubepi.agent.types import AgentTool, AgentToolResult
from cubepi.providers.base import Content, ImageContent, TextContent


def mcp_schema_to_pydantic_model(
    *,
    tool_name: str,
    input_schema: dict[str, Any],
) -> type[BaseModel]:
    """Convert an MCP JSON-Schema inputSchema to a Pydantic model class.

    cubepi.AgentTool requires `parameters: type[BaseModel]`. We synthesize
    a model from the schema's top-level properties.

    Type coverage: string/integer/number/boolean/array/object. Nested
    object schemas become dict[str, Any]. Constraint coverage on top-level
    properties: enum (via Literal), description, pattern, minLength,
    maxLength, minimum/maximum (incl. exclusive variants), minItems,
    maxItems. Anything else is preserved as a plain type without
    validation.
    """
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    fields: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        fields[prop_name] = _build_field(prop_schema, prop_name in required)

    model_name = f"MCP_{tool_name}_Input"
    return create_model(model_name, **fields)


def _build_field(prop_schema: dict[str, Any], required: bool) -> tuple[Any, Any]:
    """Map one JSON-Schema property to a (type, FieldInfo|default) tuple."""
    if "enum" in prop_schema and isinstance(prop_schema["enum"], list):
        enum_values = tuple(prop_schema["enum"])
        py_type: Any = Literal[enum_values] if enum_values else Any
    else:
        py_type = _json_schema_type_to_python(prop_schema)

    field_kwargs: dict[str, Any] = {}
    if "description" in prop_schema:
        field_kwargs["description"] = prop_schema["description"]

    # string constraints
    if "pattern" in prop_schema:
        field_kwargs["pattern"] = prop_schema["pattern"]
    if "minLength" in prop_schema:
        field_kwargs["min_length"] = prop_schema["minLength"]
    if "maxLength" in prop_schema:
        field_kwargs["max_length"] = prop_schema["maxLength"]

    # numeric constraints
    if "minimum" in prop_schema:
        field_kwargs["ge"] = prop_schema["minimum"]
    if "maximum" in prop_schema:
        field_kwargs["le"] = prop_schema["maximum"]
    if "exclusiveMinimum" in prop_schema:
        field_kwargs["gt"] = prop_schema["exclusiveMinimum"]
    if "exclusiveMaximum" in prop_schema:
        field_kwargs["lt"] = prop_schema["exclusiveMaximum"]

    # array constraints
    if "minItems" in prop_schema:
        field_kwargs["min_length"] = prop_schema["minItems"]
    if "maxItems" in prop_schema:
        field_kwargs["max_length"] = prop_schema["maxItems"]

    default = ... if required else None
    if field_kwargs:
        return (py_type, Field(default=default, **field_kwargs))
    return (py_type, default)


def _json_schema_type_to_python(schema: dict[str, Any]) -> Any:
    t = schema.get("type")
    if t == "string":
        return str
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "boolean":
        return bool
    if t == "array":
        item_t = _json_schema_type_to_python(schema.get("items", {}))
        return list[item_t]  # type: ignore[valid-type]
    if t == "object":
        return dict[str, Any]
    return Any


def make_mcp_agent_tool(
    *,
    name: str,
    description: str,
    input_schema: dict[str, Any],
    call_remote: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]],
) -> AgentTool:
    """Build a cubepi.AgentTool wrapping an MCP tool call.

    call_remote is the transport-specific RPC: given (tool_name, args_dict),
    returns the MCP tools/call response dict normalized to:
        {"content": [{"type": "text", "text": ...}, ...], "isError": bool}
    """
    parameters_model = mcp_schema_to_pydantic_model(
        tool_name=name,
        input_schema=input_schema,
    )

    async def _execute(
        tool_call_id: str,
        args,
        *,
        signal=None,
        on_update=None,
    ) -> AgentToolResult:
        # tool_call_id and on_update are unused by MCP semantics (MCP RPC is
        # not incremental), but we accept them for signature compatibility
        # with cubepi's agent loop.
        del tool_call_id, on_update
        args_dict = (
            args.model_dump(exclude_none=True)
            if hasattr(args, "model_dump")
            else dict(args)
        )
        result = await call_remote(name, args_dict)
        content_blocks: list[Content] = []
        for c in result.get("content", []):
            ctype = c.get("type")
            if ctype == "text":
                content_blocks.append(TextContent(text=c.get("text", "")))
            elif ctype == "image":
                content_blocks.append(
                    ImageContent(
                        source=c.get("data", ""),
                        media_type=c.get("mimeType", ""),
                    )
                )
        details: dict[str, Any] = {"raw_mcp_response": result}
        if "structuredContent" in result:
            details["structuredContent"] = result["structuredContent"]
        return AgentToolResult(
            content=content_blocks,
            details=details,
            is_error=True if result.get("isError") else None,
        )

    return AgentTool(
        name=name,
        description=description,
        parameters=parameters_model,
        execute=_execute,
    )
