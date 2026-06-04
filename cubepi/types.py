from __future__ import annotations
# mypy: disable-error-code=misc

from pydantic import BaseModel
from typing_extensions import TypeAliasType

JsonPrimitive = str | int | float | bool | None
JsonValue = TypeAliasType(
    "JsonValue",
    JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"],
)
JsonObject = dict[str, JsonValue]

StructuredValue = TypeAliasType(
    "StructuredValue",
    JsonPrimitive | BaseModel | list["StructuredValue"] | dict[str, "StructuredValue"],
)
StructuredObject = dict[str, StructuredValue]
