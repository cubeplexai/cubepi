from __future__ import annotations
# mypy: disable-error-code=misc

from pydantic import BaseModel, SerializeAsAny
from typing_extensions import TypeAliasType

JsonPrimitive = str | int | float | bool | None
JsonValue = TypeAliasType(
    "JsonValue",
    JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"],
)
JsonObject = dict[str, JsonValue]

# The ``SerializeAsAny[BaseModel]`` branch forces pydantic to dump the runtime
# subclass instead of the declared base. Without it, fields typed as
# ``StructuredValue`` that hold a concrete model instance silently serialize to
# ``{}`` — pydantic's "smart" union dispatch picks the empty ``BaseModel``
# schema for the union branch and ignores the instance's real fields. This
# bug bites checkpointer persistence and compaction message-ref hashing, where
# ``ToolResultMessage.details`` would lose its payload on a save round-trip.
StructuredValue = TypeAliasType(
    "StructuredValue",
    JsonPrimitive
    | SerializeAsAny[BaseModel]
    | list["StructuredValue"]
    | dict[str, "StructuredValue"],
)
StructuredObject = dict[str, StructuredValue]
