"""cubepi — Pythonic async-native agent framework."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("cubepi")
except PackageNotFoundError:  # pragma: no cover — non-installed checkout
    __version__ = "0.0.0+unknown"


from cubepi.agent import (
    Agent,
    AgentState,
    AgentTool,
    AgentToolResult,
    run_agent_loop,
    run_agent_loop_continue,
    tool,
)
from cubepi.middleware import Middleware, compose_middleware
from cubepi.providers import (
    AssistantMessage,
    BaseProvider,
    BoundModel,
    Message,
    MessageStream,
    Model,
    Provider,
    StreamEvent,
    StreamOptions,
    TextContent,
    ThinkingBudgets,
    ThinkingLevel,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    UserMessage,
    adjust_max_tokens_for_thinking,
)
from cubepi.errors import (
    ContextLengthExceeded,
    ProviderAuthFailed,
    ProviderBadRequest,
    ProviderError,
    ProviderUnavailable,
    RateLimited,
)
from cubepi.providers.capability import (
    CapabilityDescriptor,
    ReasoningLevelSpec,
    TemperatureSpec,
)
from cubepi.providers.images import (
    AssistantImages,
    BaseImagesProvider,
    ImagesAborted,
    ImagesCapabilityDescriptor,
    ImagesContext,
    ImagesCost,
    ImagesModel,
    ImagesOptions,
    ImagesProvider,
    SizeSpec,
)
from cubepi.types import JsonObject, JsonValue, StructuredObject, StructuredValue

__all__ = [
    "__version__",
    "Agent",
    "AgentState",
    "AgentTool",
    "AgentToolResult",
    "AssistantImages",
    "AssistantMessage",
    "BaseImagesProvider",
    "BaseProvider",
    "BoundModel",
    "CapabilityDescriptor",
    "ContextLengthExceeded",
    "ImagesAborted",
    "ImagesCapabilityDescriptor",
    "ImagesContext",
    "ImagesCost",
    "ImagesModel",
    "ImagesOptions",
    "ImagesProvider",
    "JsonObject",
    "JsonValue",
    "Message",
    "MessageStream",
    "Middleware",
    "Model",
    "Provider",
    "ProviderAuthFailed",
    "ProviderBadRequest",
    "ProviderError",
    "ProviderUnavailable",
    "RateLimited",
    "ReasoningLevelSpec",
    "SizeSpec",
    "StreamEvent",
    "StreamOptions",
    "StructuredObject",
    "StructuredValue",
    "TemperatureSpec",
    "TextContent",
    "ThinkingBudgets",
    "ThinkingLevel",
    "ToolCall",
    "ToolDefinition",
    "ToolResultMessage",
    "UserMessage",
    "adjust_max_tokens_for_thinking",
    "compose_middleware",
    "run_agent_loop",
    "run_agent_loop_continue",
    "tool",
]
