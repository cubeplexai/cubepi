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
)
from cubepi.middleware import Middleware, compose_middleware
from cubepi.providers import (
    AssistantMessage,
    BaseProvider,
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
from cubepi.providers.capability import (
    CapabilityDescriptor,
    ReasoningLevelSpec,
    TemperatureSpec,
)

__all__ = [
    "__version__",
    "Agent",
    "AgentState",
    "AgentTool",
    "AgentToolResult",
    "AssistantMessage",
    "BaseProvider",
    "CapabilityDescriptor",
    "Message",
    "MessageStream",
    "Middleware",
    "Model",
    "Provider",
    "ReasoningLevelSpec",
    "StreamEvent",
    "StreamOptions",
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
]
