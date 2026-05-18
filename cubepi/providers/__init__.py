from cubepi.providers.base import (
    AssistantMessage,
    BaseProvider,
    Content,
    ImageContent,
    Message,
    MessageStream,
    Model,
    ModelCost,
    OnChunkCallback,
    OnPayloadCallback,
    OnRequestCallback,
    OnResponseBodyCallback,
    OnResponseCallback,
    Provider,
    ProviderResponse,
    StreamEvent,
    StreamOptions,
    TextContent,
    ThinkingBudgets,
    ThinkingContent,
    ThinkingLevel,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    Usage,
    UserMessage,
    adjust_max_tokens_for_thinking,
)
from cubepi.providers.faux import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
)
from cubepi.providers.models import (
    THINKING_LEVELS,
    clamp_thinking_level,
    get_supported_thinking_levels,
    models_are_equal,
)


# Lazy imports for optional providers
def get_anthropic_provider():
    from cubepi.providers.anthropic import AnthropicProvider

    return AnthropicProvider


def get_openai_provider():
    from cubepi.providers.openai import OpenAIProvider

    return OpenAIProvider


def get_openai_responses_provider():
    from cubepi.providers.openai_responses import OpenAIResponsesProvider

    return OpenAIResponsesProvider


__all__ = [
    "AssistantMessage",
    "BaseProvider",
    "Content",
    "FauxProvider",
    "ImageContent",
    "Message",
    "MessageStream",
    "Model",
    "ModelCost",
    "OnChunkCallback",
    "OnPayloadCallback",
    "OnRequestCallback",
    "OnResponseBodyCallback",
    "OnResponseCallback",
    "Provider",
    "ProviderResponse",
    "StreamEvent",
    "StreamOptions",
    "TextContent",
    "ThinkingBudgets",
    "ThinkingContent",
    "ThinkingLevel",
    "ToolCall",
    "ToolDefinition",
    "ToolResultMessage",
    "Usage",
    "UserMessage",
    "adjust_max_tokens_for_thinking",
    "clamp_thinking_level",
    "faux_assistant_message",
    "faux_text",
    "faux_thinking",
    "faux_tool_call",
    "get_supported_thinking_levels",
    "models_are_equal",
    "THINKING_LEVELS",
]
