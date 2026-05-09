from cubepi.providers.base import (
    AssistantMessage,
    Content,
    ImageContent,
    Message,
    MessageStream,
    Model,
    ModelCost,
    Provider,
    StreamEvent,
    TextContent,
    ThinkingContent,
    ThinkingLevel,
    ToolCall,
    ToolDefinition,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from cubepi.providers.faux import (
    FauxProvider,
    faux_assistant_message,
    faux_text,
    faux_thinking,
    faux_tool_call,
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
    "Content",
    "FauxProvider",
    "ImageContent",
    "Message",
    "MessageStream",
    "Model",
    "ModelCost",
    "Provider",
    "StreamEvent",
    "TextContent",
    "ThinkingContent",
    "ThinkingLevel",
    "ToolCall",
    "ToolDefinition",
    "ToolResultMessage",
    "Usage",
    "UserMessage",
    "faux_assistant_message",
    "faux_text",
    "faux_thinking",
    "faux_tool_call",
]
