"""Field-name constants and pinned semconv version for cubepi tracing.

Align with OpenTelemetry GenAI Semantic Conventions v1.41.0
(https://opentelemetry.io/docs/specs/semconv/gen-ai/). Anything cubepi-
specific lives under the ``cubepi.*`` namespace per the OTel-recommended
vendor-extension pattern.
"""

from __future__ import annotations

#: Pinned semconv version. Bumping requires auditing for renamed/deprecated
#: attributes — both Resource and InstrumentationScope advertise this URL.
SCHEMA_URL = "https://opentelemetry.io/schemas/1.41.0"

# ---------------------------------------------------------------------------
# OTel GenAI semantic conventions (https://opentelemetry.io/docs/specs/semconv/gen-ai/)
# ---------------------------------------------------------------------------

# Operation discriminator
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
OP_INVOKE_AGENT = "invoke_agent"
OP_CHAT = "chat"
OP_EXECUTE_TOOL = "execute_tool"

# Provider id
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"

# Agent identity (process-level lives in Resource; per-run override possible)
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_AGENT_ID = "gen_ai.agent.id"
GEN_AI_AGENT_DESCRIPTION = "gen_ai.agent.description"
GEN_AI_AGENT_VERSION = "gen_ai.agent.version"

# Conversation / thread
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"

# Request parameters
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_REQUEST_TOP_K = "gen_ai.request.top_k"
GEN_AI_REQUEST_STOP_SEQUENCES = "gen_ai.request.stop_sequences"
GEN_AI_REQUEST_FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
GEN_AI_REQUEST_PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
GEN_AI_REQUEST_SEED = "gen_ai.request.seed"
GEN_AI_REQUEST_STREAM = "gen_ai.request.stream"
GEN_AI_REQUEST_CHOICE_COUNT = "gen_ai.request.choice.count"
GEN_AI_OUTPUT_TYPE = "gen_ai.output.type"

# Response metadata
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_ID = "gen_ai.response.id"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK = "gen_ai.response.time_to_first_chunk"

# Usage / tokens
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_USAGE_REASONING_OUTPUT_TOKENS = "gen_ai.usage.reasoning.output_tokens"
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"

# Tool execution
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
GEN_AI_TOOL_DESCRIPTION = "gen_ai.tool.description"

# Network (recommended on CLIENT spans)
SERVER_ADDRESS = "server.address"
SERVER_PORT = "server.port"

# Error classification (CR on error)
ERROR_TYPE = "error.type"

# GenAI-specific exception event name
EVENT_GEN_AI_EXCEPTION = "gen_ai.client.operation.exception"

# ---------------------------------------------------------------------------
# cubepi extension attributes
# ---------------------------------------------------------------------------

# Per-run identity
CUBEPI_RUN_ID = "cubepi.run_id"
CUBEPI_THREAD_ID = "cubepi.thread_id"

# Root span helpers
CUBEPI_AGENT_TOOLS = "cubepi.agent.tools"
CUBEPI_AGENT_SYSTEM_PROMPT_SHA256 = "cubepi.agent.system_prompt.sha256"
CUBEPI_INPUT_MESSAGES_COUNT = "cubepi.input.messages.count"
CUBEPI_OUTPUT_MESSAGES_COUNT = "cubepi.output.messages.count"
CUBEPI_ABORTED = "cubepi.aborted"

# Turn span attributes (cubepi.turn span — no gen_ai.operation.name)
CUBEPI_TURN_INDEX = "cubepi.turn.index"
CUBEPI_TURN_STOP_REASON = "cubepi.turn.stop_reason"
CUBEPI_TURN_TOOL_CALLS_COUNT = "cubepi.turn.tool_calls.count"
CUBEPI_TURN_TERMINATED_BY_TOOL = "cubepi.turn.terminated_by_tool"

# Chat span — provider-side LLM call
CUBEPI_LLM_THINKING_LEVEL = "cubepi.llm.thinking_level"
CUBEPI_LLM_RAW_REQUEST = "cubepi.llm.raw_request"
CUBEPI_LLM_RAW_RESPONSE = "cubepi.llm.raw_response"

# Tool span
CUBEPI_TOOL_EXECUTION_MODE = "cubepi.tool.execution_mode"
CUBEPI_TOOL_IS_ERROR = "cubepi.tool.is_error"
CUBEPI_TOOL_TERMINATE = "cubepi.tool.terminate"
CUBEPI_TOOL_BLOCKED_BY_HOOK = "cubepi.tool.blocked_by_hook"
CUBEPI_TOOL_BLOCK_REASON = "cubepi.tool.block_reason"

# Span names (NOT semconv keys — these are name templates)
SPAN_NAME_INVOKE_AGENT = "invoke_agent"
SPAN_NAME_TURN = "cubepi.turn"
SPAN_NAME_CHAT = "chat"
SPAN_NAME_EXECUTE_TOOL = "execute_tool"

# Instrumentation scope identity
SCOPE_NAME = "cubepi.tracing"


# ---------------------------------------------------------------------------
# Provider name mapping (cubepi.Model.provider → gen_ai.provider.name)
# ---------------------------------------------------------------------------

#: Cubepi providers identify themselves with short strings on
#: ``Model.provider``. Map to the canonical OTel ``gen_ai.provider.name``
#: values per the semconv registry. Unknown providers get the raw string
#: prefixed with ``unknown:`` so observers can spot the gap.
PROVIDER_NAME_MAP: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "azure_openai": "azure.ai.openai",
    "gemini": "gcp.gemini",
    "vertex_ai": "gcp.vertex_ai",
    "bedrock": "aws.bedrock",
    "cohere": "cohere",
    "mistral": "mistral_ai",
    "groq": "groq",
    "xai": "x_ai",
    "deepseek": "deepseek",
    "perplexity": "perplexity",
    "watsonx": "ibm.watsonx.ai",
    "faux": "faux",
}


def map_provider_name(provider: str) -> str:
    """Translate cubepi ``Model.provider`` to the canonical
    ``gen_ai.provider.name`` value.

    Unknown providers get ``unknown:<raw>`` so observers can spot a gap.
    """
    return PROVIDER_NAME_MAP.get(provider, f"unknown:{provider}")
