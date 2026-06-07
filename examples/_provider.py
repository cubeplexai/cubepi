"""Shared provider setup for cubepi examples.

Set one of the following before running any example:

    # Anthropic (or Anthropic-compatible endpoint):
    export ANTHROPIC_API_KEY=sk-ant-...
    export ANTHROPIC_BASE_URL=https://...   # optional, for compatible endpoints
    export MODEL=claude-sonnet-4-6          # optional, this is the default

    # OpenAI (or OpenAI-compatible endpoint):
    export OPENAI_API_KEY=sk-...
    export OPENAI_BASE_URL=https://...      # optional, for compatible endpoints
    export MODEL=gpt-4o                     # optional, this is the default

ANTHROPIC_API_KEY takes priority when both are set.
"""

import os
import sys

_anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
_openai_key = os.environ.get("OPENAI_API_KEY")

if _anthropic_key:
    from cubepi.providers.anthropic import AnthropicProvider

    _base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
    provider = AnthropicProvider(api_key=_anthropic_key, base_url=_base_url)
    MODEL_ID = os.environ.get("MODEL", "claude-sonnet-4-6")
elif _openai_key:
    from cubepi.providers.openai import OpenAIProvider

    _base_url = os.environ.get("OPENAI_BASE_URL") or None
    provider = OpenAIProvider(api_key=_openai_key, base_url=_base_url)
    MODEL_ID = os.environ.get("MODEL", "gpt-4o")
else:
    print(
        "Error: set ANTHROPIC_API_KEY or OPENAI_API_KEY before running examples.",
        file=sys.stderr,
    )
    sys.exit(1)
