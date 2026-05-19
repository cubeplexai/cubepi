---
title: Installation
---

# Installation

CubePi runs on **Python 3.11+**. The core has three runtime
dependencies: `pydantic`, `anthropic`, `openai`. Optional features
(SQLite, Postgres, MCP, OpenTelemetry tracing) are gated behind extras
so you only install what you use.

## With pip

```bash
pip install cubepi
```

Optional extras:

```bash
pip install "cubepi[sqlite]"        # adds aiosqlite for SQLiteCheckpointer
pip install "cubepi[postgres]"      # adds asyncpg + sqlalchemy + msgpack
pip install "cubepi[mcp]"           # adds the MCP SDK for tool loaders
pip install "cubepi[tracing]"       # adds opentelemetry-sdk for Tracer / Meter
pip install "cubepi[tracing-otlp]"  # adds the OTLP/HTTP span exporter
pip install "cubepi[sqlite,mcp,tracing]"  # combine
```

## With uv

[`uv`](https://github.com/astral-sh/uv) is significantly faster than
pip and is the recommended workflow:

```bash
uv add cubepi
uv add "cubepi[sqlite,postgres,mcp,tracing,tracing-otlp]"
```

In an existing uv project, `uv sync` re-locks the environment after
edits to `pyproject.toml`.

## With Poetry

```bash
poetry add cubepi
poetry add "cubepi[sqlite,postgres,mcp,tracing,tracing-otlp]"
```

## Verifying the install

```bash
python -c "import cubepi; print(cubepi.__doc__)"
# cubepi — Pythonic async-native agent framework.
```

If you see an `ImportError`, your interpreter is likely \< 3.11 — check
`python --version`.

## Configuring provider credentials

CubePi providers read credentials from constructor arguments. Most
deployments pull them from environment variables:

```python
import os
from cubepi.providers.anthropic import AnthropicProvider
from cubepi.providers.openai import OpenAIProvider

anthropic = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"])
openai = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])
```

You can also pass `base_url=...` to either provider to point at a
self-hosted endpoint or compatible proxy (e.g. Anthropic Bedrock,
LiteLLM, vLLM).

For the [FauxProvider](../guides/providers/custom#using-fauxprovider-in-tests)
(used in tests), no credentials are required.

## Choosing extras: which to install

| Extra | Pulls in | Install when |
|---|---|---|
| (none) | core only | You only need in-memory state, no MCP, no tracing |
| `[sqlite]` | `aiosqlite` | Single-process app needs disk persistence |
| `[postgres]` | `asyncpg`, `sqlalchemy`, `msgpack` | Multi-instance / production — see [Postgres guide](../guides/checkpointing/postgres) |
| `[mcp]` | `mcp` | You want to mount MCP server tools into your agent |
| `[tracing]` | `opentelemetry-sdk` | You want OpenTelemetry traces (and optionally metrics) — see [Tracing guide](../guides/tracing/overview) |
| `[tracing-otlp]` | `opentelemetry-exporter-otlp-proto-http` | Ship traces to an OTLP/HTTP backend (Jaeger ≥1.50, Tempo, Honeycomb, Datadog, …) |
| `[docs]` | `griffe` | You're building the docs site (contributors only) |

## Next steps

- [Quick Start](./quick-start) — your first agent in five minutes.
- [Core Concepts](./core-concepts) — what `Agent` / `Tool` / `Provider`
  / `Checkpointer` actually mean before you start gluing them.
