---
id: intro
title: CubePi
slug: /
sidebar_position: 0
---

# CubePi

**CubePi** is a Pythonic, async-native agent framework — a leaner, more
readable take on agent runtimes like
[langgraph](https://github.com/langchain-ai/langgraph). It models an
agent as a plain `async` function instead of a state graph; you can read
the entire core loop in five minutes.

```bash
pip install cubepi
```

Then jump to the [Quick Start](./getting-started/quick-start) to ship a
working tool-using agent in under five minutes.

## What you get

- **Plain async functions, not graph nodes.** The agent loop is a
  `while` loop over message turns. You build tools as `async def`
  functions; the framework routes calls and executes them in parallel.
- **One streaming pattern.** Every provider yields `StreamEvent`s
  through a `MessageStream`. You iterate it with `async for`. No
  callback registries, no separate handler types for text vs. tools.
- **Append-only checkpointing.** Persistence writes the new messages on
  each turn, not the full transcript. O(1) DB I/O regardless of
  conversation length — SQLite for laptops, Postgres for production.
- **Native multi-provider.** Anthropic and OpenAI ship in the box,
  through a `Provider` Protocol. Add a new provider in one class.
- **Five-hook middleware.** `transform_context`, `convert_to_llm`,
  `before_tool_call`, `after_tool_call`, `should_stop_after_turn` —
  each with explicit composition rules. No mystery node ordering.
- **MCP loaders.** Point at any
  [Model Context Protocol](https://modelcontextprotocol.io) server
  (HTTP or stdio) and get back a list of `AgentTool`s.
- **OpenTelemetry built in.** Attach a `Tracer` and every prompt
  produces a tree of OTel spans aligned with the GenAI Semantic
  Conventions — works with Jaeger, Tempo, Honeycomb, Datadog, or any
  OTLP-compatible backend. No payloads recorded by default; opt in
  with `record_content=True` and a `redact` callback.

## Where to go next

| If you want to… | Start here |
|---|---|
| Install and run your first agent | [Getting Started → Installation](./getting-started/installation) |
| Understand the building blocks | [Getting Started → Core Concepts](./getting-started/core-concepts) |
| Wire a real tool-using agent | [Guides → Building Your First Agent](./guides/agents/first-agent) |
| Persist a conversation across restarts | [Guides → SQLite Checkpointing](./guides/checkpointing/sqlite) |
| Ship traces to Jaeger / Tempo / Honeycomb | [Guides → Tracing](./guides/tracing/overview) |
| Look up a specific symbol | [API Reference](./api/) |
| See full working examples | [Recipes](./recipes/weather-agent) |
| Port an existing langgraph agent | [Migration → From langgraph](./migration/from-langgraph) |

## Status

CubePi is at `v0.3.0` — alpha. The public API surface is stable for
v0.3 and frozen in the [`0.3` docs snapshot](pathname:///). The `next`
channel (toggle in the top-right) tracks the unreleased main branch.

Source, issues, and discussion live on
[GitHub](https://github.com/cubeplexai/cubepi).
