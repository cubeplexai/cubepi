---
title: Compaction
description: "Use CompactionMiddleware to summarize older turns while preserving full CubePi history."
---

# Compaction

`CompactionMiddleware` keeps long conversations inside a model's context window
without deleting agent history. It summarizes older turns into `ctx.extra`, then
sends the model a compressed view: one summary message plus the most recent
messages. `agent.state.messages` and checkpointer history stay complete.

## Basic setup

Use a cheaper model for the summary pass and your normal model for the agent:

```python
from cubepi import Agent
from cubepi.middleware import CompactionMiddleware

model = main_provider.model("claude-sonnet-4-5-20250929")
summary_model = cheap_provider.model("claude-haiku-4-5")

agent = Agent(
    model=model,
    checkpointer=checkpointer,
    thread_id="conv_123",
    middleware=[
        CompactionMiddleware(
            summary_model=summary_model,
            max_tokens_before_compact=80_000,
            keep_recent_messages=8,
            max_summary_tokens=1024,
        ),
    ],
)
```

The summary call uses `Provider.generate(...)` with `temperature=0.0`,
`thinking="off"`, and `max_output_tokens=max_summary_tokens`.

## What gets persisted

The middleware writes two keys into `AgentContext.extra`:

- `compaction` — the summary state and the message refs it covers.
- `compaction_until_msg_index` — the history boundary summarized so far.

When a checkpointer is attached, CubePi saves `ctx.extra` at `agent_end`, so the
next process can resume with the existing summary. If the message refs no longer
match the current history, the middleware clears the stale state and starts over
rather than sending an invalid summary.

## Choosing thresholds

Start with conservative values:

```python
CompactionMiddleware(
    summary_model=cheap_model,
    max_tokens_before_compact=80_000,
    keep_recent_messages=8,
    max_summary_tokens=1024,
)
```

Raise `max_tokens_before_compact` if your model has a large context window and
you want fewer summary calls. Raise `keep_recent_messages` when recent tool
outputs or user corrections are especially important. Increase
`max_summary_tokens` for long-running research or coding sessions where the
summary needs more detail.

## Tracing

When `cubepi.tracing` is attached to the agent, the summarizer call is
first-class in the trace tree. `summarize()` opens a
`cubepi.compaction.summarize` parent span (tagged with
`cubepi.compaction.message_count`) around the LLM call, and the recorder
automatically subscribes the summarizer provider so its `chat` span lands
inside:

```
invoke_agent
└── cubepi.turn
    ├── cubepi.compaction.summarize
    │   └── chat <summary-model>
    └── chat <main-model>
```

The wrapper span is a no-op context manager when OpenTelemetry isn't
installed, so the middleware works the same on minimal installs. The
root `invoke_agent` span's `gen_ai.provider.name` /
`cubepi.agent.system_prompt_sha256` / `cubepi.agent.tools` continue to
reflect the agent's main provider/model, not the summarizer's — even
when summarization runs first.

## Failure behavior

If the summary provider fails, CubePi logs a warning and continues with the
previous compressed view or the original messages. The agent run is not failed
only because compaction could not refresh.

## When not to use it

Skip compaction for short tasks, stateless agents, or workflows where every
token of old tool output must be visible to the model. In those cases a simple
sliding-window `transform_context` hook can be easier to reason about.
