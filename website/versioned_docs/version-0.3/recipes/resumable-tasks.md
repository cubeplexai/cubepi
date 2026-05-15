---
title: Resumable Long Tasks
---

# Recipe: Resumable Long Tasks

When an agent is mid-flight through a long-running operation (a series
of tool calls, a multi-turn reasoning session) and the process dies,
you want to come back and pick up where it left off — not start over.
CubePi's append-only checkpointing plus `agent.resume()` makes this
trivial *between turns*; for resumption *mid-tool*, you need a little
more care.

**Time to run:** 15 minutes.
**Deps:** `cubepi[sqlite]`, an `ANTHROPIC_API_KEY`.

## The pattern

There are three crash points to think about:

1. **Between turns** — The model has answered, no tools to run, the
   loop is between iterations. `resume()` re-invokes the model. *Free
   with the checkpointer.*
2. **After tool results, before model call** — Tool results are
   persisted. `resume()` sees the last message is a `ToolResultMessage`
   and re-invokes the model. *Free with the checkpointer.*
3. **Mid-tool** — The tool started but didn't finish. Nothing is
   persisted yet (CubePi only persists messages). You need
   tool-internal idempotency. *Requires care.*

This recipe focuses on case 3.

## Idempotent tools with external state

The pattern: each tool action has a deterministic, idempotent key.
Before doing the work, check whether it's been done.

```python title="tools.py"
import os
import json
from pathlib import Path

from pydantic import BaseModel
from cubepi import AgentTool, AgentToolResult, TextContent


# Simple file-backed job store; replace with Redis / Postgres in prod.
JOB_DIR = Path(os.environ.get("JOB_DIR", "/tmp/cubepi-jobs"))
JOB_DIR.mkdir(parents=True, exist_ok=True)


class TranscodeParams(BaseModel):
    source_path: str
    output_path: str


async def transcode_video(tool_call_id, params: TranscodeParams, *, signal=None, on_update=None):
    job_key = f"transcode:{params.source_path}->{params.output_path}"
    job_file = JOB_DIR / f"{job_key.replace('/', '_')}.json"

    if job_file.exists():
        # Already done in a previous run.
        state = json.loads(job_file.read_text())
        return AgentToolResult(
            content=[TextContent(text=f"Already transcoded to {state['output_path']}.")],
            details=state,
        )

    # Do the actual work (long-running, expensive).
    # Use signal to abort cleanly if cancelled.
    output = await run_ffmpeg(params.source_path, params.output_path, signal=signal)

    # Commit the job-done marker AFTER the work succeeds.
    job_file.write_text(json.dumps({"output_path": output}))

    return AgentToolResult(
        content=[TextContent(text=f"Transcoded to {output}.")],
        details={"output_path": output},
    )


transcode_tool = AgentTool(
    name="transcode_video",
    description="Transcode a video file. Idempotent — safe to retry.",
    parameters=TranscodeParams,
    execute=transcode_video,
    execution_mode="sequential",  # one transcode at a time
)
```

Now if the process dies during `run_ffmpeg`, the next agent run sees
`job_file.exists() == False`, redoes the work, and only writes the
marker on success. If the process dies *after* the marker was
written, the next run sees the marker, returns the cached result
immediately, and the agent continues as if it had just finished.

## Resuming the agent

```python title="resume.py"
import asyncio
import os
import sys

from cubepi import Agent, Model
from cubepi.checkpointer import SQLiteCheckpointer
from cubepi.providers.anthropic import AnthropicProvider

from tools import transcode_tool   # the wrapped AgentTool from above


async def main(thread_id: str, initial_prompt: str | None):
    async with SQLiteCheckpointer("jobs.db") as cp:
        agent = Agent(
            provider=AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"]),
            model=Model(id="claude-sonnet-4-5-20250929", provider="anthropic"),
            system_prompt="You orchestrate video transcoding jobs.",
            tools=[transcode_tool],
            checkpointer=cp,
            thread_id=thread_id,
        )
        agent.subscribe(lambda e, s=None: None)

        if initial_prompt:
            # Fresh run.
            await agent.prompt(initial_prompt)
        else:
            # Resume.  If the last message is an assistant message with no
            # follow-up queued, this will raise; otherwise it picks up.
            await agent.resume()


if __name__ == "__main__":
    thread_id = sys.argv[1]
    initial = sys.argv[2] if len(sys.argv) > 2 else None
    asyncio.run(main(thread_id, initial))
```

Workflow:

```bash
# Start a job:
python resume.py job-1 "Transcode /videos/a.mov to /out/a.mp4 and /videos/b.mov to /out/b.mp4"

# Kill it mid-flight: Ctrl-C.

# Resume — agent picks up from the last persisted message:
python resume.py job-1
```

## The three resume scenarios in code

```python
async def smart_resume(agent):
    msgs = agent.state.messages
    if not msgs:
        # Brand-new conversation. Caller must prompt.
        return False

    last = msgs[-1]
    last_role = type(last).__name__

    if last_role == "AssistantMessage":
        # Either the run finished naturally, or it died right after a
        # turn ended. resume() raises unless there's queued steering.
        # Easiest path: ask the user what's next.
        return False

    # Last is UserMessage or ToolResultMessage — safe to resume.
    await agent.resume()
    return True
```

## Persistence + abort

`agent.abort()` triggers a clean teardown that emits `agent_end`. The
last *fully persisted* message is whatever made it through
`message_end`. Aborts during a tool's execution don't persist the
tool result (the tool didn't return), so `resume()` will re-dispatch
the model with the last `AssistantMessage` containing the unfinished
`ToolCall`. The model will usually re-issue the call — your
idempotency guards handle the rest.

## What about persisting partial tool state?

CubePi doesn't expose a "persist a partial tool result" API. The
intended pattern is: keep partial state in the tool's own
external store (filesystem, Redis, S3), keyed deterministically by the
tool args. That's what `transcode_video` above does with `JOB_DIR`.

## Common pitfalls

- **Non-idempotent tools** — Without deterministic keys, retries can
  charge a credit card twice or send a duplicate email. Always wrap
  external side-effects in an idempotency key.
- **Job markers in `/tmp`** — Cleared on reboot. Use a real
  persistence layer for production jobs.
- **`resume()` after an assistant message with no queue** — Raises.
  Either prompt the user for the next message or call `prompt()`
  fresh.
- **Forgetting the signal check inside the tool** — A long
  `await asyncio.sleep(...)` or a `for ... in stream` that ignores
  `signal.is_set()` won't honour `abort`. Drop a check inside any
  hot loop.

## See also

- [Multi-turn → `resume()`](../guides/agents/multi-turn#resume--continue-from-the-last-message)
  — full semantics.
- [Persistent Chat](./persistent-chat) — the simpler restartable
  scenario.
- [SQLite Checkpointing](../guides/checkpointing/sqlite) — what's
  persisted, when.
