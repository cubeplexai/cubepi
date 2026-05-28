---
title: SQLite Checkpointing
description: "Use SQLiteCheckpointer for lightweight single-process agent state persistence in CubePi."
---

# SQLite Checkpointing

`SQLiteCheckpointer` is the lightweight persistence backend: a single
local file, no server, append-only message log. It's the default
choice for laptops, single-process apps, desktop tools, and any
environment where one Python process owns the conversation.

Install the extra:

```bash
pip install "cubepi[sqlite]"
```

This pulls in `aiosqlite`.

## Basic usage

```python
import asyncio
from cubepi import Agent, Model
from cubepi.checkpointer import SQLiteCheckpointer
from cubepi.providers.anthropic import AnthropicProvider


async def main():
    provider = AnthropicProvider(api_key="…")
    async with SQLiteCheckpointer("agent.db") as cp:
        agent = Agent(
            provider=provider,
            model=Model(id="claude-sonnet-4-5-20250929", provider="anthropic"),
            checkpointer=cp,
            thread_id="user-42",
        )
        await agent.prompt("Remember: my favourite colour is teal.")

        # Restart the script later — the file persists.
        await agent.prompt("What did I say my favourite colour was?")
        # → "You said teal."


asyncio.run(main())
```

Two things to internalise:

1. **`thread_id`** is your conversation identifier — usually a user id
   or a session id. Two agents on the same `thread_id` share the same
   history.
2. **`async with SQLiteCheckpointer(...)`** is required: the context
   manager opens the connection and runs the one-time table creation
   on `__aenter__`. Using it without the context manager will raise
   `AssertionError`.

## What gets persisted

`SQLiteCheckpointer` writes two tables on first use:

```sql
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    message_json TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT (julianday('now'))
);

CREATE TABLE thread_extra (
    thread_id TEXT PRIMARY KEY,
    extra_json TEXT NOT NULL DEFAULT '{}'
);
```

- Every `UserMessage`, `AssistantMessage`, and `ToolResultMessage`
  becomes one row in `messages`. Pydantic `model_dump()` is used for
  the JSON payload.
- The `extra` dict on `AgentContext` is persisted on `agent_end` into
  `thread_extra`. Middleware that wants thread-scoped state should
  write into `context.extra`.

This schema is append-only. CubePi never updates or deletes rows.

## HITL pending table

When the [HITL](../hitl) module is in use, an additional table is
created automatically on `__aenter__`:

```sql
CREATE TABLE IF NOT EXISTS thread_pending_request (
    thread_id TEXT PRIMARY KEY,
    request_json TEXT NOT NULL,
    created_at REAL NOT NULL DEFAULT (julianday('now'))
);
```

No manual migration is needed — `CREATE TABLE IF NOT EXISTS` is
idempotent.

## When CubePi reads

On the **first** `prompt()` after instantiation, CubePi calls
`load(thread_id)`. If the thread exists, history is restored into
`agent.state.messages` and `extra` is restored into the agent's
private `_extra` dict.

Subsequent `prompt()` calls don't re-read — the in-memory state is
authoritative.

This means: **don't share one `Agent` instance across processes**.
Process A's in-memory state will diverge from Process B's writes.

## Multi-thread isolation

```python
async with SQLiteCheckpointer("agent.db") as cp:
    alice = Agent(provider=…, model=…, checkpointer=cp, thread_id="alice")
    bob   = Agent(provider=…, model=…, checkpointer=cp, thread_id="bob")
    # Each call only loads/appends its own thread.
```

You can pool one checkpointer across many users; the `thread_id`
isolates them.

## Concurrency model

The checkpointer uses an `asyncio.Lock` around every read and write.
SQLite itself can be written from multiple processes, but CubePi's
expectation is that a single agent instance owns a thread. If you
have multiple processes writing the same `agent.db`:

- Reads are safe.
- Concurrent writes to **different** threads are safe.
- Concurrent writes to the **same** thread will interleave and the
  resulting history won't make sense — you'll see two assistant
  messages back-to-back, etc.

If you need multi-process writers on shared threads, jump to
[Postgres](./postgres) which uses an advisory lock per thread.

## Where the file lives

Use an absolute path in production:

```python
SQLiteCheckpointer("/var/lib/myapp/agent.db")
```

Relative paths resolve against `os.getcwd()` at the moment of
`__aenter__`. The directory must exist; create it ahead of time.

## Backup and inspection

The file is a normal SQLite database. You can:

```bash
# Inspect history for a thread
sqlite3 agent.db "SELECT message_json FROM messages WHERE thread_id='user-42' ORDER BY id"

# Back up
cp agent.db agent.db.bak

# Vacuum to reclaim space (optional — file grows linearly with history)
sqlite3 agent.db "VACUUM"
```

## Common pitfalls

- **Used without `async with`** — `AssertionError: self._db is not
  None`. Always wrap in `async with`.
- **Two processes writing the same thread** — Interleaved history.
  Use Postgres or coordinate at the application layer.
- **WAL mode not enabled** — CubePi uses the default journal mode for
  portability. For a single-writer, many-reader app, enable WAL once
  via `sqlite3 agent.db "PRAGMA journal_mode=WAL"` for better read
  concurrency.
- **Forgetting to set `thread_id`** — Without it, the agent has no
  persistence binding. The checkpointer is silently ignored. Always
  pass both.

## See also

- [Postgres Checkpointing](./postgres) — for multi-instance deployments.
- [Custom Backends](./custom) — implement the protocol for Redis,
  DynamoDB, etc.
- [Recipes → Persistent Chat](../../recipes/persistent-chat) —
  end-to-end app with SQLite.
- [Recipes → Resumable Long Tasks](../../recipes/resumable-tasks) —
  agents that survive a process restart mid-tool.
