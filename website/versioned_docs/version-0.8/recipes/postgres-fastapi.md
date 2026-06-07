---
title: Postgres + FastAPI Service
description: "Deploy a FastAPI-backed CubePi agent with PostgresCheckpointer for production."
---

# Recipe: Postgres + FastAPI Service

A production-shaped HTTP service that fronts a CubePi agent: FastAPI
for routing, server-sent events for streaming, a shared
`PostgresCheckpointer` for persistence, and `thread_id` derived from
the authenticated user.

**Time to run:** 30 minutes.
**Deps:** `cubepi[postgres]`, `fastapi`, `uvicorn[standard]`,
`sse-starlette`, a running Postgres with the CubePi schema applied.

## Schema first

Before the service starts, run the CubePi schema migration. The
quickest way for this recipe:

```bash
psql "$DATABASE_URL" <<'SQL'
CREATE TABLE cubepi_threads (
    thread_id TEXT PRIMARY KEY,
    parent_thread_id TEXT REFERENCES cubepi_threads(thread_id),
    forked_at_seq BIGINT,
    extra JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE cubepi_messages (
    thread_id TEXT NOT NULL REFERENCES cubepi_threads(thread_id) ON DELETE CASCADE,
    seq BIGINT NOT NULL,
    role TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    payload BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (thread_id, seq)
) PARTITION BY HASH (thread_id);

-- 64 hash partitions
DO $$
BEGIN
  FOR i IN 0..63 LOOP
    EXECUTE format(
      'CREATE TABLE cubepi_messages_p%s PARTITION OF cubepi_messages FOR VALUES WITH (MODULUS 64, REMAINDER %s)',
      i, i
    );
  END LOOP;
END$$;

CREATE INDEX ix_cubepi_messages_metadata_gin
ON cubepi_messages USING gin (metadata jsonb_path_ops);

CREATE TABLE cubepi_schema_version (version INT PRIMARY KEY);
INSERT INTO cubepi_schema_version VALUES (1);
SQL
```

For a real deployment, generate this via Alembic — see
[Postgres Checkpointing → Bootstrapping via Alembic](../guides/checkpointing/postgres#bootstrapping-via-alembic).

## The service

```python title="service.py"
import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from cubepi import Agent
from cubepi.checkpointer import PostgresCheckpointer
from cubepi.providers.anthropic import AnthropicProvider


# --- App lifecycle ------------------------------------------------------

_provider = AnthropicProvider(provider_id="anthropic", api_key=os.environ["ANTHROPIC_API_KEY"])
_checkpointer: PostgresCheckpointer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _checkpointer
    _checkpointer = await PostgresCheckpointer(
        os.environ["DATABASE_URL"],
        min_pool_size=2,
        max_pool_size=20,
    ).__aenter__()
    yield
    await _checkpointer.__aexit__(None, None, None)


app = FastAPI(lifespan=lifespan)


# --- Auth (stub — replace with your real auth) -------------------------

async def current_user_id() -> str:
    # In production: decode JWT, look up session, etc.
    return "demo-user"


# --- Routes -------------------------------------------------------------

class PromptBody(BaseModel):
    text: str


@app.post("/chat/{conversation_id}/messages")
async def post_message(
    conversation_id: str,
    body: PromptBody,
    user_id: str = Depends(current_user_id),
):
    thread_id = f"{user_id}:{conversation_id}"

    async def event_generator() -> AsyncIterator[dict]:
        agent = Agent(
            model=_provider.model("claude-sonnet-4-6"),
            system_prompt="You are a helpful assistant.",
            checkpointer=_checkpointer,
            thread_id=thread_id,
        )

        queue: asyncio.Queue = asyncio.Queue()
        agent.subscribe(lambda e, s=None: queue.put_nowait(e))

        async def run():
            try:
                await agent.prompt(body.text)
            finally:
                queue.put_nowait(None)   # sentinel

        task = asyncio.create_task(run())

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                # Emit a small subset to the client.
                if event.type == "message_update" and event.stream_event.type == "text_delta":
                    yield {"event": "delta", "data": event.stream_event.delta}
                elif event.type == "tool_execution_start":
                    yield {"event": "tool_start", "data": event.tool_name}
                elif event.type == "agent_end":
                    yield {"event": "done", "data": ""}
        finally:
            await task

    return EventSourceResponse(event_generator())


@app.get("/chat/{conversation_id}/history")
async def get_history(
    conversation_id: str,
    user_id: str = Depends(current_user_id),
):
    thread_id = f"{user_id}:{conversation_id}"
    data = await _checkpointer.load(thread_id)
    if data is None:
        return {"messages": []}
    return {
        "messages": [m.model_dump(mode="json") for m in data.messages],
    }
```

Run:

```bash
pip install "cubepi[postgres]" fastapi "uvicorn[standard]" sse-starlette
export DATABASE_URL=postgresql://user:pass@localhost/cubepi
export ANTHROPIC_API_KEY=sk-…
uvicorn service:app --reload --port 8000
```

Test:

```bash
curl -N -X POST http://localhost:8000/chat/conv1/messages \
  -H "content-type: application/json" \
  -d '{"text":"hi"}'
# event: delta
# data: Hello
# event: delta
# data: !
# event: done
```

## Design notes

- **One `PostgresCheckpointer` per process, shared across requests.**
  It holds a connection pool; opening one per request would defeat
  the pool.
- **One `Agent` per request.** Agents own per-conversation state
  (steering queues, listeners). Don't reuse them.
- **`thread_id = f"{user_id}:{conversation_id}"`** — user isolation by
  prefix. The agent reads/writes only its own thread.
- **SSE for streaming.** Each text delta goes to the client as a
  separate event. Tool starts get their own event type — clients can
  render a "thinking" indicator without rebuilding event handling.
- **No load balancer affinity needed.** Because state is in Postgres,
  any service instance can pick up any conversation.

## Concurrency on the same thread

If a user double-clicks send, two `POST` requests arrive simultaneously.
Both create an `Agent` bound to the same `thread_id`. The Postgres
advisory lock serialises their appends, but the **in-memory** states
diverge — the second request's agent might not see the first's
in-progress message in its `agent.state.messages`.

For most chat UIs this is fine (the client controls send timing).
If you need strict ordering, add an application-layer mutex
(`asyncio.Lock` keyed by `thread_id`) or queue.

## Production hardening checklist

- **Auth:** Replace `current_user_id()` with real JWT / session
  validation.
- **Rate limiting:** Add a [`RateLimitMiddleware`](../guides/middleware/examples#rate-limiting)
  to the agent constructor, keyed by `user_id`.
- **Cost tracking:** Subscribe to `agent_end`, sum `usage` on each
  `AssistantMessage`, write to a billing table.
- **Observability:** Use `on_response` to capture `anthropic-*` rate
  headers; export to Prometheus.
- **Backups:** Postgres native — `pg_dump`, point-in-time recovery.
- **Graceful shutdown:** uvicorn's lifespan handler closes the pool;
  add `signal.signal(SIGTERM, ...)` if you have other resources.

## Common pitfalls

- **CubepiSchemaUninitialized at startup** — Your migrations didn't
  run. Apply the schema first.
- **Connection pool exhaustion** — Default `max_pool_size=10`. Bump
  it if your service has more concurrent agents than that.
- **SSE behind a load balancer** — Some LBs buffer SSE. Disable
  buffering (`X-Accel-Buffering: no` for nginx).
- **Long requests timing out** — Tool-heavy agents can run minutes.
  Set generous proxy timeouts and uvicorn `--timeout-keep-alive 600`.

## Run the example

A self-contained service template for this recipe is in the repository at
[`examples/postgres_fastapi.py`](https://github.com/cubeplexai/cubepi/blob/main/examples/postgres_fastapi.py).

```bash
git clone https://github.com/cubeplexai/cubepi && cd cubepi
uv sync --extra postgres
pip install fastapi "uvicorn[standard]" sse-starlette

export DATABASE_URL=postgresql://user:pass@localhost/cubepi
export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY [+ OPENAI_BASE_URL]

uvicorn examples.postgres_fastapi:app --reload --port 8000

# Test with curl:
curl -N -X POST http://localhost:8000/chat/conv1/messages \
  -H "content-type: application/json" \
  -d '{"text":"hi"}'
```

## See also

- [Postgres Checkpointing](../guides/checkpointing/postgres) — the
  backend in depth.
- [Persistent Chat](./persistent-chat) — the same flow with SQLite.
- [Multi-Provider Failover](./multi-provider-failover) — combine with
  this service for resilience.
