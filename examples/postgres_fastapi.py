"""Postgres + FastAPI service — recipe example.

A production-shaped HTTP service: FastAPI for routing, server-sent events for
streaming, a shared PostgresCheckpointer for persistence.

    pip install "cubepi[postgres]" fastapi "uvicorn[standard]" sse-starlette
    export DATABASE_URL=postgresql://user:pass@localhost/cubepi
    uvicorn examples.postgres_fastapi:app --reload --port 8000

    # Then test:
    curl -N -X POST http://localhost:8000/chat/conv1/messages \\
      -H "content-type: application/json" \\
      -d '{"text":"hi"}'

Requires: cubepi[postgres], fastapi, uvicorn[standard], sse-starlette
Set ANTHROPIC_API_KEY or OPENAI_API_KEY before running (see _provider.py).
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from cubepi import Agent
from cubepi.checkpointer import PostgresCheckpointer

# Works both when run as a script (python examples/postgres_fastapi.py) and
# when loaded as a package by uvicorn (uvicorn examples.postgres_fastapi:app).
sys.path.insert(0, str(Path(__file__).parent))
from _provider import MODEL_ID, provider

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


async def current_user_id() -> str:
    # Replace with real auth (JWT decode, session lookup, etc.)
    return "demo-user"


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
            model=provider.model(MODEL_ID),
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
                queue.put_nowait(None)

        task = asyncio.create_task(run())
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
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
    return {"messages": [m.model_dump(mode="json") for m in data.messages]}
