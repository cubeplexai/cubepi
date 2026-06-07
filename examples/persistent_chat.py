"""Persistent chat — recipe example.

A REPL chat that survives restarts. Conversation history is kept in a SQLite
file; pass a thread_id as the first argument to identify the user/session.

    uv run python examples/persistent_chat.py alice
    # Have a chat, then Ctrl-D.

    uv run python examples/persistent_chat.py alice
    # History is restored. Ask "what did I just tell you?" — the model remembers.

    uv run python examples/persistent_chat.py bob
    # Different thread, clean slate.

Requires: cubepi[sqlite]
Uses the Ark coding endpoint by default — override via CUBEPI_* env vars (see _provider.py).
"""

import asyncio
import sys

from cubepi import Agent
from cubepi.checkpointer import SQLiteCheckpointer

from _provider import MODEL_ID, provider


async def main(thread_id: str) -> None:
    async with SQLiteCheckpointer("chat.db") as cp:
        agent = Agent(
            model=provider.model(MODEL_ID),
            system_prompt="You are a concise, friendly assistant.",
            checkpointer=cp,
            thread_id=thread_id,
        )

        def on_event(event, signal=None):
            if event.type == "message_update" and event.stream_event.type == "text_delta":
                print(event.stream_event.delta, end="", flush=True)
            elif event.type == "agent_end":
                print()

        agent.subscribe(on_event)

        print(f"Chatting on thread {thread_id!r}. Ctrl-D to quit.\n")
        loop = asyncio.get_event_loop()
        while True:
            try:
                user_input = await loop.run_in_executor(None, input, "you> ")
            except EOFError:
                print()
                return
            if not user_input.strip():
                continue
            print("ai > ", end="", flush=True)
            await agent.prompt(user_input)


if __name__ == "__main__":
    thread_id = sys.argv[1] if len(sys.argv) > 1 else "default"
    asyncio.run(main(thread_id))
