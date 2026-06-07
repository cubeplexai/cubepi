# Examples

Runnable scripts demonstrating CubePi features.

Run any example with `uv`:

```bash
uv run python examples/<name>.py
```

## Provider setup

Recipe examples use a real LLM. Set one of the following before running:

```bash
# Anthropic (or Anthropic-compatible endpoint):
export ANTHROPIC_API_KEY=sk-ant-...
export ANTHROPIC_BASE_URL=https://...   # optional, for compatible endpoints
export MODEL=claude-sonnet-4-6          # optional, this is the default

# OpenAI (or OpenAI-compatible endpoint):
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://...      # optional, for compatible endpoints
export MODEL=gpt-4o                     # optional, this is the default
```

The shared `_provider.py` module reads these env vars and exposes `provider`
and `MODEL_ID` for each example to import. `ANTHROPIC_API_KEY` takes priority
when both are set.

## Recipes

| Example | What it shows | Extra deps |
|---|---|---|
| [`weather_agent.py`](weather_agent.py) | Tool calling, streaming output, Ctrl-C cancellation | `httpx` |
| [`persistent_chat.py`](persistent_chat.py) | SQLite-backed chat that survives restarts | `cubepi[sqlite]` |
| [`multi_provider_failover.py`](multi_provider_failover.py) | Automatic failover between providers on error | — |
| [`ask_user_form.py`](ask_user_form.py) | HITL multi-question form via `ask_user_tool` | — |
| [`sandbox_confirm.py`](sandbox_confirm.py) | `ApprovalPolicyMiddleware` — auto-allow, deny, or confirm tool calls | — |
| [`resumable_tasks.py`](resumable_tasks.py) | Crash-resilient tasks with idempotent tools and checkpointing | `cubepi[sqlite]` |
| [`postgres_fastapi.py`](postgres_fastapi.py) | Production HTTP service: FastAPI + SSE streaming + Postgres | `cubepi[postgres]` `fastapi` `uvicorn[standard]` `sse-starlette` |

### weather_agent.py

```bash
uv run --with httpx python examples/weather_agent.py
```

### persistent_chat.py

```bash
uv run python examples/persistent_chat.py alice
# Chat, then Ctrl-D. Re-run same command — history is restored.

uv run python examples/persistent_chat.py bob
# Different thread, clean slate.
```

Requires: `cubepi[sqlite]`

### multi_provider_failover.py

```bash
uv run python examples/multi_provider_failover.py
```

Runs with a bad primary key to demonstrate failover, then succeeds via the
real secondary key.

### ask_user_form.py

```bash
uv run python examples/ask_user_form.py
```

The host loop answers the form programmatically (simulating a UI). In a real
app you'd render the questions to a frontend.

### sandbox_confirm.py

```bash
uv run python examples/sandbox_confirm.py
```

Shows auto-allow (`echo`, `ls`, `cat`), hard-deny (`rm -rf /`), and
human-confirm (any other command, auto-approved in the simulated host).

### resumable_tasks.py

```bash
# Start a job:
uv run python examples/resumable_tasks.py job-1 start

# Kill mid-flight, then resume:
uv run python examples/resumable_tasks.py job-1
```

Items already processed are skipped on resume (idempotent tool backed by
a file-based job store in `/tmp/cubepi-jobs/`).

Requires: `cubepi[sqlite]`

### postgres_fastapi.py

```bash
uv sync --extra postgres
export DATABASE_URL=postgresql://user:pass@localhost/cubepi
uv run --with fastapi --with "uvicorn[standard]" --with sse-starlette \
  uvicorn examples.postgres_fastapi:app --reload --port 8000

curl -N -X POST http://localhost:8000/chat/conv1/messages \
  -H "content-type: application/json" \
  -d '{"text":"hi"}'
```

## Checkpointing (service integration)

| Example | What it shows | Requires |
|---|---|---|
| [`checkpointing_postgres.py`](checkpointing_postgres.py) | Persist an agent conversation in Postgres and resume it after a simulated restart | A reachable Postgres |
| [`checkpointing_mysql.py`](checkpointing_mysql.py) | Same, on MySQL 8.0.13+ | A reachable MySQL |

Both use `FauxProvider` (no API key needed) and create a **throwaway database**
that is dropped on exit — safe to re-run against a dev server.

```bash
CUBEPI_PG_DSN=postgresql://user:pass@host:5432/dbname \
    uv run python examples/checkpointing_postgres.py

CUBEPI_MYSQL_DSN=mysql://user:pass@host:3306/dbname \
    uv run python examples/checkpointing_mysql.py
```

Each script bootstraps the CubePi schema inline so it runs standalone, but in
production the schema is owned by your host application's Alembic migration.
See the host-integration runbooks for the migration recipe and version-upgrade flow:

- [`cubepi/checkpointer/postgres/README.md`](../cubepi/checkpointer/postgres/README.md)
- [`cubepi/checkpointer/mysql/README.md`](../cubepi/checkpointer/mysql/README.md)
- User guides: [Postgres](../website/docs/guides/checkpointing/postgres.md) ·
  [MySQL](../website/docs/guides/checkpointing/mysql.md)
