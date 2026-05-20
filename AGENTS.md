# cubepi

Pythonic async-native agent framework (an alternative to langgraph). Async-first,
strongly typed, deliberately few dependencies. 

## Commands (always via `uv`)

```bash
uv sync --all-extras --dev               # install everything
uv run pytest tests/                      # run tests (asyncio_mode=auto)
uv run pytest tests/path/test.py::test -v # single test
uv run ruff check cubepi/ tests/
uv run ruff format --check cubepi/ tests/ # CI checks formatting, doesn't fix
```

CI runs pytest + ruff only. **There is no mypy step** — ignore the stale
`.mypy_cache/`. Tests run on Python 3.11–3.14 (3.14 is `continue-on-error`);
local default is 3.13 (`.python-version`).

## Architecture

`cubepi/` modules: `providers/` (LLM abstraction — `Provider` protocol returning
`MessageStream`; anthropic / openai / openai_responses / faux), `agent/`
(`agent.py` stateful class, `loop.py` stateless core algorithm, `tools.py`
execution engine), `middleware/`, `checkpointer/` (memory / sqlite / postgres),
`mcp/`, `tracing/` (OTel, optional), `cli/` (`cubepi trace`). See the README
"Architecture" section for the full annotated tree.

## Conventions & gotchas

- **Lean deps**: core deps are anthropic, openai, pydantic, pyyaml only.
  Everything else (sqlite, postgres, mcp, tracing, trace-cli) is an optional
  extra in `pyproject.toml`. Don't add a hard dependency without strong reason.
- **`cubepi.tracing` is lazily importable** (PEP 562 `__getattr__`): schema
  constants import without the opentelemetry SDK, so the trace CLI works on a
  `cubepi[trace-cli]`-only install. Don't add eager OTel imports to its
  `__init__.py`.
- **Tests use `FauxProvider`** for deterministic, no-API-call runs with realistic
  streaming. Prefer it over mocking providers.
- **`cubepi/cli/**` is excluded from codecov** (`codecov.yml` ignore).
- Packaged data: `cubepi/providers/catalog/data/*.yaml` ships in the wheel.

## Development workflow

This project follows a deliberate spec → plan → code pipeline. Honor it for any
non-trivial work.

**1. Set up an isolated worktree first.** When a new requirement comes in, before
doing any work create a **date-prefixed worktree on a date-prefixed branch**
(e.g. `.worktrees/YYYY-MM-DD-<topic>` on branch `YYYY-MM-DD-<topic>`). Never work
directly on `main`. Subagents that write code must also use `isolation:
"worktree"`.

**2. Spec — collaborate, don't go autonomous.** The spec stage is interactive:
talk through and confirm requirements with the user before writing the spec.
While forming it, research prior art — **pi-agent-core, langgraph, claude code** —
to find best practices, but let our own requirements and design philosophy drive
the result. Call out notable divergences ("library does X, cubepi does Y because
Z") so they can be reviewed. Specs go in `dev/specs/` (dated
`YYYY-MM-DD-<topic>.md`), plans in `dev/plans/`.

Codex reviews happen in **two distinct phases** — local (step 3) and on the PR
(step 5).

**3. Local codex review loop — ask before entering it.** Once spec/plan/code are
ready, **check with the user before starting the local codex review loop.** If
they say go: run it autonomously without stopping to ask mid-flow — write the spec
→ codex review; write the plan → codex review; write the code → codex review and
iterate until codex is OK. Use the `codex:rescue` subagent for these reviews.

**4. Open the PR.** Never commit to `main` directly (it's a protected branch).
Open a PR from the worktree branch.

**5. PR codex review loop.** Opening the PR triggers an automatic codex review.
After that it is **not** automatic — drive it manually:

- Poll for feedback every **~2 minutes** (check the PR's review comments).
- Resolve every piece of feedback, pushing fixes to the branch.
- Once resolved, **reply `@codex` on the PR to request another review pass.**
- Repeat poll → fix → `@codex` until codex reports no remaining issues.

Merge only after the PR codex review is clean and CI passes.
