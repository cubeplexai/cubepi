# Rename CubePi → CubeLoop — Implementation Plan

**Goal:** Rebrand the library to CubeLoop / `cubeloop` in 0.14.0: package tree, imports, CLI, OTel vendor namespace, Postgres/MySQL table names, current docs, and a transitional `cubepi` PyPI shim. GitHub repo is renamed in place (not a new repo). Canonical site is `https://cubeloop.dev`.

**Spec:** [`dev/specs/2026-09-09-rename-cubeloop.md`](../specs/2026-09-09-rename-cubeloop.md)

**Architecture:** One source tree (`cubeloop/`). A second hatchling project at `compat/cubepi/` publishes the shim wheel. Schema v6 is a `RENAME TABLE` (PG/MySQL only). Tracing writes `cubeloop.*` and the CLI reads both namespaces.

**Tech stack:** Python 3.11+, hatchling, uv workspace, pytest, ruff, mypy, Docusaurus current docs (EN + zh-Hans).

Work in `.worktrees/2026-09-09-rename-cubeloop` on branch `2026-09-09-rename-cubeloop`. Never commit to `main`.

---

## Do not touch

These stay CubePi — they document what already shipped:

- `website/versioned_docs/**`
- `website/versioned_sidebars/**`
- `website/i18n/zh-Hans/docusaurus-plugin-content-docs/version-*`
- Historical `CHANGELOG.md` sections (`## [0.13.6]` and older). Add new text under `[Unreleased]` only.
- `dev/specs/**` and `dev/plans/**` except this pair.
- SQLite table names (`messages`, `runs`, `thread_extra`, …).
- Git history.

A bulk `cubepi` → `cubeloop` replace that hits those trees is a bug.

---

## File structure

- **Move** `cubepi/` → `cubeloop/`
- **Move** `skills/cubepi/` → `skills/cubeloop/`, `skills/cubepi-trace/` → `skills/cubeloop-trace/`
- **Move** `website/static/img/brand/cubepi-*` → `cubeloop-*` (same artwork)
- **Create** `compat/cubepi/` (shim distribution)
- **Create** `website/docs/migration/from-cubepi.md` + zh-Hans mirror
- **Create** `tests/compat/test_cubepi_shim.py`
- **Create** `tests/cli/trace/fixtures/` (0.13 JSONL + 0.14 JSONL)
- **Modify** `pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml`, `codecov.yml`, `AGENTS.md`, `CLAUDE.md`, `README.md`
- **Modify** `.github/workflows/{ci,docs,publish}.yml`
- **Modify** checkpointer models / runtime SQL / alembic helpers (PG + MySQL)
- **Modify** `cubeloop/tracing/schema.py` and every writer of `cubepi.*` attributes
- **Modify** current docs, `website/src/**`, `docusaurus.config.ts`, `sidebars.ts`, `website/static/_worker.js`, `website/scripts/build_api_reference.py`
- **Modify** examples, tests (except frozen historical schema SQL that still creates `cubepi_*` as a v5 fixture)

---

## Task 1: Mechanical package-tree rename

Do the filesystem move and import-path rewrite **before** identifier/table/OTel string changes, so the suite can go green on `import cubeloop` while names are still `Cubepi*` / `cubepi_*`.

- [ ] `git mv cubepi cubeloop`
- [ ] `git mv skills/cubepi skills/cubeloop`
- [ ] `git mv skills/cubepi-trace skills/cubeloop-trace`
- [ ] `git mv website/static/img/brand/cubepi-logo.svg website/static/img/brand/cubeloop-logo.svg` (and `.png`, `cubepi-social-preview.svg` / `.png`)
- [ ] Update `pyproject.toml`:
  - `name = "cubeloop"`
  - version `0.14.0` (this PR *is* the breaking 0.14 surface; tag comes later)
  - URLs → `https://cubeloop.dev`, `https://github.com/cubeplexai/cubeloop`
  - `[project.scripts] cubeloop = "cubeloop.cli.__main__:main"`
  - `[tool.mypy] files = ["cubeloop"]`
  - `[tool.hatch.build.targets.wheel] packages = ["cubeloop"]`
  - `[tool.uv.workspace] members = ["compat/cubepi"]` (member lands in Task 5; can add the table now or then)
- [ ] Rewrite imports in **allowed** paths only:

Allowed: `cubeloop/`, `tests/`, `examples/`, `website/docs/`, `website/src/`, `website/scripts/`, `website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/`, `website/docusaurus.config.ts`, `website/sidebars.ts`, `website/static/_worker.js`, `README.md`, `AGENTS.md`, `CLAUDE.md`, `.github/`, `.pre-commit-config.yaml`, `codecov.yml`, `skills/`.

Replace, in order (longest first):

1. `from cubepi` / `import cubepi` → `cubeloop`
2. `"cubepi/` path segments in CI/mypy/ruff/codecov (`cubepi/cli/**` → `cubeloop/cli/**`)
3. Do **not** yet rewrite `cubepi_threads`, `cubepi.run_id`, `CubepiSchemaError`, `CUBEPI_*` — those are later tasks.

- [ ] `uv lock && uv sync --all-extras --dev`
- [ ] CI workflow: `--cov=cubeloop`, `import cubeloop`, ruff/mypy paths, docs.yml `cubepi/**` → `cubeloop/**` and the API-mdx guard `cubepi-*.mdx` → `cubeloop-*.mdx`
- [ ] `website/scripts/build_api_reference.py`: `MODULES` tuples `cubeloop.*`; generated page ids `cubeloop-agent` etc.
- [ ] `website/sidebars.ts` current API ids `api/cubeloop-agent` …
- [ ] `website/docs/api/index.mdx` + zh-Hans current mirror

Green bar for this task is “the tree imports as cubeloop”:

```bash
uv run python -c "import cubeloop; print(cubeloop.__all__)"
uv run ruff check cubeloop/ tests/
uv run ruff format --check cubeloop/ tests/
uv run mypy cubeloop
uv run pytest tests/test_init.py tests/test_package.py tests/cli/test_entrypoint.py -q
```

Expect remaining failures only where tests still assert the string `"cubepi"` as a table/attr/package name. Do not “fix” those by reverting the import path.

---

## Task 2: Public Python identifiers

- [ ] `CubepiSchemaError` / `Uninitialized` / `Mismatch` → `Cubeloop*` in `cubeloop/checkpointer/exceptions.py` and the PG/MySQL re-exports
- [ ] ORM: `cubepi_metadata` → `cubeloop_metadata`; `CubepiBase` / `CubepiThread` / `CubepiMessage` / `CubepiRun` / `CubepiHitlAnswer` / `CubepiSchemaVersion` → `Cubeloop*`
- [ ] Tracing constants: `CUBEPI_RUN_ID` → `CUBELOOP_RUN_ID` (and the rest). **Leave the string values as `cubepi.*` until Task 4** so Task 2 is a pure identifier rename.
- [ ] `_pkg_version("cubeloop")`
- [ ] Logger / tracer *names* that are Python logging namespaces can wait for Task 4 (they are OTel/log vendor strings).
- [ ] cubeloop does **not** keep `Cubepi*` aliases. The shim is the compatibility layer.

```bash
uv run pytest tests/checkpointer/test_init.py tests/checkpointer/test_exceptions.py tests/test_init.py -q
```

---

## Task 3: Schema v5 → v6 (Postgres + MySQL)

`EXPECTED_SCHEMA_VERSION = 6`. Runtime SQL and current-schema helpers use `cubeloop_*`. Historical v1–v5 helpers keep emitting `cubepi_*` because they run against databases that still have those names.

### 3.1 Footgun — do not “fix” old helpers by renaming their SQL

Today `upgrade_v3_to_v4_op()` calls `create_runs_partitions_op()`. After this task `create_runs_partitions_op()` must emit `cubeloop_runs_pXX PARTITION OF cubeloop_runs`. If the v3→v4 helper still calls it, a v3 database will try to partition a table that does not exist yet.

- [ ] Inline the **old** `cubepi_runs_pXX` / `cubepi_messages_pXX` DDL into `upgrade_v3_to_v4_op` (and any other historical helper that currently delegates to a current-schema factory).
- [ ] Point `create_message_partitions_op` / `create_runs_partitions_op` (Postgres) and the current MySQL CREATE helpers at **new** names, for fresh installs.

`write_schema_version_op()` must **not** hardcode only `cubeloop_schema_version`. Hosts copy `op.execute(write_schema_version_op())` into each already-shipped Alembic revision. After 0.14 those revisions still run against `cubepi_schema_version` when a fresh DB replays history.

Return dialect SQL that writes `EXPECTED_SCHEMA_VERSION` to `cubeloop_schema_version` if that table exists, else to `cubepi_schema_version`.

- Postgres: one `DO $$ … $$` block (single statement).
- MySQL: **must** remain a `;`-joined list of standalone statements. Hosts do `for stmt in write_schema_version_op().split(";"): op.execute(stmt)`. Do not emit `BEGIN`/`IF`/`THEN` compound SQL. Use `SET @cp_ver_new = (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name = 'cubeloop_schema_version')`, then `SET @cp_del = IF(@cp_ver_new > 0, 'DELETE FROM cubeloop_schema_version WHERE version <> N', 'DELETE FROM cubepi_schema_version WHERE version <> N')`, then `PREPARE` / `EXECUTE` / `DEALLOCATE PREPARE`, then the same for `INSERT IGNORE`. Tests execute via that split loop, not `cursor.execute(whole_string)`.

`upgrade_v5_to_v6_op()` only renames, including the version table. The host then calls `write_schema_version_op()`, which now hits the new table.

Do **not** change historical helpers to call `write_schema_version_op` themselves.

### 3.2 Postgres `upgrade_v5_to_v6_op()`

Rename, in one helper:

| Old | New |
|---|---|
| `cubepi_threads` | `cubeloop_threads` |
| `cubepi_messages` + `cubepi_messages_p00`…`p63` | `cubeloop_messages` + matching `pXX` |
| `cubepi_runs` + `cubepi_runs_p00`…`p63` | `cubeloop_runs` + matching `pXX` |
| `cubepi_hitl_answers` | `cubeloop_hitl_answers` |
| `cubepi_schema_version` | `cubeloop_schema_version` |
| `ix_cubepi_*` (the three we named) | `ix_cubeloop_*` |

`ALTER TABLE … RENAME TO` on a partitioned parent does **not** rename children — loop `PARTITION_COUNT`. Auto-generated PK/FK constraint names may keep `cubepi_*`; do not chase them unless a test asserts the name.

Idempotency (Postgres): complete iff `cubeloop_threads` **and** `cubeloop_schema_version` both exist. `cubeloop_threads` alone is not a no-op sentinel.

### 3.3 MySQL `upgrade_v5_to_v6_op()`

One `RENAME TABLE a TO a2, b TO b2, …` covering **all** v5 tables including `cubepi_schema_version`. KEY partitions ride with the table. Do not rename the version table in a second statement.

Same complete-state idempotency as Postgres. Missing `cubepi_schema_version` makes the atomic rename fail as a whole; `_verify_schema` step 3 explains recovery. Do not split this helper on `;` — it is one statement.

### 3.4 Runtime SQL

Every string in `cubeloop/checkpointer/postgres/checkpointer.py` and `mysql/checkpointer.py` (`FROM cubepi_messages`, `INSERT INTO cubepi_threads`, mismatch hint text) uses the new names. `_schema_mismatch_hint` must name `upgrade_v5_to_v6_op()` when `actual < 6`. MySQL's current hint still says `add_run_id_column_op()` — replace it with the same step-list helper Postgres uses.

`_verify_schema` (both backends):

1. SELECT from `cubeloop_schema_version`. If the table exists, compare to 6.
2. If the new table is missing (`UndefinedTable` / MySQL 1146): SELECT `cubepi_schema_version`. If **that** exists (any stored version, including 6), raise `CubeloopSchemaMismatch(expected=6, actual=<row>)` with a hint that only names `upgrade_v5_to_v6_op()` + `write_schema_version_op()`.
3. If neither version table exists but `cubepi_threads` or `cubeloop_threads` does → `CubeloopSchemaMismatch(expected=6, actual=0)` with a “version table missing, do not CREATE fresh v6 on live data” hint (migrate guide recovery: insert version 5 then v5→v6).
4. If neither version table nor data tables exist → `CubeloopSchemaUninitialized`.

A host who deploys 0.14 before the v6 Alembic revision must not see “tables not found”.

### 3.5 Tests

Split fixtures. Today's `_setup_schema` CREATEs `cubepi_*` then calls `add_pending_request_column_op`, `create_message_partitions_op`, `upgrade_v3_to_v4_op`, `upgrade_v4_to_v5_op`, `write_schema_version_op`. After Task 3.1 those historical helpers still emit `cubepi_*` while current partition factories emit `cubeloop_*`. Do **not** “just rename the CREATE TABLE” and keep calling the old helpers.

- [ ] **Fresh-v6 builder**: one canonical v6 DDL source (same SQL the backend README / examples ship) + current `create_message_partitions_op` / `create_runs_partitions_op`. No historical helpers. Assert it matches example DDL (string or information_schema).
- [ ] **Historical-v5 builder**: keep creating `cubepi_*` and calling the frozen v1–v5 helpers; `write_schema_version_op` lands version 6-or-5 in `cubepi_schema_version` (because the new table does not exist yet). Used only as the input to v5→v6 tests and as the unmigrated-v5 mismatch case.
- [ ] **Full-chain replay test** (PG + MySQL): start empty, run the documented helper sequence a host would copy (v1-shaped CREATE + pending_request + run_id + partitions + v3→v4 + v4→v5 + write + **v5→v6** + write). Then open the checkpointer and round-trip a message. This is the test that catches `write_schema_version_op` targeting the wrong table.
- [ ] `test_expected_schema_version_is_5` → `…_is_6`; mismatch-hint tests include v5→v6
- [ ] **Keep** v2/v3 fixture SQL as `CREATE TABLE cubepi_schema_version …`. Update asserted `expected=6`. Unmigrated v5 (historical-v5 builder, no v6 helper) raises `CubeloopSchemaMismatch`, **not** Uninitialized.
- [ ] New tests (PG + MySQL):
  1. historical-v5 + `upgrade_v5_to_v6_op` + `write_schema_version_op` → checkpointer enters, message round-trips.
  2. Helper is idempotent on an already-v6 database (both `cubeloop_threads` and `cubeloop_schema_version` present).
  3. `information_schema` shows `cubeloop_threads` / `cubeloop_messages` / `cubeloop_runs` / `cubeloop_hitl_answers` / `cubeloop_schema_version`; old names are gone.
  4. Data tables present, no version table → mismatch `actual=0`, not Uninitialized.
  5. MySQL `write_schema_version_op()` split-and-execute loop (README contract) succeeds on both a v5-named and a v6-named version table.
- [ ] Env readers: `CUBELOOP_TEST_PG_DSN` then fallback `CUBEPI_TEST_PG_DSN` (same for MySQL / MCP HTTP / example DSNs). CI env keys become `CUBELOOP_*`. Ephemeral DB names `cubepi_test_*` / `cubepi_example_*` → `cubeloop_test_*` / `cubeloop_example_*`.
- [ ] Examples + backend READMEs: v6 CREATE TABLE under new names, plus a short “existing v5 DB: call `upgrade_v5_to_v6_op`” note.

```bash
uv run pytest tests/checkpointer/ -v
```

SQLite tests must stay green with **no** table rename.

---

## Task 4: OTel vendor namespace

### 4.1 Write path — cubeloop only

Start from `rg -n "cubepi\\." cubeloop/` after Task 1, not from schema.py alone. Change **string values** (constants already renamed in Task 2) **and** every remaining literal/f-string.

Schema constants plus the 0.13.6 leftovers the first inventory missed:

```
cubepi.run_id / cubepi.thread_id / cubepi.agent.* / cubepi.turn.* / cubepi.llm.* / cubepi.tool.*
cubepi.run.outcome / cubepi.aborted
cubepi.turn                  (span name)
SCOPE_NAME cubepi.tracing
cubepi.metadata.<key>
cubepi.oneshot.operation
cubepi.tags
cubepi.fork.src_thread_id / cubepi.fork.after_run_id
cubepi.agent.fork_once       (span name)
error.type cubepi.aborted / cubepi.error / cubepi.tool.error
cubepi_error_type_for() → "cubeloop.aborted"
ContextVars cubepi.tracing.run_metadata / run_tags / active_run
```

Also the non-schema literals (grep `"cubepi` in `cubeloop/` after Task 1):

| File | Old | New |
|---|---|---|
| `mcp/_tracing.py` | `_SCOPE_NAME = "cubepi.mcp"`, `error.type = "cubepi.aborted"` | `cubeloop.mcp` / `cubeloop.aborted` |
| `hitl/_trace.py` | `get_tracer("cubepi.hitl")` | `cubeloop.hitl` |
| `agent/agent.py` | `get_tracer("cubepi.agent")`, span `cubepi.agent.fork_once` | `cubeloop.agent` / `cubeloop.agent.fork_once` |
| `providers/base.py` | logger `"cubepi.providers"` | `cubeloop.providers` |
| `cli/trace/loader.py` | `_META_PREFIX = "cubepi.metadata."` | dual-read, see 4.2 |
| `cli/trace/render.py` | title `"cubepi runs"` | `"cubeloop runs"` |
| `cli/__main__.py` | `prog="cubepi"` | `prog="cubeloop"` |
| `cli/trace/loader.py` | `DEFAULT_DIR = Path("./cubepi-traces")` | `./cubeloop-traces` |

No dual-write. `gen_ai.*` unchanged.

Add schema constants for the metadata prefix and oneshot operation if they are currently ad-hoc f-strings — they should live next to the other `CUBELOOP_*` keys.

### 4.2 Read path — both namespaces

CLI (and any in-process reader that keys off schema constants) accepts old and new:

- Span name `cubepi.turn` **or** `cubeloop.turn`
- Attribute lookup: try `cubeloop.run_id`, fall back to `cubepi.run_id` (same for tags, fork attrs, outcome, metadata prefix, oneshot)
- Abort is **not** a plain key fallback. `Span.is_aborted` today reads only the boolean. Treat as aborted if `cubeloop.aborted` or `cubepi.aborted` is true **or** `error.type` is `cubeloop.aborted` or `cubepi.aborted`.

Helper: `schema.attr(attrs, CUBELOOP_RUN_ID)` for ordinary keys. Abort stays a dedicated predicate so `error.type` is not forgotten. Do not scatter `or attrs.get("cubepi…")` in every call site.

### 4.3 Tests

- [ ] `tests/tracing/test_lazy_import.py`: `CUBELOOP_RUN_ID == "cubeloop.run_id"` still imports without OpenTelemetry
- [ ] Existing tracing tests assert new keys (`cubeloop.tags`, fork attrs, `error.type` values, oneshot)
- [ ] New CLI tests with fixtures:
  - `v013_cubepi.jsonl` — `cubepi.turn`, `cubepi.run_id`, `cubepi.metadata.user_id`, `cubepi.tags`
  - `v013_abort_boolean.jsonl` — `cubepi.aborted=true`
  - `v013_abort_error_type_only.jsonl` — `error.type="cubepi.aborted"` and **no** boolean (must still show aborted)
  - `v013_fork.jsonl` / `v013_oneshot.jsonl`
  - `v014_cubeloop.jsonl` — new keys only
  - `cubeloop trace view` renders both generations; `ls --meta` matches both prefixes
- [ ] `rg "cubepi\\." cubeloop/` is empty except comments that quote the old name in the migrate guide / dual-read fallback table in schema.py

```bash
uv run pytest tests/tracing/ tests/cli/ tests/mcp/test_adapter.py -v
```

---

## Task 5: `cubepi` shim distribution

`compat/cubepi/` is a second hatchling project in the uv workspace.

```
compat/cubepi/
  pyproject.toml
  README.md          # deprecation banner; classifier Inactive
  cubepi/
    __init__.py      # warn + re-export cubeloop public API
    _redirect.py     # MetaPathFinder: cubepi.* → cubeloop.* on demand
    __main__.py      # warn, dispatch to cubeloop.cli
```

`pyproject.toml`:

- `name = "cubepi"`, `version = "0.14.0"` (keep in lockstep with root; exact pin, not a range)
- `dependencies = ["cubeloop==0.14.0"]`
- extras forward 1:1: `sqlite = ["cubeloop[sqlite]==0.14.0"]` (same for postgres, mysql, mcp, tracing, tracing-otlp, trace-cli)
- `[project.scripts] cubepi = "cubepi.__main__:main"`
- `[tool.uv.sources] cubeloop = { workspace = true }` so local `uv sync` uses the workspace package; `uv build` must emit the PyPI pin in the wheel metadata (verify with `unzip -p dist/cubepi-*.whl '*.dist-info/METADATA' | grep Requires-Dist`)

### 5.1 Import redirect

`cubepi/__init__.py` must **not** `sys.modules['cubepi'] = cubeloop` (breaks relative imports inside cubeloop). A finder that only writes `sys.modules` and returns `None` is also insufficient: `from cubepi.providers.anthropic import …` has to resolve `cubepi.providers` first, and with no `ModuleSpec` import falls through to a missing physical package.

Ship a `MetaPathFinder` + alias `Loader`:

1. `import cubepi` loads the shim `__init__.py` (finder ignores the root name).
2. `__init__.py` prints the spec warning to **stderr once per process**, emits `DeprecationWarning`, then `from cubeloop import *` and copies `__all__` / `__version__`.
3. Finder matches `fullname.startswith("cubepi.")`. `find_spec` returns `ModuleSpec(fullname, AliasLoader(real_name), origin=real.origin, is_package=…)`. Package specs set `submodule_search_locations`.
4. `AliasLoader.create_module` returns `importlib.import_module(real_name)` — the **same object** already in `sys.modules[real_name]`. `exec_module` is a no-op. cubeloop relative imports keep `__name__` under `cubeloop.*`.
5. Do **not** `walk_packages` / eagerly import `cubeloop.tracing` (lazy OTel import must survive `from cubepi.tracing import schema`).

PEP 517 wheels cannot print during `pip install`. Do not add a `.pth` that prints on every interpreter start. Do not raise in `setup()`.

### 5.2 Tests (`tests/compat/test_cubepi_shim.py`)

- [ ] `import cubepi` captures stderr containing `renamed to 'cubeloop'` and a `DeprecationWarning`
- [ ] `cubepi.Agent is cubeloop.Agent`
- [ ] `from cubepi.providers.anthropic import AnthropicProvider` is the cubeloop class
- [ ] `from cubepi.checkpointer.postgres import PostgresCheckpointer` works
- [ ] Subprocess with OpenTelemetry hidden: `from cubepi.tracing import schema` succeeds (lazy)
- [ ] `python -m cubepi trace --help` exits 0 and prints the warning on stderr
- [ ] Second `import cubepi` in the same process does not reprint the stderr banner

Workspace tests are not enough (they resolve `cubeloop` via the live project). Task 8 adds a clean-venv wheel smoke.

```bash
uv run pytest tests/compat/ -v
```

---

## Task 6: Current docs, site, skills, changelog

Allowed trees only (see Do not touch).

- [ ] `CubePi` → `CubeLoop`, `cubepi` → `cubeloop`, `cubepi.ai` → `cubeloop.dev`, `cubeplexai/cubepi` → `cubeplexai/cubeloop` in:
  - `README.md`, `AGENTS.md`, `CLAUDE.md`
  - `website/docs/**`, `website/src/**`, `website/docusaurus.config.ts` (`title`, `url`, `projectName`, logos, editUrl, keywords)
  - `website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/**`
  - `website/static/_worker.js`: `cubeloop.pages.dev` → `cubeloop.dev`
  - `examples/**`, `skills/cubeloop/**`, `skills/cubeloop-trace/**` (frontmatter `name: cubeloop` / `cubeloop-trace`)
  - checkpointer READMEs
- [ ] New page `website/docs/migration/from-cubepi.md` (+ zh-Hans current):
  - `pip install cubeloop` / import map
  - shim: `pip install -U cubepi` still works, warns, re-exports
  - alembic v5→v6 (`upgrade_v5_to_v6_op`)
  - OTel attribute map + “CLI still reads old JSONL”
  - env var map
  - GitHub redirect
- [ ] Add it to `website/sidebars.ts` Migration category
- [ ] `CHANGELOG.md` `[Unreleased]`: breaking rename, shim, schema v6, OTel namespace. Do not edit older sections. Compare-URL footer can wait for the GitHub rename; use `cubeplexai/cubeloop` now (redirects after rename).
- [ ] `dev/runbooks/cut-doc-version.md`: s/CubePi/CubeLoop/ and s/cubepi/cubeloop/ in the **runbook prose** (it is operational, not a frozen snapshot).

```bash
cd website && pnpm build && pnpm test
```

If browser tools are available, open `/docs/migration/from-cubepi`, `/docs/getting-started/installation`, and the homepage; check desktop + mobile. Frozen `/docs/0.13/` must still say CubePi.

---

## Task 7: Publish workflow

`.github/workflows/publish.yml` currently `uv build`s one dist.

- [ ] Build **both** wheels + sdists into `dist/`:
  1. root → `cubeloop-0.14.0-*`
  2. workspace member `cubepi` → `cubepi-0.14.0-*`
- [ ] Upload the whole `dist/` to TestPyPI then PyPI. Set `skip-existing: true` on both `pypa/gh-action-pypi-publish` steps so a partial TestPyPI attempt (cubeloop uploaded, cubepi rejected) can be retried without hitting “file already exists”.
- [ ] Document in the workflow comment (or a short `dev/runbooks/rename-github-and-pypi.md`) the out-of-repo steps — they are **not** this PR:

  1. Land this PR on `main`.
  2. Rename GitHub repo `cubepi` → `cubeloop` (Settings → Rename). Do not create a new repo.
  3. Cloudflare: custom domain `cubeloop.dev`; 301 `cubepi.ai` → `cubeloop.dev`; Pages GitHub connection.
  4. **PyPI and TestPyPI**: create project `cubeloop` + OIDC publisher for `cubeplexai/cubeloop` on **both** indexes (workflow `publish.yml`, matching environment names). Update the existing `cubepi` publisher's repository name on both indexes after the GitHub rename. Verify publisher identity before tagging.
  5. Tag `v0.14.0`.
  6. Cut docs version 0.14 per `cut-doc-version.md` **after** current/ is CubeLoop.

---

## Task 8: Verify

```bash
uv sync --all-extras --dev
uv run ruff check cubeloop/ tests/ compat/cubepi/
uv run ruff format --check cubeloop/ tests/ compat/cubepi/
uv run mypy cubeloop
uv run pytest tests/ -v
cd website && pnpm build && pnpm test
```

Sanity greps (must be empty except Do-not-touch trees and the migrate page / CHANGELOG Unreleased / shim / historical alembic SQL):

```bash
# Current library must not import cubepi
rg -n "from cubepi|import cubepi" cubeloop tests examples website/docs website/src \
  website/scripts website/i18n/zh-Hans/docusaurus-plugin-content-docs/current \
  README.md AGENTS.md CLAUDE.md skills --glob '!**/migration/from-cubepi.md'

# Current schema helpers / runtime SQL must not still write cubepi_* table names
# (historical upgrade_vN helpers and v2/v3 fixtures MAY still contain cubepi_*)
rg -n "INSERT INTO cubepi_|FROM cubepi_|CREATE TABLE cubepi_" cubeloop/checkpointer --glob '*checkpointer.py'
```

`uv build` at repo root and `uv build` for the cubepi member; inspect METADATA for the exact `Requires-Dist: cubeloop==0.14.0`.

Clean-venv wheel smoke (cwd **outside** the checkout). Use the host `uv`, not a venv-local `uv` (there isn’t one). `--no-index` cannot resolve `anthropic` / `pydantic` / `sqlalchemy` / `asyncpg`; pin only our two wheels from `dist/` and let the package index fill the rest:

```bash
REPO=/home/chris/cubepi/.worktrees/2026-09-09-rename-cubeloop
uv venv /tmp/cubeloop-smoke
uv pip install --python /tmp/cubeloop-smoke/bin/python \
  --find-links "$REPO/dist" \
  'cubeloop==0.14.0' 'cubepi[postgres]==0.14.0'
cd /tmp
/tmp/cubeloop-smoke/bin/python -c "import cubeloop, cubepi; from cubepi.providers.anthropic import AnthropicProvider; from cubepi.tracing import schema; import sqlalchemy, asyncpg"
/tmp/cubeloop-smoke/bin/python -m cubepi trace --help
```

Hide OpenTelemetry in a subprocess for the `from cubepi.tracing import schema` assertion. This is the test that catches path-deps, missing shim files, and extras that do not forward. Use `--find-links` without `--no-index` so third-party wheels still come from PyPI; `--find-links` prefers the local cubeloop/cubepi artifacts when versions match.

---

## Out of scope (do not do in this PR)

- Yanking `cubepi` 0.13.x
- Failing `pip install cubepi` (sklearn brownout)
- Dual-writing OTel keys
- Rewriting versioned docs or old CHANGELOG entries
- New logo artwork
- Opening a second GitHub repository
- Actually renaming the GitHub repo / DNS / PyPI publishers (runbook only)
- Tagging 0.14.0 / cutting `version-0.14` docs
