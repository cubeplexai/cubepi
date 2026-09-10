# Rename CubePi → CubeLoop

- **Date**: 2026-09-09
- **Status**: Approved for implementation
- **Branch / worktree**: `2026-09-09-rename-cubeloop` → `.worktrees/2026-09-09-rename-cubeloop`
- **Related**: release runbook in this spec §9; docs versioning [`dev/runbooks/cut-doc-version.md`](../runbooks/cut-doc-version.md)

## Confirmed with the user

| Decision | Choice |
|---|---|
| Display name | CubeLoop |
| PyPI / import / CLI | `cubeloop` |
| Domain | `cubeloop.dev` (newly registered) |
| GitHub | Rename `cubeplexai/cubepi` in place to `cubeplexai/cubeloop`. Do **not** create a second repository. |
| Dual package | Yes. `cubepi` continues on PyPI as a transitional meta-package that depends on `cubeloop` and warns. |
| SQL table names | Rename (`cubepi_*` → `cubeloop_*`) with a schema migration. |
| OTel vendor namespace | Rename (`cubepi.*` → `cubeloop.*`). |
| Historical snapshots | Frozen `versioned_docs` / old CHANGELOG entries / old `dev/specs` stay as CubePi — they document what shipped. |

## 1. Motivation

The project is rebranding from CubePi to CubeLoop. The current public identity (`cubepi` on PyPI, `from cubepi import …`, `cubepi.ai`, `cubeplexai/cubepi`) has to move together. Leaving any of those on the old name after the rest have moved creates a split identity that is worse than a clean break.

This is a breaking 0.x release (proposed **0.14.0**), not a 1.0.

## 2. Identity map

| Surface | Before | After |
|---|---|---|
| Display | CubePi | CubeLoop |
| PyPI project | `cubepi` | `cubeloop` (real). `cubepi` stays as the shim. |
| Import root | `cubepi` | `cubeloop` |
| Source tree | `cubepi/` | `cubeloop/` |
| Console script | `cubepi` | `cubeloop`. Shim also keeps `cubepi` as a wrapper. |
| Docs / homepage | `https://cubepi.ai` | `https://cubeloop.dev` |
| GitHub | `https://github.com/cubeplexai/cubepi` | `https://github.com/cubeplexai/cubeloop` |
| Trace dir default | `./cubepi-traces` | `./cubeloop-traces` |
| Skills | `cubepi`, `cubepi-trace` | `cubeloop`, `cubeloop-trace` |
| OTel scope | `cubepi.tracing` | `cubeloop.tracing` |
| OTel vendor attrs | `cubepi.run_id`, `cubepi.turn`, … | `cubeloop.run_id`, `cubeloop.turn`, … |
| SQL tables (PG/MySQL) | `cubepi_threads`, `cubepi_messages`, … | `cubeloop_threads`, `cubeloop_messages`, … |
| SQLAlchemy metadata | `cubepi_metadata` | `cubeloop_metadata` |
| Schema exceptions | `CubepiSchemaError` hierarchy | `CubeloopSchemaError` hierarchy |
| Env vars | `CUBEPI_*` | `CUBELOOP_*` (old names accepted for one 0.x window) |

Out of scope for this rename:

- New logo / visual identity (rename the files; keep the existing artwork unless new assets are supplied).
- Rewriting frozen versioned docs (0.3–0.13) or historical CHANGELOG entries.
- Creating a new GitHub repository.
- Yanking `cubepi` 0.13.x from PyPI.

## 3. Dual package (the `cubepi` shim)

### 3.1 Why a second distribution

Users who already have `cubepi` in a lockfile will `pip install -U cubepi` (or uv/poetry equivalent) and expect *something* to happen. Publishing only `cubeloop` leaves those lockfiles on 0.13.x forever.

Prior art:

- **sklearn on PyPI** (scikit-learn/sklearn-pypi-package) is an empty distribution that depends on `scikit-learn`. That works because the real wheel already provides the `sklearn` import path. cubepi cannot copy this blindly: after the rename the real wheel provides `cubeloop`, not `cubepi`.
- sklearn later switched the dummy to a **brownout that fails install**. We do **not** fail install in 0.14. We warn, pull in `cubeloop`, and keep old imports working for a transition window.
- **Pillow** ships the `PIL` import path inside the same wheel. We cannot do that as the long-term shape (the import path is part of the new name), but the shim's job is the same idea for one 0.x window.

Divergence: cubepi's import path **is** the PyPI name, so an empty dependency-only wheel would make `import cubepi` raise `ModuleNotFoundError` after a successful install. That is worse than a warning. The shim is therefore a **tiny alias package**, not a zero-file meta-package.

### 3.2 Shape of the `cubepi` 0.14+ distribution

Lives in this repo at `compat/cubepi/` (own `pyproject.toml`). The root project becomes `cubeloop`.

- `requires-python` matches cubeloop.
- Version is **exactly** the cubeloop version (pin `cubeloop==0.14.0`, not a range). Version skew between the two wheels is a bug.
- Optional extras are forwarded: `cubepi[sqlite]` → `cubeloop[sqlite]==<same>`, same for `postgres`, `mysql`, `mcp`, `tracing`, `tracing-otlp`, `trace-cli`.
- Classifier `Development Status :: 7 - Inactive`.
- PyPI description / README is a deprecation banner pointing at `pip install cubeloop` and https://cubeloop.dev.

The wheel contains:

1. `cubepi/__init__.py` — warn + public re-export of `cubeloop`.
2. An on-demand import redirect so deep imports keep working:
   `from cubepi.providers.anthropic import AnthropicProvider` loads the cubeloop module of the same suffix.
3. Console script `cubepi` → prints the same warning to stderr, then dispatches to `cubeloop.cli`.

The redirect is a `MetaPathFinder` **plus an alias `Loader`**, not a finder that only mutates `sys.modules` and returns `None`. Python resolves `cubepi.providers` before `cubepi.providers.anthropic`; if the finder does not return a `ModuleSpec` with a loader, import falls through to a missing physical package and deep imports fail.

Required shape:

- Finder matches only `fullname.startswith("cubepi.")` (the root `cubepi` package is the shim on disk).
- `find_spec` returns a `ModuleSpec(fullname, AliasLoader(real_name), origin=…, is_package=…)`.
- `AliasLoader.create_module` returns `importlib.import_module("cubeloop" + suffix)` — the **same object** as `sys.modules[real_name]`, so cubeloop's relative imports keep using `__name__ == "cubeloop…"` .
- `exec_module` is a no-op (module already executed under the real name).
- Package specs set `submodule_search_locations` so further children can be found.
- Do not `walk_packages` / eagerly import `cubeloop.tracing`. `from cubepi.tracing import schema` must still work without OpenTelemetry.

Do not assign `sys.modules["cubepi"] = cubeloop`.

### 3.3 The install-time warning

PEP 517 wheels do not run `setup.py` at install time. A print inside `setup()` only fires when building from an sdist, and `uv build` publishes a wheel, so **a true pip-install print is not reliable**. sklearn's dummy eventually failed install by raising in `setup()`; that is a brownout, not a warning, and it only works for sdist installs.

Warning surfaces we actually ship:

| When | How |
|---|---|
| Browsing PyPI | Banner README + Inactive classifier. |
| First `import cubepi` | `DeprecationWarning` **and** one stderr line (warnings are often filtered). |
| `cubepi` CLI | Same stderr line, then the cubeloop CLI. |
| Resolver | `cubepi==0.14.0` depending on `cubeloop==0.14.0` is visible in the lock. |

Do not install a `.pth` that prints on every Python startup.

The stderr / warning text:

```text
WARNING: The 'cubepi' package has been renamed to 'cubeloop'.
Install `cubeloop` and import from `cubeloop` instead.
See https://cubeloop.dev/docs/migration/from-cubepi
```

### 3.4 Shim lifetime

Keep the shim for the rest of 0.x. Removal (or a sklearn-style install brownout) is a later, explicit decision — not part of 0.14.

### 3.5 Publishing both wheels

`.github/workflows/publish.yml` builds two dists from the same tag:

1. root project → `cubeloop`
2. `compat/cubepi` → `cubepi`

PyPI trusted publishing is per project **and per index**. The publish workflow uploads to TestPyPI first, then PyPI. Before tagging:

- Register `cubeloop` on **both** pypi.org and test.pypi.org, each with an OIDC publisher for `cubeplexai/cubeloop` (workflow `publish.yml`, matching environment names).
- After the GitHub repo rename, update the existing `cubepi` publisher's repository name on **both** indexes to `cubeplexai/cubeloop`.

Partial upload: wheels are immutable. If TestPyPI accepts `cubeloop` and then rejects `cubepi` (missing publisher), a retry of the same version fails on the already-uploaded file. The publish action must set `skip-existing: true` (or an equivalent idempotent retry) so a completed file is skipped and the missing one can be uploaded. Document the same for a human `twine` recovery.

A post-build smoke test (clean venv, `--no-index --find-links dist`, cwd outside the checkout) is required before we trust the two wheels: `import cubeloop`, `import cubepi` (warns), a deep import, `from cubepi.tracing import schema` without OpenTelemetry, `python -m cubepi trace --help`, and `cubepi[postgres]` pulling sqlalchemy/asyncpg.

## 4. Library rename

### 4.1 Package tree and Python names

`git mv cubepi cubeloop`. Every `from cubepi…` / `import cubepi…` in library code, tests, examples, website scripts, and current docs becomes `cubeloop`.

Public identifiers that embed the old name:

| Old | New |
|---|---|
| `cubepi_metadata` | `cubeloop_metadata` |
| `CubepiBase`, `CubepiThread`, `CubepiMessage`, `CubepiRun`, `CubepiHitlAnswer`, `CubepiSchemaVersion` | `Cubeloop*` |
| `CubepiSchemaError`, `CubepiSchemaUninitialized`, `CubepiSchemaMismatch` | `CubeloopSchemaError`, `CubeloopSchemaUninitialized`, `CubeloopSchemaMismatch` |
| `CUBEPI_*` tracing constants | `CUBELOOP_*` |
| `_pkg_version("cubepi")` | `_pkg_version("cubeloop")` |

cubeloop itself does **not** keep the old Python names. The shim is the compatibility layer.

### 4.2 CLI

`[project.scripts] cubeloop = "cubeloop.cli.__main__:main"`.

Default JSONL directory: `./cubeloop-traces`. `--dir` is unchanged. No automatic fallback to `./cubepi-traces`.

### 4.3 Env vars

| Old | New |
|---|---|
| `CUBEPI_TEST_PG_DSN` | `CUBELOOP_TEST_PG_DSN` |
| `CUBEPI_TEST_MYSQL_DSN` | `CUBELOOP_TEST_MYSQL_DSN` |
| `CUBEPI_TEST_MCP_HTTP_URL` | `CUBELOOP_TEST_MCP_HTTP_URL` |
| `CUBEPI_PG_DSN` (examples) | `CUBELOOP_PG_DSN` |
| `CUBEPI_MYSQL_DSN` (examples) | `CUBELOOP_MYSQL_DSN` |
| `CUBEPI_DOCS_SOURCE_REF` | `CUBELOOP_DOCS_SOURCE_REF` |

Readers accept the old name if the new one is unset, for the rest of 0.x. CI and docs print the new name.

## 5. Checkpointer schema v5 → v6

`EXPECTED_SCHEMA_VERSION` becomes 6. Hosts apply `upgrade_v5_to_v6_op()` (Postgres and MySQL each get one) and then `write_schema_version_op()`.

### 5.0 Runtime verification (unmigrated v5)

0.14 `_verify_schema` SELECTs `cubeloop_schema_version`. A v5 database still has only `cubepi_schema_version`. Treating “new table missing” as `CubeloopSchemaUninitialized` / “tables not found” is the wrong diagnosis — the host needs `upgrade_v5_to_v6_op()`, not a greenfield CREATE.

Verification order:

1. `cubeloop_schema_version` present → compare `version` to 6 (today’s path). Mismatch hint names every `upgrade_vN_to_v{N+1}_op()` still required, including v5→v6.
2. Else `cubepi_schema_version` present → `CubeloopSchemaMismatch(expected=6, actual=<that row>)` with a hint that **only** names `upgrade_v5_to_v6_op()` + `write_schema_version_op()`. Do this even if the stored number is already 6 (a host who replayed live `write_schema_version_op()` before the rename).
3. Else if `cubepi_threads` or `cubeloop_threads` exists → not an empty database. Raise `CubeloopSchemaMismatch(expected=6, actual=0)` with a hint that the version table is missing so the schema cannot be auto-classified. Recovery (migrate guide): if the data tables are still `cubepi_*`, `CREATE TABLE cubepi_schema_version` and `INSERT 5`, then run `upgrade_v5_to_v6_op()`. Do **not** apply fresh v6 CREATE TABLE on top of existing data.
4. Else → `CubeloopSchemaUninitialized`.

Do not SELECT the legacy version table except as this diagnostic.

### 5.0.1 `write_schema_version_op()` must not assume the new table

Hosts copy `op.execute(write_schema_version_op())` into **each** Alembic revision (documented pattern). After 0.14 those old revisions still run against `cubepi_schema_version` when someone provisions a fresh DB by replaying history. If the helper is rewritten to only touch `cubeloop_schema_version`, replay dies in the v4→v5 revision, before v6 can rename.

`write_schema_version_op()` writes `EXPECTED_SCHEMA_VERSION` to `cubeloop_schema_version` if that table exists, else to `cubepi_schema_version`. `upgrade_v5_to_v6_op()` only renames; the host then calls `write_schema_version_op()`, which now sees the new table.

Dialect constraints:

- **Postgres:** one `DO $$ … $$` block is fine (`asyncpg` / Alembic `op.execute` send it as a single statement).
- **MySQL:** the documented contract is `for stmt in write_schema_version_op().split(";"): op.execute(stmt)`. A compound `IF` / stored procedure **cannot** be used — internal semicolons get split into invalid fragments. Return a `;`-joined sequence of **standalone** statements only, e.g. `SET @cp_ver_new = (SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name = 'cubeloop_schema_version')`, then `SET @cp_del = IF(@cp_ver_new > 0, 'DELETE FROM cubeloop_schema_version WHERE version <> N', 'DELETE FROM cubepi_schema_version WHERE version <> N')`, then `PREPARE` / `EXECUTE` / `DEALLOCATE PREPARE`, then the same for `INSERT IGNORE`. No statement in that string contains a nested `;`. Tests must split-and-execute exactly as the README does.

Historical v1–v5 DDL helpers keep emitting `cubepi_*` names. A 0.14 test must replay that full helper chain (not only a pre-built v5 snapshot) through v6 and then open the checkpointer.

### 5.1 Postgres

Rename parent tables, 64 message partitions, 64 run partitions, indexes, and the schema-version table:

| Old | New |
|---|---|
| `cubepi_threads` | `cubeloop_threads` |
| `cubepi_messages` (+ `cubepi_messages_p00`…`p63`) | `cubeloop_messages` (+ `cubeloop_messages_p00`…`p63`) |
| `cubepi_runs` (+ `cubepi_runs_p00`…`p63`) | `cubeloop_runs` (+ same partition suffix) |
| `cubepi_hitl_answers` | `cubeloop_hitl_answers` |
| `cubepi_schema_version` | `cubeloop_schema_version` |
| `ix_cubepi_*` | `ix_cubeloop_*` |

PostgreSQL `ALTER TABLE … RENAME TO` on a partitioned parent does **not** rename children. The helper must rename each partition. FKs follow the referenced table automatically.

Fresh installs create `cubeloop_*` names only. `create_message_partitions_op` / `create_runs_partitions_op` emit the new names.

Idempotency is **complete only when both** `cubeloop_threads` and `cubeloop_schema_version` exist. `cubeloop_threads` alone is not a no-op sentinel (that is the half-renamed failure mode). If data tables are already `cubeloop_*` but `cubepi_schema_version` still exists, rename the remaining objects. If neither new nor old data tables exist, this is not a v5→v6 path.

### 5.2 MySQL

One `RENAME TABLE a TO a2, b TO b2, …` statement covering **every** v5 table including `cubepi_schema_version`. Indexes and KEY partitions ride with the table. A second statement for the version table is forbidden: if it fails after data tables already moved, retry sees `cubeloop_threads` and would no-op under a naive sentinel.

The helper assumes a complete v5 (data + version table). If `cubepi_schema_version` is missing, the atomic rename fails as a whole — `_verify_schema` step 3 tells the operator to create the version table first. Same complete-state idempotency as Postgres.

### 5.3 SQLite — do not rename

SQLite already uses generic names (`messages`, `runs`, `thread_extra`, `thread_pending_request`, `thread_hitl_answers`). There is no `cubepi_` prefix to update, and renaming those files' internal tables would churn local DBs for no brand value.

This is a deliberate divergence from "rename every table": SQLite was never branded.

### 5.4 Host-managed DDL

Examples (`examples/checkpointing_postgres.py`, `examples/checkpointing_mysql.py`) and the backend READMEs ship the v6 CREATE TABLE statements under the new names, plus the v5→v6 rename helper for existing databases. Applications that copy-pasted v5 DDL must run the helper (or equivalent `RENAME TABLE`) before opening 0.14.

## 6. Tracing

### 6.1 Write path

Every cubepi-vendor identifier on the write path changes. Implementation starts from a grep of `cubepi.` in `cubeloop/` (after the tree rename), not from the schema.py constant list alone. Inventory that exists in 0.13.6 and **must** move:

| Kind | Old | New |
|---|---|---|
| Attr constants in `schema.py` | `cubepi.run_id`, `cubepi.thread_id`, `cubepi.agent.*`, `cubepi.turn.*`, `cubepi.llm.*`, `cubepi.tool.*`, `cubepi.run.outcome`, `cubepi.aborted`, counts | `cubeloop.*` same suffix |
| Span name | `cubepi.turn` | `cubeloop.turn` |
| Scope | `cubepi.tracing` | `cubeloop.tracing` |
| Ad-hoc attrs | `cubepi.tags`, `cubepi.metadata.<key>`, `cubepi.oneshot.operation`, `cubepi.fork.src_thread_id`, `cubepi.fork.after_run_id` | `cubeloop.*` |
| Span name | `cubepi.agent.fork_once` | `cubeloop.agent.fork_once` |
| `error.type` values | `cubepi.aborted`, `cubepi.error`, `cubepi.tool.error` | `cubeloop.aborted` / `.error` / `.tool.error` |
| Tracer scopes | `cubepi.mcp`, `cubepi.hitl`, `cubepi.agent` | `cubeloop.*` |
| Loggers | `cubepi.providers`, `cubepi.providers.base`, `cubepi.providers.fallback`, `cubepi.tracing` | `cubeloop.*` |
| ContextVars | `cubepi.tracing.run_metadata`, `cubepi.tracing.run_tags`, `cubepi.tracing.active_run` | `cubeloop.tracing.*` |
| Helper | `cubepi_error_type_for` → `"cubepi.aborted"` | `cubeloop_error_type_for` → `"cubeloop.aborted"` |

Promote every ad-hoc f-string / literal onto schema constants. No dual-write. `gen_ai.*` is unchanged.

### 6.2 Read path (`cubeloop trace`)

JSONL from ≤0.13 still has `cubepi.*` keys. Readers accept **both** namespaces. New files are cubeloop-only.

A generic `schema.attr(attrs, NEW_KEY)` (try new, then old) is not enough for abort: `Span.is_aborted` today reads only the boolean `cubepi.aborted`. Some 0.13 spans (and MCP adapter paths) also stamp `error.type="cubepi.aborted"`; treat as aborted if **either** boolean (`cubeloop.aborted` or `cubepi.aborted`) is true **or** `error.type` is `cubeloop.aborted` or `cubepi.aborted`. Metadata prefix, oneshot, tags, fork attrs, and span name `cubepi.turn` all dual-read.

This is the one place dual-read is required. Host dashboards are their migration.

## 7. Docs, site, brand files

### 7.1 Current docs (this PR)

Rename throughout:

- `website/docs/**` (EN current)
- `website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/**`
- `website/src/**`, `docusaurus.config.ts`, compare pages, FAQ, changelog page
- `README.md`, `AGENTS.md` / `CLAUDE.md`, `examples/**`, `skills/**`
- `website/static/_worker.js`: production hostname `cubeloop.pages.dev` → `cubeloop.dev`

Add `website/docs/migration/from-cubepi.md` (and zh-Hans current): install `cubeloop`, import map, alembic v5→v6, OTel attribute map, shim warning, GitHub redirect. Put it in the existing Migration sidebar category.

A feature without docs is not done; the migrate page is the user-facing docs for this change.

### 7.2 Frozen history — do not rewrite

`website/versioned_docs/version-0.3` … `version-0.13` and their zh-Hans mirrors stay CubePi. They are snapshots of released versions.

Optional, small, allowed: a one-line banner on those versions ("CubePi was renamed to CubeLoop in 0.14; this snapshot documents the `cubepi` package.") if Docusaurus version config supports it without editing hundreds of md files. Do not search-replace the snapshots.

`CHANGELOG.md`: add a 0.14.0 section that describes the rename. Do not rewrite older sections. Update the `[Unreleased]` / compare URLs to the new GitHub path once the repo is renamed.

`dev/specs/**` and `dev/plans/**` from before this date stay as written.

### 7.3 Brand files

Rename `website/static/img/brand/cubepi-*` → `cubeloop-*`. Same artwork. Social preview alt text / Open Graph titles become CubeLoop.

### 7.4 Domain

- Canonical site: `https://cubeloop.dev`
- Cloudflare Pages custom domain: `cubeloop.dev`
- `cubepi.ai` (and `www`) **301** to the matching `cubeloop.dev` path. Existing `_worker.js` already 301s `cubepi.pages.dev` → the custom domain; after the cut it 301s `cubeloop.pages.dev` → `cubeloop.dev`. The cubepi.ai → cubeloop.dev redirect is DNS/Cloudflare config, not this repo.

## 8. GitHub rename (in place)

GitHub's rename keeps issues, PRs, stars, Actions history, and redirects `github.com/cubeplexai/cubepi` → `…/cubeloop` (git clone of the old URL keeps working).

Things that do **not** follow automatically and must be updated in the same release window:

- This repo's URLs in `pyproject.toml`, README badges, `docusaurus.config.ts` `editUrl`, CHANGELOG compare links, skills install lines.
- PyPI trusted publisher repository name (both `cubepi` and the new `cubeloop` project).
- Codecov project (usually follows; verify the badge).
- Cloudflare Pages GitHub connection.
- skills.sh / DeepWiki entries that key on `cubeplexai/cubepi`.

Do **not** open a new empty repository and push. That would split issues and break the redirect.

## 9. Release order

The code PR can land on `main` while the GitHub repo is still named `cubepi`, but then README/docs URLs 404 until the rename. Coordinate as one cut:

1. Land this work on `main` (package tree, shim, schema v6, current docs, migrate guide).
2. **Immediately** rename the GitHub repository `cubepi` → `cubeloop`.
3. Point `cubeloop.dev` at the Pages project; 301 `cubepi.ai` → `cubeloop.dev`.
4. Update **PyPI and TestPyPI** trusted publishers; first-time-create `cubeloop` on both indexes. Confirm workflow name + environment match `publish.yml`.
5. Tag `v0.14.0`. Publish uploads both dists; `skip-existing: true` so a partial TestPyPI attempt can be retried. User-install of the shim needs cubeloop on the same index — TestPyPI/PyPI both must have both files before we call the cut done.
6. Verify from a clean machine: `pip install cubeloop` and `pip install -U cubepi` both yield importable cubeloop; the latter warns on `import cubepi`.

Docs version snapshot for 0.14 follows [`cut-doc-version.md`](../runbooks/cut-doc-version.md) **after** the current tree is CubeLoop, so `version-0.14` is born as CubeLoop.

## 10. Testing

- Full existing suite after the tree rename (pytest, ruff, mypy) against `cubeloop/`.
- Shim tests in `tests/compat/`: `import cubepi` warns; `from cubepi import Agent` is `cubeloop.Agent`; `from cubepi.providers.anthropic import AnthropicProvider` resolves; `from cubepi.tracing import schema` is still lazy. Plus a **clean-venv wheel smoke** (Task 8) — workspace tests are not sufficient.
- Schema v6: a **fresh-v6** fixture (canonical new DDL only, no historical helpers); a **historical-v5** fixture (old names + old helpers) that then runs `upgrade_v5_to_v6_op`; a **full helper-chain replay** from v1-shaped CREATE through v6; unmigrated v5 raises `CubeloopSchemaMismatch` not Uninitialized; helper is idempotent.
- Trace CLI fixtures: 0.13 JSONL covering `cubepi.turn`, `cubepi.run_id`, `cubepi.metadata.*`, `cubepi.tags`, fork attrs, oneshot, boolean abort, and `error.type="cubepi.aborted"` without the boolean; 0.14 JSONL uses only `cubeloop.*`.
- Examples and `tests/checkpointer/test_examples.py` use `CUBELOOP_*_DSN` (old env still accepted).

## 11. Non-goals

- 1.0 version jump.
- Failing `pip install cubepi` (sklearn brownout) in this release.
- Dual-writing OTel attributes.
- Renaming SQLite table names.
- Rewriting versioned docs or git history.
- A new GitHub repository.
- Redesigning the logo.
