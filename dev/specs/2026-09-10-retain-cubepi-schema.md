# Keep the CubePi persistence schema after the CubeLoop rename

- **Date:** 2026-09-10
- **Status:** Approved for implementation
- **Branch / worktree:** `2026-09-10-retain-cubepi-schema` → `.worktrees/2026-09-10-retain-cubepi-schema`
- **Supersedes:** the database-rename portions of [`2026-09-09-rename-cubeloop.md`](2026-09-09-rename-cubeloop.md)
- **Incident release:** `cubeloop==0.14.0` / `cubepi==0.14.0`

## Confirmed decisions

| Decision | Choice |
|---|---|
| Product, package, imports, CLI | Remain CubeLoop / `cubeloop` |
| Postgres and MySQL physical names | Keep `cubepi_*` permanently |
| Schema version | Remain v5; the brand rename is not a schema change |
| 0.14.0 | Yank both PyPI distributions, without deleting files or the GitHub tag/release |
| Recovery scope | No runtime dual-schema support; document an emergency reverse rename for any early adopter |
| Replacement | Publish 0.14.1 after clean review and CI |

## 1. Problem

The 0.14.0 brand rename also renamed Postgres/MySQL tables and indexes and raised
the schema version from 5 to 6. That coupled a public identity change to a
persistent storage protocol. It made rolling deploys and application rollback
unsafe and, more importantly, changed the output of helpers imported by
historical host Alembic revisions. A v1 revision that was valid with 0.13 can
therefore fail when replayed with 0.14.

Migration files must be reproducible. A helper already used by a released host
migration must retain its original SQL semantics regardless of the installed
CubeLoop version.

## 2. Prior art and deliberate divergence

- LangGraph's Postgres checkpointer keeps an append-only ordered list of literal
  migrations; a migration's position is its version and old entries are not
  rewritten. CubeLoop keeps host-owned Alembic rather than adopting runtime DDL,
  but adopts the same immutability rule for exported helpers.
- pi-agent-core / the pi coding agent versions its persisted session format
  independently of its presentation and repository identity. Claude Code also
  treats on-disk project/user state as a compatibility surface rather than a
  branding surface.
- CubeLoop deliberately keeps the historical `cubepi_*` SQL namespace while its
  Python and product namespace is `cubeloop`. The storage name is a wire-format
  identifier, not user-facing branding.

## 3. Required behavior

### 3.1 Models and runtime SQL

Postgres and MySQL SQLAlchemy models, metadata, verification queries, runtime
queries, error row references, examples, and DDL documentation use:

- `cubepi_threads`
- `cubepi_messages` and Postgres `cubepi_messages_p00`…`p63`
- `cubepi_runs` and Postgres `cubepi_runs_p00`…`p63`
- `cubepi_hitl_answers`
- `cubepi_schema_version`
- `ix_cubepi_*` indexes

Python class names and metadata variables remain `Cubeloop*` and
`cubeloop_metadata`; only their physical SQL names retain the legacy namespace.
SQLite remains unchanged.

`EXPECTED_SCHEMA_VERSION` is 5 for Postgres and MySQL. A valid 0.13.6 database
must open directly under 0.14.1 without DDL.

### 3.2 Alembic helpers are immutable

Restore the complete Postgres and MySQL helper output to the 0.13.6 semantics.
Remove `upgrade_v5_to_v6_op()` from the supported public API and documentation.
`write_schema_version_op()` writes version 5 to `cubepi_schema_version` only.

Golden tests freeze the exact normalized output of every exported 0.13.6 helper,
including statement order, types, partition count, terminators, and MySQL's
semicolon-splitting contract. `write_schema_version_op()` becomes a permanently
frozen v5 writer rather than tracking `EXPECTED_SCHEMA_VERSION`. Future schema
work adds explicitly versioned writers and helpers; it must not retarget an
existing helper or derive historical SQL from a mutable package constant.

### 3.3 0.14.0 early-adopter recovery

0.14.1 does not silently detect or mutate `cubeloop_*` databases. The migration
guide contains conspicuous, dialect-specific emergency reverse-rename SQL for a
database that actually ran the withdrawn v6 migration. It instructs operators to:

1. stop all application instances;
2. back up the database;
3. rename tables, all Postgres child partitions, and indexes back;
4. write schema version 5;
5. verify object names and row counts before starting 0.14.1.

The recovery is not part of the normal upgrade path.

The documented SQL is executable source, not an untested snippet. Integration
tests create the exact withdrawn 0.14.0 v6 schema, insert representative data,
execute the documented recovery statements from a shared source verbatim, and
verify v5 names, partitions, indexes, constraints, row counts, version 5, and a
successful 0.14.1 checkpointer round trip. A preflight must detect name
collisions or mixed v5/v6 schemas and fail before executing any rename.

### 3.4 Compatibility distribution

Publish `cubepi==0.14.1` with the real package. Its dependency and forwarded
extras use `cubeloop>=0.14.1,<0.15`, allowing compatible 0.14 patch fixes while
preventing an unreviewed minor-version jump. Both distributions remain versioned
and published together.

### 3.5 Documentation and frozen 0.14 snapshot

Update current English and zh-Hans checkpointing/migration docs, backend READMEs,
examples, API reference inputs, and the existing 0.14 versioned snapshot.
Although frozen snapshots normally do not change, 0.14.0 is withdrawn and 0.14.1
is the supported 0.14 release, so the snapshot must describe the safe 0.14
contract. Correct only the affected persistence/migration pages in place and
keep those pages at current/snapshot parity. Do **not** rerun
`docusaurus docs:version 0.14` or change version/sidebars/latest-version state.
Record this exception in the changelog and release notes.

Close PR #223 as superseded, preserving it as incident context.

## 4. Yank and release operations

Yank both `cubeloop==0.14.0` and `cubepi==0.14.0` on PyPI with the reason:

> Withdrawn: the database table rename breaks reproducible host Alembic history
> and safe rollback. Use 0.14.1 or later.

The public incident/recovery page must be live before its link is used. Release
order is: merge and deploy that page; amend the GitHub 0.14.0 release notes;
yank both PyPI projects; then publish 0.14.1. If an emergency yank must happen
before the docs deploy, put self-contained recovery instructions in the GitHub
release notes first. Do not delete artifacts, tags, or the GitHub release.
TestPyPI may be yanked too if supported, but PyPI is the release blocker.

The 0.14.1 release follows the normal patch-release runbook, but does not rerun
the docs-version creation runbook. Do not tag or publish until local Codex review
is clean, CI passes, and the PR Codex review loop is clean.

## 5. Acceptance criteria

- Both PyPI 0.14.0 projects report all files as yanked with the agreed reason.
- A v5 Postgres/MySQL fixture opens and preserves data with 0.14.1 code and no DDL.
- Fresh host Alembic history creates only `cubepi_*`, ends at version 5, and the
  checkpointer operates on it.
- Golden tests prove the complete 0.13.6 helper output and MySQL split contract
  remain unchanged.
- Both dialects recover an exact v6 fixture with representative data by executing
  the shipped recovery source; mixed/colliding schemas fail before mutation.
- No supported code or normal docs reference `upgrade_v5_to_v6_op()` or require
  `cubeloop_*` database objects.
- Unit/integration tests, Ruff, formatting, mypy, and docs build pass.
- 0.14.1 wheels smoke-test both `cubeloop` and the `cubepi` shim.
- Current and version-0.14 English/zh-Hans docs state that physical DB names stay
  `cubepi_*` and include the exceptional 0.14.0 recovery path.

## 6. Out of scope

- Reverting the Python/package/CLI/domain/GitHub rename.
- Reverting the new tracing namespace or trace directory.
- Supporting mixed old/new table sets at runtime.
- Yanking any 0.13.x release.
- Renaming the physical database namespace in a later release without a new,
  separately approved expand/migrate/contract design.
