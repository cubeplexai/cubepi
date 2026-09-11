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
| 0.14.0 | Delete both PyPI releases; keep the GitHub tag/release as incident history |
| Recovery scope | None; 0.14.0 had no users and is removed rather than supported |
| Replacement | Publish 0.14.1 after clean review and CI |
| `cubepi` package | Publish a dependency-free tombstone that always fails with migration guidance |

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

### 3.3 No 0.14.0 database recovery surface

Remove the reverse-rename SQL, withdrawn-schema runtime probing, and recovery
guidance. Delete both 0.14.0 distributions during the 0.14.1 release cutover,
after both replacement distributions are downloadable and verified, so package
resolution never falls back to 0.13.6 or to no installable CubeLoop release.
0.14.1 supports the stable v5 `cubepi_*` schema only. A database containing
`cubeloop_*` objects is outside the supported contract; the package neither
detects nor mutates it specially.

### 3.4 Compatibility distribution becomes a tombstone

Publish `cubepi==0.14.1` as a dependency-free tombstone package. It must not
depend on, import, re-export, alias, or dispatch to `cubeloop`. Importing
`cubepi` or any `cubepi.*` path raises `ImportError` immediately with a concise
message naming the replacement package/import and linking to the migration
guide. The package declares no dependencies and no optional-dependency extras.
Running the `cubepi` console script or `python -m cubepi` exits non-zero through
that same import failure, with the same actionable guidance and no CubeLoop
command dispatch. A normal Python import traceback is acceptable for these
command forms; tests assert the non-zero status and guidance text, not polished
CLI rendering.

### 3.5 Documentation and frozen 0.14 snapshot

Update current English and zh-Hans checkpointing/migration docs, backend READMEs,
examples, API reference inputs, and the existing 0.14 versioned snapshot. State
that 0.14.0 was removed and has no supported database migration.
Although frozen snapshots normally do not change, 0.14.0 is withdrawn and 0.14.1
is the supported 0.14 release, so the snapshot must describe the safe 0.14
contract. Correct only the affected persistence/migration pages in place and
keep those pages at current/snapshot parity. Do **not** rerun
`docusaurus docs:version 0.14` or change version/sidebars/latest-version state.
Record this exception in the changelog and release notes.

Close PR #223 as superseded, preserving it as incident context.

## 4. Delete and release operations

Build both 0.14.1 distributions from the reviewed commit, upload them, and
verify their files and behavior from PyPI. Then, in the same controlled release
window, delete both `cubeloop==0.14.0` and `cubepi==0.14.0` and verify that their
release JSON endpoints no longer advertise downloadable files. Preserve the
GitHub tag/release as incident history and only then amend its notes to say the
package release was removed and users should install 0.14.1. No recovery
procedure is published because the confirmed operational premise is that nobody
adopted 0.14.0.

The 0.14.1 release follows the normal patch-release runbook, but does not rerun
the docs-version creation runbook. Do not tag or publish until local Codex review
is clean, CI passes, and the PR Codex review loop is clean.

## 5. Acceptance criteria

- Both PyPI 0.14.0 projects have no downloadable release files.
- A v5 Postgres/MySQL fixture opens and preserves data with 0.14.1 code and no DDL.
- Fresh host Alembic history creates only `cubepi_*`, ends at version 5, and the
  checkpointer operates on it.
- Golden tests prove the complete 0.13.6 helper output and MySQL split contract
  remain unchanged.
- No supported code or normal docs reference `upgrade_v5_to_v6_op()` or require
  `cubeloop_*` database objects.
- Unit/integration tests, Ruff, formatting, mypy, and docs build pass.
- The 0.14.1 `cubeloop` wheel works normally; the dependency-free `cubepi`
  tombstone wheel fails all imports and commands with migration guidance.
- Current and version-0.14 English/zh-Hans docs state that physical DB names stay
  `cubepi_*` and that 0.14.0 was removed.

## 6. Out of scope

- Reverting the Python/package/CLI/domain/GitHub rename.
- Reverting the new tracing namespace or trace directory.
- Supporting, detecting, or recovering mixed old/new table sets at runtime.
- Yanking any 0.13.x release.
- Renaming the physical database namespace in a later release without a new,
  separately approved expand/migrate/contract design.
