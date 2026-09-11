# Retain `cubepi_*` persistence schema — implementation plan

**Spec:** [`dev/specs/2026-09-10-retain-cubepi-schema.md`](../specs/2026-09-10-retain-cubepi-schema.md)

**Worktree:** `.worktrees/2026-09-10-retain-cubepi-schema` on branch
`2026-09-10-retain-cubepi-schema`.

## 1. Prepare deletion communication

- Update the migration page and v0.14.0 release notes to say 0.14.0 was removed
  and 0.14.1 is the first supported CubeLoop release.

## 2. Restore the v5 persistence contract

- Change Postgres/MySQL model table/index names back to `cubepi_*` while retaining
  `Cubeloop*` Python names and `cubeloop_metadata`.
- Set both database `EXPECTED_SCHEMA_VERSION` constants to 5.
- Restore all runtime and verification SQL identifiers to `cubepi_*`; remove all
  v6-specific diagnosis and recovery paths.
- Restore Alembic helper SQL to the exact 0.13.6 storage semantics and remove
  `upgrade_v5_to_v6_op()`.
- Restore examples and backend READMEs.

## 3. Remove the withdrawn recovery surface

- Delete the PostgreSQL/MySQL reverse-rename scripts and their integration tests.
- Delete runtime probes and special error messages for `cubeloop_*` database
  objects. Retain ordinary v5 version and missing-schema validation.

## 4. Lock down migration reproducibility

- Add golden exact-output tests for every 0.13.6 helper, including order, types,
  terminators, partition count, and MySQL statement splitting.
- Add representative host migration-chain fixtures for Postgres and MySQL.
- Test empty database → v5 replay, prebuilt v5 → open, and data round trips.
- Add a regression assertion that normal schema code contains no physical
  `cubeloop_*` identifiers.

## 5. Repair packaging and documentation

- Bump root and tombstone projects to 0.14.1. Remove all tombstone dependencies,
  extras forwarding, import aliases, API exports, and CLI dispatch.
- Test top-level/deep imports and both command forms for non-zero actionable
  failure without importing or installing `cubeloop`; Python tracebacks are
  acceptable as long as they contain the guidance.
- Rewrite the current and version-0.14 migration/checkpoint docs in English and
  zh-Hans; remove every recovery link and state that 0.14.0 was deleted.
- Correct the existing 0.14 pages in place; do not rerun docs version creation or
  alter versions/sidebars/latest-version state.
- Update recipes, generated API references as required, changelog, and release
  notes. Record why the 0.14 snapshot was corrected.

## 6. Verify locally

- `uv sync --all-extras --dev`
- Targeted Postgres/MySQL/helper/shim tests.
- `uv run pytest tests/`
- `uv run ruff check cubeloop/ tests/`
- `uv run ruff format --check cubeloop/ tests/`
- `uv run mypy cubeloop`
- Website tests and production build.
- Build both distributions and smoke-test them from a clean temporary venv.

## 7. Reviews, PR, and release

- Ask the user before entering the local Codex review loop.
- Run spec → plan → code reviews with `codex:rescue`, fixing until clean.
- Commit and push the branch; open the replacement PR with test evidence.
- Run the PR Codex poll/fix/`@codex` loop until clean and wait for CI.
- Merge and wait for the migration page to deploy.
- Verify the merged `main` commit already contains the reviewed 0.14.1 versions
  and changelog from this PR. Explicitly skip `dev/runbooks/cut-doc-version.md`
  and never run `docusaurus docs:version 0.14`; the existing snapshot was
  corrected in place.
- Tag that exact clean `main` commit as v0.14.1, publish both wheels, and verify
  PyPI metadata and clean-machine behavior. In the same release window, delete
  both 0.14.0 PyPI releases and verify their files are gone. Then amend the
  v0.14.0 GitHub release and close #223 as superseded.
