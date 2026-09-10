# Retain `cubepi_*` persistence schema — implementation plan

**Spec:** [`dev/specs/2026-09-10-retain-cubepi-schema.md`](../specs/2026-09-10-retain-cubepi-schema.md)

**Worktree:** `.worktrees/2026-09-10-retain-cubepi-schema` on branch
`2026-09-10-retain-cubepi-schema`.

## 1. Prepare withdrawal communication

- Prepare the public incident/recovery page and v0.14.0 release-note amendment.
- Do not yank until the recovery page is merged and deployed (or the release note
  contains the complete recovery procedure inline).

## 2. Restore the v5 persistence contract

- Change Postgres/MySQL model table/index names back to `cubepi_*` while retaining
  `Cubeloop*` Python names and `cubeloop_metadata`.
- Set both database `EXPECTED_SCHEMA_VERSION` constants to 5.
- Restore all runtime and verification SQL identifiers to `cubepi_*`; remove the
  v6-specific diagnosis path.
- Restore Alembic helper SQL to the exact 0.13.6 storage semantics and remove
  `upgrade_v5_to_v6_op()`.
- Restore examples and backend READMEs.

## 3. Create and test the recovery source

- Add canonical machine-readable scripts at
  `website/static/recovery/0.14.0/postgres-v6-to-v5.sql` and
  `website/static/recovery/0.14.0/mysql-v6-to-v5.sql`. These are the only
  executable recovery source; EN/zh-Hans docs link to the same downloads and
  explain backup, stop-the-world, execution, and verification rather than
  maintaining translated SQL copies.
- Each script performs collision/mixed-state preflight before mutation and
  restores tables, Postgres partitions, indexes, and schema version 5.
- Test both scripts against an exact v6 fixture with data, indexes, constraints,
  and partitions; prove preflight failure leaves mixed/colliding schemas intact.

## 4. Lock down migration reproducibility

- Add golden exact-output tests for every 0.13.6 helper, including order, types,
  terminators, partition count, and MySQL statement splitting.
- Add representative host migration-chain fixtures for Postgres and MySQL.
- Test empty database → v5 replay, prebuilt v5 → open, and data round trips.
- Add a regression assertion that normal schema code contains no physical
  `cubeloop_*` identifiers.

## 5. Repair packaging and documentation

- Bump root and shim projects to 0.14.1 and change shim dependency/extras to
  `>=0.14.1,<0.15`.
- Rewrite the current and version-0.14 migration/checkpoint docs in English and
  zh-Hans; include isolated emergency reverse-rename instructions for 0.14.0.
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
- Merge and wait for the recovery page and downloadable scripts to deploy.
- Amend the v0.14.0 GitHub release, yank both 0.14.0 PyPI projects with the
  approved reason, verify their JSON metadata, and close #223 as superseded.
- Verify the merged `main` commit already contains the reviewed 0.14.1 versions
  and changelog from this PR. Explicitly skip `dev/runbooks/cut-doc-version.md`
  and never run `docusaurus docs:version 0.14`; the existing snapshot was
  corrected in place.
- Tag that exact clean `main` commit as v0.14.1, publish both wheels, then verify
  PyPI metadata and clean-machine installation of both `cubeloop==0.14.1` and
  `cubepi==0.14.1`.
