# Runbook: Cutting a docs version snapshot

When CubePi cuts a new release `X.Y`, the Docusaurus site needs a frozen
snapshot of the docs under `website/versioned_docs/version-X.Y/` (and the
zh-Hans mirror) so future docs drift on `main` doesn't rewrite history for
readers landing on the released version. This runbook captures the order of
operations and the traps the 0.5 → 0.6 → 0.7 cuts surfaced.

## When to run

Run this **right before** tagging the release — after all feature work for
`X.Y` has landed on `main` (or your release-prep worktree) and `current/` is
in the exact state you want to ship. The cut produces a frozen copy; anything
you fix in `current/` after the cut won't reach the released version without
a separate edit.

## Prerequisites

- Working in the release-prep worktree (`.worktrees/YYYY-MM-DD-release-X.Y-...`).
- `pyproject.toml` already bumped to `X.Y.0`.
- `CHANGELOG.md` has the `## [X.Y.0]` section populated (the `/changelog`
  page reads from the repo root, so it's not part of the snapshot — but
  prose in versioned guides may reference it).
- `current/` (EN + zh-Hans) and the under-development code are in sync.
  Specifically, every new public API in `X.Y` should already have a guide
  page and the page should match the shipped behavior. Use `diff -r
  website/docs website/i18n/zh-Hans/docusaurus-plugin-content-docs/current`
  to spot stragglers.

## The cut, in order

All commands run from `website/`.

```bash
# 1. Regenerate API mdx from the Python source (gitignored in current/, but
#    the snapshot will commit them under versioned_docs/version-X.Y/api/).
pnpm apiref

# 2. Snapshot. This populates:
#    - website/versioned_docs/version-X.Y/
#    - website/i18n/zh-Hans/docusaurus-plugin-content-docs/version-X.Y/
#      and version-X.Y.json (so the locale stays paired)
#    - website/versioned_sidebars/version-X.Y-sidebars.json
#    - prepends "X.Y" to website/versions.json
pnpm docusaurus docs:version X.Y

# 3. Edit website/docusaurus.config.ts (see "Config edits" below).

# 4. Build with the new version flipped to latest.
pnpm build
```

## Config edits (`docusaurus.config.ts`)

In the `classicOptions.docs` block:

- Flip `lastVersion` to `'X.Y'`.
- In `versions`:
  - Add `'X.Y': { label: 'X.Y (latest)', path: '' }`.
  - Demote the previous `(latest)` entry to `{ label: 'N-1', path: 'N-1',
    noIndex: true }`. `noIndex: true` is **mandatory** — without it,
    Google reindexes the demoted docs as duplicate content of the new
    latest, costing the new version search rank.
- In `sitemap.ignorePatterns`, add `'/docs/N-1/**'` so the demoted version
  is dropped from `sitemap.xml`.

Don't touch the `current: { label: 'Next 🚧', path: 'next', banner:
'unreleased', noIndex: true }` entry — `current/` should always live at
`/docs/next/` with `noIndex` so unreleased changes don't pollute search.

## Gotchas (the meat)

These are the things that have actually bitten us during 0.5 → 0.7 cuts. Read
them before running the cut, not after the build fails.

### 1. Run `pnpm apiref` **before** `docusaurus docs:version`

`website/docs/api/cubepi-*.mdx` is gitignored (only `index.mdx` is committed)
because it's regenerated from Python source on every build by `pnpm apiref`
(wired into `prebuild`/`prestart`). If you cut without running `apiref`
first, the snapshot copies an empty `api/` directory, the next build fails
on a broken link from `api/index.mdx`, and you have to hand-port the api
mdx files into `versioned_docs/version-X.Y/api/` after the fact. The 0.6
cut hit this — see commit `ee1056d` (`fix(docs): add generated API reference
to version-0.6 snapshot`).

### 2. The snapshot freezes `sidebars.ts` too

`docusaurus docs:version` writes the **current** sidebar into
`versioned_sidebars/version-X.Y-sidebars.json`. Any sidebar entry you add
to `sidebars.ts` after the cut shows up only on `current/` (Next 🚧), not
on `X.Y`. If you forgot a page, the fix is either (a) re-cut by deleting
`versioned_docs/version-X.Y/` and `versioned_sidebars/version-X.Y-sidebars.json`
and rerunning, or (b) hand-edit the versioned sidebar JSON.

### 3. `current/` and `version-X.Y/` must be byte-identical at cut time

Cutting at release time means a reader on `/docs/next/foo` and a reader on
`/docs/foo` should see the same thing the day of the release. Divergence
happens silently when:

- One locale was updated but the other wasn't.
- Old code samples weren't refreshed for the new API (the 0.7 cut had 13
  pages still importing the old `Model` symbol after the BoundModel
  redesign).
- A hero/landing card hardcodes a path to one specific provider page
  instead of using `custom-versionAwareDocLink` (the 0.7 hero linked to
  `/guides/providers/anthropic` instead of `/guides/providers/overview`,
  which then drifted between current and version-0.7).

After cutting, run `diff -r website/docs website/versioned_docs/version-X.Y`
and the zh-Hans equivalent. The only expected difference is `intro.mdx`
status blurb (Next vs. Released).

### 4. zh-Hans parity is your problem

`docusaurus docs:version` does mirror the zh-Hans locale into
`i18n/zh-Hans/docusaurus-plugin-content-docs/version-X.Y/` and writes
`version-X.Y.json`, but only the *files that exist* in `current/` at cut
time. If a guide page was added to EN `website/docs/` but its zh-Hans
mirror wasn't written to `website/i18n/zh-Hans/.../current/`, the snapshot
permanently has an EN-only page. Verify before cutting:

```bash
diff <(cd website/docs && find . -name '*.md*' | sort) \
     <(cd website/i18n/zh-Hans/docusaurus-plugin-content-docs/current && find . -name '*.md*' | sort)
```

Any output is a missing translation. Write it first, then cut.

### 5. Version-aware links, not hardcoded paths

Anywhere a doc, landing page, or React component links to another doc, use
the `custom-versionAwareDocLink` navbar item type or relative markdown
links (`./other-page` resolves correctly across versions). Hardcoded
absolute paths like `/docs/guides/providers/anthropic` work on the version
they were written for but send `0.6` readers to the `0.7` page after the
cut. The 0.7 hero card hit this — the fix is to point cross-section links
at section indices (`overview` pages) rather than specific subpages.

### 6. Don't commit generated API mdx into `current/`

CI has a guard that fails the build if `website/docs/api/cubepi-*.mdx` is
present in a commit (only `index.mdx` is allowed). The snapshot is allowed
to contain them because the path is different
(`versioned_docs/version-X.Y/api/`). If your local cut accidentally staged
`website/docs/api/cubepi-*.mdx`, unstage them — they regenerate on every
build.

### 7. `versions.json` order matters

The cut prepends `X.Y` to `versions.json`. Don't manually reorder — the
order is `[latest, ..., oldest]` and Docusaurus uses it for the version
dropdown. If you intentionally drop an old version, delete its entry
from `versions.json` **and** its `versioned_docs/version-N/` directory,
sidebar JSON, and any locale mirrors, plus its entry from the `versions`
block in `docusaurus.config.ts`.

### 8. Status blurb in `intro.mdx`

Both `website/docs/intro.mdx` and the zh-Hans mirror typically carry a
"Status" line that should mention what's new in `X.Y`. The snapshot
preserves whatever was in `current/` at cut time, so write the Released
copy in `current/` **before** the cut and then revert `current/`'s blurb
to the Next 🚧 copy after the cut so unreleased work isn't advertised as
shipped.

## Post-cut verification

After committing the cut, before opening the PR / tagging:

```bash
cd website
pnpm apiref               # regenerate (idempotent)
pnpm build                # must pass with no broken-link errors
pnpm test                 # vitest suite covers the version-aware nav
```

Then in the rendered site (`pnpm start` or the preview deploy):

- `/docs/` redirects to the new `X.Y` content with `X.Y (latest)` in the
  version dropdown.
- `/docs/next/` shows the same content with the `unreleased` banner.
- `/docs/N-1/` still serves the demoted version, but `<meta name="robots"
  content="noindex">` is present in its HTML head.
- `sitemap.xml` contains `/docs/...` paths but not `/docs/N-1/...` or
  `/docs/next/...`.

## After the cut

The release itself (tag, GitHub Release, PyPI publish) is a separate runbook
— but the order is: cut docs → bump pyproject.toml if not already → merge
PR → tag → push tag → create GitHub Release (triggers PyPI publish).
