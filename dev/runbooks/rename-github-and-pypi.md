# GitHub rename and PyPI publishers

Out-of-repo cutover for CubePi → CubeLoop. The code PR does **not** do
these steps; this runbook is the checklist after that PR lands (and the
parts already done before merge).

Do **not** open a second GitHub repository. GitHub's in-place rename
keeps issues, PRs, stars, Actions history, and redirects
`github.com/cubeplexai/cubepi` → `github.com/cubeplexai/cubeloop`.

## Already done (before the 0.14 PR)

These landed during the cutover window and do not need repeating:

| Item | Status |
|---|---|
| GitHub repo rename `cubeplexai/cubepi` → `cubeplexai/cubeloop` | Done |
| Cloudflare Pages custom domains `cubeloop.dev` / `www.cubeloop.dev` on project `cubepi` | Done (project name stays `cubepi`) |
| Zone Single Redirect: `cubepi.ai` / `www.cubepi.ai` → `https://cubeloop.dev${uri}` 301 | Done |
| `cubepi.pages.dev` path 301s (except Pages' built-in `/` alias) via `_worker.js` | Done in this PR |
| Google Search Console change of address `cubepi.ai` → `cubeloop.dev` | Done |
| GA4 stream URL / authorized domains; PostHog authorized URLs | Done |
| PyPI pending Trusted Publisher for **new** project `cubeloop` | Done: repo `cubeplexai/cubeloop`, workflow `publish.yml`, environment `pypi` |

The Cloudflare Pages project is still named `cubepi`. Leave it.
`docs.yml` `projectName: cubepi` is correct.

## After the code PR merges

1. **Confirm GitHub redirects.** `git clone https://github.com/cubeplexai/cubepi.git` should land in `cubeloop`. Update local remotes if they still say `cubepi.git` (`git remote set-url origin https://github.com/cubeplexai/cubeloop.git`).
2. **Cloudflare Pages GitHub connection.** If Pages is wired to the GitHub repo (this project is Direct Upload via `pages-action`, so usually nothing to change). After merge, the next `docs.yml` deploy on `main` publishes CubeLoop current docs to `cubeloop.dev`.
3. **Submit sitemaps** on the Search Console property `cubeloop.dev`:
   - `sitemap.xml`
   - `zh-Hans/sitemap.xml`
   Do this **after** the CubeLoop docs deploy, not before — live sitemaps still listed `cubepi.ai` URLs until then.
4. **PyPI / TestPyPI publishers** (verify before tagging):
   - **New project `cubeloop`:** pending publisher must match `Repository: cubeplexai/cubeloop`, `Workflow: publish.yml`, `Environment: pypi`. First successful upload **creates** the project; there is no "New project" button.
   - **TestPyPI** is a separate account: https://test.pypi.org/manage/account/publishing/ — same workflow/environment names, environment `testpypi`.
   - **Existing `cubepi`:** after the GitHub rename, edit the publisher at https://pypi.org/manage/project/cubepi/settings/publishing/ so the repository is `cubeplexai/cubeloop` (not `cubepi`). Same on TestPyPI.
5. **Tag `v0.14.0`.** `publish.yml` builds both dists (`uv build` + `uv build --package cubepi`) and uploads the whole `dist/` to TestPyPI then PyPI with `skip-existing: true`. A partial TestPyPI attempt can be retried.
6. **Smoke from a clean machine** (not the worktree):
   ```bash
   pip install cubeloop
   python -c "import cubeloop; print(cubeloop.__version__)"
   pip install -U cubepi
   python -c "import cubepi"   # DeprecationWarning + stderr banner
   ```
   Both indexes must have **both** files before calling the cut done. The shim's `Requires-Dist: cubeloop==0.14.0` will fail if only `cubepi` uploaded.
7. **Cut docs version 0.14** per [`cut-doc-version.md`](cut-doc-version.md) **after** `website/docs/` current is CubeLoop, so `version-0.14` is born as CubeLoop. Do not snapshot 0.13-branded current/.

## Do not

- Yank `cubepi` 0.13.x.
- Fail `pip install cubepi` (the 0.14 shim must keep working).
- Create `github.com/cubeplexai/cubeloop` as a new empty repo.
- Rename the Cloudflare Pages project (would change `cubepi.pages.dev`).
- Edit `website/versioned_docs/` or old CHANGELOG sections.

## Related

- Spec: `dev/specs/2026-09-09-rename-cubeloop.md`
- Plan: `dev/plans/2026-09-09-rename-cubeloop.md`
- Migrate page: `website/docs/migration/from-cubepi.md`
