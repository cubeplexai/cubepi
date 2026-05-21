# CubePi Documentation Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first public CubePi documentation site under `website/`, with versioning, EN + zh-Hans i18n, page-level PostHog feedback, a custom Operator-styled homepage, and an auto-generated API reference via `griffe`. Deploy to Cloudflare Pages on every push to `main`.

**Architecture:** Docusaurus 3.x lives in a `website/` subdirectory peer to the Python package. Python tooling (`uv` + `griffe`) feeds an auto-generated MDX API reference into the Docusaurus content tree at build time. The custom homepage is React + a hand-tuned CSS-token system in the Operator design language. PostHog is wired through a Docusaurus client module; a swizzled `theme/DocItem/Footer` mounts a 👍/👎 component that captures events directly. CI runs build-and-check on every PR (with broken-link / anchor / spelling guards) and deploys via `cloudflare/pages-action@v1` on `main`.

**Tech Stack:** Docusaurus 3.x, TypeScript, React, pnpm 9, Node 22, Python 3.11 + `uv`, `griffe>=0.45`, PostHog JS SDK, Cloudflare Pages, GitHub Actions.

**Spec:** `docs/specs/2026-05-15-cubepi-docs-site-design.md`

---

## Phase 0 — Repo prep

### Task 1: Migrate brand assets to `website/static/img/brand/`

**Files:**
- Move (4): `assets/brand/cubepi-logo.svg`, `assets/brand/cubepi-logo.png`, `assets/brand/cubepi-social-preview.svg`, `assets/brand/cubepi-social-preview.png` → `website/static/img/brand/`
- Delete: `assets/brand/` (after move)
- Delete: `assets/` (if empty after delete)
- Modify: `README.md` — image references

- [ ] **Step 1: Create the destination directory**

```bash
mkdir -p website/static/img/brand
```

- [ ] **Step 2: Move every brand asset**

```bash
git mv assets/brand/cubepi-logo.svg            website/static/img/brand/cubepi-logo.svg
git mv assets/brand/cubepi-logo.png            website/static/img/brand/cubepi-logo.png
git mv assets/brand/cubepi-social-preview.svg  website/static/img/brand/cubepi-social-preview.svg
git mv assets/brand/cubepi-social-preview.png  website/static/img/brand/cubepi-social-preview.png
```

- [ ] **Step 3: Remove empty old directories**

```bash
rmdir assets/brand 2>/dev/null
rmdir assets       2>/dev/null
```

- [ ] **Step 4: Update README image references**

Replace every occurrence of `assets/brand/` with `website/static/img/brand/` in `README.md`. Verify with:

```bash
grep -n 'assets/brand' README.md
```

Expected: no output. Then verify the new paths resolve:

```bash
grep -n 'website/static/img/brand' README.md
```

Expected: at least 1 hit (the top-of-file logo).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: relocate brand assets under website/static/img/brand/"
```

---

### Task 2: Ignore Docusaurus build artefacts

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Append website rules**

Add at the end of `.gitignore`:

```
# Docusaurus
website/.docusaurus/
website/build/
website/node_modules/
website/docs/api/*.mdx
!website/docs/api/_index.mdx
```

- [ ] **Step 2: Verify**

```bash
tail -8 .gitignore
```

Expected: shows the new block.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore website/ build artefacts and generated api MDX"
```

---

## Phase 1 — Docusaurus scaffold

### Task 3: Scaffold Docusaurus TypeScript project

**Files:**
- Create: `website/package.json`, `website/tsconfig.json`, `website/docusaurus.config.ts`, `website/sidebars.ts`, `website/src/css/custom.css`, `website/src/pages/index.tsx`, `website/docs/intro.md`, etc. (created by template)
- Modify: subsequent tasks overwrite many of these.

- [ ] **Step 1: Run the classic-TS template**

From repo root:

```bash
pnpm dlx create-docusaurus@latest website classic --typescript
```

When prompted, accept defaults.

- [ ] **Step 2: Verify the dev server boots**

```bash
cd website && pnpm install && pnpm start --no-open
```

Expected: console prints `[SUCCESS] Docusaurus website is running at: http://localhost:3000/`. Hit `Ctrl+C` once verified.

- [ ] **Step 3: Replace generated `docusaurus.config.ts`**

Overwrite `website/docusaurus.config.ts` with:

```ts
import type { Config } from '@docusaurus/types';
import { themes as prismThemes } from 'prism-react-renderer';

const POSTHOG_KEY = process.env.POSTHOG_KEY ?? '';
const POSTHOG_HOST = process.env.POSTHOG_HOST ?? 'https://us.i.posthog.com';
const GIT_SHA = process.env.GITHUB_SHA?.slice(0, 7) ?? 'dev';

const config: Config = {
  title: 'CubePi',
  tagline: 'A Pythonic, async-native agent framework',
  favicon: 'img/brand/cubepi-logo.svg',

  url: 'https://cubepi.pages.dev',
  baseUrl: '/',
  organizationName: 'cubeplexai',
  projectName: 'cubepi',

  onBrokenLinks: 'throw',
  onBrokenAnchors: 'throw',
  onBrokenMarkdownLinks: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'zh-Hans'],
    localeConfigs: {
      en:       { label: 'English' },
      'zh-Hans':{ label: '简体中文' },
    },
  },

  customFields: { POSTHOG_KEY, POSTHOG_HOST, GIT_SHA },

  clientModules: [require.resolve('./src/clientModules/posthog.ts')],

  presets: [
    ['classic', {
      docs: {
        sidebarPath: './sidebars.ts',
        editUrl: 'https://github.com/cubeplexai/cubepi/edit/main/website/',
        lastVersion: 'current',
        versions: {
          current: { label: 'Next 🚧', path: 'next', banner: 'unreleased' },
        },
      },
      blog: false,
      theme: {
        customCss: './src/css/custom.css',
      },
    } satisfies import('@docusaurus/preset-classic').Options],
  ],

  themeConfig: {
    image: 'img/brand/cubepi-social-preview.png',
    navbar: {
      title: 'CubePi',
      logo: { alt: 'CubePi logo', src: 'img/brand/cubepi-logo.svg' },
      items: [
        { to: '/getting-started/installation', label: 'Docs', position: 'left' },
        { to: '/api/',                         label: 'API',  position: 'left' },
        { to: '/recipes/',                     label: 'Recipes', position: 'left' },
        { type: 'docsVersionDropdown', position: 'right' },
        { type: 'localeDropdown',      position: 'right' },
        { href: 'https://github.com/cubeplexai/cubepi', label: 'GitHub', position: 'right' },
      ],
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'toml'],
    },
    colorMode: { defaultMode: 'light', respectPrefersColorScheme: true },
  },
};

export default config;
```

(Task 5 wires versioning to `0.3` once the first snapshot is taken. For now we run with `current` only.)

- [ ] **Step 4: Confirm config compiles**

```bash
cd website && pnpm build
```

Expected: `[SUCCESS] Generated static files in "build".` Two warnings about `clientModules` / `versionDropdown` referencing files-that-don't-exist-yet are normal — we'll add them in later tasks. **If the build errors**, fix the path before continuing.

- [ ] **Step 5: Stage and commit the scaffold**

```bash
git add website/
git commit -m "feat(website): scaffold Docusaurus 3 with TS preset and base config"
```

---

### Task 4: Add the `docs` Python extra with `griffe`

**Files:**
- Modify: `pyproject.toml:24-35`

- [ ] **Step 1: Add the extra**

After the `mcp = [...]` block in `[project.optional-dependencies]`, append:

```toml
docs = [
    "griffe>=0.45",
]
```

- [ ] **Step 2: Sync and verify import**

```bash
uv sync --extra docs
uv run python -c "import griffe; print(griffe.__version__)"
```

Expected: a version string ≥ 0.45.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build(deps): add docs extra with griffe>=0.45 for API reference generation"
```

---

## Phase 2 — Operator design tokens & fonts

### Task 5: Self-host fonts

**Files:**
- Create (6): `website/static/fonts/Inter-Regular.woff2`, `website/static/fonts/Inter-Medium.woff2`, `website/static/fonts/Inter-SemiBold.woff2`, `website/static/fonts/InterTight-SemiBold.woff2`, `website/static/fonts/JetBrainsMono-Regular.woff2`, `website/static/fonts/JetBrainsMono-Medium.woff2`

- [ ] **Step 1: Fetch font files**

Use Fontsource CDN sources (Apache 2.0 / OFL). Run from repo root:

```bash
mkdir -p website/static/fonts
cd website/static/fonts

curl -L -o Inter-Regular.woff2          'https://cdn.jsdelivr.net/fontsource/fonts/inter@latest/latin-400-normal.woff2'
curl -L -o Inter-Medium.woff2           'https://cdn.jsdelivr.net/fontsource/fonts/inter@latest/latin-500-normal.woff2'
curl -L -o Inter-SemiBold.woff2         'https://cdn.jsdelivr.net/fontsource/fonts/inter@latest/latin-600-normal.woff2'
curl -L -o InterTight-SemiBold.woff2    'https://cdn.jsdelivr.net/fontsource/fonts/inter-tight@latest/latin-600-normal.woff2'
curl -L -o JetBrainsMono-Regular.woff2  'https://cdn.jsdelivr.net/fontsource/fonts/jetbrains-mono@latest/latin-400-normal.woff2'
curl -L -o JetBrainsMono-Medium.woff2   'https://cdn.jsdelivr.net/fontsource/fonts/jetbrains-mono@latest/latin-500-normal.woff2'
```

- [ ] **Step 2: Verify all six files exist and are non-zero**

```bash
ls -lh website/static/fonts/
```

Expected: 6 entries, each between 20–80 KB.

- [ ] **Step 3: Commit**

```bash
git add website/static/fonts/
git commit -m "feat(website): self-host Inter, Inter Tight, JetBrains Mono"
```

---

### Task 6: Apply Operator design tokens

**Files:**
- Overwrite: `website/src/css/custom.css`

- [ ] **Step 1: Replace `custom.css` with Operator tokens**

```css
/* ============== Fonts ============== */
@font-face { font-family: 'Inter';        src: url('/fonts/Inter-Regular.woff2')        format('woff2'); font-weight: 400; font-display: swap; }
@font-face { font-family: 'Inter';        src: url('/fonts/Inter-Medium.woff2')         format('woff2'); font-weight: 500; font-display: swap; }
@font-face { font-family: 'Inter';        src: url('/fonts/Inter-SemiBold.woff2')       format('woff2'); font-weight: 600; font-display: swap; }
@font-face { font-family: 'Inter Tight';  src: url('/fonts/InterTight-SemiBold.woff2')  format('woff2'); font-weight: 600; font-display: swap; }
@font-face { font-family: 'JetBrains Mono'; src: url('/fonts/JetBrainsMono-Regular.woff2') format('woff2'); font-weight: 400; font-display: swap; }
@font-face { font-family: 'JetBrains Mono'; src: url('/fonts/JetBrainsMono-Medium.woff2')  format('woff2'); font-weight: 500; font-display: swap; }

/* ============== Operator tokens ============== */
:root {
  --ink-12: #0a0a0b;
  --ink-11: #1f1f22;
  --ink-9:  #52525b;
  --ink-7:  #a1a1aa;
  --ink-5:  #e4e4e7;
  --ink-3:  #f4f4f5;
  --ink-1:  #fafafa;
  --surface: #ffffff;
  --accent: #3b5bd9;
  --accent-soft: #eef1fe;
  --accent-ink: #1e3aa8;
  --ok:   #16a34a;
  --warn: #d97706;
  --err:  #dc2626;

  /* Docusaurus overrides */
  --ifm-color-primary: var(--accent);
  --ifm-color-primary-dark:        #2f4fc4;
  --ifm-color-primary-darker:      #2c4abc;
  --ifm-color-primary-darkest:     #243d9b;
  --ifm-color-primary-light:       #5470de;
  --ifm-color-primary-lighter:     #6a83e3;
  --ifm-color-primary-lightest:    #93a6ec;
  --ifm-background-color: var(--ink-1);
  --ifm-background-surface-color: var(--surface);
  --ifm-color-content: var(--ink-11);
  --ifm-heading-color: var(--ink-12);
  --ifm-font-family-base: 'Inter', system-ui, -apple-system, sans-serif;
  --ifm-heading-font-family: 'Inter Tight', 'Inter', system-ui;
  --ifm-font-family-monospace: 'JetBrains Mono', ui-monospace, monospace;
  --ifm-font-size-base: 14px;
  --ifm-line-height-base: 1.55;
  --ifm-navbar-height: 56px;
  --ifm-navbar-background-color: var(--surface);
  --ifm-navbar-shadow: none;
  --ifm-toc-border-color: var(--ink-5);
  --ifm-hr-border-color: var(--ink-5);
  --ifm-code-font-size: 92%;
  --docusaurus-highlighted-code-line-bg: var(--ink-3);
}

[data-theme='dark'] {
  --ink-12: #f4f4f5;
  --ink-11: #e4e4e7;
  --ink-9:  #a1a1aa;
  --ink-7:  #71717a;
  --ink-5:  #3f3f46;
  --ink-3:  #27272a;
  --ink-1:  #18181b;
  --surface: #0a0a0b;
  --accent: #6a83e3;
  --accent-soft: #1e3aa820;
  --accent-ink: #93a6ec;
  --ifm-color-primary: var(--accent);
  --ifm-background-color: var(--ink-1);
  --ifm-background-surface-color: var(--surface);
}

html, body { font-feature-settings: 'cv11', 'ss01'; -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility; }

/* No shadows except modals (which Docusaurus does not use in docs). */
.navbar { border-bottom: 1px solid var(--ink-5); }
.menu__link, .navbar__link { font-weight: 500; }
.menu__link--active, .navbar__link--active { color: var(--accent); }

/* Tabular numerals globally in code and tables */
code, pre, table td.num, .num { font-variant-numeric: tabular-nums; }
```

- [ ] **Step 2: Rebuild and visually inspect**

```bash
cd website && pnpm start --no-open
```

Open `http://localhost:3000` in a browser. Expected: the default Docusaurus homepage now uses Inter Tight headings, ink-1 background, and the `#3b5bd9` accent on links. Hit `Ctrl+C` when satisfied.

- [ ] **Step 3: Commit**

```bash
git add website/src/css/custom.css
git commit -m "feat(website): apply Operator design tokens and Docusaurus overrides"
```

---

## Phase 3 — Content scaffold (English)

### Task 7: Replace default content with the IA skeleton

**Files:**
- Delete: every file under `website/docs/` produced by the template (`intro.md`, `tutorial-basics/`, `tutorial-extras/`).
- Delete: `website/blog/` (whole directory — blog is disabled).
- Create (placeholder pages, one paragraph each):
  - `website/docs/intro.md`
  - `website/docs/getting-started/installation.md`
  - `website/docs/getting-started/quick-start.md`
  - `website/docs/getting-started/core-concepts.md`
  - `website/docs/guides/agents/first-agent.md`
  - `website/docs/guides/agents/tool-use.md`
  - `website/docs/guides/agents/multi-turn.md`
  - `website/docs/guides/agents/streaming.md`
  - `website/docs/guides/providers/anthropic.md`
  - `website/docs/guides/providers/openai.md`
  - `website/docs/guides/providers/custom.md`
  - `website/docs/guides/checkpointing/sqlite.md`
  - `website/docs/guides/checkpointing/postgres.md`
  - `website/docs/guides/checkpointing/custom.md`
  - `website/docs/guides/middleware/hooks.md`
  - `website/docs/guides/middleware/composition.md`
  - `website/docs/guides/middleware/examples.md`
  - `website/docs/guides/mcp/loading.md`
  - `website/docs/guides/mcp/auth.md`
  - `website/docs/api/_index.mdx`
  - `website/docs/recipes/weather-agent.md`
  - `website/docs/recipes/multi-provider-failover.md`
  - `website/docs/recipes/persistent-chat.md`
  - `website/docs/recipes/resumable-tasks.md`
  - `website/docs/recipes/postgres-fastapi.md`
  - `website/docs/migration/from-langgraph.md`

- [ ] **Step 1: Clear template content**

```bash
rm -rf website/docs website/blog
mkdir -p website/docs/getting-started \
         website/docs/guides/{agents,providers,checkpointing,middleware,mcp} \
         website/docs/api \
         website/docs/recipes \
         website/docs/migration
```

- [ ] **Step 2: Create the entry page**

`website/docs/intro.md`:

```markdown
---
id: intro
title: CubePi
slug: /
sidebar_position: 0
---

# CubePi

A Pythonic, async-native agent framework.

Start with the [Quick Start](/getting-started/quick-start) or browse the [API Reference](/api/).
```

- [ ] **Step 3: Stub every other page**

For each path in the *Create* list above (except `intro.md` and `api/_index.mdx`), create a file with the following template, substituting `<TITLE>` with a human-readable title derived from the path (e.g. `getting-started/installation.md` → `Installation`):

```markdown
---
title: <TITLE>
---

# <TITLE>

_Placeholder. Content arrives in a follow-up content PR._
```

You can do this with a script:

```bash
cd website/docs
for f in \
  getting-started/installation.md \
  getting-started/quick-start.md \
  getting-started/core-concepts.md \
  guides/agents/first-agent.md \
  guides/agents/tool-use.md \
  guides/agents/multi-turn.md \
  guides/agents/streaming.md \
  guides/providers/anthropic.md \
  guides/providers/openai.md \
  guides/providers/custom.md \
  guides/checkpointing/sqlite.md \
  guides/checkpointing/postgres.md \
  guides/checkpointing/custom.md \
  guides/middleware/hooks.md \
  guides/middleware/composition.md \
  guides/middleware/examples.md \
  guides/mcp/loading.md \
  guides/mcp/auth.md \
  recipes/weather-agent.md \
  recipes/multi-provider-failover.md \
  recipes/persistent-chat.md \
  recipes/resumable-tasks.md \
  recipes/postgres-fastapi.md \
  migration/from-langgraph.md ; do
  title=$(basename "${f%.md}" | sed 's/-/ /g' | awk '{for(i=1;i<=NF;i++)$i=toupper(substr($i,1,1))substr($i,2)} 1')
  cat > "$f" <<EOF
---
title: $title
---

# $title

_Placeholder. Content arrives in a follow-up content PR._
EOF
done
```

- [ ] **Step 4: Create the API index**

`website/docs/api/_index.mdx`:

```mdx
---
title: API Reference
slug: /api/
---

# API Reference

Auto-generated from CubePi's public modules.

- [cubepi.agent](./cubepi-agent.mdx)
- [cubepi.providers](./cubepi-providers.mdx)
- [cubepi.checkpointer](./cubepi-checkpointer.mdx)
- [cubepi.middleware](./cubepi-middleware.mdx)
- [cubepi.mcp](./cubepi-mcp.mdx)
- [cubepi.utils](./cubepi-utils.mdx)

The module pages above are generated at build time from docstrings and are not editable by hand.
```

- [ ] **Step 5: Write the sidebar**

Overwrite `website/sidebars.ts`:

```ts
import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docs: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/installation',
        'getting-started/quick-start',
        'getting-started/core-concepts',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      items: [
        { type: 'category', label: 'Agents', items: [
          'guides/agents/first-agent',
          'guides/agents/tool-use',
          'guides/agents/multi-turn',
          'guides/agents/streaming',
        ]},
        { type: 'category', label: 'Providers', items: [
          'guides/providers/anthropic',
          'guides/providers/openai',
          'guides/providers/custom',
        ]},
        { type: 'category', label: 'Checkpointing', items: [
          'guides/checkpointing/sqlite',
          'guides/checkpointing/postgres',
          'guides/checkpointing/custom',
        ]},
        { type: 'category', label: 'Middleware', items: [
          'guides/middleware/hooks',
          'guides/middleware/composition',
          'guides/middleware/examples',
        ]},
        { type: 'category', label: 'MCP', items: [
          'guides/mcp/loading',
          'guides/mcp/auth',
        ]},
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      link: { type: 'doc', id: 'api/_index' },
      items: [
        { type: 'autogenerated', dirName: 'api' },
      ],
    },
    {
      type: 'category',
      label: 'Recipes',
      items: [
        'recipes/weather-agent',
        'recipes/multi-provider-failover',
        'recipes/persistent-chat',
        'recipes/resumable-tasks',
        'recipes/postgres-fastapi',
      ],
    },
    {
      type: 'category',
      label: 'Migration',
      items: ['migration/from-langgraph'],
    },
  ],
};

export default sidebars;
```

- [ ] **Step 6: Verify the site builds**

```bash
cd website && pnpm build
```

Expected: build succeeds. `Broken link` errors mean a path is wrong — fix and rerun.

- [ ] **Step 7: Commit**

```bash
git add website/docs/ website/sidebars.ts website/docusaurus.config.ts
git rm -r --cached website/blog 2>/dev/null || true
git commit -m "feat(website): replace template content with cubepi IA skeleton"
```

---

## Phase 4 — API reference pipeline

### Task 8: Build the `griffe → MDX` generator with tests

**Files:**
- Create: `website/scripts/build-api-reference.py`
- Create: `website/scripts/tests/test_build_api_reference.py`
- Create: `website/scripts/tests/__init__.py` (empty)
- Modify: `pyproject.toml` — `[tool.pytest.ini_options].testpaths`

- [ ] **Step 1: Add the script's test path to pytest**

In `pyproject.toml`, change:

```toml
testpaths = ["tests"]
```

to:

```toml
testpaths = ["tests", "website/scripts/tests"]
```

- [ ] **Step 2: Write the first failing test**

Create `website/scripts/tests/__init__.py` empty. Then `website/scripts/tests/test_build_api_reference.py`:

```python
"""Tests for the griffe → MDX API reference generator."""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "build_api_reference.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_api_reference", SCRIPT_PATH)
    assert spec and spec.loader, f"cannot load {SCRIPT_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_api_reference"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_render_signature_includes_return_annotation():
    mod = _load_module()
    rendered = mod.render_signature(
        name="run",
        parameters=[("self", None, None), ("x", "int", None)],
        returns="str",
    )
    assert "run(self, x: int) -> str" in rendered


def test_render_docstring_parses_google_sections():
    mod = _load_module()
    doc = (
        "Summary line.\n\n"
        "Args:\n"
        "    name (str): The name.\n\n"
        "Returns:\n"
        "    bool: Whether it worked.\n"
    )
    out = mod.render_docstring(doc)
    assert "Summary line." in out
    assert "**Args**" in out
    assert "`name`" in out
    assert "**Returns**" in out


def test_emit_module_writes_generated_marker(tmp_path):
    mod = _load_module()
    module_path = tmp_path / "cubepi-utils.mdx"
    mod.emit_module(
        out_path=module_path,
        module_name="cubepi.utils",
        sidebar_position=6,
        symbols=[],
        commit_sha="abc1234",
    )
    body = module_path.read_text(encoding="utf-8")
    assert body.startswith("---\n")
    assert "title: cubepi.utils" in body
    assert "<!-- GENERATED by build-api-reference.py — DO NOT EDIT -->" in body
```

- [ ] **Step 3: Run tests, verify they fail because the script doesn't exist**

```bash
uv run pytest website/scripts/tests/ -v
```

Expected: `AssertionError: cannot load …/build_api_reference.py` or `FileNotFoundError`.

- [ ] **Step 4: Implement the script**

Create `website/scripts/build-api-reference.py` (note: hyphenated for the CLI command, but the test loads it through `importlib` so the filename hyphen is fine). Use **`build_api_reference.py`** (underscored) so Python `importlib.util` can load it as a module:

Rename: `website/scripts/build_api_reference.py` is the canonical filename. The npm `prebuild` script (Task 9) will invoke it via `python website/scripts/build_api_reference.py`.

Content:

```python
"""Generate Docusaurus-compatible MDX from cubepi public API via griffe."""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import griffe

MODULES = [
    ("cubepi.agent",        "Agents",        1),
    ("cubepi.providers",    "Providers",     2),
    ("cubepi.checkpointer", "Checkpointing", 3),
    ("cubepi.middleware",   "Middleware",    4),
    ("cubepi.mcp",          "MCP",           5),
    ("cubepi.utils",        "Utils",         6),
]


def _is_public(name: str, parent_all: list[str] | None) -> bool:
    if parent_all is not None:
        return name in parent_all
    return not name.startswith("_")


def collect_public_symbols(module: griffe.Module) -> list[griffe.Object]:
    parent_all = module.exports if hasattr(module, "exports") else None
    out: list[griffe.Object] = []
    for member_name, member in module.members.items():
        if not _is_public(member_name, parent_all):
            continue
        if member.is_alias:
            try:
                member = member.final_target
            except griffe.AliasResolutionError:
                continue
        out.append(member)
    return out


def render_signature(name: str, parameters: list[tuple[str, str | None, str | None]], returns: str | None) -> str:
    parts = []
    for pname, ptype, pdefault in parameters:
        s = pname
        if ptype:
            s += f": {ptype}"
        if pdefault:
            s += f" = {pdefault}"
        parts.append(s)
    sig = f"{name}({', '.join(parts)})"
    if returns:
        sig += f" -> {returns}"
    return f"```python\n{sig}\n```"


_GOOGLE_SECTIONS = ("Args", "Arguments", "Returns", "Raises", "Yields", "Example", "Examples", "Note", "Notes")


def render_docstring(text: str | None) -> str:
    if not text:
        return ""
    lines = text.strip("\n").splitlines()
    out: list[str] = []
    in_section: str | None = None
    for line in lines:
        m = re.match(r"^([A-Z][a-zA-Z]+):\s*$", line.strip())
        if m and m.group(1) in _GOOGLE_SECTIONS:
            in_section = m.group(1)
            out.append("")
            out.append(f"**{in_section}**")
            out.append("")
            continue
        if in_section in {"Args", "Arguments"} and line.startswith(("    ", "\t")):
            stripped = line.strip()
            arg_m = re.match(r"^(\w+)\s*(\([^)]+\))?:\s*(.*)$", stripped)
            if arg_m:
                arg, _ty, desc = arg_m.groups()
                out.append(f"- `{arg}` — {desc}")
                continue
        if in_section in {"Returns", "Yields"} and line.startswith(("    ", "\t")):
            out.append(f"- {line.strip()}")
            continue
        if in_section == "Raises" and line.startswith(("    ", "\t")):
            stripped = line.strip()
            raise_m = re.match(r"^(\w+):\s*(.*)$", stripped)
            if raise_m:
                exc, desc = raise_m.groups()
                out.append(f"- `{exc}` — {desc}")
                continue
        out.append(line)
    return "\n".join(out).strip() + "\n"


def render_symbol(symbol: griffe.Object, github_blob_root: str) -> str:
    name = symbol.name
    kind = symbol.kind.value
    block: list[str] = [f"### {name}", "", f"_{kind}_", ""]

    if hasattr(symbol, "parameters"):
        params: list[tuple[str, str | None, str | None]] = []
        for p in symbol.parameters:
            ptype = str(p.annotation) if p.annotation else None
            pdefault = str(p.default) if p.default is not None else None
            params.append((p.name, ptype, pdefault))
        returns = str(symbol.returns) if getattr(symbol, "returns", None) else None
        block.append(render_signature(name, params, returns))
        block.append("")

    block.append(render_docstring(symbol.docstring.value if symbol.docstring else None))

    if symbol.filepath and symbol.lineno:
        rel = Path(symbol.filepath).as_posix()
        rel = rel.split("cubepi/")[-1]
        link = f"{github_blob_root}/cubepi/{rel}#L{symbol.lineno}"
        block.append(f"[source]({link})")
        block.append("")

    return "\n".join(block)


def emit_module(out_path: Path, module_name: str, sidebar_position: int,
                symbols: Iterable, commit_sha: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = (
        "---\n"
        f"id: {module_name.replace('.', '-')}\n"
        f"title: {module_name}\n"
        f"sidebar_position: {sidebar_position}\n"
        "hide_table_of_contents: false\n"
        "---\n\n"
    )
    body = [f"# `{module_name}`", ""]
    for sym in symbols:
        body.append(render_symbol(sym, github_blob_root=f"https://github.com/cubeplexai/cubepi/blob/{commit_sha}"))
        body.append("")
    body.append("")
    body.append("<!-- GENERATED by build-api-reference.py — DO NOT EDIT -->")
    out_path.write_text(frontmatter + "\n".join(body), encoding="utf-8")


def current_commit_sha() -> str:
    env = os.environ.get("GITHUB_SHA")
    if env:
        return env
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "main"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path,
                        help="Output directory, typically website/docs/api/")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sha = current_commit_sha()
    loader = griffe.GriffeLoader()
    loader.load("cubepi")

    for mod_name, _label, position in MODULES:
        try:
            mod = loader.modules_collection[mod_name]
        except KeyError:
            print(f"[warn] {mod_name} not importable; skipping", file=sys.stderr)
            continue
        symbols = collect_public_symbols(mod)
        out_path = args.out / f"{mod_name.replace('.', '-')}.mdx"
        emit_module(out_path, mod_name, position, symbols, sha)
        print(f"[ok] wrote {out_path} ({len(symbols)} symbols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note: tests load by path so they pass with this filename. The hyphenated CLI alias (`build-api-reference.py`) is unnecessary — refer to it everywhere as `build_api_reference.py`. Update the spec mentally if there is a contradiction.

- [ ] **Step 5: Run the tests, verify they pass**

```bash
uv run pytest website/scripts/tests/ -v
```

Expected: 3 passing.

- [ ] **Step 6: Smoke-run the script end-to-end**

```bash
uv run python website/scripts/build_api_reference.py --out website/docs/api/
ls website/docs/api/*.mdx
```

Expected: 6 mdx files (`cubepi-agent.mdx`, …, `cubepi-utils.mdx`), each with `<!-- GENERATED` trailer.

- [ ] **Step 7: Verify Docusaurus builds with the generated content**

```bash
cd website && pnpm build
```

Expected: build succeeds. **Likely surprises:**
- Some symbols may emit raw `<` / `>` from generics — if MDX parser complains, escape with `&lt;` and `&gt;` in `render_signature` before pasting into a code fence (already inside ```python fence, so MDX shouldn't try to parse — but if a docstring contains a raw `<`, the test will still work; only fix in `render_docstring` if a real build error appears).

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml website/scripts/
git commit -m "feat(website): griffe-based API reference generator with tests"
```

---

### Task 9: Wire the prebuild hook

**Files:**
- Modify: `website/package.json`

- [ ] **Step 1: Add scripts**

In `website/package.json` `"scripts"`, add `prebuild` and `apiref`:

```json
"scripts": {
  "docusaurus": "docusaurus",
  "start": "docusaurus start",
  "build": "docusaurus build",
  "swizzle": "docusaurus swizzle",
  "deploy": "docusaurus deploy",
  "clear": "docusaurus clear",
  "serve": "docusaurus serve",
  "write-translations": "docusaurus write-translations",
  "write-heading-ids": "docusaurus write-heading-ids",
  "typecheck": "tsc",
  "apiref": "cd .. && uv run python website/scripts/build_api_reference.py --out website/docs/api/",
  "prebuild": "pnpm apiref",
  "prestart": "pnpm apiref",
  "check": "pnpm build && pnpm typecheck"
}
```

- [ ] **Step 2: Verify the generated MDX is excluded from git**

```bash
git status website/docs/api/
```

Expected: only `_index.mdx` (if any state). The generated `cubepi-*.mdx` files should not appear — they're matched by the `.gitignore` rule from Task 2.

- [ ] **Step 3: Smoke `pnpm build`**

```bash
cd website && pnpm build
```

Expected: prebuild step runs griffe; main build succeeds.

- [ ] **Step 4: Commit**

```bash
git add website/package.json
git commit -m "build(website): run griffe before every Docusaurus build/start"
```

---

## Phase 5 — Custom homepage

### Task 10: Hero component

**Files:**
- Create: `website/src/components/Home/Hero.tsx`
- Create: `website/src/components/Home/Hero.module.css`

- [ ] **Step 1: `Hero.tsx`**

```tsx
import React from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './Hero.module.css';

export default function Hero() {
  const social = useBaseUrl('/img/brand/cubepi-social-preview.png');
  const installCmd = 'pip install cubepi';

  return (
    <section className={styles.hero}>
      <img
        className={styles.banner}
        src={social}
        alt="CubePi · A Pythonic, async-native agent framework"
        width={1280}
        height={640}
      />
      <div className={styles.eyebrow}>cubepi · v0.3.0 · alpha</div>
      <h1 className={styles.h1}>A Pythonic, async-native agent framework.</h1>
      <p className={styles.lead}>
        Plain async functions instead of graph nodes. 3 deps. Append-only checkpointing.
      </p>
      <div className={styles.actions}>
        <button
          type="button"
          className={`${styles.cta} ${styles.ctaPrimary}`}
          onClick={() => navigator.clipboard?.writeText(installCmd)}
          aria-label={`Copy install command: ${installCmd}`}
        >
          <code>{installCmd}</code>
          <kbd className={styles.kbd}>⌘C</kbd>
        </button>
        <Link className={`${styles.cta} ${styles.ctaGhost}`} to="/getting-started/quick-start">
          Quick Start →
          <kbd className={styles.kbd}>G Q</kbd>
        </Link>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: `Hero.module.css`**

```css
.hero {
  max-width: 880px;
  margin: 48px auto 64px;
  padding: 0 24px;
  text-align: left;
}
.banner {
  display: block;
  width: 100%;
  height: auto;
  border: 1px solid var(--ink-5);
  border-radius: 6px;
  margin-bottom: 40px;
}
.eyebrow {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 12px;
  color: var(--ink-7);
  letter-spacing: 0.06em;
  margin-bottom: 12px;
}
.h1 {
  font-family: 'Inter Tight', 'Inter', system-ui;
  font-size: 44px;
  line-height: 1.08;
  letter-spacing: -0.02em;
  font-weight: 600;
  color: var(--ink-12);
  margin: 0 0 16px;
}
.lead {
  font-size: 16px;
  line-height: 1.55;
  color: var(--ink-9);
  margin: 0 0 28px;
  max-width: 640px;
}
.actions { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
.cta {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  height: 36px;
  padding: 0 14px;
  border: 1px solid var(--ink-5);
  border-radius: 5px;
  background: var(--surface);
  font-size: 14px;
  color: var(--ink-11);
  font-weight: 500;
  cursor: pointer;
  text-decoration: none;
}
.cta:hover { background: var(--ink-3); }
.ctaPrimary { background: var(--ink-12); border-color: var(--ink-12); color: var(--surface); }
.ctaPrimary:hover { background: var(--ink-11); }
.ctaPrimary code { background: transparent; color: inherit; font-size: 13px; }
.ctaGhost { background: transparent; }
.kbd {
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 10.5px;
  height: 18px;
  min-width: 18px;
  padding: 0 5px;
  display: inline-flex;
  align-items: center;
  border: 1px solid var(--ink-5);
  border-bottom-width: 2px;
  border-radius: 4px;
  background: var(--surface);
  color: var(--ink-9);
  opacity: 0.85;
}
.ctaPrimary .kbd { background: rgba(255,255,255,0.12); color: rgba(255,255,255,0.85); border-color: transparent; }
@media (max-width: 640px) {
  .h1 { font-size: 32px; }
  .hero { margin-top: 24px; }
}
```

- [ ] **Step 3: Commit**

```bash
git add website/src/components/Home/Hero.tsx website/src/components/Home/Hero.module.css
git commit -m "feat(website): Hero component with social preview banner"
```

---

### Task 11: WhyTable component

**Files:**
- Create: `website/src/components/Home/WhyTable.tsx`
- Create: `website/src/components/Home/WhyTable.module.css`

- [ ] **Step 1: `WhyTable.tsx`**

```tsx
import React from 'react';
import styles from './WhyTable.module.css';

const ROWS: { label: string; langgraph: string; cubepi: string }[] = [
  { label: 'Abstraction',     langgraph: 'Graph nodes + edges + channels',                          cubepi: 'Plain async functions — run_agent_loop is a while loop' },
  { label: 'Streaming',       langgraph: 'Callback-based, multiple handler types',                  cubepi: 'async for event in stream — one pattern everywhere' },
  { label: 'Checkpointing',   langgraph: 'Full snapshot per step; serializes entire message list',  cubepi: 'Append-only — O(1) DB I/O regardless of conversation length' },
  { label: 'Dependencies',    langgraph: 'langchain-core, langgraph-sdk, and transitive deps',      cubepi: '3 core deps: pydantic, anthropic, openai' },
  { label: 'Tool execution',  langgraph: 'Tools are graph nodes with manual wiring',                cubepi: 'Declare tools as functions; framework routes and parallelizes' },
  { label: 'Multi-provider',  langgraph: 'Via langchain chat model adapters',                       cubepi: 'Native Provider protocol — Anthropic, OpenAI built in' },
  { label: 'Middleware',      langgraph: 'Graph-level middleware on node entry/exit',               cubepi: '5 typed hooks with declarative composition rules' },
  { label: 'Observability',   langgraph: 'LangSmith / Langfuse integration',                        cubepi: 'Events + middleware hooks — bring your own tracing' },
];

export default function WhyTable() {
  return (
    <section className={styles.section}>
      <h2 className={styles.h2}>Why CubePi</h2>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th></th>
              <th>langgraph</th>
              <th>CubePi</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((r) => (
              <tr key={r.label}>
                <td className={styles.label}>{r.label}</td>
                <td>{r.langgraph}</td>
                <td className={styles.us}>{r.cubepi}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: `WhyTable.module.css`**

```css
.section { max-width: 1080px; margin: 0 auto 64px; padding: 0 24px; }
.h2 {
  font-family: 'Inter Tight';
  font-size: 11px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  font-weight: 600;
  color: var(--ink-7);
  margin-bottom: 16px;
}
.tableWrap { overflow-x: auto; border: 1px solid var(--ink-5); border-radius: 8px; background: var(--surface); }
.table { width: 100%; border-collapse: collapse; font-size: 13px; }
.table th, .table td {
  padding: 12px 16px;
  border-bottom: 1px solid var(--ink-5);
  text-align: left;
  vertical-align: top;
  line-height: 1.55;
}
.table tr:last-child td { border-bottom: 0; }
.table th { font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--ink-7); background: var(--ink-3); }
.label { font-weight: 500; color: var(--ink-12); width: 22%; }
.us { color: var(--ink-12); font-family: 'JetBrains Mono'; font-size: 12.5px; }
```

- [ ] **Step 3: Commit**

```bash
git add website/src/components/Home/WhyTable.tsx website/src/components/Home/WhyTable.module.css
git commit -m "feat(website): WhyTable comparison component"
```

---

### Task 12: HelloAgent component

**Files:**
- Create: `website/src/components/Home/HelloAgent.tsx`
- Create: `website/src/components/Home/HelloAgent.module.css`

- [ ] **Step 1: `HelloAgent.tsx`**

```tsx
import React from 'react';
import Link from '@docusaurus/Link';
import CodeBlock from '@theme/CodeBlock';
import styles from './HelloAgent.module.css';

const SAMPLE = `import asyncio
from cubepi import Agent, AgentTool, Model
from cubepi.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(api_key="sk-...")

def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"72°F and sunny in {city}"

agent = Agent(
    model=Model(provider=provider, model="claude-sonnet-4-5-20250929"),
    tools=[AgentTool(
        name="get_weather",
        description="Get current weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        execute=get_weather,
    )],
    system_prompt="You are a helpful weather assistant.",
)

async def main():
    stream = await agent.prompt("What's the weather in Tokyo?")
    async for event in stream:
        if event.type == "text_delta":
            print(event.delta, end="", flush=True)

asyncio.run(main())
`;

export default function HelloAgent() {
  return (
    <section className={styles.section}>
      <div className={styles.left}>
        <h2 className={styles.h2}>Hello, agent.</h2>
        <p className={styles.lede}>
          A single async function loop. One <code>Provider</code>, one <code>AgentTool</code>, and you're streaming.
        </p>
        <Link to="/getting-started/quick-start" className={styles.link}>
          Full quick-start →
        </Link>
      </div>
      <div className={styles.right}>
        <CodeBlock language="python">{SAMPLE}</CodeBlock>
      </div>
    </section>
  );
}
```

- [ ] **Step 2: `HelloAgent.module.css`**

```css
.section {
  max-width: 1080px;
  margin: 0 auto 64px;
  padding: 0 24px;
  display: grid;
  grid-template-columns: 1fr 1.4fr;
  gap: 40px;
  align-items: start;
}
.h2 { font-family: 'Inter Tight'; font-size: 24px; letter-spacing: -0.015em; color: var(--ink-12); margin: 0 0 12px; }
.lede { font-size: 14px; color: var(--ink-9); line-height: 1.55; margin: 0 0 16px; }
.link { font-size: 13px; color: var(--accent); }
.right :global(.theme-code-block) { margin: 0; }
@media (max-width: 768px) {
  .section { grid-template-columns: 1fr; gap: 16px; }
}
```

- [ ] **Step 3: Commit**

```bash
git add website/src/components/Home/HelloAgent.tsx website/src/components/Home/HelloAgent.module.css
git commit -m "feat(website): HelloAgent code-block section"
```

---

### Task 13: FeatureGrid component

**Files:**
- Create: `website/src/components/Home/FeatureGrid.tsx`
- Create: `website/src/components/Home/FeatureGrid.module.css`

- [ ] **Step 1: `FeatureGrid.tsx`**

Inline 16px Lucide-style monoline icons as plain SVG (no library — Operator dictates "no decorative icon set"). Six cards:

```tsx
import React from 'react';
import Link from '@docusaurus/Link';
import styles from './FeatureGrid.module.css';

type Icon = React.FC<React.SVGProps<SVGSVGElement>>;

const IconBox: Icon = (p) => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" {...p}>
    <rect x="2" y="2" width="12" height="12" rx="2" />
  </svg>
);
const IconStream: Icon = (p) => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" {...p}>
    <path d="M1 4h14M1 8h14M1 12h10" />
  </svg>
);
const IconTool: Icon = (p) => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" {...p}>
    <path d="M11 2l3 3-2 2-3-3zM10 5l-7 7v3h3l7-7" />
  </svg>
);
const IconPlug: Icon = (p) => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" {...p}>
    <path d="M5 1v4M11 1v4M3 5h10v4a4 4 0 01-4 4H7a4 4 0 01-4-4V5zM8 13v2" />
  </svg>
);
const IconDisk: Icon = (p) => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" {...p}>
    <ellipse cx="8" cy="4" rx="6" ry="2" />
    <path d="M2 4v8c0 1.1 2.7 2 6 2s6-.9 6-2V4M2 8c0 1.1 2.7 2 6 2s6-.9 6-2" />
  </svg>
);
const IconMcp: Icon = (p) => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" {...p}>
    <circle cx="8" cy="8" r="6" />
    <path d="M2 8h12M8 2v12" />
  </svg>
);

const CARDS = [
  { Icon: IconBox,    title: 'Agents',        body: 'One async loop, fully typed events.',     href: '/guides/agents/first-agent' },
  { Icon: IconStream, title: 'Streaming',     body: 'async for event in stream.',              href: '/guides/agents/streaming' },
  { Icon: IconTool,   title: 'Tools',         body: 'Plain functions, parallel execution.',    href: '/guides/agents/tool-use' },
  { Icon: IconPlug,   title: 'Providers',     body: 'Anthropic, OpenAI, or write your own.',   href: '/guides/providers/anthropic' },
  { Icon: IconDisk,   title: 'Checkpointing', body: 'Append-only, O(1) per turn.',             href: '/guides/checkpointing/sqlite' },
  { Icon: IconMcp,    title: 'MCP',           body: 'Load remote tools at startup.',           href: '/guides/mcp/loading' },
];

export default function FeatureGrid() {
  return (
    <section className={styles.section}>
      <div className={styles.grid}>
        {CARDS.map((c) => (
          <Link key={c.title} to={c.href} className={styles.card}>
            <c.Icon className={styles.icon} width={16} height={16} />
            <h3 className={styles.title}>{c.title}</h3>
            <p className={styles.body}>{c.body}</p>
            <span className={styles.more}>→ Guides / {c.title}</span>
          </Link>
        ))}
      </div>
    </section>
  );
}
```

- [ ] **Step 2: `FeatureGrid.module.css`**

```css
.section { max-width: 1080px; margin: 0 auto 64px; padding: 0 24px; }
.grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.card {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 18px 16px 14px;
  border: 1px solid var(--ink-5);
  border-radius: 6px;
  background: var(--surface);
  color: inherit;
  text-decoration: none;
  position: relative;
  min-height: 132px;
}
.card:hover { border-color: var(--ink-7); background: var(--ink-3); }
.icon { color: var(--ink-9); }
.title {
  font-family: 'Inter Tight';
  font-size: 15px;
  font-weight: 600;
  letter-spacing: -0.005em;
  color: var(--ink-12);
  margin: 4px 0 0;
}
.body { font-size: 13px; color: var(--ink-9); line-height: 1.5; margin: 0; }
.more {
  margin-top: auto;
  font-family: 'JetBrains Mono';
  font-size: 11px;
  color: var(--accent);
}
@media (max-width: 900px) { .grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 560px) { .grid { grid-template-columns: 1fr; } }
```

- [ ] **Step 3: Commit**

```bash
git add website/src/components/Home/FeatureGrid.tsx website/src/components/Home/FeatureGrid.module.css
git commit -m "feat(website): FeatureGrid with six monoline cards"
```

---

### Task 14: InstallMatrix component

**Files:**
- Create: `website/src/components/Home/InstallMatrix.tsx`
- Create: `website/src/components/Home/InstallMatrix.module.css`

- [ ] **Step 1: `InstallMatrix.tsx`**

```tsx
import React from 'react';
import styles from './InstallMatrix.module.css';

const ROWS: { tool: string; cmd: string }[] = [
  { tool: 'pip',    cmd: 'pip install cubepi' },
  { tool: 'uv',     cmd: 'uv add cubepi' },
  { tool: 'poetry', cmd: 'poetry add cubepi' },
  { tool: 'extras', cmd: 'pip install cubepi[sqlite,postgres,mcp]' },
];

export default function InstallMatrix() {
  return (
    <section className={styles.section}>
      <h2 className={styles.h2}>Install</h2>
      <div className={styles.table}>
        {ROWS.map((r) => (
          <div key={r.tool} className={styles.row}>
            <span className={styles.tool}>{r.tool}</span>
            <code className={styles.cmd}>{r.cmd}</code>
            <button
              type="button"
              className={styles.copy}
              onClick={() => navigator.clipboard?.writeText(r.cmd)}
              aria-label={`Copy ${r.tool} command`}
            >Copy</button>
          </div>
        ))}
      </div>
    </section>
  );
}
```

- [ ] **Step 2: `InstallMatrix.module.css`**

```css
.section { max-width: 1080px; margin: 0 auto 64px; padding: 0 24px; }
.h2 { font-family: 'Inter Tight'; font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 600; color: var(--ink-7); margin-bottom: 12px; }
.table { border: 1px solid var(--ink-5); border-radius: 8px; background: var(--surface); overflow: hidden; }
.row {
  display: grid;
  grid-template-columns: 96px 1fr auto;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  border-bottom: 1px solid var(--ink-5);
}
.row:last-child { border-bottom: 0; }
.tool { font-family: 'JetBrains Mono'; font-size: 12px; color: var(--ink-7); text-transform: uppercase; letter-spacing: 0.06em; }
.cmd { font-family: 'JetBrains Mono'; font-size: 13px; color: var(--ink-12); background: var(--ink-3); padding: 6px 10px; border: 1px solid var(--ink-5); border-radius: 5px; }
.copy { font-size: 11px; padding: 4px 10px; border: 1px solid var(--ink-5); background: var(--surface); color: var(--ink-9); border-radius: 4px; cursor: pointer; }
.copy:hover { background: var(--ink-3); color: var(--ink-11); }
```

- [ ] **Step 3: Commit**

```bash
git add website/src/components/Home/InstallMatrix.tsx website/src/components/Home/InstallMatrix.module.css
git commit -m "feat(website): InstallMatrix table"
```

---

### Task 15: MetaBar component

**Files:**
- Create: `website/src/components/Home/MetaBar.tsx`
- Create: `website/src/components/Home/MetaBar.module.css`

- [ ] **Step 1: `MetaBar.tsx`**

```tsx
import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './MetaBar.module.css';

export default function MetaBar() {
  const { siteConfig } = useDocusaurusContext();
  const sha = (siteConfig.customFields?.GIT_SHA as string | undefined) ?? 'dev';
  return (
    <section className={styles.bar}>
      <span>v0.3.0</span>
      <span className={styles.sep}>·</span>
      <span>py 3.11+</span>
      <span className={styles.sep}>·</span>
      <span>MIT</span>
      <span className={styles.sep}>·</span>
      <span>build {sha}</span>
      <span className={styles.sep}>·</span>
      <span className={styles.ok}>● ci passing</span>
      <span className={styles.sep}>·</span>
      <span>pypi · weekly downloads via shields badge</span>
    </section>
  );
}
```

- [ ] **Step 2: `MetaBar.module.css`**

```css
.bar {
  max-width: 1080px;
  margin: 0 auto;
  padding: 14px 24px;
  border-top: 1px solid var(--ink-5);
  font-family: 'JetBrains Mono';
  font-size: 11px;
  color: var(--ink-7);
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.sep { color: var(--ink-5); }
.ok { color: var(--ok); }
```

- [ ] **Step 3: Commit**

```bash
git add website/src/components/Home/MetaBar.tsx website/src/components/Home/MetaBar.module.css
git commit -m "feat(website): MetaBar status footer"
```

---

### Task 16: Wire the homepage

**Files:**
- Overwrite: `website/src/pages/index.tsx`

- [ ] **Step 1: Implementation**

```tsx
import React from 'react';
import Layout from '@theme/Layout';
import Hero from '@site/src/components/Home/Hero';
import WhyTable from '@site/src/components/Home/WhyTable';
import HelloAgent from '@site/src/components/Home/HelloAgent';
import FeatureGrid from '@site/src/components/Home/FeatureGrid';
import InstallMatrix from '@site/src/components/Home/InstallMatrix';
import MetaBar from '@site/src/components/Home/MetaBar';

export default function Home(): JSX.Element {
  return (
    <Layout title="CubePi — Pythonic async-native agent framework"
            description="CubePi is a Pythonic, async-native agent framework. Plain async functions, append-only checkpointing, 3 core dependencies.">
      <Hero />
      <WhyTable />
      <HelloAgent />
      <FeatureGrid />
      <InstallMatrix />
      <MetaBar />
    </Layout>
  );
}
```

- [ ] **Step 2: Build and visually inspect**

```bash
cd website && pnpm start --no-open
```

Open `http://localhost:3000`. Expected: full custom homepage in correct order. Resize to 600px wide → hero/feature grid collapse to single column.

- [ ] **Step 3: Commit**

```bash
git add website/src/pages/index.tsx
git commit -m "feat(website): assemble Operator-styled homepage"
```

---

## Phase 6 — PostHog + feedback widget

### Task 17: PostHog client module

**Files:**
- Create: `website/src/clientModules/posthog.ts`

- [ ] **Step 1: Add the SDK**

```bash
cd website && pnpm add posthog-js@1
```

- [ ] **Step 2: Write the module**

```ts
import posthog from 'posthog-js';
import siteConfig from '@generated/docusaurus.config';

const key   = (siteConfig.customFields?.POSTHOG_KEY  as string | undefined) ?? '';
const host  = (siteConfig.customFields?.POSTHOG_HOST as string | undefined) ?? 'https://us.i.posthog.com';

if (typeof window !== 'undefined' && key) {
  posthog.init(key, {
    api_host: host,
    capture_pageview: true,
    persistence: 'memory',
    autocapture: false,
    disable_session_recording: true,
  });
  (window as any).__cubepi_posthog = posthog;
}

export {};
```

- [ ] **Step 3: Verify the build still works without `POSTHOG_KEY` set**

```bash
unset POSTHOG_KEY
cd website && pnpm build
```

Expected: build succeeds, no console errors. (When `POSTHOG_KEY` is empty the init block is skipped.)

- [ ] **Step 4: Commit**

```bash
git add website/package.json website/pnpm-lock.yaml website/src/clientModules/posthog.ts
git commit -m "feat(website): PostHog client module gated on env"
```

---

### Task 18: DocFeedback component (TDD)

**Files:**
- Create: `website/src/components/DocFeedback/index.tsx`
- Create: `website/src/components/DocFeedback/styles.module.css`
- Create: `website/src/components/DocFeedback/index.test.tsx`
- Modify: `website/package.json` — devDependencies + scripts

- [ ] **Step 1: Add test tooling**

```bash
cd website && pnpm add -D vitest@1 @testing-library/react@16 @testing-library/jest-dom@6 jsdom@24
```

In `website/package.json` `"scripts"`, add:

```json
"test": "vitest run",
"test:watch": "vitest"
```

Create `website/vitest.config.ts`:

```ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'jsdom',
    setupFiles: ['./vitest.setup.ts'],
    globals: true,
  },
});
```

`website/vitest.setup.ts`:

```ts
import '@testing-library/jest-dom/vitest';
```

- [ ] **Step 2: Write the failing test**

`website/src/components/DocFeedback/index.test.tsx`:

```tsx
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';
import DocFeedback from './index';

declare global { interface Window { __cubepi_posthog?: { capture: (e: string, p: object) => void } } }

beforeEach(() => {
  window.__cubepi_posthog = { capture: vi.fn() };
});

describe('DocFeedback', () => {
  it('renders the prompt and two buttons', () => {
    render(<DocFeedback slug="/foo" version="0.3" locale="en" />);
    expect(screen.getByText(/was this page helpful/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /yes/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /no/i })).toBeInTheDocument();
  });

  it('captures a doc_feedback event with helpful=true on 👍', () => {
    const capture = vi.fn();
    window.__cubepi_posthog = { capture };
    render(<DocFeedback slug="/foo" version="0.3" locale="en" />);
    fireEvent.click(screen.getByRole('button', { name: /yes/i }));
    expect(capture).toHaveBeenCalledWith('doc_feedback', {
      slug: '/foo', helpful: true, version: '0.3', locale: 'en',
    });
    expect(screen.getByText(/thanks/i)).toBeInTheDocument();
  });

  it('shows a comment textarea on 👎 and captures doc_feedback_comment on submit', () => {
    const capture = vi.fn();
    window.__cubepi_posthog = { capture };
    render(<DocFeedback slug="/foo" version="0.3" locale="en" />);
    fireEvent.click(screen.getByRole('button', { name: /no/i }));
    expect(capture).toHaveBeenCalledWith('doc_feedback', {
      slug: '/foo', helpful: false, version: '0.3', locale: 'en',
    });
    const ta = screen.getByRole('textbox');
    fireEvent.change(ta, { target: { value: 'unclear example' } });
    fireEvent.click(screen.getByRole('button', { name: /submit/i }));
    expect(capture).toHaveBeenCalledWith('doc_feedback_comment', {
      slug: '/foo', version: '0.3', locale: 'en', comment: 'unclear example',
    });
  });
});
```

Run:

```bash
cd website && pnpm test
```

Expected: all 3 tests fail with "Cannot find module './index'".

- [ ] **Step 3: Implement the component**

`website/src/components/DocFeedback/index.tsx`:

```tsx
import React, { useState } from 'react';
import styles from './styles.module.css';

interface Props {
  slug: string;
  version: string;
  locale: string;
}

type Phase = 'ask' | 'thanks' | 'comment' | 'submitted';

function capture(event: string, payload: object) {
  const ph = (window as any).__cubepi_posthog;
  if (ph && typeof ph.capture === 'function') ph.capture(event, payload);
}

export default function DocFeedback({ slug, version, locale }: Props): JSX.Element {
  const [phase, setPhase] = useState<Phase>('ask');
  const [comment, setComment] = useState('');

  const onYes = () => {
    capture('doc_feedback', { slug, helpful: true, version, locale });
    setPhase('thanks');
  };
  const onNo = () => {
    capture('doc_feedback', { slug, helpful: false, version, locale });
    setPhase('comment');
  };
  const onSubmit = () => {
    capture('doc_feedback_comment', { slug, version, locale, comment });
    setPhase('submitted');
  };

  return (
    <aside className={styles.box}>
      {phase === 'ask' && (
        <>
          <span className={styles.q}>Was this page helpful?</span>
          <button type="button" className={styles.btn} onClick={onYes} aria-label="Yes">👍</button>
          <button type="button" className={styles.btn} onClick={onNo}  aria-label="No">👎</button>
        </>
      )}
      {phase === 'thanks' && <span className={styles.q}>Thanks!</span>}
      {phase === 'comment' && (
        <div className={styles.commentWrap}>
          <span className={styles.q}>What was missing?</span>
          <textarea
            className={styles.textarea}
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            rows={3}
          />
          <button type="button" className={styles.btn} onClick={onSubmit}>Submit</button>
        </div>
      )}
      {phase === 'submitted' && <span className={styles.q}>Thanks — we'll review it.</span>}
    </aside>
  );
}
```

`website/src/components/DocFeedback/styles.module.css`:

```css
.box {
  margin: 32px 0 0;
  padding: 14px 16px;
  border: 1px solid var(--ink-5);
  border-radius: 6px;
  background: var(--surface);
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 13px;
  color: var(--ink-9);
}
.q { color: var(--ink-11); font-weight: 500; }
.btn {
  font: inherit;
  background: var(--surface);
  border: 1px solid var(--ink-5);
  border-radius: 5px;
  padding: 4px 10px;
  cursor: pointer;
  color: var(--ink-11);
}
.btn:hover { background: var(--ink-3); border-color: var(--ink-7); }
.commentWrap { display: flex; flex-direction: column; gap: 8px; width: 100%; }
.textarea {
  font: inherit;
  font-family: 'Inter';
  padding: 8px 10px;
  border: 1px solid var(--ink-5);
  border-radius: 5px;
  background: var(--surface);
  color: var(--ink-12);
  resize: vertical;
}
.textarea:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(59,91,217,0.12); }
```

- [ ] **Step 4: Run tests, verify pass**

```bash
cd website && pnpm test
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add website/package.json website/pnpm-lock.yaml website/vitest.config.ts website/vitest.setup.ts website/src/components/DocFeedback/
git commit -m "feat(website): DocFeedback widget with PostHog capture and tests"
```

---

### Task 19: Swizzle `DocItem/Footer` to mount DocFeedback

**Files:**
- Create: `website/src/theme/DocItem/Footer/index.tsx`

- [ ] **Step 1: Eject the default footer to inspect it**

```bash
cd website && pnpm swizzle @docusaurus/theme-classic DocItem/Footer --wrap --typescript --danger
```

Pick "Yes" at any prompts. This creates `src/theme/DocItem/Footer/index.tsx` wrapping the upstream component.

- [ ] **Step 2: Replace the wrapper**

`website/src/theme/DocItem/Footer/index.tsx`:

```tsx
import React from 'react';
import Footer from '@theme-original/DocItem/Footer';
import { useDoc } from '@docusaurus/plugin-content-docs/client';
import { useLocation } from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import DocFeedback from '@site/src/components/DocFeedback';

export default function FooterWrapper(props: any): JSX.Element {
  const { metadata } = useDoc();
  const { i18n } = useDocusaurusContext();
  const { pathname } = useLocation();
  const version = (metadata as any).version ?? 'current';
  return (
    <>
      <DocFeedback slug={pathname} version={version} locale={i18n.currentLocale} />
      <Footer {...props} />
    </>
  );
}
```

- [ ] **Step 3: Build and verify**

```bash
cd website && pnpm build
```

Open the built site (`pnpm serve`) and navigate to `/getting-started/installation`. Expected: a 👍/👎 box appears above the prev/next footer.

- [ ] **Step 4: Commit**

```bash
git add website/src/theme/DocItem/Footer/
git commit -m "feat(website): swizzle DocItem/Footer to mount DocFeedback"
```

---

## Phase 7 — i18n

### Task 20: Enable Simplified Chinese locale with core translations

**Files:**
- Create (≥ 6): translation files under `website/i18n/zh-Hans/`

- [ ] **Step 1: Scaffold translation directories**

```bash
mkdir -p \
  website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/getting-started \
  website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/guides/agents \
  website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/guides/providers \
  website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/guides/checkpointing \
  website/i18n/zh-Hans/docusaurus-theme-classic
```

- [ ] **Step 2: Generate UI string templates**

```bash
cd website && pnpm write-translations --locale zh-Hans
```

This creates `i18n/zh-Hans/code.json` and theme JSON files. Hand-edit:

- `code.json` — translate string values, leave keys.
- `docusaurus-theme-classic/navbar.json` — translate `Docs`, `API`, `Recipes`, `GitHub`.
- `docusaurus-theme-classic/footer.json` — translate any footer labels.

- [ ] **Step 3: Copy and translate the 6 core docs**

For each of these pages, copy the English markdown to the parallel zh-Hans path and translate the prose. Leave code blocks unchanged. Six core pages:

| English source | zh-Hans destination |
|---|---|
| `docs/intro.md` | `i18n/zh-Hans/docusaurus-plugin-content-docs/current/intro.md` |
| `docs/getting-started/installation.md` | `i18n/zh-Hans/.../current/getting-started/installation.md` |
| `docs/getting-started/quick-start.md` | `i18n/zh-Hans/.../current/getting-started/quick-start.md` |
| `docs/getting-started/core-concepts.md` | `i18n/zh-Hans/.../current/getting-started/core-concepts.md` |
| `docs/guides/agents/first-agent.md` | `i18n/zh-Hans/.../current/guides/agents/first-agent.md` |
| `docs/guides/agents/streaming.md` | `i18n/zh-Hans/.../current/guides/agents/streaming.md` |
| `docs/guides/agents/tool-use.md` | `i18n/zh-Hans/.../current/guides/agents/tool-use.md` |
| `docs/guides/providers/anthropic.md` | `i18n/zh-Hans/.../current/guides/providers/anthropic.md` |
| `docs/guides/checkpointing/sqlite.md` | `i18n/zh-Hans/.../current/guides/checkpointing/sqlite.md` |

Because the English content is still placeholder, translate each placeholder paragraph to a Chinese placeholder (`_占位。内容将在后续内容 PR 中补充。_`). The real translation will follow the real English content in subsequent PRs.

- [ ] **Step 4: Verify the locale build**

```bash
cd website && pnpm build -- --locale zh-Hans
```

Expected: build succeeds; navbar Locale dropdown is visible on the built site at `/zh-Hans/`.

- [ ] **Step 5: Commit**

```bash
git add website/i18n/zh-Hans/
git commit -m "feat(website): enable zh-Hans locale with stubbed core translations"
```

---

## Phase 8 — Versioning

### Task 21: Snapshot version 0.3 and set as default

**Files:**
- Create (auto): `website/versioned_docs/version-0.3/`, `website/versioned_sidebars/version-0.3-sidebars.json`, `website/versions.json`
- Modify: `website/docusaurus.config.ts:lastVersion` and `versions`

- [ ] **Step 1: Run the snapshot**

```bash
cd website && pnpm docusaurus docs:version 0.3
```

Expected: creates `versioned_docs/version-0.3/`, `versioned_sidebars/version-0.3-sidebars.json`, and updates `versions.json` to `["0.3"]`.

- [ ] **Step 2: Re-route URLs**

Edit `website/docusaurus.config.ts` so the `presets.classic.docs` block reads:

```ts
docs: {
  sidebarPath: './sidebars.ts',
  editUrl: 'https://github.com/cubeplexai/cubepi/edit/main/website/',
  lastVersion: '0.3',
  versions: {
    current: { label: 'Next 🚧', path: 'next', banner: 'unreleased' },
    '0.3':   { label: '0.3 (latest)', path: '' },
  },
},
```

- [ ] **Step 3: Mirror the zh-Hans translation tree**

```bash
mkdir -p website/i18n/zh-Hans/docusaurus-plugin-content-docs/version-0.3
cp -r website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/* \
      website/i18n/zh-Hans/docusaurus-plugin-content-docs/version-0.3/
```

- [ ] **Step 4: Build and verify URLs**

```bash
cd website && pnpm build
```

Expected: `build/` contains both `/index.html` (for 0.3 docs) and `/next/...` (for next). Version dropdown in navbar shows `0.3 (latest)` and `Next 🚧`.

- [ ] **Step 5: Commit**

```bash
git add website/versioned_docs/ website/versioned_sidebars/ website/versions.json website/docusaurus.config.ts website/i18n/zh-Hans/docusaurus-plugin-content-docs/version-0.3/
git commit -m "feat(website): snapshot 0.3 as default; next routed to /next/"
```

---

## Phase 9 — CI

### Task 22: GitHub Actions workflow

**Files:**
- Create: `.github/workflows/docs.yml`

- [ ] **Step 1: Workflow content**

```yaml
name: docs

on:
  push:
    branches: [main]
    paths:
      - 'website/**'
      - 'cubepi/**'
      - 'pyproject.toml'
      - '.github/workflows/docs.yml'
  pull_request:
    paths:
      - 'website/**'
      - 'cubepi/**'
      - 'pyproject.toml'
      - '.github/workflows/docs.yml'

permissions:
  contents: read
  deployments: write

concurrency:
  group: docs-${{ github.ref }}
  cancel-in-progress: true

jobs:
  build-and-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
        with: { version: 9 }
      - uses: actions/setup-node@v4
        with:
          node-version: 22
          cache: 'pnpm'
          cache-dependency-path: website/pnpm-lock.yaml
      - uses: astral-sh/setup-uv@v3
        with: { python-version: '3.11' }
      - name: Install Python deps
        run: uv sync --frozen --extra docs
      - name: Install website deps
        working-directory: website
        run: pnpm install --frozen-lockfile
      - name: Guard against hand-edited API mdx
        run: |
          if find website/docs/api -type f -name 'cubepi-*.mdx' 2>/dev/null | grep .; then
            echo "::error::Generated API MDX files were committed. Delete them." && exit 1
          fi
      - name: Build (runs griffe via prebuild)
        working-directory: website
        env:
          POSTHOG_KEY: ${{ secrets.POSTHOG_KEY }}
          POSTHOG_HOST: ${{ secrets.POSTHOG_HOST }}
        run: pnpm build
      - name: Run vitest suite
        working-directory: website
        run: pnpm test
      - name: Upload artefact
        if: github.ref == 'refs/heads/main'
        uses: actions/upload-artifact@v4
        with:
          name: website-build
          path: website/build/
          if-no-files-found: error

  deploy-cloudflare:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    needs: build-and-check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          name: website-build
          path: website/build/
      - name: Deploy to Cloudflare Pages
        uses: cloudflare/pages-action@v1
        with:
          apiToken: ${{ secrets.CF_API_TOKEN }}
          accountId: ${{ secrets.CF_ACCOUNT_ID }}
          projectName: cubepi
          directory: website/build
          gitHubToken: ${{ secrets.GITHUB_TOKEN }}
          branch: main
```

- [ ] **Step 2: Local dry-run**

```bash
cd website && pnpm build && pnpm test
```

Expected: both pass.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/docs.yml
git commit -m "ci: build, test, and deploy docs to Cloudflare Pages"
```

- [ ] **Step 4: Configure repository secrets**

Manually (cannot be automated):

1. In Cloudflare dashboard → Pages → create a project named `cubepi`, "Direct upload" mode.
2. Cloudflare → My Profile → API Tokens → create token with `Cloudflare Pages:Edit`.
3. GitHub repo → Settings → Secrets and variables → Actions, add:
   - `CF_API_TOKEN`
   - `CF_ACCOUNT_ID`
   - `POSTHOG_KEY`
   - `POSTHOG_HOST` (optional; defaults to US endpoint)

Record completion of this manual step in the PR description.

---

## Phase 10 — README & links

### Task 23: README & link out

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a docs banner near the top**

After the badges block in `README.md`, insert:

```markdown
**Docs:** https://cubepi.pages.dev — Getting Started · API Reference · Recipes
```

- [ ] **Step 2: Verify image paths**

```bash
grep -n 'assets/brand' README.md
```

Expected: no output. Already fixed by Task 1, this is a sanity check.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: link README to https://cubepi.pages.dev"
```

---

## Phase 11 — Open PR

### Task 24: Push the branch and open a PR

- [ ] **Step 1: Push**

```bash
git push -u origin <current-branch>
```

- [ ] **Step 2: Open the PR**

```bash
gh pr create --title "Add CubePi documentation site (Docusaurus 3, EN+zh-Hans, PostHog feedback, CF Pages)" --body "$(cat <<'EOF'
## Summary
- New `website/` Docusaurus 3 site with Operator-styled custom homepage
- EN default + zh-Hans (core 6–8 pages stubbed) with locale + version dropdowns
- API reference auto-generated from docstrings via griffe (prebuild)
- 👍/👎 footer widget capturing `doc_feedback` / `doc_feedback_comment` to PostHog
- CI: build + test + deploy to Cloudflare Pages at `cubepi.pages.dev`
- First snapshot `0.3` is the default; `Next 🚧` at `/next/`

Spec: `docs/specs/2026-05-15-cubepi-docs-site-design.md`
Plan: `docs/plans/2026-05-15-cubepi-docs-site.md`

## Test plan
- [ ] `pnpm install` + `uv sync --extra docs` succeeds on a fresh clone
- [ ] `pnpm dev` (== `pnpm start`) serves the site at http://localhost:3000
- [ ] `pnpm build` succeeds locally
- [ ] `pnpm test` passes (3 DocFeedback + 3 build_api_reference)
- [ ] Click 👍 / 👎 on any page; verify `doc_feedback` event lands in PostHog
- [ ] Switch locale to 简体中文; verify zh-Hans pages render and API pages show fallback
- [ ] Switch version to Next 🚧; verify URL is `/next/...`
- [ ] CI green; preview URL deployable from CF Pages dashboard
EOF
)"
```

---

## Out of scope (explicit non-tasks)

- Translating full English content into zh-Hans (stubs only; content PRs follow).
- Replacing placeholder page content with real prose (separate content PR).
- Algolia DocSearch application — relies on the site being live first.
- Custom domain binding — deferred until post-1.0.
- Lighthouse perf tuning — add as follow-up issue.

---

## Pre-flight assumptions

The first executor of this plan should verify these are true before starting Task 1:

1. `pnpm dlx create-docusaurus` works locally (Node 22 installed).
2. `uv` is installed and `uv run python -V` prints 3.11+.
3. The `cubepi` package imports cleanly from a fresh `uv sync` — required for griffe.
4. `gh` CLI is authenticated for Task 24.

If any of these are not true, stop and surface the gap before starting.
