# CubePi Documentation Site — Design Spec

**Date:** 2026-05-15
**Author:** xf gong (gxf.alpha@gmail.com)
**Status:** Draft — awaiting review

## 1. Goal & Scope

Stand up the first public documentation site for **CubePi**, a Pythonic
async-native agent framework currently at `v0.3.0` alpha on PyPI.

The site is a **pure developer documentation site** (no marketing landing
page). Its single job is to make CubePi discoverable, learnable, and
referenceable — for users evaluating it against `langgraph`, for users
writing their first agent, and for users debugging a specific API.

Three hard requirements drive the framework choice:

- **Versioning**: snapshot every minor release (`0.3`, `0.4`, …)
- **i18n**: English (default) + Simplified Chinese (`zh-Hans`)
- **Feedback**: page-level 👍/👎 with optional comment, signal goes to PostHog

Out of scope: paid SaaS tooling (Mintlify / GitBook), API playground /
sandbox, blog, changelog page (handled by GitHub Releases).

## 2. Framework Selection

**Choice: Docusaurus 3.x** (latest), with TypeScript config.

Rationale — versioning is the deciding factor. Of all open-source doc
frameworks evaluated (VitePress, Nextra, Starlight, MkDocs Material,
Docusaurus), **Docusaurus is the only one with versioning as a core,
first-class command** (`docusaurus docs:version`). Starlight has community
plugin `starlight-versions` but it lags core releases and is single-maintainer
risk. VitePress and Nextra require manual directory copying for each
snapshot. MkDocs needs `mike` plus Insiders for i18n.

Docusaurus also has i18n built into core, mature React-based theming
(needed for the custom homepage), and battle-tested by Babel / Redux /
Jest / Supabase / Prisma.

The known downside — slower builds than Vite-based alternatives — is
acceptable for a docs site that builds on CI, not on every keystroke.

## 3. Tech Stack

| Layer | Choice |
|---|---|
| Framework | Docusaurus 3.x |
| Config | TypeScript (`docusaurus.config.ts`) |
| Node | 22 LTS |
| Package manager | pnpm 9 |
| API reference generator | [griffe](https://github.com/mkdocstrings/griffe) (Python) |
| Hosting | Cloudflare Pages |
| Analytics & feedback | PostHog (self-hosted-cloud) |
| Search | Algolia DocSearch (free tier for OSS) |
| CI | GitHub Actions |
| Default domain | `cubepi.ai` (custom domain TBD) |

## 4. Repository Layout

The docs site lives under a new top-level `website/` directory, peer to
the Python package `cubepi/`. Brand assets move from `assets/brand/` to
`website/static/img/brand/` so there is a single source of truth.

```
cubepi/
├── cubepi/                          # Python package — unchanged
├── docs/                            # specs / plans — unchanged
├── website/                         # ← entire docs site
│   ├── docusaurus.config.ts
│   ├── sidebars.ts
│   ├── package.json
│   ├── pnpm-lock.yaml
│   ├── tsconfig.json
│   ├── .gitignore                   # ignores build/ and docs/api/*.mdx
│   │
│   ├── docs/                        # next-version English content
│   │   ├── intro.md
│   │   ├── getting-started/
│   │   ├── guides/
│   │   ├── api/                     # griffe-generated, gitignored
│   │   │   └── _index.mdx           # hand-written overview, tracked
│   │   ├── recipes/
│   │   └── migration/
│   │
│   ├── versioned_docs/
│   │   └── version-0.3/             # first snapshot
│   ├── versioned_sidebars/
│   │   └── version-0.3-sidebars.json
│   ├── versions.json                # ["0.3"]
│   │
│   ├── i18n/
│   │   └── zh-Hans/
│   │       ├── code.json
│   │       └── docusaurus-plugin-content-docs/
│   │           ├── current/         # next-version zh-Hans
│   │           └── version-0.3/     # 0.3 zh-Hans
│   │
│   ├── src/
│   │   ├── clientModules/
│   │   │   └── posthog.ts
│   │   ├── components/
│   │   │   ├── DocFeedback/
│   │   │   └── Home/
│   │   │       ├── Hero.tsx
│   │   │       ├── WhyTable.tsx
│   │   │       ├── HelloAgent.tsx
│   │   │       ├── FeatureGrid.tsx
│   │   │       ├── InstallMatrix.tsx
│   │   │       └── MetaBar.tsx
│   │   ├── theme/
│   │   │   └── DocItem/Footer/index.tsx   # swizzled — mounts DocFeedback
│   │   ├── css/
│   │   │   └── custom.css           # Operator design tokens
│   │   └── pages/
│   │       └── index.tsx            # custom homepage
│   │
│   ├── static/
│   │   ├── img/
│   │   │   └── brand/               # ← migrated from /assets/brand/
│   │   │       ├── cubepi-logo.svg
│   │   │       ├── cubepi-logo.png
│   │   │       ├── cubepi-social-preview.svg
│   │   │       └── cubepi-social-preview.png
│   │   ├── fonts/                   # self-hosted Inter / Inter Tight / JetBrains Mono
│   │   └── llms.txt
│   │
│   └── scripts/
│       └── build-api-reference.py   # griffe → MDX
│
├── .github/
│   └── workflows/
│       └── docs.yml                 # new
│
├── pyproject.toml                   # adds [project.optional-dependencies].docs
└── README.md                        # updated image paths
```

Conventions:

- No root-level `package.json` — keeps CubePi unambiguously a Python project.
- Generated API MDX files are gitignored; the build runs `griffe` every
  time. CI failures point at the source code, not at stale MDX.
- The `docs` Python extra in `pyproject.toml` carries `griffe` and any
  other build-time helpers.

## 5. Information Architecture

Top-level navigation has five sections. The sidebar tree:

```
Getting Started
  ├─ Installation
  ├─ Quick Start (5-min agent)
  └─ Core Concepts (Agent / Tool / Provider / Stream)

Guides
  ├─ Agents
  │   ├─ Building Your First Agent
  │   ├─ Tool Use & Parallel Execution
  │   ├─ Multi-turn Conversations
  │   └─ Streaming Events
  ├─ Providers
  │   ├─ Anthropic
  │   ├─ OpenAI
  │   └─ Writing a Custom Provider
  ├─ Checkpointing
  │   ├─ SQLite
  │   ├─ Postgres
  │   └─ Custom Backends
  ├─ Middleware
  │   ├─ The 5 Hooks
  │   ├─ Composition Rules
  │   └─ Examples (rate limit / logging / retries)
  └─ MCP
      ├─ Loading MCP Tools
      └─ Server Authentication

API Reference  (griffe-generated)
  ├─ cubepi.agent
  ├─ cubepi.providers
  ├─ cubepi.checkpointer
  ├─ cubepi.middleware
  ├─ cubepi.mcp
  └─ cubepi.utils

Recipes
  ├─ Weather Agent (tool use)
  ├─ Multi-Provider Failover
  ├─ Persistent Chat (SQLite)
  ├─ Resumable Long Tasks
  └─ Postgres + FastAPI Service

Migration
  └─ From langgraph
```

zh-Hans sidebar is structurally identical (UI strings translated); see
§7 for content translation scope.

## 6. Versioning Strategy

**Snapshot every minor.** First snapshot is `0.3` at site launch.

- `next` (under development): driven by `website/docs/`, served at `/next/...`.
- Default published version: `0.3`, served at `/` (no version prefix).
- Each new minor release runs `pnpm docusaurus docs:version <X.Y>` in the
  release PR, copying the current `docs/` to `versioned_docs/version-X.Y/`
  and freezing it.
- `versions.json` keeps the array of snapshots. When it grows past 3
  active versions, the oldest is labelled `unmaintained` (still
  reachable, but flagged in UI).

Config:

```ts
presets: [['classic', {
  docs: {
    lastVersion: '0.3',
    versions: {
      current: { label: 'Next 🚧', path: 'next', banner: 'unreleased' },
      '0.3':   { label: '0.3 (latest)', path: '' },
    },
  },
}]]
```

A **version switcher** sits in the top-right navbar.

## 7. Internationalisation Strategy

- `defaultLocale: 'en'`, `locales: ['en', 'zh-Hans']`.
- Translation lives at
  `website/i18n/zh-Hans/docusaurus-plugin-content-docs/{current,version-0.3}/`.
- **Initial Chinese coverage**: Getting Started (3 pages) + Guides core
  (Building Your First Agent, Tool Use, Streaming, Anthropic provider,
  SQLite checkpointing) = **6–8 pages**, not the full site.
- **API Reference is English-only**. When a Chinese reader navigates to
  an API page, Docusaurus's built-in `<TranslationNotice>` reads
  "This page is only available in English." with a link to the
  authoritative English version.
- Translation workflow: manual edits to the mirrored Markdown files. No
  Crowdin / Lokalise integration — solo maintainer, SaaS overhead not
  justified.

A **language switcher** sits in the top-right navbar, adjacent to the
version switcher.

## 8. Feedback Mechanism

**👍 / 👎 footer widget**, signal goes to PostHog.

### PostHog setup

- Initialised in `website/src/clientModules/posthog.ts`:
  ```ts
  posthog.init(process.env.POSTHOG_KEY!, {
    api_host: 'https://us.i.posthog.com',
    capture_pageview: true,
    persistence: 'memory',   // no cookie → no GDPR banner
  });
  ```
- `POSTHOG_KEY` is set in Cloudflare Pages environment variables and
  exposed at build time via `customFields` in `docusaurus.config.ts`.

### `<DocFeedback />` component

- Mounted by swizzling `theme/DocItem/Footer`.
- Visual: 1px ink-5 border, 6px radius, no shadow.
  ```
  Was this page helpful?   [ 👍 ]   [ 👎 ]
  ```
- On click:
  - 👍 → `posthog.capture('doc_feedback', { slug, helpful: true, version, locale })`,
    button label changes to "Thanks!".
  - 👎 → same event with `helpful: false`, then reveals a textarea +
    Submit button; submitting fires `doc_feedback_comment` with
    `comment` and the same context fields.
- **No anti-spam / dedup** — every click captures an event. Operator
  decision: low-stakes signal, dedup would reduce signal more than it
  prevents abuse.

### Dashboards (PostHog side, configured manually after launch)

- Insight 1: `helpful%` per page slug, sorted ascending → worst pages
  first → editorial backlog.
- Insight 2: `helpful%` broken down by `version` → regression detection.
- Insight 3: comment stream, weekly review.

## 9. Custom Homepage

### Design language: Operator

The homepage adopts the **Direction A · Operator** design language
(reference: `/home/chris/cubebox/_design-explorations/direction-a-operator/DESIGN.md`).
Operator is a Linear / Vercel / Raycast-style "information architecture"
discipline: 1px hairlines, no shadows, no purple gradients, no emoji,
tabular numerals on every numeric span, single accent colour, kbd hints
as first-class citizens.

**Type system**: Inter Tight (display, -1.5% letter-spacing), Inter
(body, 14/20), JetBrains Mono (numerals & code, `font-variant-numeric:
tabular-nums`). Self-hosted from `static/fonts/`, never Google Fonts.

**Colour system**: 9 ink shades (`--ink-1` … `--ink-12`) + single accent
`#3B5BD9` + `ok / warn / err` (used sparingly).

### Homepage section layout (top-to-bottom, single column)

```
┌─────────────────────────────────────────────────────────────┐
│ navbar  (Docusaurus default, themed with Operator tokens)   │
├─────────────────────────────────────────────────────────────┤
│ HERO                                                        │
│   • Banner image: cubepi-social-preview.png                 │
│     - max-width 880px, centred                              │
│     - 1px ink-5 border, 6px radius, no shadow               │
│   • Eyebrow:  cubepi · v0.3.0 · alpha   (JetBrains Mono)    │
│   • H1:       A Pythonic, async-native agent framework.     │
│   • Lead:     Plain async functions instead of graph nodes. │
│               3 deps. Append-only checkpointing.            │
│   • Actions:  [ pip install cubepi  ⌘C ]                    │
│               [ Quick Start →   G Q ]                       │
│   (no "CubePi" wordmark below image — already baked in)     │
├─────────────────────────────────────────────────────────────┤
│ WHY CUBEPI                                                  │
│   • 9-row × 3-col comparison table (cubepi vs langgraph)    │
│   • Hairline rules, no shadow, JetBrains Mono in cells     │
│   • Direct port of the table from README.md                 │
├─────────────────────────────────────────────────────────────┤
│ HELLO, AGENT.                                               │
│   • Left: H2 + one-sentence tagline                         │
│   • Right: 10-line code block, JetBrains Mono, ink-3 bg     │
│   • Bottom link: "Full quick-start → /getting-started"      │
├─────────────────────────────────────────────────────────────┤
│ FEATURE GRID (3 × 2, six cards)                             │
│   • Agents · Streaming · Tools · Providers ·                │
│     Checkpointing · MCP                                     │
│   • Per card: 16px Lucide stroke-1.5 monoline icon,         │
│     Inter Tight title, ≤ 22-char ink-9 description,         │
│     "→ Guides / <topic>" link bottom-right                  │
├─────────────────────────────────────────────────────────────┤
│ INSTALL MATRIX                                              │
│   • 4-column table: pip / uv / poetry / extras              │
│   • Each row: equal-width code, copy button to the right    │
├─────────────────────────────────────────────────────────────┤
│ META STATUS BAR                                             │
│   • JetBrains Mono 11px, ink-7, 1px ink-5 top border        │
│   • Shows: v0.3.0 · py 3.11+ · MIT · build a1b2c3d ·        │
│            pypi:weekly 1.2k · ci:passing · coverage:91%     │
├─────────────────────────────────────────────────────────────┤
│ Docusaurus default footer (links, copyright)                │
└─────────────────────────────────────────────────────────────┘
```

### Responsive

Below 768px:

- Hero stays single-column; the social-preview banner stays at the top.
- Feature grid collapses to 1 column.
- WHY CUBEPI table becomes horizontally scrollable inside its own
  container (don't reflow rows — readers compare cells).

### Justified divergences from Operator

| Aspect | Operator says | Homepage does | Justification |
|---|---|---|---|
| Hero exists | "marketing distance not needed" | Keeps a restrained hero | Without one, a docs landing page is just a link list; the hero is severe and informational, no animations |
| No sidebar / inspector / statusbar chrome | Identity feature of Operator | Homepage uses a single-column layout, only carries the meta status bar at the bottom | A homepage is not an app shell; copying chrome would be cargo cult |
| kbd hints sparing | First-class citizen | Only two `<kbd>` chips, on the two hero CTAs | docs site is not an IDE; heavy kbd hints would feel performative |

### ⚠️ Outstanding review reference

`memory/feedback_design_review.md` requires comparing cubepi designs
against `pi-agent-core`. `pi-agent-core` was not present on this
machine when the spec was written. Once the repo is available, the
homepage will be re-reviewed against it and divergences documented in
the implementation plan.

## 10. API Reference Generation Pipeline

**Tool: [griffe](https://github.com/mkdocstrings/griffe)** — the engine
behind `mkdocstrings`. Pure Python, parses signatures, type annotations,
decorators, and Google-style docstrings.

### Script: `website/scripts/build-api-reference.py`

Roughly 150 lines. Flow:

1. `griffe.load("cubepi")` → full module tree.
2. Filter to public symbols (in `__all__` or without leading underscore).
3. For each top-level module (`agent`, `providers`, `checkpointer`,
   `middleware`, `mcp`, `utils`), emit one `<module>.mdx`.
4. Per symbol, render:
   - Signature block with type annotations (JetBrains Mono).
   - Docstring description (Google-style sections parsed: `Args`,
     `Returns`, `Raises`, `Example` → mdx-native tables / code blocks).
   - Source link → GitHub blob URL at the current commit, anchored to
     the line.
5. Inject frontmatter (`id`, `title`, `sidebar_position`).
6. Trailing `<!-- GENERATED by build-api-reference.py — DO NOT EDIT -->`
   marker. CI grep step rejects PRs that hand-edit these files.

### Integration

- `pnpm build` runs `"prebuild": "uv run python scripts/build-api-reference.py"`.
- CubePi is `uv sync --extra docs`-installed in CI before docs build.

### Versioning

- When `docs:version 0.4` runs, the latest generated API MDX is copied
  into `versioned_docs/version-0.4/api/` and frozen forever (so the API
  reference for `0.3` continues to reflect `0.3`'s code, not main).

### Docstring contract

- Google-style (existing convention in the codebase).
- All public API has docstrings; `ruff` rule `D` (pydocstyle) enforced
  in CI on the `cubepi/` package.

## 11. CI / Deployment / PR Preview

### `.github/workflows/docs.yml`

Two jobs:

**`build-and-check`** (PR + push):

- `actions/checkout@v4`
- `pnpm/action-setup@v4` (pnpm 9)
- `actions/setup-node@v4` (Node 22)
- `astral-sh/setup-uv@v3`
- `uv python install 3.11 && uv sync --frozen --extra docs`
- `cd website && pnpm install --frozen-lockfile`
- `pnpm build` (runs griffe prebuild → docusaurus build)
- `pnpm run check`:
  - Docusaurus `onBrokenLinks: 'throw'` + `onBrokenAnchors: 'throw'`
  - `cspell` over `docs/` and `i18n/` with a project dictionary
    (entries: `cubepi`, `langgraph`, `anthropic`, `mcp`, etc.)
  - Grep guard: reject hand-edits to `docs/api/*.mdx`

**`deploy-cloudflare`** (push to `main` only):

- Reuses the artefact from `build-and-check`.
- Uses `cloudflare/pages-action@v1` with a Wrangler API token
  (`CF_API_TOKEN`, stored in GitHub Secrets) and account ID.

### PR preview

Cloudflare Pages provides preview URLs natively (`<sha>.cubepi.ai`).
`cloudflare/pages-action@v1` posts a sticky comment with the preview
link on every PR.

### Domain

Launch domain: `cubepi.ai`. Custom domain (e.g. `cubepi.dev`,
`docs.cubepi.dev`) is deferred — site config supports late binding via
`url` / `baseUrl` swap.

## 12. Performance & Compliance Notes

- All fonts self-hosted (no Google Fonts request) → faster LCP, GDPR-clean.
- PostHog initialised with `persistence: 'memory'` → no cookies →
  no consent banner needed in EU.
- Docusaurus produces fully static output → first byte from CF edge.
- Target Lighthouse scores at launch: ≥ 95 on Performance, Accessibility,
  Best Practices, SEO for the homepage on mid-tier mobile profile.

## 13. Open Items (to resolve during implementation)

1. **`pi-agent-core` design parity review** (see §9) — needs repo
   access.
2. **Algolia DocSearch application** — apply for the OSS free tier;
   takes 1–2 weeks. Until approved, use Docusaurus's built-in local
   search plugin.
3. **PostHog project provisioning** — choose project (US vs EU
   endpoint).
4. **README.md updates** — image references must point at the new
   `website/static/img/brand/` path; add a single line linking to
   `cubepi.ai` at the top.

## 14. Out of Scope (explicitly deferred)

- Blog / changelog (use GitHub Releases).
- API playground / interactive sandbox.
- Authenticated content / paid tiers.
- Multi-repo doc aggregation.
- Crowdin / SaaS translation workflow.
- Custom domain binding (deferred until product is post-1.0).

## 15. Success Criteria

- `pnpm dev` runs locally on a fresh clone with one command after
  `pnpm install` + `uv sync --extra docs`.
- `pnpm build` produces a deployable artefact on CI in under 4 minutes.
- A new minor release can be snapshotted with a single command
  (`pnpm docusaurus docs:version <X.Y>`).
- 👍 / 👎 events are visible in PostHog within 30 seconds of clicking.
- Every page is reachable via the navbar in ≤ 2 clicks from the
  homepage.
- Lighthouse ≥ 95 on the four core categories for `/` and a representative
  Guides page on mobile.
