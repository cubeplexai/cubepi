# Changelog

All notable changes to CubePi are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2026-05-25

### Added

- **Image generation subsystem** (`cubepi.providers.images`): a pluggable,
  per-vendor model interface for image generation, with a class factory so new
  backends slot in without touching call sites.
- **`CapabilityDescriptor`**: a declarative, per-model description of what a
  model supports — temperature mode (free / fixed / ignored), reasoning-level
  mapping (int budget / effort / enum), and the `max_tokens` field name. It now
  drives the OpenAI, OpenAI Responses, Anthropic, and DeepSeek providers, and is
  exported from the top-level package.
- **`cubepi trace` CLI** (install with the `trace-cli` extra): discover, list,
  view, follow, and aggregate stats over local agent-run traces, with rich
  rendering and run-id prefix matching.
- **Tracing**: an OTLP exporter and a best-effort `trace()` scope helper. The
  tracing package now imports cleanly without `opentelemetry` installed.
- **Self-describing provider errors** that carry provider / model / cause
  context for easier debugging.

### Changed

- Provider reasoning/thinking and temperature handling is now driven by
  `CapabilityDescriptor` instead of per-provider ad-hoc payload quirks, giving
  consistent behavior across OpenAI, Anthropic, and DeepSeek.

### Fixed

- **Anthropic**: merge parallel tool results into a single user message; carry
  parsed tool arguments through the streaming `toolcall_end` event; compute
  `max_tokens` from the actual capability budget; honor per-request
  `thinking_budgets` overrides.
- **Agent loop / steering**: drain steering at the turn boundary;
  `after_model_response` now injects after tool results; backfill tool results
  for tool calls orphaned by a cancel.
- **DeepSeek**: correct reasoning-effort path and temperature range handling.

## Earlier releases

- **[0.4.0]** - 2026-05-19 — see the [release notes](https://github.com/cubeplexai/cubepi/releases/tag/v0.4.0).
- **[0.3.0]** - 2026-05-14 — see the [release notes](https://github.com/cubeplexai/cubepi/releases/tag/v0.3.0).
- **[0.2.0]** - 2026-05-10 — see the [release notes](https://github.com/cubeplexai/cubepi/releases/tag/v0.2.0).
- **[0.1.0]** - 2026-05-09 — initial release. See the [release notes](https://github.com/cubeplexai/cubepi/releases/tag/v0.1.0).

[Unreleased]: https://github.com/cubeplexai/cubepi/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/cubeplexai/cubepi/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/cubeplexai/cubepi/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/cubeplexai/cubepi/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/cubeplexai/cubepi/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cubeplexai/cubepi/releases/tag/v0.1.0
