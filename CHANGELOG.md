# Changelog

All notable changes to pycordis are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-11-11

### Changed

- New README: a 20-line quickstart, a real agent example, a comparison table
  and an FAQ. Every README program runs in the test suite and must print its
  documented output, and the config example must produce the documented
  `pycordis doctor` output.
- The full feature documentation moved to `docs/reference.md`.
- The distribution is published as `pycordis-framework`; the import name stays
  `pycordis`.

### Fixed

- Two wall-clock bounds in the test suite tolerate slow CI runners.

## [0.1.0] - 2026-11-10

First public release.

### Added

- A port of [cordis](https://github.com/cordiverse/cordis) 4.1.0 for asyncio:
  contexts, plugins running in fibers, services with dependency injection,
  events (`emit`, `parallel`, `serial`, `bail`, `waterfall`), effects and
  disposers, the reflect service, the registry, `Service`, the logger and
  `Volatile` configs, with cordis's lifecycle, event names and error messages.
- Typed service keys (`ServiceKey[T]`), typed plugin configs and
  `@plugin_meta`.
- `current_context()` / `use_context()`, per-request scopes (`ctx.scope`) and
  lifecycle-bound `Collection`s.
- Supervision: `RestartPolicy` with backoff, `load_timeout`,
  `dispose_timeout` and `LoadTimeoutError`.
- Diagnostics: `ctx.diagnose()`, dependency-cycle detection and
  `pycordis doctor`.
- Observability: lifecycle telemetry, `FiberStats`, `JsonExporter` and a
  stdlib `HealthServer` (`/healthz`, `/readyz`, `/inspect`).
- A config-file loader (TOML, JSON, optional YAML) with includes,
  `${env:...}` / `${file:...}` interpolation, an import allowlist, hot
  reload, comment-preserving write-back, profiles and JSON Schema export.
- The `pycordis` command: `start`, `inspect`, `doctor` and `schema`.
- Zero runtime dependencies; strict-typed with `py.typed`.

### Notes

Behaviour worth knowing when coming from cordis or from pre-release builds:

- **New `Context` methods shadow service names.** `ctx.use(KEY)`,
  `ctx.scope(...)` and `ctx.diagnose()` are methods, like `ctx.inspect()`, so
  services named `"use"`, `"scope"` or `"diagnose"` can no longer be read as
  attributes. Read them with `ctx.get("use")` and so on.
- `ctx.plugin(plugin, config)` type-checks `config` against the plugin's config
  parameter. A plugin typed as plain `object` takes only an `Any` or `None` config.
  A mapping or an omitted config is always accepted (see **`plugin_meta` and
  typed configs**).
- A dataclass `Config` schema accepts an instance of itself unchanged, as well as
  a mapping to validate: `ctx.plugin(chat, ChatConfig(port=2))`.
- The CLI bounds each loader entry's unload at 10 seconds by default
  (`--dispose-timeout`); the library itself still has no timeouts by default.
- Plugin attributes named `policy`, `load_timeout` and `dispose_timeout` are
  now read as supervision metadata: on a non-module plugin, such an attribute
  of the wrong type makes `ctx.plugin()` raise (`TypeError` / `ValueError`).
- **Mistyped inputs raise `TypeError` at once**, naming the problem, instead
  of failing later with an `AttributeError` or odd behaviour:
  - an event name that is not a `str` or `EventKey`
    (`event name must be a str or EventKey, got int`), including a lone one:
    `root.emit(123)` (and `parallel`, `serial`, `bail`, `waterfall`) used to
    take `123` as the receiver and fail with an `IndexError`. A receiver
    followed by a name stays optional. A call with no event name, or a
    `waterfall` without its final `next` argument, raises `TypeError` too;
  - an inject item that is not a `str` or `ServiceKey`, or an inject
    declaration that is not a list or mapping;
  - a mistyped logger intercept `name` / `level`;
  - a non-callable `Config.merge` or `model_validate`.

  When `ctx.plugin()` rejects an object whose `apply` cannot be read, its
  `TypeError` now carries the getter's error as `__cause__`.
- **Value classes are `@final`.** `ServiceKey`, `EventKey`, `RestartPolicy`,
  `Backoff`, `Diagnosis`, `Reason`, `Issue`, `Message`, `EffectMeta`,
  `EventOptions`, `LifecycleEvent`, `FiberStats`, `CollectionChange`, the
  handles `EffectDisposer`, `ListenerDisposer`, `Scope` and `Volatile`, the
  `Inject` namespace, and the loader records `Entry`, `LoadedEntry`,
  `LoaderOptions` and `Overlay` can no longer be subclassed. This is a
  type-checker error, not a runtime one. Extension points stay open: `Context`,
  `Service`, `Fiber`, `Logger`, the exporters, `MemoryExporter`, `Collection`,
  `Loader`, `HealthServer` and their `Config` classes.
- **A top-level `$schema` string is allowed in config files.** Earlier builds
  rejected it as an unknown top-level key. The loader now ignores it (in included files and
  overlays too) and keeps it on write-back, and `pycordis schema` output allows
  it, so a JSON or YAML config can point its editor at its schema. A `$schema`
  that is not a string is rejected.
- Behaviour changes for those who tried pre-release builds:
  - `/readyz` (with `ready_requires="loader"`) is `503` with a `not-started`
    reason (entry `null`) until `Loader.start()` has loaded the config; it used
    to pass while the loader had no entries yet. `Loader.started` says
    whether it has started.
  - `load_timeout` also fails a body that catches its cancellation and returns
    late (`LoadTimeoutError`), instead of letting it go `ACTIVE`.
  - A `Collection` whose provider unloads drops the items still in it, with a
    `remove` change each; a reference kept past that is empty.
- **`${file:...}` rejects non-regular files without opening them.** A FIFO,
  device, tty or directory fails with `cannot read file <name>: not a regular
  file`. Opening one could block or have side effects. A missing file now
  reports `No such file or directory`.
