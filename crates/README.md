# Crates and their tests

The workspace's crates are those of spec §2.1, one directory each; the skeleton test
(U-35, `crates/ftgsxr-core/tests/u35_skeleton.rs`) fails if one is added, removed or
renamed, or if a crate depends on one the band table does not allow.

## Where the tests live

- **Spec tests**, one file per §3 row, in `crates/<crate>/tests/`, named after the row
  (`u07_temporal_opacity_rebased.rs`, `r03_probe_p3.rs`), at the public interface (§2.7,
  Organisation).
- **Private tests** of a module's helpers in its `#[cfg(test)]` block, named after the
  helper (`tdd` skill, "Private tests").
- **Fixtures** in `tests/fixtures/` at the workspace root, reached through
  `ftgsxr_testkit::fixtures_dir()` (§2.7, Fixtures).

## Test lanes

Every test belongs to one lane, by what it needs (§2.7, D-28; "Test lane" in
`docs/GLOSSARY.md`):

| Lane | Needs | Gate |
| --- | --- | --- |
| `cpu` | nothing | none: plain `cargo test` runs it |
| `gpu` | a graphics adapter | `#[ignore = "gpu"]`, name prefix `gpu_` |
| `xr` | an OpenXR runtime (Monado on the desktop) | `#[ignore = "xr"]`, prefix `xr_` |
| `dv` | the Quest over adb | `#[ignore = "dv"]`, prefix `dv_` |
| `net` | the internet | `#[ignore = "net"]`, prefix `net_` |
| `ds` | `$OPENFTGS_DATASETS` and the trained tiny models | `#[ignore = "ds"]`, prefix `ds_` |

A lane other than `cpu` runs with `cargo test -- --ignored <prefix>`; the `gpu`, `xr`
and `dv` lanes add `--test-threads 1` (§2.7, Lanes). A `gpu` (or `ds`) test under
`crates/<crate>/tests/` starts with `let _layer = ftgsxr_testkit::layer_messages();`,
before its first device, so a Vulkan validation layer warning or error fails it in a
debug build (§1.13, R-16, Q-105). A test that runs the binary `ftgsxr` instead points
`VK_LAYER_SETTINGS_PATH` at a layer settings file and fails on an error or warning in
the layer's log (R-15's and `bench`'s runs in `ftgsxr-desktop/tests/cli.rs`), since the
child process's messages never reach the test's subscriber. The `gpu` unit tests in a
crate's `src/` (`#[cfg(test)]` blocks of `ftgsxr-render`) are the exception: the testkit
depends on `ftgsxr-render`, so they cannot use it, and they rely on the device's own
validation collector (Q-74). CI runs the `cpu` lane and the skeleton test on every push
(`.github/workflows/ci.yml`).
