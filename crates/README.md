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
and `dv` lanes add `--test-threads 1` (§2.7, Lanes). CI runs the `cpu` lane and the
skeleton test on every push (`.github/workflows/ci.yml`).
