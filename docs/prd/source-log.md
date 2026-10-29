# Implementation source log

This log continues the spec's source log (Appendix A of `docs/spec/openftgs-xr-spec.md`) for the implementation phase. Every source consulted gets one row: what it was and what was taken from it. Reading this repository and the spec needs no row.

If a forbidden source is met by accident, its row starts with **Incident** and says what was seen, that reading stopped, and that nothing was taken.

| # | Date | Source | What was taken |
| --- | --- | --- | --- || T2.1 | 2026-10-29 | crates.io API, `bevy` 0.19.1 (https://crates.io/api/v1/crates/bevy/0.19.1) | `rust_version` 1.95.0, the workspace's `rust-version` (§2.6); license field MIT OR Apache-2.0 |
| T2.2 | 2026-10-29 | crates.io API, `bevy_math` 0.19.1 dependency list (https://crates.io/api/v1/crates/bevy_math/0.19.1/dependencies) | glam `^0.32.0` (normal dependency): the workspace pins glam 0.32, not the 0.33 of §2.3's parenthesis (PRD #1, "glam before any code") |
| T2.3 | 2026-10-29 | crates.io API, dependency lists of `bevy_render` 0.19.1 and `bevy_winit` 0.19.1 | wgpu, wgpu-types and naga `^29.0.3`; winit `^0.30`: the pins of §2.3's "Bevy's pin" rows |
| T2.4 | 2026-10-29 | crates.io API, dependency lists of `bevy_mod_openxr` 0.6.0, `bevy_mod_xr` 0.6.0, `bevy_xr_utils` 0.6.0 and `openxr` 0.21.1 | openxr `^0.21.1`, openxr-sys `^0.13.1`, android-activity `=0.6.0`, jni `^0.20`, ndk-context `^0.1`, Bevy `^0.19`: the pins of §2.3's "the XR crates' pin" rows |
| T2.5 | 2026-10-29 | crates.io API, dependency lists of `bevy_egui` 0.42.0 and `bevy_kira_audio` 0.26.0 | egui `^0.36`, kira `^0.12.1` (bevy\_egui's and bevy\_kira\_audio's pins) |
| T2.6 | 2026-10-29 | crates.io API, license fields of §2.3's "knowledge" crates: sha2 0.10.9, thiserror 2.0.21, rayon 1.12.0, url 2.5.8, image 0.25.10, clap 4.6.7, proptest 1.12.0, approx 0.5.1, tiny\_http 0.12.0, cargo-deny 0.20.2 | each license as §2.3 states it (MIT OR Apache-2.0; approx Apache-2.0); the non-crate "knowledge" rows (Gradle, the Android SDK, GameActivity) wait for `android/` in M-3 |
| T2.7 | 2026-10-29 | crates.io API, version lists and license fields of §2.3's "verified" crates (safetensors, memmap2, bytemuck, half, serde, serde\_json, toml, ureq, rustls, ring, webpki-roots, directories, rfd, tracing, tracing-subscriber, android\_logger, ndk-context, jni, glam, wgpu, wgpu-hal, naga, winit, egui, kira, openxr-sys and the three XR crates) | that a release exists under each pinned version and its license field matches §2.3 |
| T2.8 | 2026-10-29 | crates.io API, feature tables of `ureq` 3.5.0 and `rustls` 0.23 | rustls 0.23's default features select the aws-lc-rs provider, so the workspace table turns rustls's default features off and D-20's `ring` is chosen by the crate that uses it; ureq 3.5's `rustls` feature uses ring and webpki-roots (for the `http` ticket) |
| T2.9 | 2026-10-29 | cargo-deny 0.20.2 (MIT OR Apache-2.0), installed with `cargo install --locked cargo-deny`; the configuration template `cargo deny init` writes | the sections and keys of `deny.toml` (`[graph]`, `[advisories]`, `[licenses]`, `[bans]`, `[sources]`), filled with §2.3's allowlist |
