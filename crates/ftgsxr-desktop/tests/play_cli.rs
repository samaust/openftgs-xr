// SPDX-License-Identifier: Apache-2.0

//! `ftgsxr play`'s command line (§1.7, D-36, Q-157, `cpu` lane): `--help`; an unknown
//! `--tier` or `--log-level` is a usage error (exit 2); a second scene file is a usage
//! error; a missing model path exits 2 before any window. Each run gets its own XDG
//! directories, so nothing reaches the user's.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use ftgsxr_testkit::scratch_dir;

fn scratch(name: &str) -> PathBuf {
    scratch_dir(env!("CARGO_TARGET_TMPDIR"), env!("CARGO_CRATE_NAME"), name)
}

/// `ftgsxr` with `args`, its home and XDG directories under `home` and no display.
fn ftgsxr(home: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ftgsxr"))
        .args(args)
        .env("HOME", home)
        .env("XDG_CONFIG_HOME", home.join("config"))
        .env("XDG_CACHE_HOME", home.join("cache"))
        .env("XDG_DATA_HOME", home.join("data"))
        .env_remove("DISPLAY")
        .env_remove("WAYLAND_DISPLAY")
        .output()
        .expect("the binary runs")
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

#[test]
fn play_help_lists_its_arguments() {
    let home = scratch("help");
    let o = ftgsxr(&home, &["play", "--help"]);
    assert_eq!(o.status.code(), Some(0), "{}", stderr(&o));
    let out = String::from_utf8(o.stdout).unwrap();
    for arg in ["MODEL|URL|SCENE", "--tier", "--settings", "--log-level"] {
        assert!(out.contains(arg), "{arg} in {out}");
    }
    #[cfg(feature = "xr")]
    assert!(out.contains("--xr"), "{out}");
    // `play` is one of the binary's subcommands.
    let o = ftgsxr(&home, &["--help"]);
    assert!(String::from_utf8(o.stdout).unwrap().contains("play"));
}

#[test]
fn an_unknown_tier_or_level_is_a_usage_error() {
    let home = scratch("unknown-tier");
    let o = ftgsxr(&home, &["play", "--tier", "ultra"]);
    assert_eq!(o.status.code(), Some(2));
    assert!(stderr(&o).contains("ultra"), "{}", stderr(&o));
    let o = ftgsxr(&home, &["play", "--log-level", "loud"]);
    assert_eq!(o.status.code(), Some(2));
}

#[test]
fn a_second_scene_file_is_a_usage_error() {
    let home = scratch("two-scenes");
    let a = home.join("a.ftgsxr-scene.json");
    let b = home.join("b.ftgsxr-scene.json");
    fs::write(&a, "{}").unwrap();
    fs::write(&b, "{}").unwrap();
    let o = ftgsxr(&home, &["play", a.to_str().unwrap(), b.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(2));
    let err = stderr(&o);
    assert!(err.contains("second scene"), "{err}");
    assert!(err.contains("Usage"), "{err}");
}

#[test]
fn a_missing_model_exits_2_before_any_window() {
    let home = scratch("missing-model");
    let missing = home.join("nothing.safetensors");
    let o = ftgsxr(&home, &["play", missing.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(2));
    let err = stderr(&o);
    assert!(err.starts_with("ftgsxr: "), "{err}");
    assert!(err.contains("nothing.safetensors"), "{err}");
    // Before the log too: no log file was made.
    assert!(!home.join("cache/openftgs-xr/logs").exists());
}

/// `xr` lane (§1.7, Q-162): `play --xr` on a desktop whose loader finds no runtime
/// (`XR_RUNTIME_JSON` naming a missing file) exits 2 with E-XR-RUNTIME on standard error.
/// The window's event loop opens before the XR instance is known (Q-220), so the test
/// keeps the session's `DISPLAY` or `WAYLAND_DISPLAY`.
#[cfg(feature = "xr")]
#[test]
#[ignore = "xr"]
fn xr_play_without_a_runtime_exits_2_with_e_xr_runtime() {
    let home = scratch("xr-no-runtime");
    let o = Command::new(env!("CARGO_BIN_EXE_ftgsxr"))
        .args(["play", "--xr"])
        .env("HOME", &home)
        .env("XDG_CONFIG_HOME", home.join("config"))
        .env("XDG_CACHE_HOME", home.join("cache"))
        .env("XDG_DATA_HOME", home.join("data"))
        .env("XR_RUNTIME_JSON", home.join("no-such-runtime.json"))
        .output()
        .expect("the binary runs");
    let err = stderr(&o);
    assert_eq!(o.status.code(), Some(2), "{err}");
    assert!(
        err.lines()
            .any(|l| l.starts_with("ftgsxr: E-XR-RUNTIME: No OpenXR runtime")),
        "{err}"
    );
}
