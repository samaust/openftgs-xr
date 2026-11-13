// SPDX-License-Identifier: Apache-2.0

//! Dv-10 (§3.4, `dv` lane; §1.5, D-49, Q-15, Q-61, Q-273): an `adb push` of a probe into
//! the models directory lists it in the library; the shim's `listTree` and `openDocument`
//! read a model from a granted tree. The driver pushes P1 as `models/dv10-probe.safetensors`
//! (removed afterwards, also when the test fails), launches `dv10` and reads its verdict.
//! The tree is the one the owner granted once through the app's picker, whose URI the app
//! logged (`picked tree <uri>`), passed from `FTGSXR_DV_TREE` in the extra `ftgsxr-tree`;
//! without the variable the tree clause is not run, which the test prints, and the ticket
//! stays unverified (`docs/agents/issue-tracker.md`, "Unverified tickets"). Prerequisites:
//! `dv_support`'s.

use crate::dv_support;

use std::path::PathBuf;
use std::time::Duration;

use dv_support::{PACKAGE, adb, apk, device, install, stop, verdict};

/// The environment variable holding the granted tree's URI (Q-273).
const TREE_VAR: &str = "FTGSXR_DV_TREE";
/// The probe's path on the device, in the app-external models directory (Q-15).
fn probe_on_device() -> String {
    format!("/sdcard/Android/data/{PACKAGE}/files/models/dv10-probe.safetensors")
}

/// Removes the pushed probe when dropped.
struct Pushed;

impl Drop for Pushed {
    fn drop(&mut self) {
        adb(&["shell", "rm", "-f", &probe_on_device()]);
    }
}

#[test]
#[ignore = "dv"]
fn dv_dv10_a_pushed_probe_is_listed_and_a_granted_tree_opens_a_model() {
    let apk = apk();
    device();
    install(&apk);
    let probe = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/probes/P1.safetensors");
    let on_device = probe_on_device();
    let dir = on_device.rsplit_once('/').map(|(d, _)| d).unwrap_or(".");
    adb(&["shell", "mkdir", "-p", dir]);
    adb(&["push", &probe.to_string_lossy(), &on_device]);
    let _pushed = Pushed;

    let tree = std::env::var(TREE_VAR).ok().filter(|t| !t.is_empty());
    adb(&["logcat", "-c"]);
    let start = std::time::Instant::now();
    let mut args = vec![
        "shell",
        "am",
        "start",
        "-S",
        "-n",
        dv_support::ACTIVITY,
        "--es",
        "ftgsxr-test",
        "dv10",
    ];
    // `am start` runs in the device's shell: the URI is quoted for it.
    let quoted = tree.as_ref().map(|t| format!("'{t}'"));
    if let Some(q) = &quoted {
        args.extend(["--es", "ftgsxr-tree", q.as_str()]);
    }
    adb(&args);
    let result = verdict("dv10", start, Duration::from_secs(150));
    stop();

    let (pass, detail, _) = result.expect("a dv10 verdict line in logcat");
    assert!(pass, "dv10 FAIL {detail}");
    assert!(
        detail.contains("dv10-probe"),
        "the probe is listed: {detail}"
    );
    if tree.is_none() {
        println!("dv10: tree clause NOT RUN ({TREE_VAR} unset); the ticket stays unverified");
    } else {
        assert!(
            detail.contains("loaded"),
            "the tree's model loaded: {detail}"
        );
    }
}
