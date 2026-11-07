// SPDX-License-Identifier: Apache-2.0

//! The device tests' runner (§3.4, Q-122): the launch intent's string extra
//! `ftgsxr-test` names a test, whose verdict is the logcat line
//! `ftgsxr-test: <name> PASS|FAIL <detail>` under the tag `ftgsxr`.

use std::path::{Path, PathBuf};

use bevy::prelude::*;

mod dv1;
mod dv10;
mod dv12;
mod dv13;
mod dv2;
mod dv3;
mod dv4;
mod dv8;
mod dv9;

/// The intent extra naming the test (§3.4).
pub(crate) const EXTRA: &str = "ftgsxr-test";

/// The directory of the pushed and written files, `dv/` under the app-external files
/// directory (Q-122).
pub(crate) fn dv_dir() -> Option<PathBuf> {
    bevy::android::ANDROID_APP
        .get()
        .and_then(|a| a.external_data_path())
        .map(|e| e.join("dv"))
}

/// Runs `name` when it is a headless test (Dv-3, Dv-4: before the XR session is created,
/// on the headless renderer's own device, Q-122) with the pushed and written files in
/// `dv`, printing its verdict; `false` when `name` is not one, for `add`.
pub(crate) fn run_headless(name: &str, dv: &Path) -> bool {
    let result = match name {
        dv3::NAME => dv3::run(dv),
        dv4::NAME => dv4::run(dv),
        _ => return false,
    };
    match result {
        Ok(detail) => verdict(name, true, &detail),
        Err(detail) => verdict(name, false, &detail),
    }
    true
}

/// Prints the verdict line (§3.4).
pub(crate) fn verdict(name: &str, pass: bool, detail: &str) {
    let word = if pass { "PASS" } else { "FAIL" };
    info!("ftgsxr-test: {name} {word} {detail}");
}

/// Adds the test named by the extra to `app`; an unknown name prints FAIL at once.
pub(crate) fn add(app: &mut App, name: &str) {
    match name {
        dv1::NAME => dv1::add(app),
        dv10::NAME => dv10::add(app),
        dv2::NAME => dv2::add(app),
        dv12::NAME => dv12::add(app),
        dv13::NAME => dv13::add(app),
        dv8::NAME => dv8::add(app),
        dv9::NAME => dv9::add(app),
        _ => verdict(name, false, "unknown test"),
    }
}
