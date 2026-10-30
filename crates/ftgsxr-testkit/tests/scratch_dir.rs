// SPDX-License-Identifier: Apache-2.0

//! Q-62 (#94): `process_dir` names a test binary's per-process folder without creating it,
//! and `scratch_dir` gives an empty directory in it on every call.

use std::fs;

use ftgsxr_testkit::{process_dir, scratch_dir};

const TMP: &str = env!("CARGO_TARGET_TMPDIR");
const BINARY: &str = env!("CARGO_CRATE_NAME");

#[test]
fn process_dir_names_the_binary_and_the_process_and_creates_nothing() {
    let dir = process_dir(TMP, "absent_binary");
    assert_eq!(
        dir,
        std::path::Path::new(TMP).join(format!("absent_binary-{}", std::process::id()))
    );
    assert!(!dir.exists(), "{}", dir.display());
}

#[test]
fn scratch_dir_starts_empty_on_every_call() {
    let dir = scratch_dir(TMP, BINARY, "starts_empty");
    assert_eq!(dir, process_dir(TMP, BINARY).join("starts_empty"));
    assert!(dir.is_dir(), "{}", dir.display());
    fs::write(dir.join("left.txt"), b"x").unwrap();
    fs::create_dir(dir.join("sub")).unwrap();

    let again = scratch_dir(TMP, BINARY, "starts_empty");
    assert_eq!(again, dir);
    assert_eq!(fs::read_dir(&again).unwrap().count(), 0);
}
