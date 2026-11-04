// SPDX-License-Identifier: Apache-2.0

//! Q-62 (#94, #413): `process_dir` names a test file's per-process folder without creating
//! it, and `scratch_dir` gives an empty directory in it on every call; two test files of
//! one test binary that use the same `name` keep folders of their own.

use std::fs;

use ftgsxr_testkit::{process_dir, scratch_dir};

const TMP: &str = env!("CARGO_TARGET_TMPDIR");

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
    let stem = crate::test_file_stem();
    let dir = scratch_dir(TMP, stem, "starts_empty");
    assert_eq!(dir, process_dir(TMP, stem).join("starts_empty"));
    assert!(dir.is_dir(), "{}", dir.display());
    fs::write(dir.join("left.txt"), b"x").unwrap();
    fs::create_dir(dir.join("sub")).unwrap();

    let again = scratch_dir(TMP, stem, "starts_empty");
    assert_eq!(again, dir);
    assert_eq!(fs::read_dir(&again).unwrap().count(), 0);
}

#[test]
fn two_test_files_of_one_binary_with_the_same_name_keep_their_own_folders() {
    let stem = crate::test_file_stem();
    assert_eq!(stem, "scratch_dir");
    let mine = scratch_dir(TMP, stem, "same_name");
    fs::write(mine.join("mine.txt"), b"x").unwrap();

    let theirs = crate::scratch_dir_twin::scratch("same_name");
    assert_eq!(
        theirs,
        process_dir(TMP, "scratch_dir_twin").join("same_name")
    );
    assert_ne!(theirs, mine);
    assert!(theirs.is_dir() && fs::read_dir(&theirs).unwrap().count() == 0);
    // The twin's call left this file's folder as it was.
    assert!(mine.join("mine.txt").is_file());
}
