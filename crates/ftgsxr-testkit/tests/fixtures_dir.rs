// SPDX-License-Identifier: Apache-2.0

//! §2.7 Fixtures (D-29), Q-1: `fixtures_dir` returns the committed `tests/fixtures/`
//! directory once its `SHA256SUMS` has been verified.

use ftgsxr_testkit::fixtures_dir;

#[test]
fn fixtures_dir_is_the_committed_directory() {
    let dir = fixtures_dir();
    assert!(dir.is_absolute(), "{}", dir.display());
    assert!(dir.ends_with("tests/fixtures"), "{}", dir.display());
    for file in ["SHA256SUMS", "README.md", "probes/P1.safetensors"] {
        assert!(dir.join(file).is_file(), "{file} in {}", dir.display());
    }
}

#[test]
fn fixtures_dir_is_stable_across_calls() {
    assert_eq!(fixtures_dir(), fixtures_dir());
}
