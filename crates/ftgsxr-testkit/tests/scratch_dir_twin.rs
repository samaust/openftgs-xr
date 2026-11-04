// SPDX-License-Identifier: Apache-2.0

//! Q-62 (#413): a second test file of the testkit's test binary, whose scratch folders
//! `scratch_dir.rs` checks against its own.

use std::path::PathBuf;

use ftgsxr_testkit::scratch_dir;

/// A scratch directory of this test file.
pub(crate) fn scratch(name: &str) -> PathBuf {
    scratch_dir(env!("CARGO_TARGET_TMPDIR"), crate::test_file_stem(), name)
}
