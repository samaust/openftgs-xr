// SPDX-License-Identifier: Apache-2.0

//! Scratch directories of one test binary in one process (#94, Q-62).

use std::fs;
use std::path::{Path, PathBuf};

/// `<tmp>/<binary>-<pid>`: the folder of the test binary `binary` in this process, under the
/// temporary directory `tmp` (an integration test passes `env!("CARGO_TARGET_TMPDIR")` and
/// `env!("CARGO_CRATE_NAME")`). No other binary and no overlapping run, in this worktree or
/// another sharing the target directory, uses it. Only the path: nothing is created (Q-62).
pub fn process_dir(tmp: impl AsRef<Path>, binary: &str) -> PathBuf {
    tmp.as_ref()
        .join(format!("{binary}-{}", std::process::id()))
}

/// A fresh, empty directory `name` in [`process_dir`]`(tmp, binary)`: whatever an earlier
/// call left there is removed first (Q-62).
pub fn scratch_dir(tmp: impl AsRef<Path>, binary: &str, name: &str) -> PathBuf {
    let dir = process_dir(tmp, binary).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create the scratch directory");
    dir
}
