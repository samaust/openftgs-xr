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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_dir_names_the_binary_and_the_process_and_creates_nothing() {
        let tmp =
            std::env::temp_dir().join(format!("ftgsxr-testkit-scratch-{}", std::process::id()));
        let dir = process_dir(&tmp, "u99_example");
        assert_eq!(dir, tmp.join(format!("u99_example-{}", std::process::id())));
        assert!(!dir.exists());
    }

    #[test]
    fn scratch_dir_starts_empty_on_every_call() {
        let tmp = std::env::temp_dir().join(format!(
            "ftgsxr-testkit-scratch-fresh-{}",
            std::process::id()
        ));
        let dir = scratch_dir(&tmp, "u99_example", "case");
        assert_eq!(dir, process_dir(&tmp, "u99_example").join("case"));
        assert!(dir.is_dir());
        fs::write(dir.join("left.txt"), b"x").unwrap();
        fs::create_dir(dir.join("sub")).unwrap();

        let again = scratch_dir(&tmp, "u99_example", "case");
        assert_eq!(again, dir);
        assert_eq!(fs::read_dir(&again).unwrap().count(), 0);
        let _ = fs::remove_dir_all(&tmp);
    }
}
