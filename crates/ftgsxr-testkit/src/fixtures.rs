// SPDX-License-Identifier: Apache-2.0

//! The committed fixtures directory and its checksum gate (§2.7 Fixtures, D-29).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use sha2::{Digest, Sha256};

/// The checksum list's name inside the fixtures directory.
const SUMS: &str = "SHA256SUMS";

/// The committed `tests/fixtures/` directory of the workspace, after verifying every file
/// its `SHA256SUMS` lists (§2.7: "A fixture's checksum mismatch fails the suite before any
/// test"; Q-1). The check runs once per process; while it fails, every call panics with a
/// message naming the first file, in `SHA256SUMS`'s order, that is missing or whose
/// SHA-256 differs from its line.
pub fn fixtures_dir() -> PathBuf {
    static VERIFIED: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    match VERIFIED.get_or_init(|| {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
        let dir = dir
            .canonicalize()
            .map_err(|e| format!("the fixtures directory {}: {e}", dir.display()))?;
        verify(&dir)?;
        Ok(dir)
    }) {
        Ok(dir) => dir.clone(),
        Err(message) => panic!("fixture checksum check failed (§2.7): {message}"),
    }
}

/// Checks every line `<64 hex digits><space><space or *><path>` of `dir/SHA256SUMS`
/// against the file it names, in order; the error names the first file missing or
/// mismatching. Paths may hold spaces: everything after the separator is the path.
pub(crate) fn verify(dir: &Path) -> Result<(), String> {
    let sums_path = dir.join(SUMS);
    let sums =
        fs::read_to_string(&sums_path).map_err(|e| format!("{}: {e}", sums_path.display()))?;
    let mut checked = 0usize;
    for (number, line) in sums.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let bad_line = || {
            format!(
                "{SUMS} line {}: {line:?} is not `<sha256>  <path>`",
                number + 1
            )
        };
        let (digest, rest) = line.split_at_checked(64).ok_or_else(bad_line)?;
        let name = rest
            .strip_prefix("  ")
            .or_else(|| rest.strip_prefix(" *"))
            .filter(|name| !name.is_empty())
            .ok_or_else(bad_line)?;
        if !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(bad_line());
        }
        let bytes = fs::read(dir.join(name)).map_err(|e| format!("{name}: missing ({e})"))?;
        let actual = hex(&Sha256::digest(&bytes));
        if !actual.eq_ignore_ascii_case(digest) {
            return Err(format!(
                "{name}: SHA-256 {actual}, {SUMS} says {}",
                digest.to_ascii_lowercase()
            ));
        }
        checked += 1;
    }
    if checked == 0 {
        return Err(format!("{} lists no file", sums_path.display()));
    }
    Ok(())
}

/// Lowercase hexadecimal digits of `bytes`.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A fresh empty directory under the system's temporary directory.
    fn scratch_dir() -> PathBuf {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ftgsxr-testkit-fixtures-{}-{}",
            std::process::id(),
            COUNT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("sub")).expect("a scratch directory");
        dir
    }

    /// A two-file fixtures directory with a correct `SHA256SUMS`, one path holding a space.
    fn good_dir() -> PathBuf {
        let dir = scratch_dir();
        fs::write(dir.join("a.txt"), b"alpha\n").unwrap();
        fs::write(dir.join("sub/b c.bin"), [0u8, 1, 2, 255]).unwrap();
        let sums = format!(
            "{}  a.txt\n{}  sub/b c.bin\n",
            hex(&Sha256::digest(b"alpha\n")),
            hex(&Sha256::digest([0u8, 1, 2, 255]))
        );
        fs::write(dir.join(SUMS), sums).unwrap();
        dir
    }

    #[test]
    fn verify_accepts_matching_files() {
        let dir = good_dir();
        assert_eq!(verify(&dir), Ok(()));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn verify_names_the_corrupted_file() {
        let dir = good_dir();
        fs::write(dir.join("sub/b c.bin"), [0u8, 1, 2, 254]).unwrap();
        let err = verify(&dir).unwrap_err();
        assert!(err.starts_with("sub/b c.bin: SHA-256 "), "{err}");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn verify_names_a_missing_file() {
        let dir = good_dir();
        fs::remove_file(dir.join("a.txt")).unwrap();
        let err = verify(&dir).unwrap_err();
        assert!(err.starts_with("a.txt: missing"), "{err}");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn verify_names_the_first_mismatch_in_list_order() {
        let dir = good_dir();
        fs::write(dir.join("a.txt"), b"beta\n").unwrap();
        fs::write(dir.join("sub/b c.bin"), b"").unwrap();
        let err = verify(&dir).unwrap_err();
        assert!(err.starts_with("a.txt: "), "{err}");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn verify_refuses_a_malformed_list() {
        let dir = good_dir();
        fs::write(dir.join(SUMS), "not a checksum line\n").unwrap();
        assert!(verify(&dir).unwrap_err().contains("line 1"));
        fs::write(dir.join(SUMS), "").unwrap();
        assert!(verify(&dir).unwrap_err().contains("lists no file"));
        fs::remove_dir_all(dir).unwrap();
    }

    /// The acceptance criterion's corrupted copy: the committed directory copied, one byte
    /// of one fixture changed, and the check names that file.
    #[test]
    fn verify_names_the_file_changed_in_a_copy_of_the_fixtures() {
        let committed = fixtures_dir();
        let copy = scratch_dir();
        copy_tree(&committed, &copy);
        assert_eq!(verify(&copy), Ok(()));
        let victim = copy.join("probes/P3.safetensors");
        let mut bytes = fs::read(&victim).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        fs::write(&victim, bytes).unwrap();
        let err = verify(&copy).unwrap_err();
        assert!(err.starts_with("probes/P3.safetensors: SHA-256 "), "{err}");
        fs::remove_dir_all(copy).unwrap();
    }

    fn copy_tree(from: &Path, to: &Path) {
        for entry in fs::read_dir(from).unwrap() {
            let entry = entry.unwrap();
            let target = to.join(entry.file_name());
            if entry.file_type().unwrap().is_dir() {
                fs::create_dir_all(&target).unwrap();
                copy_tree(&entry.path(), &target);
            } else {
                fs::copy(entry.path(), target).unwrap();
            }
        }
    }
}
