// SPDX-License-Identifier: Apache-2.0

//! `list_dir` (§1.5's local directory paragraph, `cpu` lane): the part of U-31's `LocalDir`
//! clause that the listing module carries. A temporary directory holding a probe, the tiny
//! checkpoint directory, the PLY snapshot and stray files lists exactly the three models,
//! with their kinds (Q-38 for the names, locations, sizes and what is skipped).

use std::fs;
use std::path::{Path, PathBuf};

use ftgsxr_io::{Entry, EntryKind, list_dir};
use ftgsxr_testkit::fixtures_dir;

/// A fresh, empty directory under Cargo's temporary directory for integration tests.
fn fresh_dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn entry(dir: &Path, file: &str, name: &str, kind: EntryKind, size: u64) -> Entry {
    Entry {
        name: name.to_owned(),
        location: dir.join(file).to_str().unwrap().to_owned(),
        kind,
        size: Some(size),
        mtime: None,
        sha256: None,
        thumbnail: None,
        audio: None,
        scene: None,
    }
}

#[test]
fn u31_list_dir_one_level_with_kinds() {
    let fixtures = fixtures_dir();
    let models = fixtures.join("listing/models");
    let dir = fresh_dir("u31_list_dir_one_level_with_kinds");
    let copy = |from: &Path, to: &str| {
        fs::copy(from, dir.join(to)).unwrap();
    };
    // The three models of the listing fixture.
    copy(&models.join("probe p1.safetensors"), "probe p1.safetensors");
    copy(&models.join("snapshot.ply"), "snapshot.ply");
    fs::create_dir(dir.join("tiny")).unwrap();
    copy(&models.join("tiny/manifest.json"), "tiny/manifest.json");
    copy(
        &models.join("tiny/params.safetensors"),
        "tiny/params.safetensors",
    );
    // Stray files: not a model; a safetensors file of another format (`ftgs-renders`); a
    // dot-name; an exchange model cut short, whose header no longer covers the file.
    copy(&models.join("notes.txt"), "notes.txt");
    copy(
        &fixtures.join("probes/P1.renders.safetensors"),
        "P1.renders.safetensors",
    );
    copy(&models.join("probe p1.safetensors"), ".hidden.safetensors");
    let probe = fs::read(models.join("probe p1.safetensors")).unwrap();
    fs::write(dir.join("cut.safetensors"), &probe[..probe.len() - 4]).unwrap();
    // Directories that are not checkpoints: empty, and one whose manifest has another format.
    fs::create_dir(dir.join("empty")).unwrap();
    fs::create_dir(dir.join("scene")).unwrap();
    fs::write(
        dir.join("scene/manifest.json"),
        r#"{"format": "ftgs-scene", "version": 1}"#,
    )
    .unwrap();
    // The checkpoint inside `tiny/` is one level down and its params file is not listed.

    let entries = list_dir(&dir).unwrap();
    assert_eq!(
        entries,
        vec![
            entry(
                &dir,
                "probe p1.safetensors",
                "probe p1",
                EntryKind::Exchange,
                1092
            ),
            entry(&dir, "snapshot.ply", "snapshot", EntryKind::Ply, 2022),
            // manifest.json (2605 bytes) + params.safetensors (17072 bytes), as the index.
            entry(&dir, "tiny", "tiny", EntryKind::Checkpoint, 19677),
        ]
    );
}

#[test]
fn u31_list_dir_missing_directory_is_e_io() {
    let dir = fresh_dir("u31_list_dir_missing_directory_is_e_io").join("absent");
    assert_eq!(list_dir(&dir).unwrap_err().code(), "E-IO");
}
