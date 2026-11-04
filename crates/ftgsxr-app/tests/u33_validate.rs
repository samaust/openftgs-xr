// SPDX-License-Identifier: Apache-2.0

//! U-33, the `cmd_validate` clause (§1.7, `cpu` lane): the fixtures validate (exit 0) with
//! their header information; an invalid file is refused with its E-FILE code and exit 2,
//! and validation stops at the first invalid file.

use std::fs;
use std::path::PathBuf;

use ftgsxr_app::{cmd_validate, exit_code};
use ftgsxr_core::{ModelKind, SAFETENSORS_PREFIX, header_len, read_info};
use ftgsxr_testkit::{exchange_bytes, fixtures_dir, random_model, scratch_dir};

fn fixture(rel: &str) -> PathBuf {
    fixtures_dir().join(rel)
}

/// The header-only information of an exchange file, read independently of `cmd_validate`.
fn header_info(path: &PathBuf) -> ftgsxr_core::ModelInfo {
    let bytes = fs::read(path).unwrap();
    let prefix: [u8; SAFETENSORS_PREFIX] = bytes[..SAFETENSORS_PREFIX].try_into().unwrap();
    let n = header_len(&prefix) as usize;
    read_info(
        &bytes[SAFETENSORS_PREFIX..SAFETENSORS_PREFIX + n],
        bytes.len() as u64,
    )
    .unwrap()
}

#[test]
fn u33_validate_the_fixtures() {
    let exchange: Vec<PathBuf> = [
        "probes/P1.safetensors",
        "probes/P10.safetensors",
        "checkpoint/tiny.safetensors",
        "checkpoint/single.safetensors",
    ]
    .iter()
    .map(|r| fixture(r))
    .collect();
    let mut paths = exchange.clone();
    paths.extend([
        fixture("checkpoint/tiny"),
        fixture("checkpoint/single"),
        fixture("checkpoint/tiny.tau0.ply"),
    ]);
    let infos = cmd_validate(&paths).expect("the fixtures are valid");
    assert_eq!(infos.len(), paths.len());

    // An exchange file: what its header tells.
    for (path, info) in exchange.iter().zip(&infos) {
        assert_eq!(info, &header_info(path), "{}", path.display());
    }

    // A checkpoint: its export's information but for the kind, the producer and the size,
    // which is that of `params.safetensors`, the file its identity hashes (D-40, Q-76).
    for (k, name) in ["tiny", "single"].iter().enumerate() {
        let info = &infos[exchange.len() + k];
        let export = header_info(&fixture(&format!("checkpoint/{name}.safetensors")));
        assert_eq!(
            (info.n, info.sh_degree),
            (export.n, export.sh_degree),
            "{name}"
        );
        assert_eq!(info.time, export.time, "{name}");
        assert_eq!(info.conventions, export.conventions, "{name}");
        assert!(matches!(info.kind, ModelKind::Checkpoint { .. }), "{name}");
        let params = fixture(&format!("checkpoint/{name}/params.safetensors"));
        assert_eq!(
            info.file_bytes,
            fs::metadata(params).unwrap().len(),
            "{name}"
        );
    }

    // The PLY snapshot of the tiny model at τ₀: its N, a single timestamp, the file's size.
    let ply = &infos[exchange.len() + 2];
    let tiny = &infos[0..exchange.len()][2];
    assert_eq!(ply.kind, ModelKind::Ply);
    assert_eq!(ply.n, tiny.n);
    assert_eq!((ply.time.tau0, ply.time.tau1), (0.0, 0.0));
    assert_eq!(
        ply.file_bytes,
        fs::metadata(fixture("checkpoint/tiny.tau0.ply"))
            .unwrap()
            .len()
    );
}

/// Every model in the fixture set, found through `tests/fixtures/SHA256SUMS` so a new
/// fixture is covered without editing this test (#208): each `.safetensors` file but the
/// renders files, each `.ply` file, and each checkpoint directory (the parent of a
/// `manifest.json`) in place of the files inside it. `http/empty.safetensors` is left out:
/// it is the 0-byte file of U-31's HTTP read, invalid on purpose (`tests/fixtures/README.md`).
fn every_model_fixture() -> Vec<PathBuf> {
    let sums = fs::read_to_string(fixtures_dir().join("SHA256SUMS")).unwrap();
    // `<digest>  <path>`; a path may hold spaces.
    let listed: Vec<&str> = sums
        .lines()
        .filter_map(|l| l.split_once("  ").map(|(_, p)| p))
        .collect();
    let checkpoints: Vec<&str> = listed
        .iter()
        .filter_map(|p| p.strip_suffix("/manifest.json"))
        .collect();
    let mut models: Vec<PathBuf> = checkpoints.iter().map(|c| fixture(c)).collect();
    for rel in &listed {
        let model = (rel.ends_with(".safetensors") && !rel.ends_with(".renders.safetensors"))
            || rel.ends_with(".ply");
        let in_checkpoint = checkpoints
            .iter()
            .any(|c| rel.starts_with(&format!("{c}/")));
        if model && !in_checkpoint && *rel != "http/empty.safetensors" {
            models.push(fixture(rel));
        }
    }
    models
}

#[test]
fn u33_validate_every_model_fixture() {
    let paths = every_model_fixture();
    // The probes P1–P10 at least, so a change to SHA256SUMS' form can't empty the list.
    assert!(paths.len() >= 10, "{paths:?}");
    for path in &paths {
        let infos = cmd_validate(std::slice::from_ref(path))
            .unwrap_or_else(|e| panic!("{}: {}: {}", path.display(), e.code(), e.user_text()));
        assert_eq!(infos.len(), 1, "{}", path.display());
    }
}

#[test]
fn u33_validate_refuses_an_invalid_file_with_exit_2() {
    let dir = scratch_dir(
        env!("CARGO_TARGET_TMPDIR"),
        crate::test_file_stem(),
        "invalid",
    );
    let mut model = random_model(7, 8, 1);
    model.opacity[3] = 1.0; // outside (0, 1): E-FILE-VALUE
    let bad = dir.join("bad.safetensors");
    fs::write(&bad, exchange_bytes(&model)).unwrap();
    let missing = dir.join("missing.safetensors");

    let e = cmd_validate(&[fixture("probes/P1.safetensors"), bad.clone(), missing])
        .expect_err("the opacity of 1 is refused");
    assert_eq!(e.code(), "E-FILE-VALUE");
    assert_eq!(exit_code(&e), 2);
    // Q-75: the text names the file; the first invalid file stops the run.
    assert!(
        e.user_text().contains("bad.safetensors"),
        "{}",
        e.user_text()
    );

    let e = cmd_validate(&[dir.join("missing.safetensors")]).unwrap_err();
    assert_eq!((e.code(), exit_code(&e)), ("E-IO", 2));
}

// Q-260
#[test]
fn u33_validate_takes_a_checkpoints_manifest_path() {
    // Q-260: a checkpoint's `manifest.json` names the checkpoint, so `validate` of it
    // gives the directory's information, the size being `params.safetensors`' (Q-76).
    let dir = fixture("checkpoint/tiny");
    let by_manifest = cmd_validate(&[dir.join("manifest.json")]).expect("the manifest path");
    assert_eq!(by_manifest, cmd_validate(&[dir.clone()]).unwrap());
    assert_eq!(
        by_manifest[0].file_bytes,
        fs::metadata(dir.join("params.safetensors")).unwrap().len()
    );
    // A `manifest.json` of another format is E-FILE-FORMAT, exit 2, as its directory.
    let scratch = scratch_dir(
        env!("CARGO_TARGET_TMPDIR"),
        crate::test_file_stem(),
        "other-format",
    );
    let manifest = scratch.join("manifest.json");
    fs::write(&manifest, br#"{"format": "ftgs-scene", "version": 1}"#).unwrap();
    let e = cmd_validate(&[manifest]).unwrap_err();
    assert_eq!((e.code(), exit_code(&e)), ("E-FILE-FORMAT", 2));
    assert_eq!(
        e.code(),
        cmd_validate(&[scratch]).unwrap_err().code(),
        "the manifest path fails as its directory does"
    );
}
