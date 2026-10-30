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

#[test]
fn u33_validate_refuses_an_invalid_file_with_exit_2() {
    let dir = scratch_dir(
        env!("CARGO_TARGET_TMPDIR"),
        env!("CARGO_CRATE_NAME"),
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
