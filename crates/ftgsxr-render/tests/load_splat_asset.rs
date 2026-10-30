// SPDX-License-Identifier: Apache-2.0

//! `load_splat_asset` (§2.1, §1.2, D-51, Q-T119.2) on the committed fixtures: an exchange
//! file and a PLY snapshot from bytes (the kind sniffed), a checkpoint by path, and the
//! explicit reduction (D-13). Each asset's information, box and identity agree with the
//! core readers'.

use ftgsxr_core::{GpuLayoutOptions, ModelKind, open_model, read_checkpoint, read_ply};
use ftgsxr_io::ModelBytes;
use ftgsxr_render::load_splat_asset;
use ftgsxr_testkit::{fixtures_dir, probe};

fn opts() -> GpuLayoutOptions {
    GpuLayoutOptions::default()
}

#[test]
fn exchange_bytes_load_without_a_hint() {
    let (model, _) = probe("P1");
    let bytes = std::fs::read(fixtures_dir().join("probes/P1.safetensors")).unwrap();
    let len = bytes.len() as u64;
    let a = load_splat_asset(&ModelBytes::Owned(bytes), None, &opts(), None).unwrap();
    assert_eq!(a.info.kind, ModelKind::ExchangeFile);
    assert_eq!(a.info.n, model.len() as u64);
    assert_eq!(a.info.sh_degree, model.sh_degree);
    assert_eq!(a.info.file_bytes, len);
    assert_eq!(a.identity, model.identity);
    assert_eq!(a.bounds, model.bounds(8));
}

#[test]
fn ply_bytes_load_without_a_hint() {
    let path = fixtures_dir().join("checkpoint/tiny.tau0.ply");
    let bytes = std::fs::read(&path).unwrap();
    let model = read_ply(&bytes).unwrap();
    let a = load_splat_asset(&ModelBytes::Owned(bytes), None, &opts(), None).unwrap();
    assert_eq!(a.info.kind, ModelKind::Ply);
    assert_eq!(a.info.n, model.len() as u64);
    assert_eq!(a.identity, model.identity);
}

#[test]
fn a_checkpoint_loads_by_path() {
    let dir = fixtures_dir().join("checkpoint/tiny");
    let model = read_checkpoint(&dir).unwrap();
    let a = load_splat_asset(&ModelBytes::Checkpoint(dir.clone()), None, &opts(), None).unwrap();
    assert!(matches!(a.info.kind, ModelKind::Checkpoint { .. }));
    assert_eq!(a.info.n, model.len() as u64);
    assert_eq!(a.identity, model.identity);
    // Q-76: the size of the file the identity hashes.
    let params = std::fs::metadata(dir.join("params.safetensors"))
        .unwrap()
        .len();
    assert_eq!(a.info.file_bytes, params);
}

#[test]
fn reduce_to_keeps_that_many_rows() {
    let path = fixtures_dir().join("checkpoint/tiny.safetensors");
    let model = open_model(&path).unwrap();
    assert!(model.len() > 2);
    let k = model.len() / 2;
    let bytes = std::fs::read(&path).unwrap();
    let a = load_splat_asset(&ModelBytes::Owned(bytes), None, &opts(), Some(k)).unwrap();
    assert_eq!(a.info.n, k as u64);
    assert_eq!(a.identity, model.identity);
}
