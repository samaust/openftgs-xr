// SPDX-License-Identifier: Apache-2.0

//! `load_splat_asset` (§2.1, §1.2, D-51, Q-99) on the committed fixtures: an exchange
//! file and a PLY snapshot from bytes (the kind sniffed), a checkpoint by path, and the
//! explicit reduction (D-13). Each asset's information, box and identity agree with the
//! core readers'. Then each kind hint, the refusals (a checkpoint hint on owned bytes,
//! malformed bytes) and the reduction keeping the rows of highest importance (§1.2 load
//! option 4) or, at a budget of N or more, every row.

use ftgsxr_core::{
    GpuLayoutOptions, Model, ModelKind, open_model, read_checkpoint, read_exchange, read_ply,
};
use ftgsxr_io::ModelBytes;
use ftgsxr_render::{RenderError, load_splat_asset};
use ftgsxr_testkit::{exchange_bytes, fixtures_dir, probe, random_model};

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
    assert_eq!(a.bounds, model.bounds());
    assert_eq!(a.content_bounds, model.content_bounds());
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

#[test]
fn the_exchange_hint_loads_an_exchange_file() {
    let bytes = std::fs::read(fixtures_dir().join("probes/P1.safetensors")).unwrap();
    let model = read_exchange(&bytes).unwrap();
    let a = load_splat_asset(
        &ModelBytes::Owned(bytes),
        Some(ModelKind::ExchangeFile),
        &opts(),
        None,
    )
    .unwrap();
    assert_eq!(a.info.kind, ModelKind::ExchangeFile);
    assert_eq!(a.info.n, model.len() as u64);
    assert_eq!(a.identity, model.identity);
}

#[test]
fn the_ply_hint_loads_a_ply_snapshot() {
    let bytes = std::fs::read(fixtures_dir().join("checkpoint/tiny.tau0.ply")).unwrap();
    let model = read_ply(&bytes).unwrap();
    let a = load_splat_asset(
        &ModelBytes::Owned(bytes),
        Some(ModelKind::Ply),
        &opts(),
        None,
    )
    .unwrap();
    assert_eq!(a.info.kind, ModelKind::Ply);
    assert_eq!(a.info.n, model.len() as u64);
    assert_eq!(a.identity, model.identity);
}

#[test]
fn the_hint_decides_the_reader() {
    // Exchange bytes read as a PLY snapshot are the PLY reader's refusal, not a sniffed load.
    let bytes = std::fs::read(fixtures_dir().join("probes/P1.safetensors")).unwrap();
    let e = read_ply(&bytes).unwrap_err();
    let err = load_splat_asset(
        &ModelBytes::Owned(bytes),
        Some(ModelKind::Ply),
        &opts(),
        None,
    )
    .err()
    .unwrap();
    assert!(matches!(err, RenderError::Core(_)), "{err}");
    assert_eq!(err.code(), e.code());
}

#[test]
fn a_checkpoint_hint_on_owned_bytes_is_a_format_error() {
    // D-51: a checkpoint is a directory and loads by path.
    let bytes = std::fs::read(fixtures_dir().join("checkpoint/tiny/params.safetensors")).unwrap();
    let hint = match open_model(&fixtures_dir().join("checkpoint/tiny"))
        .unwrap()
        .kind
    {
        k @ ModelKind::Checkpoint { .. } => k,
        k => panic!("the fixture is a checkpoint, not {k:?}"),
    };
    let err = load_splat_asset(&ModelBytes::Owned(bytes), Some(hint), &opts(), None)
        .err()
        .unwrap();
    assert!(matches!(err, RenderError::Core(_)), "{err}");
    assert_eq!(err.code(), "E-FILE-FORMAT");
}

#[test]
fn malformed_bytes_give_the_core_error() {
    let bytes = b"not a model file at all".to_vec();
    let e = read_exchange(&bytes).unwrap_err();
    let err = load_splat_asset(&ModelBytes::Owned(bytes), None, &opts(), None)
        .err()
        .unwrap();
    match &err {
        RenderError::Core(inner) => assert_eq!(inner.code(), e.code()),
        other => panic!("expected RenderError::Core, got {other}"),
    }
    assert_eq!(err.code(), e.code());
    assert!(err.code().starts_with("E-"), "{}", err.code());
}

#[test]
fn reduce_to_n_or_more_keeps_the_full_model() {
    let path = fixtures_dir().join("checkpoint/tiny.safetensors");
    let model = open_model(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let full = load_splat_asset(&ModelBytes::Owned(bytes.clone()), None, &opts(), None).unwrap();
    for budget in [model.len(), model.len() + 5] {
        let a = load_splat_asset(
            &ModelBytes::Owned(bytes.clone()),
            None,
            &opts(),
            Some(budget),
        )
        .unwrap();
        assert_eq!(a.info.n, model.len() as u64);
        assert_eq!(a.bounds, full.bounds);
        assert_eq!(a.content_bounds, full.content_bounds);
        assert_eq!(a.identity, model.identity);
    }
}

/// Eight static rows on the x axis, row i at x = i, equal scales, so a row's importance
/// (opacity·scale_x·scale_y·scale_z, §1.2 option 4) is its opacity; rows 3 and 4 are the
/// two most important.
fn ranked_model() -> Model {
    let mut m = random_model(7, 8, 0);
    let opacity = [0.30, 0.20, 0.40, 0.95, 0.90, 0.35, 0.25, 0.50];
    for (i, &o) in opacity.iter().enumerate() {
        m.position[i] = [i as f32, 0.0, 0.0];
        m.velocity[i] = [0.0; 3];
        m.scale[i] = [0.01; 3];
        m.opacity[i] = o;
        m.time_center[i] = 0.95;
        m.duration[i] = 10.0;
    }
    m
}

/// The rows `rows` of `m`, in file order.
fn rows_of(m: &Model, rows: &[usize]) -> Model {
    let k = (usize::from(m.sh_degree) + 1).pow(2) * 3;
    Model {
        position: rows.iter().map(|&i| m.position[i]).collect(),
        time_center: rows.iter().map(|&i| m.time_center[i]).collect(),
        duration: rows.iter().map(|&i| m.duration[i]).collect(),
        velocity: rows.iter().map(|&i| m.velocity[i]).collect(),
        scale: rows.iter().map(|&i| m.scale[i]).collect(),
        rotation: rows.iter().map(|&i| m.rotation[i]).collect(),
        opacity: rows.iter().map(|&i| m.opacity[i]).collect(),
        sh: rows
            .iter()
            .flat_map(|&i| m.sh[i * k..(i + 1) * k].iter().copied())
            .collect(),
        ..m.clone()
    }
}

#[test]
fn reduce_to_keeps_the_rows_of_highest_importance() {
    let model = ranked_model();
    let bytes = exchange_bytes(&model);
    let identity = read_exchange(&bytes).unwrap().identity;
    let a = load_splat_asset(&ModelBytes::Owned(bytes), None, &opts(), Some(2)).unwrap();
    assert_eq!(a.info.n, 2);
    // The reduced model's box is that of rows 3 and 4 alone: x in [3, 4] widened by 3 σ.
    let top = rows_of(&model, &[3, 4]);
    let expected = load_splat_asset(
        &ModelBytes::Owned(exchange_bytes(&top)),
        None,
        &opts(),
        None,
    )
    .unwrap();
    assert_eq!(a.bounds, expected.bounds);
    assert!(
        (a.bounds.min.x - (3.0 - 0.03)).abs() < 1e-5,
        "{:?}",
        a.bounds
    );
    assert!(
        (a.bounds.max.x - (4.0 + 0.03)).abs() < 1e-5,
        "{:?}",
        a.bounds
    );
    // The identity stays the file's (D-40).
    assert_eq!(a.identity, identity);
}
