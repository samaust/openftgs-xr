// SPDX-License-Identifier: Apache-2.0

//! The exchange-file writer of `import` and `export` (§1.2, §1.7, openftgs §2.2; Q-65,
//! Q-T109.1, #109): a model written by `write_exchange` reads back through `read_exchange`
//! with equal arrays, time span and metadata, and the producer `openftgs-xr <version>`.

use std::fs;
use std::path::PathBuf;

use ftgsxr_core::formats::{encode_exchange, write_exchange};
use ftgsxr_core::{Model, ModelKind, RenderConventions, TimeSpan, read_exchange, sha256};
use ftgsxr_testkit::{random_model, scratch_dir};

/// A scratch directory of this test binary's own, emptied first (Q-62).
fn scratch(name: &str) -> PathBuf {
    scratch_dir(env!("CARGO_TARGET_TMPDIR"), env!("CARGO_CRATE_NAME"), name)
}

/// A two-row degree-1 model with a background and a producer of its own, worked by hand.
fn two_rows() -> Model {
    Model {
        position: vec![[0.5, -1.0, 2.0], [0.0, 0.25, -0.75]],
        time_center: vec![0.3, 1.6],
        duration: vec![0.2, 5.0],
        velocity: vec![[0.1, 0.0, -0.2], [0.0, 0.0, 0.0]],
        scale: vec![[0.05, 0.1, 0.02], [0.3, 0.3, 0.3]],
        rotation: vec![[1.0, 0.0, 0.0, 0.0], [0.0, 0.6, 0.8, 0.0]],
        opacity: vec![0.9, 0.125],
        sh: (0..2 * 4 * 3).map(|i| i as f32 * 0.25 - 1.0).collect(),
        sh_degree: 1,
        time: TimeSpan {
            tau0: 0.1,
            tau1: 1.9,
            time_unit: "s".to_owned(),
        },
        conventions: RenderConventions {
            background: [1.0, 0.5, 0.25],
            eps2d: 0.25,
            transmittance_stop: None,
            ..RenderConventions::default()
        },
        producer: "openftgs 0.3.1".to_owned(),
        identity: [0; 32],
        kind: ModelKind::Ply,
    }
}

/// The producer the writer records (Q-65).
fn our_producer() -> String {
    format!("openftgs-xr {}", env!("CARGO_PKG_VERSION"))
}

/// `written` equals `model` in every array, the degree, the time span and the conventions;
/// its producer is the writer's, its kind an exchange file, its identity the file's SHA-256.
fn assert_round_trip(model: &Model, written: &Model, bytes: &[u8], what: &str) {
    let expected = Model {
        producer: our_producer(),
        identity: sha256(bytes),
        kind: ModelKind::ExchangeFile,
        ..model.clone()
    };
    assert_eq!(written, &expected, "{what}");
}

#[test]
fn write_exchange_reads_back_equal() {
    let dir = scratch("round_trip");
    let mut models = vec![("two rows".to_owned(), two_rows())];
    for degree in 0..=3u8 {
        models.push((
            format!("random degree {degree}"),
            random_model(11, 500, degree),
        ));
    }
    let mut single = random_model(3, 50, 2);
    single.time.tau1 = single.time.tau0;
    single.time_center.fill(single.time.tau0);
    single.velocity.fill([0.0; 3]);
    single.validate().expect("a valid single-timestamp model");
    models.push(("single timestamp".to_owned(), single));

    for (what, model) in &models {
        let path = dir.join("model.safetensors");
        write_exchange(&path, model).unwrap_or_else(|e| panic!("{what}: {e}"));
        let bytes = fs::read(&path).unwrap();
        let written = read_exchange(&bytes).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert_round_trip(model, &written, &bytes, what);
    }
}

/// The names in `dir`, sorted.
fn listing(dir: &std::path::Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// The writer validates first, as openftgs's `io.write_exchange` does (Q-T109.1): an invalid
/// model is refused with §1.2's code, the file already at the path is kept, and no
/// temporary is left.
#[test]
fn write_exchange_refuses_an_invalid_model() {
    let dir = scratch("invalid");
    let path = dir.join("model.safetensors");
    fs::write(&path, b"the old file").unwrap();
    let mut model = two_rows();
    model.opacity[1] = 1.0;
    let err = write_exchange(&path, &model).unwrap_err();
    assert_eq!(err.code(), "E-FILE-VALUE", "{err}");
    let mut model = two_rows();
    model.sh.pop();
    assert_eq!(
        write_exchange(&path, &model).unwrap_err().code(),
        "E-FILE-SH"
    );
    assert_eq!(fs::read(&path).unwrap(), b"the old file");
    assert_eq!(listing(&dir), ["model.safetensors"]);
}

/// The write replaces an existing file through a temporary sibling and leaves nothing else;
/// a directory that does not exist is E-IO and leaves no file.
#[test]
fn write_exchange_replaces_atomically() {
    let dir = scratch("replace");
    let path = dir.join("model.safetensors");
    fs::write(&path, b"the old file").unwrap();
    write_exchange(&path, &two_rows()).unwrap();
    assert!(read_exchange(&fs::read(&path).unwrap()).is_ok());
    assert_eq!(listing(&dir), ["model.safetensors"]);

    let missing = dir.join("no such directory").join("model.safetensors");
    let err = write_exchange(&missing, &two_rows()).unwrap_err();
    assert_eq!(err.code(), "E-IO", "{err}");
    assert_eq!(listing(&dir), ["model.safetensors"]);
}

/// `encode_exchange` (Q-T109.1): the same file in memory with the model's own producer,
/// written unchecked, so a test can make a file the reader refuses.
#[test]
fn encode_exchange_keeps_the_producer_and_writes_unchecked() {
    let model = two_rows();
    let bytes = encode_exchange(&model);
    let expected = Model {
        identity: sha256(&bytes),
        kind: ModelKind::ExchangeFile,
        ..model.clone()
    };
    assert_eq!(read_exchange(&bytes).unwrap(), expected);
    assert_eq!(encode_exchange(&model), bytes, "deterministic");

    let mut invalid = two_rows();
    invalid.duration[0] = -1.0;
    let err = read_exchange(&encode_exchange(&invalid)).unwrap_err();
    assert_eq!(err.code(), "E-FILE-VALUE", "{err}");
}
