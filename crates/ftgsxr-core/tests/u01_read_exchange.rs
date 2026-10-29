// SPDX-License-Identifier: Apache-2.0

//! U-1, the reader's part (§1.2 `ftgs-if`, `cpu` lane): the committed probe files load with
//! every field, dtype, shape and N; `sh_degree`, `tau0`, `tau1`, `time_unit` and `producer`
//! parsed; `render_conventions` recovered with openftgs's values; an extra tensor and an
//! extra metadata key ignored. The oracle is an independent parse of each file written here
//! from §1.2's container description (serde_json and the little-endian bytes).

use std::fs;
use std::path::PathBuf;

use ftgsxr_core::{
    Model, ModelKind, RenderConventions, SAFETENSORS_PREFIX, header_len, open_model, read_exchange,
    read_info,
};
use ftgsxr_testkit::{SafetensorsBuilder, exchange_bytes, fixtures_dir, random_model};
use serde_json::Value;

/// The probe models: every `probes/*.safetensors` but the renders files.
fn probe_paths() -> Vec<PathBuf> {
    let dir = fixtures_dir().join("probes");
    let mut paths: Vec<PathBuf> = fs::read_dir(&dir)
        .expect("the probes directory")
        .map(|e| e.expect("a directory entry").path())
        .filter(|p| {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            name.ends_with(".safetensors") && !name.ends_with(".renders.safetensors")
        })
        .collect();
    paths.sort();
    assert!(paths.len() >= 18, "the probes of tests/fixtures/README.md");
    paths
}

/// A file split by hand: its header JSON and its data section.
struct Parsed {
    header: serde_json::Map<String, Value>,
    data: Vec<u8>,
}

fn parse(bytes: &[u8]) -> Parsed {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    Parsed {
        header: header.as_object().unwrap().clone(),
        data: bytes[8 + n..].to_vec(),
    }
}

impl Parsed {
    fn meta(&self, key: &str) -> Option<&str> {
        self.header["__metadata__"].get(key).and_then(Value::as_str)
    }

    /// The tensor's dtype, shape and bytes.
    fn tensor(&self, key: &str) -> (String, Vec<usize>, &[u8]) {
        let t = &self.header[key];
        let shape = t["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let o = t["data_offsets"].as_array().unwrap();
        let (s, e) = (
            o[0].as_u64().unwrap() as usize,
            o[1].as_u64().unwrap() as usize,
        );
        (
            t["dtype"].as_str().unwrap().to_owned(),
            shape,
            &self.data[s..e],
        )
    }

    fn f32s(&self, key: &str, dtype_shape: (&str, &[usize])) -> Vec<f32> {
        let (dtype, shape, bytes) = self.tensor(key);
        assert_eq!((dtype.as_str(), shape.as_slice()), dtype_shape, "{key}");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    fn f64s(&self, key: &str, dtype_shape: (&str, &[usize])) -> Vec<f64> {
        let (dtype, shape, bytes) = self.tensor(key);
        assert_eq!((dtype.as_str(), shape.as_slice()), dtype_shape, "{key}");
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect()
    }
}

fn flat<const K: usize>(rows: &[[f32; K]]) -> Vec<f32> {
    rows.iter().flatten().copied().collect()
}

/// Compares float32 arrays bit for bit, so −0.0 and NaN payloads count.
fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|x| x.to_bits()).collect()
}

/// Checks a model read from `bytes` against the independent parse, field by field.
fn check_against_file(model: &Model, bytes: &[u8], what: &str) {
    let p = parse(bytes);
    let n = model.len();
    let l = p.meta("sh_degree").unwrap().parse::<u8>().unwrap();
    let k = (usize::from(l) + 1).pow(2);
    assert_eq!(p.tensor("position").1[0], n, "{what}: N");
    assert_eq!(model.sh_degree, l, "{what}");
    assert_eq!(
        model.time.tau0,
        p.meta("tau0").unwrap().parse::<f64>().unwrap(),
        "{what}"
    );
    assert_eq!(
        model.time.tau1,
        p.meta("tau1").unwrap().parse::<f64>().unwrap(),
        "{what}"
    );
    assert_eq!(model.time.time_unit, p.meta("time_unit").unwrap(), "{what}");
    assert_eq!(model.producer, p.meta("producer").unwrap_or(""), "{what}");
    assert_eq!(
        bits(&flat(&model.position)),
        bits(&p.f32s("position", ("F32", &[n, 3]))),
        "{what}"
    );
    assert_eq!(
        model.time_center,
        p.f64s("time_center", ("F64", &[n])),
        "{what}"
    );
    assert_eq!(model.duration, p.f64s("duration", ("F64", &[n])), "{what}");
    assert_eq!(
        bits(&flat(&model.velocity)),
        bits(&p.f32s("velocity", ("F32", &[n, 3]))),
        "{what}"
    );
    assert_eq!(
        bits(&flat(&model.scale)),
        bits(&p.f32s("scale", ("F32", &[n, 3]))),
        "{what}"
    );
    // openftgs writes unit quaternions with the canonical sign, which the reader keeps.
    assert_eq!(
        bits(&flat(&model.rotation)),
        bits(&p.f32s("rotation", ("F32", &[n, 4]))),
        "{what}"
    );
    assert_eq!(
        bits(&model.opacity),
        bits(&p.f32s("opacity", ("F32", &[n]))),
        "{what}"
    );
    assert_eq!(
        bits(&model.sh),
        bits(&p.f32s("sh", ("F32", &[n, k, 3]))),
        "{what}"
    );
    assert_eq!(model.kind, ModelKind::ExchangeFile, "{what}");
}

/// The conventions openftgs writes: its defaults with the model's background (openftgs
/// D-73, D-84), here read from the block by hand.
fn expected_conventions(bytes: &[u8]) -> RenderConventions {
    let p = parse(bytes);
    let block: Value = serde_json::from_str(p.meta("render_conventions").unwrap()).unwrap();
    let bg: Vec<f32> = block["background"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    RenderConventions {
        background: [bg[0], bg[1], bg[2]],
        ..RenderConventions::default()
    }
}

#[test]
fn u01_probes_load_every_field() {
    for path in probe_paths() {
        let bytes = fs::read(&path).unwrap();
        let what = path.display().to_string();
        let model = open_model(&path).unwrap_or_else(|e| panic!("{what}: {e}"));
        check_against_file(&model, &bytes, &what);
        assert_eq!(model.conventions, expected_conventions(&bytes), "{what}");
        assert_eq!(model, read_exchange(&bytes).unwrap(), "{what}");
    }
}

#[test]
fn u01_probe_metadata_values() {
    let p1 = open_model(&fixtures_dir().join("probes/P1.safetensors")).unwrap();
    assert_eq!(p1.time.tau0, 0.0);
    assert_eq!(p1.time.tau1, 1.9);
    assert_eq!(p1.time.time_unit, "s");
    assert_eq!(p1.producer, "openftgs-xr tools/fixtures");
    assert_eq!(p1.conventions, RenderConventions::default());
    let white = open_model(&fixtures_dir().join("probes/R2_white_background.safetensors")).unwrap();
    assert_eq!(white.conventions.background, [1.0, 1.0, 1.0]);
    let deg3 = open_model(&fixtures_dir().join("probes/R2_overlap_deg3.safetensors")).unwrap();
    assert_eq!(deg3.sh_degree, 3);
    assert_eq!(deg3.sh.len(), deg3.len() * 16 * 3);
}

/// The file's tensors and metadata, rebuilt through the testkit's builder with `extra`
/// additions.
fn rebuild(
    bytes: &[u8],
    extra_meta: &[(&str, &str)],
    extra_tensors: &[(&str, &[usize])],
) -> Vec<u8> {
    let p = parse(bytes);
    let mut b = SafetensorsBuilder::new();
    for (key, value) in p.header["__metadata__"].as_object().unwrap() {
        b = b.metadata(key, value.as_str().unwrap());
    }
    for (key, value) in extra_meta {
        b = b.metadata(key, value);
    }
    for key in p.header.keys().filter(|k| *k != "__metadata__") {
        let (dtype, shape, data) = p.tensor(key);
        b = b.tensor(key, &dtype, &shape, data.to_vec());
    }
    for (key, shape) in extra_tensors {
        let count: usize = shape.iter().product();
        b = b.tensor(key, "F32", shape, vec![0x7f; 4 * count]);
    }
    b.build()
}

#[test]
fn u01_extra_tensor_and_metadata_key_ignored() {
    let bytes = fs::read(fixtures_dir().join("probes/P1.safetensors")).unwrap();
    let plain = read_exchange(&bytes).unwrap();
    // An extra tensor full of NaN (0x7f7f7f7f) and of another shape, and an unknown key.
    let extra = rebuild(
        &bytes,
        &[("comment", "not a key of ftgs-if")],
        &[("extra_tensor", &[5, 7])],
    );
    let mut read = read_exchange(&extra).unwrap();
    assert_ne!(read.identity, plain.identity);
    read.identity = plain.identity;
    assert_eq!(read, plain);
}

#[test]
fn u01_header_only_information() {
    for path in probe_paths() {
        let bytes = fs::read(&path).unwrap();
        let n = header_len(bytes[..SAFETENSORS_PREFIX].try_into().unwrap()) as usize;
        let header = &bytes[SAFETENSORS_PREFIX..SAFETENSORS_PREFIX + n];
        let info = read_info(header, bytes.len() as u64).unwrap();
        let model = read_exchange(&bytes).unwrap();
        assert_eq!(info.n, model.len() as u64);
        assert_eq!(info.sh_degree, model.sh_degree);
        assert_eq!(info.time, model.time);
        assert_eq!(info.producer, model.producer);
        assert_eq!(info.conventions, model.conventions);
        assert_eq!(info.kind, ModelKind::ExchangeFile);
        assert_eq!(info.file_bytes, bytes.len() as u64);
        // A header that does not describe a file of this size is refused.
        let err = read_info(header, bytes.len() as u64 + 1).unwrap_err();
        assert_eq!(err.code(), "E-FILE-FORMAT");
    }
}

#[test]
fn u01_random_models_round_trip() {
    for (seed, n, degree) in [(1, 1, 0), (2, 17, 1), (3, 64, 2), (4, 300, 3)] {
        let model = random_model(seed, n, degree);
        assert_eq!(read_exchange(&exchange_bytes(&model)).unwrap(), model);
    }
}

#[test]
fn u01_a_large_model_round_trips() {
    // Above the readers' parallel threshold.
    let model = random_model(5, 70_000, 1);
    assert_eq!(read_exchange(&exchange_bytes(&model)).unwrap(), model);
}
