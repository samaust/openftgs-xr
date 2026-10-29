// SPDX-License-Identifier: Apache-2.0

//! `exchange_bytes` (Q-1): a `Model` as a `ftgs-if` v1 file, §1.2's table and metadata.
//! The loader is not used: the bytes are parsed here by hand, from §1.2's description of
//! the container, and every tensor decoded back.

use ftgsxr_core::{Model, ModelKind, RenderConventions, TimeSpan};
use ftgsxr_testkit::{exchange_bytes, random_model};
use serde_json::{Map, Value};

/// A two-row degree-1 model, worked by hand.
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
            tau0: 0.0,
            tau1: 1.9,
            time_unit: "s".to_owned(),
        },
        conventions: RenderConventions {
            background: [1.0, 0.5, 0.25],
            ..RenderConventions::default()
        },
        producer: "a \"quoted\" producer".to_owned(),
        identity: [0; 32],
        kind: ModelKind::ExchangeFile,
    }
}

/// (header length, header JSON, data) per §1.2's container.
fn split(bytes: &[u8]) -> (usize, Map<String, Value>, &[u8]) {
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let text = std::str::from_utf8(&bytes[8..8 + len]).expect("a UTF-8 header");
    assert!(text.starts_with('{'), "{text:?}");
    let Value::Object(map) = serde_json::from_str(text).expect("a JSON header") else {
        panic!("the header is not an object");
    };
    (len, map, &bytes[8 + len..])
}

/// One tensor's dtype, shape and bytes, with its offsets checked against the data.
fn tensor<'a>(map: &Map<String, Value>, data: &'a [u8], key: &str) -> (String, Vec<u64>, &'a [u8]) {
    let entry = &map[key];
    let dtype = entry["dtype"].as_str().unwrap().to_owned();
    let shape: Vec<u64> = entry["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    let offsets: Vec<usize> = entry["data_offsets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    (dtype, shape, &data[offsets[0]..offsets[1]])
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    let (chunks, rest) = bytes.as_chunks::<4>();
    assert!(rest.is_empty());
    chunks.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

fn f64s(bytes: &[u8]) -> Vec<f64> {
    let (chunks, rest) = bytes.as_chunks::<8>();
    assert!(rest.is_empty());
    chunks.iter().map(|c| f64::from_le_bytes(*c)).collect()
}

const KEYS: [&str; 8] = [
    "position",
    "time_center",
    "duration",
    "velocity",
    "scale",
    "rotation",
    "opacity",
    "sh",
];

#[test]
fn exchange_bytes_writes_the_metadata_of_section_1_2() {
    let model = two_rows();
    let (_, map, _) = split(&exchange_bytes(&model));
    let meta = map["__metadata__"].as_object().expect("__metadata__");
    let get = |k: &str| {
        meta[k]
            .as_str()
            .unwrap_or_else(|| panic!("{k} is a string"))
    };
    assert_eq!(get("format"), "ftgs-if");
    assert_eq!(get("version"), "1");
    assert_eq!(get("sh_degree"), "1");
    assert_eq!(get("time_unit"), "s");
    assert_eq!(get("tau0"), "0.0");
    assert_eq!(get("tau1"), "1.9");
    assert_eq!(get("producer"), "a \"quoted\" producer");
    let conventions = get("render_conventions");
    assert_eq!(conventions, model.conventions.to_json());
    assert_eq!(
        RenderConventions::from_json(conventions).unwrap(),
        model.conventions
    );
    assert_eq!(meta.len(), 8, "{meta:?}");
}

#[test]
fn exchange_bytes_writes_the_tensors_of_section_1_2() {
    let model = two_rows();
    let bytes = exchange_bytes(&model);
    let (len, map, data) = split(&bytes);
    assert_eq!(map.len(), 9, "the eight tensors and __metadata__");

    let expect = [
        ("position", "F32", vec![2, 3]),
        ("time_center", "F64", vec![2]),
        ("duration", "F64", vec![2]),
        ("velocity", "F32", vec![2, 3]),
        ("scale", "F32", vec![2, 3]),
        ("rotation", "F32", vec![2, 4]),
        ("opacity", "F32", vec![2]),
        ("sh", "F32", vec![2, 4, 3]),
    ];
    let mut covered = Vec::new();
    for (key, dtype, shape) in expect {
        let (got_dtype, got_shape, _) = tensor(&map, data, key);
        assert_eq!((got_dtype.as_str(), got_shape), (dtype, shape), "{key}");
        let offsets: Vec<usize> = map[key]["data_offsets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let size = if dtype == "F64" { 8 } else { 4 };
        assert_eq!(
            (8 + len + offsets[0]) % size,
            0,
            "{key} is aligned in the file"
        );
        covered.push((offsets[0], offsets[1]));
    }
    covered.sort();
    assert_eq!(covered[0].0, 0);
    assert!(
        covered.windows(2).all(|w| w[0].1 == w[1].0),
        "no holes: {covered:?}"
    );
    assert_eq!(covered.last().unwrap().1, data.len());

    let flat3 = |v: &[[f32; 3]]| v.iter().flatten().copied().collect::<Vec<_>>();
    assert_eq!(
        f32s(tensor(&map, data, "position").2),
        flat3(&model.position)
    );
    assert_eq!(f64s(tensor(&map, data, "time_center").2), model.time_center);
    assert_eq!(f64s(tensor(&map, data, "duration").2), model.duration);
    assert_eq!(
        f32s(tensor(&map, data, "velocity").2),
        flat3(&model.velocity)
    );
    assert_eq!(f32s(tensor(&map, data, "scale").2), flat3(&model.scale));
    assert_eq!(
        f32s(tensor(&map, data, "rotation").2),
        model.rotation.iter().flatten().copied().collect::<Vec<_>>()
    );
    assert_eq!(f32s(tensor(&map, data, "opacity").2), model.opacity);
    assert_eq!(f32s(tensor(&map, data, "sh").2), model.sh);
}

/// The acceptance criterion: a random model's file carries §1.2's keys and metadata.
#[test]
fn exchange_bytes_of_a_random_model_carries_every_key() {
    for degree in 0..=3u8 {
        let model = random_model(7, 1000, degree);
        let bytes = exchange_bytes(&model);
        let (_, map, data) = split(&bytes);
        for key in KEYS {
            let (_, shape, bytes) = tensor(&map, data, key);
            assert_eq!(shape[0], 1000, "{key}");
            assert!(!bytes.is_empty(), "{key}");
        }
        let k = (u64::from(degree) + 1).pow(2);
        assert_eq!(tensor(&map, data, "sh").1, vec![1000, k, 3]);
        let meta = map["__metadata__"].as_object().unwrap();
        assert_eq!(meta["sh_degree"], Value::String(degree.to_string()));
        for key in [
            "format",
            "version",
            "time_unit",
            "tau0",
            "tau1",
            "producer",
            "render_conventions",
        ] {
            assert!(meta.contains_key(key), "{key}");
        }
    }
}

#[test]
fn exchange_bytes_ignores_identity_and_kind() {
    let a = two_rows();
    let mut b = two_rows();
    b.identity = [7; 32];
    b.kind = ModelKind::Ply;
    assert_eq!(exchange_bytes(&a), exchange_bytes(&b));
}
