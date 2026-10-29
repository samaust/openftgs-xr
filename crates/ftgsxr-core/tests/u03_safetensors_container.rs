// SPDX-License-Identifier: Apache-2.0

//! U-3 (§1.2 the safetensors container, `cpu` lane): a padded header; an unaligned data
//! offset (the copy path) against an aligned one, giving identical models; a header length
//! above 100 MB refused.

use ftgsxr_core::{Model, SAFETENSORS_PREFIX, header_len, read_exchange};
use ftgsxr_testkit::{SafetensorsBuilder, random_model};

fn f32_bytes(xs: impl IntoIterator<Item = f32>) -> Vec<u8> {
    xs.into_iter().flat_map(f32::to_le_bytes).collect()
}

fn f64_bytes(xs: impl IntoIterator<Item = f64>) -> Vec<u8> {
    xs.into_iter().flat_map(f64::to_le_bytes).collect()
}

/// The model's exchange file with the header changed by `shape`. Float32 tensors come
/// first, then a float64 tensor, so a misaligned data section misaligns both kinds.
fn file(m: &Model, shape: impl FnOnce(SafetensorsBuilder) -> SafetensorsBuilder) -> Vec<u8> {
    let n = m.len();
    let k = (usize::from(m.sh_degree) + 1).pow(2);
    let b = SafetensorsBuilder::new()
        .metadata("format", "ftgs-if")
        .metadata("version", "1")
        .metadata("sh_degree", &m.sh_degree.to_string())
        .metadata("time_unit", &m.time.time_unit)
        .metadata("tau0", &format!("{:?}", m.time.tau0))
        .metadata("tau1", &format!("{:?}", m.time.tau1))
        .metadata("producer", &m.producer)
        .metadata("render_conventions", &m.conventions.to_json())
        .tensor("opacity", "F32", &[n], f32_bytes(m.opacity.iter().copied()))
        .tensor(
            "position",
            "F32",
            &[n, 3],
            f32_bytes(m.position.iter().flatten().copied()),
        )
        .tensor(
            "time_center",
            "F64",
            &[n],
            f64_bytes(m.time_center.iter().copied()),
        )
        .tensor(
            "rotation",
            "F32",
            &[n, 4],
            f32_bytes(m.rotation.iter().flatten().copied()),
        )
        .tensor(
            "duration",
            "F64",
            &[n],
            f64_bytes(m.duration.iter().copied()),
        )
        .tensor(
            "scale",
            "F32",
            &[n, 3],
            f32_bytes(m.scale.iter().flatten().copied()),
        )
        .tensor("sh", "F32", &[n, k, 3], f32_bytes(m.sh.iter().copied()))
        .tensor(
            "velocity",
            "F32",
            &[n, 3],
            f32_bytes(m.velocity.iter().flatten().copied()),
        );
    shape(b).build()
}

/// The model read from `bytes`, without its identity (the bytes differ by construction).
/// The bytes are first copied to an 8-byte-aligned address, so each tensor is aligned in
/// memory exactly when it is aligned in the file.
fn read(bytes: &[u8]) -> Model {
    let mut words = vec![0u64; bytes.len().div_ceil(8)];
    let buffer = &mut bytemuck::cast_slice_mut::<u64, u8>(&mut words)[..bytes.len()];
    buffer.copy_from_slice(bytes);
    let mut m = read_exchange(buffer).unwrap();
    m.identity = [0; 32];
    m
}

/// Each tensor of the file `bytes` as (name, dtype size in bytes, offset of its data from
/// the file's first byte), from the header's `dtype` and `data_offsets` (§1.2).
fn tensor_offsets(bytes: &[u8]) -> Vec<(String, usize, usize)> {
    let n = header_len(bytes[..SAFETENSORS_PREFIX].try_into().unwrap()) as usize;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[SAFETENSORS_PREFIX..SAFETENSORS_PREFIX + n]).unwrap();
    header
        .iter()
        .filter(|(name, _)| *name != "__metadata__")
        .map(|(name, entry)| {
            let size = match entry["dtype"].as_str().unwrap() {
                "F32" => 4,
                "F64" => 8,
                other => panic!("{name}: dtype {other}"),
            };
            let begin = entry["data_offsets"][0].as_u64().unwrap() as usize;
            (name.clone(), size, SAFETENSORS_PREFIX + n + begin)
        })
        .collect()
}

fn expected(m: &Model) -> Model {
    let mut m = m.clone();
    m.identity = [0; 32];
    m
}

#[test]
fn u03_padded_header() {
    let m = random_model(3, 33, 2);
    let plain = file(&m, |b| b);
    for spaces in [1, 7, 64, 1000] {
        let padded = file(&m, |b| b.pad_header(spaces));
        assert!(padded.len() >= plain.len() + spaces);
        let n = header_len(padded[..SAFETENSORS_PREFIX].try_into().unwrap()) as usize;
        assert_eq!(padded[SAFETENSORS_PREFIX + n - 1], b' ');
        assert_eq!(read(&padded), expected(&m), "{spaces} spaces");
    }
    assert_eq!(read(&plain), expected(&m));
}

#[test]
fn u03_unaligned_data_against_aligned() {
    let m = random_model(4, 51, 3);
    let aligned = file(&m, |b| b);
    let offsets = tensor_offsets(&aligned);
    assert_eq!(offsets.len(), 8);
    for (name, size, offset) in &offsets {
        assert_eq!(
            offset % size,
            0,
            "{name} aligned for its dtype in the reference"
        );
    }
    let reference = read(&aligned);
    assert_eq!(reference, expected(&m));
    for shift in 1..8 {
        let unaligned = file(&m, |b| b.misalign_data(shift));
        for (name, size, offset) in tensor_offsets(&unaligned) {
            // Every float64 tensor is misaligned; every float32 one too unless the shift
            // is 4.
            let misaligned = offset % size != 0;
            assert_eq!(misaligned, size == 8 || shift != 4, "{name}, shift {shift}");
        }
        assert_eq!(read(&unaligned), reference, "data shifted by {shift}");
    }
}

#[test]
fn u03_header_above_100_mb_refused() {
    for len in [100_000_001u64, u64::MAX] {
        let mut bytes = len.to_le_bytes().to_vec();
        bytes.extend_from_slice(b"{}");
        let err = read_exchange(&bytes).unwrap_err();
        assert_eq!(err.code(), "E-FILE-FORMAT", "header length {len}");
    }
}

#[test]
fn u03_header_len_is_the_little_endian_prefix() {
    assert_eq!(header_len(&[0x10, 0x02, 0, 0, 0, 0, 0, 0]), 0x0210);
    assert_eq!(header_len(&[0xff; 8]), u64::MAX);
}
