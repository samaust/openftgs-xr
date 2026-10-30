// SPDX-License-Identifier: Apache-2.0

//! U-25, the `ftgs-renders` writer and reader (§1.7, openftgs §2.2, `cpu` lane): a
//! two-item round trip with the keys `rgb/<id>` and `alpha/<id>` and the metadata fields,
//! ids sorted on read; an openftgs-written renders fixture reads; a `request_sha256`
//! mismatch is refused (E-FILE-CHECKSUM, §1.15).

use std::path::PathBuf;

use ftgsxr_core::{CoreError, RenderImage, read_renders, read_request, write_renders};
use ftgsxr_testkit::fixtures_dir;
use serde_json::Value;

/// A scratch directory of this test binary's own, emptied first.
/// Named after this test binary and the process, so no other binary and no overlapping run
/// in another session or worktree shares it (#94).
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("u25_renders-{}", std::process::id()))
        .join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// An image whose values count up from `start` by `step`, so every pixel differs.
fn image(width: u32, height: u32, start: f32, step: f32) -> RenderImage {
    let n = (width * height) as usize;
    RenderImage {
        width,
        height,
        rgb: (0..3 * n).map(|i| start + step * i as f32).collect(),
        alpha: (0..n).map(|i| (i as f32 + 0.5) / n as f32).collect(),
    }
}

/// The header of a safetensors file, read by hand: the 8-byte little-endian length, then
/// the JSON header.
fn header(bytes: &[u8]) -> Value {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    serde_json::from_slice(&bytes[8..8 + n]).unwrap()
}

fn digest(byte: u8) -> [u8; 32] {
    let mut d = [0u8; 32];
    for (i, x) in d.iter_mut().enumerate() {
        *x = byte.wrapping_add((i as u8).wrapping_mul(17));
    }
    d
}

#[test]
fn u25_two_item_round_trip_with_its_keys_and_metadata() {
    let dir = scratch("round_trip");
    let path = dir.join("out.safetensors");
    // Out of order, of two sizes, with values above 1 and below 0 (no clamping).
    let b = image(3, 2, -0.75, 0.25);
    let a = image(2, 4, 0.5, 0.125);
    let sha = digest(0xa5);
    let items = vec![
        ("r0001".to_owned(), b.clone()),
        ("r0000".to_owned(), a.clone()),
    ];
    write_renders(&path, &items, "openftgs-xr 0.1.0 (test adapter)", sha).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let h = header(&bytes);
    let meta = h["__metadata__"].as_object().unwrap();
    assert_eq!(meta["format"], "ftgs-renders");
    assert_eq!(meta["version"], "1");
    assert_eq!(meta["producer"], "openftgs-xr 0.1.0 (test adapter)");
    let hex: String = sha.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(meta["request_sha256"], hex.as_str());
    let mut keys: Vec<&str> = h.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort();
    assert_eq!(
        keys,
        [
            "__metadata__",
            "alpha/r0000",
            "alpha/r0001",
            "rgb/r0000",
            "rgb/r0001"
        ]
    );
    assert_eq!(h["rgb/r0000"]["dtype"], "F32");
    assert_eq!(h["rgb/r0000"]["shape"], serde_json::json!([4, 2, 3]));
    assert_eq!(h["alpha/r0000"]["shape"], serde_json::json!([4, 2, 1]));
    assert_eq!(h["rgb/r0001"]["shape"], serde_json::json!([2, 3, 3]));
    assert_eq!(h["alpha/r0001"]["dtype"], "F32");

    let back = read_renders(&bytes, Some(sha)).unwrap();
    assert_eq!(back, vec![("r0000".to_owned(), a), ("r0001".to_owned(), b)]);
    assert_eq!(read_renders(&bytes, None).unwrap().len(), 2);
    assert_eq!(
        std::fs::read_dir(&dir).unwrap().count(),
        1,
        "no temporary left"
    );
}

#[test]
fn u25_a_request_sha256_mismatch_is_refused() {
    let dir = scratch("mismatch");
    let path = dir.join("out.safetensors");
    let items = vec![("r0000".to_owned(), image(2, 2, 0.0, 0.1))];
    write_renders(&path, &items, "p", digest(1)).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert!(matches!(
        read_renders(&bytes, Some(digest(2))),
        Err(CoreError::Checksum(_))
    ));
}

#[test]
fn u25_an_openftgs_written_renders_fixture_reads() {
    let probes = fixtures_dir().join("probes");
    let request = read_request(&std::fs::read(probes.join("P1.request.json")).unwrap()).unwrap();
    let bytes = std::fs::read(probes.join("P1.renders.safetensors")).unwrap();
    let renders = read_renders(&bytes, Some(request.sha256)).unwrap();
    let mut ids: Vec<&str> = request.items.iter().map(|i| i.id.as_str()).collect();
    ids.sort();
    let got: Vec<&str> = renders.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(got, ids);
    for ((_, image), item) in renders.iter().zip(&request.items) {
        assert_eq!(
            (image.width, image.height),
            (item.camera.width, item.camera.height)
        );
        assert_eq!(image.rgb.len(), 3 * (image.width * image.height) as usize);
        assert_eq!(image.alpha.len(), (image.width * image.height) as usize);
    }
    // The first float32 of `alpha/r0000`, read from the bytes by hand.
    let h = header(&bytes);
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let start = h["alpha/r0000"]["data_offsets"][0].as_u64().unwrap() as usize + 8 + n;
    let first = f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
    assert_eq!(renders[0].1.alpha[0].to_bits(), first.to_bits());
}

#[test]
fn u25_another_format_or_version_is_refused() {
    let request = std::fs::read(fixtures_dir().join("probes/P1.request.json")).unwrap();
    assert!(matches!(
        read_renders(&request, None),
        Err(CoreError::Format(_))
    ));
    // A ftgs-if model is a safetensors file of another format.
    let model = std::fs::read(fixtures_dir().join("probes/P1.safetensors")).unwrap();
    assert!(matches!(
        read_renders(&model, None),
        Err(CoreError::Format(_))
    ));
}

#[test]
fn u25_writing_buffers_that_disagree_with_the_size_is_refused() {
    let dir = scratch("bad_size");
    let mut bad = image(2, 2, 0.0, 0.1);
    bad.alpha.pop();
    let items = vec![("r0000".to_owned(), bad)];
    assert!(matches!(
        write_renders(&dir.join("o.safetensors"), &items, "p", digest(0)),
        Err(CoreError::Shape(_))
    ));
}

#[test]
fn u25_writing_a_non_finite_value_is_refused() {
    let dir = scratch("non_finite");
    let mut bad = image(2, 2, 0.0, 0.1);
    bad.rgb[5] = f32::NAN;
    let items = vec![("r0000".to_owned(), bad)];
    assert!(matches!(
        write_renders(&dir.join("o.safetensors"), &items, "p", digest(0)),
        Err(CoreError::Value(_))
    ));
}

#[test]
fn u25_writing_no_item_or_a_repeated_id_is_refused() {
    let dir = scratch("ids");
    let path = dir.join("o.safetensors");
    assert!(write_renders(&path, &[], "p", digest(0)).is_err());
    let one = image(1, 1, 0.0, 0.1);
    let twice = vec![("a".to_owned(), one.clone()), ("a".to_owned(), one)];
    assert!(matches!(
        write_renders(&path, &twice, "p", digest(0)),
        Err(CoreError::Value(_))
    ));
    assert!(!path.exists());
}
