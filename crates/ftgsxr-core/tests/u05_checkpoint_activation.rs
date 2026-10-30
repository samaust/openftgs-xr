// SPDX-License-Identifier: Apache-2.0

//! U-5 (§1.2 checkpoint activation, D-8, `cpu` lane): openftgs's tiny checkpoint against its
//! `ftgs export` file: positions exact, `time_center` and `duration` within 10⁻⁹ relative,
//! velocities and scales within 10⁻⁶ relative, rotations equal after normalization and the
//! canonical sign, opacities within 10⁻⁷, SH exact; a single-timestamp checkpoint uses the
//! unit 1; a `files` digest mismatch is E-FILE-CHECKSUM.

use std::fs;
use std::path::{Path, PathBuf};

use ftgsxr_core::{Model, ModelKind, hex, open_model, read_checkpoint, read_exchange, sha256};
use ftgsxr_testkit::{fixtures_dir, scratch_dir};
use serde_json::Value;

fn rel(a: f64, b: f64) -> f64 {
    if a == b {
        0.0
    } else {
        (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
    }
}

/// A quaternion normalized in float64 with the canonical sign (§1.2), for comparing
/// rotations rounded by two writers.
fn canonical(q: [f32; 4]) -> [f64; 4] {
    let q = q.map(f64::from);
    let norm = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    let u = q.map(|x| x / norm);
    let lead = *u.iter().find(|x| **x != 0.0).unwrap();
    if lead < 0.0 { u.map(|x| -x) } else { u }
}

/// Compares a checkpoint's activation with its export within U-5's tolerances.
fn compare(ckpt: &Model, export: &Model) {
    assert_eq!(ckpt.len(), export.len());
    assert_eq!(ckpt.sh_degree, export.sh_degree);
    assert_eq!(ckpt.time, export.time);
    assert_eq!(ckpt.position, export.position, "positions exact");
    for i in 0..ckpt.len() {
        assert!(
            rel(ckpt.time_center[i], export.time_center[i]) <= 1e-9,
            "time_center[{i}]: {} vs {}",
            ckpt.time_center[i],
            export.time_center[i]
        );
        assert!(
            rel(ckpt.duration[i], export.duration[i]) <= 1e-9,
            "duration[{i}]"
        );
        for c in 0..3 {
            let (a, b) = (
                f64::from(ckpt.velocity[i][c]),
                f64::from(export.velocity[i][c]),
            );
            assert!(rel(a, b) <= 1e-6, "velocity[{i}][{c}]: {a} vs {b}");
            let (a, b) = (f64::from(ckpt.scale[i][c]), f64::from(export.scale[i][c]));
            assert!(rel(a, b) <= 1e-6, "scale[{i}][{c}]: {a} vs {b}");
        }
        // Equal after normalization and the canonical sign, up to float32 rounding.
        let (a, b) = (canonical(ckpt.rotation[i]), canonical(export.rotation[i]));
        for c in 0..4 {
            assert!(
                (a[c] - b[c]).abs() <= 2.0 * f64::from(f32::EPSILON),
                "rotation[{i}]"
            );
        }
        assert!(
            (f64::from(ckpt.opacity[i]) - f64::from(export.opacity[i])).abs() <= 1e-7,
            "opacity[{i}]"
        );
    }
    assert_eq!(ckpt.sh, export.sh, "SH exact");
    assert_eq!(ckpt.conventions, export.conventions);
    assert_eq!(ckpt.producer, export.producer);
}

fn manifest(dir: &Path) -> Value {
    serde_json::from_slice(&fs::read(dir.join("manifest.json")).unwrap()).unwrap()
}

#[test]
fn u05_tiny_checkpoint_against_its_export() {
    let dir = fixtures_dir().join("checkpoint/tiny");
    let ckpt = read_checkpoint(&dir).unwrap();
    let export =
        read_exchange(&fs::read(fixtures_dir().join("checkpoint/tiny.safetensors")).unwrap())
            .unwrap();
    assert!(!ckpt.is_single_timestamp());
    compare(&ckpt, &export);
    let m = manifest(&dir);
    assert_eq!(
        ckpt.kind,
        ModelKind::Checkpoint {
            spatial_scale: m["spatial_scale"].as_f64().unwrap(),
            frame_dt: m["frame_dt"].as_f64().unwrap(),
        }
    );
    assert_eq!(
        ckpt.identity,
        sha256(&fs::read(dir.join("params.safetensors")).unwrap())
    );
    assert_eq!(open_model(&dir).unwrap(), ckpt);
}

#[test]
fn u05_single_timestamp_checkpoint_uses_the_unit_1() {
    let dir = fixtures_dir().join("checkpoint/single");
    let ckpt = read_checkpoint(&dir).unwrap();
    let export =
        read_exchange(&fs::read(fixtures_dir().join("checkpoint/single.safetensors")).unwrap())
            .unwrap();
    assert!(ckpt.is_single_timestamp());
    compare(&ckpt, &export);
    // With T_τ = 0 the unit is 1: duration = exp(log_durations), velocity 0, centers τ₀.
    let params = fs::read(dir.join("params.safetensors")).unwrap();
    let n = u64::from_le_bytes(params[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&params[8..8 + n]).unwrap();
    let offsets = header["log_durations"]["data_offsets"].as_array().unwrap();
    let start = 8 + n + offsets[0].as_u64().unwrap() as usize;
    for i in 0..ckpt.len() {
        let at = start + 4 * i;
        let log_d = f32::from_le_bytes(params[at..at + 4].try_into().unwrap());
        assert!(
            rel(ckpt.duration[i], f64::from(log_d).exp()) <= 1e-12,
            "duration[{i}]"
        );
        assert_eq!(ckpt.time_center[i], ckpt.time.tau0);
        assert_eq!(ckpt.velocity[i], [0.0; 3]);
    }
}

/// A copy of the tiny checkpoint's two read files in a scratch directory, the binary's own
/// per-process folder (#94, Q-62).
fn copy_tiny(name: &str) -> PathBuf {
    let dir = scratch_dir(env!("CARGO_TARGET_TMPDIR"), env!("CARGO_CRATE_NAME"), name);
    let src = fixtures_dir().join("checkpoint/tiny");
    for file in ["manifest.json", "params.safetensors"] {
        fs::copy(src.join(file), dir.join(file)).unwrap();
    }
    dir
}

/// A copy of the tiny checkpoint whose manifest `edit` changed.
fn edited_tiny(name: &str, edit: impl FnOnce(&mut Value)) -> PathBuf {
    let dir = copy_tiny(name);
    let mut m = manifest(&dir);
    edit(&mut m);
    fs::write(dir.join("manifest.json"), serde_json::to_vec(&m).unwrap()).unwrap();
    dir
}

#[test]
fn u05_files_digest_mismatch() {
    // The other files are ignored: the copy without them loads.
    let dir = copy_tiny("ok");
    assert!(read_checkpoint(&dir).is_ok());

    // The manifest's digest changed.
    let dir = edited_tiny("manifest", |m| {
        m["files"]["params.safetensors"] = Value::String(hex(&[7; 32]))
    });
    assert_eq!(read_checkpoint(&dir).unwrap_err().code(), "E-FILE-CHECKSUM");

    // One byte of the parameters changed, the file still a valid one.
    let dir = copy_tiny("params");
    let path = dir.join("params.safetensors");
    let mut bytes = fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last - 3] ^= 0x01;
    fs::write(&path, &bytes).unwrap();
    assert_eq!(read_checkpoint(&dir).unwrap_err().code(), "E-FILE-CHECKSUM");
}

/// The tiny checkpoint read after `edit` changed its manifest.
fn read_edited(name: &str, edit: impl FnOnce(&mut Value)) -> Model {
    read_checkpoint(&edited_tiny(name, edit)).unwrap()
}

/// A manifest edit: its name and the edit.
type Edit = (&'static str, fn(&mut Value));

#[test]
fn u05_producer_without_an_openftgs_version() {
    // Q-34 (owner, #55): `openftgs <version>` when the manifest names one, else
    // `openftgs`, as openftgs's own producer string falls back.
    let named = read_edited("producer-named", |m| {
        m["versions"] = serde_json::json!({"openftgs": "0.3.1"})
    });
    assert_eq!(named.producer, "openftgs 0.3.1");
    let cases: [Edit; 5] = [
        ("null", |m| m["versions"]["openftgs"] = Value::Null),
        ("absent", |m| {
            m.as_object_mut().unwrap().remove("versions");
        }),
        ("no-entry", |m| m["versions"] = serde_json::json!({})),
        ("not-string", |m| {
            m["versions"] = serde_json::json!({"openftgs": 3})
        }),
        ("versions-not-object", |m| m["versions"] = "0.3.1".into()),
    ];
    for (name, edit) in cases {
        let ckpt = read_edited(&format!("producer-{name}"), edit);
        assert_eq!(ckpt.producer, "openftgs", "{name}");
    }
}

#[test]
fn u05_string_version_loads() {
    // Q-34: `version` may be the string "1", as openftgs's `load_checkpoint` compares
    // `str(version)`.
    let ckpt = read_edited("version-string", |m| m["version"] = "1".into());
    assert_eq!(
        ckpt,
        read_checkpoint(&fixtures_dir().join("checkpoint/tiny")).unwrap()
    );
}

/// A manifest rule: its name, the edit that breaks it, the code expected.
type Case = (&'static str, fn(&mut Value), &'static str);

#[test]
fn u05_manifest_rules() {
    let cases: [Case; 6] = [
        (
            "format",
            |m| m["format"] = "ftgs-if".into(),
            "E-FILE-FORMAT",
        ),
        ("version", |m| m["version"] = 2.into(), "E-FILE-VERSION"),
        (
            "tau1",
            |m| {
                m.as_object_mut().unwrap().remove("tau1");
            },
            "E-FILE-KEYS",
        ),
        (
            "files",
            |m| {
                m["files"]
                    .as_object_mut()
                    .unwrap()
                    .remove("params.safetensors");
            },
            "E-FILE-KEYS",
        ),
        ("sh_degree", |m| m["sh_degree"] = 2.into(), "E-FILE-SH"),
        (
            "background",
            |m| m["config"]["background"] = serde_json::json!([0, 2, 0]),
            "E-FILE-VALUE",
        ),
    ];
    for (name, edit, code) in cases {
        let dir = edited_tiny(&format!("rule-{name}"), edit);
        assert_eq!(read_checkpoint(&dir).unwrap_err().code(), code, "{name}");
    }
    let dir = copy_tiny("not-json");
    fs::write(dir.join("manifest.json"), b"{\"format\": ").unwrap();
    assert_eq!(read_checkpoint(&dir).unwrap_err().code(), "E-FILE-FORMAT");
    let dir = copy_tiny("missing-params");
    fs::remove_file(dir.join("params.safetensors")).unwrap();
    assert_eq!(read_checkpoint(&dir).unwrap_err().code(), "E-IO");
}
