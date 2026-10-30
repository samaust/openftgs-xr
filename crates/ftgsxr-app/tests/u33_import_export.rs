// SPDX-License-Identifier: Apache-2.0

//! U-33, the `cmd_import` and `cmd_export` clause (§1.7, `cpu` lane): both commands copy a
//! `ftgs-if` byte for byte, and write the activated form of a checkpoint, which reads back
//! equal to U-5's activation (`read_checkpoint`), the producer, identity and kind set aside
//! (the writer names `openftgs-xr <version>`; Q-65, Q-72). An invalid input writes nothing
//! and exits 2.

use std::fs;
use std::path::{Path, PathBuf};

use ftgsxr_app::{HeadlessError, cmd_export, cmd_import, exit_code};
use ftgsxr_core::{Model, ModelKind, open_model, read_checkpoint, read_exchange, sha256};
use ftgsxr_testkit::{exchange_bytes, fixtures_dir, random_model, scratch_dir};

type Command = fn(&Path, &Path) -> Result<(), HeadlessError>;
const COMMANDS: [(&str, Command); 2] = [("import", cmd_import), ("export", cmd_export)];

fn scratch(name: &str) -> PathBuf {
    scratch_dir(env!("CARGO_TARGET_TMPDIR"), env!("CARGO_CRATE_NAME"), name)
}

/// A quaternion normalized in float64 with the canonical sign (§1.2), as U-5 compares them.
fn canonical(q: [f32; 4]) -> [f64; 4] {
    let q = q.map(f64::from);
    let norm = q.iter().map(|x| x * x).sum::<f64>().sqrt();
    let u = q.map(|x| x / norm);
    let lead = *u.iter().find(|x| **x != 0.0).unwrap();
    if lead < 0.0 { u.map(|x| -x) } else { u }
}

/// `written` equals `activated` but for the producer, identity and kind; rotations are
/// equal after normalization and the canonical sign, up to float32 rounding, since every
/// reader renormalizes them.
fn assert_reads_back_equal(written: &Model, activated: &Model) {
    assert_eq!(written.producer, "openftgs-xr 0.1.0");
    assert_eq!(written.kind, ModelKind::ExchangeFile);
    let mut w = written.clone();
    let mut a = activated.clone();
    for (x, y) in w.rotation.iter().zip(&a.rotation) {
        let (x, y) = (canonical(*x), canonical(*y));
        for c in 0..4 {
            assert!((x[c] - y[c]).abs() <= 2.0 * f64::from(f32::EPSILON));
        }
    }
    w.rotation.clear();
    a.rotation.clear();
    (w.producer, w.identity, w.kind) = (a.producer.clone(), a.identity, a.kind);
    assert_eq!(w, a);
}

#[test]
fn u33_a_ftgs_if_input_is_copied_byte_for_byte() {
    let dir = scratch("copy");
    for (name, command) in COMMANDS {
        for fixture in ["probes/P1.safetensors", "checkpoint/single.safetensors"] {
            let input = fixtures_dir().join(fixture);
            let out = dir.join(format!("{name}.safetensors"));
            command(&input, &out).unwrap();
            assert_eq!(
                fs::read(&out).unwrap(),
                fs::read(&input).unwrap(),
                "{name} {fixture}"
            );
        }
    }
}

#[test]
fn u33_a_checkpoint_is_written_in_its_activated_form() {
    let dir = scratch("activated");
    for (name, command) in COMMANDS {
        for ckpt in ["tiny", "single"] {
            let input = fixtures_dir().join("checkpoint").join(ckpt);
            let out = dir.join(format!("{name}-{ckpt}.safetensors"));
            command(&input, &out).unwrap();
            let bytes = fs::read(&out).unwrap();
            let written = read_exchange(&bytes).unwrap();
            assert_eq!(written.identity, sha256(&bytes));
            assert_reads_back_equal(&written, &read_checkpoint(&input).unwrap());
        }
    }
}

#[test]
fn u33_a_ply_is_written_in_its_activated_form() {
    let dir = scratch("ply");
    let input = fixtures_dir().join("checkpoint/tiny.tau0.ply");
    for (name, command) in COMMANDS {
        let out = dir.join(format!("{name}.safetensors"));
        command(&input, &out).unwrap();
        let written = read_exchange(&fs::read(&out).unwrap()).unwrap();
        assert_reads_back_equal(&written, &open_model(&input).unwrap());
    }
}

#[test]
fn u33_an_invalid_input_writes_nothing_and_exits_2() {
    let dir = scratch("invalid");
    let mut model = random_model(3, 6, 0);
    model.scale[2][1] = 0.0; // E-FILE-VALUE
    let bad = dir.join("bad.safetensors");
    fs::write(&bad, exchange_bytes(&model)).unwrap();
    for (name, command) in COMMANDS {
        let out = dir.join(format!("{name}.safetensors"));
        let e = command(&bad, &out).unwrap_err();
        assert_eq!((e.code(), exit_code(&e)), ("E-FILE-VALUE", 2), "{name}");
        assert!(!out.exists(), "{name} wrote a file");
        assert_eq!(
            fs::read_dir(&dir).unwrap().count(),
            1,
            "{name} left a temporary"
        );
    }
}
