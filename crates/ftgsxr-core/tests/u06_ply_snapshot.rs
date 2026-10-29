// SPDX-License-Identifier: Apache-2.0

//! U-6 (§1.2 the PLY snapshot, D-41, `cpu` lane): openftgs's `write_ply` output of the probe
//! `R2_overlap_deg3` at τ = 1.3 (Q-10) against the probe's own per-time arrays at τ (E1, E2):
//! positions within 10⁻⁶, scales within 10⁻⁶ relative (log round trip), opacity within 10⁻⁶
//! (logit round trip), SH reordered exactly, quaternions equal; an ASCII PLY and an unknown
//! property name are refused.

use std::fs;

use ftgsxr_core::{
    Model, ModelKind, RenderConventions, Vec3, effective_opacity, moved_position, open_model,
    read_exchange, read_ply, sha256, temporal_opacity,
};
use ftgsxr_testkit::fixtures_dir;

const TAU: f64 = 1.3;

fn snapshot_bytes() -> Vec<u8> {
    fs::read(fixtures_dir().join("probes/R2_overlap_deg3.tau1.3.ply")).unwrap()
}

fn probe() -> Model {
    read_exchange(&fs::read(fixtures_dir().join("probes/R2_overlap_deg3.safetensors")).unwrap())
        .unwrap()
}

#[test]
fn u06_snapshot_against_the_probe_at_tau() {
    let bytes = snapshot_bytes();
    let ply = read_ply(&bytes).unwrap();
    let probe = probe();
    assert_eq!(ply.len(), probe.len());
    assert_eq!(ply.sh_degree, probe.sh_degree);
    for i in 0..probe.len() {
        // E1: the moved position at τ.
        let expected = moved_position(
            Vec3::from_array(probe.position[i]),
            Vec3::from_array(probe.velocity[i]),
            probe.time_center[i],
            TAU,
        );
        for c in 0..3 {
            assert!(
                (ply.position[i][c] - expected[c]).abs() <= 1e-6,
                "position[{i}][{c}]: {} vs {}",
                ply.position[i][c],
                expected[c]
            );
            let (a, b) = (f64::from(ply.scale[i][c]), f64::from(probe.scale[i][c]));
            assert!((a - b).abs() <= 1e-6 * b, "scale[{i}][{c}]: {a} vs {b}");
        }
        // E2: the effective opacity at τ, clamped as `write_ply` clamps it before the logit.
        let sigma_t = temporal_opacity(TAU, probe.time_center[i], probe.duration[i], false);
        let expected =
            f64::from(effective_opacity(probe.opacity[i], sigma_t)).clamp(1e-7, 1.0 - 1e-7);
        assert!(
            (f64::from(ply.opacity[i]) - expected).abs() <= 1e-6,
            "opacity[{i}]: {} vs {expected}",
            ply.opacity[i]
        );
        for c in 0..4 {
            assert!(
                (ply.rotation[i][c] - probe.rotation[i][c]).abs() <= f32::EPSILON,
                "rotation[{i}]: {:?} vs {:?}",
                ply.rotation[i],
                probe.rotation[i]
            );
        }
    }
    assert_eq!(ply.sh, probe.sh, "SH reordered exactly");
}

#[test]
fn u06_snapshot_is_a_single_timestamp_model() {
    let bytes = snapshot_bytes();
    let ply = open_model(&fixtures_dir().join("probes/R2_overlap_deg3.tau1.3.ply")).unwrap();
    assert!(ply.is_single_timestamp());
    assert_eq!((ply.time.tau0, ply.time.tau1), (0.0, 0.0));
    assert_eq!(ply.time.time_unit, "s");
    assert!(ply.time_center.iter().all(|&t| t == 0.0));
    assert!(ply.duration.iter().all(|&d| d == 1.0));
    assert!(ply.velocity.iter().all(|v| *v == [0.0; 3]));
    assert_eq!(ply.conventions, RenderConventions::default());
    assert_eq!(ply.producer, "");
    assert_eq!(ply.kind, ModelKind::Ply);
    assert_eq!(ply.identity, sha256(&bytes));
}

#[test]
fn u06_checkpoint_snapshot_loads() {
    let ply = open_model(&fixtures_dir().join("checkpoint/tiny.tau0.ply")).unwrap();
    assert_eq!(ply.len(), 64);
    assert_eq!(ply.sh_degree, 3);
}

/// The snapshot with its header text changed by `edit`; the body kept.
fn with_header(edit: impl Fn(&str) -> String) -> Vec<u8> {
    let bytes = snapshot_bytes();
    let end = bytes
        .windows(11)
        .position(|w| w == b"end_header\n")
        .unwrap()
        + 11;
    let header = std::str::from_utf8(&bytes[..end]).unwrap();
    let mut out = edit(header).into_bytes();
    out.extend_from_slice(&bytes[end..]);
    out
}

#[test]
fn u06_ascii_ply_refused() {
    let ascii = with_header(|h| h.replace("binary_little_endian", "ascii"));
    assert_eq!(read_ply(&ascii).unwrap_err().code(), "E-FILE-FORMAT");
    let big = with_header(|h| h.replace("binary_little_endian", "binary_big_endian"));
    assert_eq!(read_ply(&big).unwrap_err().code(), "E-FILE-FORMAT");
}

#[test]
fn u06_unknown_property_refused() {
    let unknown = with_header(|h| h.replace("property float nx\n", "property float normal_x\n"));
    assert_eq!(read_ply(&unknown).unwrap_err().code(), "E-FILE-FORMAT");
}

#[test]
fn u06_other_faults() {
    // A property of another type.
    let double = with_header(|h| h.replace("property float opacity", "property double opacity"));
    assert_eq!(read_ply(&double).unwrap_err().code(), "E-FILE-SHAPE");
    // A comment names the producer.
    let named = with_header(|h| h.replacen("format", "comment openftgs 0.1.0\nformat", 1));
    assert_eq!(read_ply(&named).unwrap().producer, "openftgs 0.1.0");
    // A truncated body.
    let mut short = snapshot_bytes();
    short.pop();
    assert_eq!(read_ply(&short).unwrap_err().code(), "E-FILE-FORMAT");
    // Not a PLY at all.
    assert_eq!(
        read_ply(b"\x08\0\0\0\0\0\0\0{}      ").unwrap_err().code(),
        "E-FILE-FORMAT"
    );
}

#[test]
fn u06_f_rest_count_and_missing_names() {
    // Dropping three f_rest properties leaves 42, which no degree has; the body is cut to
    // match so only the count is wrong.
    let bytes = snapshot_bytes();
    let end = bytes
        .windows(11)
        .position(|w| w == b"end_header\n")
        .unwrap()
        + 11;
    let header = std::str::from_utf8(&bytes[..end]).unwrap();
    let header = header
        .replace("property float f_rest_42\n", "")
        .replace("property float f_rest_43\n", "")
        .replace("property float f_rest_44\n", "");
    let props = 62 - 3;
    let rows = (bytes.len() - end) / (62 * 4);
    let mut out = header.into_bytes();
    out.extend(vec![0u8; rows * props * 4]);
    assert_eq!(read_ply(&out).unwrap_err().code(), "E-FILE-SH");
    // A required property missing.
    let missing = with_header(|h| h.replace("property float rot_3\n", ""));
    assert_eq!(read_ply(&missing).unwrap_err().code(), "E-FILE-KEYS");
}
