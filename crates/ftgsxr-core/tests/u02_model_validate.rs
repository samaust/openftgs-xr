// SPDX-License-Identifier: Apache-2.0

//! U-2's rules on hand-built models (§1.2 reader rules, §1.15 codes, `cpu` lane):
//! `Model::validate` rejects each rule a `Model` in memory can break with its E-FILE code,
//! reporting the first broken rule in §1.2's order. The file-level clauses of U-2
//! (another `format`, major version 2, a missing key, a wrong dtype, `render_conventions`
//! that is not an object, version `1.3`) are the loader's.

use ftgsxr_core::{Model, ModelKind, RenderConventions, TimeSpan};

/// A valid two-primitive, degree-1 model over τ ∈ [0, 2] s.
fn valid() -> Model {
    Model {
        position: vec![[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        time_center: vec![0.5, 1.5],
        duration: vec![0.25, 1.0],
        velocity: vec![[0.1, 0.0, 0.0], [0.0, -0.2, 0.0]],
        scale: vec![[0.1, 0.2, 0.3], [0.05, 0.05, 0.05]],
        rotation: vec![[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]],
        opacity: vec![0.5, 0.9],
        sh: vec![0.1; 2 * 4 * 3],
        sh_degree: 1,
        time: TimeSpan {
            tau0: 0.0,
            tau1: 2.0,
            time_unit: "s".into(),
        },
        conventions: RenderConventions::default(),
        producer: "openftgs 0.1.0".into(),
        identity: [0; 32],
        kind: ModelKind::ExchangeFile,
    }
}

/// A valid single-timestamp model at τ₀ = τ₁ = 3 (openftgs D-70, D-71).
fn single() -> Model {
    let mut m = valid();
    m.time.tau0 = 3.0;
    m.time.tau1 = 3.0;
    m.time_center = vec![3.0, 3.0];
    m.duration = vec![1.0, 1.0];
    m.velocity = vec![[0.0; 3]; 2];
    m
}

fn code_of(m: &Model) -> &'static str {
    m.validate().expect_err("the model breaks a rule").code()
}

#[test]
fn u02_valid_models_pass() {
    valid().validate().expect("the baseline is valid");
    single()
        .validate()
        .expect("the single-timestamp baseline is valid");
    let mut m = valid();
    m.rotation[0] = [2.0, 0.0, 0.0, 0.0]; // non-unit quaternions are accepted
    m.sh_degree = 0;
    m.sh = vec![0.0; 2 * 3];
    m.validate()
        .expect("a non-unit rotation and degree 0 are valid");
}

#[test]
fn u02_sh_degree_out_of_range_is_e_file_sh() {
    let mut m = valid();
    m.sh_degree = 4;
    m.sh = vec![0.0; 2 * 25 * 3];
    assert_eq!(code_of(&m), "E-FILE-SH");
}

#[test]
fn u02_sh_shape_disagreeing_with_degree_is_e_file_sh() {
    let mut m = valid();
    m.sh_degree = 2; // the sh array still holds degree 1's 4 coefficients
    assert_eq!(code_of(&m), "E-FILE-SH");
}

#[test]
fn u02_tau1_before_tau0_is_e_file_value() {
    let mut m = valid();
    m.time.tau0 = 2.0;
    m.time.tau1 = 1.0;
    assert_eq!(code_of(&m), "E-FILE-VALUE");
    let mut m = valid();
    m.time.tau1 = f64::INFINITY;
    assert_eq!(code_of(&m), "E-FILE-VALUE");
}

#[test]
fn u02_wrong_row_count_or_empty_is_e_file_shape() {
    let mut m = valid();
    m.opacity.pop();
    assert_eq!(code_of(&m), "E-FILE-SHAPE");
    let mut m = valid();
    m.duration.push(1.0);
    assert_eq!(code_of(&m), "E-FILE-SHAPE");
    let mut m = valid();
    for v in [&mut m.time_center, &mut m.duration] {
        v.clear();
    }
    m.position.clear();
    m.velocity.clear();
    m.scale.clear();
    m.rotation.clear();
    m.opacity.clear();
    m.sh.clear();
    assert_eq!(code_of(&m), "E-FILE-SHAPE", "N = 0");
}

#[test]
fn u02_nan_anywhere_is_e_file_value() {
    let edits: [fn(&mut Model); 8] = [
        |m| m.position[1][2] = f32::NAN,
        |m| m.time_center[0] = f64::NAN,
        |m| m.duration[1] = f64::INFINITY,
        |m| m.velocity[0][1] = f32::NAN,
        |m| m.scale[0][0] = f32::NAN,
        |m| m.rotation[1][3] = f32::NAN,
        |m| m.opacity[0] = f32::NAN,
        |m| m.sh[5] = f32::NEG_INFINITY,
    ];
    for (k, edit) in edits.iter().enumerate() {
        let mut m = valid();
        edit(&mut m);
        assert_eq!(code_of(&m), "E-FILE-VALUE", "edit {k}");
    }
}

#[test]
fn u02_non_positive_duration_or_scale_is_e_file_value() {
    for d in [0.0, -1.0] {
        let mut m = valid();
        m.duration[1] = d;
        assert_eq!(code_of(&m), "E-FILE-VALUE", "duration {d}");
    }
    for s in [0.0, -0.1] {
        let mut m = valid();
        m.scale[1][2] = s;
        assert_eq!(code_of(&m), "E-FILE-VALUE", "scale {s}");
    }
}

#[test]
fn u02_rotation_norm_1e_9_is_e_file_value() {
    let mut m = valid();
    m.rotation[0] = [1e-9, 0.0, 0.0, 0.0];
    assert_eq!(code_of(&m), "E-FILE-VALUE");
    let mut m = valid();
    m.rotation[0] = [0.0; 4];
    assert_eq!(code_of(&m), "E-FILE-VALUE");
}

#[test]
fn u02_opacity_0_and_1_are_e_file_value() {
    for o in [0.0, 1.0, 1.5, -0.5] {
        let mut m = valid();
        m.opacity[1] = o;
        assert_eq!(code_of(&m), "E-FILE-VALUE", "opacity {o}");
    }
}

#[test]
fn u02_single_timestamp_with_motion_is_e_file_single() {
    let mut m = single();
    m.velocity[1] = [0.0, 0.0, 0.01];
    assert_eq!(code_of(&m), "E-FILE-SINGLE");
    let mut m = single();
    m.time_center[0] = 3.5;
    assert_eq!(code_of(&m), "E-FILE-SINGLE");
}

#[test]
fn u02_background_out_of_range_is_e_file_value() {
    let mut m = valid();
    m.conventions.background = [1.5, 0.0, 0.0];
    assert_eq!(code_of(&m), "E-FILE-VALUE");
}

#[test]
fn u02_first_broken_rule_in_order_is_reported() {
    // §1.2's order: SH degree, then shapes, then values, then the single-timestamp rule.
    let mut m = single();
    m.velocity[0] = [1.0, 0.0, 0.0]; // E-FILE-SINGLE
    m.opacity[0] = 1.0; // E-FILE-VALUE, an earlier rule
    assert_eq!(code_of(&m), "E-FILE-VALUE");
    m.opacity.pop(); // E-FILE-SHAPE, earlier still
    assert_eq!(code_of(&m), "E-FILE-SHAPE");
    m.sh_degree = 7; // E-FILE-SH, the first
    assert_eq!(code_of(&m), "E-FILE-SH");
    // A conventions fault comes after the single-timestamp rule.
    let mut m = single();
    m.velocity[0] = [1.0, 0.0, 0.0];
    m.conventions.background = [2.0, 0.0, 0.0];
    assert_eq!(code_of(&m), "E-FILE-SINGLE");
}
