// SPDX-License-Identifier: Apache-2.0

//! U-10 (E5, `cpu` lane): the (w, x, y, z) ↔ engine order round trip with the canonical
//! sign; Σ symmetric positive definite with eigenvalues scale²; q, 3q and −q give one Σ;
//! `posed_covariance` = s² R Σ Rᵀ; R(q) equals openftgs E5's matrix on random q. 10⁻⁶.
//!
//! The canonical sign is §1.2's: w ≥ 0, and where w = 0 the first nonzero component
//! positive. openftgs E5's matrix R(w, x, y, z) is written out below from
//! `docs/spec/openftgs-spec.md` E5, sharing no code with `ftgsxr_core`.

use approx::{assert_relative_eq, relative_eq};
use proptest::prelude::*;

use ftgsxr_core::{
    Mat3, Pose, Quat, Vec3, covariance, posed_covariance, quat_from_wxyz, quat_to_wxyz,
};

/// The row's tolerance.
const TOL: f32 = 1e-6;

/// openftgs E5's R(w, x, y, z) for q normalized first, in float64, row-major.
fn openftgs_r(wxyz: [f32; 4]) -> [[f64; 3]; 3] {
    let n = wxyz
        .iter()
        .map(|c| f64::from(*c).powi(2))
        .sum::<f64>()
        .sqrt();
    let [w, x, y, z] = wxyz.map(|c| f64::from(c) / n);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// a · b for row-major 3 × 3 matrices.
fn mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for (i, row) in c.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}

/// aᵀ for a row-major 3 × 3 matrix.
fn transpose(a: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut t = [[0.0; 3]; 3];
    for (i, row) in t.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a[j][i];
        }
    }
    t
}

/// glam's column-major `Mat3` as row-major float64 entries: `m.col(j)[i]` is row i, column j.
fn rows(m: Mat3) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    for (i, row) in r.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = f64::from(m.col(j)[i]);
        }
    }
    r
}

/// Asserts two row-major matrices are equal within `tol`, absolute or relative.
fn assert_mat_eq(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3], tol: f64) {
    for i in 0..3 {
        for j in 0..3 {
            assert_relative_eq!(a[i][j], b[i][j], epsilon = tol, max_relative = tol);
        }
    }
}

/// Asserts two quaternions in file order are equal within the row's tolerance.
fn assert_wxyz_eq(a: [f32; 4], b: [f32; 4]) {
    for (x, y) in a.iter().zip(b.iter()) {
        assert_relative_eq!(*x, *y, epsilon = TOL);
    }
}

#[test]
fn u10_wxyz_round_trip_keeps_a_canonical_unit_quaternion() {
    // (w, x, y, z) = (0.5, 0.5, −0.5, 0.5): unit, w > 0.
    let file = [0.5, 0.5, -0.5, 0.5];
    let q = quat_from_wxyz(file);
    // The engine's order is (x, y, z, w).
    assert_wxyz_eq([q.x, q.y, q.z, q.w], [0.5, -0.5, 0.5, 0.5]);
    assert_wxyz_eq(quat_to_wxyz(q), file);
}

#[test]
fn u10_wxyz_round_trip_normalizes_and_takes_the_canonical_sign() {
    // A non-unit quaternion with w < 0: normalized, then negated (w ≥ 0).
    let q = quat_from_wxyz([-2.0, 2.0, -2.0, 2.0]);
    assert_relative_eq!(q.length(), 1.0, epsilon = TOL);
    assert_wxyz_eq(quat_to_wxyz(q), [0.5, -0.5, 0.5, -0.5]);
    // w = 0: the first nonzero of (x, y, z) is made positive.
    assert_wxyz_eq(
        quat_to_wxyz(quat_from_wxyz([0.0, -0.6, 0.8, 0.0])),
        [0.0, 0.6, -0.8, 0.0],
    );
    assert_wxyz_eq(
        quat_to_wxyz(quat_from_wxyz([0.0, 0.0, -1.0, 0.0])),
        [0.0, 0.0, 1.0, 0.0],
    );
    assert_wxyz_eq(
        quat_to_wxyz(quat_from_wxyz([0.0, 0.0, 0.0, -3.0])),
        [0.0, 0.0, 0.0, 1.0],
    );
    // −0.0 counts as w = 0, not as a positive w.
    assert_wxyz_eq(
        quat_to_wxyz(quat_from_wxyz([-0.0, 0.0, -1.0, 0.0])),
        [0.0, 0.0, 1.0, 0.0],
    );
    // Already canonical with w = 0: unchanged.
    assert_wxyz_eq(
        quat_to_wxyz(quat_from_wxyz([0.0, 0.6, -0.8, 0.0])),
        [0.0, 0.6, -0.8, 0.0],
    );
}

#[test]
fn u10_covariance_is_symmetric_positive_definite_with_eigenvalues_scale_squared() {
    let q = quat_from_wxyz([0.9, 0.3, -0.2, 0.25]);
    let scale = Vec3::new(0.5, 2.0, 0.1);
    let sigma = rows(covariance(q, scale));
    let tol = f64::from(TOL);
    assert_mat_eq(&sigma, &transpose(&sigma), tol);
    // The eigenvalues λ_k = scale_k² are the roots of the characteristic polynomial, whose
    // coefficients are trace, the sum of the principal 2 × 2 minors and the determinant.
    let l = [0.25f64, 4.0, 0.01];
    let trace = sigma[0][0] + sigma[1][1] + sigma[2][2];
    let minors = sigma[0][0] * sigma[1][1] - sigma[0][1] * sigma[1][0] + sigma[0][0] * sigma[2][2]
        - sigma[0][2] * sigma[2][0]
        + sigma[1][1] * sigma[2][2]
        - sigma[1][2] * sigma[2][1];
    let det = sigma[0][0] * (sigma[1][1] * sigma[2][2] - sigma[1][2] * sigma[2][1])
        - sigma[0][1] * (sigma[1][0] * sigma[2][2] - sigma[1][2] * sigma[2][0])
        + sigma[0][2] * (sigma[1][0] * sigma[2][1] - sigma[1][1] * sigma[2][0]);
    assert_relative_eq!(trace, l[0] + l[1] + l[2], max_relative = tol);
    assert_relative_eq!(
        minors,
        l[0] * l[1] + l[0] * l[2] + l[1] * l[2],
        max_relative = tol
    );
    assert_relative_eq!(det, l[0] * l[1] * l[2], max_relative = 10.0 * tol);
    // Positive definite: the leading principal minors are positive (Sylvester).
    assert!(sigma[0][0] > 0.0);
    assert!(sigma[0][0] * sigma[1][1] - sigma[0][1] * sigma[1][0] > 0.0);
    assert!(det > 0.0);
}

#[test]
fn u10_q_3q_and_minus_q_give_one_covariance() {
    let (w, x, y, z) = (0.6f32, -0.2, 0.7, 0.33);
    let scale = Vec3::new(0.3, 0.07, 1.5);
    let one = rows(covariance(Quat::from_xyzw(x, y, z, w).normalize(), scale));
    let three = rows(covariance(
        Quat::from_xyzw(3.0 * x, 3.0 * y, 3.0 * z, 3.0 * w),
        scale,
    ));
    let minus = rows(covariance(Quat::from_xyzw(-x, -y, -z, -w), scale));
    assert_mat_eq(&three, &one, f64::from(TOL));
    assert_mat_eq(&minus, &one, f64::from(TOL));
    // The same through the file order.
    let file = rows(covariance(
        quat_from_wxyz([3.0 * w, 3.0 * x, 3.0 * y, 3.0 * z]),
        scale,
    ));
    assert_mat_eq(&file, &one, f64::from(TOL));
}

#[test]
fn u10_covariance_uses_openftgs_e5_matrix() {
    let wxyz = [0.9f32, 0.3, -0.2, 0.25];
    let scale = [0.5f64, 2.0, 0.1];
    let r = openftgs_r(wxyz);
    let s2 = [
        [scale[0] * scale[0], 0.0, 0.0],
        [0.0, scale[1] * scale[1], 0.0],
        [0.0, 0.0, scale[2] * scale[2]],
    ];
    let expected = mul(&mul(&r, &s2), &transpose(&r));
    let got = rows(covariance(quat_from_wxyz(wxyz), Vec3::new(0.5, 2.0, 0.1)));
    assert_mat_eq(&got, &expected, f64::from(TOL));
}

#[test]
fn u10_posed_covariance_is_s_squared_r_sigma_r_transpose() {
    // A generic symmetric Σ with off-diagonal entries, and a pose with s = 1.7.
    let sigma = [[0.5f64, 0.1, -0.2], [0.1, 0.3, 0.05], [-0.2, 0.05, 0.9]];
    let cov = Mat3::from_cols(
        Vec3::new(0.5, 0.1, -0.2),
        Vec3::new(0.1, 0.3, 0.05),
        Vec3::new(-0.2, 0.05, 0.9),
    );
    let wxyz = [0.8f32, -0.1, 0.4, 0.3];
    let pose = Pose {
        translation: Vec3::new(1.0, -2.0, 3.0),
        rotation: quat_from_wxyz(wxyz),
        scale: 1.7,
    };
    let r = openftgs_r(wxyz);
    let mut expected = mul(&mul(&r, &sigma), &transpose(&r));
    for row in expected.iter_mut() {
        for cell in row.iter_mut() {
            *cell *= 1.7f64 * 1.7;
        }
    }
    let got = rows(posed_covariance(cov, &pose));
    assert_mat_eq(&got, &expected, f64::from(TOL));
}

#[test]
fn u10_posed_covariance_composes_with_the_primitive_rotation_and_scale() {
    // s_m² R(q_m) Σ(q, a) R(q_m)ᵀ = Σ(q_m q, s_m a): the posed primitive is a primitive.
    let q = quat_from_wxyz([0.6, -0.2, 0.7, 0.33]);
    let a = Vec3::new(0.3, 0.07, 1.5);
    let pose = Pose {
        translation: Vec3::ZERO,
        rotation: quat_from_wxyz([0.2, 0.9, -0.3, 0.1]),
        scale: 0.4,
    };
    let got = rows(posed_covariance(covariance(q, a), &pose));
    let expected = rows(covariance(pose.rotation * q, 0.4 * a));
    assert_mat_eq(&got, &expected, f64::from(TOL));
}

/// §1.2's canonical sign, written from the rule: w ≥ 0, and where w = 0 the first nonzero
/// component positive.
fn canonical(q: [f64; 4]) -> [f64; 4] {
    match q.iter().find(|c| **c != 0.0) {
        Some(c) if *c < 0.0 => q.map(|c| -c),
        _ => q,
    }
}

/// The largest magnitude of a row-major matrix, the scale of its entries' tolerance.
fn max_abs(a: &[[f64; 3]; 3]) -> f64 {
    a.iter().flatten().fold(0.0, |m, x| m.max(x.abs()))
}

/// Whether two row-major matrices agree within 10⁻⁶ of the larger one's magnitude.
fn mat_close(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> bool {
    let tol = f64::from(TOL) * max_abs(a).max(max_abs(b)).max(1.0);
    (0..3).all(|i| (0..3).all(|j| relative_eq!(a[i][j], b[i][j], epsilon = tol)))
}

/// A random quaternion in file order, away from 0 so that its normalization is well defined.
fn quaternion() -> impl Strategy<Value = [f32; 4]> {
    prop::array::uniform4(-1.0f32..1.0).prop_filter("norm > 0.1", |q| {
        q.iter().map(|c| c * c).sum::<f32>() > 0.01
    })
}

/// A random positive scale.
fn scales() -> impl Strategy<Value = [f32; 3]> {
    prop::array::uniform3(0.01f32..10.0)
}

proptest! {
    #[test]
    fn u10_proptest_round_trip_with_the_canonical_sign(wxyz in quaternion()) {
        let n = wxyz.iter().map(|c| f64::from(*c).powi(2)).sum::<f64>().sqrt();
        let expected = canonical(wxyz.map(|c| f64::from(c) / n));
        let got = quat_to_wxyz(quat_from_wxyz(wxyz));
        for (g, e) in got.iter().zip(expected.iter()) {
            prop_assert!(relative_eq!(f64::from(*g), *e, epsilon = f64::from(TOL)));
        }
        // A second round trip changes nothing.
        let again = quat_to_wxyz(quat_from_wxyz(got));
        for (a, g) in again.iter().zip(got.iter()) {
            prop_assert!(relative_eq!(*a, *g, epsilon = TOL));
        }
    }

    #[test]
    fn u10_proptest_covariance_is_openftgs_e5_with_eigenvalues_scale_squared(
        wxyz in quaternion(),
        a in scales(),
        k in 0.1f32..10.0,
    ) {
        let r = openftgs_r(wxyz);
        let s2 = a.map(|x| f64::from(x) * f64::from(x));
        let d = [[s2[0], 0.0, 0.0], [0.0, s2[1], 0.0], [0.0, 0.0, s2[2]]];
        let expected = mul(&mul(&r, &d), &transpose(&r));
        let scale = Vec3::from_array(a);
        let sigma = rows(covariance(quat_from_wxyz(wxyz), scale));
        prop_assert!(mat_close(&sigma, &expected));
        prop_assert!(mat_close(&sigma, &transpose(&sigma)));
        // Column k of openftgs's R is an eigenvector with eigenvalue scale_k².
        let tol = f64::from(TOL) * max_abs(&expected).max(1.0);
        for (col, l) in s2.iter().enumerate() {
            for i in 0..3 {
                let sv: f64 = (0..3).map(|j| sigma[i][j] * r[j][col]).sum();
                prop_assert!(relative_eq!(sv, l * r[i][col], epsilon = tol));
            }
        }
        // kq and −q give the same Σ as q.
        let [w, x, y, z] = wxyz;
        let scaled = rows(covariance(Quat::from_xyzw(k * x, k * y, k * z, k * w), scale));
        let negated = rows(covariance(Quat::from_xyzw(-x, -y, -z, -w), scale));
        prop_assert!(mat_close(&scaled, &sigma));
        prop_assert!(mat_close(&negated, &sigma));
    }

    #[test]
    fn u10_proptest_posed_covariance_is_s_squared_r_sigma_r_transpose(
        wxyz in quaternion(),
        pose_wxyz in quaternion(),
        a in scales(),
        s in 0.01f32..100.0,
    ) {
        let cov = covariance(quat_from_wxyz(wxyz), Vec3::from_array(a));
        let pose = Pose { translation: Vec3::ZERO, rotation: quat_from_wxyz(pose_wxyz), scale: s };
        let r = openftgs_r(pose_wxyz);
        let mut expected = mul(&mul(&r, &rows(cov)), &transpose(&r));
        for cell in expected.iter_mut().flatten() {
            *cell *= f64::from(s) * f64::from(s);
        }
        prop_assert!(mat_close(&rows(posed_covariance(cov, &pose)), &expected));
    }
}
