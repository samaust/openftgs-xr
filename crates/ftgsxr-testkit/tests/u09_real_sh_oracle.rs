// SPDX-License-Identifier: Apache-2.0

//! U-9's independent real-SH evaluation (E4, Q-1): `real_sh` reproduces E4's table, its
//! constants and signs, so U-9 can trust it against `ftgsxr_core::sh`. The expected values
//! are E4's table worked by hand at the axes and at (1, 1, 1)/√3.

use approx::assert_relative_eq;
use ftgsxr_core::Vec3;
use ftgsxr_testkit::real_sh;

// E4's constants, as the spec prints them.
const C0: f64 = 0.28209479177387814;
const C1: f64 = 0.4886025119029199;
const C20: f64 = 0.31539156525252005;
const C21: f64 = 1.0925484305920792;
const C22: f64 = 0.5462742152960396;
const C30: f64 = 0.3731763325901154;
const C31: f64 = 0.4570457994644658;
const C32: f64 = 1.445305721320277;
const C33: f64 = 0.5900435899266435;

const TOL: f64 = 1e-12;

fn assert_basis(got: [f64; 16], want: [f64; 16], what: &str) {
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= TOL,
            "{what}: coefficient {k} is {g}, E4 gives {w}"
        );
    }
}

/// E4's table at +x: x = 1, y = z = 0.
const AT_X: [f64; 16] = [
    C0, // l = 0
    0.0, 0.0, -C1, // l = 1: −C₁y, C₁z, −C₁x
    0.0, 0.0, -C20, 0.0,
    C22, // l = 2: C₂₀(3z² − 1) = −C₂₀; C₂₂(x² − y²) = C₂₂
    0.0, 0.0, 0.0, 0.0, C31, 0.0,
    -C33, // l = 3: −C₃₁x(5z² − 1) = C₃₁; −C₃₃x(x² − 3y²) = −C₃₃
];

/// E4's table at +y: y = 1, x = z = 0.
const AT_Y: [f64; 16] = [
    C0, -C1, 0.0, 0.0, 0.0, 0.0, -C20, 0.0, -C22, // C₂₂(x² − y²) = −C₂₂
    C33, 0.0, C31, 0.0, 0.0, 0.0,
    0.0, // −C₃₃y(3x² − y²) = C₃₃; −C₃₁y(5z² − 1) = C₃₁
];

/// E4's table at +z: z = 1, x = y = 0.
const AT_Z: [f64; 16] = [
    C0,
    0.0,
    C1,
    0.0,
    0.0,
    0.0,
    2.0 * C20,
    0.0,
    0.0, // C₂₀(3 − 1)
    0.0,
    0.0,
    0.0,
    2.0 * C30,
    0.0,
    0.0,
    0.0, // C₃₀(5 − 3)
];

#[test]
fn u09_real_sh_gives_e4_at_the_positive_axes() {
    assert_basis(real_sh(3, Vec3::X), AT_X, "+x");
    assert_basis(real_sh(3, Vec3::Y), AT_Y, "+y");
    assert_basis(real_sh(3, Vec3::Z), AT_Z, "+z");
}

/// Y_lm(−d) = (−1)^l Y_lm(d): every polynomial of band l in E4's table has degree l.
#[test]
fn u09_real_sh_gives_e4_at_the_negative_axes() {
    let parity = |want: [f64; 16]| {
        let mut out = want;
        for (k, v) in out.iter_mut().enumerate() {
            let l = (k as f64).sqrt().floor() as i32;
            *v *= f64::from((-1i32).pow(l as u32));
        }
        out
    };
    assert_basis(real_sh(3, Vec3::NEG_X), parity(AT_X), "−x");
    assert_basis(real_sh(3, Vec3::NEG_Y), parity(AT_Y), "−y");
    assert_basis(real_sh(3, Vec3::NEG_Z), parity(AT_Z), "−z");
}

/// Off the axes, where E4's mixed terms (xy, yz, xz, xyz) and their signs show.
#[test]
fn u09_real_sh_gives_e4_on_the_diagonal() {
    let s = 1.0 / 3f64.sqrt();
    let s3 = s * s * s;
    let want = [
        C0,
        -C1 * s,
        C1 * s,
        -C1 * s,
        C21 / 3.0,
        -C21 / 3.0,
        0.0,
        -C21 / 3.0,
        0.0,
        -2.0 * C33 * s3,        // −C₃₃y(3x² − y²) = −2C₃₃s³
        2.0 * C32 * s3,         // 2C₃₂xyz
        -C31 * s * (2.0 / 3.0), // −C₃₁y(5z² − 1)
        -C30 * s * (4.0 / 3.0), // C₃₀z(5z² − 3)
        -C31 * s * (2.0 / 3.0), // −C₃₁x(5z² − 1)
        0.0,                    // C₃₂z(x² − y²)
        2.0 * C33 * s3,         // −C₃₃x(x² − 3y²) = 2C₃₃s³
    ];
    assert_basis(real_sh(3, Vec3::ONE), want, "(1, 1, 1)/√3");
}

#[test]
fn u09_real_sh_normalizes_its_direction() {
    let d = Vec3::new(0.3, -1.2, 0.7);
    let unit = real_sh(3, d.normalize());
    let long = real_sh(3, d * 5.0);
    for (a, b) in unit.iter().zip(long) {
        assert_relative_eq!(*a, b, epsilon = 1e-7);
    }
}

#[test]
fn u09_real_sh_zeroes_the_bands_above_the_degree() {
    let d = Vec3::new(0.3, -1.2, 0.7);
    let full = real_sh(3, d);
    for degree in 0..=3u8 {
        let k = (usize::from(degree) + 1).pow(2);
        let got = real_sh(degree, d);
        assert_eq!(got[..k], full[..k], "degree {degree}");
        assert!(got[k..].iter().all(|&v| v == 0.0), "degree {degree}");
    }
}

/// The basis is orthonormal on the sphere: a Monte Carlo-free check through the
/// addition theorem, Σ_m Y_lm(d)² = (2l + 1)/(4π) for every unit d.
#[test]
fn u09_real_sh_satisfies_the_addition_theorem() {
    for d in [
        Vec3::new(0.3, -1.2, 0.7),
        Vec3::new(-0.9, 0.1, -0.4),
        Vec3::ONE,
    ] {
        let y = real_sh(3, d);
        for l in 0..=3usize {
            let sum: f64 = y[l * l..(l + 1) * (l + 1)].iter().map(|v| v * v).sum();
            let want = (2 * l + 1) as f64 / (4.0 * std::f64::consts::PI);
            assert_relative_eq!(sum, want, max_relative = 1e-12);
        }
    }
}
