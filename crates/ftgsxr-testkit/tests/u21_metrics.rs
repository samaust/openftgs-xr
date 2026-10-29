// SPDX-License-Identifier: Apache-2.0

//! U-21 (E15, D-46): `metrics` gives PSNR over the clamped RGB values, the share s of pixels
//! whose largest channel difference exceeds 1/255, and the same share s_α for alpha; 10⁻⁹.

use ftgsxr_core::RenderImage;
use ftgsxr_testkit::metrics;

const TOL: f64 = 1e-9;
const W: u32 = 100;
const H: u32 = 100;

/// A W × H image of constant colour and alpha.
fn flat(rgb: [f32; 3], alpha: f32) -> RenderImage {
    let n = (W * H) as usize;
    RenderImage {
        width: W,
        height: H,
        rgb: rgb.iter().copied().cycle().take(3 * n).collect(),
        alpha: vec![alpha; n],
    }
}

/// The pixels changed in the 1 % cases: every hundredth pixel, 100 of 10 000.
fn one_percent() -> impl Iterator<Item = usize> {
    (0..(W * H) as usize).step_by(100)
}

#[test]
fn u21_identical_images_have_infinite_psnr_and_no_differing_pixel() {
    let a = flat([0.2, 0.5, 0.9], 0.7);
    let m = metrics(&a, &a.clone());
    assert_eq!(m.psnr, f64::INFINITY);
    assert_eq!(m.share, 0.0);
    assert_eq!(m.alpha, 0.0);
}

#[test]
fn u21_one_channel_off_by_two_levels_in_one_percent_of_the_pixels() {
    let a = flat([0.0, 0.0, 0.0], 1.0);
    let mut b = a.clone();
    let offset = 2.0f32 / 255.0;
    for p in one_percent() {
        b.rgb[3 * p + 1] = offset;
    }
    let m = metrics(&a, &b);
    // By hand: 100 of the 30 000 values differ by d (the offset as stored, widened), so
    // MSE = 100 d² / 30 000 and PSNR = 10 log10(300 / d²).
    let d = f64::from(offset);
    let expected = 10.0 * (300.0 / (d * d)).log10();
    assert!((m.psnr - expected).abs() <= TOL, "{} vs {expected}", m.psnr);
    assert!((m.share - 0.01).abs() <= TOL, "{}", m.share);
    assert!(m.alpha.abs() <= TOL, "{}", m.alpha);
    // Symmetric in its arguments.
    let n = metrics(&b, &a);
    assert!((n.psnr - expected).abs() <= TOL && (n.share - 0.01).abs() <= TOL);
}

#[test]
fn u21_alpha_off_by_two_levels_in_one_percent_of_the_pixels() {
    let a = flat([0.3, 0.3, 0.3], 0.5);
    let mut b = a.clone();
    for p in one_percent() {
        b.alpha[p] = 0.5 + 2.0 / 255.0;
    }
    let m = metrics(&a, &b);
    assert_eq!(m.psnr, f64::INFINITY);
    assert!(m.share.abs() <= TOL, "{}", m.share);
    assert!((m.alpha - 0.01).abs() <= TOL, "{}", m.alpha);
}

#[test]
fn u21_values_beyond_the_unit_interval_are_clamped_before_comparing() {
    let a = flat([1.0, 0.0, 1.0], 1.0);
    let b = flat([1.5, -0.5, 1.0], 1.5);
    let m = metrics(&a, &b);
    assert_eq!(m.psnr, f64::INFINITY);
    assert_eq!(m.share, 0.0);
    assert_eq!(m.alpha, 0.0);
}

/// A 1 × 1 image.
fn pixel(rgb: [f32; 3], alpha: f32) -> RenderImage {
    RenderImage {
        width: 1,
        height: 1,
        rgb: rgb.to_vec(),
        alpha: vec![alpha],
    }
}

/// Q-41: a NaN value counts its pixel in s and s_α and makes the PSNR NaN, so a NaN in a
/// player render fails the tier rather than passing unseen.
#[test]
fn u21_a_nan_pixel_counts_as_differing() {
    let m = metrics(
        &pixel([0.5; 3], 0.5),
        &pixel([0.5, f32::NAN, 0.5], f32::NAN),
    );
    assert!(m.psnr.is_nan());
    assert_eq!((m.share, m.alpha), (1.0, 1.0));
}

/// Q-41: renders of different sizes are refused with a panic.
#[test]
#[should_panic(expected = "different sizes")]
fn u21_renders_of_different_sizes_are_refused() {
    let mut wide = pixel([0.0; 3], 0.0);
    wide.width = 2;
    metrics(&pixel([0.0; 3], 0.0), &wide);
}
