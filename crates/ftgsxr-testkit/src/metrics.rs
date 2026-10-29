// SPDX-License-Identifier: Apache-2.0

//! E15's conformance metrics (D-46, Q-1).

use ftgsxr_core::RenderImage;

/// E15's three numbers for two renders of one request item (Q-1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Metrics {
    /// PSNR in dB over the H·W·3 clamped RGB values; `f64::INFINITY` when the MSE is 0.
    pub psnr: f64,
    /// s: the share of pixels whose largest RGB channel difference exceeds 1/255.
    pub share: f64,
    /// s_α: the share of pixels whose alpha difference exceeds 1/255.
    pub alpha: f64,
}

/// A difference above this counts a pixel in s or s_α (E15).
const LEVEL: f64 = 1.0 / 255.0;

/// E15 for renders `a` and `b`, computed in float64 after clamping every RGB and alpha value
/// of both to [0, 1] (D-46): PSNR = 10 log10(1 / MSE) over the H·W·3 RGB values, infinite
/// at MSE = 0; s and s_α count a pixel when its largest RGB difference, or its alpha
/// difference, exceeds 1/255. A NaN value counts its pixel as differing and makes the PSNR
/// NaN (Q-41).
///
/// # Panics
///
/// When the two renders differ in size, or either one's arrays do not hold W·H·3 and W·H
/// values.
pub fn metrics(a: &RenderImage, b: &RenderImage) -> Metrics {
    assert_eq!(
        (a.width, a.height),
        (b.width, b.height),
        "metrics: renders of different sizes"
    );
    let pixels = a.width as usize * a.height as usize;
    for image in [a, b] {
        assert!(
            image.rgb.len() == 3 * pixels && image.alpha.len() == pixels,
            "metrics: a {}×{} render needs {} RGB and {pixels} alpha values, has {} and {}",
            image.width,
            image.height,
            3 * pixels,
            image.rgb.len(),
            image.alpha.len()
        );
    }
    let mut squared = 0.0f64;
    let mut differing = 0usize;
    for (pa, pb) in a
        .rgb
        .as_chunks::<3>()
        .0
        .iter()
        .zip(b.rgb.as_chunks::<3>().0)
    {
        let mut over = false;
        for (&ca, &cb) in pa.iter().zip(pb) {
            let d = unit(ca) - unit(cb);
            squared += d * d;
            over |= exceeds(d);
        }
        differing += usize::from(over);
    }
    let alpha_differing = a
        .alpha
        .iter()
        .zip(&b.alpha)
        .filter(|&(&x, &y)| exceeds(unit(x) - unit(y)))
        .count();
    let mse = squared / (3 * pixels) as f64;
    Metrics {
        psnr: if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (1.0 / mse).log10()
        },
        share: differing as f64 / pixels as f64,
        alpha: alpha_differing as f64 / pixels as f64,
    }
}

/// A value widened to float64 and clamped to [0, 1] (D-46); NaN stays NaN.
fn unit(value: f32) -> f64 {
    f64::from(value).clamp(0.0, 1.0)
}

/// Whether a difference counts its pixel: more than 1/255, or NaN (Q-41).
fn exceeds(difference: f64) -> bool {
    difference.is_nan() || difference.abs() > LEVEL
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exceeds_is_strict_at_one_level() {
        assert!(!exceeds(LEVEL));
        assert!(!exceeds(-LEVEL));
        assert!(exceeds(LEVEL * (1.0 + 1e-12)));
        assert!(exceeds(f64::NAN));
    }
}
