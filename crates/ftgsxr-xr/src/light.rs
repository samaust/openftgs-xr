// SPDX-License-Identifier: Apache-2.0

//! The light estimator's pure part (§1.9, E14, D-15, D-38): the lighting mode read from
//! the settings, the thumbnail statistics, the band mask, the smoother and the gains.

use bevy::math::Vec3;
use ftgsxr_app::{CameraThumb, FeatureState};
use ftgsxr_io::{EnvironmentSettings, ManualLighting};
use ftgsxr_render::ColorTransform;

/// E14's clamp of g and each w_c.
const GAIN_MIN: f32 = 0.25;
const GAIN_MAX: f32 = 4.0;
/// E14's local fallback: fewer than 1 % of the pixels in the band take the frame.
const BAND_MIN_FRACTION: f32 = 0.01;

/// §1.9's four lighting modes (E14).
#[derive(Clone, Debug, PartialEq)]
pub enum LightingMode {
    /// No transform.
    Off,
    /// E14's manual sliders.
    Manual(ManualLighting),
    /// E14 over the whole camera frame.
    AutoGlobal,
    /// E14 over the selected instance's depth band.
    AutoLocal,
}

impl LightingMode {
    /// The effective mode: `lighting_mode`'s spellings (Q-50: `off`, `manual`,
    /// `auto-global`, `auto-local`; any other reads as `off`), an auto mode counting as
    /// `Manual` with the manual sliders unless `camera` is `Available`.
    // Q-382, Q-384
    pub fn from_settings(env: &EnvironmentSettings, camera: FeatureState) -> LightingMode {
        let manual = || LightingMode::Manual(env.lighting_manual.clone());
        let auto = |mode| {
            if camera == FeatureState::Available {
                mode
            } else {
                manual()
            }
        };
        match env.lighting_mode.as_str() {
            "manual" => manual(),
            "auto-global" => auto(LightingMode::AutoGlobal),
            "auto-local" => auto(LightingMode::AutoLocal),
            _ => LightingMode::Off,
        }
    }
}

/// E14's estimate: the mean luma Ȳ and the mean RGB (r̄, ḡ, b̄) over Ω.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightEstimate {
    /// Ȳ.
    pub luma: f32,
    /// (r̄, ḡ, b̄).
    pub rgb: [f32; 3],
}

impl LightEstimate {
    /// The mean RGB over the masked pixels (all when `mask` is `None`) and BT.709's luma
    /// of that mean; `None` for an empty thumbnail, a mask of another length than the
    /// pixel count, or a mask selecting nothing.
    // Q-382
    pub fn from_thumb(thumb: &CameraThumb, mask: Option<&[bool]>) -> Option<LightEstimate> {
        let (pixels, _) = thumb.rgb.as_chunks::<3>();
        let n = pixels.len();
        if n == 0 || mask.is_some_and(|m| m.len() != n) {
            return None;
        }
        let mut sum = [0f64; 3];
        let mut count = 0usize;
        for (i, p) in pixels.iter().enumerate() {
            if mask.is_none_or(|m| m[i]) {
                for c in 0..3 {
                    sum[c] += f64::from(p[c]);
                }
                count += 1;
            }
        }
        if count == 0 {
            return None;
        }
        let rgb = sum.map(|s| (s / count as f64) as f32);
        let luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        Some(LightEstimate { luma, rgb })
    }
}

/// E14's local region: `Some` of the pixels whose depth is finite and within
/// `d_m ± delta_m`, `None` when fewer than 1 % qualify (the caller then takes the frame).
// Q-382
pub fn band_mask(depths_m: &[f32], d_m: f32, delta_m: f32) -> Option<Vec<bool>> {
    let mask: Vec<bool> = depths_m
        .iter()
        .map(|&z| z.is_finite() && (z - d_m).abs() <= delta_m)
        .collect();
    let qualifying = mask.iter().filter(|&&b| b).count();
    if depths_m.is_empty() || (qualifying as f32) < BAND_MIN_FRACTION * depths_m.len() as f32 {
        return None;
    }
    Some(mask)
}

/// E14's exponential moving average of the estimate, time constant τ.
// Q-382
#[derive(Clone, Debug)]
pub struct LightSmoother {
    time_constant_s: f32,
    state: Option<(LightEstimate, f64)>,
}

impl LightSmoother {
    /// A smoother with time constant τ = `time_constant_s` (E14: 0.5 s), empty.
    pub fn new(time_constant_s: f32) -> LightSmoother {
        LightSmoother {
            time_constant_s,
            state: None,
        }
    }

    /// Adds the sample taken at `at_s` and returns the smoothed estimate: the first
    /// sample as it is, then each component moved by 1 − exp(−Δt/τ), Δt from the previous
    /// sample's `at_s`; a Δt ≤ 0 leaves the estimate unchanged.
    pub fn push(&mut self, sample: LightEstimate, at_s: f64) -> LightEstimate {
        let next = match self.state {
            None => (sample, at_s),
            Some((current, last)) => {
                let dt = at_s - last;
                if dt <= 0.0 {
                    return current;
                }
                let k = 1.0 - (-(dt / f64::from(self.time_constant_s))).exp();
                let k = k as f32;
                let mix = |a: f32, b: f32| a + k * (b - a);
                let rgb = [0, 1, 2].map(|c| mix(current.rgb[c], sample.rgb[c]));
                (
                    LightEstimate {
                        luma: mix(current.luma, sample.luma),
                        rgb,
                    },
                    at_s,
                )
            }
        };
        self.state = Some(next);
        next.0
    }

    /// The smoothed estimate, `None` before the first sample.
    pub fn current(&self) -> Option<LightEstimate> {
        self.state.map(|(e, _)| e)
    }

    /// Forgets the samples (the camera restarted).
    pub fn reset(&mut self) {
        self.state = None;
    }
}

/// a / b, read as 1 when b is zero or the quotient is not finite (Q-382).
fn ratio(a: f32, b: f32) -> f32 {
    let r = a / b;
    if b == 0.0 || !r.is_finite() { 1.0 } else { r }
}

/// w scaled to mean 1 and each channel clamped to E14's range.
fn normalize_clamp(w: [f32; 3]) -> Vec3 {
    let mean = (w[0] + w[1] + w[2]) / 3.0;
    let w = if mean > 0.0 && mean.is_finite() {
        w.map(|c| c / mean)
    } else {
        [1.0; 3]
    };
    Vec3::from_array(w.map(|c| c.clamp(GAIN_MIN, GAIN_MAX)))
}

/// E14, pure: the gains from the smoothed estimate and the reference.
// Q-382
pub fn light_gains(
    estimate: &LightEstimate,
    reference: &LightEstimate,
    gamma: f32,
) -> ColorTransform {
    let g = ratio(estimate.luma, reference.luma)
        .powf(gamma)
        .clamp(GAIN_MIN, GAIN_MAX);
    let w = [0, 1, 2].map(|c| {
        ratio(
            ratio(estimate.rgb[c], estimate.luma),
            ratio(reference.rgb[c], reference.luma),
        )
    });
    ColorTransform {
        gain: g * normalize_clamp(w),
    }
}

/// E14's manual mode: `ColorTransform::from_manual` (Q-250).
pub fn manual_gains(manual: &ManualLighting) -> ColorTransform {
    ColorTransform::from_manual(manual)
}

/// E14's grey-world mode (D-38): g = 1, w_c = Ȳ/c̄ scaled to mean 1 and clamped; the
/// identity when any channel mean is 0.
// Q-382
pub fn grey_world_gains(estimate: &LightEstimate) -> ColorTransform {
    if estimate.rgb.contains(&0.0) {
        return ColorTransform::IDENTITY;
    }
    let w = estimate.rgb.map(|c| ratio(estimate.luma, c));
    ColorTransform {
        gain: normalize_clamp(w),
    }
}
