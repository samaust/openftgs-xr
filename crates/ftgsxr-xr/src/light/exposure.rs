// SPDX-License-Identifier: Apache-2.0

//! E14's exposure match, pure (§1.9, D-38, §2.1): the room's mean linear luma L_room,
//! the model's L_model from the layer probe's sums, and EV_new with the 1 % rule.

use bevy::math::URect;
use ftgsxr_app::CameraThumb;
use ftgsxr_render::LayerSums;

/// E14's EV range.
const EV_MIN: f32 = -4.0;
const EV_MAX: f32 = 4.0;
/// E14's 1 % rule: the summed α over the measured cells against their pixels.
const ALPHA_MIN_FRACTION: f64 = 0.01;
/// BT.601's luma weights (E14).
const LUMA: [f64; 3] = [0.299, 0.587, 0.114];

/// The piecewise sRGB decode.
fn srgb_decode(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// E14's exposure match's outcome.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExposureMatch {
    /// The new manual exposure, and whether E14's clamp to [−4, 4] was hit.
    Set {
        /// EV_new.
        ev: f32,
        /// The clamp was hit.
        clamped: bool,
    },
    /// The 1 % rule, no cell measured, or a luma zero or not finite: nothing changes.
    NotInView,
}

/// E14's exposure match, pure: L_room over the thumbnail's pixels (all, or those `mask`
/// sets); None for an empty thumbnail, a mask of another length than the pixel count, or
/// a mask selecting nothing.
pub fn linear_luma(thumb: &CameraThumb, mask: Option<&[bool]>) -> Option<f32> {
    let (pixels, _) = thumb.rgb.as_chunks::<3>();
    let n = pixels.len();
    if n == 0 || mask.is_some_and(|m| m.len() != n) {
        return None;
    }
    let mut sum = 0f64;
    let mut count = 0usize;
    for (i, p) in pixels.iter().enumerate() {
        if mask.is_none_or(|m| m[i]) {
            sum += (0..3)
                .map(|c| LUMA[c] * srgb_decode(f64::from(p[c])))
                .sum::<f64>();
            count += 1;
        }
    }
    (count > 0).then(|| (sum / count as f64) as f32)
}

/// The pixels of a layer `size` wide in cell `i` of a grid `cells` wide: the x with
/// ⌊cells · x / size⌋ = i (E14's partition).
fn cell_pixels(i: u32, cells: u32, size: u32) -> u64 {
    let start = |i: u32| (u64::from(i) * u64::from(size)).div_ceil(u64::from(cells));
    start(i + 1) - start(i)
}

/// EV_new from EV_old, L_room and the layer probe's sums over `cells` (half-open ranges of
/// the grid; all cells when None); `NotInView` by E14's 1 % rule, for no cell, or for a
/// luma zero or not finite.
pub fn matched_exposure_ev(
    current_ev: f32,
    room_luma: f32,
    model: &LayerSums,
    cells: Option<URect>,
) -> ExposureMatch {
    let (w, h) = (model.width, model.height);
    let rect = cells.unwrap_or(URect::new(0, 0, w, h));
    let (x0, x1) = (rect.min.x.min(w), rect.max.x.min(w));
    let (y0, y1) = (rect.min.y.min(h), rect.max.y.min(h));
    let mut luma = 0f64;
    let mut alpha = 0f64;
    let mut pixels = 0u64;
    let mut any = false;
    for j in y0..y1 {
        for i in x0..x1 {
            let Some(c) = model.cells.get((j * w + i) as usize) else {
                continue;
            };
            any = true;
            luma += (0..3).map(|k| LUMA[k] * f64::from(c[k])).sum::<f64>();
            alpha += f64::from(c[3]);
            pixels += cell_pixels(i, w, model.layer.x) * cell_pixels(j, h, model.layer.y);
        }
    }
    if !any || alpha.is_nan() || alpha < ALPHA_MIN_FRACTION * pixels as f64 || alpha <= 0.0 {
        return ExposureMatch::NotInView;
    }
    let l_model = luma / alpha;
    let l_room = f64::from(room_luma);
    let valid = |l: f64| l.is_finite() && l > 0.0;
    if !valid(l_model) || !valid(l_room) {
        return ExposureMatch::NotInView;
    }
    let raw = f64::from(current_ev) + (l_room / l_model).log2();
    let ev = raw.clamp(f64::from(EV_MIN), f64::from(EV_MAX));
    ExposureMatch::Set {
        ev: ev as f32,
        clamped: ev != raw,
    }
}
