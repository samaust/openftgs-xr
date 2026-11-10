// SPDX-License-Identifier: Apache-2.0

//! E14's exposure match, pure (§1.9, D-38, §2.1): the room's mean linear luma L_room,
//! the model's L_model from the layer probe's sums, and EV_new with the 1 % rule.

use bevy::math::{Mat4, URect, UVec2, Vec2, Vec3};
use ftgsxr_app::CameraThumb;
use ftgsxr_render::LayerSums;

/// E14's EV range.
const EV_MIN: f32 = -4.0;
const EV_MAX: f32 = 4.0;
/// E14's 1 % rule: the summed α over the measured cells against their pixels.
const ALPHA_MIN_FRACTION: f64 = 0.01;
/// BT.601's luma weights (E14).
pub(crate) const LUMA: [f64; 3] = [0.299, 0.587, 0.114];

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

/// The cells of a `grid` over the left view that a box overlaps once projected (E14's
/// measured cells while "per instance" has a target): the box's corners `min`..`max` in
/// model space through `clip_from_model`, the cells of their screen bounding rectangle,
/// half-open; all cells when a corner is at or behind the eye while another is in front
/// (the box straddles the eye plane); an empty rectangle when no corner is in front or
/// the rectangle misses the view.
// Q-397
pub(crate) fn box_cells(clip_from_model: Mat4, min: Vec3, max: Vec3, grid: UVec2) -> URect {
    let empty = URect::new(0, 0, 0, 0);
    let mut lo = Vec2::splat(f32::INFINITY);
    let mut hi = Vec2::splat(f32::NEG_INFINITY);
    let mut behind = false;
    for k in 0..8 {
        let pick = |bit: u32, a: f32, b: f32| if k & bit == 0 { a } else { b };
        let corner = Vec3::new(
            pick(1, min.x, max.x),
            pick(2, min.y, max.y),
            pick(4, min.z, max.z),
        );
        let clip = clip_from_model * corner.extend(1.0);
        if clip.w.is_nan() || clip.w <= 0.0 {
            behind = true;
            continue;
        }
        // NDC to the layer's [0, 1]², x right and y down from the top-left.
        let ndc = clip.truncate().truncate() / clip.w;
        let uv = Vec2::new((ndc.x + 1.0) / 2.0, (1.0 - ndc.y) / 2.0);
        lo = lo.min(uv);
        hi = hi.max(uv);
    }
    if !lo.is_finite() || !hi.is_finite() {
        return empty;
    }
    if behind {
        return URect::new(0, 0, grid.x, grid.y);
    }
    let size = grid.as_vec2();
    let a = (lo * size).floor().clamp(Vec2::ZERO, size);
    let b = (hi * size).ceil().clamp(Vec2::ZERO, size);
    if a.x >= b.x || a.y >= b.y {
        return empty;
    }
    URect::new(a.x as u32, a.y as u32, b.x as u32, b.y as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 90° vertical field of view, 4 : 3, Bevy's reversed infinite projection, the
    /// eye at the origin looking down −z.
    fn view() -> Mat4 {
        Mat4::perspective_infinite_reverse_rh(std::f32::consts::FRAC_PI_2, 4.0 / 3.0, 0.1)
    }

    const GRID: UVec2 = UVec2::new(64, 48);

    /// A box ±0.45 m about (0, 0, −2): its nearest face at 1.55 m spans
    /// tan = 0.45 / 1.55 = 0.290323, so x ∈ ±0.290323 · 3/4 and y ∈ ±0.290323 in NDC,
    /// u ∈ [0.391129, 0.608871] → cells 25.03…38.97, v ∈ [0.354839, 0.645161] →
    /// rows 17.03…30.97.
    #[test]
    fn box_cells_of_a_box_ahead() {
        let c = Vec3::new(0.0, 0.0, -2.0);
        let r = box_cells(view(), c - 0.45, c + 0.45, GRID);
        assert_eq!(r, URect::new(25, 17, 39, 31));
        // Moved to the right by its pose: the cells move right.
        let posed = view() * Mat4::from_translation(Vec3::new(0.5, 0.0, 0.0));
        let r2 = box_cells(posed, c - 0.45, c + 0.45, GRID);
        assert!(r2.min.x > r.min.x && r2.min.y == r.min.y, "{r2:?}");
    }

    /// An empty projection (the box behind the eye, or beside the view) measures no
    /// cell, so the match gives `NotInView`; a box around the eye measures them all.
    #[test]
    fn box_cells_empty_projection_is_not_in_view() {
        let model = LayerSums {
            cells: vec![[0.2, 0.2, 0.2, 1.0]; 64 * 48],
            width: 64,
            height: 48,
            layer: GRID,
        };
        for c in [Vec3::new(0.0, 0.0, 2.0), Vec3::new(20.0, 0.0, -2.0)] {
            let r = box_cells(view(), c - 0.45, c + 0.45, GRID);
            assert!(r.is_empty(), "{c}: {r:?}");
            assert_eq!(
                matched_exposure_ev(0.0, 0.4, &model, Some(r)),
                ExposureMatch::NotInView
            );
        }
        let r = box_cells(view(), Vec3::splat(-1.0), Vec3::splat(1.0), GRID);
        assert_eq!(r, URect::new(0, 0, 64, 48));
    }
}
