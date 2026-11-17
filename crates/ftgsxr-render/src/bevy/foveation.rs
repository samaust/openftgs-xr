// SPDX-License-Identifier: Apache-2.0

//! Each eye's fragment density map for its layer pass (§1.12 Headset display, D-58,
//! §1.4 stages 5 and 6): `SplatFoveation`, the level a camera's layer is foveated at
//! (Q-458), and the private builder of the map from a level, the layer's size and the
//! camera's projection (Q-459, Q-460, Q-457).

use ::bevy::ecs::prelude::*;
use ::bevy::math::{Mat4, UVec2};

use crate::gpu::Foveation;

/// The foveation level of a splat camera's layer pass (§1.12 Headset display, D-58).
///
/// A camera without it, or with it at `Off`, draws its layer exactly as without
/// foveation, with no density map attached, on every device (Q-457). A camera with
/// another level draws its layer with its own density map where the device has
/// `wgpu::Features::FRAGMENT_DENSITY_MAP`, and with no map elsewhere. `ftgsxr_xr` keeps it
/// on each eye equal to `DisplaySettings.foveation`; the desktop window's camera never has
/// it. Only a `LinearOver` layer is foveated: a `FileEncoded` camera draws without a map.
// Q-458
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplatFoveation(pub Foveation);

/// A density map's texels, row-major from the framebuffer's origin (top left), two bytes
/// (`Rg8Unorm`: the x and y densities) each.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DensityMap {
    /// Texels on each axis: ⌈layer ÷ texel⌉ (Q-460).
    pub(crate) size: UVec2,
    pub(crate) texels: Vec<u8>,
}

/// A level's regions (§1.12 Headset display): (radius in degrees from the optical axis,
/// density) inside out, a view angle strictly below the radius inside it, and the density
/// beyond the last; `None` at off (Q-457). Dynamic takes high's.
fn pattern(level: Foveation) -> Option<(&'static [(f32, f32)], f32)> {
    match level {
        Foveation::Off => None,
        Foveation::Low => Some((&[(25.0, 1.0)], 0.5)),
        Foveation::Medium => Some((&[(15.0, 1.0), (25.0, 0.5)], 0.25)),
        Foveation::High | Foveation::Dynamic => Some((&[(8.0, 1.0), (15.0, 0.5)], 0.25)),
    }
}

/// The density at view angle `theta` degrees from the optical axis under `level`.
fn density_at(level: Foveation, theta: f32) -> Option<f32> {
    let (rings, outer) = pattern(level)?;
    Some(
        rings
            .iter()
            .find(|(r, _)| theta < *r)
            .map_or(outer, |(_, d)| *d),
    )
}

/// The density map of a `layer`-sized layer pass drawn with the perspective
/// `clip_from_view` at `level`, on a grid of `texel` pixels (the device's minimum,
/// Q-460); `None` at off (Q-457) or for an empty texel.
///
/// The optical axis is the pixel the view's forward axis projects to, and each texel's
/// view angle is taken at its center (clamped into the layer) from the projection's
/// intrinsics at the layer's size (Q-459; `view_camera`'s mapping): c_x = (1 − P₀₂) W/2,
/// c_y = (1 + P₁₂) H/2, f_x = P₀₀ W/2, f_y = P₁₁ H/2, θ = atan‖((x − c_x)/f_x, (y − c_y)/f_y)‖.
/// A density d is stored as round(255 d) on both axes (Q-T808.1).
// Q-T808.1
pub(crate) fn density_map(
    level: Foveation,
    layer: UVec2,
    clip_from_view: Mat4,
    texel: UVec2,
) -> Option<DensityMap> {
    pattern(level)?;
    if texel.x == 0 || texel.y == 0 || layer.x == 0 || layer.y == 0 {
        return None;
    }
    let size = UVec2::new(layer.x.div_ceil(texel.x), layer.y.div_ceil(texel.y));
    let p = clip_from_view;
    let (w, h) = (layer.x as f32, layer.y as f32);
    let (fx, fy) = (p.x_axis.x * w / 2.0, p.y_axis.y * h / 2.0);
    let (cx, cy) = ((1.0 - p.z_axis.x) * w / 2.0, (1.0 + p.z_axis.y) * h / 2.0);
    let mut texels = Vec::with_capacity((size.x * size.y * 2) as usize);
    for j in 0..size.y {
        for i in 0..size.x {
            let x = ((i as f32 + 0.5) * texel.x as f32).min(w);
            let y = ((j as f32 + 0.5) * texel.y as f32).min(h);
            let theta = ((x - cx) / fx).hypot((y - cy) / fy).atan().to_degrees();
            let d = density_at(level, theta)?;
            let v = (d * 255.0).round() as u8;
            texels.extend([v, v]);
        }
    }
    Some(DensityMap { size, texels })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::bevy::math::Vec4;

    /// A reverse-Z infinite perspective from the tangents of its four half-angles
    /// (left and down negative), as an XR eye's located field of view gives it:
    /// x_ndc = 2/(r − l)·t_x − (r + l)/(r − l), and likewise in y.
    fn frustum(l: f32, r: f32, d: f32, u: f32) -> Mat4 {
        Mat4::from_cols(
            Vec4::new(2.0 / (r - l), 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0 / (u - d), 0.0, 0.0),
            Vec4::new((r + l) / (r - l), (u + d) / (u - d), 0.0, -1.0),
            Vec4::new(0.0, 0.0, 0.01, 0.0),
        )
    }

    const TEXEL: UVec2 = UVec2::new(16, 16);

    fn at(map: &DensityMap, i: u32, j: u32) -> [u8; 2] {
        let k = ((j * map.size.x + i) * 2) as usize;
        [map.texels[k], map.texels[k + 1]]
    }

    /// §1.12, D-58: with an asymmetric frustum the full-density region is centered on the
    /// pixel the optical axis projects to, within one map texel, at render scale 1.0 and
    /// 0.6 (layers 1680 × 1760 and 1008 × 1056).
    #[test]
    fn density_map_full_density_region_is_centered_on_the_optical_axis() {
        // Tangents asymmetric by different amounts in x and y.
        let (l, r, d, u) = (-1.2f32, 0.8, -1.1, 0.7);
        let proj = frustum(l, r, d, u);
        for layer in [UVec2::new(1680, 1760), UVec2::new(1008, 1056)] {
            let map = density_map(Foveation::High, layer, proj, TEXEL).expect("a map");
            // The axis pixel by hand: W·|l|/(|l| + r) across, H·u/(u + |d|) down from the top.
            let ax = layer.x as f32 * 1.2 / 2.0;
            let ay = layer.y as f32 * 0.7 / 1.8;
            let (mut sx, mut sy, mut n) = (0.0f64, 0.0f64, 0u32);
            for j in 0..map.size.y {
                for i in 0..map.size.x {
                    if at(&map, i, j) == [255, 255] {
                        sx += f64::from(i) + 0.5;
                        sy += f64::from(j) + 0.5;
                        n += 1;
                    }
                }
            }
            assert!(n > 4, "a full-density region at {layer}");
            let (mx, my) = (sx / f64::from(n) * 16.0, sy / f64::from(n) * 16.0);
            assert!(
                (mx - f64::from(ax)).abs() <= 16.0 && (my - f64::from(ay)).abs() <= 16.0,
                "{layer}: centroid ({mx}, {my}) against the axis ({ax}, {ay})"
            );
            // Not the image's center, which is far from the axis here.
            assert!((mx - f64::from(layer.x) / 2.0).abs() > 64.0);
        }
    }

    /// Q-457: off builds no map.
    #[test]
    fn density_map_off_builds_no_map() {
        let proj = frustum(-1.0, 1.0, -1.0, 1.0);
        assert_eq!(
            density_map(Foveation::Off, UVec2::new(1680, 1760), proj, TEXEL),
            None
        );
    }

    /// §1.12: dynamic takes high's pattern (and high's differs from medium's).
    #[test]
    fn density_map_dynamic_gives_highs_pattern() {
        let proj = frustum(-1.2, 0.8, -1.1, 0.7);
        let layer = UVec2::new(1008, 1056);
        let high = density_map(Foveation::High, layer, proj, TEXEL);
        assert_eq!(density_map(Foveation::Dynamic, layer, proj, TEXEL), high);
        assert_ne!(density_map(Foveation::Medium, layer, proj, TEXEL), high);
    }

    /// Q-460: the map is ⌈layer ÷ minimum texel⌉ on each axis, which meets Vulkan's
    /// minimum size ⌈framebuffer ÷ maximum texel⌉ (256 × 256 on the Quest 3).
    #[test]
    fn density_map_covers_the_layer_under_vulkans_minimum_size() {
        let proj = frustum(-1.0, 1.0, -1.0, 1.0);
        for (layer, want) in [
            (UVec2::new(1680, 1760), UVec2::new(105, 110)),
            (UVec2::new(1008, 1056), UVec2::new(63, 66)),
            (UVec2::new(1001, 17), UVec2::new(63, 2)),
        ] {
            let map = density_map(Foveation::Low, layer, proj, TEXEL).expect("a map");
            assert_eq!(map.size, want, "{layer}");
            assert_eq!(map.texels.len() as u32, want.x * want.y * 2);
            assert!(map.size.x >= layer.x.div_ceil(256) && map.size.y >= layer.y.div_ceil(256));
        }
    }

    /// §1.12's calibrated radii and densities, at hand-picked view angles.
    #[test]
    fn density_at_follows_section_1_12s_pattern() {
        use Foveation::*;
        let cases = [
            (Low, 24.9, 1.0),
            (Low, 25.1, 0.5),
            (Medium, 14.9, 1.0),
            (Medium, 15.1, 0.5),
            (Medium, 24.9, 0.5),
            (Medium, 25.1, 0.25),
            (High, 7.9, 1.0),
            (High, 8.1, 0.5),
            (High, 14.9, 0.5),
            (High, 15.1, 0.25),
        ];
        for (level, theta, d) in cases {
            assert_eq!(density_at(level, theta), Some(d), "{level:?} at {theta}°");
        }
        assert_eq!(density_at(Off, 0.0), None);
    }
}
