// SPDX-License-Identifier: Apache-2.0

//! Auto local's band (E14, Q-388): the left view's depth image read back from the GPU,
//! E13's decode mirrored in Rust, and each thumbnail pixel's environment depth.

use bevy::log::{info, warn};
use bevy::math::{DMat3, DVec3};
use bevy::prelude::{Commands, Entity, On, Query, Res, ResMut, Resource, With};
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use ftgsxr_app::{Camera, CameraThumb, FeatureState, Selected, SettingsRes, XrFeatures};
use ftgsxr_render::SplatOcclusion;

use super::LightingMode;
use crate::passthrough::{Environment, EnvironmentMode};

/// A readback still pending after this many frames is dropped (PRD #664's spike check
/// 6's fallback: the band stays on the older depth, or the frame).
const READBACK_FRAMES_MAX: u32 = 10;
/// The frames a readback is expected to take (§2.4 rule 1's two frames late).
const READBACK_FRAMES_EXPECTED: u32 = 2;

/// Auto local's depth (Q-388): the newest left view's image read back, and the one-shot
/// readback in flight.
#[derive(Resource, Default)]
pub(crate) struct DepthReadback {
    /// The newest snapshot, `None` while the provider is stopped.
    pub(crate) latest: Option<DepthSnapshot>,
    /// The readback's entity and the frames since it was spawned.
    pending: Option<(Entity, u32)>,
    /// A readback completed in this process (the first is logged, spike check 6).
    logged: bool,
}

/// The left view's depth image as read back: its stored depths, top row first, with
/// the camera and planes of the `DepthFrame` it came from.
#[derive(Clone, Debug)]
pub(crate) struct DepthSnapshot {
    /// `camera.width × camera.height` stored depths d.
    pub(crate) stored: Vec<f32>,
    /// The depth camera in stage space (E6, Q-341).
    pub(crate) camera: Camera,
    pub(crate) near: f32,
    pub(crate) far: f32,
}

/// E13's decode (Q-340), `occlusion.wgsl`'s `depth_metres` mirrored: the stored depth d
/// of an OpenXR projection (zero at `near`, one at `far`) as metres along the depth
/// camera's z axis, Z = f·n / (f − d·(f − n)), or Z = n / (1 − d) with an infinite far
/// plane (`far < near` or `far` not finite); `None` unless 0 ≤ d < 1 and Z > 0.
// Q-340, Q-388
pub(crate) fn stored_depth_metres(d: f32, near: f32, far: f32) -> Option<f32> {
    if !(0.0..1.0).contains(&d) {
        return None;
    }
    let z = if far < near || !far.is_finite() {
        near / (1.0 - d)
    } else {
        far * near / (far - d * (far - near))
    };
    (z > 0.0).then_some(z)
}

/// A readback's bytes as the `width × height` floats of an `R32Float` image, rows top
/// first, each row padded to 256 bytes as `gpu_readback` lays the copy out; `None` when
/// the bytes are too few.
// Q-388
pub(crate) fn r32_floats(data: &[u8], width: u32, height: u32) -> Option<Vec<f32>> {
    let (w, h) = (width as usize, height as usize);
    let row = 4 * w;
    let stride = row.div_ceil(256) * 256;
    if h == 0 || data.len() < (h - 1) * stride + row {
        return None;
    }
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        let (texels, _) = data[y * stride..y * stride + row].as_chunks::<4>();
        out.extend(texels.iter().map(|b| f32::from_le_bytes(*b)));
    }
    Some(out)
}

/// Each thumbnail pixel's environment depth in metres along the depth camera's z axis
/// (E13's decode), `NaN` where the pixel has none: the ray through the pixel's centre and
/// `thumb.camera`'s intrinsics, rotated into the depth camera with the two centres taken
/// as coincident, projected into the depth image, the nearest texel decoded; a texel
/// invalid to the decode, a ray behind the depth camera or a projection outside the image
/// gives `NaN`. Without `thumb.camera` the pixel maps by normalized image coordinates.
/// `thumb.width × thumb.height` values, row-major.
// Q-388
pub(crate) fn thumb_depths(thumb: &CameraThumb, depth: &DepthSnapshot) -> Vec<f32> {
    let (tw, th) = (thumb.width as usize, thumb.height as usize);
    let c = &depth.camera;
    let (dw, dh) = (f64::from(c.width), f64::from(c.height));
    let texel = |x: f64, y: f64| -> f32 {
        if !(0.0..dw).contains(&x) || !(0.0..dh).contains(&y) {
            return f32::NAN;
        }
        let i = y as usize * c.width as usize + x as usize;
        depth
            .stored
            .get(i)
            .and_then(|&d| stored_depth_metres(d, depth.near, depth.far))
            .unwrap_or(f32::NAN)
    };
    // The thumbnail's pixel ray to the depth camera's: R_d R_tᵀ K_t⁻¹ (centres coincident).
    let to_depth = thumb.camera.as_ref().map(|t| {
        let r_t = DMat3::from_mat4(t.world_to_camera);
        let r_d = DMat3::from_mat4(c.world_to_camera);
        (r_d * r_t.transpose() * t.k.inverse(), c.k)
    });
    let mut out = Vec::with_capacity(tw * th);
    for v in 0..th {
        for u in 0..tw {
            let (pu, pv) = (u as f64 + 0.5, v as f64 + 0.5);
            out.push(match &to_depth {
                Some((m, k)) => {
                    let q = *m * DVec3::new(pu, pv, 1.0);
                    if q.z > 0.0 {
                        let p = *k * (q / q.z);
                        texel(p.x, p.y)
                    } else {
                        f32::NAN
                    }
                }
                None => texel(pu / tw as f64 * dw, pv / th as f64 * dh),
            });
        }
    }
    out
}

/// Whether a readback is spawned for the left view's capture `age_frames` old: the
/// effective `mode` auto local, an instance selected (the band needs d_m), passthrough
/// mode and a new capture (Q-388).
// Q-388
fn readback_wanted(
    mode: &LightingMode,
    selected: bool,
    passthrough: bool,
    age_frames: u32,
) -> bool {
    matches!(mode, LightingMode::AutoLocal) && selected && passthrough && age_frames == 0
}

/// The frame's request (main world, `Update`, before the control): while auto local is
/// in force (`XrFeatures.lighting` as of the last frame's control), an instance selected,
/// in passthrough mode and the provider runs, a one-shot `Readback::texture` of the
/// left view's `DepthFrame.image` is spawned when a new capture arrived (`age_frames`
/// 0) and none is in flight. Its `Readback` is removed the frame after (one extract
/// copies it once, encoded by Bevy's render system after the drive's median pass was
/// submitted in `PrepareResources`), the entity kept for `ReadbackComplete`. Nothing
/// waits for the GPU (§2.4 rule 1).
// Q-388
pub(crate) fn request(
    mut rb: ResMut<DepthReadback>,
    occlusion: Option<Res<SplatOcclusion>>,
    settings: Option<Res<SettingsRes>>,
    env: Option<Res<Environment>>,
    features: Option<Res<XrFeatures>>,
    selected: Query<(), With<Selected>>,
    mut commands: Commands,
) {
    let left = occlusion.as_deref().and_then(|o| o.frames.first());
    if left.is_none() {
        rb.latest = None;
    }
    if let Some((entity, frames)) = rb.pending.as_mut() {
        *frames += 1;
        let (entity, frames) = (*entity, *frames);
        if frames == 1 {
            commands.entity(entity).remove::<Readback>();
        } else if frames > READBACK_FRAMES_MAX {
            warn!("light: the depth readback did not return in {frames} frames; dropped");
            commands.entity(entity).despawn();
            rb.pending = None;
        }
        return;
    }
    let camera = features.map_or(FeatureState::Absent, |f| f.lighting);
    let mode = settings.map_or(LightingMode::Off, |s| {
        LightingMode::from_settings(&s.0.environment, camera)
    });
    let passthrough = env.is_some_and(|e| e.mode == EnvironmentMode::Passthrough);
    if let Some(f) = left
        && readback_wanted(&mode, !selected.is_empty(), passthrough, f.age_frames)
    {
        let entity = commands
            .spawn(Readback::texture(f.image.clone()))
            .observe(complete)
            .id();
        rb.pending = Some((entity, 0));
    }
}

/// The readback's floats kept as the snapshot, with the camera and planes of the left
/// view's frame current now: a capture newer than the one asked for (30 Hz against the
/// display's 72 Hz) is read whole, since the copy follows that frame's median pass, and
/// is paired with its own camera unless it arrived in the frame of the copy itself (a few
/// milliseconds of head motion, against Δ = 0.5 m).
// Q-388
fn complete(
    event: On<ReadbackComplete>,
    mut rb: ResMut<DepthReadback>,
    occlusion: Option<Res<SplatOcclusion>>,
    mut commands: Commands,
) {
    let entity = event.event().entity;
    commands.entity(entity).despawn();
    let frames = match rb.pending {
        Some((e, f)) if e == entity => f,
        _ => return,
    };
    rb.pending = None;
    let Some(f) = occlusion.as_deref().and_then(|o| o.frames.first()) else {
        return;
    };
    let Some(stored) = r32_floats(&event.data, f.camera.width, f.camera.height) else {
        warn!("light: the depth readback is shorter than its image; dropped");
        return;
    };
    if !rb.logged || frames > READBACK_FRAMES_EXPECTED {
        rb.logged = true;
        info!(
            "light: the depth read back in {frames} frames ({} x {})",
            f.camera.width, f.camera.height
        );
    }
    rb.latest = Some(DepthSnapshot {
        stored,
        camera: f.camera,
        near: f.near,
        far: f.far,
    });
}

#[cfg(test)]
mod tests {
    use bevy::math::{Quat, Vec3};
    use ftgsxr_app::{FeatureState, Fov};

    use super::super::band_mask;
    use super::*;

    /// Q-388: the readback is wanted only in the effective auto local mode (an auto local
    /// setting without the camera is manual), with an instance selected, in passthrough
    /// mode, for a new capture.
    #[test]
    fn readback_wanted_needs_effective_auto_local_a_selection_and_a_new_capture() {
        let mut env = ftgsxr_io::Settings::default().environment;
        env.lighting_mode = "auto-local".into();
        let local = LightingMode::from_settings(&env, FeatureState::Available);
        assert!(readback_wanted(&local, true, true, 0));
        assert!(!readback_wanted(&local, false, true, 0), "no selection");
        assert!(!readback_wanted(&local, true, false, 0), "VR mode");
        assert!(!readback_wanted(&local, true, true, 1), "no new capture");
        let gated = LightingMode::from_settings(&env, FeatureState::Absent);
        assert!(
            !readback_wanted(&gated, true, true, 0),
            "auto local gated to manual"
        );
        assert!(!readback_wanted(&LightingMode::AutoGlobal, true, true, 0));
    }

    /// Q-340's hand-worked values, the ones `passes.rs`'s
    /// `gpu_depth_metres_decodes_hand_worked_values` pins the shader on: with n = 0.25
    /// and f = 10, d = 0 is 0.25 m, d = 0.5 is 2.5/5.125 m and d = 0.975 is 2.5/0.49375 m;
    /// with the infinite far plane (f < n, or f = +∞) d = 0.5 is 0.5 m and d = 0.875 is
    /// 2 m; d = 1, d < 0 and a NaN are invalid.
    #[test]
    fn stored_depth_metres_agrees_with_the_shaders_hand_worked_values() {
        let inf = f32::INFINITY;
        let cases: [(f32, f32, f32, Option<f64>); 12] = [
            (0.0, 0.25, 10.0, Some(0.25)),
            (0.5, 0.25, 10.0, Some(2.5 / 5.125)),
            (0.975, 0.25, 10.0, Some(2.5 / 0.49375)),
            (1.0, 0.25, 10.0, None),
            (-0.1, 0.25, 10.0, None),
            (f32::NAN, 0.25, 10.0, None),
            (0.5, 0.25, 0.0, Some(0.5)),
            (0.875, 0.25, 0.0, Some(2.0)),
            (1.0, 0.25, 0.0, None),
            (0.5, 0.25, inf, Some(0.5)),
            (0.875, 0.25, inf, Some(2.0)),
            (1.0, 0.25, inf, None),
        ];
        for (d, n, f, want) in cases {
            let got = stored_depth_metres(d, n, f);
            match want {
                None => assert!(got.is_none(), "d {d} n {n} f {f}: {got:?}"),
                Some(z) => {
                    let g = got.unwrap_or_else(|| panic!("d {d} n {n} f {f}: invalid"));
                    assert!((f64::from(g) - z).abs() < 1e-5, "d {d} n {n} f {f}: {g}");
                }
            }
        }
    }

    /// A 5 × 2 image's rows are 20 bytes padded to 256: the floats come back row by
    /// row, the padding skipped; too few bytes give `None`.
    #[test]
    fn r32_floats_skips_the_row_padding() {
        let mut data = vec![0xffu8; 256 + 20];
        for (i, v) in [0.0f32, 0.1, 0.2, 0.3, 0.4].iter().enumerate() {
            data[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (i, v) in [1.0f32, 1.1, 1.2, 1.3, 1.4].iter().enumerate() {
            data[256 + 4 * i..256 + 4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        assert_eq!(
            r32_floats(&data, 5, 2),
            Some(vec![0.0, 0.1, 0.2, 0.3, 0.4, 1.0, 1.1, 1.2, 1.3, 1.4])
        );
        assert_eq!(r32_floats(&data[..256 + 19], 5, 2), None);
    }

    /// ±45° each way.
    const FOV: Fov = Fov::Angles {
        left: -std::f32::consts::FRAC_PI_4,
        right: std::f32::consts::FRAC_PI_4,
        up: std::f32::consts::FRAC_PI_4,
        down: -std::f32::consts::FRAC_PI_4,
    };

    fn thumb(camera: Option<Camera>) -> CameraThumb {
        CameraThumb {
            width: 64,
            height: 48,
            rgb: vec![0.5; 64 * 48 * 3],
            at_s: 0.0,
            camera,
        }
    }

    /// E14 local, Q-388: a 320 × 320 depth image (near 0.1 m, the infinite far plane)
    /// seen by the thumbnail's own pose and field of view, a wall at 2 m (d = 1 − 0.1/2 =
    /// 0.95) over its left half and nothing (d = 1) over its right half. d_m = 2 m gives
    /// the thumbnail's left half (32 of 64 columns, every row), d_m = 4 m fewer than 1 %,
    /// so the frame; the same without the thumbnail's camera (normalized coordinates);
    /// a thumbnail camera turned away from the depth camera sees no depth at all.
    #[test]
    fn thumb_depths_maps_a_wall_over_half_the_thumbnail() {
        let at = Vec3::new(0.3, 1.6, -0.2);
        let turn = Quat::from_rotation_y(0.4);
        let depth = DepthSnapshot {
            stored: (0..320 * 320)
                .map(|i| if i % 320 < 160 { 0.95 } else { 1.0 })
                .collect(),
            camera: Camera::from_stage_pose(at, turn, FOV, 320, 320),
            near: 0.1,
            far: f32::INFINITY,
        };
        let half: Vec<bool> = (0..64 * 48).map(|i| i % 64 < 32).collect();
        for camera in [Some(Camera::from_stage_pose(at, turn, FOV, 64, 48)), None] {
            let label = format!("camera {}", camera.is_some());
            let z = thumb_depths(&thumb(camera), &depth);
            assert_eq!(z.len(), 64 * 48, "{label}");
            assert_eq!(band_mask(&z, 2.0, 0.5), Some(half.clone()), "{label}");
            assert_eq!(band_mask(&z, 4.0, 0.5), None, "{label}");
        }
        let away = Camera::from_stage_pose(at, turn * Quat::from_rotation_y(3.0), FOV, 64, 48);
        let z = thumb_depths(&thumb(Some(away)), &depth);
        assert!(z.iter().all(|z| z.is_nan()));
    }
}
