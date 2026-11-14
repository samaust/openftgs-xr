// SPDX-License-Identifier: Apache-2.0

//! R-12 (§3.2), E13 through `SplatPasses::encode_draw` with an `OcclusionInput`: a
//! synthetic depth image of a wall at 2 m hides a primitive at 3 m and leaves one at 1 m;
//! the soft weights across z_e ± δ within 10⁻³; a wall tilted 45° crossing a large
//! primitive gives an occlusion edge within 1 px of its analytic position, following each
//! fragment's own ray depth (D-45). The tier is the conformance tier with
//! `occlusion: true` (Q-347); the images are written as an OpenXR projection writes them
//! (Q-340, `stored_depth`), and each occluded frame is compared with the same frame drawn
//! without an input, so the weight is the ratio of the two layers' alphas. The wall's
//! visibility clause is also checked in shared-center mode for two eyes (D-4, #721), each
//! eye's fragments weighed at that eye's z\_c (Q-400).
//!
//! Q-445's exactness check (#790): the occluded layers of every R-12 scene, per view and in
//! shared-center mode in both record layouts, against those recorded at the commit before
//! an exact change to E13's evaluation, change no pixel by E15's rule (s = 0, s\_α = 0).

use std::path::PathBuf;

use ftgsxr_core::{Camera, DMat3, DMat4, DVec3, RenderImage, UVec2, occlusion_weight};
use ftgsxr_render::{ColorTransform, RenderMode, SplatDevice};
use ftgsxr_testkit::{gpu_context, layer_messages, metrics};

use crate::display_support::{
    Occlusion, Row, Run, draw, model, render_occluded, render_shared_stereo, stored_depth, upload,
};

const W: u32 = 64;
const H: u32 = 48;
const RUN: Run = Run {
    resolve: None,
    depth: false,
};

/// A pinhole camera of `w` × `h` with focal `f` and the principal point at the center,
/// whose world-to-camera rotation is `r` and whose center is `c` (E6).
fn camera(w: u32, h: u32, f: f64, r: DMat3, c: DVec3) -> Camera {
    let mut w2c = DMat4::from_mat3(r);
    w2c.w_axis = (-(r * c)).extend(1.0);
    Camera {
        width: w,
        height: h,
        k: DMat3::from_cols(
            DVec3::new(f, 0.0, 0.0),
            DVec3::new(0.0, f, 0.0),
            DVec3::new(f64::from(w) / 2.0, f64::from(h) / 2.0, 1.0),
        ),
        world_to_camera: w2c,
    }
}

/// The view: 64 × 48, focal 60, at the origin with the world's axes (x right, y down,
/// z forward), so a point's z is its depth z_c.
fn view() -> Camera {
    camera(W, H, 60.0, DMat3::IDENTITY, DVec3::ZERO)
}

fn row(position: [f32; 3], scale: f32, opacity: f32) -> Row {
    Row {
        position,
        velocity: [0.0; 3],
        time_center: 0.95,
        duration: 100.0,
        scale,
        opacity,
        color: [0.8, 0.6, 0.4],
    }
}

/// The layer's alphas of `rows` drawn from the view at τ = 0.95, without and with
/// `occlusion`.
fn alphas(dev: &SplatDevice, rows: &[Row], occlusion: &Occlusion) -> (Vec<f32>, Vec<f32>) {
    let m = upload(dev, &model(rows));
    let draws = [draw(&m, 0.95, RenderMode::Splats, ColorTransform::IDENTITY)];
    let alpha = |f: crate::display_support::Frame| f.layer.iter().map(|p| p[3]).collect();
    let reference = alpha(render_occluded(dev, &draws, &view(), &RUN, None));
    let occluded = alpha(render_occluded(dev, &draws, &view(), &RUN, Some(occlusion)));
    (reference, occluded)
}

/// R-12: a wall at 2 m (stored with nearZ 0.25 and farZ 20) over the right half of the
/// image, the left half invalid (d = 1, the far plane): the primitive at 3 m behind the
/// wall is hidden, the one at 1 m in front of it is drawn as without occlusion, and the
/// one at 3 m in the invalid half is drawn as without occlusion (E13: w_occ = 1). The
/// image is bound as `R32Float` and as `Depth16Unorm`, the depth swapchain's expected
/// format (#570), whose 16-bit rounding leaves the wall far from both primitives.
#[test]
#[ignore = "gpu"]
fn gpu_r12_a_wall_at_2_m_hides_3_m_and_keeps_1_m() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let (near, far) = (0.25, 20.0);
    let wall = stored_depth(2.0, near, far);
    for format in [
        wgpu::TextureFormat::R32Float,
        wgpu::TextureFormat::Depth16Unorm,
    ] {
        wall_at_2_m(&dev, near, far, wall, format);
    }
}

/// The wall scene of `gpu_r12_a_wall_at_2_m_hides_3_m_and_keeps_1_m`: its rows and its
/// depth image, the wall stored as `wall` in `format`.
fn wall_scene(
    near: f64,
    far: f64,
    wall: f32,
    format: wgpu::TextureFormat,
) -> ([Row; 3], Occlusion) {
    let depth = (0..W * H)
        .map(|i| if i % W < W / 2 { 1.0 } else { wall })
        .collect();
    let occlusion = Occlusion {
        depth,
        size: UVec2::new(W, H),
        format,
        camera: view(),
        near: near as f32,
        far: far as f32,
        soft: true,
        delta_m: 0.05,
    };
    // u = 32 + 60 x / z, v = 24 + 60 y / z: (8, 24) in the invalid half, (44, 18) behind
    // the wall, (44, 30) in front of it.
    let rows = [
        row([-1.2, 0.0, 3.0], 0.03, 0.9),
        row([0.6, -0.3, 3.0], 0.03, 0.9),
        row([0.2, 0.1, 1.0], 0.02, 0.9),
    ];
    (rows, occlusion)
}

fn wall_at_2_m(dev: &SplatDevice, near: f64, far: f64, wall: f32, format: wgpu::TextureFormat) {
    let (rows, occlusion) = wall_scene(near, far, wall, format);
    let (reference, occluded) = alphas(dev, &rows, &occlusion);
    let region = |u: u32, v: u32| match (u < W / 2, v < H / 2) {
        (true, _) => "invalid",
        (false, true) => "behind",
        (false, false) => "front",
    };
    for name in ["invalid", "behind", "front"] {
        let covered = (0..W * H)
            .filter(|&i| region(i % W, i / W) == name)
            .filter(|&i| reference[i as usize] > 0.1)
            .count();
        assert!(
            covered > 0,
            "{name}: the primitive is drawn without occlusion"
        );
    }
    for i in 0..(W * H) as usize {
        let (u, v) = (i as u32 % W, i as u32 / W);
        let expected = match region(u, v) {
            "behind" => 0.0,
            _ => reference[i],
        };
        assert_eq!(
            occluded[i],
            expected,
            "pixel ({u}, {v}) in {}",
            region(u, v)
        );
    }
}

/// The soft-weight scenes' cases: soft or hard, and the primitive at z_e + kδ.
const SOFT_CASES: [(bool, f32); 11] = [
    (true, -1.2),
    (true, -1.0),
    (true, -0.5),
    (true, -0.25),
    (true, 0.0),
    (true, 0.25),
    (true, 0.5),
    (true, 1.0),
    (true, 1.2),
    (false, 0.9),
    (false, 1.1),
];

/// The scene of `gpu_r12_the_soft_weight_across_z_e_plus_minus_delta` for one case: the
/// primitive's depth z, its row and the depth image of the wall at 2 m (δ = 5 cm, stored
/// with the infinite far plane).
fn soft_scene(soft: bool, k: f32) -> (f32, [Row; 1], Occlusion) {
    let (near, far) = (0.2, 0.1); // far < near: the infinite far plane
    let delta = 0.05f32;
    let wall = stored_depth(2.0, near, far);
    let occlusion = Occlusion {
        depth: vec![wall; (W * H) as usize],
        size: UVec2::new(W, H),
        format: wgpu::TextureFormat::R32Float,
        camera: view(),
        near: near as f32,
        far: far as f32,
        soft,
        delta_m: delta,
    };
    let z = 2.0 + k * delta;
    (z, [row([0.0, 0.0, z], 0.05, 0.999)], occlusion)
}

/// R-12: the soft weight across z_e ± δ (E13, δ = 5 cm, the wall at z_e = 2 m stored with
/// the infinite far plane, farZ < nearZ): a primitive at 2 m + s, for s from −1.2δ to
/// 1.2δ, drawn with the depth camera the view's, so every fragment's z_f is z_c = 2 + s;
/// where the unoccluded alpha exceeds 0.9 the ratio of the alphas is
/// `ftgsxr_core::occlusion_weight` (the CPU oracle of Q-63, U-19) within 10⁻³; the hard
/// weight is 1 at s = 0.9δ and 0 at 1.1δ.
#[test]
#[ignore = "gpu"]
fn gpu_r12_the_soft_weight_across_z_e_plus_minus_delta() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    for (soft, k) in SOFT_CASES {
        let (z, rows, occlusion) = soft_scene(soft, k);
        let delta = occlusion.delta_m;
        let (reference, occluded) = alphas(&dev, &rows, &occlusion);
        let expected = occlusion_weight(z, Some(2.0), delta, soft);
        let mut covered = 0;
        for i in 0..(W * H) as usize {
            if reference[i] > 0.9 {
                covered += 1;
                let w = occluded[i] / reference[i];
                assert!(
                    (w - expected).abs() <= 1e-3,
                    "soft {soft}, z_f = z_e + {k}δ, pixel {i}: weight {w}, expected {expected}"
                );
            }
        }
        assert!(
            covered > 0,
            "soft {soft}, {k}δ: the primitive covers pixels"
        );
    }
}

/// The scene of `gpu_r12_a_45_degree_wall_gives_the_edge_of_each_fragments_own_ray`: the
/// edge's column u*, the large primitive's row and the tilted wall's depth image.
fn tilted_scene() -> (f64, [Row; 1], Occlusion) {
    let u_star = 21.3f64;
    let x_star = 2.0 * (u_star - 32.0) / 60.0;
    let (near, far) = (0.1, 50.0);
    let (dw, dh, df) = (384u32, 288u32, 240.0);
    let turn = DMat3::from_rotation_y(5f64.to_radians());
    let center = DVec3::new(0.03, -0.01, 0.0);
    let env = camera(dw, dh, df, turn, center);
    // Each texel's ray from the depth camera's center meets the wall z − x = 2 − x* at
    // s times its direction R_dᵀ (K⁻¹ (i + ½, j + ½, 1)), whose depth along the depth
    // camera's axis is s.
    let k = 2.0 - x_star;
    let depth = (0..dw * dh)
        .map(|t| {
            let (i, j) = (f64::from(t % dw) + 0.5, f64::from(t / dw) + 0.5);
            let r = DVec3::new(
                (i - f64::from(dw) / 2.0) / df,
                (j - f64::from(dh) / 2.0) / df,
                1.0,
            );
            let dir = turn.transpose() * r;
            let s = (k - (center.z - center.x)) / (dir.z - dir.x);
            if s > 0.0 {
                stored_depth(s, near, far)
            } else {
                1.0
            }
        })
        .collect();
    let occlusion = Occlusion {
        depth,
        size: UVec2::new(dw, dh),
        format: wgpu::TextureFormat::R32Float,
        camera: env,
        near: near as f32,
        far: far as f32,
        soft: false,
        delta_m: 0.0,
    };
    let rows = [row([x_star as f32, 0.0, 2.0], 0.5, 0.999)];
    (u_star, rows, occlusion)
}

/// R-12: a wall tilted 45° crossing a large primitive. The wall is the plane
/// z = 2 + (x − x*) of the view's camera space, x* = 2 (u* − 32)/60 with u* = 21.3, so it
/// meets the primitive's depth plane z_c = 2 at the view's column u*; the depth camera
/// sits 3 cm right and 1 cm up of the view, turned 5° about its y axis, with an image of
/// 384 × 288 at focal 240 (stored with nearZ 0.1 and farZ 50), and the occlusion is hard
/// with δ = 0. Each fragment's p_f on its own ray at z_c lies behind the wall exactly when
/// its x is below x*, so the pixels whose sample point lies more than 1 px left of u* are
/// hidden and those more than 1 px right of it are drawn as without occlusion; a weight
/// taken at the primitive's center (x = x*) would hide all or none of it.
#[test]
#[ignore = "gpu"]
fn gpu_r12_a_45_degree_wall_gives_the_edge_of_each_fragments_own_ray() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let (u_star, rows, occlusion) = tilted_scene();
    let (reference, occluded) = alphas(&dev, &rows, &occlusion);
    let (mut hidden, mut kept) = (0, 0);
    for i in 0..(W * H) as usize {
        if reference[i] <= 0.5 {
            continue;
        }
        let sample = f64::from(i as u32 % W) + 0.5;
        if sample < u_star - 1.0 {
            hidden += 1;
            assert_eq!(occluded[i], 0.0, "pixel {i} (u + ½ = {sample}) is hidden");
        } else if sample > u_star + 1.0 {
            kept += 1;
            assert_eq!(
                occluded[i], reference[i],
                "pixel {i} (u + ½ = {sample}) is kept"
            );
        }
    }
    assert!(
        hidden > 100 && kept > 100,
        "both sides covered: {hidden} hidden, {kept} kept"
    );
}

/// The scene of `gpu_r12_shared_center_a_wall_at_2_m_hides_3_m_and_keeps_1_m_in_both_eyes`:
/// its rows, the two eyes 64 mm apart and each eye's depth image of the wall at 2 m.
fn shared_scene() -> ([Row; 2], [Camera; 2], [Occlusion; 2]) {
    let (near, far) = (0.25, 20.0);
    let wall = stored_depth(2.0, near, far);
    let eyes =
        [-0.032, 0.032].map(|x| camera(W, H, 60.0, DMat3::IDENTITY, DVec3::new(x, 0.0, 0.0)));
    let occlusion = eyes.map(|eye| Occlusion {
        depth: vec![wall; (W * H) as usize],
        size: UVec2::new(W, H),
        format: wgpu::TextureFormat::R32Float,
        camera: eye,
        near: near as f32,
        far: far as f32,
        soft: true,
        delta_m: 0.05,
    });
    // u = 32 + 60 (x − x_eye) / z: about 11 for the row at 3 m, 55 for the row at 1 m.
    let rows = [
        row([-1.0, -0.2, 3.0], 0.05, 0.9),
        row([0.4, 0.1, 1.0], 0.02, 0.9),
    ];
    (rows, eyes, occlusion)
}

/// R-12's visibility clause in shared-center mode (D-4, #721, Q-400): two eyes 64 mm apart
/// share one prepare, each with a synthetic depth image of a wall at 2 m over its whole
/// image (its own camera, nearZ 0.25, farZ 20, soft with δ = 5 cm). In both eyes the
/// primitive at 3 m (left half of the image) is hidden and the one at 1 m (right half) is
/// drawn exactly as the same shared-center frame without occlusion, in both record layouts.
#[test]
#[ignore = "gpu"]
fn gpu_r12_shared_center_a_wall_at_2_m_hides_3_m_and_keeps_1_m_in_both_eyes() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let (rows, eyes, occlusion) = shared_scene();
    let m = upload(&dev, &model(&rows));
    let draws = [draw(&m, 0.95, RenderMode::Splats, ColorTransform::IDENTITY)];
    for half_records in [false, true] {
        let alphas = |occ: [Option<&Occlusion>; 2]| {
            render_shared_stereo(&dev, &draws, &eyes, &RUN, occ, half_records)
                .0
                .map(|f| f.layer.iter().map(|p| p[3]).collect::<Vec<f32>>())
        };
        let reference = alphas([None, None]);
        let occluded = alphas([Some(&occlusion[0]), Some(&occlusion[1])]);
        for k in 0..2 {
            let (mut behind, mut front) = (0, 0);
            for i in 0..(W * H) as usize {
                let u = i as u32 % W;
                if reference[k][i] > 0.1 {
                    if u < W / 2 {
                        behind += 1;
                    } else {
                        front += 1;
                    }
                }
                let expected = if u < W / 2 { 0.0 } else { reference[k][i] };
                assert_eq!(
                    occluded[k][i],
                    expected,
                    "eye {k}, pixel ({u}, {}), half Σ^S: {half_records}",
                    i as u32 / W
                );
            }
            assert!(
                behind > 0 && front > 0,
                "eye {k}: both primitives drawn without occlusion ({behind}, {front})"
            );
        }
    }
}

/// Q-445: the recorded occluded layers of R-12's scenes (`exact_layers`), sparse: per
/// layer its id, then each pixel that is not (0, 0, 0, 0) as its index and its four f32.
fn recorded_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/r12_exact.bin")
}

/// Q-445: set to record `exact_layers` into `recorded_path()` instead of comparing, at the
/// commit before a change to E13's evaluation.
const RECORD_VAR: &str = "FTGSXR_RECORD_R12_EXACT";

/// The occluded layers of R-12's scenes (Q-445), by id: per view, the wall at 2 m in both
/// formats, each soft-weight case, the 45° wall, and two scenes where the 2 × 2 blend sets
/// the weight over many pixels: `step_scene` hard and soft, and `field_scene` hard and soft
/// with δ = 0 and δ > 0, with finite and infinite far planes; in shared-center mode in both
/// record layouts, each of those scenes with both eyes the view (the first eye's layer)
/// and the shared-center wall scene (both eyes' layers).
fn exact_layers(dev: &SplatDevice) -> Vec<(String, Vec<[f32; 4]>)> {
    let (near, far) = (0.25, 20.0);
    let wall = stored_depth(2.0, near, far);
    let mut scenes: Vec<(String, Vec<Row>, Occlusion)> = Vec::new();
    for (name, format) in [
        ("r32float", wgpu::TextureFormat::R32Float),
        ("depth16", wgpu::TextureFormat::Depth16Unorm),
    ] {
        let (rows, occlusion) = wall_scene(near, far, wall, format);
        scenes.push((format!("wall-{name}"), rows.to_vec(), occlusion));
    }
    for (soft, k) in SOFT_CASES {
        let (_, rows, occlusion) = soft_scene(soft, k);
        let mode = if soft { "soft" } else { "hard" };
        scenes.push((format!("{mode}-{k:+.2}"), rows.to_vec(), occlusion));
    }
    let (_, rows, occlusion) = tilted_scene();
    scenes.push(("tilted".to_owned(), rows.to_vec(), occlusion));
    for (soft, delta_m) in [(false, 0.0), (true, 0.05)] {
        let (rows, occlusion) = step_scene(soft, delta_m);
        scenes.push((format!("step-{delta_m:.2}"), rows.to_vec(), occlusion));
    }
    for (soft, delta_m) in [(false, 0.0), (true, 0.3), (true, 0.0), (false, 0.3)] {
        for (planes, (near, far)) in [("finite", (0.25, 20.0)), ("infinite", (0.2, 0.1))] {
            let (rows, occlusion) = field_scene(soft, delta_m, near, far);
            let mode = if soft { "soft" } else { "hard" };
            scenes.push((
                format!("field-{mode}-{delta_m:.2}-{planes}"),
                rows.to_vec(),
                occlusion,
            ));
        }
    }

    let mut out = Vec::new();
    for (id, rows, occlusion) in &scenes {
        let m = upload(dev, &model(rows));
        let draws = [draw(&m, 0.95, RenderMode::Splats, ColorTransform::IDENTITY)];
        let frame = render_occluded(dev, &draws, &view(), &RUN, Some(occlusion));
        out.push((format!("per-view/{id}"), frame.layer));
        for half_records in [false, true] {
            let [left, _] = render_shared_stereo(
                dev,
                &draws,
                &[view(), view()],
                &RUN,
                [Some(occlusion), Some(occlusion)],
                half_records,
            )
            .0;
            out.push((format!("{}/{id}", layout(half_records)), left.layer));
        }
    }
    let (rows, eyes, occlusion) = shared_scene();
    let m = upload(dev, &model(&rows));
    let draws = [draw(&m, 0.95, RenderMode::Splats, ColorTransform::IDENTITY)];
    for half_records in [false, true] {
        let frames = render_shared_stereo(
            dev,
            &draws,
            &eyes,
            &RUN,
            [Some(&occlusion[0]), Some(&occlusion[1])],
            half_records,
        )
        .0;
        for (k, f) in frames.into_iter().enumerate() {
            out.push((
                format!("{}/shared-wall-eye{k}", layout(half_records)),
                f.layer,
            ));
        }
    }
    out
}

/// Q-445: `occlusion_edges`' step (one depth texel spans 4 × 4 view pixels, texel columns
/// 0–7 at 1 m and 8–15 at 3 m, a large primitive at 2 m), where the 2 × 2 blend's x
/// fraction sets the weight across the edge.
fn step_scene(soft: bool, delta_m: f32) -> ([Row; 1], Occlusion) {
    let (near, far) = (0.25, 20.0);
    let (dw, dh) = (16u32, 12u32);
    let (occluder, surface) = (stored_depth(1.0, near, far), stored_depth(3.0, near, far));
    let occlusion = Occlusion {
        depth: (0..dw * dh)
            .map(|t| if t % dw < 8 { occluder } else { surface })
            .collect(),
        size: UVec2::new(dw, dh),
        format: wgpu::TextureFormat::R32Float,
        camera: camera(dw, dh, 15.0, DMat3::IDENTITY, DVec3::ZERO),
        near: near as f32,
        far: far as f32,
        soft,
        delta_m,
    };
    ([row([0.0, 0.0, 2.0], 0.5, 0.999)], occlusion)
}

/// Q-445: a depth field that varies in both directions, so that the projection q, the
/// texels and both of the 2 × 2 blend's fractions set the weight over most of the image:
/// a 23 × 17 image at focal 21 (about 3 view pixels per texel), the depth camera 2 cm
/// right and 1 cm down of the view and turned 3° about y and 2° about x, each texel a
/// pseudo-random depth in [1.6 m, 2.4 m] (a fixed linear congruential sequence) around a
/// large primitive at 2 m, every 7th texel invalid (d = 1), stored with `near` and `far`
/// (far < near: the infinite far plane, as the Quest's depth images are).
fn field_scene(soft: bool, delta_m: f32, near: f64, far: f64) -> ([Row; 1], Occlusion) {
    let (dw, dh) = (23u32, 17u32);
    let turn =
        DMat3::from_rotation_y(3f64.to_radians()) * DMat3::from_rotation_x(2f64.to_radians());
    let mut state = 0x2545_f491u32;
    let depth = (0..dw * dh)
        .map(|t| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            if t % 7 == 3 {
                1.0
            } else {
                let u = f64::from(state >> 8) / f64::from(1u32 << 24);
                stored_depth(1.6 + 0.8 * u, near, far)
            }
        })
        .collect();
    let occlusion = Occlusion {
        depth,
        size: UVec2::new(dw, dh),
        format: wgpu::TextureFormat::R32Float,
        camera: camera(dw, dh, 21.0, turn, DVec3::new(0.02, 0.01, 0.0)),
        near: near as f32,
        far: far as f32,
        soft,
        delta_m,
    };
    ([row([0.0, 0.0, 2.0], 0.5, 0.999)], occlusion)
}

/// The shared-center record layout's name in an id.
fn layout(half_records: bool) -> &'static str {
    if half_records {
        "shared-half"
    } else {
        "shared-full"
    }
}

/// The sparse encoding of `recorded_path()`.
fn encode(layers: &[(String, Vec<[f32; 4]>)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(layers.len() as u32).to_le_bytes());
    for (id, layer) in layers {
        out.extend_from_slice(&(id.len() as u32).to_le_bytes());
        out.extend_from_slice(id.as_bytes());
        out.extend_from_slice(&(layer.len() as u32).to_le_bytes());
        let set: Vec<(usize, &[f32; 4])> = layer
            .iter()
            .enumerate()
            .filter(|(_, p)| p.iter().any(|v| v.to_bits() != 0))
            .collect();
        out.extend_from_slice(&(set.len() as u32).to_le_bytes());
        for (i, p) in set {
            out.extend_from_slice(&(i as u32).to_le_bytes());
            for v in p {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    out
}

fn decode(bytes: &[u8]) -> Vec<(String, Vec<[f32; 4]>)> {
    let mut at = 0;
    let take = |at: &mut usize, n: usize| {
        let slice = &bytes[*at..*at + n];
        *at += n;
        slice
    };
    let word = |at: &mut usize| u32::from_le_bytes(take(at, 4).try_into().expect("four bytes"));
    let n = word(&mut at);
    let mut out = Vec::new();
    for _ in 0..n {
        let len = word(&mut at) as usize;
        let id = String::from_utf8(take(&mut at, len).to_vec()).expect("a UTF-8 id");
        let pixels = word(&mut at) as usize;
        let mut layer = vec![[0.0f32; 4]; pixels];
        for _ in 0..word(&mut at) {
            let i = word(&mut at) as usize;
            for v in &mut layer[i] {
                *v = f32::from_bits(word(&mut at));
            }
        }
        out.push((id, layer));
    }
    assert_eq!(at, bytes.len(), "the recorded file ends after its layers");
    out
}

/// A W × H layer as E15's image: its (C, A) as the RGB and the alpha.
fn image(layer: &[[f32; 4]]) -> RenderImage {
    RenderImage {
        width: W,
        height: H,
        rgb: layer.iter().flat_map(|p| [p[0], p[1], p[2]]).collect(),
        alpha: layer.iter().map(|p| p[3]).collect(),
    }
}

/// Q-445 (#790): an exact change to E13's evaluation changes no pixel. The occluded layers
/// of R-12's scenes (`exact_layers`: per view, and shared-center in both record layouts,
/// with the soft edge across z_e ± δ among them) against those recorded at the commit
/// before the change: s = 0 and s_α = 0 by E15's rule (D-46). With `FTGSXR_RECORD_R12_EXACT`
/// set it records them instead.
#[test]
#[ignore = "gpu"]
fn gpu_r12_exact_e13_changes_no_pixel_of_the_recorded_layers() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let layers = exact_layers(&dev);
    if std::env::var_os(RECORD_VAR).is_some_and(|v| !v.is_empty()) {
        std::fs::write(recorded_path(), encode(&layers)).expect("the recorded layers written");
        return;
    }
    let bytes = std::fs::read(recorded_path()).expect("the recorded layers (Q-445)");
    let recorded = decode(&bytes);
    let ids =
        |l: &[(String, Vec<[f32; 4]>)]| l.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>();
    assert_eq!(ids(&layers), ids(&recorded), "the same scenes");
    let drawn = layers
        .iter()
        .filter(|(_, l)| l.iter().any(|p| p[3] > 0.0))
        .count();
    assert!(
        drawn * 2 > layers.len(),
        "most occluded layers draw something"
    );
    for ((id, now), (_, then)) in layers.iter().zip(&recorded) {
        let m = metrics(&image(then), &image(now));
        assert!(
            m.share == 0.0 && m.alpha == 0.0,
            "{id}: s = {}, s_α = {} against the recorded layer",
            m.share,
            m.alpha
        );
    }
}
