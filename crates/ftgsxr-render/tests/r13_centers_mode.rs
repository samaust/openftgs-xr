// SPDX-License-Identifier: Apache-2.0

//! R-13, centers mode (§3.2; §1.4 stage 6, Q-78): discs drawn at the moved positions' μ′
//! (E1, E5, E6) with radius ρ, colored by the DC colormap C₀c₀₀ + 0.5: every pixel whose
//! sample point lies within ρ − 1 px of a center shows that center's color within 1/255,
//! opaque, and every covered pixel lies within ρ + 1 px of the center whose color it shows.
//! The depth-tested form writes reverse-Z depth near / z_c (Q-87); in `Both` mode the
//! centers draw over the splats (Q-96). The time-center colormap maps \[τ₀, τ₁\] onto
//! viridis \[0, 1\], and a single-timestamp model to viridis(0) (Q-92, Q-97). In
//! shared-center mode (D-4, #721) the centers project μ^S for each eye (Q-400), with the
//! same criteria in both eyes.

use crate::display_support;

use display_support::{
    Frame, Row, Run, draw, model, probe_camera, render, render_shared_stereo, upload,
};
use ftgsxr_core::{Camera, Vec2};
use ftgsxr_render::{ColorTransform, Colormap, RenderMode};
use ftgsxr_testkit::{gpu_context, layer_messages};

const RHO: f32 = 3.0;
const TAU: f64 = 1.2;

fn rows() -> Vec<Row> {
    let colors = [
        [0.9, 0.2, 0.1],
        [0.1, 0.8, 0.3],
        [0.2, 0.3, 0.9],
        [0.7, 0.7, 0.1],
        [0.6, 0.1, 0.7],
    ];
    (0..5)
        .map(|k| Row {
            position: [
                -1.6 + 0.8 * k as f32,
                0.3 * k as f32 - 0.6,
                if k % 2 == 0 { 0.5 } else { -0.5 },
            ],
            velocity: [0.4, -0.2, 0.3],
            time_center: 0.95,
            duration: 100.0,
            scale: 0.01,
            opacity: 0.9,
            color: colors[k],
        })
        .collect()
}

/// E1 then E6 by hand for the probe camera (z_c = y + 4, x_c = x, y_c = −z): μ′ and z_c.
fn projected(r: &Row) -> (Vec2, f32) {
    projected_from(r, 0.0)
}

/// `projected` for the probe camera moved by `dx` along x (x_c = x − dx).
fn projected_from(r: &Row, dx: f32) -> (Vec2, f32) {
    let dt = (TAU - r.time_center) as f32;
    let p = [0, 1, 2].map(|i| r.position[i] + r.velocity[i] * dt);
    let z = p[1] + 4.0;
    (
        Vec2::new(60.0 * (p[0] - dx) / z + 32.0, 60.0 * -p[2] / z + 24.0),
        z,
    )
}

fn check_discs(frame: &Frame, rows: &[Row]) {
    let colors: Vec<[f32; 3]> = rows.iter().map(|r| r.color).collect();
    check_disc_colors(frame, rows, &colors);
}

/// Every pixel within ρ − 1 px of row k's center shows `colors[k]` within 1/255, opaque,
/// and every covered pixel lies within ρ + 1 px of a center.
fn check_disc_colors(frame: &Frame, rows: &[Row], colors: &[[f32; 3]]) {
    let centers: Vec<(Vec2, [f32; 3])> = rows
        .iter()
        .zip(colors)
        .map(|(r, &c)| (projected(r).0, c))
        .collect();
    check_centers(frame, &centers);
}

/// Every pixel within ρ − 1 px of a center shows its color within 1/255, opaque, and every
/// covered pixel lies within ρ + 1 px of a center.
fn check_centers(frame: &Frame, centers: &[(Vec2, [f32; 3])]) {
    for (a, b) in centers.iter().zip(centers.iter().skip(1)) {
        assert!(a.0.distance(b.0) > 2.0 * RHO + 2.0, "the discs are apart");
    }
    for (i, &texel) in frame.layer.iter().enumerate() {
        let p = Vec2::new((i % 64) as f32 + 0.5, (i / 64) as f32 + 0.5);
        let (nearest, color) = centers
            .iter()
            .min_by(|a, b| a.0.distance(p).total_cmp(&b.0.distance(p)))
            .copied()
            .unwrap();
        let d = nearest.distance(p);
        if d <= RHO - 1.0 {
            for k in 0..3 {
                assert!(
                    (texel[k] - color[k]).abs() <= 1.0 / 255.0,
                    "pixel {i} channel {k}: {} against the color {}",
                    texel[k],
                    color[k]
                );
            }
            assert_eq!(texel[3], 1.0, "pixel {i}: the disc is opaque");
        }
        if texel[3] > 0.0 && d > RHO + 1.0 {
            panic!("pixel {i} is covered {d} px from the nearest center");
        }
    }
}

#[test]
#[ignore = "gpu"]
fn gpu_r13_centers_are_discs_of_radius_rho_in_the_dc_color() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let rows = rows();
    let model = model(&rows);
    let gpu_model = upload(&dev, &model);
    let mode = RenderMode::Centers {
        pixel_radius: RHO,
        colormap: Colormap::Dc,
    };
    let draws = [draw(&gpu_model, TAU, mode, ColorTransform::IDENTITY)];
    let cam = probe_camera();

    let frame = render(
        &dev,
        &draws,
        &cam,
        &Run {
            resolve: None,
            depth: false,
        },
    );
    check_discs(&frame, &rows);
    let covered = frame.layer.iter().filter(|t| t[3] > 0.0).count();
    assert!(covered >= 5 * 9, "five discs drawn ({covered} pixels)");

    // Depth-tested: the discs still drawn, the depth near / z_c at their centers.
    let frame = render(
        &dev,
        &draws,
        &cam,
        &Run {
            resolve: None,
            depth: true,
        },
    );
    check_discs(&frame, &rows);
    let depth = frame.depth.as_ref().expect("depth-tested");
    for r in &rows {
        let (mean, z) = projected(r);
        let i = mean.y.floor() as usize * 64 + mean.x.floor() as usize;
        let want = 0.01 / z;
        assert!(
            (depth[i] - want).abs() <= 1e-5 * want,
            "depth {} against near / z_c = {want}",
            depth[i]
        );
    }
    for (i, t) in frame.layer.iter().enumerate() {
        if t[3] == 0.0 {
            assert_eq!(depth[i], 0.0, "pixel {i}: no depth outside the discs");
        }
    }

    // Both: the splats drawn first, the centers over them (Q-96).
    let both = RenderMode::Both {
        pixel_radius: RHO,
        colormap: Colormap::Dc,
    };
    let draws = [draw(&gpu_model, TAU, both, ColorTransform::IDENTITY)];
    let frame = render(
        &dev,
        &draws,
        &cam,
        &Run {
            resolve: None,
            depth: false,
        },
    );
    check_discs(&frame, &rows);
}

/// Viridis at 0, 0.5 and 1: entries 0, 128 and 255 of matplotlib's 256-entry
/// `_viridis_data` (v3.9.2; source-log row 76), sRGB-encoded display values (Q-92).
const VIRIDIS_0: [f32; 3] = [0.267_004, 0.004_874, 0.329_415];
const VIRIDIS_HALF: [f32; 3] = [0.127_568, 0.566_949, 0.550_556];
const VIRIDIS_1: [f32; 3] = [0.993_248, 0.906_157, 0.143_936];

/// Three still rows at the given time centers, apart in the probe camera's image.
fn timed_rows(time_centers: [f64; 3]) -> Vec<Row> {
    time_centers
        .iter()
        .enumerate()
        .map(|(k, &time_center)| Row {
            position: [-1.2 + 1.2 * k as f32, 0.0, 0.4 * k as f32 - 0.4],
            velocity: [0.0; 3],
            time_center,
            duration: 100.0,
            scale: 0.01,
            opacity: 0.9,
            color: [0.5; 3],
        })
        .collect()
}

fn time_center_frame(model: &ftgsxr_core::Model, tau: f64) -> Frame {
    let dev = gpu_context();
    let gpu_model = upload(&dev, model);
    let mode = RenderMode::Centers {
        pixel_radius: RHO,
        colormap: Colormap::TimeCenter,
    };
    let draws = [draw(&gpu_model, tau, mode, ColorTransform::IDENTITY)];
    render(
        &dev,
        &draws,
        &probe_camera(),
        &Run {
            resolve: None,
            depth: false,
        },
    )
}

#[test]
#[ignore = "gpu"]
fn gpu_r13_time_center_colormap_spans_tau0_to_tau1() {
    let _layer = layer_messages(); // R-16, Q-105
    // τ₀ ≠ 0, so the inverse span is 1 / (τ₁ − τ₀), not 1 / τ₁ (Q-97).
    let rows = timed_rows([10.0, 11.0, 12.0]);
    let mut m = model(&rows);
    m.time.tau0 = 10.0;
    m.time.tau1 = 12.0;
    m.validate().expect("a valid model");
    let frame = time_center_frame(&m, 11.0);
    check_disc_colors(&frame, &rows, &[VIRIDIS_0, VIRIDIS_HALF, VIRIDIS_1]);
    let covered = frame.layer.iter().filter(|t| t[3] > 0.0).count();
    assert!(covered >= 3 * 9, "three discs drawn ({covered} pixels)");

    // A single timestamp (D-9), τ₀ = τ₁: span 0, every center at viridis(0).
    let rows = timed_rows([0.95; 3]);
    let mut m = model(&rows);
    m.time.tau0 = 0.95;
    m.time.tau1 = 0.95;
    m.validate().expect("a valid single-timestamp model");
    let frame = time_center_frame(&m, 0.95);
    check_disc_colors(&frame, &rows, &[VIRIDIS_0; 3]);
    assert!(
        frame.layer.iter().flatten().all(|v| v.is_finite()),
        "no NaN from a zero span"
    );
    let covered = frame.layer.iter().filter(|t| t[3] > 0.0).count();
    assert!(covered >= 3 * 9, "three discs drawn ({covered} pixels)");
}

/// The probe camera moved by `dx` along x: an eye of a pair.
fn eye(dx: f64) -> Camera {
    let mut c = probe_camera();
    // t_w = −R_w c with R_w's first row (1, 0, 0).
    c.world_to_camera.w_axis.x -= dx;
    c
}

/// R-13 in shared-center mode (D-4, #721, Q-400): two eyes 16 cm apart, so that every
/// disc's μ′ differs by more than 2 px between them, beyond the 1 px criterion, share one prepare
/// at their midpoint; in each eye the discs lie at the moved positions' μ′ in that eye
/// (within 1 px: the pixels within ρ − 1 px of it show its DC color within 1/255, opaque,
/// and none beyond ρ + 1 px is covered), in both shared-center record layouts and in
/// `Both` mode; depth-tested, each eye's depth at a disc's center is near / z_c of that
/// eye (Q-87).
#[test]
#[ignore = "gpu"]
fn gpu_r13_shared_center_centers_lie_at_each_eyes_mu_prime() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let rows = rows();
    let model = model(&rows);
    let gpu_model = upload(&dev, &model);
    const HALF_BASE: f32 = 0.08;
    let eyes = [eye(-f64::from(HALF_BASE)), eye(f64::from(HALF_BASE))];
    let centers = |dx: f32| -> Vec<(Vec2, [f32; 3])> {
        rows.iter()
            .map(|r| (projected_from(r, dx).0, r.color))
            .collect()
    };
    let expected = [centers(-HALF_BASE), centers(HALF_BASE)];
    // Every disc's μ′ differs by more than 2 px between the eyes, so a disc drawn at the
    // other eye's μ′ (or view 0's) fails the 1 px criterion.
    for (l, r) in expected[0].iter().zip(&expected[1]) {
        assert!(l.0.distance(r.0) > 2.0, "disparity {}", l.0.distance(r.0));
    }
    let modes = [
        RenderMode::Centers {
            pixel_radius: RHO,
            colormap: Colormap::Dc,
        },
        RenderMode::Both {
            pixel_radius: RHO,
            colormap: Colormap::Dc,
        },
    ];
    for (mode, half_records) in modes.into_iter().flat_map(|m| [(m, false), (m, true)]) {
        let draws = [draw(&gpu_model, TAU, mode, ColorTransform::IDENTITY)];
        let (frames, contents) = render_shared_stereo(
            &dev,
            &draws,
            &eyes,
            &Run {
                resolve: None,
                depth: true,
            },
            [None, None],
            half_records,
        );
        assert_eq!(contents.visible_centers, 5);
        for (k, frame) in frames.iter().enumerate() {
            check_centers(frame, &expected[k]);
            let covered = frame.layer.iter().filter(|t| t[3] > 0.0).count();
            assert!(
                covered >= 5 * 9,
                "eye {k}: five discs drawn ({covered} pixels)"
            );
            let depth = frame.depth.as_ref().expect("depth-tested");
            for r in &rows {
                let (mean, z) = projected_from(r, if k == 0 { -HALF_BASE } else { HALF_BASE });
                let i = mean.y.floor() as usize * 64 + mean.x.floor() as usize;
                let want = 0.01 / z;
                assert!(
                    (depth[i] - want).abs() <= 1e-5 * want,
                    "eye {k}: depth {} against near / z_c = {want}",
                    depth[i]
                );
            }
        }
    }
}

/// Q-422: in shared-center mode a center behind one eye (its z_c below the near plane)
/// draws nothing in that eye, while the other eye, which keeps it in the cull, draws its
/// disc. The second eye is the probe camera moved to y = 2, past the rows (z_c = y − 2 < 0):
/// projected without the cull, μ′ = f·x/z_c + c would mirror the discs onto its image.
#[test]
#[ignore = "gpu"]
fn gpu_r13_shared_center_center_behind_one_eye_draws_nothing_there() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let rows = rows();
    let gpu_model = upload(&dev, &model(&rows));
    let mut behind = probe_camera();
    behind.world_to_camera.w_axis.z = -2.0;
    let mode = RenderMode::Centers {
        pixel_radius: RHO,
        colormap: Colormap::Dc,
    };
    let draws = [draw(&gpu_model, TAU, mode, ColorTransform::IDENTITY)];
    // The mirrored μ′ of at least one row lies on the second eye's image.
    assert!(rows.iter().any(|r| {
        let dt = (TAU - r.time_center) as f32;
        let p = [0, 1, 2].map(|i| r.position[i] + r.velocity[i] * dt);
        let z = p[1] - 2.0;
        let m = Vec2::new(60.0 * p[0] / z + 32.0, 60.0 * -p[2] / z + 24.0);
        (0.0..64.0).contains(&m.x) && (0.0..48.0).contains(&m.y)
    }));
    let (frames, contents) = render_shared_stereo(
        &dev,
        &draws,
        &[probe_camera(), behind],
        &Run {
            resolve: None,
            depth: false,
        },
        [None, None],
        false,
    );
    assert_eq!(contents.visible_centers, 5);
    check_discs(&frames[0], &rows);
    assert!(
        frames[1].layer.iter().all(|t| t[3] == 0.0),
        "no disc in the eye the centers are behind"
    );
}
