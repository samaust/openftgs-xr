// SPDX-License-Identifier: Apache-2.0

//! R-13, centers mode (§3.2; §1.4 stage 6, Q-78): discs drawn at the moved positions' μ′
//! (E1, E5, E6) with radius ρ, colored by the DC colormap C₀c₀₀ + 0.5: every pixel whose
//! sample point lies within ρ − 1 px of a center shows that center's color within 1/255,
//! opaque, and every covered pixel lies within ρ + 1 px of the center whose color it shows.
//! The depth-tested form writes reverse-Z depth near / z_c (Q-87); in `Both` mode the
//! centers draw over the splats (Q-T116.6).

mod display_support;

use display_support::{Frame, Row, Run, draw, model, probe_camera, render, upload};
use ftgsxr_core::Vec2;
use ftgsxr_render::{ColorTransform, Colormap, RenderMode};
use ftgsxr_testkit::gpu_context;

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
    let dt = (TAU - r.time_center) as f32;
    let p = [0, 1, 2].map(|i| r.position[i] + r.velocity[i] * dt);
    let z = p[1] + 4.0;
    (
        Vec2::new(60.0 * p[0] / z + 32.0, 60.0 * -p[2] / z + 24.0),
        z,
    )
}

fn check_discs(frame: &Frame, rows: &[Row]) {
    let centers: Vec<(Vec2, [f32; 3])> = rows.iter().map(|r| (projected(r).0, r.color)).collect();
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
                    "pixel {i} channel {k}: {} against the DC color {}",
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

    // Both: the splats drawn first, the centers over them (Q-T116.6).
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
