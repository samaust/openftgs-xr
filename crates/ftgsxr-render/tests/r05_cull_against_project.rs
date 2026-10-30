// SPDX-License-Identifier: Apache-2.0

//! R-5, the cull pass against `project` (§3.2; §1.4 stage 2, E1–E3, E5, E6, D-10): the
//! visible set, μ′, the conics, the radii and the depths of the per-view records against
//! the float64 oracle `ftgsxr_core::project`, and the 1/255 bound (Q-28, owner comment on
//! PRD #76), read through the frame-buffer read back (Q-69). The random views look at the
//! model from inside it and from outside it, with the tolerances scaled to float32's
//! propagation (Q-82).

mod cull_support;

use cull_support::{
    Rng, draw, look_at, oracle, passes, prepare, row_at, sort_pose, test_pose, upload, view,
};
use ftgsxr_core::{
    Camera, DMat3, DVec3, Mat3, Pose, Projected, RenderConventions, UVec2, Vec3, effective_opacity,
    temporal_opacity,
};
use ftgsxr_render::{FrameBuffers, QualityTier, RecordView, RenderMode};
use ftgsxr_testkit::{gpu_context, layer_messages, random_model};

/// R-5's tolerances. μ′ takes `MEAN_PX` plus float32's propagation
/// `MEAN_ROUNDING`·f·2⁻²³(‖R_w μ^S‖ + ‖t_w‖)/z_c, the rounding of x_c = R_w μ^S + t_w's
/// operands, which nearly cancel inside the model, magnified by f/z_c; and a radius whose
/// pre-ceiling value lies within `CEIL_REL` relative of an integer may differ by 1 (Q-82).
// Q-82
const MEAN_PX: f64 = 1e-3;
// Q-82
const MEAN_ROUNDING: f64 = 8.0;
// Q-112
const CONIC_REL: f32 = 1e-4;
const DEPTH: f32 = 1e-5;
// Q-82
const CEIL_REL: f64 = 1e-5;

/// The oracle's quantities that `Projected` does not carry, from the same float64 path as
/// `project` (E6): the camera-space depth z_c, the magnitude ‖R_w μ^S‖ + ‖t_w‖ of x_c's
/// operands, and the radii before their ceiling, extent·√Σ′₀₀ and extent·√Σ′₁₁ (Q-82).
// Q-82
fn pre_ceiling(
    mean: Vec3,
    cov: Mat3,
    sigma_eff: f32,
    cam: &Camera,
    conv: &RenderConventions,
) -> (f64, f64, [f64; 2]) {
    let k = cam.k;
    let (fx, fy, cx, cy) = (k.x_axis.x, k.y_axis.y, k.z_axis.x, k.z_axis.y);
    let (w, h) = (f64::from(cam.width), f64::from(cam.height));
    let r_w = DMat3::from_mat4(cam.world_to_camera);
    let (rotated, t_w) = (r_w * mean.as_dvec3(), cam.world_to_camera.w_axis.truncate());
    let x_c = rotated + t_w;
    let z = x_c.z;
    let (u, v) = (x_c.x / z, x_c.y / z);
    let m = f64::from(conv.jacobian_clamp_margin);
    let (mx, my) = (m * w / fx, m * h / fy);
    let tx = u.clamp(-(cx / fx + mx), (w - cx) / fx + mx);
    let ty = v.clamp(-(cy / fy + my), (h - cy) / fy + my);
    let j0 = DVec3::new(fx / z, 0.0, -fx * tx / z);
    let j1 = DVec3::new(0.0, fy / z, -fy * ty / z);
    let sigma_c = r_w * cov.as_dmat3() * r_w.transpose();
    let eps = f64::from(conv.eps2d);
    let s00 = j0.dot(sigma_c * j0) + eps;
    let s11 = j1.dot(sigma_c * j1) + eps;
    let extent = (2.0 * (f64::from(sigma_eff) / f64::from(conv.alpha_skip)).ln())
        .sqrt()
        .min(f64::from(conv.screen_extent_max));
    let operands = rotated.length() + t_w.length();
    (z, operands, [extent * s00.sqrt(), extent * s11.sqrt()])
}

/// Q-82's radius test: exact, or within 1 where the pre-ceiling value lies within
/// `CEIL_REL` relative of a positive integer, where float32 rounding can flip the ceiling.
// Q-82
fn radius_ok(got: u32, oracle: u32, pre: f64) -> bool {
    let n = pre.round();
    let near_integer = n > 0.0 && (pre - n).abs() <= CEIL_REL * n;
    got == oracle || (near_integer && got.abs_diff(oracle) <= 1)
}

/// Pairs each oracle entry with the GPU record of the same depth (within `DEPTH`) and the
/// nearest μ′: records carry no row, and the slots' order is the atomics'.
fn pair(
    cpu: &[(usize, Projected)],
    gpu: &[RecordView],
) -> Vec<(usize, Projected, Option<RecordView>)> {
    let mut order: Vec<usize> = (0..gpu.len()).collect();
    order.sort_by(|&a, &b| gpu[a].depth.total_cmp(&gpu[b].depth));
    let depths: Vec<f32> = order.iter().map(|&k| gpu[k].depth).collect();
    let mut used = vec![false; gpu.len()];
    cpu.iter()
        .map(|&(row, p)| {
            let lo = depths.partition_point(|&d| d < p.depth - DEPTH);
            let hi = depths.partition_point(|&d| d <= p.depth + DEPTH);
            let best = (lo..hi)
                .map(|j| order[j])
                .filter(|&k| !used[k])
                .min_by(|&a, &b| {
                    let da = (gpu[a].mean_px - p.mean).length();
                    let db = (gpu[b].mean_px - p.mean).length();
                    da.total_cmp(&db)
                });
            if let Some(k) = best {
                used[k] = true;
            }
            (row, p, best.map(|k| gpu[k]))
        })
        .collect()
}

/// R-5 on 10⁴ random primitives and 20 random views, ten from inside the model and ten from
/// outside it (Q-82): the visible set equal; μ′ within 10⁻³ px plus
/// 8·f·2⁻²³(‖R_w μ^S‖ + ‖t_w‖)/z_c per axis, each conic component 10⁻⁴ relative with a
/// floor of 10⁻⁴ of the largest component (Q-112), depths 10⁻⁵, radii
/// exact but within 1 where the oracle's pre-ceiling value is within 10⁻⁵ relative of an
/// integer.
// Q-82
#[test]
#[ignore = "gpu"]
fn gpu_r05_cull_matches_project_on_random_primitives_and_views() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let tier = QualityTier::conformance();
    let model = random_model(0x5_05, 10_000, 3);
    let gpu_model = upload(&dev, &model, &tier);
    let passes = passes(&dev, &tier);
    let frame = FrameBuffers::new(&dev, &tier, model.len() as u32);
    let pose = test_pose();
    let center = pose.translation;
    let mut rng = Rng(0x05_2026);
    let mut failures = Vec::new();
    let mut compared = 0usize;
    // Each conic component's largest error as a fraction of its bound, with the oracle's
    // and the record's values there (Q-112).
    let mut worst_conic = [(0f32, 0f32, 0f32); 3];
    for v in 0..20 {
        // Even views from inside the model, 0.3–1.2 m from its center, odd views from
        // outside it, 3–6 m: sizes, focal lengths and principal points vary, and τ
        // includes times outside [τ₀, τ₁] (Q-82).
        // Q-82
        let inside = v % 2 == 0;
        let side = if inside { "inside" } else { "outside" };
        let dist = if inside {
            rng.range(0.3, 1.2)
        } else {
            rng.range(3.0, 6.0)
        } as f32;
        let eye = center + rng.unit() * dist;
        let target = center + rng.unit() * rng.range(0.0, 0.5) as f32;
        let size = UVec2::new(rng.range(64.0, 400.0) as u32, rng.range(48.0, 300.0) as u32);
        let f = rng.range(0.6, 1.6) as f32 * size.x as f32;
        let principal = [
            (size.x as f64 * rng.range(0.4, 0.6)) as f32,
            (size.y as f64 * rng.range(0.4, 0.6)) as f32,
        ];
        let cam = look_at(eye, target, f, size, principal);
        let tau = rng.range(-0.2, 2.1);

        let cpu = oracle(&model, &pose, tau, &cam);
        let draws = [draw(&gpu_model, pose, tau, RenderMode::Splats)];
        let got = prepare(
            &dev,
            &passes,
            &frame,
            &draws,
            &sort_pose(&cam),
            &[view(cam)],
        );
        assert!(!got.overflow, "view {v}: C = N cannot overflow");
        if got.visible_splats as usize != cpu.len() {
            failures.push(format!(
                "view {v} ({side}, τ = {tau}): {} records, the oracle keeps {}",
                got.visible_splats,
                cpu.len()
            ));
        }
        for (row, p, rec) in pair(&cpu, &got.records) {
            compared += 1;
            let Some(r) = rec else {
                failures.push(format!(
                    "view {v} ({side}): row {row} kept by the oracle has no record: {p:?}"
                ));
                continue;
            };
            // Q-82: the tolerances scaled to float32's propagation.
            let (sigma_eff, mu_s, cov) = row_at(&model, row, &pose, tau);
            let (z_c, operands, pre) = pre_ceiling(mu_s, cov, sigma_eff, &cam, &model.conventions);
            // The mirror must ceil to the oracle's radii, else it has drifted from `project`.
            assert_eq!(
                pre.map(|r| r.ceil() as u32),
                p.radii,
                "view {v}: row {row}: the test's pre-ceiling radii drifted from project"
            );
            let spread = MEAN_ROUNDING * 2f64.powi(-23) * operands / z_c;
            let focal = [cam.k.x_axis.x, cam.k.y_axis.y];
            let mean_ok = (0..2).all(|k| {
                (f64::from(r.mean_px[k]) - f64::from(p.mean[k])).abs()
                    <= MEAN_PX + focal[k] * spread
            });
            // Q-112: each component within 10⁻⁴ of its own value, with a floor of 10⁻⁴ of
            // the conic's largest component (a near-zero off-diagonal loses its relative
            // digits to float32 cancellation).
            let largest = p.conic.iter().fold(0f32, |m, c| m.max(c.abs()));
            let mut conic_ok = true;
            for (k, worst) in worst_conic.iter_mut().enumerate() {
                let bound = CONIC_REL * p.conic[k].abs().max(largest);
                let fraction = (r.conic[k] - p.conic[k]).abs() / bound;
                conic_ok &= fraction <= 1.0;
                if fraction > worst.0 {
                    *worst = (fraction, p.conic[k], r.conic[k]);
                }
            }
            let radii = [r.half_extent.x as u32, r.half_extent.y as u32];
            let radii_ok = (0..2).all(|k| radius_ok(radii[k], p.radii[k], pre[k]));
            if !mean_ok || !conic_ok || !radii_ok || (r.depth - p.depth).abs() > DEPTH {
                failures.push(format!(
                    "view {v} ({side}): row {row}: oracle {p:?}, pre-ceiling {pre:?}, \
                     record {r:?}"
                ));
            }
        }
    }
    for (k, (fraction, want, got)) in worst_conic.iter().enumerate() {
        eprintln!(
            "R-5 conic[{k}]: worst error {fraction:.3} of its bound (oracle {want:e}, record {got:e})"
        );
    }
    assert!(
        compared > 10_000,
        "the views keep enough rows to compare ({compared})"
    );
    assert!(
        failures.is_empty(),
        "{} disagreement(s) with project:\n{}",
        failures.len(),
        failures
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert!(dev.take_validation_messages().is_empty());
}

/// R-5 at the 1/255 bound (Q-28, owner comment on PRD #76): in a model with more than one
/// timestamp, a row at opacity exactly α_skip whose window would not otherwise be reached
/// (every other row persistent, so w_max = 0) passes E3's exact test σ_eff ≥ α_skip at
/// its center and within the float32 band |τ − c| < 2⁻¹²·duration, where σt rounds to 1,
/// and fails it outside; a row just below α_skip fails it at every τ, its center included.
/// Both agree with `effective_opacity`'s rounding to f32. Neither has a record at any τ:
/// at σ_eff = α_skip E6's extent is 0 and both radii 0 (Q-21), and below it E3 culls. The
/// test pass is observed through `FrameContents::active_rows` (Q-79).
#[test]
#[ignore = "gpu"]
fn gpu_r05_bound_rows_agree_with_e3_and_effective_opacity() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let tier = QualityTier::conformance();
    let mut model = random_model(0x5_28, 3, 0);
    let alpha_skip = model.conventions.alpha_skip;
    let (c, d) = (0.95, 0.1);
    // Row 0 persistent and always active; row 1 at the bound; row 2 one float32 below it.
    model.opacity = vec![0.9, alpha_skip, f32::from_bits(alpha_skip.to_bits() - 1)];
    model.time_center = vec![c, c, c];
    model.duration = vec![100.0, d, d];
    model.position = vec![[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]];
    model.velocity = vec![[0.0; 3]; 3];
    model.scale = vec![[0.05; 3]; 3];
    model.validate().expect("a valid model");

    let gpu_model = upload(&dev, &model, &tier);
    let passes = passes(&dev, &tier);
    let frame = FrameBuffers::new(&dev, &tier, 3);
    let cam = look_at(
        Vec3::new(0.0, 0.0, -3.0),
        Vec3::ZERO,
        200.0,
        UVec2::new(128, 96),
        [64.0, 48.0],
    );
    let band = 2f64.powi(-12) * d;
    for (offset, in_band) in [
        (0.0, true),
        (band / 2.0, true),
        (-band / 2.0, true),
        (band * 8.0, false),
        (-band * 8.0, false),
        (d, false),
    ] {
        let tau = c + offset;
        // The CPU side: `effective_opacity`'s rounding to f32 keeps row 1 exactly in the
        // band, and never row 2.
        let kept = |i: usize| {
            effective_opacity(
                model.opacity[i],
                temporal_opacity(tau, c, model.duration[i], false),
            ) >= alpha_skip
        };
        assert_eq!(kept(1), in_band, "effective_opacity at τ = c + {offset}");
        assert!(
            !kept(2),
            "effective_opacity below the bound at τ = c + {offset}"
        );

        let draws = [draw(&gpu_model, Pose::IDENTITY, tau, RenderMode::Splats)];
        let got = prepare(
            &dev,
            &passes,
            &frame,
            &draws,
            &sort_pose(&cam),
            &[view(cam)],
        );
        let active = 1 + u32::from(in_band);
        assert_eq!(
            got.active_rows, active,
            "rows passing E3's test at τ = c + {offset}"
        );
        // E3 and E6: only row 0 is drawn, as the oracle says.
        let cpu = oracle(&model, &Pose::IDENTITY, tau, &cam);
        assert_eq!(cpu.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![0]);
        assert_eq!(
            got.visible_splats, 1,
            "only row 0 has a record at τ = c + {offset}"
        );
        let (_, mu, _) = row_at(&model, 0, &Pose::IDENTITY, tau);
        assert_eq!(mu, Vec3::ZERO);
    }
    assert!(dev.take_validation_messages().is_empty());
}
