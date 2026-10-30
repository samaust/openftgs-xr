// SPDX-License-Identifier: Apache-2.0

//! R-5, the cull pass against `project` (§3.2; §1.4 stage 2, E1–E3, E5, E6, D-10): the
//! visible set, μ′, the conics, the radii and the depths of the per-view records against
//! the float64 oracle `ftgsxr_core::project`, and the 1/255 bound (Q-28, owner comment on
//! PRD #76), read through the frame-buffer read back (Q-69).

mod cull_support;

use cull_support::{
    Rng, draw, look_at, oracle, passes, prepare, row_at, sort_pose, test_pose, upload, view,
};
use ftgsxr_core::{Pose, Projected, UVec2, Vec3, effective_opacity, temporal_opacity};
use ftgsxr_render::{FrameBuffers, QualityTier, RecordView, RenderMode};
use ftgsxr_testkit::{gpu_context, random_model};

/// R-5's tolerances.
const MEAN_PX: f32 = 1e-3;
const CONIC_REL: f32 = 1e-4;
const DEPTH: f32 = 1e-5;

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

#[test]
#[ignore = "gpu"]
fn gpu_r05_cull_matches_project_on_random_primitives_and_views() {
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
    for v in 0..20 {
        // Views from outside the model, 3–6 m from its center: sizes, focal lengths and
        // principal points vary, and τ includes times outside [τ₀, τ₁]. Views from inside
        // the model (0.3–1.2 m) are left out: there float32 meets R-5's tolerances only
        // up to its own rounding (a μ′ off by 2·10⁻³ px at z_c = 0.04, a radius whose
        // pre-ceiling value lies within float32 rounding of an integer), Q-T112.5.
        let dist = rng.range(3.0, 6.0) as f32;
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
                "view {v} (τ = {tau}): {} records, the oracle keeps {}",
                got.visible_splats,
                cpu.len()
            ));
        }
        for (row, p, rec) in pair(&cpu, &got.records) {
            compared += 1;
            let Some(r) = rec else {
                failures.push(format!(
                    "view {v}: row {row} kept by the oracle has no record: {p:?}"
                ));
                continue;
            };
            let scale = p.conic.iter().fold(0f32, |m, c| m.max(c.abs()));
            let conic_ok = (0..3).all(|k| (r.conic[k] - p.conic[k]).abs() <= CONIC_REL * scale);
            let radii = [r.half_extent.x as u32, r.half_extent.y as u32];
            if (r.mean_px - p.mean).abs().max_element() > MEAN_PX
                || !conic_ok
                || radii != p.radii
                || (r.depth - p.depth).abs() > DEPTH
            {
                failures.push(format!("view {v}: row {row}: oracle {p:?}, record {r:?}"));
            }
        }
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
/// test pass is observed through `FrameContents::active_rows` (Q-T112.2).
#[test]
#[ignore = "gpu"]
fn gpu_r05_bound_rows_agree_with_e3_and_effective_opacity() {
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
