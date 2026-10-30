// SPDX-License-Identifier: Apache-2.0

//! R-6, the visible count and overflow (§3.2; §1.4 stages 2, 3 and 8, D-34, P-5): the
//! count equals the CPU count; C below it sets the overflow flag and clamps; the
//! statistics arrive two frames later; the frame loop never waits longer than 1 ms.

mod cull_support;

use std::time::{Duration, Instant};

use cull_support::{draw, look_at, oracle, passes, prepare, sort_pose, test_pose, upload, view};
use ftgsxr_core::{UVec2, Vec3};
use ftgsxr_render::{Colormap, FrameBuffers, FrameStats, QualityTier, RenderMode};
use ftgsxr_testkit::{gpu_context, random_model};

/// The watchdog of R-6: no call of the frame loop may wait longer than this.
const WATCHDOG: Duration = Duration::from_millis(1);

fn camera() -> ftgsxr_core::Camera {
    let center = test_pose().translation;
    look_at(
        center + Vec3::new(0.5, 0.8, -4.0),
        center,
        300.0,
        UVec2::new(320, 240),
        [160.0, 120.0],
    )
}

#[test]
#[ignore = "gpu"]
fn gpu_r06_visible_count_equals_the_cpu_count_and_overflow_clamps() {
    let dev = gpu_context();
    let tier = QualityTier::conformance();
    let model = random_model(0x6_06, 20_000, 1);
    let gpu_model = upload(&dev, &model, &tier);
    let passes = passes(&dev, &tier);
    let pose = test_pose();
    let cam = camera();
    let tau = 0.7;
    let expected = oracle(&model, &pose, tau, &cam).len() as u32;
    assert!(expected > 100, "the view keeps rows ({expected})");
    let draws = [draw(&gpu_model, pose, tau, RenderMode::Splats)];

    // C = N: the count is the CPU count, no overflow; the sort is sized from it.
    let frame = FrameBuffers::new(&dev, &tier, model.len() as u32);
    let got = prepare(
        &dev,
        &passes,
        &frame,
        &draws,
        &sort_pose(&cam),
        &[view(cam)],
    );
    assert_eq!(got.visible_splats, expected);
    assert!(!got.overflow);
    assert_eq!(got.sort_dispatch, [expected.div_ceil(256), 1, 1]);
    assert_eq!(got.records.len() as u32, expected);
    assert_eq!(got.sorted.len() as u32, expected);

    // C exactly the count: still no overflow.
    let frame = FrameBuffers::new(&dev, &tier, expected);
    let got = prepare(
        &dev,
        &passes,
        &frame,
        &draws,
        &sort_pose(&cam),
        &[view(cam)],
    );
    assert_eq!((got.visible_splats, got.overflow), (expected, false));

    // C below the count: the flag is raised and the count clamped to C.
    let c = expected / 3;
    let frame = FrameBuffers::new(&dev, &tier, c);
    let got = prepare(
        &dev,
        &passes,
        &frame,
        &draws,
        &sort_pose(&cam),
        &[view(cam)],
    );
    assert_eq!((got.visible_splats, got.overflow), (c, true));
    assert_eq!(got.records.len() as u32, c);
    assert_eq!(got.sort_dispatch, [c.div_ceil(256), 1, 1]);
    // Every slot below C holds a record the oracle keeps (same depth as one of its rows).
    let depths: Vec<f32> = oracle(&model, &pose, tau, &cam)
        .iter()
        .map(|(_, p)| p.depth)
        .collect();
    for r in &got.records {
        assert!(
            depths.iter().any(|d| (d - r.depth).abs() <= 1e-5),
            "a kept record {r:?} is one of the oracle's"
        );
    }
    assert!(dev.take_validation_messages().is_empty());
}

/// The centers region (Q-78): an instance in `Both` mode adds one center per row that
/// passes E3's test and lies on the image, after the splats; a C between the splats and
/// splats + centers clamps the centers and raises the flag.
#[test]
#[ignore = "gpu"]
fn gpu_r06_centers_follow_the_splats_under_the_same_capacity() {
    let dev = gpu_context();
    let tier = QualityTier::conformance();
    let model = random_model(0x6_07, 5_000, 0);
    let gpu_model = upload(&dev, &model, &tier);
    let passes = passes(&dev, &tier);
    let pose = test_pose();
    let cam = camera();
    let tau = 1.1;
    let rho = 3.0;
    let splats = oracle(&model, &pose, tau, &cam).len() as u32;
    // The oracle of a center: σ_eff ≥ α_skip, z_c within [near, far], and the disc's box
    // μ′ ± ρ not wholly off the image.
    let conv = &model.conventions;
    let centers = (0..model.len())
        .filter(|&i| {
            let (sigma_eff, mu_s, _) = cull_support::row_at(&model, i, &pose, tau);
            let r = ftgsxr_core::DMat3::from_mat4(cam.world_to_camera);
            let xc = r * mu_s.as_dvec3() + cam.world_to_camera.w_axis.truncate();
            let (u, v) = (
                cam.k.x_axis.x * xc.x / xc.z + cam.k.z_axis.x,
                cam.k.y_axis.y * xc.y / xc.z + cam.k.z_axis.y,
            );
            let rho = f64::from(rho);
            sigma_eff >= conv.alpha_skip
                && xc.z >= f64::from(conv.near)
                && xc.z <= f64::from(conv.far)
                && u + rho > 0.0
                && u - rho < f64::from(cam.width)
                && v + rho > 0.0
                && v - rho < f64::from(cam.height)
        })
        .count() as u32;
    assert!(centers > 100 && splats > 100);
    let mode = RenderMode::Both {
        pixel_radius: rho,
        colormap: Colormap::Dc,
    };
    let draws = [draw(&gpu_model, pose, tau, mode)];

    let frame = FrameBuffers::new(&dev, &tier, 2 * model.len() as u32);
    let got = prepare(
        &dev,
        &passes,
        &frame,
        &draws,
        &sort_pose(&cam),
        &[view(cam)],
    );
    assert_eq!(
        (got.visible_splats, got.visible_centers, got.overflow),
        (splats, centers, false)
    );

    let c = splats + centers / 2;
    let frame = FrameBuffers::new(&dev, &tier, c);
    let got = prepare(
        &dev,
        &passes,
        &frame,
        &draws,
        &sort_pose(&cam),
        &[view(cam)],
    );
    assert_eq!(
        (got.visible_splats, got.visible_centers, got.overflow),
        (splats, c - splats, true)
    );
    assert!(dev.take_validation_messages().is_empty());
}

/// Stage 8 under R-6's watchdog: each frame encodes stages 1–3 and the statistics, is
/// submitted, and the GPU is let finish outside the timed calls (a display period's
/// stand-in); `FrameStats::poll` gives `None` at frames 0 and 1 and frame k − 2's
/// statistics at frame k, and neither it nor `encode_stats` waits longer than 1 ms.
#[test]
#[ignore = "gpu"]
fn gpu_r06_statistics_arrive_two_frames_later_without_blocking() {
    let dev = gpu_context();
    let tier = QualityTier::conformance();
    let model = random_model(0x6_08, 20_000, 0);
    let gpu_model = upload(&dev, &model, &tier);
    let passes = passes(&dev, &tier);
    let pose = test_pose();
    let cam = camera();
    let frame = FrameBuffers::new(&dev, &tier, model.len() as u32);
    let mut counts = Vec::new();
    let mut slowest = Duration::ZERO;
    for k in 0..8usize {
        // τ moves so that each frame's count differs.
        let tau = 0.2 * k as f64;
        let draws = [draw(&gpu_model, pose, tau, RenderMode::Splats)];
        let mut enc = dev
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        passes.encode_prepare(&mut enc, &frame, &draws, &sort_pose(&cam), &[view(cam)]);
        let t = Instant::now();
        passes.encode_stats(&mut enc, &frame);
        slowest = slowest.max(t.elapsed());
        dev.queue().submit([enc.finish()]);
        dev.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the frame completes");
        counts.push(oracle(&model, &pose, tau, &cam).len() as u32);

        let t = Instant::now();
        let stats = FrameStats::poll(&frame, &dev);
        slowest = slowest.max(t.elapsed());
        if k < 2 {
            assert_eq!(stats, None, "frame {k}: nothing two frames old yet");
            continue;
        }
        let stats = stats.unwrap_or_else(|| panic!("frame {k}: frame {}'s statistics", k - 2));
        assert_eq!(
            stats.visible_splats,
            counts[k - 2],
            "frame {k} reads frame {}",
            k - 2
        );
        assert_eq!(stats.visible_centers, 0);
        assert_eq!(stats.capacity, model.len() as u32);
        assert!(!stats.overflow);
        let t = stats.gpu_times_ms;
        assert!(t.cull >= 0.0 && t.total >= t.cull, "{t:?}");
        if dev
            .device()
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            assert!(t.cull > 0.0, "the cull pass is timed: {t:?}");
        }
        // Polling again in the same frame gives nothing new.
        assert_eq!(FrameStats::poll(&frame, &dev), None);
    }
    counts.dedup();
    assert!(
        counts.len() > 4,
        "the counts differ between frames: {counts:?}"
    );
    assert!(slowest <= WATCHDOG, "a frame-loop call waited {slowest:?}");
    assert!(dev.take_validation_messages().is_empty());
}
