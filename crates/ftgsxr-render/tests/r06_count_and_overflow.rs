// SPDX-License-Identifier: Apache-2.0

//! R-6, the visible count and overflow (§3.2; §1.4 stages 2, 3 and 8, D-34, P-5): the
//! count equals the CPU count; C below it sets the overflow flag and clamps; the
//! statistics arrive two frames later; the frame loop never waits longer than 1 ms.

mod cull_support;

use std::time::{Duration, Instant};

use cull_support::{draw, look_at, oracle, passes, prepare, sort_pose, test_pose, upload, view};
use ftgsxr_core::{UVec2, Vec3};
use ftgsxr_render::{Colormap, FrameBuffers, FrameStats, GpuModel, QualityTier, RenderMode};
use ftgsxr_testkit::{gpu_context, layer_messages, random_model};

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
    let _layer = layer_messages(); // R-16, Q-105
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
    assert_eq!(got.sort_dispatch, [expected.div_ceil(1024), 1, 1]);
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
    assert_eq!(got.sort_dispatch, [c.div_ceil(1024), 1, 1]);
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
    let _layer = layer_messages(); // R-16, Q-105
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

/// The frame loop's calls under R-6's watchdog: the slowest of each, by name.
#[derive(Default)]
struct Watchdog {
    slowest: Vec<(&'static str, Duration, usize)>,
}

impl Watchdog {
    /// Runs `f` as the frame loop's call `name` of frame `k`, keeping each name's slowest.
    fn time<T>(&mut self, k: usize, name: &'static str, f: impl FnOnce() -> T) -> T {
        let t = Instant::now();
        let out = f();
        let elapsed = t.elapsed();
        match self.slowest.iter_mut().find(|(n, _, _)| *n == name) {
            Some(entry) if elapsed > entry.1 => *entry = (name, elapsed, k),
            Some(_) => {}
            None => self.slowest.push((name, elapsed, k)),
        }
        out
    }
}

/// The lagging run's model size: large enough that the GPU's frame may outlast the loop's.
const LAGGING_ROWS: usize = 200_000;

/// Stage 8 under R-6's watchdog: each frame encodes stages 1–3 and the statistics, is
/// submitted and polled, and each of those calls (`encode_prepare`, `encode_stats`, the
/// submit, `FrameStats::poll`) is timed. In the first run the GPU is let finish after each
/// frame outside the timed calls (a display period's stand-in), so `poll` gives `None` at
/// frames 0 and 1 and frame k − 2's statistics at frame k. In the second the frames follow
/// one another without waiting, so the GPU may lag the loop: two frames is a floor (Q-81),
/// so `poll` may give `None` while frame k − 2 is still running, and whatever it gives at
/// frame k is the block of a frame j ≤ k − 2 that no earlier poll gave, j growing from
/// one block to the next. No call of either run may take longer than 1 ms; the submit's
/// bound holds in a release build, and a debug build, where wgpu's submit spends about
/// 1 ms of CPU without waiting on the GPU, times and reports it only (Q-111).
// Q-111
#[test]
#[ignore = "gpu"]
fn gpu_r06_statistics_arrive_two_frames_later_without_blocking() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let tier = QualityTier::conformance();
    let model = random_model(0x6_08, 20_000, 0);
    let gpu_model = upload(&dev, &model, &tier);
    let passes = passes(&dev, &tier);
    let pose = test_pose();
    let cam = camera();
    let mut watchdog = Watchdog::default();
    // One frame of the loop at τ, timed call by call; the statistics `poll` gives.
    let run_frame = |gpu_model: &GpuModel,
                     frame: &FrameBuffers,
                     watchdog: &mut Watchdog,
                     k: usize,
                     tau: f64| {
        let draws = [draw(gpu_model, pose, tau, RenderMode::Splats)];
        let mut enc = dev
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        watchdog.time(k, "encode_prepare", || {
            passes.encode_prepare(&mut enc, frame, &draws, &sort_pose(&cam), &[view(cam)]);
        });
        watchdog.time(k, "encode_stats", || passes.encode_stats(&mut enc, frame));
        watchdog.time(k, "submit", || dev.queue().submit([enc.finish()]));
        watchdog.time(k, "poll", || FrameStats::poll(frame, &dev))
    };

    // The paced run.
    let frame = FrameBuffers::new(&dev, &tier, model.len() as u32);
    let mut counts = Vec::new();
    for k in 0..8usize {
        // τ moves so that each frame's count differs.
        let tau = 0.2 * k as f64;
        counts.push(oracle(&model, &pose, tau, &cam).len() as u32);
        let stats = run_frame(&gpu_model, &frame, &mut watchdog, k, tau);
        // The display period, outside the timed calls: the frame finishes before the next.
        dev.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the frame completes");
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
    let mut distinct = counts.clone();
    distinct.dedup();
    assert!(
        distinct.len() > 4,
        "the counts differ between frames: {counts:?}"
    );

    // The lagging run: a model fifty times larger, so the GPU falls behind the loop, and
    // the frames back to back; each frame's statistics are checked once the loop is done.
    let big = random_model(0x6_09, LAGGING_ROWS, 0);
    let big_gpu = upload(&dev, &big, &tier);
    let frame = FrameBuffers::new(&dev, &tier, big.len() as u32);
    let frames = 24usize;
    let tau_of = |k: usize| 0.1 * k as f64;
    let polled: Vec<Option<FrameStats>> = (0..frames)
        .map(|k| run_frame(&big_gpu, &frame, &mut watchdog, k, tau_of(k)))
        .collect();
    dev.device()
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the frames complete");
    // Q-81: frame k reads the earliest frame j ≤ k − 2 after the last one read whose count
    // it gives, so each frame's block is read at most once and in order.
    let expected: Vec<u32> = (0..frames)
        .map(|j| oracle(&big, &pose, tau_of(j), &cam).len() as u32)
        .collect();
    let mut distinct = expected.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(
        distinct.len(),
        frames,
        "each lagging frame's count names its frame: {expected:?}"
    );
    let mut next = 0usize;
    let (mut arrived, mut late) = (0usize, 0usize);
    for (k, stats) in polled.iter().enumerate() {
        let Some(stats) = stats else {
            continue;
        };
        assert!(k >= 2, "frame {k}: nothing two frames old yet");
        let j = (next..=k - 2)
            .find(|&j| expected[j] == stats.visible_splats)
            .unwrap_or_else(|| {
                panic!(
                    "frame {k}: {} splats, no unread frame {next} to {} has that count: {:?}",
                    stats.visible_splats,
                    k - 2,
                    &expected[next.min(k - 1)..=k - 2]
                )
            });
        next = j + 1;
        arrived += 1;
        late += usize::from(j < k - 2);
    }
    eprintln!(
        "R-6: {arrived} of {} lagging frames had statistics, {late} of them late",
        frames - 2
    );
    for &(name, slowest, k) in &watchdog.slowest {
        eprintln!("R-6: slowest {name}: {slowest:?} (frame {k})");
    }
    let over: Vec<String> = watchdog
        .slowest
        .iter()
        // Q-111: the submit is bounded in a release build only.
        .filter(|&&(name, slowest, _)| {
            (name != "submit" || !cfg!(debug_assertions)) && slowest > WATCHDOG
        })
        .map(|(name, slowest, k)| format!("{name} took {slowest:?} at frame {k}"))
        .collect();
    assert!(
        over.is_empty() && watchdog.slowest.len() == 4,
        "a frame-loop call waited longer than {WATCHDOG:?}: {over:?}"
    );
    assert!(dev.take_validation_messages().is_empty());
}
