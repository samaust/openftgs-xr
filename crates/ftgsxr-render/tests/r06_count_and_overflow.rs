// SPDX-License-Identifier: Apache-2.0

//! R-6, the visible count and overflow (§3.2; §1.4 stages 2, 3 and 8, D-34, P-5): the
//! count equals the CPU count; C below it sets the overflow flag and clamps; the
//! statistics arrive two frames later; the frame loop never waits longer than 1 ms.

use crate::cull_support;

use std::time::{Duration, Instant};

use cull_support::{draw, look_at, oracle, passes, prepare, sort_pose, test_pose, upload, view};
use ftgsxr_core::{Camera, DMat3, DMat4, DVec3, Model, Pose, UVec2, Vec3};
use ftgsxr_render::{
    Colormap, FrameBuffers, FrameStats, GpuModel, ModelDraw, QualityTier, RenderMode, SortMode,
    SortPose, SplatDevice, SplatPasses, ViewParams,
};
use ftgsxr_testkit::{gpu_context, layer_messages, random_model};

/// The watchdog of R-6: no call of the frame loop may wait longer than this.
const WATCHDOG: Duration = Duration::from_millis(1);

fn camera() -> Camera {
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
            let r = DMat3::from_mat4(cam.world_to_camera);
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

/// A thread's context switches (Q-111): those it made itself, waiting, and those the
/// scheduler made, preempting it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Switches {
    voluntary: u64,
    involuntary: u64,
}

impl Switches {
    /// The switches made since `before`.
    fn since(self, before: Switches) -> Switches {
        Switches {
            voluntary: self.voluntary - before.voluntary,
            involuntary: self.involuntary - before.involuntary,
        }
    }
}

/// What R-6's watchdog makes of one timed call (Q-111).
#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    /// Within the 1 ms bound.
    Within,
    /// Over it, but the thread was preempted rather than waiting: reported, not failed.
    Preempted,
    /// Over it: a wait, or counts that couldn't be read.
    Over,
}

/// The verdict on a call that took `elapsed`, given the thread's context switches
/// during it, `None` where they couldn't be read.
// Q-111
fn verdict(elapsed: Duration, switches: Option<Switches>) -> Verdict {
    match switches {
        _ if elapsed <= WATCHDOG => Verdict::Within,
        Some(Switches {
            voluntary: 0,
            involuntary,
        }) if involuntary > 0 => Verdict::Preempted,
        _ => Verdict::Over,
    }
}

/// Q-111: a call within 1 ms passes; a call over it fails unless the thread made no
/// voluntary context switch and at least one involuntary one (it was preempted, not
/// waiting); counts that couldn't be read excuse nothing.
// Q-111
#[test]
fn verdict_excuses_only_a_preempted_call() {
    let switches = |voluntary, involuntary| Switches {
        voluntary,
        involuntary,
    };
    let over = WATCHDOG + Duration::from_micros(2_500);
    let within = WATCHDOG - Duration::from_micros(1);
    assert_eq!(verdict(within, Some(switches(3, 2))), Verdict::Within);
    assert_eq!(verdict(WATCHDOG, None), Verdict::Within);
    assert_eq!(verdict(over, Some(switches(0, 1))), Verdict::Preempted);
    assert_eq!(verdict(over, Some(switches(1, 4))), Verdict::Over);
    assert_eq!(verdict(over, Some(switches(2, 0))), Verdict::Over);
    assert_eq!(verdict(over, Some(switches(0, 0))), Verdict::Over);
    assert_eq!(verdict(over, None), Verdict::Over);
}

/// The calling thread's context switches so far, from `/proc/thread-self/status`; `None`
/// where they can't be read (Q-111).
fn context_switches() -> Option<Switches> {
    let status = std::fs::read_to_string("/proc/thread-self/status").ok()?;
    let field = |key: &str| {
        status
            .lines()
            .find_map(|line| line.strip_prefix(key))
            .and_then(|rest| rest.trim().parse::<u64>().ok())
    };
    Some(Switches {
        voluntary: field("voluntary_ctxt_switches:")?,
        involuntary: field("nonvoluntary_ctxt_switches:")?,
    })
}

/// One frame-loop call's record under R-6's watchdog.
struct CallRecord {
    name: &'static str,
    /// The slowest time and its frame.
    slowest: (Duration, usize),
    /// How many times the call went over 1 ms preempted (Q-111): reported, not failed.
    preempted: usize,
    /// Each time it went over 1 ms otherwise: the time, the frame and the context
    /// switches during the call, where they could be read.
    over: Vec<(Duration, usize, Option<Switches>)>,
}

/// The frame loop's calls under R-6's watchdog, by name.
#[derive(Default)]
struct Watchdog {
    calls: Vec<CallRecord>,
}

impl Watchdog {
    /// Runs `f` as the frame loop's call `name` of frame `k` and records its verdict. The
    /// context switches are read outside the timed span.
    // Q-111
    fn time<T>(&mut self, k: usize, name: &'static str, f: impl FnOnce() -> T) -> T {
        let before = context_switches();
        let t = Instant::now();
        let out = f();
        let elapsed = t.elapsed();
        let after = context_switches();
        let switches = before.zip(after).map(|(before, after)| after.since(before));
        let index = match self.calls.iter().position(|c| c.name == name) {
            Some(i) => i,
            None => {
                self.calls.push(CallRecord {
                    name,
                    slowest: (Duration::ZERO, k),
                    preempted: 0,
                    over: Vec::new(),
                });
                self.calls.len() - 1
            }
        };
        let call = &mut self.calls[index];
        if elapsed > call.slowest.0 {
            call.slowest = (elapsed, k);
        }
        match verdict(elapsed, switches) {
            Verdict::Within => {}
            Verdict::Preempted => call.preempted += 1,
            Verdict::Over => call.over.push((elapsed, k, switches)),
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
/// 1 ms of CPU without waiting on the GPU, times and reports it only. A call over 1 ms
/// that was preempted rather than waiting (no voluntary context switch, at least one
/// involuntary one) is reported, not failed, and the bounds are checked only when
/// `FTGSXR_WATCHDOG=1` is set, by a run holding the machine alone (Q-111).
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
    // Q-111: the bounds apply in a run holding the machine alone, which sets this.
    let bounds_checked = std::env::var("FTGSXR_WATCHDOG").as_deref() == Ok("1");
    for call in &watchdog.calls {
        let (slowest, k) = call.slowest;
        eprintln!(
            "R-6: slowest {}: {slowest:?} (frame {k}); over 1 ms {} times, {} of them preempted",
            call.name,
            call.over.len() + call.preempted,
            call.preempted
        );
    }
    if !bounds_checked {
        eprintln!("R-6: FTGSXR_WATCHDOG is not 1, so the 1 ms bound was reported, not checked");
    }
    let over: Vec<String> = watchdog
        .calls
        .iter()
        // Q-111: the submit is bounded in a release build only.
        .filter(|call| call.name != "submit" || !cfg!(debug_assertions))
        .flat_map(|call| {
            call.over.iter().map(|(took, k, switches)| {
                format!(
                    "{} took {took:?} at frame {k}, context switches {switches:?}",
                    call.name
                )
            })
        })
        .collect();
    assert_eq!(
        watchdog.calls.len(),
        4,
        "the four frame-loop calls are timed"
    );
    assert!(
        !bounds_checked || over.is_empty(),
        "a frame-loop call waited longer than {WATCHDOG:?}: {over:?}"
    );
    assert!(dev.take_validation_messages().is_empty());
}

/// `model` with twelve of the rows the view keeps at τ made non-finite: four with a NaN
/// position, four with an infinite position, four with an infinite scale. Returns the
/// rows kept and the twelve, in that order of kinds.
fn with_non_finite_rows(
    model: &mut Model,
    pose: &Pose,
    tau: f64,
    cam: &Camera,
) -> (Vec<usize>, Vec<usize>) {
    let kept: Vec<usize> = oracle(model, pose, tau, cam)
        .iter()
        .map(|(i, _)| *i)
        .collect();
    assert!(kept.len() > 100, "the view keeps rows ({})", kept.len());
    let bad: Vec<usize> = kept
        .iter()
        .copied()
        .step_by(kept.len() / 12)
        .take(12)
        .collect();
    for (k, &i) in bad.iter().enumerate() {
        match k % 3 {
            0 => model.position[i][0] = f32::NAN,
            1 => model.position[i][2] = f32::INFINITY,
            _ => model.scale[i][1] = f32::INFINITY,
        }
    }
    (kept, bad)
}

/// Three frames of `draws` with the statistics; the third polls the first's block.
fn stats_of(
    dev: &SplatDevice,
    passes: &SplatPasses,
    frame: &FrameBuffers,
    draws: &[ModelDraw],
    pose: &SortPose,
    views: &[ViewParams],
) -> FrameStats {
    let mut stats = None;
    for _ in 0..3 {
        let mut enc = dev
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        passes.encode_prepare(&mut enc, frame, draws, pose, views);
        passes.encode_stats(&mut enc, frame);
        dev.queue().submit([enc.finish()]);
        dev.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the frame completes");
        stats = FrameStats::poll(frame, dev);
    }
    stats.expect("frame 0's statistics at frame 2")
}

/// E6 and §1.4 stage 8 (revision 13): rows the view would keep, given a non-finite
/// position or scale, are culled, neither counted visible nor drawn, and counted in
/// `FrameStats::non_finite` two frames later.
#[test]
#[ignore = "gpu"]
fn gpu_r06_non_finite_rows_are_culled_and_counted() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let tier = QualityTier::conformance();
    let mut model = random_model(0x6_0a, 5_000, 0);
    let pose = test_pose();
    let cam = camera();
    let tau = 0.7;
    let (kept, bad) = with_non_finite_rows(&mut model, &pose, tau, &cam);
    let gpu_model = upload(&dev, &model, &tier);
    let passes = passes(&dev, &tier);
    let draws = [draw(&gpu_model, pose, tau, RenderMode::Splats)];
    let frame = FrameBuffers::new(&dev, &tier, model.len() as u32);
    let expected = (kept.len() - bad.len()) as u32;

    let got = prepare(
        &dev,
        &passes,
        &frame,
        &draws,
        &sort_pose(&cam),
        &[view(cam)],
    );
    assert_eq!(
        got.visible_splats, expected,
        "the non-finite rows are not drawn"
    );
    assert_eq!(got.records.len() as u32, expected);
    for r in &got.records {
        assert!(
            r.depth.is_finite() && r.mean_px.is_finite(),
            "no record of a non-finite row: {r:?}"
        );
    }

    let stats = stats_of(
        &dev,
        &passes,
        &frame,
        &draws,
        &sort_pose(&cam),
        &[view(cam)],
    );
    assert_eq!(stats.non_finite, bad.len() as u32);
    assert_eq!(stats.visible_splats, expected);
    assert!(dev.take_validation_messages().is_empty());
}

/// Q-118 (E6, §1.4 stage 8): a center counts when its μ^S is non-finite, and an instance
/// drawn as splats and centers counts each such row once, not once per dispatch: the
/// NaN and infinite positions give 8 in centers mode, and in both modes 12 (those 8 and
/// the 4 infinite scales, which only the splats project).
#[test]
#[ignore = "gpu"]
fn gpu_r06_non_finite_counts_centers_once_per_row_and_instance() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let tier = QualityTier::conformance();
    let mut model = random_model(0x6_0b, 5_000, 0);
    let pose = test_pose();
    let cam = camera();
    let tau = 0.7;
    let (_, bad) = with_non_finite_rows(&mut model, &pose, tau, &cam);
    let gpu_model = upload(&dev, &model, &tier);
    let passes = passes(&dev, &tier);
    let frame = FrameBuffers::new(&dev, &tier, 2 * model.len() as u32);
    let (pixel_radius, colormap) = (3.0, Colormap::Dc);
    for (mode, expected) in [
        (
            RenderMode::Centers {
                pixel_radius,
                colormap,
            },
            8,
        ),
        (
            RenderMode::Both {
                pixel_radius,
                colormap,
            },
            bad.len() as u32,
        ),
    ] {
        let draws = [draw(&gpu_model, pose, tau, mode)];
        let stats = stats_of(
            &dev,
            &passes,
            &frame,
            &draws,
            &sort_pose(&cam),
            &[view(cam)],
        );
        assert_eq!(stats.non_finite, expected, "{mode:?}");
    }
    assert!(dev.take_validation_messages().is_empty());
}

/// Q-118 (E6, §1.4 stage 8, D-4): the count is of primitives, once per row and
/// instance: two eyes culling one shared-center set count the twelve rows once, and two
/// instances of the model in one frame count them once each.
#[test]
#[ignore = "gpu"]
fn gpu_r06_non_finite_counts_once_per_row_and_instance_whatever_the_views() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let mut model = random_model(0x6_0c, 5_000, 0);
    let pose = test_pose();
    let cam = camera();
    let tau = 0.7;
    let (_, bad) = with_non_finite_rows(&mut model, &pose, tau, &cam);
    let n = bad.len() as u32;

    // Two eyes 64 mm apart about `cam`, one shared-center set (D-4).
    let center = pose.translation;
    let head = center + Vec3::new(0.5, 0.8, -4.0);
    let eye = |dx: f32| {
        look_at(
            head + Vec3::X * dx,
            center,
            300.0,
            UVec2::new(320, 240),
            [160.0, 120.0],
        )
    };
    let tier = QualityTier {
        sort_mode: SortMode::SharedCenter,
        ..QualityTier::conformance()
    };
    let gpu_model = upload(&dev, &model, &tier);
    let passes_shared = passes(&dev, &tier);
    let frame = FrameBuffers::new(&dev, &tier, model.len() as u32);
    let sort = SortPose {
        origin: head,
        forward: (center - head).normalize(),
    };
    let draws = [draw(&gpu_model, pose, tau, RenderMode::Splats)];
    let stats = stats_of(
        &dev,
        &passes_shared,
        &frame,
        &draws,
        &sort,
        &[view(eye(-0.032)), view(eye(0.032))],
    );
    assert_eq!(stats.non_finite, n, "two eyes, one count per row");

    // Two instances of the model, one view: one count per row and instance.
    let tier = QualityTier::conformance();
    let gpu_model = upload(&dev, &model, &tier);
    let passes = passes(&dev, &tier);
    let frame = FrameBuffers::new(&dev, &tier, 2 * model.len() as u32);
    let draws = [
        draw(&gpu_model, pose, tau, RenderMode::Splats),
        draw(&gpu_model, pose, tau, RenderMode::Splats),
    ];
    let stats = stats_of(
        &dev,
        &passes,
        &frame,
        &draws,
        &sort_pose(&cam),
        &[view(cam)],
    );
    assert_eq!(
        stats.non_finite,
        2 * n,
        "two instances, one count per row each"
    );
    assert!(dev.take_validation_messages().is_empty());
}

/// A 320 × 240 camera of focal 300 at the origin looking along +z, its world-to-camera
/// transform the identity, so x_c = μ^S exactly under the identity pose.
fn identity_camera() -> Camera {
    Camera {
        width: 320,
        height: 240,
        k: DMat3::from_cols(
            DVec3::new(300.0, 0.0, 0.0),
            DVec3::new(0.0, 300.0, 0.0),
            DVec3::new(160.0, 120.0, 1.0),
        ),
        world_to_camera: DMat4::IDENTITY,
    }
}

/// Q-118 (E6, E3, §1.4 stage 8): a row E3 culls (σ_eff < α_skip) is not counted, even
/// with an infinite scale, while the same row above α_skip is.
#[test]
#[ignore = "gpu"]
fn gpu_r06_non_finite_does_not_count_a_row_e3_culls() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let tier = QualityTier::conformance();
    let mut model = random_model(0x6_0d, 8, 0);
    let tau = 0.7;
    // An identity pose and camera, every row at (0, 0, 1) in the middle of the image.
    let cam = identity_camera();
    for i in 0..8 {
        // Every row active at τ (σ_t = 1) and still, so E3 decides on the opacity alone:
        // 0.9 for rows 0–3, 0.001 < α_skip = 1/255 for rows 4–7.
        model.time_center[i] = tau;
        model.duration[i] = 10.0;
        model.velocity[i] = [0.0; 3];
        model.position[i] = [0.0, 0.0, 1.0];
        model.scale[i] = [0.05, f32::INFINITY, 0.05];
        model.opacity[i] = if i < 4 { 0.9 } else { 0.001 };
    }
    let gpu_model = upload(&dev, &model, &tier);
    let passes = passes(&dev, &tier);
    let frame = FrameBuffers::new(&dev, &tier, model.len() as u32);
    let draws = [draw(&gpu_model, Pose::IDENTITY, tau, RenderMode::Splats)];
    let stats = stats_of(
        &dev,
        &passes,
        &frame,
        &draws,
        &sort_pose(&cam),
        &[view(cam)],
    );
    assert_eq!(stats.non_finite, 4, "rows 0–3 only");
    assert_eq!(stats.visible_splats, 0);
    assert!(dev.take_validation_messages().is_empty());
}

/// Q-118 (E6, §1.4 stage 8): a row whose camera-space position is finite but whose μ′
/// lies beyond float32's range, off the image, is counted rather than culled as off the
/// image, whichever way it points; its overflow is not left to the GPU, where float32
/// overflow need not give ∞ (#219).
#[test]
#[ignore = "gpu"]
fn gpu_r06_non_finite_counts_a_mean_beyond_float32() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let tier = QualityTier::conformance();
    let mut model = random_model(0x6_0e, 6, 0);
    let tau = 0.7;
    let cam = identity_camera();
    // x_c = μ^S, z_c within the near and far culls; f · |x_c| / z_c ≥ 3·10³⁹ > 3.4·10³⁸
    // for rows 0–4 (row 4's u = x / z itself overflows); row 5, |μ′| = 3·10³⁷, is finite.
    let positions = [
        [1e37, 0.0, 1.0],
        [-1e37, 0.0, 1.0],
        [0.0, 1e37, 1.0],
        [0.0, -1e37, 1.0],
        [1e37, 0.0, 0.02],
        [1e35, 0.0, 1.0],
    ];
    for (i, p) in positions.iter().enumerate() {
        model.time_center[i] = tau;
        model.duration[i] = 10.0;
        model.velocity[i] = [0.0; 3];
        model.scale[i] = [0.05; 3];
        model.opacity[i] = 0.9;
        model.position[i] = *p;
    }
    let gpu_model = upload(&dev, &model, &tier);
    let passes = passes(&dev, &tier);
    let frame = FrameBuffers::new(&dev, &tier, model.len() as u32);
    let draws = [draw(&gpu_model, Pose::IDENTITY, tau, RenderMode::Splats)];
    let stats = stats_of(
        &dev,
        &passes,
        &frame,
        &draws,
        &sort_pose(&cam),
        &[view(cam)],
    );
    assert_eq!(stats.non_finite, 5, "rows 0–4, not the finite row 5");
    assert_eq!(stats.visible_splats, 0, "all six are off the image");
    assert!(dev.take_validation_messages().is_empty());
}
