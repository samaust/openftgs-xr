// SPDX-License-Identifier: Apache-2.0

//! The session benchmarks' protocol (§3.5 B-2, B-7, B-8, B-9; Q-407, Q-432): the
//! orbit transplanted to a head, the setups of each benchmark, and the figures and
//! budgets a run's cameras give. Pure, so the `cpu` lane checks it; `session` drives it.
// Q-407, Q-432

use bevy::math::Vec4;
use ftgsxr_core::{Aabb, Pose, Quat, Vec3};
use ftgsxr_render::SortMode;

use crate::benchmark::json::Json;
use crate::benchmark::{Budget, Measured, Rule, look_at, median};

/// The instance's pose that puts a view at `head` (its stage pose, −z forward) where
/// `camera` (the same kind of pose, in model space) sees the model: H·K⁻¹, scale 1
/// (Q-407's transplant).
// Q-407
pub(crate) fn transplant(head: Pose, camera: Pose) -> Pose {
    let rotation = (head.rotation * camera.rotation.inverse()).normalize();
    Pose {
        translation: head.translation - rotation * camera.translation,
        rotation,
        scale: 1.0,
    }
}

/// The orbit's cameras (Q-102).
pub(crate) const ORBIT: usize = 72;

/// Where a setup's head is (Q-407).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum View {
    /// On Q-102's orbit about the content box: the budgeted view.
    Orbit,
    /// At the content box's center, with the orbit's orientations and times: reported.
    Center,
}

impl View {
    /// The setup's `view` value.
    pub(crate) fn name(self) -> &'static str {
        match self {
            View::Orbit => "orbit",
            View::Center => "center",
        }
    }
}

/// The view's `ORBIT` steps about the content box `bounds` over \[τ₀, τ₁\]: each camera's
/// pose in model space (−z forward, y up) and its τ (Q-102's orbit about B_c, Q-407).
/// On the orbit camera k is at 5k° on a horizontal circle of radius the box's diagonal,
/// a quarter of the radius above the center, looking at it, at τ₀ + k/71 (τ₁ − τ₀); at
/// the center it keeps camera k's orientation and τ.
// Q-102, Q-407, Q-432
pub(crate) fn steps(view: View, bounds: &Aabb, tau0: f64, tau1: f64) -> Vec<(Pose, f64)> {
    let center = (bounds.min + bounds.max) / 2.0;
    let radius = (bounds.max - bounds.min).length().max(1e-3);
    (0..ORBIT)
        .map(|k| {
            let angle = (k as f32 * 5.0).to_radians();
            let eye = center + Vec3::new(radius * angle.cos(), 0.25 * radius, radius * angle.sin());
            let f = k as f64 / (ORBIT - 1) as f64;
            let pose = Pose {
                translation: match view {
                    View::Orbit => eye,
                    View::Center => center,
                },
                rotation: look_at(center - eye),
                scale: 1.0,
            };
            (pose, tau0 * (1.0 - f) + tau1 * f)
        })
        .collect()
}

/// The eyes' midpoint (Q-407, as Q-399's sort pose): the mean of the eyes' positions and
/// the normalized sum of their rotations (sign-aligned to the first), scale 1; `None`
/// without an eye.
// Q-399, Q-432
pub(crate) fn midpoint(eyes: &[Pose]) -> Option<Pose> {
    let first = eyes.first()?;
    let n = eyes.len() as f32;
    let translation = eyes.iter().map(|e| e.translation).sum::<Vec3>() / n;
    let sum = eyes.iter().fold(Vec4::ZERO, |acc, e| {
        let q = Vec4::from(e.rotation);
        acc + if q.dot(Vec4::from(first.rotation)) < 0.0 {
            -q
        } else {
            q
        }
    });
    Some(Pose {
        translation,
        rotation: Quat::from_vec4(sum).normalize(),
        scale: 1.0,
    })
}

/// One setup of a session benchmark (Q-407, Q-432): passthrough on and the governor
/// off in all of them.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Setup {
    /// The quality tier's name (§1.12).
    pub(crate) tier: &'static str,
    /// The sort mode, when not the tier's (B-7).
    pub(crate) sort: Option<SortMode>,
    /// `[xr] refresh_rate`: `None` is the runtime's rate (P-1's 72 Hz).
    pub(crate) refresh_hz: Option<f32>,
    /// `[environment] occlusion`.
    pub(crate) occlusion: bool,
    /// `[environment] lighting_mode`.
    pub(crate) lighting: &'static str,
    /// Where the head is.
    pub(crate) view: View,
}

/// The setups of B-2, B-7 and B-8, in their order (Q-432); `None` for another name.
// Q-432
pub(crate) fn setups(name: &str) -> Option<Vec<Setup>> {
    let base = Setup {
        tier: "quest-balanced",
        sort: None,
        refresh_hz: None,
        occlusion: true,
        lighting: "off",
        view: View::Orbit,
    };
    let list = match name {
        "B-2" => [
            ("quest-balanced", None),
            ("quest-high", None),
            ("quest-low", None),
            // Last: an unset rate requests nothing, so the runtime's would not return.
            ("quest-balanced", Some(90.0)),
        ]
        .into_iter()
        .flat_map(|(tier, refresh_hz)| {
            [View::Orbit, View::Center].map(|view| Setup {
                tier,
                refresh_hz,
                view,
                ..base.clone()
            })
        })
        .collect(),
        "B-7" => [SortMode::SharedCenter, SortMode::PerView]
            .map(|sort| Setup {
                sort: Some(sort),
                ..base.clone()
            })
            .to_vec(),
        "B-8" => [
            (false, "off"),
            (true, "off"),
            (true, "manual"),
            (true, "auto-global"),
            (true, "auto-local"),
        ]
        .map(|(occlusion, lighting)| Setup {
            occlusion,
            lighting,
            ..base.clone()
        })
        .to_vec(),
        _ => return None,
    };
    Some(list)
}

/// `json` (an object) with `key` added last.
pub(crate) fn with(json: &Json, key: &str, value: Json) -> Json {
    let mut out = json.clone();
    if let Json::Obj(members) = &mut out {
        members.push((key.to_owned(), value));
    }
    out
}

/// One camera's held frames: the figures of each distinct statistics block that arrived
/// in them (Q-432).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct CameraRun {
    /// Each block's `gpu_times_ms.total` (f_gpu, Q-403).
    pub(crate) gpu_ms: Vec<f64>,
    /// Each block's `gpu_times_ms.sort`.
    pub(crate) sort_ms: Vec<f64>,
    /// Each block's `visible_splats`.
    pub(crate) visible: Vec<u32>,
    /// Whether any block overflowed.
    pub(crate) overflow: bool,
}

/// What a setup's held frames showed besides the blocks (Q-432, Q-433).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Observed {
    /// `XrDisplay.refresh_hz`, when known.
    pub(crate) refresh_hz: Option<f32>,
    /// `XrDisplay.swapchain`.
    pub(crate) swapchain: [u32; 2],
    /// Each runtime counter's path, unit and valid values over the held frames.
    pub(crate) counters: Vec<(String, &'static str, Vec<f64>)>,
}

/// A setup's figures that its budgets and comparisons read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Summary {
    /// The slowest camera's median f_gpu.
    pub(crate) gpu_ms_max: f64,
    /// The median over the cameras of their median f_gpu.
    pub(crate) gpu_ms_median: f64,
    /// The median over the cameras of their median sort time.
    pub(crate) sort_ms_median: f64,
    /// Whether any camera's block overflowed.
    pub(crate) overflow: bool,
}

/// Records a setup's figures under `setup` (Q-432): per camera k the medians `gpu_ms`
/// and `sort_ms` and the largest `visible_splats`, with `camera` k in the setup; per
/// setup `gpu_ms_max`, `gpu_ms_median`, `sort_ms_median`, `sort_ms_max`, `visible_max`,
/// `overflow_cameras`, `blocks`, the observed `refresh_hz` and swapchain size, and each
/// runtime counter's median as `runtime_counter` with its path in `counter` (Q-433).
// Q-432, Q-433
pub(crate) fn summarize(
    m: &mut Measured,
    setup: &Json,
    cameras: &[CameraRun],
    observed: &Observed,
) -> Summary {
    let mut gpu = Vec::new();
    let mut sort = Vec::new();
    let mut visible_max = 0u32;
    for (k, c) in cameras.iter().enumerate() {
        let at = with(setup, "camera", Json::Num(k as f64));
        let g = median(&mut c.gpu_ms.clone());
        let o = median(&mut c.sort_ms.clone());
        let v = c.visible.iter().copied().max().unwrap_or(0);
        m.figure(&at, "gpu_ms", g, "ms");
        m.figure(&at, "sort_ms", o, "ms");
        m.figure(&at, "visible_splats", f64::from(v), "splats");
        gpu.push(g);
        sort.push(o);
        visible_max = visible_max.max(v);
    }
    let gpu_ms_max = gpu.iter().copied().fold(0.0, f64::max);
    let sort_ms_max = sort.iter().copied().fold(0.0, f64::max);
    let overflowed = cameras.iter().filter(|c| c.overflow).count();
    let blocks: usize = cameras.iter().map(|c| c.gpu_ms.len()).sum();
    let summary = Summary {
        gpu_ms_max,
        gpu_ms_median: median(&mut gpu),
        sort_ms_median: median(&mut sort),
        overflow: overflowed > 0,
    };
    m.figure(setup, "gpu_ms_max", summary.gpu_ms_max, "ms");
    m.figure(setup, "gpu_ms_median", summary.gpu_ms_median, "ms");
    m.figure(setup, "sort_ms_median", summary.sort_ms_median, "ms");
    m.figure(setup, "sort_ms_max", sort_ms_max, "ms");
    m.figure(setup, "visible_max", f64::from(visible_max), "splats");
    m.figure(setup, "overflow_cameras", overflowed as f64, "cameras");
    m.figure(setup, "blocks", blocks as f64, "blocks");
    observe(m, setup, observed);
    summary
}

/// A setup's run: its figures' summary, or why the device did not reach it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Outcome {
    /// Measured.
    Ran(Summary),
    /// Not reached (Q-432): the reason, already recorded with each measure.
    Skipped(String),
}

/// The note when P-1 is not judged because a camera overflowed (Q-407).
pub(crate) const P1_OVERFLOW_NOTE: &str = "P-1 not judged: a camera's visible count exceeded the tier's C, so the figure is reported only (Q-407)";

/// The budgets and the figures derived across setups (Q-432): B-2's P-1 on
/// `quest-balanced`'s orbit (12.4 ms at the runtime's rate, 9.6 ms at 90 Hz, §1.12), none
/// where a camera overflowed; B-7's `sort_ratio` (per view over shared center, at least
/// 1.5, §3.5) and `gpu_ratio`; B-8's `cost_ms` against occlusion off, and each lighting
/// mode's against occlusion on with lighting off.
// Q-407, Q-432
pub(crate) fn conclude(name: &str, m: &mut Measured, runs: &[(Setup, Json, Outcome)]) {
    if name == "B-2" {
        for (setup, json, outcome) in runs {
            let Outcome::Ran(summary) = outcome else {
                continue;
            };
            if setup.tier != "quest-balanced" || setup.view != View::Orbit {
                continue;
            }
            if summary.overflow {
                if !m.notes.contains(&P1_OVERFLOW_NOTE) {
                    m.notes.push(P1_OVERFLOW_NOTE);
                }
                continue;
            }
            // §1.12 P-1: 72 Hz with 1.5 ms margin, and 90 Hz when selected.
            let limit = if setup.refresh_hz == Some(90.0) {
                9.6
            } else {
                12.4
            };
            m.budgets.push(Budget {
                id: "P-1",
                setup: json.clone(),
                measure: "gpu_ms_max",
                rule: Rule::AtMost,
                limit,
                value: summary.gpu_ms_max,
            });
        }
    }
    let ran = |pick: &dyn Fn(&Setup) -> bool| {
        runs.iter().find_map(|(s, _, o)| match o {
            Outcome::Ran(summary) if pick(s) => Some(*summary),
            _ => None,
        })
    };
    if name == "B-7" {
        let shared = ran(&|s| s.sort == Some(SortMode::SharedCenter));
        let per_view = ran(&|s| s.sort == Some(SortMode::PerView));
        if let (Some(shared), Some(per_view)) = (shared, per_view) {
            let setup = Json::obj([
                ("tier", Json::str("quest-balanced")),
                ("view", Json::str(View::Orbit.name())),
                ("ratio", Json::str("per view / shared center")),
            ]);
            let sort_ratio = per_view.sort_ms_median / shared.sort_ms_median;
            m.figure(&setup, "sort_ratio", sort_ratio, "ratio");
            m.figure(
                &setup,
                "gpu_ratio",
                per_view.gpu_ms_median / shared.gpu_ms_median,
                "ratio",
            );
            // §3.5 B-7: shared center at least 1.5× faster in the sort (calibrated).
            m.budgets.push(Budget {
                id: "B-7",
                setup,
                measure: "sort_ratio",
                rule: Rule::AtLeast,
                limit: 1.5,
                value: sort_ratio,
            });
        }
    }
    if name == "B-8" {
        let off = ran(&|s| !s.occlusion);
        let on = ran(&|s| s.occlusion && s.lighting == "off");
        for (setup, json, outcome) in runs {
            let Outcome::Ran(summary) = outcome else {
                continue;
            };
            let reference = match (setup.occlusion, setup.lighting) {
                (false, _) => None,
                (true, "off") => off,
                (true, _) => on,
            };
            if let Some(r) = reference {
                m.figure(
                    json,
                    "cost_ms",
                    summary.gpu_ms_median - r.gpu_ms_median,
                    "ms",
                );
            }
        }
    }
}

/// The blocks of Dv-5's play (Q-438): each distinct statistics block that arrived
/// while the instance played over its span.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Play {
    /// Each block's `gpu_times_ms.total` (f_gpu, Q-403).
    pub(crate) gpu_ms: Vec<f64>,
    /// Each block's `gpu_times_ms.sort`.
    pub(crate) sort_ms: Vec<f64>,
    /// Each block's `visible_splats`.
    pub(crate) visible: Vec<u32>,
    /// The blocks that overflowed.
    pub(crate) overflow_blocks: usize,
    /// The app's frames over the play.
    pub(crate) frames: u32,
    /// The play's wall time, s.
    pub(crate) play_s: f64,
}

/// §1.12 P-1's limit at the runtime's 72 Hz.
pub(crate) const P1_MS: f64 = 12.4;
/// §1.12 P-4's Quest limit.
pub(crate) const P4_MS: f64 = 2.5;
/// §1.12 P-3's limit, s per 100 MB.
pub(crate) const P3_S_PER_100MB: f64 = 1.0;

/// Dv-5's τ after `elapsed` seconds of play at speed 1 from τ₀ (§1.6: τ advances by
/// speed · Δt), held at τ₁ once the span is played (Q-438).
// Q-438
pub(crate) fn play_tau(tau0: f64, tau1: f64, elapsed: f64) -> f64 {
    (tau0 + elapsed).min(tau1)
}

/// The note when Dv-5's P-1 is not judged because a block overflowed (Q-407).
pub(crate) const DV5_OVERFLOW_NOTE: &str = "P-1 not judged: a block's visible count exceeded the tier's C, so the figure is reported only (Q-407)";

/// Records Dv-5's figures and its budgets under `setup` (§3.4 Dv-5, Q-407, Q-438):
/// P-1 on the largest f_gpu of every block (none when a block overflowed), P-4 on the
/// largest sort time of every block, and P-3 on the load's `load_s` per 10⁸ of the
/// file's `file_bytes` (B-5's figure, Q-102); the runtime's counters beside them.
// Q-407, Q-438, Q-439
pub(crate) fn conclude_dv5(
    m: &mut Measured,
    setup: &Json,
    play: &Play,
    observed: &Observed,
    load_s: f64,
    file_bytes: u64,
) {
    let gpu_max = play.gpu_ms.iter().copied().fold(0.0, f64::max);
    let sort_max = play.sort_ms.iter().copied().fold(0.0, f64::max);
    let over = play.gpu_ms.iter().filter(|&&g| g > P1_MS).count();
    let sort_over = play.sort_ms.iter().filter(|&&s| s > P4_MS).count();
    let visible_max = play.visible.iter().copied().max().unwrap_or(0);
    let per_100mb = load_s * 1e8 / file_bytes as f64;
    m.figure(setup, "frame_gpu_ms_max", gpu_max, "ms");
    m.figure(
        setup,
        "frame_gpu_ms_median",
        median(&mut play.gpu_ms.clone()),
        "ms",
    );
    m.figure(setup, "frame_sort_ms_max", sort_max, "ms");
    m.figure(
        setup,
        "frame_sort_ms_median",
        median(&mut play.sort_ms.clone()),
        "ms",
    );
    m.figure(setup, "blocks_over_p1", over as f64, "blocks");
    m.figure(setup, "blocks_over_p4", sort_over as f64, "blocks");
    m.figure(setup, "blocks", play.gpu_ms.len() as f64, "blocks");
    m.figure(setup, "frames", f64::from(play.frames), "frames");
    m.figure(setup, "play_s", play.play_s, "s");
    m.figure(setup, "visible_max", f64::from(visible_max), "splats");
    m.figure(
        setup,
        "overflow_blocks",
        play.overflow_blocks as f64,
        "blocks",
    );
    m.figure(setup, "load_s", load_s, "s");
    m.figure(setup, "file_bytes", file_bytes as f64, "bytes");
    m.figure(setup, "s_per_100mb", per_100mb, "s");
    observe(m, setup, observed);
    if play.overflow_blocks > 0 {
        m.notes.push(DV5_OVERFLOW_NOTE);
    } else {
        m.budgets.push(Budget {
            id: "P-1",
            setup: setup.clone(),
            measure: "frame_gpu_ms_max",
            rule: Rule::AtMost,
            limit: P1_MS,
            value: gpu_max,
        });
    }
    m.budgets.push(Budget {
        id: "P-3",
        setup: setup.clone(),
        measure: "s_per_100mb",
        rule: Rule::AtMost,
        limit: P3_S_PER_100MB,
        value: per_100mb,
    });
    m.budgets.push(Budget {
        id: "P-4",
        setup: setup.clone(),
        measure: "frame_sort_ms_max",
        rule: Rule::AtMost,
        limit: P4_MS,
        value: sort_max,
    });
}

/// The observed refresh rate, swapchain size and each runtime counter's median (Q-433).
fn observe(m: &mut Measured, setup: &Json, observed: &Observed) {
    if let Some(hz) = observed.refresh_hz {
        m.figure(setup, "refresh_hz", f64::from(hz), "Hz");
    }
    m.figure(
        setup,
        "swapchain_width",
        f64::from(observed.swapchain[0]),
        "px",
    );
    m.figure(
        setup,
        "swapchain_height",
        f64::from(observed.swapchain[1]),
        "px",
    );
    for (path, unit, values) in &observed.counters {
        if !values.is_empty() {
            let at = with(setup, "counter", Json::str(path.clone()));
            m.figure(&at, "runtime_counter", median(&mut values.clone()), unit);
        }
    }
}

/// B-9's first frame (Q-222, Q-434): `ends` holds each update's end, in seconds from
/// the start, up to the one whose statistics first counted a splat; the frame drawn is
/// the update two before (the statistics' delay, §1.4 stage 8), at least `uploaded`.
// Q-222, Q-434
pub(crate) fn first_frame(ends: &[f64], uploaded: f64) -> f64 {
    let frame = ends.len().saturating_sub(3);
    ends.get(frame).map_or(uploaded, |&end| end.max(uploaded))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).abs().max_element() <= 1e-5
    }

    /// Q-407: with the instance at `transplant(H, K)`, the eyes' midpoint H, taken into
    /// model space, is camera K, its position and its axes (within 10⁻⁵), here worked by
    /// hand: H at (1, 1.6, −2) turned 90° about y, K at (0, 0, 3) looking along −z, so
    /// the instance turns 90° about y and sits at (1, 1.6, −2) − (3, 0, 0).
    #[test]
    fn transplant_puts_the_eyes_midpoint_at_the_orbit_camera() {
        let turn = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let head = Pose {
            translation: Vec3::new(1.0, 1.6, -2.0),
            rotation: turn,
            scale: 1.0,
        };
        let camera = Pose {
            translation: Vec3::new(0.0, 0.0, 3.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
        };
        let m = transplant(head, camera);
        assert!(close(m.translation, Vec3::new(-2.0, 1.6, -2.0)), "{m:?}");
        assert!(m.rotation.angle_between(turn) <= 1e-5, "{m:?}");
        assert_eq!(m.scale, 1.0);

        // A camera turned and tilted, a head elsewhere: the head in model space is the
        // camera, and each of the head's axes is the camera's.
        let camera = Pose {
            translation: Vec3::new(4.0, 2.5, -1.0),
            rotation: Quat::from_euler(bevy::math::EulerRot::YXZ, 0.7, -0.3, 0.1),
            scale: 1.0,
        };
        let head = Pose {
            translation: Vec3::new(-0.3, 1.2, 0.4),
            rotation: Quat::from_euler(bevy::math::EulerRot::YXZ, -1.1, 0.2, 0.0),
            scale: 1.0,
        };
        let m = transplant(head, camera);
        let to_model = |p: Vec3| m.rotation.inverse() * (p - m.translation) / m.scale;
        assert!(
            close(to_model(head.translation), camera.translation),
            "{m:?}"
        );
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            let head_axis = m.rotation.inverse() * (head.rotation * axis);
            assert!(close(head_axis, camera.rotation * axis), "{axis}");
        }
    }

    /// Q-102's orbit about B_c, worked by hand for the box \[−1, 1\] × \[0, 2\] × \[−1, 1\]:
    /// center (0, 1, 0), radius 2√3, height 1 + √3/2; camera 0 on +x, camera 18 on +z.
    #[test]
    fn orbit_circles_the_content_box_over_the_span() {
        let bounds = Aabb {
            min: Vec3::new(-1.0, 0.0, -1.0),
            max: Vec3::new(1.0, 2.0, 1.0),
        };
        let r = 2.0 * 3f32.sqrt();
        let orbit = steps(View::Orbit, &bounds, 1.0, 8.1);
        assert_eq!(orbit.len(), ORBIT);
        assert!(close(
            orbit[0].0.translation,
            Vec3::new(r, 1.0 + 0.25 * r, 0.0)
        ));
        assert!(close(
            orbit[18].0.translation,
            Vec3::new(0.0, 1.0 + 0.25 * r, r)
        ));
        assert_eq!((orbit[0].1, orbit[ORBIT - 1].1), (1.0, 8.1));
        assert!((orbit[1].1 - (1.0 + 7.1 / 71.0)).abs() < 1e-12);
        let center = Vec3::new(0.0, 1.0, 0.0);
        for (pose, _) in &orbit {
            // Looking at the center (−z), y up (no roll: x stays horizontal).
            let forward = pose.rotation * Vec3::NEG_Z;
            assert!(close(forward, (center - pose.translation).normalize()));
            assert!((pose.rotation * Vec3::X).y.abs() < 1e-6);
            assert_eq!(pose.scale, 1.0);
        }
        let inside = steps(View::Center, &bounds, 1.0, 8.1);
        assert_eq!(inside.len(), ORBIT);
        for (k, (pose, tau)) in inside.iter().enumerate() {
            assert!(close(pose.translation, center));
            assert_eq!(pose.rotation, orbit[k].0.rotation);
            assert_eq!(*tau, orbit[k].1);
        }
    }

    /// Two eyes 64 mm apart, one yawed 2° more: the midpoint between them, yawed 1°.
    #[test]
    fn midpoint_averages_the_eyes() {
        let yaw = |deg: f32| Quat::from_rotation_y(deg.to_radians());
        let eye = |x: f32, deg: f32| Pose {
            translation: Vec3::new(x, 1.6, 0.5),
            rotation: yaw(deg),
            scale: 1.0,
        };
        let m = midpoint(&[eye(-0.032, 10.0), eye(0.032, 12.0)]).expect("a midpoint");
        assert!(close(m.translation, Vec3::new(0.0, 1.6, 0.5)));
        assert!(m.rotation.angle_between(yaw(11.0)) <= 1e-5);
        // The same rotation written with the other sign averages the same.
        let mut flipped = eye(0.032, 12.0);
        flipped.rotation = -flipped.rotation;
        let m2 = midpoint(&[eye(-0.032, 10.0), flipped]).expect("a midpoint");
        assert!(m2.rotation.angle_between(yaw(11.0)) <= 1e-5);
        assert_eq!(midpoint(&[]), None);
    }

    /// Q-432's lists: B-2 each Quest tier at the runtime's rate, then `quest-balanced`
    /// at 90 Hz, each orbit then center, occlusion on, lighting off; B-7 `quest-balanced`'s
    /// orbit shared center then per view; B-8 occlusion off and on, then each lighting
    /// mode with occlusion on.
    #[test]
    fn the_setups_of_b2_b7_b8() {
        let setup = |tier, refresh_hz, view| Setup {
            tier,
            sort: None,
            refresh_hz,
            occlusion: true,
            lighting: "off",
            view,
        };
        let b2 = setups("B-2").expect("B-2");
        let mut expected = Vec::new();
        for (tier, rate) in [
            ("quest-balanced", None),
            ("quest-high", None),
            ("quest-low", None),
            ("quest-balanced", Some(90.0)),
        ] {
            expected.push(setup(tier, rate, View::Orbit));
            expected.push(setup(tier, rate, View::Center));
        }
        assert_eq!(b2, expected);

        let orbit = setup("quest-balanced", None, View::Orbit);
        let b7 = setups("B-7").expect("B-7");
        let with_sort = |sort| Setup {
            sort: Some(sort),
            ..orbit.clone()
        };
        assert_eq!(
            b7,
            [
                with_sort(SortMode::SharedCenter),
                with_sort(SortMode::PerView)
            ]
        );

        let b8 = setups("B-8").expect("B-8");
        let env = |occlusion, lighting| Setup {
            occlusion,
            lighting,
            ..orbit.clone()
        };
        assert_eq!(
            b8,
            [
                env(false, "off"),
                env(true, "off"),
                env(true, "manual"),
                env(true, "auto-global"),
                env(true, "auto-local"),
            ]
        );
        for name in ["B-1", "B-3", "B-9", "B-10"] {
            assert_eq!(setups(name), None, "{name}");
        }
    }

    /// The value of the figure `measure` under `setup`.
    fn figure(m: &Measured, setup: &Json, measure: &str) -> Option<f64> {
        m.figures
            .iter()
            .find(|f| f.measure == measure && f.setup.same(setup))
            .map(|f| f.value)
    }

    /// Q-407: three cameras whose block medians are 11, 25 and 5.5 ms: the budgeted
    /// figure is the slowest camera's, 25; the median over cameras 11.
    #[test]
    fn summary_keeps_the_slowest_cameras_median() {
        let cam = |gpu: &[f64], sort: &[f64], visible: &[u32], overflow| CameraRun {
            gpu_ms: gpu.to_vec(),
            sort_ms: sort.to_vec(),
            visible: visible.to_vec(),
            overflow,
        };
        let cameras = [
            cam(
                &[10.0, 12.0, 11.0],
                &[2.0, 2.0, 3.0],
                &[900, 950, 940],
                false,
            ),
            cam(
                &[20.0, 30.0, 25.0],
                &[2.5, 2.5, 2.5],
                &[990, 1000, 980],
                true,
            ),
            cam(&[5.0, 6.0], &[1.0, 2.0], &[10, 20], false),
        ];
        let observed = Observed {
            refresh_hz: Some(72.0),
            swapchain: [1680, 1760],
            counters: vec![(
                "/perfmetrics_meta/app/gpu_frametime".into(),
                "ms",
                vec![18.0, 19.0, 20.0],
            )],
        };
        let setup = Json::obj([("tier", Json::str("quest-balanced"))]);
        let mut m = Measured::default();
        let s = summarize(&mut m, &setup, &cameras, &observed);
        assert_eq!(
            s,
            Summary {
                gpu_ms_max: 25.0,
                gpu_ms_median: 11.0,
                sort_ms_median: 2.0,
                overflow: true,
            }
        );
        let camera = |k: f64| with(&setup, "camera", Json::Num(k));
        assert_eq!(figure(&m, &camera(0.0), "gpu_ms"), Some(11.0));
        assert_eq!(figure(&m, &camera(2.0), "gpu_ms"), Some(5.5));
        assert_eq!(figure(&m, &camera(1.0), "sort_ms"), Some(2.5));
        assert_eq!(figure(&m, &camera(1.0), "visible_splats"), Some(1000.0));
        assert_eq!(figure(&m, &setup, "gpu_ms_max"), Some(25.0));
        assert_eq!(figure(&m, &setup, "gpu_ms_median"), Some(11.0));
        assert_eq!(figure(&m, &setup, "sort_ms_median"), Some(2.0));
        assert_eq!(figure(&m, &setup, "sort_ms_max"), Some(2.5));
        assert_eq!(figure(&m, &setup, "visible_max"), Some(1000.0));
        assert_eq!(figure(&m, &setup, "overflow_cameras"), Some(1.0));
        assert_eq!(figure(&m, &setup, "blocks"), Some(8.0));
        assert_eq!(figure(&m, &setup, "refresh_hz"), Some(72.0));
        assert_eq!(figure(&m, &setup, "swapchain_width"), Some(1680.0));
        assert_eq!(figure(&m, &setup, "swapchain_height"), Some(1760.0));
        let counter = with(
            &setup,
            "counter",
            Json::str("/perfmetrics_meta/app/gpu_frametime"),
        );
        assert_eq!(figure(&m, &counter, "runtime_counter"), Some(19.0));
    }

    fn ran(gpu_ms_max: f64, gpu_ms_median: f64, sort_ms_median: f64, overflow: bool) -> Outcome {
        Outcome::Ran(Summary {
            gpu_ms_max,
            gpu_ms_median,
            sort_ms_median,
            overflow,
        })
    }

    /// The runs of `name`'s setups with `outcomes`, each setup's JSON its index.
    fn runs(name: &str, outcomes: Vec<Outcome>) -> Vec<(Setup, Json, Outcome)> {
        setups(name)
            .expect("a session benchmark")
            .into_iter()
            .zip(outcomes)
            .enumerate()
            .map(|(i, (s, o))| (s, Json::obj([("i", Json::Num(i as f64))]), o))
            .collect()
    }

    /// §1.12 P-1: `quest-balanced`'s orbit against 12.4 ms at the runtime's rate and 9.6 ms
    /// at 90 Hz, on the slowest camera's figure; nothing for the other tiers or the center
    /// view; not judged when a camera overflowed (Q-407).
    #[test]
    fn p1_budgets_only_quest_balanced_orbit_without_overflow() {
        let outcomes = (0..8)
            .map(|i| ran(10.0 + f64::from(i), 9.0, 2.0, false))
            .collect();
        let mut m = Measured::default();
        conclude("B-2", &mut m, &runs("B-2", outcomes));
        let got: Vec<(&str, f64, f64, Rule, f64)> = m
            .budgets
            .iter()
            .map(|b| {
                (
                    b.id,
                    b.setup.get("i").and_then(Json::as_f64).unwrap_or(-1.0),
                    b.limit,
                    b.rule,
                    b.value,
                )
            })
            .collect();
        assert_eq!(
            got,
            [
                ("P-1", 0.0, 12.4, Rule::AtMost, 10.0),
                ("P-1", 6.0, 9.6, Rule::AtMost, 16.0),
            ]
        );
        assert!(m.budgets.iter().all(|b| b.measure == "gpu_ms_max"));
        assert!(m.notes.is_empty());

        // An overflow at 72 Hz, a setup not reached at 90 Hz: no budget, one note.
        let mut outcomes: Vec<Outcome> = (0..8).map(|_| ran(10.0, 9.0, 2.0, false)).collect();
        outcomes[0] = ran(10.0, 9.0, 2.0, true);
        outcomes[6] = Outcome::Skipped("90 Hz not reached".into());
        let mut m = Measured::default();
        conclude("B-2", &mut m, &runs("B-2", outcomes));
        assert!(m.budgets.is_empty(), "{:?}", m.budgets);
        assert_eq!(m.notes, [P1_OVERFLOW_NOTE]);
    }

    /// §3.5 B-7: per view's sort over shared center's, at least 1.5 (here 6 / 2.5 = 2.4),
    /// with the frame times' ratio (24 / 16 = 1.5) beside it. B-8: each setup's median
    /// frame time against its reference: occlusion on against off (+1.5), each lighting
    /// mode against occlusion on with lighting off; a mode not reached has no cost.
    #[test]
    fn b7_ratio_and_b8_costs() {
        let mut m = Measured::default();
        let b7 = runs(
            "B-7",
            vec![ran(20.0, 16.0, 2.5, false), ran(30.0, 24.0, 6.0, false)],
        );
        conclude("B-7", &mut m, &b7);
        assert_eq!(m.budgets.len(), 1, "{:?}", m.budgets);
        let b = &m.budgets[0];
        assert_eq!(
            (b.id, b.measure, b.rule, b.limit),
            ("B-7", "sort_ratio", Rule::AtLeast, 1.5)
        );
        assert!((b.value - 2.4).abs() < 1e-12);
        let ratio = |measure: &str| {
            m.figures
                .iter()
                .find(|f| f.measure == measure)
                .map(|f| f.value)
        };
        assert!((ratio("sort_ratio").expect("sort_ratio") - 2.4).abs() < 1e-12);
        assert!((ratio("gpu_ratio").expect("gpu_ratio") - 1.5).abs() < 1e-12);
        let setup = &b.setup;
        assert_eq!(
            setup.get("tier").and_then(Json::as_str),
            Some("quest-balanced")
        );
        assert_eq!(setup.get("view").and_then(Json::as_str), Some("orbit"));

        let mut m = Measured::default();
        let b8 = runs(
            "B-8",
            vec![
                ran(9.0, 8.0, 2.0, false),
                ran(11.0, 9.5, 2.0, false),
                ran(11.5, 9.75, 2.0, false),
                Outcome::Skipped("auto lighting not available".into()),
                ran(13.0, 11.0, 2.0, false),
            ],
        );
        conclude("B-8", &mut m, &b8);
        assert!(m.budgets.is_empty());
        let costs: Vec<(f64, f64)> = m
            .figures
            .iter()
            .filter(|f| f.measure == "cost_ms")
            .map(|f| {
                (
                    f.setup.get("i").and_then(Json::as_f64).unwrap_or(-1.0),
                    f.value,
                )
            })
            .collect();
        assert_eq!(costs, [(1.0, 1.5), (2.0, 0.25), (4.0, 1.5)]);
    }

    /// Q-222: the statistics of the fifth update count the first splat, so the frame is
    /// the third's (0.7 s), unless the upload ended later.
    #[test]
    fn b9_first_frame_is_two_updates_before_the_first_counted_splat() {
        let ends = [0.5, 0.6, 0.7, 0.8, 0.9];
        assert_eq!(first_frame(&ends, 0.55), 0.7);
        assert_eq!(first_frame(&ends, 0.75), 0.75);
        assert_eq!(first_frame(&ends[..1], 0.4), 0.5);
    }

    fn play(gpu: &[f64], sort: &[f64], overflow_blocks: usize) -> Play {
        Play {
            gpu_ms: gpu.to_vec(),
            sort_ms: sort.to_vec(),
            visible: vec![990_000; gpu.len()],
            overflow_blocks,
            frames: 150,
            play_s: 10.0,
        }
    }

    /// The budget `id`'s (measure, limit, value, pass).
    fn budget(m: &Measured, id: &str) -> Option<(&'static str, f64, f64, bool)> {
        m.budgets
            .iter()
            .find(|b| b.id == id)
            .map(|b| (b.measure, b.limit, b.value, b.pass()))
    }

    /// §3.4 Dv-5, §1.12 P-1, P-3, P-4 (Q-407, Q-438): every block's f_gpu against
    /// 12.4 ms, so one block of 13 ms among 10 ms ones fails P-1 although the median
    /// meets it; each block's sort against 2.5 ms (worst 2.4 here, met); the budget
    /// model's 180,001,152 bytes loaded in 1.80001152 s is 1 s per 100 MB, met at the
    /// limit, and 0.9 s for 90 MB of 10⁶ bytes is 1 s too.
    #[test]
    fn dv5_judges_every_block_and_the_load_per_100_mb() {
        let setup = Json::obj([("tier", Json::str("quest-balanced"))]);
        let p = play(
            &[10.0, 10.5, 13.0, 10.0, 10.2],
            &[2.0, 2.4, 2.1, 2.2, 2.0],
            0,
        );
        let mut m = Measured::default();
        conclude_dv5(
            &mut m,
            &setup,
            &p,
            &Observed::default(),
            1.800_011_52,
            180_001_152,
        );
        assert_eq!(
            budget(&m, "P-1"),
            Some(("frame_gpu_ms_max", 12.4, 13.0, false))
        );
        assert_eq!(
            budget(&m, "P-4"),
            Some(("frame_sort_ms_max", 2.5, 2.4, true))
        );
        let (measure, limit, value, _) = budget(&m, "P-3").expect("P-3");
        assert_eq!((measure, limit), ("s_per_100mb", 1.0));
        assert!((value - 1.0).abs() < 1e-12, "{value}");
        assert_eq!(m.budgets.len(), 3);
        assert_eq!(figure(&m, &setup, "frame_gpu_ms_median"), Some(10.2));
        assert_eq!(figure(&m, &setup, "frame_gpu_ms_max"), Some(13.0));
        assert_eq!(figure(&m, &setup, "frame_sort_ms_max"), Some(2.4));
        assert_eq!(figure(&m, &setup, "blocks_over_p1"), Some(1.0));
        assert_eq!(figure(&m, &setup, "blocks"), Some(5.0));
        assert_eq!(figure(&m, &setup, "frames"), Some(150.0));
        assert_eq!(figure(&m, &setup, "play_s"), Some(10.0));
        assert_eq!(figure(&m, &setup, "load_s"), Some(1.800_011_52));
        assert_eq!(figure(&m, &setup, "file_bytes"), Some(180_001_152.0));
        assert_eq!(figure(&m, &setup, "visible_max"), Some(990_000.0));

        let mut m = Measured::default();
        conclude_dv5(
            &mut m,
            &setup,
            &play(&[9.0], &[1.0], 0),
            &Observed::default(),
            0.9,
            90_000_000,
        );
        let (_, _, value, pass) = budget(&m, "P-3").expect("P-3");
        assert!((value - 1.0).abs() < 1e-12 && pass, "{value}");
        assert_eq!(
            budget(&m, "P-1"),
            Some(("frame_gpu_ms_max", 12.4, 9.0, true))
        );
    }

    /// Q-407: a block whose visible count exceeded C leaves P-1 reported, not judged,
    /// with the note; P-3 and P-4 are still judged.
    #[test]
    fn dv5_overflow_reports_p1_without_judging_it() {
        let setup = Json::obj([("tier", Json::str("quest-balanced"))]);
        let mut m = Measured::default();
        let p = play(&[70.0, 76.0], &[3.8, 3.9], 1);
        conclude_dv5(&mut m, &setup, &p, &Observed::default(), 1.0, 180_001_152);
        assert_eq!(budget(&m, "P-1"), None);
        assert_eq!(m.notes, [DV5_OVERFLOW_NOTE]);
        assert_eq!(figure(&m, &setup, "frame_gpu_ms_max"), Some(76.0));
        assert_eq!(figure(&m, &setup, "overflow_blocks"), Some(1.0));
        assert_eq!(
            budget(&m, "P-4"),
            Some(("frame_sort_ms_max", 2.5, 3.9, false))
        );
        assert!(budget(&m, "P-3").is_some_and(|b| b.3));
    }

    /// §1.6 at speed 1: τ advances by the elapsed time from τ₀ and stays at τ₁ once the
    /// span is played (Dv-5's "10 s at every τ").
    #[test]
    fn play_tau_runs_at_speed_one_and_holds_the_span_end() {
        assert_eq!(play_tau(0.0, 10.0, 0.0), 0.0);
        assert_eq!(play_tau(0.0, 10.0, 3.5), 3.5);
        assert_eq!(play_tau(2.0, 12.0, 3.5), 5.5);
        assert_eq!(play_tau(0.0, 10.0, 12.0), 10.0);
    }
}
