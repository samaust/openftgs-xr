// SPDX-License-Identifier: Apache-2.0

//! Dv-11 on the device (§3.4, §1.9, E14, Q-390). The clauses read `light`'s
//! crate-private control (the thumbnails, the raw and smoothed estimates), so they live
//! here; the quest runner spawns an entity named [`TRIGGER`] (Q-308), and this prints the
//! verdict line `ftgsxr-test: dv11 PASS|FAIL <detail>` (§3.4, Q-122).
//!
//! Once the session runs, the test sets passthrough mode and auto global (the driver
//! granted both camera permissions), waits for the camera to run and its first
//! thumbnail, counts the thumbnails for [`COUNT`] by their changing timestamps (clause 1:
//! 30 Hz or more), logs `dv11: cover` as the cue, and for up to [`WATCH`] watches the raw
//! thumbnails of the left passthrough camera, the one the shim opens: the first raw luma
//! at or below half the luma at the cue (the smoothed estimate then) marks the cover's
//! instant, and clause 2 passes when the smoothed estimate falls to half the luma at the
//! cue or less within [`WITHIN`] of it. Clause 2 is reported as not run, not FAIL
//! (Q-390, Q-273), when no cover comes within the window, when the luma at the cue is
//! below [`MIN_CUE_LUMA`] (a dark room), or when a miss follows a partial cover: the
//! raw luma's mean over the [`WITHIN`] after the cover stayed above [`DEEP`] of the luma
//! at the cue, where even a correct smoother need not halve. Then `lighting_mode` goes
//! to `off` and the camera must stop. The detail reports the rate, whether the camera
//! pose arrived, the luma ratio reached, the raw minimum and the time to half.

use std::time::{Duration, Instant};

use bevy::app::{App, Update};
use bevy::log::info;
use bevy::prelude::{
    Commands, IntoScheduleConfigs, Name, Query, Res, ResMut, Resource, not, resource_exists,
};
use bevy_mod_xr::session::XrState;
use ftgsxr_app::SettingsRes;

use super::control::Control;

/// The name of the entity the quest runner spawns to start the test (Q-308).
// Q-308, Q-390
pub(crate) const TRIGGER: &str = "ftgsxr-test: dv11";
const NAME: &str = "dv11";
/// Clause 1's count window.
const COUNT: Duration = Duration::from_secs(3);
/// Clause 1's rate.
const MIN_HZ: f64 = 30.0;
/// The window for the cover after the cue.
const WATCH: Duration = Duration::from_secs(20);
/// Clause 2: the smoothed estimate halved within this of the cover.
const WITHIN: Duration = Duration::from_secs(1);
/// A cover counts against the smoother only when the raw luma's mean over [`WITHIN`]
/// after it is at most this fraction of the luma at the cue: with E14's 0.5 s average a
/// correct smoother then reaches 0.25 + 0.75·e⁻² ≈ 0.35 at 1 s, one twice as slow
/// 0.25 + 0.75·e⁻¹ ≈ 0.53; above (0.5 − e⁻²)/(1 − e⁻²) ≈ 0.42 none can pass (Q-390).
// Q-390
const DEEP: f32 = 0.25;
/// The least luma at the cue (BT.709 luma, 0–1) clause 2 is judged at: below, a dark
/// room's noise would decide it (Q-390).
// Q-390
const MIN_CUE_LUMA: f32 = 0.05;
const START_TIMEOUT: Duration = Duration::from_secs(60);
const FIRST_TIMEOUT: Duration = Duration::from_secs(20);
const STOP_TIMEOUT: Duration = Duration::from_secs(5);

/// Clause 1's verdict from the thumbnails counted over `window` (Q-390).
// Q-390
pub(crate) fn rate_hz(thumbs: u64, window: Duration) -> f64 {
    thumbs as f64 / window.as_secs_f64()
}

/// Clause 2's state (Q-390).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Cover {
    /// No raw luma at or below half the luma at the cue yet.
    Waiting,
    /// Covered at the instant; the smoothed estimate not yet halved. Holds the sum and
    /// the count of the raw luma's ratios to the luma at the cue since the cover.
    Covered { at: Instant, sum: f32, n: u32 },
    /// Halved this long after the cover, within the window.
    Pass(Duration),
    /// Not halved within the window of a deep cover.
    Fail,
    /// Not halved within the window, but the cover was partial: its mean raw ratio.
    Partial(f32),
}

/// One observation of clause 2: `raw` and `smoothed` luma against `cue_luma` at `now`.
// Q-390
pub(crate) fn watch(cover: Cover, cue_luma: f32, raw: f32, smoothed: f32, now: Instant) -> Cover {
    let half = 0.5 * cue_luma;
    match cover {
        Cover::Waiting if raw <= half => {
            let c = Cover::Covered {
                at: now,
                sum: 0.0,
                n: 0,
            };
            watch(c, cue_luma, raw, smoothed, now)
        }
        Cover::Covered { at, .. } if smoothed <= half && now - at <= WITHIN => {
            Cover::Pass(now - at)
        }
        Cover::Covered { at, sum, n } if now - at > WITHIN => {
            let mean = if n > 0 { sum / n as f32 } else { f32::INFINITY };
            if mean <= DEEP {
                Cover::Fail
            } else {
                Cover::Partial(mean)
            }
        }
        Cover::Covered { at, sum, n } => Cover::Covered {
            at,
            sum: sum + raw / cue_luma,
            n: n + 1,
        },
        c => c,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Phase {
    Start,
    First,
    Counting {
        at: Instant,
        from: u64,
    },
    Watching {
        at: Instant,
        cue_luma: f32,
        cover: Cover,
    },
    Stopping,
    Done,
}

/// The test's state, inserted when the runner's entity appears.
#[derive(Resource, Debug)]
pub(crate) struct Dv11 {
    phase: Phase,
    since: Instant,
    parts: Vec<String>,
    pass: bool,
    min_ratio: f32,
    min_raw: f32,
}

/// Adds the test's systems: inert until an entity named [`TRIGGER`] exists.
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn add(app: &mut App) {
    app.add_systems(
        Update,
        (
            detect.run_if(not(resource_exists::<Dv11>)),
            run.run_if(resource_exists::<Dv11>),
        ),
    );
}

#[cfg_attr(test, allow(dead_code))]
fn detect(names: Query<&Name>, mut commands: Commands) {
    if names.iter().any(|n| n.as_str() == TRIGGER) {
        info!("dv11: started");
        commands.insert_resource(Dv11 {
            phase: Phase::Start,
            since: Instant::now(),
            parts: Vec::new(),
            pass: true,
            min_ratio: f32::INFINITY,
            min_raw: f32::INFINITY,
        });
    }
}

#[cfg_attr(test, allow(dead_code))]
// Q-390
fn run(
    mut test: ResMut<Dv11>,
    state: Option<Res<XrState>>,
    control: Res<Control>,
    mut settings: Option<ResMut<SettingsRes>>,
) {
    let now = Instant::now();
    let elapsed = now - test.since;
    match test.phase {
        Phase::Done => {}
        Phase::Start => {
            if state.as_deref() == Some(&XrState::Running) {
                if let Some(s) = settings.as_deref_mut() {
                    s.0.environment.mode = "passthrough".into();
                    s.0.environment.lighting_mode = "auto-global".into();
                }
                test.enter(Phase::First);
            } else if elapsed > START_TIMEOUT {
                test.fail(format!(
                    "no running session in {} s",
                    START_TIMEOUT.as_secs()
                ));
            }
        }
        Phase::First => {
            if control.running() && control.thumbs > 0 {
                let ms = elapsed.as_millis();
                test.parts.push(format!("first thumbnail after {ms} ms"));
                test.enter(Phase::Counting {
                    at: now,
                    from: control.thumbs,
                });
            } else if elapsed > FIRST_TIMEOUT {
                test.fail(format!(
                    "no thumbnail in {} s (flow {:?}, camera running {})",
                    FIRST_TIMEOUT.as_secs(),
                    control.flow,
                    control.running()
                ));
            }
        }
        Phase::Counting { at, from } => {
            if now - at >= COUNT {
                let hz = rate_hz(control.thumbs - from, now - at);
                let ok = hz >= MIN_HZ;
                test.parts.push(format!(
                    "clause 1 {}: {hz:.1} Hz",
                    if ok { "PASS" } else { "FAIL" }
                ));
                test.pass &= ok;
                test.parts
                    .push(format!("camera pose arrived: {}", control.pose_seen));
                let cue_luma = control.smoother.current().map_or(0.0, |e| e.luma);
                test.parts.push(format!("luma at the cue {cue_luma:.4}"));
                info!("dv11: cover");
                test.enter(Phase::Watching {
                    at: now,
                    cue_luma,
                    cover: Cover::Waiting,
                });
            }
        }
        Phase::Watching {
            at,
            cue_luma,
            cover,
        } => {
            let raw = control.raw.map_or(f32::INFINITY, |e| e.luma);
            let smoothed = control.smoother.current().map_or(f32::INFINITY, |e| e.luma);
            let lit = cue_luma >= MIN_CUE_LUMA;
            if cue_luma > 0.0 {
                test.min_ratio = test.min_ratio.min(smoothed / cue_luma);
                test.min_raw = test.min_raw.min(raw / cue_luma);
            }
            let cover = if lit {
                watch(cover, cue_luma, raw, smoothed, now)
            } else {
                cover
            };
            let end = match cover {
                _ if !lit => Some(format!(
                    "clause 2 not run: luma at the cue below {MIN_CUE_LUMA}"
                )),
                Cover::Pass(d) => Some(format!(
                    "clause 2 PASS: halved {} ms after the cover",
                    d.as_millis()
                )),
                Cover::Fail => {
                    test.pass = false;
                    Some("clause 2 FAIL: not halved within 1 s of the cover".to_owned())
                }
                Cover::Partial(mean) => Some(format!(
                    "clause 2 not run: partial cover (raw mean {mean:.3} of the cue's, above {DEEP})"
                )),
                Cover::Waiting if now - at > WATCH => Some(format!(
                    "clause 2 not run: no cover in {} s",
                    WATCH.as_secs()
                )),
                _ => None,
            };
            match end {
                Some(text) => {
                    test.parts.push(text);
                    let r = test.min_ratio;
                    test.parts.push(format!("luma ratio reached {r:.3}"));
                    let m = test.min_raw;
                    test.parts.push(format!("raw luma ratio minimum {m:.3}"));
                    if let Some(s) = settings.as_deref_mut() {
                        s.0.environment.lighting_mode = "off".into();
                    }
                    test.enter(Phase::Stopping);
                }
                None => {
                    test.phase = Phase::Watching {
                        at,
                        cue_luma,
                        cover,
                    }
                }
            }
        }
        Phase::Stopping => {
            if !control.running() {
                test.parts.push("camera stopped".into());
                test.done();
            } else if elapsed > STOP_TIMEOUT {
                test.fail("the camera still runs after lighting off".into());
            }
        }
    }
}

impl Dv11 {
    fn enter(&mut self, phase: Phase) {
        self.phase = phase;
        self.since = Instant::now();
    }

    fn fail(&mut self, why: String) {
        self.pass = false;
        self.parts.push(why);
        self.done();
    }

    fn done(&mut self) {
        let word = if self.pass { "PASS" } else { "FAIL" };
        info!("ftgsxr-test: {NAME} {word} {}", self.parts.join("; "));
        self.phase = Phase::Done;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Q-390: clause 1's rate is the thumbnails over the window.
    #[test]
    fn rate_hz_counts_over_the_window() {
        assert_eq!(rate_hz(90, Duration::from_secs(3)), 30.0);
        assert!(rate_hz(89, Duration::from_secs(3)) < MIN_HZ);
    }

    /// Q-390: the first raw luma at or below half the luma at the cue marks the cover;
    /// clause 2 passes when the smoothed estimate is halved within 1 s of it, fails after.
    #[test]
    fn watch_marks_the_cover_and_times_the_smoothed_drop() {
        let t = Instant::now();
        let s = |ms| t + Duration::from_millis(ms);
        assert_eq!(watch(Cover::Waiting, 1.0, 0.6, 1.0, t), Cover::Waiting);
        let c = watch(Cover::Waiting, 1.0, 0.5, 0.9, t);
        assert_eq!(
            c,
            Cover::Covered {
                at: t,
                sum: 0.5,
                n: 1
            }
        );
        let c = watch(c, 1.0, 0.1, 0.6, s(500));
        assert_eq!(
            c,
            Cover::Covered {
                at: t,
                sum: 0.6,
                n: 2
            }
        );
        assert_eq!(
            watch(c, 1.0, 0.1, 0.5, s(1000)),
            Cover::Pass(Duration::from_secs(1))
        );
        // A deep cover (raw mean 0.3 / 3 = 0.1 ≤ 0.25) missed: FAIL.
        let deep = Cover::Covered {
            at: t,
            sum: 0.3,
            n: 3,
        };
        assert_eq!(watch(deep, 1.0, 0.1, 0.6, s(1001)), Cover::Fail);
        // Raw and smoothed halved in one frame: the cover and the pass at once.
        assert_eq!(
            watch(Cover::Waiting, 1.0, 0.1, 0.4, t),
            Cover::Pass(Duration::ZERO)
        );
    }

    /// Q-390: a miss after a partial cover (the raw mean above 0.25 of the cue's) is
    /// not run, not FAIL; at 0.25 it is judged.
    #[test]
    fn watch_reports_a_partial_cover_not_a_fail() {
        let t = Instant::now();
        let late = t + Duration::from_millis(1001);
        let partial = Cover::Covered {
            at: t,
            sum: 0.9,
            n: 2,
        };
        assert_eq!(watch(partial, 1.0, 0.45, 0.6, late), Cover::Partial(0.45));
        let edge = Cover::Covered {
            at: t,
            sum: 0.5,
            n: 2,
        };
        assert_eq!(watch(edge, 1.0, 0.25, 0.6, late), Cover::Fail);
    }

    /// Q-390: a correct 0.5 s smoother passes a cover to 0.25 of the cue's luma within
    /// 1 s; one twice as slow fails it.
    #[test]
    fn deep_cover_tells_a_correct_smoother_from_a_slow_one() {
        let at_1s = |tau: f32| DEEP + (1.0 - DEEP) * (-1.0 / tau).exp();
        assert!(at_1s(0.5) <= 0.5);
        assert!(at_1s(1.0) > 0.5);
    }
}
