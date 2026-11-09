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
//! thumbnails: the first raw luma at or below half the baseline (the smoothed estimate at
//! the cue) marks the cover's instant, and clause 2 passes when the smoothed estimate
//! falls to half the baseline or less within [`WITHIN`] of it; no cover within the window
//! reports clause 2 as not run, not FAIL (Q-390, Q-273). Then `lighting_mode` goes to
//! `off` and the camera must stop. The detail reports the rate, whether the camera pose
//! arrived, and the luma ratio reached.

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
const START_TIMEOUT: Duration = Duration::from_secs(60);
const FIRST_TIMEOUT: Duration = Duration::from_secs(20);
const STOP_TIMEOUT: Duration = Duration::from_secs(5);

/// Clause 1's verdict from the thumbnails counted over `window` (Q-390).
pub(crate) fn rate_hz(thumbs: u64, window: Duration) -> f64 {
    thumbs as f64 / window.as_secs_f64()
}

/// Clause 2's state (Q-390).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Cover {
    /// No raw luma at or below half the baseline yet.
    Waiting,
    /// Covered at the instant; the smoothed estimate not yet halved.
    Covered(Instant),
    /// Halved within the window of the cover.
    Pass,
    /// Not halved within the window of the cover.
    Fail,
}

/// One observation of clause 2: `raw` and `smoothed` luma against `baseline` at `now`.
pub(crate) fn watch(cover: Cover, baseline: f32, raw: f32, smoothed: f32, now: Instant) -> Cover {
    let half = 0.5 * baseline;
    match cover {
        Cover::Waiting if raw <= half => watch(Cover::Covered(now), baseline, raw, smoothed, now),
        Cover::Covered(at) if smoothed <= half && now - at <= WITHIN => Cover::Pass,
        Cover::Covered(at) if now - at > WITHIN => Cover::Fail,
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
        baseline: f32,
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
        });
    }
}

#[cfg_attr(test, allow(dead_code))]
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
                let baseline = control.smoother.current().map_or(0.0, |e| e.luma);
                test.parts.push(format!("baseline luma {baseline:.4}"));
                info!("dv11: cover");
                test.enter(Phase::Watching {
                    at: now,
                    baseline,
                    cover: Cover::Waiting,
                });
            }
        }
        Phase::Watching {
            at,
            baseline,
            cover,
        } => {
            let raw = control.raw.map_or(f32::INFINITY, |e| e.luma);
            let smoothed = control.smoother.current().map_or(f32::INFINITY, |e| e.luma);
            if baseline > 0.0 {
                test.min_ratio = test.min_ratio.min(smoothed / baseline);
            }
            let cover = if baseline > 0.0 {
                watch(cover, baseline, raw, smoothed, now)
            } else {
                cover
            };
            let end = match cover {
                Cover::Pass => Some("clause 2 PASS".to_owned()),
                Cover::Fail => {
                    test.pass = false;
                    Some("clause 2 FAIL: not halved within 1 s of the cover".to_owned())
                }
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
                    if let Some(s) = settings.as_deref_mut() {
                        s.0.environment.lighting_mode = "off".into();
                    }
                    test.enter(Phase::Stopping);
                }
                None => {
                    test.phase = Phase::Watching {
                        at,
                        baseline,
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

    /// Q-390: the first raw luma at or below half the baseline marks the cover; clause 2
    /// passes when the smoothed estimate is halved within 1 s of it, fails after.
    #[test]
    fn watch_marks_the_cover_and_times_the_smoothed_drop() {
        let t = Instant::now();
        let s = |ms| t + Duration::from_millis(ms);
        assert_eq!(watch(Cover::Waiting, 1.0, 0.6, 1.0, t), Cover::Waiting);
        let c = watch(Cover::Waiting, 1.0, 0.5, 0.9, t);
        assert_eq!(c, Cover::Covered(t));
        assert_eq!(watch(c, 1.0, 0.1, 0.6, s(500)), c);
        assert_eq!(watch(c, 1.0, 0.1, 0.5, s(1000)), Cover::Pass);
        assert_eq!(watch(c, 1.0, 0.1, 0.6, s(1001)), Cover::Fail);
        // Raw and smoothed halved in one frame: the cover and the pass at once.
        assert_eq!(watch(Cover::Waiting, 1.0, 0.1, 0.4, t), Cover::Pass);
    }
}
