// SPDX-License-Identifier: Apache-2.0

//! Dv-12 on the device (§3.4, #228): selects 72 Hz and then 90 Hz (so the change event to
//! 90 Hz fires even when the headset starts at 90), steps through every foveation
//! level, then sets the render scale to 0.8 and waits for the session to run again, then
//! to 1.0 and waits again (#729, Q-412: both re-created sizes logged). The verdict is PASS
//! once the sequence completes; the host driver judges the rest from `display`'s log lines
//! (the rate's change event, each level applied, the new sizes).

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_mod_xr::session::XrState;
use ftgsxr_xr::{DisplaySettings, Foveation};

use crate::device_test::verdict;

/// The test's name in the intent extra.
pub(crate) const NAME: &str = "dv12";
const TIMEOUT: Duration = Duration::from_secs(90);
/// How long each step is left to take effect.
const STEP: Duration = Duration::from_secs(2);
const LEVELS: [Foveation; 5] = [
    Foveation::Off,
    Foveation::Low,
    Foveation::Medium,
    Foveation::High,
    Foveation::Dynamic,
];

#[derive(Debug, PartialEq)]
enum Step {
    WaitRunning,
    Rate72,
    Rate,
    Level(usize),
    /// The session re-created at `scale`: `left_running` once it stopped.
    Scale {
        scale: f32,
        left_running: bool,
    },
    Done,
}

#[derive(Resource)]
struct Dv12 {
    started: Instant,
    since: Instant,
    step: Step,
}

/// Adds the test.
pub(super) fn add(app: &mut App) {
    let now = Instant::now();
    app.insert_resource(Dv12 {
        started: now,
        since: now,
        step: Step::WaitRunning,
    })
    .add_systems(Update, run);
}

fn run(
    mut test: ResMut<Dv12>,
    state: Option<Res<XrState>>,
    display: Option<ResMut<DisplaySettings>>,
) {
    if test.step == Step::Done {
        return;
    }
    if test.started.elapsed() > TIMEOUT {
        verdict(
            NAME,
            false,
            &format!("timed out at {:?}: {state:?}", test.step),
        );
        test.step = Step::Done;
        return;
    }
    let running = state.as_deref() == Some(&XrState::Running);
    let Some(mut ds) = display else { return };
    let waited = test.since.elapsed() >= STEP;
    let next = match test.step {
        Step::WaitRunning if running => {
            info!("dv12: rates {:?}, requesting 72 Hz", ds.available_rates);
            ds.refresh_rate = Some(72.0);
            Some(Step::Rate72)
        }
        Step::Rate72 if waited => {
            info!("dv12: requesting 90 Hz");
            ds.refresh_rate = Some(90.0);
            Some(Step::Rate)
        }
        Step::Rate if waited => {
            ds.foveation = LEVELS[0];
            Some(Step::Level(0))
        }
        Step::Level(i) if waited && i + 1 < LEVELS.len() => {
            ds.foveation = LEVELS[i + 1];
            Some(Step::Level(i + 1))
        }
        Step::Level(_) if waited => {
            ds.render_scale = 0.8;
            Some(Step::Scale {
                scale: 0.8,
                left_running: false,
            })
        }
        Step::Scale {
            scale,
            left_running: false,
        } if !running => Some(Step::Scale {
            scale,
            left_running: true,
        }),
        Step::Scale {
            scale,
            left_running: true,
        } if running && scale < 1.0 && waited => {
            info!("dv12: render scale 0.8 applied, requesting 1.0");
            ds.render_scale = 1.0;
            Some(Step::Scale {
                scale: 1.0,
                left_running: false,
            })
        }
        Step::Scale {
            left_running: true, ..
        } if running && waited => {
            verdict(
                NAME,
                true,
                "90 Hz, every level, render scale 0.8 then 1.0 requested",
            );
            Some(Step::Done)
        }
        _ => None,
    };
    if let Some(step) = next {
        test.step = step;
        test.since = Instant::now();
    }
}
