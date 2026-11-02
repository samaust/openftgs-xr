// SPDX-License-Identifier: Apache-2.0

//! Dv-13 on the device (§3.4, §1.8, D-18, #229): once the session runs it logs
//! `dv13: running`, the driver's cue to put the headset to sleep; PASS once
//! `XrStateChanged` has left Running and returned to it. While Android has the app paused
//! the runner runs no update, so the states of the sleep arrive after the wake; the verdict
//! needs only that the session leaves Running and comes back.

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_mod_xr::session::{XrState, XrStateChanged};

use crate::device_test::verdict;

/// The test's name in the intent extra.
pub(crate) const NAME: &str = "dv13";
/// How long the first session may take to run.
const FIRST_RUN: Duration = Duration::from_secs(30);

#[derive(Debug, PartialEq, Clone, Copy)]
enum Step {
    WaitRunning,
    WaitLeave,
    WaitReturn,
    Done,
}

#[derive(Resource)]
struct Dv13 {
    started: Instant,
    step: Step,
    seen: Vec<String>,
}

/// Adds the test.
pub(super) fn add(app: &mut App) {
    app.insert_resource(Dv13 {
        started: Instant::now(),
        step: Step::WaitRunning,
        seen: Vec::new(),
    })
    .add_systems(Update, run);
}

fn run(mut test: ResMut<Dv13>, mut changed: MessageReader<XrStateChanged>) {
    for XrStateChanged(state) in changed.read() {
        if test.step == Step::Done {
            continue;
        }
        test.seen.push(format!("{state:?}"));
        let running = *state == XrState::Running;
        let next = match test.step {
            Step::WaitRunning if running => {
                info!("dv13: running");
                Step::WaitLeave
            }
            Step::WaitLeave if !running => Step::WaitReturn,
            Step::WaitReturn if running => {
                let detail = format!("states {}", test.seen.join(" -> "));
                verdict(NAME, true, &detail);
                Step::Done
            }
            s => s,
        };
        test.step = next;
    }
    if test.step == Step::WaitRunning && test.started.elapsed() > FIRST_RUN {
        let detail = format!("session not running after {} s", FIRST_RUN.as_secs());
        verdict(NAME, false, &detail);
        test.step = Step::Done;
    }
}
