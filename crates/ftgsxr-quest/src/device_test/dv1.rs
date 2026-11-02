// SPDX-License-Identifier: Apache-2.0

//! Dv-1 on the device (§3.4): PASS once the session is running. The host's 10 s
//! criterion is judged from the launch by the driver.

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_mod_xr::session::XrState;

use crate::device_test::verdict;

/// The test's name in the intent extra.
pub(crate) const NAME: &str = "dv1";
/// How long the device waits for the running session before printing FAIL.
const TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Resource)]
struct Dv1 {
    started: Instant,
    done: bool,
}

/// Adds the test.
pub(super) fn add(app: &mut App) {
    app.insert_resource(Dv1 {
        started: Instant::now(),
        done: false,
    })
    .add_systems(Update, run);
}

fn run(mut test: ResMut<Dv1>, state: Option<Res<XrState>>) {
    if test.done {
        return;
    }
    let elapsed = test.started.elapsed();
    if state.as_deref() == Some(&XrState::Running) {
        verdict(
            NAME,
            true,
            &format!("session running after {:.1} s", elapsed.as_secs_f32()),
        );
        test.done = true;
    } else if elapsed > TIMEOUT {
        verdict(
            NAME,
            false,
            &format!(
                "session not running after {} s: {state:?}",
                TIMEOUT.as_secs()
            ),
        );
        test.done = true;
    }
}
