// SPDX-License-Identifier: Apache-2.0

//! The device tests' runner (§3.4, Q-122): the launch intent's string extra
//! `ftgsxr-test` names a test, whose verdict is the logcat line
//! `ftgsxr-test: <name> PASS|FAIL <detail>` under the tag `ftgsxr`.

use std::path::Path;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_mod_xr::session::XrState;

/// The intent extra naming the test (§3.4).
pub(crate) const EXTRA: &str = "ftgsxr-test";

/// How long `dv1` waits for the running session on the device before printing FAIL; the
/// host's 10 s criterion (Dv-1) is judged from the launch by the driver.
const DV1_TIMEOUT: Duration = Duration::from_secs(30);

mod dv3;
mod dv4;

/// Runs `name` when it is a headless test (Dv-3, Dv-4: before the XR session is created,
/// on the headless renderer's own device, Q-122) with the pushed and written files in
/// `dv`, printing its verdict; `false` when `name` is not one, for `add`.
pub(crate) fn run_headless(name: &str, dv: &Path) -> bool {
    let result = match name {
        "dv3" => dv3::run(dv),
        "dv4" => dv4::run(dv),
        _ => return false,
    };
    match result {
        Ok(detail) => verdict(name, true, &detail),
        Err(detail) => verdict(name, false, &detail),
    }
    true
}

/// The test this launch runs and whether its verdict is printed.
#[derive(Resource)]
struct DeviceTest {
    name: String,
    started: Instant,
    done: bool,
}

/// Prints the verdict line (§3.4).
fn verdict(name: &str, pass: bool, detail: &str) {
    let word = if pass { "PASS" } else { "FAIL" };
    info!("ftgsxr-test: {name} {word} {detail}");
}

/// Adds the test named by the extra to `app`; an unknown name prints FAIL at once.
pub(crate) fn add(app: &mut App, name: String) {
    match name.as_str() {
        "dv1" => {
            app.insert_resource(DeviceTest {
                name,
                started: Instant::now(),
                done: false,
            })
            .add_systems(Update, dv1);
        }
        _ => verdict(&name, false, "unknown test"),
    }
}

/// Dv-1: PASS once the session is running.
fn dv1(mut test: ResMut<DeviceTest>, state: Option<Res<XrState>>) {
    if test.done {
        return;
    }
    let elapsed = test.started.elapsed();
    if state.as_deref() == Some(&XrState::Running) {
        verdict(
            &test.name,
            true,
            &format!("session running after {:.1} s", elapsed.as_secs_f32()),
        );
        test.done = true;
    } else if elapsed > DV1_TIMEOUT {
        verdict(
            &test.name,
            false,
            &format!(
                "session not running after {} s: {state:?}",
                DV1_TIMEOUT.as_secs()
            ),
        );
        test.done = true;
    }
}
