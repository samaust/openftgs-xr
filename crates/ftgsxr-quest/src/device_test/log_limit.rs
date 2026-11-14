// SPDX-License-Identifier: Apache-2.0

//! The Quest's per-frame limit on the device (§1.13 Logs, Q-167, Q-445, #793): once the
//! session runs, two sites log on every frame for a few seconds, one marked with Q-167's
//! `per_frame` field, the other unmarked, each line numbered; the verdict reports how
//! many of each were logged. The host driver judges logcat's lines.

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_mod_xr::session::XrState;

use crate::device_test::verdict;

/// The test's name in the intent extra.
pub(crate) const NAME: &str = "log-limit";
/// How long the session may take to run before FAIL.
const TIMEOUT: Duration = Duration::from_secs(30);
/// How long both sites log on every frame.
const WINDOW: Duration = Duration::from_secs(4);

#[derive(Resource)]
struct LogLimit {
    started: Instant,
    logging: Option<Instant>,
    marked: u32,
    unmarked: u32,
    done: bool,
}

/// Adds the test.
pub(super) fn add(app: &mut App) {
    app.insert_resource(LogLimit {
        started: Instant::now(),
        logging: None,
        marked: 0,
        unmarked: 0,
        done: false,
    })
    .add_systems(Update, run);
}

fn run(mut test: ResMut<LogLimit>, state: Option<Res<XrState>>) {
    if test.done {
        return;
    }
    let running = state.as_deref() == Some(&XrState::Running);
    match test.logging {
        None if running => test.logging = Some(Instant::now()),
        None if test.started.elapsed() > TIMEOUT => {
            verdict(
                NAME,
                false,
                &format!("no running session after {TIMEOUT:?}: {state:?}"),
            );
            test.done = true;
            return;
        }
        Some(since) if since.elapsed() >= WINDOW => {
            let detail = format!("marked {} unmarked {}", test.marked, test.unmarked);
            verdict(NAME, true, &detail);
            test.done = true;
            return;
        }
        _ => {}
    }
    if test.logging.is_some() {
        // Q-167's marker: one site, limited to one line a second (Q-445).
        info!(per_frame = true, "log-limit: marked {}", test.marked);
        test.marked += 1;
        // A different site, unmarked: every line reaches logcat.
        info!("log-limit: unmarked {}", test.unmarked);
        test.unmarked += 1;
    }
}
