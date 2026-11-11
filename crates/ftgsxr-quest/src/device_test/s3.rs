// SPDX-License-Identifier: Apache-2.0

//! Spike S3 on the device (PRD #718, #729, Q-412): the app runs in `quest-low` (whose
//! render scale is 0.8) with `[xr] render_scale` unset; once both eyes are splat cameras
//! and have drawn for a few seconds the verdict is PASS. The host driver reads the
//! swapchain's size from `display`'s line and the splat layer's from the adapter's (§1.12
//! at revision 56: the swapchain at the recommended size, the layer at 0.8 of it).

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_mod_xr::camera::XrCamera;
use bevy_mod_xr::session::XrState;
use ftgsxr_app::SplatCamera;

use crate::device_test::verdict;

/// The test's name in the intent extra.
pub(crate) const NAME: &str = "s3";
/// The tier the app runs in for this test.
pub(crate) const TIER: &str = "quest-low";
/// How long the setup may take before FAIL.
const TIMEOUT: Duration = Duration::from_secs(30);
/// How long both eyes draw before the verdict.
const DRAW: Duration = Duration::from_secs(3);

#[derive(Resource)]
struct S3 {
    started: Instant,
    drawing: Option<Instant>,
    done: bool,
}

/// Adds the test.
pub(super) fn add(app: &mut App) {
    app.insert_resource(S3 {
        started: Instant::now(),
        drawing: None,
        done: false,
    })
    .add_systems(Update, run);
}

fn run(
    mut test: ResMut<S3>,
    state: Option<Res<XrState>>,
    eyes: Query<(), (With<XrCamera>, With<SplatCamera>)>,
) {
    if test.done {
        return;
    }
    let running = state.as_deref() == Some(&XrState::Running);
    match test.drawing {
        None if running && eyes.iter().count() == 2 => {
            info!("s3: both eyes are splat cameras in {TIER}");
            test.drawing = Some(Instant::now());
        }
        None if test.started.elapsed() > TIMEOUT => {
            verdict(
                NAME,
                false,
                &format!("no two splat eyes after {TIMEOUT:?}: {state:?}"),
            );
            test.done = true;
        }
        Some(since) if since.elapsed() >= DRAW => {
            verdict(NAME, true, &format!("both eyes drew in {TIER}"));
            test.done = true;
        }
        _ => {}
    }
}
