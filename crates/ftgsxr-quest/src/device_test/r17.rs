// SPDX-License-Identifier: Apache-2.0

//! R-17 on the device (§3.2, E16, D-47, #730, Q-431): the app runs in
//! `quest-balanced`; the model the driver pushed as `dv/probe.safetensors` loads through
//! `SplatLoad` and is kept 1 m ahead of the left eye each frame (Dv-2's pose), with the
//! governor held off. Once it is uploaded, both eyes are splat cameras and the session
//! runs, the runner spawns the entity `ftgsxr-test: r17`, and `ftgsxr_xr`'s private
//! `display::r17` runs the sequence and prints the verdict (Q-308): it reads
//! `SplatTierState`, which this crate cannot name (§2.1's band table).
// Q-431, Q-308

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_mod_xr::camera::XrCamera;
use bevy_mod_xr::session::XrState;
use ftgsxr_app::{Governor, SplatCamera, SplatEvent};

use crate::android::splat_load;
use crate::device_test::{dv_dir, verdict};

/// The test's name in the intent extra.
pub(crate) const NAME: &str = "r17";
/// The tier the app runs in for this test (#730).
pub(crate) const TIER: &str = "quest-balanced";
/// The marker `ftgsxr_xr` waits for (Q-308).
const TRIGGER: &str = "ftgsxr-test: r17";
/// How long the load, the upload and both eyes may take before FAIL.
const SETUP_TIMEOUT: Duration = Duration::from_secs(60);
/// The probe's distance ahead of the left eye, metres (Dv-2's).
const AHEAD_M: f32 = 1.0;

#[derive(Resource)]
struct R17 {
    started: Instant,
    model: Entity,
    uploaded: bool,
    /// Whether the marker is spawned (`ftgsxr_xr` runs the rest), or the test failed.
    handed_over: bool,
    done: bool,
}

/// Adds the test.
pub(super) fn add(app: &mut App) {
    let path = dv_dir().map(|d| d.join("probe.safetensors"));
    let bytes = match path.as_deref().map(std::fs::read) {
        Some(Ok(b)) => b,
        other => {
            verdict(NAME, false, &format!("no probe at {path:?}: {other:?}"));
            return;
        }
    };
    info!("r17: probe {} bytes", bytes.len());
    let model = app
        .world_mut()
        .spawn((splat_load(bytes), Transform::IDENTITY))
        .id();
    app.insert_resource(R17 {
        started: Instant::now(),
        model,
        uploaded: false,
        handed_over: false,
        done: false,
    })
    .add_observer(on_event)
    .add_systems(Update, run);
}

fn on_event(event: On<SplatEvent>, mut test: ResMut<R17>) {
    match event.event() {
        SplatEvent::Uploaded { entity, bytes, .. } if *entity == test.model => {
            info!("r17: probe uploaded, {bytes} bytes");
            test.uploaded = true;
        }
        SplatEvent::Failed { error, .. } if !test.done => {
            verdict(NAME, false, &format!("probe load failed: {error}"));
            test.done = true;
        }
        _ => {}
    }
}

fn run(
    mut test: ResMut<R17>,
    state: Option<Res<XrState>>,
    governor: Option<ResMut<Governor>>,
    eyes: Query<(&XrCamera, &GlobalTransform, Has<SplatCamera>)>,
    mut transforms: Query<&mut Transform>,
    mut commands: Commands,
) {
    if test.done {
        return;
    }
    // Head-locked, so the probe's cost stays the same wherever the headset looks.
    let left = eyes.iter().find(|(c, _, _)| c.0 == 0).map(|(_, g, _)| *g);
    if let (Some(left), Ok(mut t)) = (left, transforms.get_mut(test.model)) {
        t.translation = left.translation() + left.forward() * AHEAD_M;
    }
    if test.handed_over {
        return;
    }
    // Held off until `ftgsxr_xr` starts the sequence (E16).
    if let Some(mut g) = governor
        && g.enabled
    {
        g.enabled = false;
    }
    let splat_eyes = eyes.iter().filter(|(_, _, s)| *s).count();
    let running = state.as_deref() == Some(&XrState::Running);
    if test.uploaded && splat_eyes == 2 && running {
        info!("r17: both eyes are splat cameras in {TIER}; starting");
        commands.spawn(Name::new(TRIGGER));
        test.handed_over = true;
    } else if test.started.elapsed() > SETUP_TIMEOUT {
        let detail = format!(
            "setup not done after {} s: uploaded {}, splat eyes {splat_eyes}, {state:?}",
            SETUP_TIMEOUT.as_secs(),
            test.uploaded,
        );
        verdict(NAME, false, &detail);
        test.done = true;
    }
}
