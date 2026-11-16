// SPDX-License-Identifier: Apache-2.0

//! Spike S-F1 (PRD #804, #805; §1.12 Headset display, D-58, E16's floor), on Q-420's
//! pattern: the model the driver pushed as `dv/spike.safetensors` (B-1's every-row-
//! persistent kind) loads in `quest-balanced` at τ_m 0.95 s, held, and is kept 2 m ahead
//! of the left eye each frame, with the governor held off. Once it is uploaded, both eyes
//! are splat cameras and the session runs, the runner spawns `ftgsxr-test: sf1` and
//! `ftgsxr_xr`'s private `display::sf1` runs the rest and prints the verdict (Q-308): it
//! writes `SplatTierState.render_scale`, which this crate cannot name (Q-133).
// Q-454, Q-420, Q-308

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_mod_xr::camera::XrCamera;
use bevy_mod_xr::session::XrState;
use ftgsxr_app::{Governor, SplatCamera, SplatEvent, SplatTime};

use crate::android::splat_load;
use crate::device_test::{dv_dir, verdict};

/// The test's name in the intent extra.
pub(crate) const NAME: &str = "sf1";
/// The tier the spike runs in (Q-420's default tier, named so the settings cannot move it).
pub(crate) const TIER: &str = "quest-balanced";
/// The marker `ftgsxr_xr` waits for (Q-308).
const TRIGGER: &str = "ftgsxr-test: sf1";
/// The pushed model's file name under `dv/` (Q-420's).
const MODEL: &str = "spike.safetensors";
/// The model's distance ahead of the left eye, metres (Q-420).
const AHEAD_M: f32 = 2.0;
/// The load, the upload and both eyes.
const SETUP_TIMEOUT: Duration = Duration::from_secs(90);
/// After setup, before the marker, so the first frames' costs are out of the windows.
const WARM_UP: Duration = Duration::from_secs(2);
/// τ_m, held: the span's midpoint (Q-102's B-3 time).
const TAU: f64 = 0.95;

#[derive(Resource)]
struct Sf1 {
    started: Instant,
    ready: Option<Instant>,
    model: Entity,
    uploaded: bool,
    handed_over: bool,
    done: bool,
}

/// Adds the test.
pub(super) fn add(app: &mut App) {
    let path = dv_dir().map(|d| d.join(MODEL));
    let bytes = match path.as_deref().map(std::fs::read) {
        Some(Ok(b)) => b,
        other => {
            verdict(NAME, false, &format!("no model at {path:?}: {other:?}"));
            return;
        }
    };
    info!("sf1: model {} bytes", bytes.len());
    let model = app
        .world_mut()
        .spawn((
            splat_load(bytes),
            Transform::IDENTITY,
            SplatTime {
                tau: TAU,
                extrapolating: false,
            },
        ))
        .id();
    app.insert_resource(Sf1 {
        started: Instant::now(),
        ready: None,
        model,
        uploaded: false,
        handed_over: false,
        done: false,
    })
    .add_observer(on_event)
    .add_systems(Update, run);
}

fn on_event(event: On<SplatEvent>, mut test: ResMut<Sf1>) {
    match event.event() {
        SplatEvent::Uploaded { entity, bytes, .. } if *entity == test.model => {
            info!("sf1: model uploaded, {bytes} bytes");
            test.uploaded = true;
        }
        SplatEvent::Failed { error, .. } if !test.done => {
            verdict(NAME, false, &format!("model load failed: {error}"));
            test.done = true;
        }
        _ => {}
    }
}

fn run(
    mut test: ResMut<Sf1>,
    state: Option<Res<XrState>>,
    governor: Option<ResMut<Governor>>,
    eyes: Query<(&XrCamera, &GlobalTransform, Has<SplatCamera>)>,
    mut transforms: Query<&mut Transform>,
    mut commands: Commands,
) {
    if test.done {
        return;
    }
    // Head-locked, so the load stays the same wherever the headset looks (Q-420).
    let left = eyes.iter().find(|(c, _, _)| c.0 == 0).map(|(_, g, _)| *g);
    if let (Some(left), Ok(mut t)) = (left, transforms.get_mut(test.model)) {
        t.translation = left.translation() + left.forward() * AHEAD_M;
    }
    // Held off (E16); `display::sf1` holds it too once started.
    if let Some(mut g) = governor
        && g.enabled
    {
        g.enabled = false;
    }
    if test.handed_over {
        return;
    }
    let splat_eyes = eyes.iter().filter(|(_, _, s)| *s).count();
    let running = state.as_deref() == Some(&XrState::Running);
    if test.uploaded && splat_eyes == 2 && running {
        let ready = *test.ready.get_or_insert_with(Instant::now);
        if ready.elapsed() >= WARM_UP {
            info!("sf1: both eyes are splat cameras in {TIER}; starting");
            commands.spawn(Name::new(TRIGGER));
            test.handed_over = true;
        }
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
