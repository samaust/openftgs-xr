// SPDX-License-Identifier: Apache-2.0

//! Dv-2 (§3.4, D-4 at the running tier's sort mode, Q-399, spike S5): the probe the
//! driver pushed as `dv/probe.safetensors` loads through `SplatLoad`, is kept 1 m ahead
//! of the left eye once both eyes are splat cameras (kept there each frame), and is
//! drawn by both for 10 s; PASS with the frame count when the frames came at the default
//! 72 Hz, neither fewer nor more (`dv2_rate`, Q-138, #269), and the first eye's
//! statistics saw the probe. The driver judges the validation layer's messages (Q-123).

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_mod_xr::camera::XrCamera;
// Q-137
use ftgsxr_app::{SplatCamera, SplatEvent, SplatStats};

use crate::android::splat_load;
use crate::device_test::{dv_dir, verdict};
use crate::dv2_rate::at_default_rate;

/// The test's name in the intent extra.
pub(crate) const NAME: &str = "dv2";

/// The drawing time Dv-2 asks for.
const WINDOW: Duration = Duration::from_secs(10);
/// How long the load, upload and the eyes may take before FAIL.
const SETUP_TIMEOUT: Duration = Duration::from_secs(30);
/// The probe's distance ahead of the left eye, metres.
const AHEAD_M: f32 = 1.0;

#[derive(Resource)]
struct Dv2 {
    started: Instant,
    model: Entity,
    uploaded: bool,
    window: Option<(Instant, u32)>,
    last_report: Instant,
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
    let model = app
        .world_mut()
        .spawn((splat_load(bytes), Transform::IDENTITY))
        .id();
    app.insert_resource(Dv2 {
        started: Instant::now(),
        model,
        uploaded: false,
        window: None,
        last_report: Instant::now(),
        done: false,
    })
    .add_observer(on_event)
    .add_systems(Update, run);
}

fn on_event(event: On<SplatEvent>, mut test: ResMut<Dv2>) {
    match event.event() {
        SplatEvent::Uploaded { entity, bytes, .. } if *entity == test.model => {
            info!("dv2: probe uploaded, {bytes} bytes");
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
    mut test: ResMut<Dv2>,
    eyes: Query<(&XrCamera, &GlobalTransform, Has<SplatCamera>)>,
    mut transforms: Query<&mut Transform>,
    stats: Res<SplatStats>,
) {
    if test.done {
        return;
    }
    let now = Instant::now();
    let Some((start, frames)) = test.window else {
        let ready = eyes.iter().filter(|(_, _, s)| *s).count() == 2;
        if test.uploaded && ready {
            info!(
                "dv2: both eyes are splat cameras; drawing for {} s",
                WINDOW.as_secs()
            );
            test.window = Some((now, 0));
        } else if test.started.elapsed() > SETUP_TIMEOUT {
            let detail = format!(
                "setup not done after {} s: uploaded {}, splat eyes {}",
                SETUP_TIMEOUT.as_secs(),
                test.uploaded,
                eyes.iter().filter(|(_, _, s)| *s).count()
            );
            verdict(NAME, false, &detail);
            test.done = true;
        }
        return;
    };
    // Head-locked, so the probe stays in view wherever the headset looks.
    let left = eyes.iter().find(|(c, _, _)| c.0 == 0).map(|(_, g, _)| *g);
    if let (Some(left), Ok(mut t)) = (left, transforms.get_mut(test.model)) {
        t.translation = left.translation() + left.forward() * AHEAD_M;
    }
    let frames = frames + 1;
    test.window = Some((start, frames));
    if now.duration_since(test.last_report) >= Duration::from_secs(1) {
        test.last_report = now;
        info!("dv2: frame {frames}, first eye's statistics {:?}", stats.0);
    }
    let elapsed = now.duration_since(start);
    if elapsed < WINDOW {
        return;
    }
    let rate = f64::from(frames) / elapsed.as_secs_f64();
    // Q-138
    let at_rate = at_default_rate(frames, elapsed.as_secs_f64());
    let seen = stats.0.is_some_and(|s| s.visible_splats > 0);
    let mut detail = format!(
        "{frames} frames in {:.2} s ({rate:.1} Hz), first eye's statistics {:?}",
        elapsed.as_secs_f64(),
        stats.0
    );
    if let Err(e) = &at_rate {
        detail.push_str(&format!("; {e}"));
    }
    verdict(NAME, at_rate.is_ok() && seen, &detail);
    test.done = true;
}
