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
pub(crate) fn verdict(name: &str, pass: bool, detail: &str) {
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
        "dv2" => dv2::add(app, name),
        "dv12" => crate::dv12::add(app),
        "dv13" => crate::dv13::add(app),
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

/// Dv-2 (§3.4, D-4 per view, Q-121): the probe the driver pushed as `dv/probe.safetensors`
/// loads through `SplatLoad`, is kept 1 m ahead of the left eye once both eyes are
/// splat cameras (kept there each frame), and is drawn by both for 10 s; PASS with the frame count when the
/// frames came at the default 72 Hz and the first eye's statistics saw the probe. The
/// driver judges the validation layer's messages (Q-123).
mod dv2 {
    use std::time::{Duration, Instant};

    use bevy::prelude::*;
    use bevy_mod_xr::camera::XrCamera;
    // Q-137
    use ftgsxr_app::{ModelBytes, SplatCamera, SplatEvent, SplatLoad, SplatStats};

    use super::verdict;

    /// The drawing time Dv-2 asks for.
    const WINDOW: Duration = Duration::from_secs(10);
    /// The default refresh rate (§1.12).
    const RATE_HZ: f64 = 72.0;
    /// The share of 72 Hz's frames the window must hold.
    // Q-138
    const MIN_RATE_SHARE: f64 = 0.95;
    /// How long the load, upload and the eyes may take before FAIL.
    const SETUP_TIMEOUT: Duration = Duration::from_secs(30);
    /// The probe's distance ahead of the left eye, metres.
    const AHEAD_M: f32 = 1.0;

    #[derive(Resource)]
    struct Dv2 {
        name: String,
        started: Instant,
        model: Entity,
        uploaded: bool,
        window: Option<(Instant, u32)>,
        last_report: Instant,
        done: bool,
    }

    pub(super) fn add(app: &mut App, name: String) {
        let path = bevy::android::ANDROID_APP
            .get()
            .and_then(|a| a.external_data_path())
            .map(|p| p.join("dv/probe.safetensors"));
        let bytes = match path.as_deref().map(std::fs::read) {
            Some(Ok(b)) => b,
            other => {
                verdict(&name, false, &format!("no probe at {path:?}: {other:?}"));
                return;
            }
        };
        let model = app
            .world_mut()
            .spawn((
                SplatLoad {
                    bytes: ModelBytes::Owned(bytes),
                    kind_hint: None,
                    reduce_to: None,
                },
                Transform::IDENTITY,
            ))
            .id();
        app.insert_resource(Dv2 {
            name,
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
            SplatEvent::Uploaded { entity, bytes } if *entity == test.model => {
                info!("dv2: probe uploaded, {bytes} bytes");
                test.uploaded = true;
            }
            SplatEvent::Failed { error, .. } if !test.done => {
                verdict(&test.name, false, &format!("probe load failed: {error}"));
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
                verdict(&test.name, false, &detail);
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
        let min = MIN_RATE_SHARE * RATE_HZ * elapsed.as_secs_f64();
        let seen = stats.0.is_some_and(|s| s.visible_splats > 0);
        let detail = format!(
            "{frames} frames in {:.2} s ({rate:.1} Hz), first eye's statistics {:?}",
            elapsed.as_secs_f64(),
            stats.0
        );
        verdict(&test.name, f64::from(frames) >= min && seen, &detail);
        test.done = true;
    }
}
