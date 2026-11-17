// SPDX-License-Identifier: Apache-2.0

//! Dv-2 (§3.4, D-4 at the running tier's sort mode, Q-399, spike S5): the probe the
//! driver pushed as `dv/probe.safetensors` loads through `SplatLoad`, is kept 1 m ahead
//! of the left eye once both eyes are splat cameras (kept there each frame), and is
//! drawn by both for 10 s; PASS with the frame count when the frames came at the default
//! 72 Hz, neither fewer nor more (`dv2_rate`, Q-138, #269), and the first eye's
//! statistics saw the probe. The driver judges the validation layer's messages (Q-123).
//!
//! The re-run with the map attached (#810, Q-464): the intent extra
//! `ftgsxr-foveation` names a level (`Off`, the no-map control, `Low`, `Medium`, `High`, `Dynamic`) held through
//! `DisplaySettings.foveation` from the start, or `cycle`, which steps high, medium,
//! off, medium and high, 2 s each, before the 10 s window, drawn at high. Every launch,
//! with or without the extra, opens its window 2 s after both eyes draw, so start-up's
//! first second is not counted (owner, 2026-11-16).
// Q-464

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy_mod_xr::camera::XrCamera;
// Q-137
use ftgsxr_app::{SplatCamera, SplatEvent, SplatStats};

use crate::android::splat_load;
use crate::device_test::{dv_dir, verdict};
use crate::dv2_rate::{FirstSecond, at_default_rate, window_lead};
use crate::jni_calls::intent_extra;
use ftgsxr_xr::{DisplaySettings, Foveation};

/// The test's name in the intent extra.
pub(crate) const NAME: &str = "dv2";

/// The drawing time Dv-2 asks for.
const WINDOW: Duration = Duration::from_secs(10);
/// How long the load, upload and the eyes may take before FAIL.
const SETUP_TIMEOUT: Duration = Duration::from_secs(30);
/// The probe's distance ahead of the left eye, metres.
const AHEAD_M: f32 = 1.0;
/// The intent extra naming the level (Q-464).
const LEVEL_EXTRA: &str = "ftgsxr-foveation";
/// The cycle's levels, each held [`CYCLE_STEP`] (Q-464).
const CYCLE: [Foveation; 5] = [
    Foveation::High,
    Foveation::Medium,
    Foveation::Off,
    Foveation::Medium,
    Foveation::High,
];
const CYCLE_STEP: Duration = Duration::from_secs(2);

/// What the extra asks for.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Levels {
    Settings,
    Hold(Foveation),
    Cycle,
}

fn levels(extra: Option<&str>) -> Option<Levels> {
    Some(match extra {
        None => Levels::Settings,
        Some("cycle") => Levels::Cycle,
        // The no-map control (Q-457): off draws without a map.
        Some("Off") => Levels::Hold(Foveation::Off),
        Some("Low") => Levels::Hold(Foveation::Low),
        Some("Medium") => Levels::Hold(Foveation::Medium),
        Some("High") => Levels::Hold(Foveation::High),
        Some("Dynamic") => Levels::Hold(Foveation::Dynamic),
        Some(_) => return None,
    })
}

#[derive(Resource)]
struct Dv2 {
    levels: Levels,
    /// When the cycle or the held level's warm-up began, once both eyes draw.
    cycle: Option<Instant>,
    /// Start-up's first second after both eyes draw, logged once (Q-464, #854).
    first_second: Option<FirstSecond>,
    started: Instant,
    model: Entity,
    uploaded: bool,
    window: Option<(Instant, u32)>,
    last_report: Instant,
    done: bool,
}

/// Adds the test.
pub(super) fn add(app: &mut App) {
    let extra = intent_extra(LEVEL_EXTRA);
    let Some(levels) = levels(extra.as_deref()) else {
        verdict(NAME, false, &format!("unknown {LEVEL_EXTRA} {extra:?}"));
        return;
    };
    info!("dv2: levels {levels:?}");
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
        levels,
        cycle: None,
        first_second: None,
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
    display: Option<ResMut<DisplaySettings>>,
) {
    if test.done {
        return;
    }
    let now = Instant::now();
    // Q-464: the level the extra asks for now.
    let want = match test.levels {
        Levels::Settings => None,
        Levels::Hold(l) => Some(l),
        Levels::Cycle => Some(match test.cycle {
            Some(c) => {
                let i = (now.duration_since(c).as_secs_f64() / CYCLE_STEP.as_secs_f64()) as usize;
                CYCLE[i.min(CYCLE.len() - 1)]
            }
            None => CYCLE[0],
        }),
    };
    if let (Some(l), Some(mut d)) = (want, display)
        && d.foveation != l
    {
        info!("dv2: foveation {l:?}");
        d.foveation = l;
    }
    // Q-464, #854: the frames of the first second after both eyes draw (with the cycle,
    // its first step's), reported and never judged.
    if let Some(lines) = test.first_second.as_mut().and_then(|f| f.frame(now)) {
        for line in lines {
            info!("{line}");
        }
    }
    let Some((start, frames)) = test.window else {
        let ready = eyes.iter().filter(|(_, _, s)| *s).count() == 2;
        if test.uploaded && ready {
            // The warm-up, or the cycle, runs before the window, so neither start-up's
            // first second nor the cycle's re-makes are timed (Q-464).
            if test.cycle.is_none() {
                info!("dv2: both eyes draw; the first second begins");
                test.first_second = Some(FirstSecond::new(now));
            }
            let c = *test.cycle.get_or_insert(now);
            let lead = window_lead(
                (test.levels == Levels::Cycle).then(|| CYCLE_STEP * CYCLE.len() as u32),
            );
            if now.duration_since(c) < lead {
                let left = eyes.iter().find(|(c, _, _)| c.0 == 0).map(|(_, g, _)| *g);
                if let (Some(left), Ok(mut t)) = (left, transforms.get_mut(test.model)) {
                    t.translation = left.translation() + left.forward() * AHEAD_M;
                }
                return;
            }
        }
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
