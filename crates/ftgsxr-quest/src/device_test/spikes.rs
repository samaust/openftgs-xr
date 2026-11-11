// SPDX-License-Identifier: Apache-2.0

//! PRD #718's session spikes S1 and S2 on the device (#725), on Dv-12's pattern: this
//! runner drives them and logs what it measures; the host's `dv` driver reads the lines.
//! They judge nothing: the verdict is PASS once the sequence completes, FAIL only when it
//! cannot (no model, a load failure, a timeout).
//!
//! The model the driver pushed as `dv/spike.safetensors` (Q-T725.1: B-1's every-row-
//! persistent kind) loads at τ_m 0.95 s, held (nothing plays it), and is kept 2 m ahead of the left eye each frame; the governor
//! is held off throughout so no knob moves under the measurement.
//!
//! - S1: whether the XR device has timestamp queries, then a 5 s window in which the
//!   first eye's statistics blocks (Q-125) are counted with those of non-zero
//!   `gpu_times_ms.total`, and `ftgsxr_xr`'s counters window `s1` runs (the runtime's
//!   counters, their units and how often they change).
//! - S2: at foveation off, low, medium and high in turn, 2 s to settle, then the median
//!   GPU times of 120 distinct first-eye blocks, beside counters window `s2 <level>`.
//!
//! Lines: `spikes: timestamp queries on the XR device: <bool>`, `spikes: s1 …`,
//! `spikes: s2 <level> median total <ms> ms …`; `ftgsxr_xr` logs `perf metrics …` and
//! `display` the swapchain's size.
// Q-T725.1

use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::render::renderer::RenderDevice;
use bevy::render::settings::WgpuFeatures;
use bevy_mod_xr::camera::XrCamera;
use bevy_mod_xr::session::XrState;
use ftgsxr_app::{Governor, SplatCamera, SplatEvent, SplatStats, SplatTime};
use ftgsxr_xr::{DisplaySettings, Foveation};

use crate::android::splat_load;
use crate::device_test::{dv_dir, verdict};

/// The test's name in the intent extra.
pub(crate) const NAME: &str = "spikes";
/// The pushed model's file name under `dv/`.
const MODEL: &str = "spike.safetensors";
/// The model's distance ahead of the left eye, metres (Q-T725.1).
const AHEAD_M: f32 = 2.0;
/// The load, the upload and both eyes.
const SETUP_TIMEOUT: Duration = Duration::from_secs(90);
/// Before S1, so the first frames' costs are out of it.
const WARM_UP: Duration = Duration::from_secs(2);
/// S1's window.
const S1_WINDOW: Duration = Duration::from_secs(5);
/// How long a new foveation level is left before its blocks count.
const SETTLE: Duration = Duration::from_secs(2);
/// S2's blocks per level.
const BLOCKS: usize = 120;
/// How long a level's blocks may take.
const LEVEL_TIMEOUT: Duration = Duration::from_secs(30);
const LEVELS: [Foveation; 4] = [
    Foveation::Off,
    Foveation::Low,
    Foveation::Medium,
    Foveation::High,
];
/// How long the last counters window is left to log before the verdict.
const CLOSING: Duration = Duration::from_millis(500);
/// τ_m, held: the span's midpoint (Q-102's B-3 time; the model's span is \[0, 1.9\]).
const TAU: f64 = 0.95;
/// The window entities' prefix `ftgsxr_xr`'s counters wait for (Q-308's pattern).
const METRICS: &str = "ftgsxr-test: metrics ";

#[derive(Debug, PartialEq)]
enum Step {
    Setup,
    WarmUp,
    S1,
    Settle(usize),
    Measure(usize),
    /// The last window's counters are logged a frame after its marker goes.
    Closing,
    Done,
}

/// One statistics block's figures.
#[derive(Clone, Copy, Debug)]
struct Block {
    cull: f32,
    sort: f32,
    draw: f32,
    total: f32,
    visible: u32,
    overflow: bool,
}

#[derive(Resource)]
struct Spikes {
    started: Instant,
    since: Instant,
    step: Step,
    model: Entity,
    uploaded: bool,
    marker: Option<Entity>,
    blocks: Vec<Block>,
    sample: Option<String>,
    done_levels: Vec<String>,
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
    info!("spikes: model {} bytes", bytes.len());
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
    let now = Instant::now();
    app.insert_resource(Spikes {
        started: now,
        since: now,
        step: Step::Setup,
        model,
        uploaded: false,
        marker: None,
        blocks: Vec::new(),
        sample: None,
        done_levels: Vec::new(),
    })
    .add_observer(on_event)
    .add_systems(Update, run);
}

fn on_event(event: On<SplatEvent>, mut test: ResMut<Spikes>) {
    match event.event() {
        SplatEvent::Uploaded { entity, bytes, .. } if *entity == test.model => {
            info!("spikes: model uploaded, {bytes} bytes");
            test.uploaded = true;
        }
        SplatEvent::Failed { error, .. } if test.step != Step::Done => {
            verdict(NAME, false, &format!("model load failed: {error}"));
            test.step = Step::Done;
        }
        _ => {}
    }
}

/// The upper median (the middle value, or the upper of the two middle ones).
fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(f32::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

#[allow(clippy::too_many_arguments)]
fn run(
    mut test: ResMut<Spikes>,
    state: Option<Res<XrState>>,
    display: Option<ResMut<DisplaySettings>>,
    governor: Option<ResMut<Governor>>,
    device: Option<Res<RenderDevice>>,
    stats: Res<SplatStats>,
    eyes: Query<(&XrCamera, &GlobalTransform, Has<SplatCamera>)>,
    mut transforms: Query<&mut Transform>,
    mut commands: Commands,
) {
    if test.step == Step::Done {
        return;
    }
    // Held off, so no knob moves under the measurement (E16).
    if let Some(mut g) = governor
        && g.enabled
    {
        g.enabled = false;
    }
    let left = eyes.iter().find(|(c, _, _)| c.0 == 0).map(|(_, g, _)| *g);
    if let (Some(left), Ok(mut t)) = (left, transforms.get_mut(test.model)) {
        t.translation = left.translation() + left.forward() * AHEAD_M;
    }
    let fresh = stats
        .is_changed()
        .then_some(stats.0)
        .flatten()
        .map(|s| Block {
            cull: s.gpu_times_ms.cull,
            sort: s.gpu_times_ms.sort,
            draw: s.gpu_times_ms.draw,
            total: s.gpu_times_ms.total,
            visible: s.visible_splats,
            overflow: s.overflow,
        });
    let running = state.as_deref() == Some(&XrState::Running);
    let elapsed = test.since.elapsed();
    let Some(mut ds) = display else { return };
    let next = match test.step {
        Step::Setup => {
            let splat_eyes = eyes.iter().filter(|(_, _, s)| *s).count();
            if test.uploaded && splat_eyes == 2 && running {
                let ts = device
                    .as_deref()
                    .map(|d| d.features().contains(WgpuFeatures::TIMESTAMP_QUERY));
                info!(
                    "spikes: timestamp queries on the XR device: {}",
                    ts.map_or("unknown (no render device)".to_owned(), |t| t.to_string())
                );
                Some(Step::WarmUp)
            } else if test.started.elapsed() > SETUP_TIMEOUT {
                let detail = format!(
                    "setup not done after {} s: uploaded {}, splat eyes {splat_eyes}, {state:?}",
                    SETUP_TIMEOUT.as_secs(),
                    test.uploaded,
                );
                verdict(NAME, false, &detail);
                Some(Step::Done)
            } else {
                None
            }
        }
        Step::WarmUp if elapsed >= WARM_UP => {
            test.blocks.clear();
            test.marker = Some(commands.spawn(Name::new(format!("{METRICS}s1"))).id());
            Some(Step::S1)
        }
        Step::S1 => {
            if let Some(b) = fresh {
                test.blocks.push(b);
                test.sample = Some(format!("{:?}", stats.0.map(|s| s.gpu_times_ms)));
            }
            if elapsed >= S1_WINDOW {
                let nonzero = test.blocks.iter().filter(|b| b.total > 0.0).count();
                info!(
                    "spikes: s1 {} blocks in {:.2} s, {nonzero} with a non-zero total, sample GpuTimes {}",
                    test.blocks.len(),
                    elapsed.as_secs_f64(),
                    test.sample.as_deref().unwrap_or("none")
                );
                if let Some(m) = test.marker.take() {
                    commands.entity(m).despawn();
                }
                ds.foveation = LEVELS[0];
                Some(Step::Settle(0))
            } else {
                None
            }
        }
        Step::Settle(i) if elapsed >= SETTLE => {
            test.blocks.clear();
            let label = format!("{METRICS}s2 {:?}", LEVELS[i]);
            test.marker = Some(commands.spawn(Name::new(label)).id());
            Some(Step::Measure(i))
        }
        Step::Measure(i) => {
            if let Some(b) = fresh {
                test.blocks.push(b);
            }
            let full = test.blocks.len() >= BLOCKS;
            if full || elapsed > LEVEL_TIMEOUT {
                let b = &test.blocks;
                let pick = |f: fn(&Block) -> f32| median(b.iter().map(f).collect());
                let line = format!(
                    "s2 {:?} median total {} ms, cull {} sort {} draw {} over {} blocks in {:.2} s, visible {}..{}, overflow {}",
                    LEVELS[i],
                    pick(|b| b.total),
                    pick(|b| b.cull),
                    pick(|b| b.sort),
                    pick(|b| b.draw),
                    b.len(),
                    elapsed.as_secs_f64(),
                    b.iter().map(|b| b.visible).min().unwrap_or(0),
                    b.iter().map(|b| b.visible).max().unwrap_or(0),
                    b.iter().any(|b| b.overflow),
                );
                info!("spikes: {line}");
                if let Some(m) = test.marker.take() {
                    commands.entity(m).despawn();
                }
                if !full {
                    verdict(NAME, false, &format!("too few blocks: {line}"));
                    Some(Step::Done)
                } else if i + 1 < LEVELS.len() {
                    test.done_levels.push(format!("{:?}", LEVELS[i]));
                    ds.foveation = LEVELS[i + 1];
                    Some(Step::Settle(i + 1))
                } else {
                    test.done_levels.push(format!("{:?}", LEVELS[i]));
                    Some(Step::Closing)
                }
            } else {
                None
            }
        }
        Step::Closing if elapsed >= CLOSING => {
            let detail = format!("s1 and s2 at {}", test.done_levels.join(", "));
            verdict(NAME, true, &detail);
            Some(Step::Done)
        }
        _ => None,
    };
    if let Some(step) = next {
        test.step = step;
        test.since = Instant::now();
    }
}
