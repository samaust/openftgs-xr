// SPDX-License-Identifier: Apache-2.0

//! Dv-5 on the headset (§3.4 Dv-5, §1.12 P-1, P-3, P-4; Q-407, Q-T732.1, Q-T732.2): the
//! session benchmark `Dv-5`. In `quest-balanced`, with passthrough and occlusion on and
//! the governor off, the budget model loads as a held instance, its load timed from the
//! file's read to its `Uploaded` (B-5's span, per 10⁸ of the file's bytes); once the
//! device shows the setup the instance is posed each frame where the eyes' midpoint sees
//! it as orbit camera 0 does, and played at speed 1 over the model's span, every distinct
//! statistics block's f_gpu and sort time kept. The file holds P-1 on the largest f_gpu,
//! P-3 on the load and P-4 on the largest sort time (`protocol::conclude_dv5`); the host
//! judges them (Q-T732.1).
// Q-407, Q-T732.1, Q-T732.2

use std::path::PathBuf;
use std::time::{Duration, Instant};

use bevy::log::{info, warn};
use bevy::prelude::{Entity, On, ResMut, Resource, World};
use ftgsxr_core::{Pose, hex, open_model};
use ftgsxr_io::Settings;
use ftgsxr_render::{FrameStats, SplatEvent, SplatStats, SplatTierState};

use super::protocol::{Observed, Play, Setup, View, conclude_dv5, play_tau, steps, with};
use super::{
    LOAD_TIMEOUT, Prepared, REACH_TIMEOUT, SAVE_WAIT, SETTLE, WARM_UP, adapter, finish, head,
    pose_instance, reached, setup_json, spawn, tier_of, write,
};
use crate::benchmark::Measured;
use crate::benchmark::json::Json;
use crate::persist::SettingsRes;
use crate::ui::{HudData, XrDisplay};

/// The benchmark's name in `SessionBench` (Q-T732.1).
pub(crate) const NAME: &str = "Dv-5";
/// Dv-5's tier (§3.4).
const TIER: &str = "quest-balanced";
/// Frames left for a tier switch (a new device, Q-239) before the load's clock starts.
const SWITCH_FRAMES: u32 = 30;
/// Blocks arrive two frames late (Q-81): the play's first frames bring the warm-up's,
/// and it runs this many frames past τ₁ for its own last ones.
const LAG_FRAMES: u32 = 2;
/// How long the play may run past its span without a block before the run fails.
const PLAY_SLACK: Duration = Duration::from_secs(20);

/// The note on Dv-5's figures.
const NOTE: &str = "Dv-5 (Q-407, Q-T732.1): the instance is posed each frame where the eyes' midpoint sees it as orbit camera 0 does and played at speed 1 over the model's span; every distinct statistics block of the play (the first two frames' blocks, the warm-up's, left out; two frames past the span's end kept) is judged: P-1 on frame_gpu_ms_max, the largest gpu_times_ms.total (f_gpu, Q-403), P-4 on frame_sort_ms_max, the largest gpu_times_ms.sort; blocks skipped while the GPU falls behind (Q-81) are not seen, so blocks may be fewer than frames";
/// The note on Dv-5's load time.
const LOAD_NOTE: &str = "Dv-5's load_s (Q-T732.2) is one load in the session, from the start of the file's read to the instance's Uploaded (read, parse, validate, pack on the task pool, upload at 64 MB per frame), against B-5's median of three on the headless device; the file was just read for its bounds, so the page cache may hold it and the time is a lower bound for a cold read (as B-5's)";
/// The note when the running tier was switched before the load.
const SWITCH_NOTE: &str = "the app started in another tier; Dv-5 switched to quest-balanced before the load's clock started";

/// Where the run is.
enum Phase {
    /// The tier was switched this many frames ago.
    Switch(u32),
    /// The load's clock runs since the read's start.
    Load(Instant),
    /// Uploaded; waiting until the device shows the setup.
    Reach(Instant),
    /// Reached; settling.
    Settle(Instant),
    /// Warm-up frames at camera 0, τ₀.
    Warm(u32),
    /// Playing since the instant, this many frames so far.
    Play(Instant, u32),
    /// The settings are restored; the result waits for their save.
    Restore(Instant, Result<(), String>),
}

/// A Dv-5 run.
#[derive(Resource)]
pub(super) struct Dv5 {
    out: PathBuf,
    headset: Option<String>,
    path: PathBuf,
    model: Prepared,
    file_bytes: u64,
    saved: Settings,
    switched: bool,
    camera: Pose,
    phase: Phase,
    instance: Option<Entity>,
    uploaded: bool,
    failed: Option<String>,
    load_s: f64,
    last_block: Option<FrameStats>,
    /// The play's frames since τ₁.
    past: u32,
    play: Play,
    observed: Observed,
}

/// The setup: passthrough and occlusion on, lighting off, the runtime's rate (§3.4).
fn setup() -> Setup {
    Setup {
        tier: TIER,
        sort: None,
        refresh_hz: None,
        occlusion: true,
        lighting: "off",
        view: View::Orbit,
    }
}

pub(super) fn on_event(event: On<SplatEvent>, run: Option<ResMut<Dv5>>) {
    let Some(mut r) = run else { return };
    match event.event() {
        SplatEvent::Uploaded { entity, .. } if Some(*entity) == r.instance => r.uploaded = true,
        SplatEvent::Failed { entity, error } if Some(*entity) == r.instance => {
            r.failed = Some(error.clone());
        }
        _ => {}
    }
}

/// Starts Dv-5 on the model at `path`: reads its figures (outside the load's span),
/// applies the setup's settings and tier.
pub(super) fn start(world: &mut World, out: PathBuf, headset: Option<String>, path: PathBuf) {
    let model = match open_model(&path) {
        Ok(m) => m,
        Err(e) => return finish(world, Err(format!("{}: {}: {e}", e.code(), path.display()))),
    };
    let file_bytes = match std::fs::metadata(&path) {
        Ok(m) => m.len(),
        Err(e) => return finish(world, Err(format!("E-IO: {}: {e}", path.display()))),
    };
    let prepared = Prepared {
        path: path.clone(),
        bytes: Vec::new(),
        label: Json::str(hex(&model.identity)),
        n: model.len(),
        bounds: model.content_bounds(),
        tau: (model.time.tau0, model.time.tau1),
    };
    drop(model);
    let camera = steps(
        View::Orbit,
        &prepared.bounds,
        prepared.tau.0,
        prepared.tau.1,
    )[0]
    .0;
    let s = setup();
    let saved = world
        .get_resource::<SettingsRes>()
        .map(|s| s.0.clone())
        .unwrap_or_default();
    if let Some(mut settings) = world.get_resource_mut::<SettingsRes>() {
        let env = &mut settings.0.environment;
        env.mode = "passthrough".into();
        env.occlusion = s.occlusion;
        env.lighting_mode = s.lighting.into();
        settings.0.xr.refresh_rate = s.refresh_hz;
    }
    let tier = tier_of(TIER);
    let switched = world
        .get_resource::<SplatTierState>()
        .is_none_or(|t| t.tier.name != TIER);
    if switched {
        world.insert_resource(SplatTierState::of(&tier));
    } else if let Some(mut t) = world.get_resource_mut::<SplatTierState>()
        && t.sort_mode != tier.sort_mode
    {
        t.sort_mode = tier.sort_mode;
    }
    info!(
        "session bench {NAME}: {} rows, {file_bytes} bytes from {}, span [{}, {}]{}",
        prepared.n,
        path.display(),
        prepared.tau.0,
        prepared.tau.1,
        if switched { ", tier switched" } else { "" }
    );
    let mut r = Dv5 {
        out,
        headset,
        path,
        model: prepared,
        file_bytes,
        saved,
        switched,
        camera,
        phase: Phase::Switch(0),
        instance: None,
        uploaded: false,
        failed: None,
        load_s: 0.0,
        last_block: None,
        past: 0,
        play: Play::default(),
        observed: Observed::default(),
    };
    if !switched {
        r.phase = Phase::Switch(SWITCH_FRAMES);
    }
    world.insert_resource(r);
}

/// Starts the load's clock, reads the file and spawns the held instance (B-5's span
/// begins at the read, Q-T732.2).
fn begin_load(world: &mut World) -> Phase {
    let started = Instant::now();
    let path = world.resource::<Dv5>().path.clone();
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            restore(world, Err(format!("E-IO: {}: {e}", path.display())));
            return take_phase(world);
        }
    };
    let tau0 = world.resource::<Dv5>().model.tau.0;
    let e = spawn(world, bytes, tau0);
    world.resource_mut::<Dv5>().instance = Some(e);
    Phase::Load(started)
}

fn take_phase(world: &mut World) -> Phase {
    std::mem::replace(&mut world.resource_mut::<Dv5>().phase, Phase::Warm(0))
}

/// Restores the settings the run changed, then waits for their save.
fn restore(world: &mut World, result: Result<(), String>) {
    let saved = world.resource::<Dv5>().saved.clone();
    if let Some(mut settings) = world.get_resource_mut::<SettingsRes>() {
        settings.0.environment = saved.environment;
        settings.0.xr.refresh_rate = saved.xr.refresh_rate;
    }
    world.resource_mut::<Dv5>().phase = Phase::Restore(Instant::now(), result);
}

/// Advances the run a frame.
pub(super) fn step(world: &mut World) {
    if let Some(e) = world.resource_mut::<Dv5>().failed.take() {
        return restore(world, Err(format!("the model failed to load: {e}")));
    }
    let (instance, camera, tau) = {
        let r = world.resource::<Dv5>();
        let tau = match r.phase {
            Phase::Play(since, _) => {
                play_tau(r.model.tau.0, r.model.tau.1, since.elapsed().as_secs_f64())
            }
            _ => r.model.tau.0,
        };
        (r.instance, r.camera, tau)
    };
    if let Some(e) = instance {
        pose_instance(world, e, camera, tau);
    }
    let block = world.get_resource::<SplatStats>().and_then(|s| s.0);
    let fresh = {
        let mut r = world.resource_mut::<Dv5>();
        let fresh = block.filter(|b| r.last_block != Some(*b));
        if fresh.is_some() {
            r.last_block = block;
        }
        fresh
    };
    let phase = take_phase(world);
    let next = match phase {
        Phase::Switch(n) if n >= SWITCH_FRAMES => begin_load(world),
        Phase::Switch(n) => Phase::Switch(n + 1),
        Phase::Load(started) => {
            if world.resource::<Dv5>().uploaded {
                let load_s = started.elapsed().as_secs_f64();
                world.resource_mut::<Dv5>().load_s = load_s;
                info!("session bench {NAME}: loaded in {load_s:.3} s");
                Phase::Reach(Instant::now())
            } else if started.elapsed() > LOAD_TIMEOUT {
                let e = format!(
                    "the model did not upload within {} s",
                    LOAD_TIMEOUT.as_secs()
                );
                return restore(world, Err(e));
            } else {
                Phase::Load(started)
            }
        }
        Phase::Reach(since) => {
            let heads = head(world).is_some();
            match reached(world, &setup()) {
                Ok(()) if heads => Phase::Settle(Instant::now()),
                result if since.elapsed() > REACH_TIMEOUT => {
                    let reason = match result {
                        Err(e) => e,
                        Ok(()) => "no eye camera".to_owned(),
                    };
                    return restore(world, Err(format!("setup not reached: {reason}")));
                }
                _ => Phase::Reach(since),
            }
        }
        Phase::Settle(since) if since.elapsed() >= SETTLE => Phase::Warm(0),
        Phase::Settle(since) => Phase::Settle(since),
        Phase::Warm(n) if n + 1 >= WARM_UP => Phase::Play(Instant::now(), 0),
        Phase::Warm(n) => Phase::Warm(n + 1),
        Phase::Play(since, frames) => return play_frame(world, since, frames, fresh),
        Phase::Restore(since, result) => {
            if since.elapsed() >= SAVE_WAIT {
                world.remove_resource::<Dv5>();
                return finish(world, result);
            }
            Phase::Restore(since, result)
        }
    };
    world.resource_mut::<Dv5>().phase = next;
}

/// One frame of the play: its block kept, the end after the span and the lag.
fn play_frame(world: &mut World, since: Instant, frames: u32, fresh: Option<FrameStats>) {
    let counters = world
        .get_resource::<HudData>()
        .map(|h| h.runtime_counters.clone())
        .unwrap_or_default();
    let display = world
        .get_resource::<XrDisplay>()
        .map(|d| (d.refresh_hz, [d.swapchain.x, d.swapchain.y]));
    let elapsed = since.elapsed().as_secs_f64();
    let frames = frames + 1;
    let mut r = world.resource_mut::<Dv5>();
    let span = r.model.tau.1 - r.model.tau.0;
    if let Some(b) = fresh
        && frames > LAG_FRAMES
    {
        r.play.gpu_ms.push(f64::from(b.gpu_times_ms.total));
        r.play.sort_ms.push(f64::from(b.gpu_times_ms.sort));
        r.play.visible.push(b.visible_splats);
        if b.overflow {
            r.play.overflow_blocks += 1;
        }
    }
    for c in counters {
        let Some(v) = c.value else { continue };
        match r.observed.counters.iter_mut().find(|(p, ..)| *p == c.path) {
            Some((_, _, values)) => values.push(v),
            None => r.observed.counters.push((c.path, c.unit, vec![v])),
        }
    }
    if let Some((hz, size)) = display {
        r.observed.refresh_hz = hz;
        r.observed.swapchain = size;
    }
    r.play.frames = frames;
    // The frames past the span's end that bring its last blocks.
    if elapsed >= span {
        r.past += 1;
    }
    let past = r.past;
    let has_blocks = !r.play.gpu_ms.is_empty();
    if past > LAG_FRAMES && has_blocks {
        r.play.play_s = elapsed;
        let result = write_run(world);
        return restore(world, result);
    }
    if elapsed > span + PLAY_SLACK.as_secs_f64() {
        let e = format!(
            "no statistics block within {} s of the play (no GPU times?)",
            PLAY_SLACK.as_secs()
        );
        return restore(world, Err(e));
    }
    r.phase = Phase::Play(since, frames);
}

/// Writes the run's file; the error's text otherwise.
fn write_run(world: &mut World) -> Result<(), String> {
    let gpu = adapter(world);
    let r = world.resource::<Dv5>();
    let base = setup_json(&r.model, &setup());
    let json = with(
        &with(
            &with(&base, "orbit_camera", Json::Num(0.0)),
            "speed",
            Json::Num(1.0),
        ),
        "span_s",
        Json::Num(r.model.tau.1 - r.model.tau.0),
    );
    let mut m = Measured::default();
    conclude_dv5(&mut m, &json, &r.play, &r.observed, r.load_s, r.file_bytes);
    m.notes.push(NOTE);
    m.notes.push(LOAD_NOTE);
    if r.switched {
        m.notes.push(SWITCH_NOTE);
    }
    let worst = |id: &str| {
        m.budgets.iter().find(|b| b.id == id).map_or_else(
            || "not judged".to_owned(),
            |b| {
                format!(
                    "{:.3} (limit {}, {})",
                    b.value,
                    b.limit,
                    if b.pass() { "met" } else { "missed" }
                )
            },
        )
    };
    let summary = format!(
        "session bench {NAME}: {} blocks over {} frames in {:.2} s; P-1 {}; P-3 {}; P-4 {}",
        r.play.gpu_ms.len(),
        r.play.frames,
        r.play.play_s,
        worst("P-1"),
        worst("P-3"),
        worst("P-4"),
    );
    if m.budgets.iter().all(|b| b.pass()) {
        info!("{summary}");
    } else {
        warn!("{summary}");
    }
    let notes = m.notes.iter().map(|n| (*n).to_owned()).collect();
    write(NAME, &r.out, &gpu, &m, notes, r.headset.as_deref())
}
