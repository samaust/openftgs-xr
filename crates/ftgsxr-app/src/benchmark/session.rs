// SPDX-License-Identifier: Apache-2.0

//! The session benchmarks B-2, B-7, B-8 and B-9's Quest half (§3.5, Q-406, Q-407,
//! Q-432, Q-434): the runner inserts `SessionBench`, and `AppPlugin`'s system runs
//! the benchmark it names in the XR session, writes its `ftgsxr-bench` file at `out` and
//! fills `result`.
//!
//! B-2, B-7 and B-8 load `model` as a held instance (a `SplatLoad` outside the library,
//! so nothing else poses or plays it), and for each setup of `protocol::setups` apply its
//! tier, sort mode, rate, occlusion and lighting, wait until the device shows them, then
//! pose the instance for each of the view's 72 steps where the eyes' midpoint sees it as
//! orbit camera k does, 8 frames of warm-up and 30 held, gathering each distinct
//! statistics block's GPU times and the runtime's counters (`HudData.runtime_counters`).
//! B-9 times the startup to the first frame drawn of `model` (Q-222). At the end the
//! settings the run changed are restored and given time to save.
// Q-406, Q-407, Q-432, Q-434

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use bevy::app::{App, Last};
use bevy::camera::{Camera, RenderTarget};
use bevy::log::{info, warn};
use bevy::prelude::{Entity, GlobalTransform, On, ResMut, Resource, Transform, With, World};
use bevy::render::renderer::RenderAdapterInfo;
use bevy::time::{Real, Time};
use ftgsxr_core::{Aabb, Pose, hex, open_model};
use ftgsxr_io::{ModelBytes, Settings};
use ftgsxr_render::{
    QualityTier, SplatCamera, SplatEvent, SplatLoad, SplatStats, SplatTierState, SplatTime,
};

use crate::benchmark::json::Json;
use crate::benchmark::{Measured, report_json};
use crate::capture::write_atomic;
use crate::governor::Governor;
use crate::manipulate::head_candidate;
use crate::persist::SettingsRes;
use crate::ui::{FeatureState, HudData, XrDisplay, XrFeatures};

pub(crate) mod protocol;

use protocol::{
    CameraRun, Observed, Outcome, Setup, conclude, first_frame, midpoint, setups, steps, summarize,
    transplant,
};

/// The session benchmark the Quest's runner names (Q-406): `AppPlugin` runs `name`
/// (B-2, B-7, B-8 or B-9) in the XR session on the model at `model`, writes its
/// `ftgsxr-bench` file at `out` with `headset` as `machine.headset`, and fills `result`,
/// `Err` with the failure's text.
// Q-406
#[derive(Resource)]
pub struct SessionBench {
    /// The benchmark's name.
    pub name: String,
    /// The model it runs on.
    pub model: Option<PathBuf>,
    /// Where its file is written.
    pub out: PathBuf,
    /// The headset's build fingerprint, for `machine.headset`.
    pub headset: Option<String>,
    /// `None` while it runs.
    pub result: Option<Result<(), String>>,
}

/// Frames before a camera's figures count (Q-407).
const WARM_UP: u32 = 8;
/// Frames a camera is held (Q-407).
const HELD: u32 = 30;
/// How long a camera whose held frames brought no block may be held (Q-432).
const BLOCK_TIMEOUT: Duration = Duration::from_secs(20);
/// How long a setup's conditions may take to show (Q-432).
const REACH_TIMEOUT: Duration = Duration::from_secs(15);
/// How long a setup is left to settle once reached, before its first camera.
const SETTLE: Duration = Duration::from_secs(2);
/// How long a load and upload may take.
const LOAD_TIMEOUT: Duration = Duration::from_secs(180);
/// The settings' save after the restore (§1.11: at most once a second after a change).
const SAVE_WAIT: Duration = Duration::from_millis(1500);
/// The requested rate within this shows as reached.
const RATE_TOLERANCE_HZ: f32 = 0.5;
/// The measures of a setup, each recorded skipped when the setup is not reached.
const MEASURES: [(&str, &str); 3] = [
    ("gpu_ms_max", "ms"),
    ("gpu_ms_median", "ms"),
    ("sort_ms_median", "ms"),
];

/// Adds the system and its observer (`AppPlugin`); inert without `SessionBench`.
pub(crate) fn add(app: &mut App) {
    app.add_systems(Last, run).add_observer(on_event);
}

/// The model's figures the run needs, read once.
struct Prepared {
    path: PathBuf,
    bytes: Vec<u8>,
    label: Json,
    n: usize,
    bounds: Aabb,
    tau: (f64, f64),
}

impl Prepared {
    fn read(path: &Path) -> Result<Prepared, String> {
        let model =
            open_model(path).map_err(|e| format!("{}: {}: {e}", e.code(), path.display()))?;
        let bytes = std::fs::read(path).map_err(|e| format!("E-IO: {}: {e}", path.display()))?;
        Ok(Prepared {
            path: path.to_path_buf(),
            bytes,
            label: Json::str(hex(&model.identity)),
            n: model.len(),
            bounds: model.content_bounds(),
            tau: (model.time.tau0, model.time.tau1),
        })
    }
}

/// Where a setup's run is.
enum Phase {
    /// The instance is loading in the setup's tier.
    Load(Instant),
    /// The setup's settings are applied; waiting until the device shows them.
    Reach(Instant),
    /// Reached; settling before the first camera.
    Settle(Instant),
    /// Camera `step`'s warm-up frames so far.
    Warm(u32),
    /// Camera `step`'s held frames so far, since when.
    Hold(u32, Instant),
    /// The settings are restored; the result waits for their save.
    Restore(Instant, Result<(), String>),
}

/// A B-2, B-7 or B-8 run.
#[derive(Resource)]
struct Run {
    name: String,
    out: PathBuf,
    headset: Option<String>,
    model: Prepared,
    saved: Settings,
    setups: Vec<Setup>,
    index: usize,
    steps: Vec<(Pose, f64)>,
    step: usize,
    phase: Phase,
    instance: Option<Entity>,
    loaded_tier: Option<&'static str>,
    uploaded: bool,
    failed: Option<String>,
    last_block: Option<ftgsxr_render::FrameStats>,
    camera: CameraRun,
    cameras: Vec<CameraRun>,
    observed: Observed,
    runs: Vec<(Setup, Json, Outcome)>,
    measured: Measured,
}

/// A B-9 run (Q-434).
#[derive(Resource)]
struct B9 {
    out: PathBuf,
    headset: Option<String>,
    path: PathBuf,
    instance: Entity,
    uploaded: bool,
    uploaded_s: Option<f64>,
    failed: Option<String>,
    ends: Vec<f64>,
    restore: Option<(Instant, Result<(), String>)>,
}

fn on_event(event: On<SplatEvent>, run: Option<ResMut<Run>>, b9: Option<ResMut<B9>>) {
    match (event.event(), run, b9) {
        (SplatEvent::Uploaded { entity, .. }, Some(mut r), _) if Some(*entity) == r.instance => {
            r.uploaded = true;
        }
        (SplatEvent::Failed { entity, error }, Some(mut r), _) if Some(*entity) == r.instance => {
            r.failed = Some(error.clone());
        }
        (SplatEvent::Uploaded { entity, .. }, _, Some(mut b)) if *entity == b.instance => {
            // Timed by the frame's system, at the frame's end.
            b.uploaded = true;
        }
        (SplatEvent::Failed { entity, error }, _, Some(mut b)) if *entity == b.instance => {
            b.failed = Some(error.clone());
        }
        _ => {}
    }
}

/// The system: starts the run `SessionBench` names, then advances it a frame.
fn run(world: &mut World) {
    let Some(bench) = world.get_resource::<SessionBench>() else {
        return;
    };
    if bench.result.is_some() {
        return;
    }
    // E16 off: no knob moves under the measurement (Q-407).
    if let Some(mut g) = world.get_resource_mut::<Governor>()
        && g.enabled
    {
        g.enabled = false;
    }
    if world.contains_resource::<Run>() {
        step_run(world);
    } else if world.contains_resource::<B9>() {
        step_b9(world);
    } else {
        start(world);
    }
}

/// Fills `SessionBench.result` (the runner prints it).
fn finish(world: &mut World, result: Result<(), String>) {
    match &result {
        Ok(()) => info!("session bench: done"),
        Err(e) => warn!("session bench: {e}"),
    }
    if let Some(mut b) = world.get_resource_mut::<SessionBench>() {
        b.result = Some(result);
    }
}

fn start(world: &mut World) {
    let bench = world.resource::<SessionBench>();
    let (name, out, headset) = (bench.name.clone(), bench.out.clone(), bench.headset.clone());
    let Some(path) = bench.model.clone() else {
        finish(
            world,
            Err(format!(
                "E-FILE-FORMAT: {name}: the headset's session benchmarks need a model"
            )),
        );
        return;
    };
    if name == "B-9" {
        // Q-434: the probe loads at once; it is read for its figures afterwards.
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => return finish(world, Err(format!("E-IO: {}: {e}", path.display()))),
        };
        let instance = spawn(world, bytes, 0.0);
        world.insert_resource(B9 {
            out,
            headset,
            path,
            instance,
            uploaded: false,
            uploaded_s: None,
            failed: None,
            ends: Vec::new(),
            restore: None,
        });
        return;
    }
    let Some(list) = setups(&name) else {
        return finish(
            world,
            Err(format!(
                "E-FILE-FORMAT: {name}: not a session benchmark (B-2, B-7, B-8, B-9)"
            )),
        );
    };
    let model = match Prepared::read(&path) {
        Ok(m) => m,
        Err(e) => return finish(world, Err(e)),
    };
    info!(
        "session bench {name}: {} rows from {}, {} setups",
        model.n,
        model.path.display(),
        list.len()
    );
    let saved = world
        .get_resource::<SettingsRes>()
        .map(|s| s.0.clone())
        .unwrap_or_default();
    world.insert_resource(Run {
        name,
        out,
        headset,
        model,
        saved,
        setups: list,
        index: 0,
        steps: Vec::new(),
        step: 0,
        phase: Phase::Load(Instant::now()),
        instance: None,
        loaded_tier: None,
        uploaded: false,
        failed: None,
        last_block: None,
        camera: CameraRun::default(),
        cameras: Vec::new(),
        observed: Observed::default(),
        runs: Vec::new(),
        measured: Measured::default(),
    });
    begin_setup(world);
}

/// Spawns the held instance (Q-432).
fn spawn(world: &mut World, bytes: Vec<u8>, tau: f64) -> Entity {
    world
        .spawn((
            SplatLoad {
                bytes: ModelBytes::Owned(bytes),
                kind_hint: None,
                reduce_to: None,
            },
            Transform::IDENTITY,
            SplatTime {
                tau,
                extrapolating: false,
            },
        ))
        .id()
}

/// The eyes' midpoint this frame (Q-432): the splat cameras that draw to a window or
/// the headset's views.
fn head(world: &mut World) -> Option<Pose> {
    let mut q =
        world.query_filtered::<(&Camera, &RenderTarget, &GlobalTransform), With<SplatCamera>>();
    let eyes: Vec<Pose> = q
        .iter(world)
        .filter(|(c, t, _)| head_candidate(c, t))
        .map(|(_, _, g)| {
            let (_, rotation, translation) = g.to_scale_rotation_translation();
            Pose {
                translation,
                rotation,
                scale: 1.0,
            }
        })
        .collect();
    midpoint(&eyes)
}

/// Poses `instance` where `camera` (model space) sees it from the head, at `tau`.
fn pose_instance(world: &mut World, instance: Entity, camera: Pose, tau: f64) {
    let Some(head) = head(world) else { return };
    let m = transplant(head, camera);
    if let Some(mut t) = world.get_mut::<Transform>(instance) {
        *t = Transform {
            translation: m.translation,
            rotation: m.rotation,
            scale: bevy::math::Vec3::splat(m.scale),
        };
    }
    if let Some(mut s) = world.get_mut::<SplatTime>(instance) {
        s.tau = tau;
    }
}

/// The setup's tier, as §1.12 writes it.
fn tier_of(name: &str) -> QualityTier {
    QualityTier::by_name(name).expect("invariant: the setups name §1.12's tiers")
}

/// The JSON of the setup (Q-432): the model, the tier and its knobs (the governor
/// is off, so they are the tier's), the sort mode, the requested rate, the environment
/// and the view.
fn setup_json(model: &Prepared, s: &Setup) -> Json {
    let tier = tier_of(s.tier);
    let sort = s.sort.unwrap_or(tier.sort_mode);
    Json::obj([
        ("model", model.label.clone()),
        ("n", Json::Num(model.n as f64)),
        ("tier", Json::str(s.tier)),
        ("render_scale", Json::Num(f64::from(tier.render_scale))),
        (
            "sh_cap",
            Json::Num(f64::from(SplatTierState::of(&tier).sh_cap)),
        ),
        ("sort", Json::str(format!("{sort:?}"))),
        ("foveation", Json::str(format!("{:?}", tier.foveation))),
        (
            "refresh_rate",
            s.refresh_hz
                .map_or_else(|| Json::str("runtime"), |r| Json::Num(f64::from(r))),
        ),
        ("passthrough", Json::Bool(true)),
        ("occlusion", Json::Bool(s.occlusion)),
        ("lighting", Json::str(s.lighting)),
        ("view", Json::str(s.view.name())),
    ])
}

/// Applies setup `index`'s settings and tier, loading the instance again on a change of
/// tier (Q-432).
fn begin_setup(world: &mut World) {
    let (setup, changed) = {
        let r = world.resource::<Run>();
        let s = r.setups[r.index].clone();
        let changed = r.loaded_tier != Some(s.tier);
        (s, changed)
    };
    if let Some(mut settings) = world.get_resource_mut::<SettingsRes>() {
        let env = &mut settings.0.environment;
        env.mode = "passthrough".into();
        env.occlusion = setup.occlusion;
        env.lighting_mode = setup.lighting.into();
        settings.0.xr.refresh_rate = setup.refresh_hz;
    }
    let tier = tier_of(setup.tier);
    if changed {
        world.insert_resource(SplatTierState::of(&tier));
    }
    if let Some(mut t) = world.get_resource_mut::<SplatTierState>() {
        let sort = setup.sort.unwrap_or(tier.sort_mode);
        if t.sort_mode != sort {
            t.sort_mode = sort;
        }
    }
    let (bounds, tau, old) = {
        let r = world.resource::<Run>();
        (r.model.bounds, r.model.tau, r.instance)
    };
    let list = steps(setup.view, &bounds, tau.0, tau.1);
    let now = Instant::now();
    let phase = if changed {
        if let Some(e) = old {
            world.despawn(e);
        }
        let bytes = world.resource::<Run>().model.bytes.clone();
        let e = spawn(world, bytes, list[0].1);
        let mut r = world.resource_mut::<Run>();
        r.instance = Some(e);
        r.uploaded = false;
        r.loaded_tier = Some(setup.tier);
        Phase::Load(now)
    } else {
        Phase::Reach(now)
    };
    info!(
        "session bench: setup {} of {}: {setup:?}",
        world.resource::<Run>().index + 1,
        world.resource::<Run>().setups.len()
    );
    let mut r = world.resource_mut::<Run>();
    r.steps = list;
    r.step = 0;
    r.cameras.clear();
    r.observed = Observed::default();
    r.phase = phase;
}

/// Whether the device shows setup `s` (Q-432); the reason it does not otherwise.
fn reached(world: &World, s: &Setup) -> Result<(), String> {
    if let Some(want) = s.refresh_hz {
        let now = world.get_resource::<XrDisplay>().and_then(|d| d.refresh_hz);
        if !now.is_some_and(|hz| (hz - want).abs() <= RATE_TOLERANCE_HZ) {
            return Err(format!("refresh rate {want} Hz not reached (at {now:?})"));
        }
    }
    let hud = world.get_resource::<HudData>();
    if s.occlusion && hud.and_then(|h| h.depth_image).is_none() {
        return Err("occlusion on, but no depth image arrived".into());
    }
    if s.lighting.starts_with("auto") {
        let lighting = world
            .get_resource::<XrFeatures>()
            .map_or(FeatureState::Absent, |f| f.lighting);
        if lighting != FeatureState::Available || hud.and_then(|h| h.light_gains).is_none() {
            return Err(format!(
                "lighting {} not running (camera {lighting:?}, no light estimate)",
                s.lighting
            ));
        }
    }
    Ok(())
}

/// The next setup, or the end of the run.
fn next_setup(world: &mut World) {
    let done = {
        let mut r = world.resource_mut::<Run>();
        r.index += 1;
        r.index >= r.setups.len()
    };
    if done {
        let result = write_run(world);
        restore(world, result);
    } else {
        begin_setup(world);
    }
}

/// Records the setup not reached: each measure skipped with `reason` (Q-417's shape).
fn skip_setup(world: &mut World, reason: String) {
    let mut r = world.resource_mut::<Run>();
    let s = r.setups[r.index].clone();
    let json = setup_json(&r.model, &s);
    for (measure, unit) in MEASURES {
        r.measured.skip(&json, measure, unit, reason.clone());
    }
    r.runs.push((s, json, Outcome::Skipped(reason)));
}

/// Restores the settings the run changed, then waits for their save (Q-432).
fn restore(world: &mut World, result: Result<(), String>) {
    let saved = world.resource::<Run>().saved.clone();
    if let Some(mut settings) = world.get_resource_mut::<SettingsRes>() {
        settings.0.environment = saved.environment;
        settings.0.xr.refresh_rate = saved.xr.refresh_rate;
    }
    world.resource_mut::<Run>().phase = Phase::Restore(Instant::now(), result);
}

fn step_run(world: &mut World) {
    if let Some(e) = world.resource::<Run>().failed.clone() {
        world.resource_mut::<Run>().failed = None;
        return restore(world, Err(format!("the model failed to load: {e}")));
    }
    // The instance, posed each frame where the current camera sees it.
    let (instance, camera, tau) = {
        let r = world.resource::<Run>();
        let (pose, tau) = r
            .steps
            .get(r.step)
            .copied()
            .unwrap_or((Pose::IDENTITY, 0.0));
        (r.instance, pose, tau)
    };
    if let Some(e) = instance {
        pose_instance(world, e, camera, tau);
    }
    let block = world.get_resource::<SplatStats>().and_then(|s| s.0);
    let fresh = {
        let mut r = world.resource_mut::<Run>();
        let fresh = block.filter(|b| r.last_block != Some(*b));
        if fresh.is_some() {
            r.last_block = block;
        }
        fresh
    };
    let phase = std::mem::replace(&mut world.resource_mut::<Run>().phase, Phase::Warm(0));
    let next = match phase {
        Phase::Load(since) => {
            if world.resource::<Run>().uploaded {
                Phase::Reach(Instant::now())
            } else if since.elapsed() > LOAD_TIMEOUT {
                restore(
                    world,
                    Err(format!(
                        "the model did not upload within {} s",
                        LOAD_TIMEOUT.as_secs()
                    )),
                );
                return;
            } else {
                Phase::Load(since)
            }
        }
        Phase::Reach(since) => {
            let s = {
                let r = world.resource::<Run>();
                r.setups[r.index].clone()
            };
            let heads = head(world).is_some();
            match reached(world, &s) {
                Ok(()) if heads => Phase::Settle(Instant::now()),
                result if since.elapsed() > REACH_TIMEOUT => {
                    let reason = match result {
                        Err(e) => e,
                        Ok(()) => "no eye camera".to_owned(),
                    };
                    warn!("session bench: setup skipped: {reason}");
                    skip_setup(world, reason);
                    next_setup(world);
                    return;
                }
                _ => Phase::Reach(since),
            }
        }
        Phase::Settle(since) if since.elapsed() >= SETTLE => Phase::Warm(0),
        Phase::Settle(since) => Phase::Settle(since),
        Phase::Warm(n) if n + 1 >= WARM_UP => {
            let mut r = world.resource_mut::<Run>();
            r.camera = CameraRun::default();
            Phase::Hold(0, Instant::now())
        }
        Phase::Warm(n) => Phase::Warm(n + 1),
        Phase::Hold(n, since) => {
            let counters = world
                .get_resource::<HudData>()
                .map(|h| h.runtime_counters.clone())
                .unwrap_or_default();
            let display = world
                .get_resource::<XrDisplay>()
                .map(|d| (d.refresh_hz, [d.swapchain.x, d.swapchain.y]));
            let mut r = world.resource_mut::<Run>();
            if let Some(b) = fresh {
                r.camera.gpu_ms.push(f64::from(b.gpu_times_ms.total));
                r.camera.sort_ms.push(f64::from(b.gpu_times_ms.sort));
                r.camera.visible.push(b.visible_splats);
                r.camera.overflow |= b.overflow;
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
            let n = n + 1;
            let has_block = !r.camera.gpu_ms.is_empty();
            let k = r.step;
            if n >= HELD && has_block {
                let camera = std::mem::take(&mut r.camera);
                r.cameras.push(camera);
                r.step += 1;
                if r.step < r.steps.len() {
                    Phase::Warm(0)
                } else {
                    end_setup(world);
                    return;
                }
            } else if n >= HELD && since.elapsed() > BLOCK_TIMEOUT {
                let e = format!(
                    "camera {k}: no statistics block within {} s (no GPU times?)",
                    BLOCK_TIMEOUT.as_secs()
                );
                restore(world, Err(e));
                return;
            } else {
                Phase::Hold(n, since)
            }
        }
        Phase::Restore(since, result) => {
            if since.elapsed() >= SAVE_WAIT {
                world.remove_resource::<Run>();
                finish(world, result);
                return;
            }
            Phase::Restore(since, result)
        }
    };
    world.resource_mut::<Run>().phase = next;
}

/// Summarizes the setup's cameras and moves on.
fn end_setup(world: &mut World) {
    {
        let mut r = world.resource_mut::<Run>();
        let r = &mut *r;
        let s = r.setups[r.index].clone();
        let json = setup_json(&r.model, &s);
        let summary = summarize(&mut r.measured, &json, &r.cameras, &r.observed);
        info!(
            "session bench: {} {} {}: gpu_ms_max {:.3} gpu_ms_median {:.3} sort_ms_median {:.3} overflow {}",
            s.tier,
            s.view.name(),
            json.compact(),
            summary.gpu_ms_max,
            summary.gpu_ms_median,
            summary.sort_ms_median,
            summary.overflow
        );
        r.runs.push((s, json, Outcome::Ran(summary)));
    }
    next_setup(world);
}

/// The note on the session benchmarks' figures.
const SESSION_NOTE: &str = "gpu_ms is f_gpu, the statistics block's gpu_times_ms.total (Q-403): in shared-center mode the cull's begin to the last eye's draw end, in per-view mode the sum over the eyes; per camera the median over the distinct blocks of its 30 held frames after 8 warm-up frames, gpu_ms_max the slowest camera's (Q-407, Q-432); runtime_counter is XR_META_performance_metrics' median over the held frames, reported beside it, not judged (Q-433)";

/// The profile, as `headless`'s bench writes it (Q-103).
fn profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

/// Writes the run's file; the error's text otherwise.
fn write_run(world: &mut World) -> Result<(), String> {
    let gpu = adapter(world);
    let mut r = world.resource_mut::<Run>();
    let r = &mut *r;
    conclude(&r.name, &mut r.measured, &r.runs);
    r.measured.notes.push(SESSION_NOTE);
    let notes = r.measured.notes.iter().map(|n| (*n).to_owned()).collect();
    write(
        &r.name,
        &r.out,
        &gpu,
        &r.measured,
        notes,
        r.headset.as_deref(),
    )
}

fn adapter(world: &World) -> String {
    world
        .get_resource::<RenderAdapterInfo>()
        .map_or_else(|| "unknown".to_owned(), |i| i.name.clone())
}

fn write(
    name: &str,
    out: &Path,
    gpu: &str,
    measured: &Measured,
    notes: Vec<String>,
    headset: Option<&str>,
) -> Result<(), String> {
    let json = report_json(name, gpu, profile(), measured, Json::Null, notes, headset);
    write_atomic(out, json.as_bytes()).map_err(|e| format!("E-IO: {}: {e}", out.display()))?;
    info!("session bench {name}: wrote {}", out.display());
    Ok(())
}

/// Q-222's desktop camera, 3 m along +z looking at the origin, where B-9's probe is
/// posed from the head (Q-434).
fn b9_camera() -> Pose {
    Pose {
        translation: bevy::math::Vec3::new(0.0, 0.0, 3.0),
        rotation: ftgsxr_core::Quat::IDENTITY,
        scale: 1.0,
    }
}

/// The note on B-9's figures on the headset.
const B9_NOTE: &str = "B-9 on the headset times one startup per launch, from the time plugin's start (App::new()'s first plugins) to the first frame drawn after the probe's Uploaded, the update two before the statistics first count a splat (Q-222, Q-434); the probe was pushed by the host, so its read may come from the page cache";

fn step_b9(world: &mut World) {
    let elapsed = world
        .resource::<Time<Real>>()
        .startup()
        .elapsed()
        .as_secs_f64();
    let instance = world.resource::<B9>().instance;
    pose_instance(world, instance, b9_camera(), 0.0);
    if let Some((since, _)) = &world.resource::<B9>().restore {
        if since.elapsed() >= SAVE_WAIT {
            let (_, result) = world
                .resource_mut::<B9>()
                .restore
                .take()
                .expect("invariant: checked above");
            world.remove_resource::<B9>();
            finish(world, result);
        }
        return;
    }
    if let Some(e) = world.resource_mut::<B9>().failed.take() {
        world.resource_mut::<B9>().restore = Some((
            Instant::now(),
            Err(format!("the probe failed to load: {e}")),
        ));
        return;
    }
    {
        let mut b = world.resource_mut::<B9>();
        if b.uploaded && b.uploaded_s.is_none() {
            b.uploaded_s = Some(elapsed);
        }
        b.ends.push(elapsed);
    }
    let drawn = world
        .get_resource::<SplatStats>()
        .and_then(|s| s.0)
        .is_some_and(|s| s.visible_splats > 0);
    let Some(uploaded_s) = world.resource::<B9>().uploaded_s else {
        return;
    };
    if !drawn {
        return;
    }
    let first_frame_s = first_frame(&world.resource::<B9>().ends, uploaded_s);
    let result = write_b9(world, uploaded_s, first_frame_s);
    world.resource_mut::<B9>().restore = Some((Instant::now(), result));
}

fn write_b9(world: &mut World, uploaded_s: f64, first_frame_s: f64) -> Result<(), String> {
    let gpu = adapter(world);
    let tier = world
        .get_resource::<SplatTierState>()
        .map_or_else(|| "unknown".to_owned(), |t| t.tier.name.clone());
    let size = world
        .get_resource::<XrDisplay>()
        .map_or([0, 0], |d| [d.swapchain.x, d.swapchain.y]);
    let b = world.resource::<B9>();
    let model =
        open_model(&b.path).map_err(|e| format!("{}: {}: {e}", e.code(), b.path.display()))?;
    let setup = Json::obj([
        ("model", Json::str(hex(&model.identity))),
        ("n", Json::Num(model.len() as f64)),
        ("tier", Json::str(tier)),
        ("width", Json::Num(f64::from(size[0]))),
        ("height", Json::Num(f64::from(size[1]))),
        ("ui", Json::Bool(true)),
        ("run", Json::str("first")),
    ]);
    let mut m = Measured::default();
    m.figure(&setup, "uploaded_s", uploaded_s, "s");
    m.figure(&setup, "first_frame_s", first_frame_s, "s");
    info!("session bench B-9: uploaded {uploaded_s:.3} s, first frame {first_frame_s:.3} s");
    write(
        "B-9",
        &b.out,
        &gpu,
        &m,
        vec![B9_NOTE.to_owned()],
        b.headset.as_deref(),
    )
}
