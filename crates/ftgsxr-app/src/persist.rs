// SPDX-License-Identifier: Apache-2.0

//! Settings, scenes and poses to and from the world (§2.1 `ftgsxr_app::persist`, §1.5
//! Poses and Scene files, §1.11, D-23). `SettingsRes` is the source of truth (PRD #275):
//! its tables are mapped into the resources the features read whenever it changes, and
//! back from them when the user changes one; it is saved at most once a second after a
//! change and at exit, to `SettingsPath` (Q-163). `poses.json` follows `Poses`; scene files
//! live under the config directory's `scenes/`, their model locations relative to
//! `AppPaths.models` by Q-3's rule.

use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use bevy::app::{App, AppExit, Last, Plugin, PostUpdate, Update};
use bevy::camera::visibility::Visibility;
use bevy::camera::{PerspectiveProjection, Projection};
use bevy::ecs::message::MessageCursor;
use bevy::ecs::prelude::*;
use bevy::ecs::world::CommandQueue;
use bevy::log::warn;
use bevy::time::{Real, Time};
use bevy::transform::components::Transform;
use ftgsxr_io::{
    AudioSettings, CameraState, ClockState, Entry, IoError, LocomotionSettings, ModelRef,
    PoseStore, SceneFile, SceneModel, Settings, SourceSpec, Url, classify_name, load_settings,
    read_scene, save_settings, write_scene,
};
use ftgsxr_render::{Colormap, Overlays, RenderMode, SplatMode, SplatOverlays};

use crate::audio::AudioState;
use crate::intent::{InstanceAction, Intent, OverlayToggle, PlaybackCommand, UiCommand};
use crate::library::{
    EntryState, Library, LibraryCommand, LibraryEntry, LibraryFailure, LoadedModel, ShowOnUpload,
};
use crate::manipulate::{Poses, now_rfc3339, pose_of, transform_of};
use crate::platform::PlatformRes;
use crate::playback::{LoopSetting, Playback, RangeSetting};
use crate::rig::{CameraController, DesktopCamera, DesktopView, Locomotion, PlayerRig};
use crate::ui::{DockHooks, SceneLoadConfirmed, Tab, ToastRequest, UiState};

mod panels;

/// The player's settings in the world (§1.11, §2.1).
#[derive(Resource, Clone, Debug, Default)]
pub struct SettingsRes(pub Settings);

/// The settings file the player reads at startup and saves to (§1.7 `--settings FILE`,
/// §1.11): inserted by the caller to override the platform's `settings.toml`, else by
/// `AppPlugin` with that default.
// Q-163
#[derive(Resource, Clone, Debug, PartialEq, Eq)]
pub struct SettingsPath(pub PathBuf);

/// The warnings of a settings load the caller made itself before inserting `SettingsRes`
/// (`play` reads the file for the log level and the tier first, Q-170): `AppPlugin`
/// raises each E-SETTINGS one as a toast, as for its own load (§1.15).
// Q-T275.2
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct SettingsWarnings(pub Vec<String>);

/// What the UI and the input adapters ask of `persist` (§2.1). Like `LibraryCommand`, it
/// derives `Message` and `Event` (Q-166). A relative scene path names a file under the
/// config directory's `scenes/`, `.ftgsxr-scene.json` appended when it lacks it; the
/// commands ask nothing (the UI confirms a load first).
// Q-200
#[derive(Message, Event, Clone, Debug, PartialEq)]
pub enum PersistCommand {
    /// Writes `SettingsRes` to `SettingsPath` now.
    SaveSettings,
    /// Writes `scene_from_world` to the scene file.
    SaveScene(PathBuf),
    /// Replaces the current scene with the scene file's.
    LoadScene(PathBuf),
    /// Saves the instance's pose into `Poses` (through `manipulate`) and `poses.json`.
    SavePose(Entity),
    /// Forgets the instance's saved pose in `Poses` and `poses.json`.
    ForgetPose(Entity),
}

/// A failure `ui` raises as a toast (§1.15): E-SETTINGS from the settings' load, E-IO,
/// E-JSON or E-SCENE-FORMAT from a scene or pose file, E-SCENE-MISSING per missing model.
/// `text` is `<code>: <user text>`.
// Q-201
#[derive(Message, Clone, Debug, PartialEq)]
pub(crate) struct PersistFailure {
    pub(crate) code: String,
    pub(crate) text: String,
}

impl PersistFailure {
    fn of(e: &IoError) -> PersistFailure {
        PersistFailure {
            code: e.code().to_owned(),
            text: format!("{}: {}", e.code(), e.user_text()),
        }
    }
}

/// The most recent eight scene files, newest first (§1.5 Scene files): the UI's one-tap
/// list, refreshed at startup and after each save and load.
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub(crate) struct RecentScenes(pub(crate) Vec<PathBuf>);

/// The files `persist` writes beside the settings.
#[derive(Resource, Clone, Debug)]
struct Files {
    poses: PathBuf,
    scenes: PathBuf,
    models: PathBuf,
}

/// `poses.json`'s content as last read or written.
#[derive(Resource)]
struct PosesSaved(PoseStore);

/// Whether no settings file existed at startup: Help opens once (§1.7, Q-T275.4).
#[derive(Resource, Default)]
struct FirstRun(bool);

/// §1.7 Help: "opened once on the first run".
// Q-T275.4
fn open_help_on_first_run(first: Res<FirstRun>, mut ui: ResMut<UiState>) {
    if first.0 {
        ui.dock_open = true;
        ui.tab = Tab::Help;
    }
}

/// What the load at build time reported, raised in the first frame.
#[derive(Resource, Default)]
struct StartupFailures(Vec<PersistFailure>);

/// The debounce of the settings' save (§1.11: at most once a second after a change).
#[derive(Resource, Default)]
struct Saver {
    dirty: bool,
    last: Option<Duration>,
}

/// A loaded scene's instance attributes and clock, applied as the instances appear.
#[derive(Resource, Default)]
struct ScenePending {
    models: Vec<SceneModel>,
    clock: Option<ClockState>,
    /// Set once every instance appeared: the clock waits a frame for `playback` to
    /// follow their T_len.
    settled: bool,
    /// The missing models' attributes, for an instance of their identity that appears
    /// once located (Q-T275.3); the clock does not wait for them.
    located: Vec<SceneModel>,
}

/// Q-163: when `SettingsRes` is absent, reads the settings from `SettingsPath`, else
/// from `default` (inserting `SettingsPath` with it), and inserts `SettingsRes`; each
/// warning is logged and kept for an E-SETTINGS toast. A `SettingsRes` the caller
/// inserted is left alone. `Poses` is read from `poses.json` beside the default settings
/// when absent (Q-185).
// Q-163, Q-205, Q-51
pub(crate) fn load_into(world: &mut World, default: PathBuf) {
    let mut failures = Vec::new();
    let path = match world.get_resource::<SettingsPath>() {
        Some(p) => p.0.clone(),
        None => {
            world.insert_resource(SettingsPath(default.clone()));
            default.clone()
        }
    };
    // §1.15: E-SETTINGS is the set-aside file; other warnings are logged.
    let set_aside = |w: &str, failures: &mut Vec<PersistFailure>| {
        if w.starts_with("E-SETTINGS") {
            failures.push(PersistFailure {
                code: "E-SETTINGS".into(),
                text: w.to_owned(),
            });
        }
    };
    // Q-T275.4: the first run is the one without a settings file.
    world.insert_resource(FirstRun(!path.exists()));
    if world.contains_resource::<SettingsRes>() {
        // Q-T275.2: the caller's own load.
        if let Some(w) = world.remove_resource::<SettingsWarnings>() {
            for w in &w.0 {
                if let Some(text) = w.strip_prefix("E-IO") {
                    failures.push(PersistFailure {
                        code: "E-IO".into(),
                        text: format!("E-IO{text}"),
                    });
                } else {
                    set_aside(w, &mut failures);
                }
            }
        }
    } else {
        let settings = match load_settings(&path) {
            Ok((settings, warnings)) => {
                for w in warnings {
                    bevy::log::warn!("{}: {w}", path.display());
                    set_aside(&w, &mut failures);
                }
                settings
            }
            Err(e) => {
                // E-SETTINGS and E-IO are recoverable (§1.15): the defaults, with the warning.
                bevy::log::warn!("{}: {}: {e}", path.display(), e.code());
                failures.push(PersistFailure::of(&e));
                Settings::default()
            }
        };
        world.insert_resource(SettingsRes(settings));
    }
    // Q-195, Q-196: rig's resources from the settings, before its plugin builds, so the
    // desktop camera and the grid start as the settings say.
    let (view, loco) = {
        let s = &world.resource::<SettingsRes>().0;
        (desktop_view(s), locomotion_of(s))
    };
    world.insert_resource(view);
    world.insert_resource(loco);
    let config = default.parent().map(Path::to_path_buf).unwrap_or_default();
    let poses = config.join("poses.json");
    if !world.contains_resource::<Poses>() {
        let store = match PoseStore::load(&poses) {
            Ok(store) => store,
            Err(e) => {
                warn!("{}: {}: {e}", poses.display(), e.code());
                failures.push(PersistFailure::of(&e));
                PoseStore::default()
            }
        };
        world.insert_resource(Poses(store));
    }
    let baseline = world.resource::<Poses>().0.clone();
    world.insert_resource(PosesSaved(baseline));
    let paths = world
        .get_resource::<PlatformRes>()
        .map(|p| p.0.paths().clone());
    world.insert_resource(Files {
        poses,
        scenes: paths
            .as_ref()
            .map_or_else(|| config.join("scenes"), |p| p.scenes.clone()),
        models: paths.map(|p| p.models).unwrap_or_default(),
    });
    world.insert_resource(StartupFailures(failures));
}

/// The module's systems, added by `AppPlugin` after the feature plugins, with
/// `AppPlugin.scene` loaded at startup.
pub(crate) struct PersistPlugin {
    pub(crate) scene: Option<PathBuf>,
}

impl Plugin for PersistPlugin {
    fn build(&self, app: &mut App) {
        // §1.7 Panels (D-21): the settings' sections of ui's dock.
        {
            let mut hooks = app
                .init_resource::<DockHooks>()
                .world_mut()
                .resource_mut::<DockHooks>();
            hooks.add(Tab::Display, panels::display_section);
            hooks.add(Tab::Environment, panels::environment_section);
            hooks.add(Tab::Settings, panels::settings_tables);
        }
        app.add_message::<PersistCommand>()
            .add_message::<PersistFailure>()
            .add_message::<SceneLoadConfirmed>()
            .add_message::<ToastRequest>()
            .add_message::<Intent>()
            .add_message::<LibraryCommand>()
            .init_resource::<Saver>()
            .init_resource::<ScenePending>()
            .init_resource::<StartupFailures>()
            .init_resource::<FirstRun>()
            .add_systems(
                Update,
                (
                    raise_startup,
                    ui_bridge,
                    run_intents,
                    run_commands,
                    sync_settings,
                    centers_follow_settings,
                    save_settings_debounced,
                    save_poses,
                )
                    .chain()
                    // Before the library reads its commands: the first sync compares the
                    // settings' sources with the library's as they were before this
                    // frame's commands, so a source a command adds in the first frame is
                    // not removed as absent from the settings. Unordered, the
                    // multi-threaded executor could run the library first (§2.4 Threads).
                    .before(crate::library::LibraryCommands),
            )
            .add_systems(Update, ui_scale_follows_settings)
            .add_systems(
                bevy::app::Startup,
                open_help_on_first_run.run_if(resource_exists::<UiState>),
            )
            .add_systems(
                PostUpdate,
                (apply_pending_instances, apply_pending_clock).chain(),
            )
            .add_systems(Last, save_at_exit);
        let recent = app
            .world()
            .get_resource::<Files>()
            .map(|f| recent_scenes(&f.scenes))
            .unwrap_or_default();
        app.insert_resource(RecentScenes(recent));
        if let Some(scene) = &self.scene {
            app.world_mut()
                .write_message(PersistCommand::LoadScene(scene.clone()));
        }
    }
}

/// ui's confirmed scene loads become `PersistCommand::LoadScene`, and persist's failures
/// ui's toasts: `ui` comes before `persist` in §2.1's order, so the mapping lives here
/// (PRD #275 "Dependency direction").
// Q-200, Q-201
fn ui_bridge(
    mut confirmed: MessageReader<SceneLoadConfirmed>,
    mut failures: MessageReader<PersistFailure>,
    mut commands: MessageWriter<PersistCommand>,
    mut toasts: MessageWriter<ToastRequest>,
) {
    for SceneLoadConfirmed(p) in confirmed.read() {
        commands.write(PersistCommand::LoadScene(p.clone()));
    }
    for f in failures.read() {
        toasts.write(ToastRequest::error(f.text.clone()));
    }
}

fn raise_startup(mut startup: ResMut<StartupFailures>, mut out: MessageWriter<PersistFailure>) {
    for f in startup.0.drain(..) {
        out.write(f);
    }
}

/// `Intent::Ui(SaveScene)` saves a scene named by the time (Q-200); `LoadScene` is the
/// UI's, which confirms and sends `PersistCommand::LoadScene`. The overlay toggles (Appendix D.1 B, V; the Display tab): an overlay
/// is a `[display.overlays]` setting, flipped here and mapped onto every instance by
/// `sync_settings` (Q-207).
fn run_intents(
    mut reader: MessageReader<Intent>,
    mut out: MessageWriter<PersistCommand>,
    mut settings: ResMut<SettingsRes>,
) {
    for intent in reader.read() {
        match intent {
            Intent::Ui(UiCommand::SaveScene) => {
                out.write(PersistCommand::SaveScene(PathBuf::from(
                    default_scene_name(),
                )));
            }
            Intent::Overlay(o) => {
                let ov = &mut settings.0.display.overlays;
                match o {
                    OverlayToggle::Boxes => ov.boxes = !ov.boxes,
                    OverlayToggle::Velocities => ov.velocities = !ov.velocities,
                    OverlayToggle::TimeTint => ov.time_tint = !ov.time_tint,
                }
            }
            _ => {}
        }
    }
}

/// `scene-YYYYMMDD-HHMMSS` in UTC, from `saved_at`'s clock.
// Q-200
fn default_scene_name() -> String {
    let now = now_rfc3339();
    let digits: String = now.chars().take(19).filter(char::is_ascii_digit).collect();
    format!("scene-{}-{}", &digits[..8], &digits[8..])
}

/// A scene path: relative ones under `scenes/`, with the suffix (Q-200).
fn scene_path(scenes: &Path, p: &Path) -> PathBuf {
    const SUFFIX: &str = ".ftgsxr-scene.json";
    let mut p = if p.is_absolute() {
        p.to_path_buf()
    } else {
        scenes.join(p)
    };
    let name = p.file_name().map(|n| n.to_string_lossy().into_owned());
    if let Some(name) = name
        && !name.ends_with(SUFFIX)
    {
        p.set_file_name(format!("{name}{SUFFIX}"));
    }
    p
}

/// The most recent eight `*.ftgsxr-scene.json` under `dir`, newest first.
pub(crate) fn recent_scenes(dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut files: Vec<(std::time::SystemTime, PathBuf)> = rd
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .ends_with(".ftgsxr-scene.json")
        })
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    files.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    files.into_iter().take(8).map(|(_, p)| p).collect()
}

/// The commands, in order, with the world at hand.
fn run_commands(world: &mut World, mut cursor: Local<MessageCursor<PersistCommand>>) {
    let commands: Vec<PersistCommand> = cursor
        .read(world.resource::<Messages<PersistCommand>>())
        .cloned()
        .collect();
    for c in commands {
        match c {
            PersistCommand::SaveSettings => save_now(world),
            // Q-205: the pose goes through `manipulate`'s action, `poses.json` follows
            // `Poses` (`save_poses`).
            PersistCommand::SavePose(instance) => {
                world.write_message(Intent::Instance {
                    instance,
                    action: InstanceAction::SavePose,
                });
            }
            PersistCommand::ForgetPose(instance) => {
                world.write_message(Intent::Instance {
                    instance,
                    action: InstanceAction::ForgetPose,
                });
            }
            PersistCommand::SaveScene(p) => {
                let files = world.resource::<Files>().clone();
                let path = scene_path(&files.scenes, &p);
                let scene = scene_from_world(world);
                if let Err(e) = write_scene(&path, &scene) {
                    warn!("{}: {}: {e}", path.display(), e.code());
                    world.write_message(PersistFailure::of(&e));
                }
                world.insert_resource(RecentScenes(recent_scenes(&files.scenes)));
            }
            PersistCommand::LoadScene(p) => {
                let files = world.resource::<Files>().clone();
                let path = scene_path(&files.scenes, &p);
                let scene = std::fs::read(&path)
                    .map_err(IoError::from)
                    .and_then(|b| read_scene(&b));
                let mut scene = match scene {
                    Ok(s) => s,
                    Err(e) => {
                        warn!("{}: {}: {e}", path.display(), e.code());
                        world.write_message(PersistFailure::of(&e));
                        continue;
                    }
                };
                resolve_locations(&mut scene, &files.models);
                world.write_message(LibraryCommand::UnloadAll);
                let mut queue = CommandQueue::default();
                let missing = world.resource_scope(|world, mut library: Mut<Library>| {
                    let mut commands = Commands::new(&mut queue, world);
                    apply_scene(&mut commands, &scene, &mut library)
                });
                queue.apply(world);
                for name in missing {
                    warn!("scene {}: model {name} not found", path.display());
                    world.write_message(PersistFailure {
                        code: "E-SCENE-MISSING".into(),
                        text: format!("E-SCENE-MISSING: Model {name} not found; locate it"),
                    });
                }
                world.insert_resource(RecentScenes(recent_scenes(&files.scenes)));
            }
        }
    }
}

/// `p` with `.` and `..` removed lexically (Q-3: no symlink resolved, no disk touched).
fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            c => out.push(c.as_os_str()),
        }
    }
    out
}

/// Q-3: a local path lexically under `models` is written relative to it; others as given.
// Q-3
fn relative_location(path: &Path, models: &Path) -> String {
    let (p, m) = (normalize(path), normalize(models));
    if !m.as_os_str().is_empty()
        && let Ok(rel) = p.strip_prefix(&m)
    {
        return rel.to_string_lossy().into_owned();
    }
    path.to_string_lossy().into_owned()
}

/// Q-3: a relative local location joined onto `models`.
fn resolve_location(location: &str, models: &Path) -> String {
    let p = Path::new(location);
    if p.is_relative() && Url::parse(location).is_err() {
        models.join(p).to_string_lossy().into_owned()
    } else {
        location.to_owned()
    }
}

fn resolve_locations(scene: &mut SceneFile, models: &Path) {
    for m in &mut scene.models {
        if m.source.kind == "dir" {
            m.source.location = resolve_location(&m.source.location, models);
        }
        if let Some(a) = &mut m.audio {
            *a = resolve_location(a, models);
        }
    }
}

/// A scene model's source as a reference: `dir` a file path, `http` a URL, `documents`
/// a content URI (Q-203).
fn model_ref(spec: &SourceSpec) -> Option<ModelRef> {
    match spec.kind.as_str() {
        "http" => Url::parse(&spec.location).ok().map(ModelRef::Http),
        "documents" => Some(ModelRef::Document(spec.location.clone())),
        _ => Some(ModelRef::File(PathBuf::from(&spec.location))),
    }
}

fn ref_location(r: &ModelRef, models: &Path) -> (String, String) {
    match r {
        ModelRef::File(p) => ("dir".into(), relative_location(p, models)),
        ModelRef::Http(u) => ("http".into(), u.to_string()),
        ModelRef::Document(d) => ("documents".into(), d.clone()),
    }
}

fn colormap_of(s: &str) -> Colormap {
    match s {
        "time" => Colormap::TimeCenter,
        "speed" => Colormap::Speed,
        "opacity" => Colormap::Opacity,
        _ => Colormap::Dc,
    }
}

fn mode_name(m: Option<&SplatMode>) -> String {
    match m.map(|m| m.0) {
        None | Some(RenderMode::Splats) => "splats",
        Some(RenderMode::Centers { .. }) => "centers",
        Some(RenderMode::Both { .. }) => "both",
        Some(RenderMode::Hidden) => "hidden",
    }
    .into()
}

/// A render mode spelled as `[display] mode` (Q-50), the centers' radius and colormap
/// from `[display]`.
fn mode_of(name: &str, display: &ftgsxr_io::DisplaySettings) -> RenderMode {
    let pixel_radius = display.centers_radius;
    let colormap = colormap_of(&display.colormap);
    match name {
        "centers" => RenderMode::Centers {
            pixel_radius,
            colormap,
        },
        "both" => RenderMode::Both {
            pixel_radius,
            colormap,
        },
        "hidden" => RenderMode::Hidden,
        _ => RenderMode::Splats,
    }
}

fn overlays_of(o: &ftgsxr_io::OverlaySettings) -> Overlays {
    Overlays {
        boxes: o.boxes,
        velocities: o.velocities.then_some(o.velocity_every.max(1)),
        time_tint: o.time_tint,
    }
}

/// The scene as the world holds it (§1.5 Scene files, §1.11): every instance in entity
/// order with its local pose (under `ContentGroup`), the clock, `SettingsRes.environment`,
/// the rig's pose, the desktop camera's local pose under the rig with its field of view
/// and mode, and the bookmarks set, in slot order (Q-206). Model and audio locations
/// follow Q-3 against `AppPaths.models`. No anchor (M-5) and no per-instance lighting yet.
// Q-206, Q-3
pub fn scene_from_world(world: &World) -> SceneFile {
    let models_dir = world
        .get_resource::<PlatformRes>()
        .map(|p| p.0.paths().models.clone())
        .unwrap_or_default();
    let mut models = Vec::new();
    if let Some(mut q) = world.try_query::<(
        Entity,
        &LoadedModel,
        &Transform,
        Option<&Visibility>,
        Option<&SplatMode>,
        Option<&ShowOnUpload>,
    )>() {
        let mut rows: Vec<_> = q.iter(world).collect();
        rows.sort_by_key(|r| r.0);
        for (_, m, t, vis, mode, show) in rows {
            let (kind, location) = ref_location(&m.source, &models_dir);
            let visible = match show {
                Some(s) => s.0,
                None => vis != Some(&Visibility::Hidden),
            };
            models.push(SceneModel {
                source: SourceSpec {
                    kind,
                    location,
                    name: m.name.clone(),
                    flavor: None,
                },
                identity: m.identity.clone(),
                name: m.name.clone(),
                pose: pose_of(t),
                visible,
                mode: mode_name(mode),
                time_offset_s: m.time.offset_s,
                end_rule: m.time.end_rule,
                lighting: None,
                audio: m.audio.as_ref().map(|a| ref_location(a, &models_dir).1),
            });
        }
    }
    let clock = world.get_resource::<Playback>().map(|p| &p.clock);
    let clock = ClockState {
        t: clock.map_or(0.0, |c| c.time()),
        speed: clock.map_or(1.0, |c| c.speed()),
        loop_mode: world
            .get_resource::<LoopSetting>()
            .map_or(LoopSetting::default().0, |l| l.0),
        range: world.get_resource::<RangeSetting>().and_then(|r| r.0),
    };
    let environment = world
        .get_resource::<SettingsRes>()
        .map(|s| s.0.environment.clone())
        .unwrap_or_default();
    let rig = world
        .try_query_filtered::<&Transform, With<PlayerRig>>()
        .and_then(|mut q| q.iter(world).next().map(pose_of));
    let ctl = world.get_resource::<CameraController>();
    let camera = world
        .try_query_filtered::<(&Transform, &Projection), With<DesktopCamera>>()
        .and_then(|mut q| {
            q.iter(world).next().map(|(t, p)| CameraState {
                pose: pose_of(t),
                fov_deg: match p {
                    Projection::Perspective(p) => p.fov.to_degrees(),
                    _ => 60.0,
                },
                orbit: ctl.is_some_and(|c| c.orbit),
            })
        });
    let bookmarks = ctl
        .map(|c| c.bookmarks.iter().flatten().cloned().collect())
        .unwrap_or_default();
    SceneFile {
        models,
        clock,
        environment,
        anchor: None,
        rig,
        camera,
        bookmarks,
    }
}

/// Replaces the current scene with `scene` (§1.5 Scene files): each model whose source
/// still offers it loads through the library (`LibraryCommand::LoadRef`), its pose,
/// visibility, mode, offset, end rule and audio applied when the instance appears; a
/// local file that no longer exists is listed in `library` as `EntryState::Missing` and
/// its name returned (Q-54, E-SCENE-MISSING). The clock, `SettingsRes.environment`, the
/// rig, the desktop camera and the bookmarks are set through `commands`; T and the range
/// once the scene's instances have appeared (T_len is 0 before). The caller unloads the
/// current instances and resolves relative locations first (`LoadScene`, Q-3).
// Q-203
pub fn apply_scene(
    commands: &mut Commands,
    scene: &SceneFile,
    library: &mut Library,
) -> Vec<String> {
    let mut missing = Vec::new();
    let mut located = Vec::new();
    let mut pending = Vec::new();
    for m in &scene.models {
        let Some(r) = model_ref(&m.source) else {
            missing.push(m.name.clone());
            continue;
        };
        if let ModelRef::File(p) = &r
            && !p.exists()
        {
            missing.push(m.name.clone());
            located.push(m.clone());
            let source = library
                .sources
                .iter()
                .position(|(s, _)| s.kind == "dir" && p.starts_with(&s.location))
                .unwrap_or(0);
            if library.sources.is_empty() {
                continue;
            }
            let location = m.source.location.clone();
            if library
                .entries
                .iter()
                .any(|e| e.source == source && e.entry.location == location)
            {
                for e in &mut library.entries {
                    if e.source == source && e.entry.location == location {
                        e.state = EntryState::Missing;
                    }
                }
                continue;
            }
            library.entries.push(LibraryEntry {
                source,
                entry: Entry {
                    name: m.name.clone(),
                    kind: classify_name(&location),
                    location,
                    size: None,
                    mtime: None,
                    sha256: None,
                    thumbnail: None,
                    audio: m.audio.clone(),
                    scene: None,
                },
                info: None,
                identity: Some(m.identity.clone()),
                thumbnail: None,
                state: EntryState::Missing,
            });
            continue;
        }
        commands.write_message(LibraryCommand::LoadRef(r));
        pending.push(m.clone());
    }
    let clock = scene.clock.clone();
    let environment = scene.environment.clone();
    let rig = scene.rig;
    let camera = scene.camera.clone();
    let bookmarks = scene.bookmarks.clone();
    commands.queue(move |world: &mut World| {
        if let Some(mut p) = world.get_resource_mut::<Playback>() {
            let _ = p.clock.set_speed(clock.speed);
            p.clock.set_loop(clock.loop_mode);
        }
        if let Some(mut l) = world.get_resource_mut::<LoopSetting>() {
            l.0 = clock.loop_mode;
        }
        if let Some(mut s) = world.get_resource_mut::<SettingsRes>()
            && s.0.environment != environment
        {
            s.0.environment = environment;
        }
        if let Some(rig) = rig
            && let Some(mut q) = world.try_query_filtered::<&mut Transform, With<PlayerRig>>()
        {
            for mut t in q.iter_mut(world) {
                *t = transform_of(rig);
            }
        }
        if let Some(cam) = &camera
            && let Some(mut q) =
                world.try_query_filtered::<(&mut Transform, &mut Projection), With<DesktopCamera>>()
        {
            for (mut t, mut p) in q.iter_mut(world) {
                *t = transform_of(cam.pose);
                if let Projection::Perspective(PerspectiveProjection { fov, .. }) = &mut *p {
                    *fov = cam.fov_deg.to_radians();
                }
            }
        }
        if let Some(mut ctl) = world.get_resource_mut::<CameraController>() {
            if let Some(cam) = &camera {
                ctl.orbit = cam.orbit;
            }
            ctl.bookmarks = Default::default();
            for (slot, b) in ctl.bookmarks.iter_mut().zip(bookmarks) {
                *slot = Some(b);
            }
        }
        let mut sp = world.get_resource_or_init::<ScenePending>();
        sp.models = pending;
        sp.located = located;
        sp.clock = Some(clock);
        sp.settled = false;
    });
    missing
}

/// A scene's attributes on each instance as it appears, matched by identity; other new
/// instances take `[display]`'s mode and overlays.
fn apply_pending_instances(
    mut pending: ResMut<ScenePending>,
    mut failures: MessageReader<LibraryFailure>,
    settings: Res<SettingsRes>,
    mut fresh: Query<(Entity, &mut LoadedModel, &mut Transform), Added<LoadedModel>>,
    mut commands: Commands,
) {
    // A scene model whose load failed waits no more (Q-203, Q-173).
    for f in failures.read() {
        if let Some(r) = &f.reference
            && let Some(at) = pending
                .models
                .iter()
                .position(|p| model_ref(&p.source).as_ref() == Some(r))
        {
            pending.models.remove(at);
        }
    }
    let display = &settings.0.display;
    for (e, mut m, mut t) in &mut fresh {
        let at = pending.models.iter().position(|p| p.identity == m.identity);
        // Q-T275.3: a located model takes its scene attributes too.
        let p = match at {
            Some(at) => Some(pending.models.remove(at)),
            None => pending
                .located
                .iter()
                .position(|p| p.identity == m.identity)
                .map(|at| pending.located.remove(at)),
        };
        let Some(p) = p else {
            let mode = mode_of(&display.mode, display);
            if mode != RenderMode::Splats {
                commands.entity(e).try_insert(SplatMode(mode));
            }
            commands
                .entity(e)
                .try_insert(SplatOverlays(overlays_of(&display.overlays)));
            continue;
        };
        *t = transform_of(p.pose);
        m.name = p.name.clone();
        m.time.offset_s = p.time_offset_s;
        m.time.end_rule = p.end_rule;
        if let Some(a) = &p.audio {
            m.audio = model_ref(&SourceSpec {
                kind: if Url::parse(a).is_ok() { "http" } else { "dir" }.into(),
                location: a.clone(),
                name: String::new(),
                flavor: None,
            });
        }
        let mut ec = commands.entity(e);
        ec.try_insert(SplatMode(mode_of(&p.mode, display)));
        ec.try_insert(SplatOverlays(overlays_of(&display.overlays)));
        if !p.visible {
            ec.try_insert((ShowOnUpload(false), Visibility::Hidden));
        }
    }
}

/// The scene's T and range a frame after its instances have appeared (T_len is theirs).
fn apply_pending_clock(
    mut pending: ResMut<ScenePending>,
    mut playback: ResMut<Playback>,
    mut range: ResMut<RangeSetting>,
) {
    if !pending.models.is_empty() || pending.clock.is_none() {
        return;
    }
    if !pending.settled {
        pending.settled = true;
        return;
    }
    let Some(clock) = pending.clock.take() else {
        return;
    };
    playback.clock.set_range(clock.range);
    playback.clock.seek(clock.t);
    range.0 = clock.range;
}

fn audio_settings(a: &AudioState) -> AudioSettings {
    // Q-188: gains in [0, 1], percent in the file.
    AudioSettings {
        muted: a.muted,
        master: a.master * 100.0,
        ui: a.ui * 100.0,
        soundtrack: a.soundtrack * 100.0,
        spatial: a.spatial,
        sounds: a.sounds.clone(),
    }
}

fn audio_state(s: &AudioSettings) -> AudioState {
    AudioState {
        muted: s.muted,
        master: (s.master / 100.0).clamp(0.0, 1.0),
        ui: (s.ui / 100.0).clamp(0.0, 1.0),
        soundtrack: (s.soundtrack / 100.0).clamp(0.0, 1.0),
        spatial: s.spatial,
        sounds: s.sounds.clone(),
    }
}

fn desktop_view(s: &Settings) -> DesktopView {
    DesktopView {
        fly_speed_mps: s.desktop.fly_speed_mps,
        fov_deg: s.desktop.fov_deg,
        grid: s.display.grid,
        background: s.environment.background,
    }
}

/// D-17: stick locomotion off in passthrough mode (Q-195).
fn locomotion_of(s: &Settings) -> Locomotion {
    Locomotion::from_settings(&s.locomotion, s.environment.mode != "passthrough")
}

/// The fields of `view` that differ from `base`'s view, written into `s`.
fn view_into(view: &DesktopView, base: &DesktopView, s: &mut Settings) {
    if view.fly_speed_mps != base.fly_speed_mps {
        s.desktop.fly_speed_mps = view.fly_speed_mps;
    }
    if view.fov_deg != base.fov_deg {
        s.desktop.fov_deg = view.fov_deg;
    }
    if view.grid != base.grid {
        s.display.grid = view.grid;
    }
    if view.background != base.background {
        s.environment.background = view.background;
    }
}

fn locomotion_settings(l: &Locomotion, base: &LocomotionSettings) -> LocomotionSettings {
    LocomotionSettings {
        speed_mps: l.speed_mps,
        snap_turn: l.snap,
        snap_deg: l.snap_deg,
        smooth_dps: l.smooth_dps,
        vignette: l.vignette,
        teleport: l.teleport,
        height_offset_m: l.height_offset_m,
        ..base.clone()
    }
}

/// The library's sources other than the platform's models directory (the first, Q-49).
fn listed_sources(library: &Library, models: &Path) -> Vec<SourceSpec> {
    let models = models.to_string_lossy();
    library
        .sources
        .iter()
        .map(|(s, _)| s)
        .filter(|s| !(s.kind == "dir" && s.location == models))
        .cloned()
        .collect()
}

/// The two-way mapping (PRD #275, "Settings as the source of truth"): a resource the user
/// changed is copied into `SettingsRes` first, then a changed `SettingsRes` table is
/// mapped into its resource. Every write compares first, and `last` holds the tables as
/// last mapped, so neither direction undoes the other.
#[allow(clippy::too_many_arguments)]
// Q-207
fn sync_settings(
    mut settings: ResMut<SettingsRes>,
    mut last: Local<Option<Settings>>,
    mut outstanding: Local<u8>,
    mut seen: Local<Vec<SourceSpec>>,
    files: Res<Files>,
    mut intents: MessageReader<Intent>,
    mut audio: ResMut<AudioState>,
    mut loco: ResMut<Locomotion>,
    mut view: ResMut<DesktopView>,
    mut playback: ResMut<Playback>,
    mut looping: ResMut<LoopSetting>,
    library: Res<Library>,
    mut library_out: MessageWriter<LibraryCommand>,
    mut instances: Query<(Entity, Option<&mut SplatMode>), With<LoadedModel>>,
    mut commands: Commands,
) {
    let first = last.is_none();
    if first {
        *seen = listed_sources(&library, &files.models);
    }
    let snap = last.get_or_insert_with(|| settings.0.clone());
    // The world into the settings.
    if !first {
        if audio.is_changed() {
            let t = audio_settings(&audio);
            if settings.0.audio != t {
                settings.0.audio = t.clone();
            }
            snap.audio = t;
        }
        if loco.is_changed() {
            let t = locomotion_settings(&loco, &settings.0.locomotion);
            if settings.0.locomotion != t {
                settings.0.locomotion = t.clone();
            }
            snap.locomotion = t;
        }
        if view.is_changed() {
            let base = desktop_view(snap);
            if *view != base {
                let mut t = settings.0.clone();
                view_into(&view, &base, &mut t);
                view_into(&view, &base, snap);
                if settings.0 != t {
                    settings.0 = t;
                }
            }
        }
        for intent in intents.read() {
            if let Intent::Playback(PlaybackCommand::Speed(s)) = *intent
                && s != 0.0
                && s.abs() <= 4.0
                && settings.0.playback.speed != s
            {
                settings.0.playback.speed = s;
                snap.playback.speed = s;
            }
        }
        if looping.is_changed() && settings.0.playback.loop_mode != looping.0 {
            settings.0.playback.loop_mode = looping.0;
            snap.playback.loop_mode = looping.0;
        }
        if *outstanding > 0 {
            *outstanding -= 1;
            if *outstanding == 0 {
                *seen = listed_sources(&library, &files.models);
            }
        } else if library.is_changed() && listed_sources(&library, &files.models) != *seen {
            // Only a change of the source list itself, not of its entries: a settings
            // source the library could not open stays in the settings.
            let actual = listed_sources(&library, &files.models);
            *seen = actual.clone();
            if actual != snap.sources {
                if settings.0.sources != actual {
                    settings.0.sources = actual.clone();
                }
                snap.sources = actual;
            }
        }
    } else {
        intents.clear();
    }
    if !(first || settings.is_changed()) {
        return;
    }
    // The settings into the world.
    let s = &settings.0;
    if first || s.audio != snap.audio {
        let a = audio_state(&s.audio);
        if *audio != a {
            *audio = a;
        }
    }
    if first || s.locomotion != snap.locomotion || s.environment.mode != snap.environment.mode {
        let l = locomotion_of(s);
        if *loco != l {
            *loco = l;
        }
    }
    {
        let v = desktop_view(s);
        if (first || v != desktop_view(snap)) && *view != v {
            *view = v;
        }
    }
    if first || s.playback != snap.playback {
        if playback.clock.speed() != s.playback.speed {
            let _ = playback.clock.set_speed(s.playback.speed);
        }
        if looping.0 != s.playback.loop_mode {
            looping.0 = s.playback.loop_mode;
            playback.clock.set_loop(s.playback.loop_mode);
        } else if first {
            playback.clock.set_loop(s.playback.loop_mode);
        }
    }
    if !first && s.display != snap.display {
        let mode = mode_of(&s.display.mode, &s.display);
        let overlays = overlays_of(&s.display.overlays);
        // Each instance keeps its mode (§1.5 Instances); its centers take the radius
        // and colormap; `[display] mode` is new instances' (Q-207).
        let _ = mode;
        let (radius, colormap) = (s.display.centers_radius, colormap_of(&s.display.colormap));
        for (e, m) in &mut instances {
            if let Some(mut m) = m {
                let next = match m.0 {
                    RenderMode::Centers { .. } => RenderMode::Centers {
                        pixel_radius: radius,
                        colormap,
                    },
                    RenderMode::Both { .. } => RenderMode::Both {
                        pixel_radius: radius,
                        colormap,
                    },
                    other => other,
                };
                if m.0 != next {
                    m.0 = next;
                }
            }
            commands.entity(e).try_insert(SplatOverlays(overlays));
        }
    }
    if first || s.sources != snap.sources {
        let models = files.models.to_string_lossy().into_owned();
        let is_models = |x: &SourceSpec| x.kind == "dir" && x.location == models;
        let mut issued = false;
        for (i, (spec, _)) in library.sources.iter().enumerate().rev() {
            if !is_models(spec) && !s.sources.contains(spec) {
                library_out.write(LibraryCommand::RemoveSource(i));
                issued = true;
            }
        }
        for spec in &s.sources {
            if !is_models(spec) && !library.sources.iter().any(|(x, _)| x == spec) {
                library_out.write(LibraryCommand::AddSource(spec.clone()));
                issued = true;
            }
        }
        if issued {
            *outstanding = 2;
        }
    }
    *snap = settings.0.clone();
}

/// `[general] ui_scale` on the screen's egui context (§1.7 DPI-aware UI, §1.11): egui's
/// zoom factor, over the window's own scale factor.
fn ui_scale_follows_settings(
    settings: Res<SettingsRes>,
    mut contexts: Query<&mut bevy_egui::EguiContext, With<bevy_egui::PrimaryEguiContext>>,
) {
    let scale = settings.0.general.ui_scale;
    if !(scale.is_finite() && scale > 0.0) {
        return;
    }
    for mut c in &mut contexts {
        let ctx = c.get_mut();
        if ctx.zoom_factor() != scale {
            ctx.set_zoom_factor(scale);
        }
    }
}

/// An instance switched to centers (the Scene tab, a key) takes `[display]`'s radius and
/// colormap, as `sync_settings` gives them on a change of the settings (Q-218).
// Q-218
fn centers_follow_settings(
    settings: Res<SettingsRes>,
    mut modes: Query<&mut SplatMode, Changed<SplatMode>>,
) {
    let d = &settings.0.display;
    let (pixel_radius, colormap) = (d.centers_radius, colormap_of(&d.colormap));
    for mut m in &mut modes {
        let next = match m.0 {
            RenderMode::Centers { .. } => RenderMode::Centers {
                pixel_radius,
                colormap,
            },
            RenderMode::Both { .. } => RenderMode::Both {
                pixel_radius,
                colormap,
            },
            other => other,
        };
        if m.0 != next {
            m.0 = next;
        }
    }
}

/// §1.13: a failed save is logged and raised as an E-IO (or its own code's) toast.
fn save_to(world: &mut World) {
    let (Some(path), Some(s)) = (
        world.get_resource::<SettingsPath>(),
        world.get_resource::<SettingsRes>(),
    ) else {
        return;
    };
    if let Err(e) = save_settings(&path.0, &s.0) {
        warn!("{}: {}: {e}", path.0.display(), e.code());
        world.write_message(PersistFailure::of(&e));
    }
}

fn save_now(world: &mut World) {
    save_to(world);
    let now = world.resource::<Time<Real>>().elapsed();
    let mut saver = world.resource_mut::<Saver>();
    saver.dirty = false;
    saver.last = Some(now);
}

/// §1.11: saved at most once a second after a change; a change is written within 1 s.
// Q-207
fn save_settings_debounced(
    settings: Res<SettingsRes>,
    path: Res<SettingsPath>,
    time: Res<Time<Real>>,
    mut saver: ResMut<Saver>,
    mut failures: MessageWriter<PersistFailure>,
) {
    if settings.is_changed() && !settings.is_added() {
        saver.dirty = true;
    }
    let now = time.elapsed();
    let due = saver
        .last
        .is_none_or(|l| now.saturating_sub(l) >= Duration::from_secs(1));
    if saver.dirty && due {
        if let Err(e) = save_settings(&path.0, &settings.0) {
            warn!("{}: {}: {e}", path.0.display(), e.code());
            failures.write(PersistFailure::of(&e));
        }
        saver.dirty = false;
        saver.last = Some(now);
    }
}

/// `poses.json` follows `Poses` (§1.5 Poses, §1.11; written atomically by
/// `PoseStore::save`).
// Q-205
fn save_poses(
    poses: Res<Poses>,
    files: Res<Files>,
    mut last: ResMut<PosesSaved>,
    mut out: MessageWriter<PersistFailure>,
) {
    if !poses.is_changed() || last.0 == poses.0 {
        return;
    }
    match poses.0.save(&files.poses) {
        Ok(()) => last.0 = poses.0.clone(),
        Err(e) => {
            warn!("{}: {}: {e}", files.poses.display(), e.code());
            out.write(PersistFailure::of(&e));
        }
    }
}

/// The settings at `AppExit` (§1.11).
fn save_at_exit(world: &mut World, mut cursor: Local<MessageCursor<AppExit>>) {
    let Some(exits) = world.get_resource::<Messages<AppExit>>() else {
        return;
    };
    if cursor.read(exits).next().is_some() {
        save_now(world);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locations_under_models_are_relative_and_resolve_back() {
        let models = Path::new("/data/openftgs-xr/models");
        let inside = Path::new("/data/openftgs-xr/models/./a/../b/m.safetensors");
        let rel = relative_location(inside, models);
        assert_eq!(rel, "b/m.safetensors");
        assert_eq!(
            resolve_location(&rel, models),
            "/data/openftgs-xr/models/b/m.safetensors"
        );
        let outside = Path::new("/elsewhere/m.ply");
        assert_eq!(relative_location(outside, models), "/elsewhere/m.ply");
        assert_eq!(
            resolve_location("/elsewhere/m.ply", models),
            "/elsewhere/m.ply"
        );
        // Lexical: `models/../x` is outside.
        let up = Path::new("/data/openftgs-xr/models/../x.ply");
        assert_eq!(relative_location(up, models), up.to_string_lossy());
        let url = "http://example.test/m.safetensors";
        assert_eq!(resolve_location(url, models), url);
    }

    #[test]
    fn scene_paths_go_under_scenes_with_the_suffix() {
        let scenes = Path::new("/c/scenes");
        assert_eq!(
            scene_path(scenes, Path::new("demo")),
            Path::new("/c/scenes/demo.ftgsxr-scene.json")
        );
        assert_eq!(
            scene_path(scenes, Path::new("/x/y.ftgsxr-scene.json")),
            Path::new("/x/y.ftgsxr-scene.json")
        );
        let name = default_scene_name();
        assert_eq!(name.len(), "scene-20261102-120000".len(), "{name}");
    }

    #[test]
    fn recent_scenes_are_the_newest_eight() {
        let dir = std::env::temp_dir().join(format!("ftgsxr-recent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let base = std::time::SystemTime::now() - Duration::from_secs(100);
        for k in 0..10u64 {
            let p = dir.join(format!("s{k}.ftgsxr-scene.json"));
            std::fs::write(&p, "{}").unwrap();
            let f = std::fs::File::options().write(true).open(&p).unwrap();
            f.set_modified(base + Duration::from_secs(k)).unwrap();
        }
        std::fs::write(dir.join("other.json"), "{}").unwrap();
        let recent = recent_scenes(&dir);
        assert_eq!(recent.len(), 8);
        assert_eq!(recent[0], dir.join("s9.ftgsxr-scene.json"));
        assert_eq!(recent[7], dir.join("s2.ftgsxr-scene.json"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn audio_percent_maps_to_gains_and_back() {
        let s = AudioSettings {
            master: 50.0,
            ui: 25.0,
            ..AudioSettings::default()
        };
        let a = audio_state(&s);
        assert!((a.master - 0.5).abs() < 1e-6 && (a.ui - 0.25).abs() < 1e-6);
        assert_eq!(audio_settings(&a), s);
    }
}
