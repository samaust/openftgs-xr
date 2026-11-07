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
    AudioSettings, CameraState, ClockState, IoError, LocomotionSettings, ManualLighting, ModelRef,
    PoseStore, SceneFile, SceneModel, Settings, SourceSpec, Url, load_settings, read_scene,
    save_settings, write_scene,
};
use ftgsxr_render::{Colormap, Overlays, RenderMode, SplatMode, SplatOverlays};

use crate::audio::AudioState;
use crate::intent::{Feature, InstanceAction, Intent, OverlayToggle, PlaybackCommand, UiCommand};
use crate::library::{
    Library, LibraryCommand, LibraryFailure, LoadedModel, MissingModel, Replaces, SceneLoad,
    ShowOnUpload, list_missing,
};
use crate::manipulate::{
    HighContrast, ManipulateSettings, Poses, now_rfc3339, pose_of, transform_of,
};
use crate::platform::{AskPermissionAgain, DeniedPermissions, Headset, PlatformRes};
use crate::playback::{LoopSetting, Playback, RangeSetting};
use crate::rig::{CameraController, DesktopCamera, DesktopView, Locomotion, PlayerRig};
use crate::ui::{
    DockHooks, InstanceLighting, RecentScenes, SceneLoadConfirmed, Tab, ToastRequest, UiState,
    UiSurface, recent_scenes,
};

mod panels;

/// A quality tier the user picked in the dock (§1.7 Panels): `governor` switches the
/// running tier to it, even when it equals the saved `[general] tier`, so the saved tier
/// can be picked back after a `--tier` override (Q-240). Only a pick switches the tier.
// Q-240
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub(crate) struct TierPick(pub(crate) String);

/// A pick of `tier` (Q-240): when `QualityTier::by_name` knows it, writes `[general] tier`
/// to `SettingsRes` if it differs, so the debounce saves it (Q-207); a `TierPick` for
/// `governor` either way (it warns of a name it does not know).
// Q-240, Q-207
pub(crate) fn pick_tier(world: &mut World, tier: String) {
    if ftgsxr_render::QualityTier::by_name(&tier).is_some()
        && let Some(mut s) = world.get_resource_mut::<SettingsRes>()
        && s.0.general.tier != tier
    {
        s.0.general.tier = tier.clone();
    }
    if let Some(mut picks) = world.get_resource_mut::<Messages<TierPick>>() {
        picks.write(TierPick(tier));
    }
}

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
// Q-224
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct SettingsWarnings(pub Vec<String>);

/// The scene anchor's handshake with the XR crate (§1.9 Floor, shadows and anchors,
/// Q-343): `persist` sets `Creating` at a scene save under the `Headset` marker in
/// passthrough mode and writes the file once the XR crate answers `Ready(uuid)`, `None` at
/// any other save (written with a null anchor) unless a save waits, and sets
/// `Locating(uuid)` at the load of a file with an anchor on the headset; the XR crate
/// answers `Ready` once located, or `Lost(uuid)`, which `persist` toasts.
/// `scene_from_world` writes `anchor` from `Ready`. While a save waits, any other answer
/// than `Ready` is a failure (Q-350).
// Q-343, Q-350
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub enum SceneAnchor {
    /// No anchor.
    #[default]
    None,
    /// A save waits for the anchor's creation.
    Creating,
    /// The anchor, created and persisted, or located: its UUID.
    Ready(String),
    /// A loaded scene's anchor, being located.
    Locating(String),
    /// A loaded scene's anchor that the runtime did not find.
    Lost(String),
}

/// How long a save waits for its anchor (Q-343).
// Q-343
const ANCHOR_WAIT: Duration = Duration::from_secs(2);

/// The scene save waiting for its anchor: the file and when the wait began.
// Q-343
#[derive(Resource, Default)]
struct AnchorWait(Option<(PathBuf, Duration)>);

/// The toast of a save written without its anchor (Q-350).
// Q-350
const ANCHOR_NOT_CREATED: &str = "anchor not created; scene saved without one";

/// §1.9's toast when a loaded scene's anchor cannot be located.
const ANCHOR_LOST: &str = "anchor not found; using the saved pose";

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

/// Whether no settings file existed at startup: Help opens once (§1.7, Q-226).
#[derive(Resource, Default)]
struct FirstRun(bool);

/// §1.7 Help: "opened once on the first run".
// Q-226
fn open_help_on_first_run(first: Res<FirstRun>, mut ui: ResMut<UiState>) {
    if first.0 {
        ui.dock_open = true;
        ui.tab = Tab::Help;
    }
}

/// Whether the UI is on the window, where the dock is §1.7's: `[desktop]`'s dock keys
/// are the desktop's, not a world panel's (§1.8).
fn dock_on_screen(ui: Option<Res<UiState>>, surface: Option<Res<UiSurface>>) -> bool {
    ui.is_some() && surface.is_none_or(|s| *s == UiSurface::Screen)
}

/// §1.7 Panels (spec revision 24): the dock opens at `[desktop]`'s width and collapsed
/// state, before its first draw; Help's first run opens it after (Q-226).
fn dock_from_settings(settings: Res<SettingsRes>, mut ui: ResMut<UiState>) {
    let d = &settings.0.desktop;
    if ui.dock_width != d.dock_width || ui.dock_open != d.dock_open {
        ui.dock_width = d.dock_width;
        ui.dock_open = d.dock_open;
    }
}

/// §1.7 Panels, §1.11: the dock's width, in whole points, and whether it is open go into
/// `[desktop]`; the width is the one the user dragged the dock to, a window too narrow
/// for it clamping only the drawing (Q-252). The dragged and the saved width are
/// compared in whole points, so a drag to a fractional width (egui rounds to 1/32 point)
/// does not mark the settings changed every frame (Q-207).
// Q-252
fn dock_into_settings(ui: Res<UiState>, mut settings: ResMut<SettingsRes>) {
    if !ui.is_changed() {
        return;
    }
    let width = ui.dock_width.round();
    let saved = &settings.0.desktop;
    let resized = width.is_finite() && width != saved.dock_width.round();
    if resized || ui.dock_open != saved.dock_open {
        let desktop = &mut settings.0.desktop;
        if resized {
            desktop.dock_width = width;
        }
        desktop.dock_open = ui.dock_open;
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
    /// once located (Q-225); the clock does not wait for them.
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
    // Q-226: the first run is the one without a settings file.
    world.insert_resource(FirstRun(!path.exists()));
    if world.contains_resource::<SettingsRes>() {
        // Q-224: the caller's own load.
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
    // Q-335: the Quest's `main` and the tests insert the marker before `AppPlugin`.
    let headset = world.contains_resource::<Headset>();
    let (view, loco) = {
        let s = &world.resource::<SettingsRes>().0;
        (desktop_view(s, headset), locomotion_of(s, headset))
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
            hooks.add(Tab::Comfort, panels::comfort_section);
        }
        app.add_message::<PersistCommand>()
            .add_message::<TierPick>()
            .add_message::<PersistFailure>()
            .add_message::<SceneLoadConfirmed>()
            .add_message::<ToastRequest>()
            .add_message::<AskPermissionAgain>()
            .init_resource::<DeniedPermissions>()
            .add_message::<Intent>()
            .add_message::<LibraryCommand>()
            .init_resource::<Saver>()
            .init_resource::<ScenePending>()
            .init_resource::<SceneAnchor>()
            .init_resource::<AnchorWait>()
            .init_resource::<StartupFailures>()
            .init_resource::<FirstRun>()
            .add_systems(
                Update,
                (
                    raise_startup,
                    ui_bridge,
                    run_intents,
                    run_commands,
                    finish_anchor_wait,
                    anchor_toasts,
                    sync_settings,
                    dock_into_settings.run_if(dock_on_screen),
                    centers_follow_settings,
                    high_contrast_follows_settings,
                    keep_upright_sync,
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
            // Before egui's pass begins, so the zoom applies from the first drawn frame
            // and the dock's range is taken in the zoomed points (§1.7 Panels, Q-252).
            .add_systems(
                bevy::app::PreUpdate,
                ui_scale_follows_settings
                    .after(bevy_egui::EguiPreUpdateSet::InitContexts)
                    .before(bevy_egui::EguiPreUpdateSet::BeginPass),
            )
            .add_systems(
                bevy::app::Startup,
                (
                    dock_from_settings.run_if(dock_on_screen),
                    open_help_on_first_run.run_if(resource_exists::<UiState>),
                )
                    .chain(),
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

/// `[display] high_contrast` into `manipulate`'s highlight (Q-278, Q-312).
// Q-278, Q-312
fn high_contrast_follows_settings(settings: Res<SettingsRes>, mut hc: ResMut<HighContrast>) {
    let want = HighContrast(settings.0.display.high_contrast);
    if *hc != want {
        *hc = want;
    }
}

/// `[xr] keep_upright` and `ManipulateSettings.keep_upright` both ways (Q-330, Q-207):
/// the settings' value goes into `manipulate` at the first frame and whenever it changes,
/// and a value changed in the world goes back into the settings, which then save it.
/// `last` is the value as last mapped, so neither direction undoes the other. A system of
/// its own: `sync_settings` already takes 15 of a Bevy system's 16 parameters.
// Q-330
fn keep_upright_sync(
    mut settings: ResMut<SettingsRes>,
    mut manipulate: ResMut<ManipulateSettings>,
    mut last: Local<Option<bool>>,
) {
    let wanted = settings.0.xr.keep_upright;
    match *last {
        Some(l) if wanted == l => {
            if manipulate.keep_upright != l {
                settings.0.xr.keep_upright = manipulate.keep_upright;
            }
        }
        _ => {
            if manipulate.keep_upright != wanted {
                manipulate.keep_upright = wanted;
            }
        }
    }
    *last = Some(settings.0.xr.keep_upright);
}

/// `Intent::Ui(SaveScene)` saves a scene named by the time (Q-200); `LoadScene` is the
/// UI's, which confirms and sends `PersistCommand::LoadScene`. The overlay toggles (Appendix D.1 B, V; the Display tab): an overlay
/// is a `[display.overlays]` setting, flipped here and mapped onto every instance by
/// `sync_settings` (Q-207). `Toggle(Feature::Occlusion)` flips `[environment] occlusion`
/// (Q-336); `Toggle(Feature::Passthrough)` is the XR crate's (Q-335).
// Q-336
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
            Intent::Toggle(Feature::Occlusion) => {
                let e = &mut settings.0.environment;
                e.occlusion = !e.occlusion;
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
                // Q-343, Q-335: on the headset in passthrough mode the anchor first, the
                // mode as gated (`passthrough_mode`).
                let passthrough = passthrough_mode(
                    &world.resource::<SettingsRes>().0,
                    world.contains_resource::<Headset>(),
                );
                if passthrough {
                    let now = world.resource::<Time<Real>>().elapsed();
                    world.insert_resource(SceneAnchor::Creating);
                    world.resource_mut::<AnchorWait>().0 = Some((path, now));
                } else {
                    // §1.9: an anchor only for a save in passthrough mode on the headset,
                    // so this one writes null and the earlier anchor is let go (a save
                    // still waiting in `Creating` keeps its wait).
                    if *world.resource::<SceneAnchor>() != SceneAnchor::Creating {
                        world.insert_resource(SceneAnchor::None);
                    }
                    write_scene_file(world, &path);
                }
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
                // Q-343: the headset locates the file's anchor; elsewhere it is ignored.
                if world.contains_resource::<Headset>() {
                    let anchor = match &scene.anchor {
                        Some(uuid) => SceneAnchor::Locating(uuid.clone()),
                        None => SceneAnchor::None,
                    };
                    world.insert_resource(anchor);
                }
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

/// Writes the world's scene to `path`, a failure toasted, and refreshes the recent scenes.
fn write_scene_file(world: &mut World, path: &Path) {
    let scene = scene_from_world(world);
    if let Err(e) = write_scene(path, &scene) {
        warn!("{}: {}: {e}", path.display(), e.code());
        world.write_message(PersistFailure::of(&e));
    }
    let scenes = world.resource::<Files>().scenes.clone();
    world.insert_resource(RecentScenes(recent_scenes(&scenes)));
}

/// The waiting save's end (Q-343): written with its anchor at `Ready`; without one, and
/// toasted, at any other answer or after `ANCHOR_WAIT` (the state then `None`,
/// Q-350).
// Q-343, Q-350
fn finish_anchor_wait(world: &mut World) {
    let Some((path, since)) = world.resource::<AnchorWait>().0.clone() else {
        return;
    };
    let now = world.resource::<Time<Real>>().elapsed();
    let ok = match world.resource::<SceneAnchor>() {
        SceneAnchor::Ready(_) => true,
        SceneAnchor::Creating if now.saturating_sub(since) < ANCHOR_WAIT => return,
        SceneAnchor::Creating => {
            world.insert_resource(SceneAnchor::None);
            false
        }
        _ => false,
    };
    world.resource_mut::<AnchorWait>().0 = None;
    if !ok {
        warn!("{}: {ANCHOR_NOT_CREATED}", path.display());
        world.write_message(ToastRequest::warning(ANCHOR_NOT_CREATED));
    }
    write_scene_file(world, &path);
}

/// §1.9's toast when the XR crate reports a loaded scene's anchor lost (Q-343).
// Q-343
fn anchor_toasts(anchor: Res<SceneAnchor>, mut toasts: MessageWriter<ToastRequest>) {
    if anchor.is_changed()
        && let SceneAnchor::Lost(uuid) = &*anchor
    {
        warn!("anchor {uuid}: {ANCHOR_LOST}");
        toasts.write(ToastRequest::warning(ANCHOR_LOST));
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
/// follow Q-3 against `AppPaths.models`. Each instance's `lighting` is its Scene-tab
/// manual values, null at the identity; `anchor` is `SceneAnchor::Ready`'s UUID, null
/// otherwise (Q-343).
// Q-206, Q-3, Q-343
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
        Option<&InstanceLighting>,
    )>() {
        let mut rows: Vec<_> = q.iter(world).collect();
        rows.sort_by_key(|r| r.0);
        for (_, m, t, vis, mode, show, light) in rows {
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
                lighting: light
                    .filter(|l| l.0 != ManualLighting::default())
                    .map(|l| l.0.clone()),
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
    let anchor = match world.get_resource::<SceneAnchor>() {
        Some(SceneAnchor::Ready(uuid)) => Some(uuid.clone()),
        _ => None,
    };
    SceneFile {
        models,
        clock,
        environment,
        anchor,
        rig,
        camera,
        bookmarks,
    }
}

/// Replaces the current scene with `scene` (§1.5 Scene files): each model whose source
/// still offers it loads through the library (`LibraryCommand::LoadRef`), its pose,
/// visibility, mode, offset, end rule, lighting and audio applied when the instance appears; a
/// local file that no longer exists is listed in `library` as `EntryState::Missing` and
/// its name returned (Q-54, E-SCENE-MISSING); a file read with another identity than the
/// scene's is refused and its model listed as missing then (Q-245). The clock,
/// `SettingsRes.environment`, the rig, the desktop camera and the bookmarks are set
/// through `commands`; T and the range once the scene's instances have appeared (T_len
/// is 0 before). The caller unloads the current instances and resolves relative
/// locations first (`LoadScene`, Q-3).
// Q-203, Q-245
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
        let model = MissingModel {
            kind: m.source.kind.clone(),
            location: m.source.location.clone(),
            name: m.name.clone(),
            audio: m.audio.clone(),
        };
        if let ModelRef::File(p) = &r
            && !p.exists()
        {
            missing.push(m.name.clone());
            located.push(m.clone());
            list_missing(library, &model, &m.identity, true);
            continue;
        }
        // Q-245: the file read must have the scene's identity, else it is missing too.
        commands.write_message(SceneLoad {
            reference: r,
            identity: m.identity.clone(),
            model,
        });
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
    mut fresh: Query<
        (Entity, &mut LoadedModel, &mut Transform, Option<&Replaces>),
        Added<LoadedModel>,
    >,
    mut commands: Commands,
) {
    // A scene model whose load failed waits no more (Q-203, Q-173); one whose file was
    // another model keeps its attributes for a later load of its identity (Q-245).
    for f in failures.read() {
        if let Some(r) = &f.reference
            && let Some(at) = pending
                .models
                .iter()
                .position(|p| model_ref(&p.source).as_ref() == Some(r))
        {
            let m = pending.models.remove(at);
            if f.code == "E-SCENE-MISSING" {
                pending.located.push(m);
            }
        }
    }
    let display = &settings.0.display;
    for (e, mut m, mut t, replaces) in &mut fresh {
        // Q-255: a replacement takes the attributes of the scene model it replaces.
        let at = match replaces {
            Some(_) => None,
            None => pending.models.iter().position(|p| p.identity == m.identity),
        };
        let key = replaces.map_or(&m.identity, |r| &r.0);
        // Q-225: a located model takes its scene attributes too.
        let p = match at {
            Some(at) => Some(pending.models.remove(at)),
            None => pending
                .located
                .iter()
                .position(|p| &p.identity == key)
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
        // Q-206: a null `lighting` leaves the identity.
        if let Some(l) = &p.lighting {
            ec.try_insert(InstanceLighting::bundle(l.clone()));
        }
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

fn desktop_view(s: &Settings, headset: bool) -> DesktopView {
    DesktopView {
        fly_speed_mps: s.desktop.fly_speed_mps,
        fov_deg: s.desktop.fov_deg,
        grid: s.display.grid,
        background: s.environment.background,
        passthrough: passthrough_mode(s, headset),
    }
}

/// Whether the player is in passthrough mode: `[environment] mode` is the wanted mode,
/// and counts only under the `Headset` marker, so the desktop and `play --xr` stay in
/// VR mode whatever the setting says (§1.4 Backgrounds, Q-335).
// Q-335
fn passthrough_mode(s: &Settings, headset: bool) -> bool {
    headset && s.environment.mode == "passthrough"
}

/// D-17: stick locomotion off in passthrough mode (Q-195), as gated (Q-335).
// Q-195, Q-335
fn locomotion_of(s: &Settings, headset: bool) -> Locomotion {
    Locomotion::from_settings(&s.locomotion, !passthrough_mode(s, headset))
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
    headset: Option<Res<Headset>>,
) {
    let first = last.is_none();
    // Q-335: the gate, and a marker inserted after the first frame maps the mode again.
    let gate_added = headset.as_ref().is_some_and(|h| h.is_added());
    let headset = headset.is_some();
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
            let base = desktop_view(snap, headset);
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
    if !(first || gate_added || settings.is_changed()) {
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
    if first
        || gate_added
        || s.locomotion != snap.locomotion
        || s.environment.mode != snap.environment.mode
    {
        let l = locomotion_of(s, headset);
        if *loco != l {
            *loco = l;
        }
    }
    {
        let v = desktop_view(s, headset);
        if (first || gate_added || v != desktop_view(snap, headset)) && *view != v {
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
/// zoom factor, over the window's own scale factor; on the panels' contexts (§1.8 User
/// interface) the panel zoom times the scale (Q-267).
// Q-267
#[allow(clippy::type_complexity)]
fn ui_scale_follows_settings(
    settings: Res<SettingsRes>,
    mut contexts: Query<
        (&mut bevy_egui::EguiContext, Has<crate::ui::PanelCamera>),
        Or<(
            With<bevy_egui::PrimaryEguiContext>,
            With<crate::ui::PanelCamera>,
        )>,
    >,
) {
    let scale = settings.0.general.ui_scale;
    if !(scale.is_finite() && scale > 0.0) {
        return;
    }
    for (mut c, panel) in &mut contexts {
        let zoom = if panel {
            crate::ui::panel_zoom(scale)
        } else {
            scale
        };
        let ctx = c.get_mut();
        if ctx.zoom_factor() != zoom {
            ctx.set_zoom_factor(zoom);
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

/// The settings at `AppExit` (§1.11); after `exit_wait`'s hold (#376, Q-236).
pub(crate) fn save_at_exit(world: &mut World, mut cursor: Local<MessageCursor<AppExit>>) {
    let Some(exits) = world.get_resource::<Messages<AppExit>>() else {
        return;
    };
    if cursor.read(exits).next().is_some() {
        save_now(world);
    }
}

#[cfg(test)]
mod tests {

    /// Q-278, Q-312: `[display] high_contrast` reaches `manipulate`'s highlight, and
    /// back off.
    #[test]
    fn high_contrast_follows_the_display_setting() {
        let mut app = App::new();
        app.insert_resource(SettingsRes(Settings::default()))
            .init_resource::<HighContrast>()
            .add_systems(Update, high_contrast_follows_settings);
        app.update();
        assert_eq!(*app.world().resource::<HighContrast>(), HighContrast(false));
        app.world_mut()
            .resource_mut::<SettingsRes>()
            .0
            .display
            .high_contrast = true;
        app.update();
        assert_eq!(*app.world().resource::<HighContrast>(), HighContrast(true));
        app.world_mut()
            .resource_mut::<SettingsRes>()
            .0
            .display
            .high_contrast = false;
        app.update();
        assert_eq!(*app.world().resource::<HighContrast>(), HighContrast(false));
    }
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

    // Q-200 (#340): an absolute path without the suffix stays where it is and gains it.
    #[test]
    fn an_absolute_scene_path_without_the_suffix_gains_it() {
        assert_eq!(
            scene_path(Path::new("/c/scenes"), Path::new("/x/y")),
            Path::new("/x/y.ftgsxr-scene.json")
        );
    }

    // §1.5 Scene files, Q-200 (#417): with ten older scene files, a save refreshes ui's
    // recent list to the eight newest, the saved scene first.
    #[test]
    fn a_save_heads_the_recent_scenes_of_the_newest_eight() {
        let dir = std::env::temp_dir().join(format!("ftgsxr-recent-save-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let scenes = dir.join("scenes");
        std::fs::create_dir_all(&scenes).unwrap();
        let base = std::time::SystemTime::now() - Duration::from_secs(100);
        for k in 0..10u64 {
            let p = scenes.join(format!("s{k}.ftgsxr-scene.json"));
            std::fs::write(&p, "{}").unwrap();
            let f = std::fs::File::options().write(true).open(&p).unwrap();
            f.set_modified(base + Duration::from_secs(k)).unwrap();
        }
        let mut app = App::new();
        app.add_message::<PersistCommand>()
            .add_message::<PersistFailure>()
            .add_message::<Intent>()
            .add_message::<LibraryCommand>()
            .insert_resource(Files {
                poses: dir.join("poses.json"),
                scenes: scenes.clone(),
                models: dir.join("models"),
            })
            .init_resource::<RecentScenes>()
            .add_systems(Update, run_commands);
        app.world_mut()
            .write_message(PersistCommand::SaveScene("new".into()));
        app.update();
        let recent = app.world().resource::<RecentScenes>().0.clone();
        let names: Vec<String> = recent
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        let mut expected = vec!["new.ftgsxr-scene.json".to_owned()];
        expected.extend((3..10u64).rev().map(|k| format!("s{k}.ftgsxr-scene.json")));
        assert_eq!(names, expected);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // §1.7 Panels (spec revision 24, #428): the saved width and collapsed state are the
    // dock's at startup, before its first draw.
    #[test]
    fn dock_from_settings_gives_the_dock_the_saved_layout() {
        let mut settings = Settings::default();
        settings.desktop.dock_width = 412.5;
        settings.desktop.dock_open = false;
        let mut app = App::new();
        app.insert_resource(SettingsRes(settings))
            .init_resource::<UiState>()
            .add_systems(bevy::app::Startup, dock_from_settings);
        app.update();
        let ui = app.world().resource::<UiState>();
        assert_eq!((ui.dock_width, ui.dock_open), (412.5, false));
    }

    /// Counts the frames in which `SettingsRes` changed after it was added.
    #[derive(Resource, Default)]
    struct SettingsChanges(u32);

    fn count_settings_changes(s: Res<SettingsRes>, mut n: ResMut<SettingsChanges>) {
        if s.is_changed() && !s.is_added() {
            n.0 += 1;
        }
    }

    // §1.7 Panels, §1.11, I-10, Q-207, Q-252 (#428): a drag's width goes into `[desktop]` in
    // whole points, a change within the same whole point (a half point included) is no
    // change, a collapse is one, and a change is written at once (the debounce's first
    // save).
    #[test]
    fn dock_into_settings_saves_whole_points_and_the_collapse() {
        let dir = std::env::temp_dir().join(format!("ftgsxr-dock-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.toml");
        let mut app = App::new();
        app.add_message::<PersistFailure>()
            .insert_resource(SettingsRes(Settings::default()))
            .insert_resource(SettingsPath(path.clone()))
            .init_resource::<Time<Real>>()
            .init_resource::<Saver>()
            .init_resource::<UiState>()
            .init_resource::<SettingsChanges>()
            .add_systems(
                Update,
                (
                    dock_into_settings,
                    count_settings_changes,
                    save_settings_debounced,
                )
                    .chain(),
            );
        app.update();
        let changes = |app: &App| app.world().resource::<SettingsChanges>().0;
        let desktop = |app: &App| app.world().resource::<SettingsRes>().0.desktop.clone();
        // The drawn width a frame later, 0.3 points off: no change.
        app.world_mut().resource_mut::<UiState>().dock_width = 320.3;
        app.update();
        assert_eq!(changes(&app), 0);
        assert_eq!(desktop(&app).dock_width, 320.0);
        assert!(!path.exists(), "nothing to save");
        // A drag to 412.6 points: 413 in the settings, and on disk at once.
        app.world_mut().resource_mut::<UiState>().dock_width = 412.6;
        app.update();
        assert_eq!(changes(&app), 1);
        assert_eq!(desktop(&app).dock_width, 413.0);
        let saved = load_settings(&path).expect("saved").0;
        assert_eq!(saved.desktop.dock_width, 413.0);
        // The same width, drawn again: no change.
        app.update();
        assert_eq!(changes(&app), 1);
        // Exactly half a point off the saved 413, drawn frame after frame: no change.
        app.world_mut().resource_mut::<UiState>().dock_width = 412.5;
        app.update();
        app.update();
        assert_eq!(changes(&app), 1);
        assert_eq!(desktop(&app).dock_width, 413.0);
        // A collapse.
        app.world_mut().resource_mut::<UiState>().dock_open = false;
        app.update();
        assert_eq!(changes(&app), 2);
        assert!(!desktop(&app).dock_open);
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
