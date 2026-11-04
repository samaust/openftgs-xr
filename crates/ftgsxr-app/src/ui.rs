// SPDX-License-Identifier: Apache-2.0

//! `ftgsxr_app::ui` (§2.1): the panels on screen (§1.7 Panels, D-21) or on a world quad
//! (§1.8 User interface), the toasts (§1.13 Logs, §1.15), the HUD's data (§1.13 HUD) and the
//! ray-to-pointer mapping (§1.8, I-11, U-32).
//!
//! The module comes before `audio`, `persist`, `governor` and `desktop_input` in §2.1's
//! order, so it names none of their types (PRD #275 "Dependency direction"): the later
//! modules write ui's crate-private `ToastRequest` for their failures, map its `UiCue`
//! to their UI sounds, read its `SceneLoadConfirmed`, and add their own sections to the
//! dock's tabs through `DockHooks`.
//!
//! What runs without egui (the toasts and their rate limit, the confirmations, the UI
//! commands, `HudData`) is added in every app; the drawing and the panel pointer only
//! when `EguiPlugin` is (PRD #275's spike check 1 and its fallback).

mod cameras;
mod dock;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use bevy::app::{App, Plugin, PostUpdate, PreUpdate, Update};
use bevy::asset::Handle;
use bevy::camera::RenderTarget;
use bevy::ecs::prelude::*;
use bevy::image::Image;
use bevy::log::warn;
use bevy::math::{UVec2, Vec2, Vec3};
use bevy::time::Time;
use bevy::transform::components::GlobalTransform;
use bevy::window::{MonitorSelection, PrimaryWindow, Window, WindowMode};
use bevy_egui::input::EguiInputEvent;
use bevy_egui::{EguiPlugin, EguiPostUpdateSet, EguiPrimaryContextPass, egui};
use ftgsxr_core::Pose;
use ftgsxr_io::{DesktopSettings, ManualLighting};
use ftgsxr_render::{
    BudgetSource, ColorTransform, Foveation, GpuTimes, SortMode, SplatEvent, SplatGpu, SplatStats,
    SplatTierState, SplatTime, SplatTransform,
};

use crate::intent::{Feature, InstanceAction, Intent, UiCommand, UnloadRequest};
use crate::library::{EntryState, Library, LibraryCommand, LibraryFailure, LoadedModel};
use crate::manipulate::Selected;
use crate::platform::PickKind;
use crate::playback::{Playback, PlaybackWarning, Scrubbing};

/// Where the UI is drawn (§1.8): on the window (the desktop), or on a world quad whose
/// entity carries `WorldPanel` and the quad's pose, `size_px` its texture's size and
/// `size_m` its size in metres. The quad's egui context is the camera that renders to
/// `WorldPanel.image` (bevy_egui's render-to-image path, D-21).
// Q-210
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub enum UiSurface {
    /// The window.
    #[default]
    Screen,
    /// A world quad.
    Panel {
        /// The quad's entity (`WorldPanel`, its `GlobalTransform` the quad's pose).
        entity: Entity,
        /// The texture's size in pixels.
        size_px: UVec2,
        /// The quad's size in metres.
        size_m: Vec2,
    },
}

/// The main panel's quad (§1.8): the image its egui context renders to; placed in M-5.
#[derive(Component, Clone, Debug, PartialEq)]
pub struct WorldPanel {
    /// The texture the panel is drawn into.
    pub image: Handle<Image>,
}

/// The wrist menu's quad (§1.8); placed in M-5.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WristMenu;

/// The ray's hit on the panel quad as a pointer position in pixels; None when it misses
/// (§1.8). The quad is centred on `panel.translation`, spans `size_m` along the pose's x
/// (right) and y (up) axes and faces its +z; only its front is hit. Its top-left corner
/// is (0, 0) and its bottom-right (W, H) = `size_px`; `panel.scale` is not applied (the
/// quad's size is `size_m`).
// Q-210
pub fn panel_hit(
    origin: Vec3,
    dir: Vec3,
    panel: Pose,
    size_m: Vec2,
    size_px: UVec2,
) -> Option<Vec2> {
    let normal = panel.rotation * Vec3::Z;
    let along = dir.dot(normal);
    if along.is_nan() || along >= 0.0 || size_m.x <= 0.0 || size_m.y <= 0.0 {
        return None;
    }
    let t = (panel.translation - origin).dot(normal) / along;
    if !t.is_finite() || t < 0.0 {
        return None;
    }
    let local = panel.rotation.inverse() * (origin + dir * t - panel.translation);
    let u = local.x / size_m.x + 0.5;
    let v = 0.5 - local.y / size_m.y;
    if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
        return None;
    }
    Some(Vec2::new(u * size_px.x as f32, v * size_px.y as f32))
}

/// The pointer events of one frame for a context at one pixel per point (I-11): the
/// move to `hit`, then the select button's changes there; `PointerGone` when the ray left
/// the quad. `last` is the previous frame's hit.
pub(crate) fn pointer_events(
    last: Option<Vec2>,
    hit: Option<Vec2>,
    select: &[bool],
) -> Vec<egui::Event> {
    let mut out = Vec::new();
    match hit {
        Some(p) => {
            let pos = egui::pos2(p.x, p.y);
            out.push(egui::Event::PointerMoved(pos));
            for &pressed in select {
                out.push(egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: egui::Modifiers::NONE,
                });
            }
        }
        None if last.is_some() => out.push(egui::Event::PointerGone),
        None => {}
    }
    out
}

/// The HUD's data (§1.13 HUD), filled every frame as far as the app can; the overlay
/// that draws it is M-9's, the "active" count and the XR fields join in M-9 and M-5.
// Q-159
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct HudData {
    /// Frames per second, from the frame time.
    pub fps: f32,
    /// The display's refresh rate, when known.
    pub refresh_hz: Option<f32>,
    /// The CPU frame time in milliseconds.
    pub cpu_ms: f32,
    /// The GPU's stage times in milliseconds.
    pub gpu: GpuTimes,
    /// The loaded primitives, summed over the instances.
    pub loaded: u64,
    /// The visible splat records.
    pub visible_splats: u32,
    /// The visible center records.
    pub visible_centers: u32,
    /// C.
    pub capacity: u32,
    /// The overflow warning (§1.4 Several models).
    pub overflow: bool,
    /// The non-finite rows (E6).
    pub non_finite: u32,
    /// The instances' static GPU bytes.
    pub gpu_bytes_used: u64,
    /// The budget in bytes.
    pub gpu_bytes_budget: u64,
    /// Where the budget comes from, once a device exists.
    pub budget_source: Option<BudgetSource>,
    /// The clock T in seconds.
    pub clock_t: f64,
    /// Each instance's τ and badges.
    pub instances: Vec<HudInstance>,
    /// The active tier's name.
    pub tier: String,
    /// The governor's current knobs: render scale, SH cap, sort mode, foveation.
    pub knobs: (f32, u8, Option<SortMode>, Option<Foveation>),
}

/// One instance's line of the HUD (Q-159).
// Q-159
#[derive(Clone, Debug, PartialEq)]
pub struct HudInstance {
    /// Its name.
    pub name: String,
    /// τ_m in the model's unit.
    pub tau: f64,
    /// The "extrapolating" badge.
    pub extrapolating: bool,
    /// The "reduced" label (D-13).
    pub reduced: bool,
}

/// A message line for the user (§1.8, §1.13, §1.15). Like `Intent`, it derives `Message`
/// (read with a `MessageReader`) and `Event` (Q-166); ui writes it once its rate limit
/// lets a request through.
// Q-166, Q-208
#[derive(Message, Event, Clone, Debug, PartialEq)]
pub struct Toast {
    /// The text, `<code>: <user text>` for an error (Q-64).
    pub text: String,
    /// Its level.
    pub level: ToastLevel,
}

/// A toast's level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ToastLevel {
    /// Information ("pose saved").
    Info,
    /// A recoverable condition.
    Warning,
    /// A failure.
    Error,
}

/// A toast asked for by any module; ui's rate limit turns it into a `Toast` (§1.13).
/// Its site, the rate limit's key, is the toast's text.
// Q-208
#[derive(Message, Clone, Debug, PartialEq)]
pub(crate) struct ToastRequest {
    pub(crate) toast: Toast,
}

impl ToastRequest {
    pub(crate) fn new(toast: Toast) -> ToastRequest {
        ToastRequest { toast }
    }

    pub(crate) fn error(text: impl Into<String>) -> ToastRequest {
        ToastRequest::new(Toast {
            text: text.into(),
            level: ToastLevel::Error,
        })
    }

    pub(crate) fn warning(text: impl Into<String>) -> ToastRequest {
        ToastRequest::new(Toast {
            text: text.into(),
            level: ToastLevel::Warning,
        })
    }
}

/// What ui asks the UI sounds for (§1.10's triggers); `audio` maps it to `PlayUiSound`.
#[derive(Message, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UiCue {
    Hover,
    Select,
    Click,
    Error,
    Confirm,
}

/// A scene load the user confirmed (Q-200); `persist` loads it as `PersistCommand::LoadScene`.
// Q-200
#[derive(Message, Clone, Debug, PartialEq)]
pub(crate) struct SceneLoadConfirmed(pub(crate) PathBuf);

/// An instance's E14 manual values, edited in the Scene tab and saved as its scene
/// `lighting` (§1.5 Instances, §1.11); an instance without one is at the identity.
// Q-218, Q-206
#[derive(Component, Clone, Debug, Default, PartialEq)]
pub(crate) struct InstanceLighting(pub(crate) ManualLighting);

impl InstanceLighting {
    /// The component with the `SplatTransform` its values give.
    pub(crate) fn bundle(m: ManualLighting) -> (InstanceLighting, SplatTransform) {
        let gain = manual_gain(&m);
        (InstanceLighting(m), SplatTransform(ColorTransform { gain }))
    }
}

/// E14's manual form: g = 2^EV and w = (1 + 0.4 t, 1 − 0.4 u, 1 − 0.4 t) scaled to mean
/// 1, unclamped; the gain is g w, the identity at zero. The app's copy of
/// `ftgsxr_xr::light::manual_gains`, which `ftgsxr-app` may not call (§2.1's bands).
// Q-250
pub(crate) fn manual_gain(m: &ManualLighting) -> Vec3 {
    let g = m.exposure_ev.exp2();
    let w = Vec3::new(
        1.0 + 0.4 * m.temperature,
        1.0 - 0.4 * m.tint,
        1.0 - 0.4 * m.temperature,
    );
    let mean = (w.x + w.y + w.z) / 3.0;
    g * w / mean
}

/// The most recent eight scene files, newest first (§1.5 Scene files, Q-200): the scene
/// load dialog's list. `persist` refreshes it at startup and after each save and load; it
/// lives here because `ui` comes before `persist` in §2.1's order (#417).
// Q-200
#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub(crate) struct RecentScenes(pub(crate) Vec<PathBuf>);

/// The most recent eight `*.ftgsxr-scene.json` under `dir` by modification time, newest
/// first (Q-200); none when `dir` cannot be read.
// Q-200
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

/// A section a later module draws into a dock tab (`persist`'s Settings tables, the
/// governor's knobs, `audio`'s volumes), so ui names none of their types.
pub(crate) type DockHook = fn(&mut World, &mut egui::Ui);

/// The dock's tabs (§1.7 Panels).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) enum Tab {
    #[default]
    Library,
    Scene,
    Playback,
    Display,
    Environment,
    Settings,
    Help,
}

impl Tab {
    pub(crate) const ALL: [Tab; 7] = [
        Tab::Library,
        Tab::Scene,
        Tab::Playback,
        Tab::Display,
        Tab::Environment,
        Tab::Settings,
        Tab::Help,
    ];
}

/// The later modules' sections, per tab.
#[derive(Resource, Default)]
pub(crate) struct DockHooks(pub(crate) HashMap<Tab, Vec<DockHook>>);

impl DockHooks {
    pub(crate) fn add(&mut self, tab: Tab, hook: DockHook) {
        self.0.entry(tab).or_default().push(hook);
    }
}

/// A question the user must answer before the command runs (Appendix D.1, §1.5).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Confirm {
    /// Unload this instance (Q-199).
    Unload(Entity),
    /// Replace the scene by a file chosen from the list (Q-200).
    LoadScene(Option<PathBuf>),
}

/// The UI's state: what is shown, the dock, the open question, the toast line.
#[derive(Resource, Clone, Debug)]
pub(crate) struct UiState {
    /// F2 (§1.7).
    pub(crate) visible: bool,
    /// F3's flag; the HUD overlay is M-9's.
    pub(crate) hud: bool,
    pub(crate) dock_open: bool,
    /// The dock's width in points (remembered, §1.7).
    pub(crate) dock_width: f32,
    pub(crate) tab: Tab,
    /// The question asked, if any.
    pub(crate) confirm: Option<Confirm>,
    /// Frames to skip drawing (capture's request, #287).
    pub(crate) hidden_frames: u8,
    /// The toasts shown, with the time they leave the line.
    pub(crate) line: Vec<(Toast, f64)>,
    /// The Library tab's filters.
    pub(crate) filter_name: String,
    pub(crate) filter_kinds: [bool; 4],
    /// Load the next model reduced to the budget (D-13).
    pub(crate) reduce: bool,
    /// The Library tab's new source: an HTTP index URL or a local folder path (§1.5
    /// Sources; the path field is Q-160's fallback for the picker).
    pub(crate) new_source: String,
}

impl Default for UiState {
    fn default() -> Self {
        UiState {
            visible: true,
            hud: false,
            // §1.7 Panels' defaults, as `[desktop]`'s.
            dock_open: DesktopSettings::default().dock_open,
            dock_width: DesktopSettings::default().dock_width,
            tab: Tab::Library,
            confirm: None,
            hidden_frames: 0,
            line: Vec::new(),
            filter_name: String::new(),
            filter_kinds: [true; 4],
            reduce: false,
            new_source: String::new(),
        }
    }
}

impl UiState {
    /// Hides the UI for the next drawn frame (capture's screenshot, #287).
    pub(crate) fn hide_for_one_frame(&mut self) {
        self.hidden_frames = self.hidden_frames.max(1);
    }

    /// The answer to the open question: yes gives its command, no drops it.
    pub(crate) fn answer(&mut self, yes: bool) -> Answered {
        let mut a = Answered {
            intent: None,
            scene: None,
            cue: UiCue::Click,
        };
        match self.confirm.take() {
            Some(Confirm::Unload(instance)) if yes => {
                a.intent = Some(Intent::Instance {
                    instance,
                    action: InstanceAction::Unload,
                });
                a.cue = UiCue::Confirm;
            }
            Some(Confirm::LoadScene(Some(path))) if yes => {
                a.scene = Some(path);
                a.cue = UiCue::Confirm;
            }
            Some(Confirm::LoadScene(None)) if yes => self.confirm = Some(Confirm::LoadScene(None)),
            _ => {}
        }
        a
    }
}

/// What an answer sends.
#[derive(Debug, PartialEq)]
pub(crate) struct Answered {
    pub(crate) intent: Option<Intent>,
    pub(crate) scene: Option<PathBuf>,
    pub(crate) cue: UiCue,
}

/// How long a toast stays on the line, in seconds.
// Q-208
const TOAST_SHOWN_S: f64 = 5.0;
/// The rate limit: one toast of the same text per second (§1.13 Logs).
// Q-208
const TOAST_PERIOD_S: f64 = 1.0;

/// The rate limit's last time per text.
#[derive(Resource, Default)]
struct ToastLimiter(HashMap<String, f64>);

/// The entries' failures already toasted, by source and location.
#[derive(Resource, Default)]
struct SeenFailures(HashMap<(usize, String), String>);

/// What each instance's latest `SplatEvent::Uploaded` reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Upload {
    /// Its static GPU bytes, for the HUD.
    bytes: u64,
    /// The SH degree it was uploaded at, for the replay row (Q-241).
    sh_degree: u8,
}

/// The latest upload per instance; a reload's upload replaces it (Q-238, Q-241).
// Q-241
#[derive(Resource, Default)]
struct Uploads(HashMap<Entity, Upload>);

/// The panel pointer's last hit and the select changes waiting for it.
#[derive(Resource, Default)]
struct PanelPointer {
    last: Option<Vec2>,
}

/// The ui systems that turn state into toasts and commands, ordered after the modules
/// that produce them.
#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct UiSet;

/// The module's plugin, added by `AppPlugin` after `RigPlugin`.
pub(crate) struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<Toast>()
            .add_message::<ToastRequest>()
            .add_message::<UiCue>()
            .add_message::<SceneLoadConfirmed>()
            .add_message::<UnloadRequest>()
            .add_message::<Intent>()
            .add_message::<LibraryCommand>()
            .add_message::<LibraryFailure>()
            .add_message::<PlaybackWarning>()
            .init_resource::<UiSurface>()
            .init_resource::<UiState>()
            .init_resource::<RecentScenes>()
            .init_resource::<HudData>()
            .init_resource::<DockHooks>()
            .init_resource::<ToastLimiter>()
            .init_resource::<SeenFailures>()
            .init_resource::<Uploads>()
            .add_systems(
                Update,
                (ui_commands, library_toasts, clock_toasts).in_set(UiSet),
            )
            .add_systems(PostUpdate, (raise_toasts, fill_hud).chain())
            .add_observer(splat_toasts)
            .add_observer(triggered_toasts);
        // `persist`'s scene query needs the component known before any instance has one.
        app.world_mut().register_component::<InstanceLighting>();
        // §1.7 Training cameras and requests.
        cameras::build(app);
        if app.is_plugin_added::<EguiPlugin>() {
            app.init_resource::<PanelPointer>()
                .init_resource::<Scrubbing>()
                .add_systems(PreUpdate, panel_pointer)
                .add_systems(PostUpdate, end_scrub.before(EguiPostUpdateSet::EndPass))
                .add_systems(EguiPrimaryContextPass, dock::draw);
        }
    }
}

/// Ends the scrub before egui's pass, every frame: the dock's playback tab sets it again
/// while the scrub bar is pressed, so a release, a frame without the bar or without an
/// egui context ends it (Q-246).
// Q-246
fn end_scrub(mut scrubbing: ResMut<Scrubbing>) {
    scrubbing.0 = false;
}

/// `UiCommand::Help`, `HideUi`, `Hud`, `Fullscreen`, `OpenFiles` and `LoadScene`
/// (Appendix D.1), and Delete's request (Q-199).
#[allow(clippy::too_many_arguments)]
fn ui_commands(
    mut intents: MessageReader<Intent>,
    mut unloads: MessageReader<UnloadRequest>,
    mut state: ResMut<UiState>,
    mut library: MessageWriter<LibraryCommand>,
    mut cues: MessageWriter<UiCue>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    selected: Query<(), Added<Selected>>,
) {
    for i in intents.read() {
        match i {
            Intent::Ui(UiCommand::Help) => {
                state.visible = true;
                state.dock_open = true;
                state.tab = Tab::Help;
            }
            Intent::Ui(UiCommand::HideUi) => state.visible = !state.visible,
            Intent::Ui(UiCommand::Hud) | Intent::Toggle(Feature::Hud) => state.hud = !state.hud,
            Intent::Ui(UiCommand::Fullscreen) => {
                for mut w in &mut windows {
                    w.mode = match w.mode {
                        WindowMode::Windowed => {
                            WindowMode::BorderlessFullscreen(MonitorSelection::Current)
                        }
                        _ => WindowMode::Windowed,
                    };
                }
            }
            Intent::Ui(UiCommand::OpenFiles) => {
                library.write(LibraryCommand::Pick(PickKind::Files));
            }
            // Q-200: ui confirms; the dialog lists the scenes.
            Intent::Ui(UiCommand::LoadScene) => {
                state.visible = true;
                state.confirm = Some(Confirm::LoadScene(None));
            }
            _ => {}
        }
    }
    // Q-199
    for UnloadRequest(e) in unloads.read() {
        state.visible = true;
        state.confirm = Some(Confirm::Unload(*e));
    }
    if !selected.is_empty() {
        cues.write(UiCue::Select);
    }
}

/// `EntryState::Failed` once per failure (§1.15) and the library's own failures (Q-173,
/// Q-193). An entry's failure is keyed by its source and location, since listings move.
fn library_toasts(
    library: Option<Res<Library>>,
    mut seen: ResMut<SeenFailures>,
    mut failures: MessageReader<LibraryFailure>,
    mut out: MessageWriter<ToastRequest>,
) {
    if let Some(library) = library.filter(|l| l.is_changed()) {
        let mut now = HashMap::new();
        for e in &library.entries {
            if let EntryState::Failed(text) = &e.state {
                let key = (e.source, e.entry.location.clone());
                if seen.0.get(&key) != Some(text) {
                    out.write(ToastRequest::error(text.clone()));
                }
                now.insert(key, text.clone());
            }
        }
        seen.0 = now;
    }
    for f in failures.read() {
        out.write(ToastRequest::error(f.text.clone()));
    }
}

/// The clock's warnings (§1.13), as Warning toasts (Q-217).
// Q-217
fn clock_toasts(
    mut warnings: MessageReader<PlaybackWarning>,
    mut out: MessageWriter<ToastRequest>,
) {
    for PlaybackWarning(text) in warnings.read() {
        out.write(ToastRequest::warning(text.clone()));
    }
}

/// The overflow warning (§1.4 Several models). `SplatEvent::Failed` reaches the user
/// through the library, which turns it into `EntryState::Failed` or a `LibraryFailure`
/// with N added (Q-117, Q-173), so it is not toasted twice here.
// Q-209
fn splat_toasts(
    event: On<SplatEvent>,
    mut out: MessageWriter<ToastRequest>,
    mut uploads: ResMut<Uploads>,
) {
    {
        match &*event {
            SplatEvent::Overflow { capacity, .. } => {
                out.write(ToastRequest::warning(overflow_text(*capacity)));
            }
            SplatEvent::Uploaded {
                entity,
                bytes: b,
                sh_degree,
            } => {
                uploads.0.insert(
                    *entity,
                    Upload {
                        bytes: *b,
                        sh_degree: *sh_degree,
                    },
                );
            }
            SplatEvent::Failed { .. } => {}
        }
    }
}

/// The overflow toast's text.
// Q-209
pub(crate) fn overflow_text(capacity: u32) -> String {
    // C only: under the overflow flag the visible count is clamped to C (§1.4 stage 3,
    // Q-78), so it would repeat C. The text carries no per-frame number because the rate
    // limit keys on it (Q-208): a per-frame overflow shows once a second.
    format!("Too many primitives: capacity {capacity} reached; some were dropped")
}

/// A `Toast` triggered from outside the crate (`ftgsxr_xr::display`'s E-XR-UNSUPPORTED,
/// Q-161) asks for a toast like any module's `ToastRequest`, through the rate limit.
// Q-221
fn triggered_toasts(toast: On<Toast>, mut out: MessageWriter<ToastRequest>) {
    out.write(ToastRequest::new(toast.event().clone()));
}

/// The rate limit (§1.13: one per second per site, the site being the text): the
/// requests become `Toast`s, enter the line, and an error plays its sound.
// Q-208
fn raise_toasts(
    time: Res<Time>,
    mut limiter: ResMut<ToastLimiter>,
    mut state: ResMut<UiState>,
    mut requests: MessageReader<ToastRequest>,
    mut toasts: MessageWriter<Toast>,
    mut cues: MessageWriter<UiCue>,
) {
    let now = time.elapsed_secs_f64();
    limiter.0.retain(|_, t| now - *t < TOAST_PERIOD_S);
    state.line.retain(|(_, until)| *until > now);
    for r in requests.read() {
        let t = &r.toast;
        if limiter.0.contains_key(&t.text) {
            continue;
        }
        limiter.0.insert(t.text.clone(), now);
        if t.level == ToastLevel::Error {
            cues.write(UiCue::Error);
        }
        if t.level != ToastLevel::Info {
            warn!("{}", t.text);
        }
        state.line.push((t.clone(), now + TOAST_SHOWN_S));
        toasts.write(t.clone());
    }
}

/// `HudData` from the adapter's statistics, the tier, the clock and the instances.
#[allow(clippy::too_many_arguments)]
fn fill_hud(
    time: Res<Time>,
    stats: Option<Res<SplatStats>>,
    tier: Option<Res<SplatTierState>>,
    gpu: Option<Res<SplatGpu>>,
    playback: Option<Res<Playback>>,
    mut uploads: ResMut<Uploads>,
    instances: Query<(Entity, &LoadedModel, Option<&SplatTime>)>,
    mut hud: ResMut<HudData>,
) {
    let dt = time.delta_secs();
    hud.cpu_ms = dt * 1000.0;
    hud.fps = if dt > 0.0 { 1.0 / dt } else { 0.0 };
    if let Some(s) = stats.as_ref().and_then(|s| s.0.as_ref()) {
        hud.gpu = s.gpu_times_ms;
        hud.visible_splats = s.visible_splats;
        hud.visible_centers = s.visible_centers;
        hud.capacity = s.capacity;
        hud.overflow = s.overflow;
        hud.non_finite = s.non_finite;
    }
    if let Some(t) = tier {
        hud.tier = t.tier.name.clone();
        hud.gpu_bytes_budget = t.tier.memory_budget_bytes;
        hud.knobs = (
            t.render_scale,
            t.sh_cap,
            Some(t.sort_mode),
            Some(t.foveation),
        );
    }
    if let Some(g) = gpu {
        let b = g.device().memory_budget();
        hud.budget_source = Some(b.source);
        if hud.gpu_bytes_budget == 0 {
            hud.gpu_bytes_budget = b.bytes;
        }
    }
    if let Some(p) = playback {
        hud.clock_t = p.clock.time();
    }
    uploads.0.retain(|e, _| instances.contains(*e));
    hud.gpu_bytes_used = uploads.0.values().map(|u| u.bytes).sum();
    hud.loaded = instances.iter().map(|(_, m, _)| m.info.n).sum();
    hud.instances = instances
        .iter()
        .map(|(_, m, t)| HudInstance {
            name: m.name.clone(),
            tau: t.map_or(0.0, |t| t.tau),
            extrapolating: t.is_some_and(|t| t.extrapolating),
            reduced: m.reduced,
        })
        .collect();
}

/// The panel pointer (§1.8, I-11): the rays' hits on the panel quad become pointer
/// events of the context rendering to its image. The intents are the last frame's,
/// read before egui takes its input.
fn panel_pointer(
    surface: Res<UiSurface>,
    mut intents: MessageReader<Intent>,
    panels: Query<(&WorldPanel, &GlobalTransform)>,
    contexts: Query<(Entity, &RenderTarget)>,
    mut pointer: ResMut<PanelPointer>,
    mut out: MessageWriter<EguiInputEvent>,
) {
    let UiSurface::Panel {
        entity,
        size_px,
        size_m,
    } = *surface
    else {
        intents.clear();
        return;
    };
    let Ok((panel, at)) = panels.get(entity) else {
        intents.clear();
        return;
    };
    let Some(context) = contexts.iter().find_map(|(e, t)| match t {
        RenderTarget::Image(i) if i.handle == panel.image => Some(e),
        _ => None,
    }) else {
        intents.clear();
        return;
    };
    let (scale, rotation, translation) = at.to_scale_rotation_translation();
    let pose = Pose {
        translation,
        rotation,
        scale: scale.x,
    };
    let mut hit = pointer.last;
    let mut select = Vec::new();
    let mut moved = false;
    for i in intents.read() {
        match i {
            Intent::Pointer { origin, dir, .. } => {
                hit = panel_hit(*origin, *dir, pose, size_m, size_px);
                moved = true;
            }
            Intent::Select { pressed, .. } => select.push(*pressed),
            _ => {}
        }
    }
    if !moved && select.is_empty() {
        return;
    }
    for event in pointer_events(pointer.last, hit, &select) {
        out.write(EguiInputEvent { context, event });
    }
    pointer.last = hit;
}

/// The first readable system font: egui's bundled fonts are off (#277), so the desktop
/// loads one of the system's sans-serif fonts at startup; without one the UI draws no
/// text and a warning is logged.
// Q-211
pub(crate) const SYSTEM_FONTS: [&str; 7] = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/system/fonts/Roboto-Regular.ttf",
];

/// Installs the first readable font of `SYSTEM_FONTS` in `ctx`; whether one was.
// Q-211
pub(crate) fn load_system_font(ctx: &egui::Context) -> bool {
    for path in SYSTEM_FONTS {
        if let Ok(bytes) = std::fs::read(path) {
            let mut fonts = egui::FontDefinitions::empty();
            fonts
                .font_data
                .insert("system".into(), egui::FontData::from_owned(bytes).into());
            for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                fonts.families.insert(family, vec!["system".into()]);
            }
            ctx.set_fonts(fonts);
            return true;
        }
    }
    warn!("no system font found; the UI draws no text");
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    // Q-250: U-20's manual case (EV 1 → g = 2), the identity at zero, and E14's w.
    #[test]
    fn manual_gain_gives_e14s_manual_form() {
        let at = |exposure_ev, temperature, tint| {
            manual_gain(&ManualLighting {
                exposure_ev,
                temperature,
                tint,
            })
        };
        assert_eq!(at(0.0, 0.0, 0.0), Vec3::ONE);
        assert!((at(1.0, 0.0, 0.0) - Vec3::splat(2.0)).abs().max_element() < 1e-6);
        // w = (1.4, 1, 0.6), already of mean 1: warmer.
        assert!(
            (at(0.0, 1.0, 0.0) - Vec3::new(1.4, 1.0, 0.6))
                .abs()
                .max_element()
                < 1e-6
        );
        // w = (1, 0.6, 1) / (2.6 / 3): less green.
        let k = 3.0 / 2.6;
        assert!(
            (at(0.0, 0.0, 1.0) - Vec3::new(k, 0.6 * k, k))
                .abs()
                .max_element()
                < 1e-6
        );
        // Unclamped: EV 3 → g = 8.
        assert!((at(3.0, 0.0, 0.0) - Vec3::splat(8.0)).abs().max_element() < 1e-5);
    }

    // Q-200 (moved from `persist`, #417): the newest eight scene files, newest first,
    // other files left out.
    #[test]
    fn recent_scenes_are_the_newest_eight() {
        let dir = std::env::temp_dir().join(format!("ftgsxr-recent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let base = std::time::SystemTime::now() - std::time::Duration::from_secs(100);
        for k in 0..10u64 {
            let p = dir.join(format!("s{k}.ftgsxr-scene.json"));
            std::fs::write(&p, "{}").unwrap();
            let f = std::fs::File::options().write(true).open(&p).unwrap();
            f.set_modified(base + std::time::Duration::from_secs(k))
                .unwrap();
        }
        std::fs::write(dir.join("other.json"), "{}").unwrap();
        let recent = recent_scenes(&dir);
        assert_eq!(recent.len(), 8);
        assert_eq!(recent[0], dir.join("s9.ftgsxr-scene.json"));
        assert_eq!(recent[7], dir.join("s2.ftgsxr-scene.json"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // Q-199: Delete's request asks; yes unloads, no leaves the instance.
    #[test]
    fn an_unload_request_unloads_only_once_confirmed() {
        let e = Entity::from_raw_u32(7).unwrap();
        let mut s = UiState {
            confirm: Some(Confirm::Unload(e)),
            ..UiState::default()
        };
        let no = s.answer(false);
        assert_eq!((no.intent, no.cue), (None, UiCue::Click));
        assert_eq!(s.confirm, None);
        s.confirm = Some(Confirm::Unload(e));
        let yes = s.answer(true);
        assert_eq!(
            yes.intent,
            Some(Intent::Instance {
                instance: e,
                action: InstanceAction::Unload
            })
        );
        assert_eq!(yes.cue, UiCue::Confirm);
    }

    // Q-200: a confirmed scene load goes to persist with its path.
    #[test]
    fn a_scene_load_waits_for_its_file_and_the_confirmation() {
        let mut s = UiState {
            confirm: Some(Confirm::LoadScene(None)),
            ..UiState::default()
        };
        assert_eq!(s.answer(true).scene, None);
        assert_eq!(s.confirm, Some(Confirm::LoadScene(None)));
        let p = PathBuf::from("a.ftgsxr-scene.json");
        s.confirm = Some(Confirm::LoadScene(Some(p.clone())));
        assert_eq!(s.answer(true).scene, Some(p));
    }

    #[test]
    fn pointer_events_move_then_press_and_leave() {
        let p = Vec2::new(3.0, 4.0);
        let ev = pointer_events(None, Some(p), &[true]);
        assert_eq!(ev[0], egui::Event::PointerMoved(egui::pos2(3.0, 4.0)));
        assert!(matches!(
            ev[1],
            egui::Event::PointerButton { pressed: true, .. }
        ));
        assert_eq!(
            pointer_events(Some(p), None, &[]),
            vec![egui::Event::PointerGone]
        );
        assert!(pointer_events(None, None, &[true]).is_empty());
    }

    /// `pointer_events` from a ray's hit drive a second egui context: the pointer at the
    /// hit's pixel (1 px) and a select press and release clicking a button there. I-11's
    /// row is pinned at the public interface in `tests/i11_panel_pointer.rs`.
    #[test]
    fn pointer_events_from_a_quad_hit_click_in_a_second_context() {
        let size_px = UVec2::new(1024, 768);
        let size_m = Vec2::new(0.56, 0.42);
        let panel = Pose {
            translation: Vec3::new(0.2, 1.4, -1.0),
            rotation: bevy::math::Quat::from_rotation_y(0.3),
            scale: 1.0,
        };
        // The button covers pixels (600, 300) to (680, 340); the ray aims at (640.5, 320.25).
        let button = egui::Rect::from_min_max(egui::pos2(600.0, 300.0), egui::pos2(680.0, 340.0));
        let target_px = Vec2::new(640.5, 320.25);
        let local = Vec2::new(
            (target_px.x / size_px.x as f32 - 0.5) * size_m.x,
            (0.5 - target_px.y / size_px.y as f32) * size_m.y,
        );
        let origin = Vec3::new(0.0, 1.6, 0.0);
        let dir = (panel.translation + panel.rotation * local.extend(0.0) - origin).normalize();
        let hit = panel_hit(origin, dir, panel, size_m, size_px).expect("a hit");
        assert!((hit - target_px).abs().max_element() <= 1.0, "{hit}");

        let ctx = egui::Context::default();
        let mut toggled = false;
        let mut last = None;
        let mut pointer = None;
        for select in [vec![], vec![true], vec![false], vec![]] {
            let events = pointer_events(last, Some(hit), &select);
            last = Some(hit);
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(size_px.x as f32, size_px.y as f32),
                )),
                events,
                ..Default::default()
            };
            let mut output = ctx.run_ui(input, |ui| {
                let r = ui.allocate_rect(button, egui::Sense::click());
                if r.clicked() {
                    toggled = !toggled;
                }
            });
            output.textures_delta.clear();
            pointer = ctx.input(|i| i.pointer.latest_pos());
        }
        let p = pointer.expect("a pointer position");
        assert!(
            (p.x - target_px.x).abs() <= 1.0 && (p.y - target_px.y).abs() <= 1.0,
            "{p:?}"
        );
        assert!(toggled, "the click toggled the button");
    }
}
