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
mod panel;
mod pointer_hand;
mod touch;
mod trim_row;

pub(crate) use panel::{PanelCamera, panel_zoom};

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
use bevy::window::{MonitorSelection, PrimaryWindow, Window, WindowMode};
use bevy_egui::input::EguiInputEvent;
use bevy_egui::{
    EguiContext, EguiGlobalSettings, EguiOutput, EguiPlugin, EguiPostUpdateSet,
    EguiPrimaryContextPass, egui,
};
use ftgsxr_core::Pose;
use ftgsxr_io::{DesktopSettings, ManualLighting, ModelRef};
use ftgsxr_render::{
    BudgetSource, ColorTransform, Foveation, GpuTimes, SortMode, SplatEvent, SplatGpu, SplatStats,
    SplatTierState, SplatTime, SplatTransform,
};

use crate::intent::{Feature, Hand, InstanceAction, Intent, UiCommand, UnloadRequest};
use crate::library::{
    EntryState, Library, LibraryCommand, LibraryFailure, LoadedModel, ReplaceEntry,
};
use crate::manipulate::Selected;
use crate::platform::{Headset, PickHandle, PickKind, PlatformRes};
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

/// A hand's ray on a panel quad for one frame (§1.8 User interface, I-11): the XR
/// adapter tests each hand's ray against the main panel and the wrist menu first and, on
/// a hit, writes this instead of that hand's `Intent::Pointer` and `Intent::Select`
/// (Q-263). `panel` is the quad's entity (it carries `WorldPanel`), `pixel` the hit in
/// the texture's pixels (`panel_hit`), `select` and `grab` the frame's presses (`true`)
/// and releases (`false`) of the select and grab inputs in order, and `scroll_px` the
/// frame's scroll in pixels: positive moves the content down, revealing what is above
/// (egui's wheel convention; Q-279). A frame without one for a panel takes the
/// pointer off it.
// Q-263, Q-279
#[derive(Message, Clone, Debug, PartialEq)]
pub struct PanelPointer {
    /// The hand whose ray hits the panel.
    pub hand: Hand,
    /// The panel quad's entity.
    pub panel: Entity,
    /// The hit in the panel texture's pixels, (0, 0) its top-left corner.
    pub pixel: Vec2,
    /// The select input's presses and releases this frame, in order.
    pub select: Vec<bool>,
    /// The grab input's presses and releases this frame, in order.
    pub grab: Vec<bool>,
    /// The scroll this frame in pixels, positive moving the content down.
    pub scroll_px: f32,
}

/// A panel texture's pixel (or a scroll in pixels) in the points of a context drawing
/// `ppp` pixels per point (its zoom, Q-267), as bevy_egui sizes an image target's
/// screen: one pixel per point at zoom 1 (Q-279); a `ppp` not positive and finite counts
/// as 1.
// Q-290
pub(crate) fn panel_point(pixel: Vec2, ppp: f32) -> Vec2 {
    if ppp.is_finite() && ppp > 0.0 {
        pixel / ppp
    } else {
        pixel
    }
}

/// The pointer events of one frame for a context, in its points (I-11): the
/// move to `hit`, then the select button's changes there, then the scroll as a wheel
/// event; `PointerGone` when the ray left the quad. `last` is the previous frame's hit.
pub(crate) fn pointer_events(
    last: Option<Vec2>,
    hit: Option<Vec2>,
    select: &[bool],
    scroll_px: f32,
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
            // Q-279
            if scroll_px != 0.0 && scroll_px.is_finite() {
                out.push(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: egui::vec2(0.0, scroll_px),
                    phase: egui::TouchPhase::Move,
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
/// that draws it is M-9's, the "active" count joins in M-9. The XR fields are the XR
/// crate's to write each frame, untracked without a session (Q-274).
// Q-159, Q-274
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
    /// The rows outside their instance's trim (E17, §1.13).
    pub trimmed: u32,
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
    /// Whether the head is tracked this frame; false without an XR session.
    // Q-274
    pub head_tracked: bool,
    /// The left side's device and whether its aim is tracked this frame; `None` while
    /// no device is active for the side and without an XR session.
    // Q-274
    pub left_hand: Option<(HudHandSource, bool)>,
    /// The right side's device and whether its aim is tracked; as `left_hand`.
    // Q-274
    pub right_hand: Option<(HudHandSource, bool)>,
    /// The depth image's resolution and its age in frames (§1.9 Depth occlusion);
    /// `None` without an environment-depth provider. The XR crate's to write.
    // Q-345
    pub depth_image: Option<(UVec2, u32)>,
    /// The scene anchor's state (§1.9 Floor, shadows and anchors); `None` without one.
    /// The XR crate's to write.
    // Q-345
    pub anchor: Option<HudAnchor>,
    /// The light estimate's gains (§1.13 HUD) while an auto lighting mode runs the
    /// camera; `None` otherwise. The value is g·w per channel, as `ColorTransform.gain`
    /// holds it, not E14's g and w apart, and without the manual offset (Q-394). The XR
    /// crate's to write.
    // Q-389, Q-394
    pub light_gains: Option<Vec3>,
}

/// The scene anchor's state on the HUD (§1.13 HUD, Q-345).
// Q-345
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HudAnchor {
    /// Being created at a scene save.
    Creating,
    /// Created and persisted.
    Persisted,
    /// Being located at a scene load.
    Locating,
    /// Located: the content group is on it.
    Located,
    /// Not found: the file's pose is used.
    Lost,
}

/// Whether an XR feature exists on this device (§1.9, §1.15 E-XR-UNSUPPORTED and
/// E-XR-PERMISSION): `Absent` without the runtime's extension (the desktop's, and the
/// default), `Available`, or `Denied` when its permission was refused (Q-336).
// Q-336
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FeatureState {
    /// The runtime offers no extension for it: its control is hidden.
    #[default]
    Absent,
    /// It can be used.
    Available,
    /// Its permission was denied: the control stays, with a button to ask again.
    Denied,
}

/// The XR features the panels' controls follow (§1.8 User interface, §1.9): the wrist
/// menu's passthrough button and the Environment tab's passthrough and occlusion
/// controls show only where their feature is not `Absent`. The XR crate writes it each
/// frame; without it every feature is `Absent` (Q-336).
// Q-336
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct XrFeatures {
    /// Passthrough (XR_FB_passthrough).
    pub passthrough: FeatureState,
    /// Depth occlusion (XR_META_environment_depth).
    pub occlusion: FeatureState,
    /// Spatial anchors (XR_FB_spatial_entity, XR_META_spatial_entity_persistence).
    pub anchors: FeatureState,
    /// Hand tracking (XR_EXT_hand_tracking with XR_FB_hand_tracking_aim), which the tracked
    /// hands' gauntlets need (§1.9, Q-366).
    // Q-366
    pub hands: FeatureState,
    /// Light estimation's passthrough camera (§1.9): `Absent` without the XR crate, a
    /// session, the platform's `CameraFrames` or a passthrough camera; `Denied` after a
    /// denial. The Environment tab offers the auto lighting modes only while `Available`.
    // Q-384
    pub lighting: FeatureState,
}

/// Which device a side's HUD line shows (Q-274): the XR crate's `HandSource`, which the
/// app cannot name, mapped by the XR crate.
// Q-274
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HudHandSource {
    /// A controller.
    Controller,
    /// Hand tracking.
    Hand,
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

/// The environment's lighting transform (§1.9 Light estimation and color matching):
/// the identity in off mode, E14's manual form of `[environment] lighting_manual` in
/// manual mode (`persist`'s settings sync), and in the auto modes that manual form times
/// E14's gains, or the manual form alone where the gains are not in force (the XR crate;
/// E14, revision 52).
/// `compose_transforms` multiplies it into every instance's own transform while
/// `per_instance` is `None`, or only into the target's while it is `Some(e)`, whatever is
/// selected later; `per_instance` is session state, `None` at start and saved nowhere.
/// `too_dark` is the XR crate's, for the Environment tab: an auto mode waits for a
/// reference the room is too dark to give (Q-385).
// Q-383, Q-385
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct Lighting {
    /// The environment's gains, the identity by default.
    pub transform: ColorTransform,
    /// The one instance the environment's factor goes to ("per instance", the instance
    /// selected when it was turned on), `None` for every instance.
    pub per_instance: Option<Entity>,
    /// An auto mode is in force, `lighting_reference` is `None`, the warm-up is over and
    /// the smoothed estimate is below the reference's floor (Q-385).
    pub too_dark: bool,
}

impl Default for Lighting {
    fn default() -> Lighting {
        Lighting {
            transform: ColorTransform::IDENTITY,
            per_instance: None,
            too_dark: false,
        }
    }
}

/// An instance as `compose_transforms` reads it.
type ComposeItem = (
    Entity,
    Option<&'static InstanceLighting>,
    Option<&'static mut SplatTransform>,
);

/// `SplatTransform`'s one writer: each instance's own E14 manual transform (the identity
/// without `InstanceLighting`) times the environment's per channel, the environment's
/// factor going to every instance while `per_instance` is `None`, or only to its target
/// while it is `Some(e)`, whatever is selected. A target no longer loaded (removed)
/// returns `per_instance` to `None`. Writes only what changed.
// Q-383
fn compose_transforms(
    mut lighting: ResMut<Lighting>,
    mut instances: Query<ComposeItem, With<LoadedModel>>,
    mut commands: Commands,
) {
    if lighting
        .per_instance
        .is_some_and(|t| !instances.contains(t))
    {
        lighting.per_instance = None;
    }
    for (e, own, current) in &mut instances {
        let own = own.map_or(ColorTransform::IDENTITY, |l| {
            ColorTransform::from_manual(&l.0)
        });
        let gain = if lighting.per_instance.is_none_or(|t| t == e) {
            own.gain * lighting.transform.gain
        } else {
            own.gain
        };
        let next = SplatTransform(ColorTransform { gain });
        match current {
            Some(mut c) => {
                c.set_if_neq(next);
            }
            None => {
                commands.entity(e).try_insert(next);
            }
        }
    }
}

/// The set of `compose_transforms`, for the modules that insert `InstanceLighting` in
/// `PostUpdate` to order themselves before it.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct ComposeTransforms;

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

/// The dock's tabs (§1.7 Panels), and the main panel's Comfort tab (§1.8 User
/// interface), shown in panel mode only.
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
    Comfort,
}

impl Tab {
    /// The screen's tabs (§1.7 Panels).
    pub(crate) const ALL: [Tab; 7] = [
        Tab::Library,
        Tab::Scene,
        Tab::Playback,
        Tab::Display,
        Tab::Environment,
        Tab::Settings,
        Tab::Help,
    ];
    /// The main panel's tabs (§1.8 User interface): the desktop's and Comfort.
    pub(crate) const PANEL: [Tab; 8] = [
        Tab::Library,
        Tab::Scene,
        Tab::Playback,
        Tab::Display,
        Tab::Environment,
        Tab::Comfort,
        Tab::Settings,
        Tab::Help,
    ];
}

/// The later modules' sections, per tab, each with whether it is the screen's only (a
/// mouse-only or keyboard item, hidden on the main panel, §1.8 User interface).
#[derive(Resource, Default)]
pub(crate) struct DockHooks(pub(crate) HashMap<Tab, Vec<(DockHook, bool)>>);

impl DockHooks {
    /// A section drawn on both surfaces.
    pub(crate) fn add(&mut self, tab: Tab, hook: DockHook) {
        self.0.entry(tab).or_default().push((hook, false));
    }

    /// A section drawn on the screen only (Help's keyboard shortcuts).
    // Q-294
    pub(crate) fn add_screen_only(&mut self, tab: Tab, hook: DockHook) {
        self.0.entry(tab).or_default().push((hook, true));
    }

    /// The sections of `tab` drawn on the main panel (`panel`) or the screen.
    pub(crate) fn sections(&self, tab: Tab, panel: bool) -> Vec<DockHook> {
        self.0.get(&tab).map_or_else(Vec::new, |v| {
            v.iter()
                .filter(|(_, screen_only)| !(panel && *screen_only))
                .map(|(h, _)| *h)
                .collect()
        })
    }
}

/// A question the user must answer before the command runs (Appendix D.1, §1.5).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Confirm {
    /// Unload this instance (Q-199).
    Unload(Entity),
    /// Replace the scene by a file chosen from the list (Q-200).
    LoadScene(Option<PathBuf>),
    /// Put the picked `file` in the place of the missing scene model `model` of `identity`
    /// (§1.5 Library, Q-256).
    // Q-256
    Replace {
        identity: String,
        model: String,
        file: ModelRef,
    },
}

impl Confirm {
    /// A replacement's question, naming both models (Q-256): the missing entry's name and
    /// the picked file's path or URL, which tells the two apart even when the file has the
    /// model's name (a model retrained in place).
    // Q-256
    pub(crate) fn replace_text(model: &str, file: &ModelRef) -> String {
        let at = match file {
            ModelRef::File(p) => p.display().to_string(),
            ModelRef::Http(url) => url.to_string(),
            ModelRef::Document(uri) => uri.clone(),
        };
        format!("Replace {model} with {at}?")
    }
}

/// "Replace with…" on a missing entry (§1.5 Library, Q-255, Q-256): the dock writes it;
/// `ui` opens the picker and asks the confirmation once a file is picked.
// Q-256
#[derive(Message, Clone, Debug, PartialEq)]
pub(crate) struct ReplaceRequest(pub(crate) usize);

/// The open "Replace with…" pickers, each with the missing entry's identity and name.
// Q-256
#[derive(Resource, Default)]
struct ReplacePicks(Vec<(String, String, PickHandle)>);

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
    /// The main panel's open sub-page, if any (§1.8 User interface); an open question
    /// is the panel's sub-page too.
    pub(crate) page: Option<Page>,
}

/// A sub-page of the main panel, with a Back button at its top (§1.8 User interface).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Page {
    /// The Scene tab's training cameras (§1.7).
    Cameras,
    /// The Library tab's request replay (§1.7).
    Replay,
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
            page: None,
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
            replace: None,
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
            // Q-256: yes replaces; no drops the pick, the entry staying missing.
            Some(Confirm::Replace { identity, file, .. }) if yes => {
                a.replace = Some(ReplaceEntry { identity, file });
                a.cue = UiCue::Confirm;
            }
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
    /// A confirmed replacement (Q-255).
    // Q-255
    pub(crate) replace: Option<ReplaceEntry>,
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

/// Each panel context's pointer, its pointer hand and last position, by the context's
/// entity (Q-334).
#[derive(Resource, Default)]
struct PanelPointerState {
    pointers: HashMap<Entity, pointer_hand::ContextPointer>,
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
            .add_message::<ReplaceRequest>()
            .add_message::<ReplaceEntry>()
            .add_message::<Intent>()
            .add_message::<PanelPointer>()
            .add_message::<LibraryCommand>()
            .add_message::<LibraryFailure>()
            .add_message::<PlaybackWarning>()
            .init_resource::<UiSurface>()
            .init_resource::<UiState>()
            .init_resource::<RecentScenes>()
            .init_resource::<HudData>()
            // Q-336
            .init_resource::<XrFeatures>()
            // Q-383
            .init_resource::<Lighting>()
            .init_resource::<DockHooks>()
            .init_resource::<ToastLimiter>()
            .init_resource::<SeenFailures>()
            .init_resource::<Uploads>()
            .init_resource::<ReplacePicks>()
            .add_systems(
                Update,
                (ui_commands, replace_picks, library_toasts, clock_toasts).in_set(UiSet),
            )
            .add_systems(PostUpdate, (raise_toasts, fill_hud).chain())
            .add_systems(PostUpdate, compose_transforms.in_set(ComposeTransforms))
            .add_observer(splat_toasts)
            .add_observer(triggered_toasts);
        // `persist`'s scene query needs the component known before any instance has one.
        app.world_mut().register_component::<InstanceLighting>();
        // §1.7 Training cameras and requests.
        cameras::build(app);
        if app.is_plugin_added::<EguiPlugin>() {
            // §1.8 User interface, Q-266: on the headset bevy_egui's primary context
            // would latch onto the first eye camera, so it is off; the panels have their
            // own contexts, and the dock draws only with a primary window.
            if app.world().contains_resource::<Headset>()
                && let Some(mut g) = app.world_mut().get_resource_mut::<EguiGlobalSettings>()
            {
                g.auto_create_primary_context = false;
            }
            app.init_resource::<PanelPointerState>()
                .init_resource::<Scrubbing>()
                .init_schedule(panel::PanelPass)
                .init_schedule(panel::WristPass)
                .add_systems(PreUpdate, panel_pointer)
                .add_systems(Update, panel::follow_surfaces)
                .add_systems(PostUpdate, end_scrub.before(EguiPostUpdateSet::EndPass))
                // Q-324: after `Update`'s visibility rules, before the panels' passes.
                .add_systems(
                    PostUpdate,
                    panel::idle_hidden.before(EguiPostUpdateSet::EndPass),
                )
                .add_systems(
                    EguiPrimaryContextPass,
                    dock::draw.run_if(any_with_component::<PrimaryWindow>),
                )
                .add_systems(panel::PanelPass, panel::draw_panel)
                .add_systems(panel::WristPass, panel::draw_wrist);
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

/// "Replace with…" (§1.5 Library, Q-256): each request on a missing entry with an identity
/// opens the picker; its first picked file is the question `Confirm::Replace`, asked once
/// no other question is open. An empty pick asks nothing.
// Q-256, Q-257
fn replace_picks(
    mut requests: MessageReader<ReplaceRequest>,
    mut picks: ResMut<ReplacePicks>,
    mut state: ResMut<UiState>,
    library: Option<Res<Library>>,
    platform: Option<Res<PlatformRes>>,
) {
    let Some(platform) = platform else {
        requests.clear();
        return;
    };
    for ReplaceRequest(i) in requests.read() {
        let Some(e) = library.as_ref().and_then(|l| l.entries.get(*i)) else {
            continue;
        };
        if e.state != EntryState::Missing {
            continue;
        }
        let Some(id) = e.identity.clone() else {
            continue;
        };
        let name = e.entry.name.clone();
        let h = platform.0.pick(PickKind::Files);
        picks.0.push((id, name, h));
    }
    let open = std::mem::take(&mut picks.0);
    for (identity, model, h) in open {
        if state.confirm.is_some() {
            picks.0.push((identity, model, h));
            continue;
        }
        match platform.0.poll_pick(h) {
            std::task::Poll::Pending => picks.0.push((identity, model, h)),
            std::task::Poll::Ready(Err(e)) => warn!("the picker failed: {e}"),
            std::task::Poll::Ready(Ok(refs)) => {
                if let Some(file) = refs.into_iter().next() {
                    state.visible = true;
                    state.confirm = Some(Confirm::Replace {
                        identity,
                        model,
                        file,
                    });
                }
            }
        }
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
        hud.trimmed = s.trimmed;
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

/// The panel pointer (§1.8, I-11): each `PanelPointer` is a hand's ray on the panel of
/// the context rendering to its image; the context follows one hand at a time, its
/// pointer hand, and that hand's hits become its pointer events (Q-334); a context that
/// had the pointer and gets no message in a frame sees it leave (Q-263). The messages are
/// the last frame's, read before egui takes its input, and their pixels reach the context
/// as its points (Q-290). Each context's events are rewritten for the touch press (Q-333).
// Q-263, Q-290, Q-333, Q-334
fn panel_pointer(
    mut pointers: MessageReader<PanelPointer>,
    panels: Query<&WorldPanel>,
    contexts: Query<(Entity, &RenderTarget, Option<&EguiOutput>)>,
    mut egui_contexts: Query<&mut EguiContext>,
    mut state: ResMut<PanelPointerState>,
    mut out: MessageWriter<EguiInputEvent>,
) {
    let mut frame: Vec<(Entity, Vec<pointer_hand::HandHit>)> = Vec::new();
    for p in pointers.read() {
        let Ok(panel) = panels.get(p.panel) else {
            continue;
        };
        let Some((context, ppp)) = contexts.iter().find_map(|(e, t, o)| match t {
            RenderTarget::Image(i) if i.handle == panel.image => {
                Some((e, o.map_or(1.0, |o| o.pixels_per_point)))
            }
            _ => None,
        }) else {
            continue;
        };
        // Q-290: the texture's pixels in the context's points.
        let hit = pointer_hand::HandHit {
            hand: p.hand,
            at: panel_point(p.pixel, ppp),
            select: p.select.clone(),
            scroll: panel_point(Vec2::new(0.0, p.scroll_px), ppp).y,
        };
        match frame.iter_mut().find(|(c, _)| *c == context) {
            Some((_, hits)) => hits.push(hit),
            None => frame.push((context, vec![hit])),
        }
    }
    for &context in state.pointers.keys() {
        if !frame.iter().any(|(c, _)| *c == context) {
            frame.push((context, Vec::new()));
        }
    }
    let mut next = HashMap::new();
    for (context, hits) in frame {
        let mut ctx = egui_contexts.get_mut(context).ok();
        // Q-333
        let armed = ctx.as_mut().is_some_and(|c| touch::armed(c.get_mut()));
        let prev = state.pointers.get(&context).copied();
        // Q-334
        let (events, pointer) = pointer_hand::hand_events(prev, &hits, armed);
        if let Some(pointer) = pointer {
            next.insert(context, pointer);
        }
        // Q-333
        let events = match ctx {
            Some(mut ctx) => touch::touch_events(ctx.get_mut(), events),
            None => events,
        };
        for event in events {
            out.write(EguiInputEvent { context, event });
        }
    }
    state.pointers = next;
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

    /// A marked instance for the composer: `LoadedModel` from the P1 probe.
    fn probe() -> LoadedModel {
        let path = ftgsxr_testkit::fixtures_dir().join("probes/P1.safetensors");
        let bytes = std::fs::read(&path).unwrap();
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let info = ftgsxr_core::read_info(&bytes[8..8 + n], bytes.len() as u64).unwrap();
        LoadedModel {
            source: ModelRef::File(path),
            identity: "0".repeat(64),
            time: ftgsxr_core::ModelTime::from_span(&info.time),
            info,
            name: "P1".into(),
            reduced: false,
            audio: None,
            frame_step_s: 1.0 / 30.0,
        }
    }

    /// Q-383 (`cpu`): every instance's `SplatTransform` is the per-channel product of its
    /// own E14 manual transform and the environment's while `per_instance` is `None`;
    /// while it is `Some(a)`, only `a` takes the environment's factor, whatever is selected
    /// later (another instance, or nothing); removing `a` returns `per_instance` to `None`.
    #[test]
    fn compose_transforms_multiplies_the_instance_and_the_environment() {
        let mut app = App::new();
        app.init_resource::<Lighting>()
            .add_systems(PostUpdate, compose_transforms);
        // EV 1, neutral white balance: g = 2^1 = 2, w = (1, 1, 1).
        let ev1 = ManualLighting {
            exposure_ev: 1.0,
            ..ManualLighting::default()
        };
        let a = app.world_mut().spawn((probe(), InstanceLighting(ev1))).id();
        let b = app.world_mut().spawn(probe()).id();
        app.world_mut().resource_mut::<Lighting>().transform = ColorTransform {
            gain: Vec3::new(2.0, 1.0, 1.0),
        };
        let gain = |app: &App, e: Entity| app.world().get::<SplatTransform>(e).unwrap().0.gain;
        // `None`: every instance takes the product.
        app.update();
        assert_eq!(gain(&app, a), Vec3::new(4.0, 2.0, 2.0));
        assert_eq!(gain(&app, b), Vec3::new(2.0, 1.0, 1.0));
        // Pinned to A, then B selected: A keeps the product, B its own transform.
        app.world_mut().resource_mut::<Lighting>().per_instance = Some(a);
        app.world_mut().entity_mut(b).insert(Selected);
        app.update();
        assert_eq!(gain(&app, a), Vec3::new(4.0, 2.0, 2.0));
        assert_eq!(gain(&app, b), Vec3::ONE);
        // Nothing selected: still A only.
        app.world_mut().entity_mut(b).remove::<Selected>();
        app.update();
        assert_eq!(gain(&app, a), Vec3::new(4.0, 2.0, 2.0));
        assert_eq!(gain(&app, b), Vec3::ONE);
        // A removed: `per_instance` returns to `None`, and B takes the product again.
        app.world_mut().despawn(a);
        app.update();
        assert_eq!(app.world().resource::<Lighting>().per_instance, None);
        app.update();
        assert_eq!(gain(&app, b), Vec3::new(2.0, 1.0, 1.0));
    }

    // Q-256: the question names the missing model and the picked file's path.
    #[test]
    fn replace_text_names_the_model_and_the_files_path() {
        let file = ModelRef::File(PathBuf::from("/m/moved.safetensors"));
        assert_eq!(
            Confirm::replace_text("moved", &file),
            "Replace moved with /m/moved.safetensors?"
        );
        let url = ModelRef::Http(ftgsxr_io::Url::parse("http://h/m/v1.2.safetensors").unwrap());
        assert_eq!(
            Confirm::replace_text("v1.2", &url),
            "Replace v1.2 with http://h/m/v1.2.safetensors?"
        );
    }

    // Q-256: yes gives the replacement; no drops it.
    #[test]
    fn answer_to_a_replacement_replaces_only_on_yes() {
        let file = ModelRef::File(PathBuf::from("/m/retrained.safetensors"));
        let ask = Confirm::Replace {
            identity: "ab".repeat(32),
            model: "moved".into(),
            file: file.clone(),
        };
        let mut s = UiState {
            confirm: Some(ask.clone()),
            ..UiState::default()
        };
        let no = s.answer(false);
        assert_eq!((no.replace, no.cue), (None, UiCue::Click));
        assert_eq!(s.confirm, None);
        s.confirm = Some(ask);
        let yes = s.answer(true);
        assert_eq!(
            yes.replace,
            Some(ReplaceEntry {
                identity: "ab".repeat(32),
                file
            })
        );
        assert_eq!(yes.cue, UiCue::Confirm);
    }

    /// A platform whose picks answer `answer`, counting the pickers opened.
    struct OnePick {
        paths: ftgsxr_io::AppPaths,
        answer: Vec<ModelRef>,
        opened: std::sync::Mutex<Vec<PickKind>>,
    }

    impl crate::platform::Platform for OnePick {
        fn paths(&self) -> &ftgsxr_io::AppPaths {
            &self.paths
        }
        fn pick(&self, kind: PickKind) -> PickHandle {
            self.opened.lock().unwrap().push(kind);
            PickHandle(0)
        }
        fn poll_pick(&self, _: PickHandle) -> std::task::Poll<Result<Vec<ModelRef>, String>> {
            std::task::Poll::Ready(Ok(self.answer.clone()))
        }
        fn request_permission(
            &self,
            _: crate::platform::Permission,
        ) -> crate::platform::PermissionHandle {
            crate::platform::PermissionHandle(0)
        }
        fn poll_permission(&self, _: crate::platform::PermissionHandle) -> std::task::Poll<bool> {
            std::task::Poll::Ready(true)
        }
        fn document_provider(&self) -> Option<std::sync::Arc<dyn ftgsxr_io::DocumentProvider>> {
            None
        }
        fn camera_frames(&self) -> Option<std::sync::Arc<dyn crate::platform::CameraFrames>> {
            None
        }
    }

    /// An app with `replace_picks`, a library whose entry 0 is a missing model "moved",
    /// and a platform whose pick answers `answer`.
    fn replace_app(answer: Vec<ModelRef>) -> (App, std::sync::Arc<OnePick>) {
        let root = PathBuf::from("/nonexistent");
        let platform = std::sync::Arc::new(OnePick {
            paths: ftgsxr_io::AppPaths {
                config: root.clone(),
                cache: root.clone(),
                models: root.clone(),
                scenes: root.clone(),
                screenshots: root.clone(),
                logs: root,
            },
            answer,
            opened: Default::default(),
        });
        let mut library = Library::default();
        library.entries.push(crate::library::LibraryEntry {
            source: 0,
            entry: ftgsxr_io::Entry {
                name: "moved".into(),
                kind: ftgsxr_io::EntryKind::Exchange,
                location: "/m/moved.safetensors".into(),
                size: None,
                mtime: None,
                sha256: None,
                thumbnail: None,
                audio: None,
                scene: None,
            },
            info: None,
            identity: Some("ab".repeat(32)),
            thumbnail: None,
            state: EntryState::Missing,
        });
        let mut app = App::new();
        app.add_message::<ReplaceRequest>()
            .init_resource::<ReplacePicks>()
            .init_resource::<UiState>()
            .insert_resource(library)
            .insert_resource(PlatformRes(platform.clone()))
            .add_systems(Update, replace_picks);
        (app, platform)
    }

    // Q-256, Q-257: "Replace with…" opens the files picker; its first file is the question.
    #[test]
    fn replace_picks_open_the_files_picker_then_ask() {
        let file = ModelRef::File(PathBuf::from("/m/retrained.safetensors"));
        let other = ModelRef::File(PathBuf::from("/m/other.safetensors"));
        let (mut app, platform) = replace_app(vec![file.clone(), other]);
        app.world_mut().write_message(ReplaceRequest(0));
        app.update();
        assert_eq!(*platform.opened.lock().unwrap(), [PickKind::Files]);
        assert_eq!(
            app.world().resource::<UiState>().confirm,
            Some(Confirm::Replace {
                identity: "ab".repeat(32),
                model: "moved".into(),
                file,
            })
        );
    }

    // Q-256: an empty pick asks nothing.
    #[test]
    fn replace_picks_ask_nothing_for_an_empty_pick() {
        let (mut app, _) = replace_app(Vec::new());
        app.world_mut().write_message(ReplaceRequest(0));
        app.update();
        app.update();
        assert_eq!(app.world().resource::<UiState>().confirm, None);
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
        let ev = pointer_events(None, Some(p), &[true], 0.0);
        assert_eq!(ev[0], egui::Event::PointerMoved(egui::pos2(3.0, 4.0)));
        assert!(matches!(
            ev[1],
            egui::Event::PointerButton { pressed: true, .. }
        ));
        assert_eq!(
            pointer_events(Some(p), None, &[], 0.0),
            vec![egui::Event::PointerGone]
        );
        assert!(pointer_events(None, None, &[true], 0.0).is_empty());
    }

    /// A panel pixel is a point divided by the context's pixels per point (its zoom),
    /// one to one at zoom 1 or a bad zoom (Q-290).
    // Q-290
    #[test]
    fn panel_point_divides_by_the_pixels_per_point() {
        let p = Vec2::new(700.0, 350.0);
        assert_eq!(panel_point(p, 1.75), Vec2::new(400.0, 200.0));
        assert_eq!(panel_point(p, 1.0), p);
        assert_eq!(panel_point(p, 0.0), p);
        assert_eq!(panel_point(p, f32::NAN), p);
    }

    /// A panel scroll is a wheel event in points after the move, egui's sign kept
    /// (Q-279); no scroll, or one off the quad, adds none.
    // Q-279
    #[test]
    fn pointer_events_scroll_as_a_wheel_event() {
        let p = Vec2::new(3.0, 4.0);
        let ev = pointer_events(None, Some(p), &[], 12.5);
        assert_eq!(ev.len(), 2);
        assert!(matches!(
            ev[1],
            egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta,
                ..
            } if delta == egui::vec2(0.0, 12.5)
        ));
        assert_eq!(pointer_events(None, Some(p), &[], 0.0).len(), 1);
        assert!(pointer_events(None, None, &[], 12.5).is_empty());
    }

    /// The context events `panel_pointer` wrote, in order.
    #[derive(Resource, Default)]
    struct Written(Vec<(Entity, egui::Event)>);

    fn written(mut r: MessageReader<EguiInputEvent>, mut w: ResMut<Written>) {
        w.0.extend(r.read().map(|e| (e.context, e.event.clone())));
    }

    /// Q-334 (#563, `cpu`): `panel_pointer` with two image-target contexts, the main
    /// panel's and the wrist menu's: each context follows its own pointer hand. With
    /// both rays on the main panel it gets the first ray's move only; the right hand's
    /// press there moves its pointer to the right's hit before the press; the left ray
    /// moving to the wrist menu points there while the right's release reaches the main
    /// panel; with no ray left, each context sees the pointer leave, once.
    #[test]
    fn panel_pointer_each_context_follows_its_pointer_hand() {
        use bevy::asset::uuid::Uuid;
        let mut app = App::new();
        app.add_message::<PanelPointer>()
            .add_message::<EguiInputEvent>()
            .init_resource::<PanelPointerState>()
            .init_resource::<Written>()
            .add_systems(Update, (panel_pointer, written).chain());
        let mut context = |n: u128| {
            let image: Handle<Image> = Handle::Uuid(Uuid::from_u128(n), Default::default());
            let context = app
                .world_mut()
                .spawn((
                    RenderTarget::Image(image.clone().into()),
                    EguiContext::default(),
                ))
                .id();
            let panel = app.world_mut().spawn(WorldPanel { image }).id();
            (context, panel)
        };
        let (main, main_panel) = context(1);
        let (wrist, wrist_panel) = context(2);
        let (a, b, c) = (
            Vec2::new(100.0, 50.0),
            Vec2::new(400.0, 300.0),
            Vec2::new(20.0, 30.0),
        );
        let pos = |p: Vec2| egui::pos2(p.x, p.y);
        let button = |p: Vec2, pressed| egui::Event::PointerButton {
            pos: pos(p),
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        // Each ray's hand, panel, hit pixel and select changes.
        type Ray = (Hand, Entity, Vec2, Vec<bool>);
        let frames: [Vec<Ray>; 5] = [
            vec![
                (Hand::Left, main_panel, a, vec![]),
                (Hand::Right, main_panel, b, vec![]),
            ],
            vec![
                (Hand::Left, main_panel, a, vec![]),
                (Hand::Right, main_panel, b, vec![true]),
            ],
            vec![
                (Hand::Left, wrist_panel, c, vec![]),
                (Hand::Right, main_panel, b, vec![false]),
            ],
            vec![],
            vec![],
        ];
        let want: [Vec<(Entity, egui::Event)>; 5] = [
            vec![(main, egui::Event::PointerMoved(pos(a)))],
            vec![
                (main, egui::Event::PointerMoved(pos(a))),
                (main, egui::Event::PointerMoved(pos(b))),
                (main, button(b, true)),
            ],
            vec![
                (main, egui::Event::PointerMoved(pos(b))),
                (main, button(b, false)),
                (wrist, egui::Event::PointerMoved(pos(c))),
            ],
            vec![
                (main, egui::Event::PointerGone),
                (wrist, egui::Event::PointerGone),
            ],
            vec![],
        ];
        for (i, (frame, want)) in frames.into_iter().zip(want).enumerate() {
            for (hand, panel, pixel, select) in frame {
                app.world_mut().write_message(PanelPointer {
                    hand,
                    panel,
                    pixel,
                    select,
                    grab: vec![],
                    scroll_px: 0.0,
                });
            }
            app.update();
            let mut got = std::mem::take(&mut app.world_mut().resource_mut::<Written>().0);
            // The contexts' order within a frame is the system's own.
            got.sort_by_key(|(e, _)| *e != main);
            assert_eq!(got, want, "frame {i}");
        }
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
            let events = pointer_events(last, Some(hit), &select, 0.0);
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
