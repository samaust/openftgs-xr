// SPDX-License-Identifier: Apache-2.0

//! The desktop's keyboard and mouse to intents (§2.1 `ftgsxr_app::desktop_input`,
//! §1.7, Appendix D.1): every D.1 row becomes its `Intent` in `PreUpdate`, the mouse ray
//! from the active camera through the cursor is `Intent::Pointer { hand: Mouse }` every
//! frame, and `KeyMap` holds D.1's bindings overridden by `Settings.desktop.keys`
//! (Q-152, Q-197), rebuilt when the settings change. Keys and buttons egui wants stay
//! with the UI. Quit is the window's close action, which Bevy's `WindowPlugin` already
//! turns into the app's exit; nothing is built for it here.

use std::collections::BTreeMap;

use bevy::camera::RenderTarget;
use bevy::input::InputSystems;
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit};
use bevy::prelude::*;
use bevy::reflect::FromReflect;
use bevy::reflect::enums::{DynamicEnum, DynamicVariant};
use bevy::window::PrimaryWindow;
use bevy_egui::input::EguiWantsInput;
use ftgsxr_render::{Colormap, RenderMode, SplatMode};

use crate::intent::{
    CameraCommand, GizmoMode, Hand, InstanceAction, Intent, OverlayToggle, PlaybackCommand,
    UiCommand, UnloadRequest,
};
use crate::manipulate::{Selected, head_candidate};
use crate::persist::SettingsRes;
use crate::playback::{Playback, speed_preset};
use crate::rig::CameraController;
use crate::ui::{DockHooks, Tab};

/// Appendix D.1's actions under Q-152's names, with their default keys (Q-197's
/// spelling: winit's `KeyCode` name with `Ctrl+`, `Shift+`, `Alt+` prefixes in that
/// order; `Digitn` stands for the digits 1 to 9 of the bookmark rows).
// Q-152, Q-197
const DEFAULTS: &[(&str, &str)] = &[
    ("move_forward", "KeyW"),
    ("move_left", "KeyA"),
    ("move_back", "KeyS"),
    ("move_right", "KeyD"),
    ("move_up", "KeyE"),
    ("move_down", "KeyQ"),
    ("toggle_fly_orbit", "KeyO"),
    ("frame_selected", "KeyF"),
    ("reset_view", "Home"),
    ("save_bookmark_n", "Ctrl+Digitn"),
    ("recall_bookmark_n", "Alt+Digitn"),
    ("deselect", "Escape"),
    ("gizmo_translate", "KeyG"),
    ("gizmo_rotate", "KeyR"),
    ("gizmo_scale", "KeyT"),
    ("nudge_left", "ArrowLeft"),
    ("nudge_right", "ArrowRight"),
    ("nudge_forward", "ArrowUp"),
    ("nudge_back", "ArrowDown"),
    ("nudge_up", "PageUp"),
    ("nudge_down", "PageDown"),
    ("turn_left", "BracketLeft"),
    ("turn_right", "BracketRight"),
    ("fit", "Ctrl+KeyF"),
    ("save_pose", "Ctrl+KeyP"),
    ("reset_pose", "Ctrl+Backspace"),
    ("unload", "Delete"),
    ("open_files", "Ctrl+KeyO"),
    ("save_scene", "Ctrl+Shift+KeyS"),
    ("load_scene", "Ctrl+KeyL"),
    ("play_pause", "Space"),
    ("step_back", "Comma"),
    ("step_forward", "Period"),
    ("speed_down", "Shift+Comma"),
    ("speed_up", "Shift+Period"),
    ("cycle_loop", "KeyL"),
    ("clock_start", "Ctrl+Home"),
    ("clock_end", "Ctrl+End"),
    ("cycle_mode", "KeyM"),
    ("toggle_boxes", "KeyB"),
    ("toggle_velocities", "KeyV"),
    ("help", "F1"),
    ("hide_ui", "F2"),
    ("hud", "F3"),
    ("fullscreen", "F11"),
    ("screenshot", "F12"),
];

/// The action names of the bookmark rows, whose key is `Digitn`.
const BOOKMARKS: [&str; 2] = ["save_bookmark_n", "recall_bookmark_n"];

/// The nudge in metres, ×10 with Shift, and the turn in degrees (§1.7).
const NUDGE_M: f32 = 0.01;
const TURN_DEG: f32 = 5.0;

/// Q-198: a double click is a second left press within this time and distance.
// Q-198
const DOUBLE_CLICK_S: f64 = 0.4;
const DOUBLE_CLICK_PX: f32 = 4.0;
/// Q-198: the pixels of a pixel-unit wheel event that make one step.
// Q-198
const WHEEL_PX_PER_STEP: f32 = 100.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
struct Mods {
    ctrl: bool,
    shift: bool,
    alt: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Key {
    Code(KeyCode),
    /// The digits 1 to 9 (the bookmark rows).
    Digits,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Chord {
    mods: Mods,
    key: Key,
}

impl Chord {
    fn name(&self) -> String {
        let mut s = String::new();
        for (on, p) in [
            (self.mods.ctrl, "Ctrl+"),
            (self.mods.shift, "Shift+"),
            (self.mods.alt, "Alt+"),
        ] {
            if on {
                s.push_str(p);
            }
        }
        match self.key {
            Key::Code(k) => s.push_str(&format!("{k:?}")),
            Key::Digits => s.push_str("Digitn"),
        }
        s
    }
}

/// `name` as a unit variant of `KeyCode` (winit's name), if it is one.
fn key_code(name: &str) -> Option<KeyCode> {
    let k = KeyCode::from_reflect(&DynamicEnum::new(name, DynamicVariant::Unit))?;
    (format!("{k:?}") == name).then_some(k)
}

fn parse(action: &str, text: &str) -> Result<Chord, String> {
    let mut parts: Vec<&str> = text.split('+').collect();
    let key = parts.pop().unwrap_or_default();
    let mut mods = Mods::default();
    for p in parts {
        let slot = match p {
            "Ctrl" => &mut mods.ctrl,
            "Shift" => &mut mods.shift,
            "Alt" => &mut mods.alt,
            _ => return Err(format!("unknown modifier {p:?}")),
        };
        if *slot {
            return Err(format!("modifier {p:?} twice"));
        }
        *slot = true;
    }
    let bookmark = BOOKMARKS.contains(&action);
    let key = match (key, bookmark) {
        ("Digitn", true) => Key::Digits,
        (_, true) => return Err("a bookmark key is Digitn".into()),
        (k, false) => Key::Code(key_code(k).ok_or_else(|| format!("unknown key {k:?}"))?),
    };
    Ok(Chord { mods, key })
}

/// The desktop's key bindings (§2.1, Appendix D.1): action → key, D.1's defaults
/// overridden by `Settings.desktop.keys` (Q-152). The mouse rows are not rebindable.
// Q-197
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct KeyMap {
    chords: BTreeMap<&'static str, Chord>,
}

impl Default for KeyMap {
    fn default() -> Self {
        KeyMap {
            chords: DEFAULTS
                .iter()
                .map(|&(a, k)| (a, parse(a, k).expect("D.1's defaults parse")))
                .collect(),
        }
    }
}

impl KeyMap {
    /// The action names, in Appendix D.1's order.
    pub fn actions() -> Vec<&'static str> {
        DEFAULTS.iter().map(|(a, _)| *a).collect()
    }

    /// D.1's defaults overridden by `keys` (action name → key name), and a warning for
    /// each entry that names an unknown action or key, which keeps the default.
    pub fn from_keys(keys: &BTreeMap<String, String>) -> (KeyMap, Vec<String>) {
        let mut map = KeyMap::default();
        let mut warnings = Vec::new();
        for (action, text) in keys {
            let Some(&(name, _)) = DEFAULTS.iter().find(|(a, _)| a == action) else {
                warnings.push(format!("[desktop] keys: unknown action {action:?}"));
                continue;
            };
            match parse(name, text) {
                Ok(c) => {
                    map.chords.insert(name, c);
                }
                Err(e) => warnings.push(format!("[desktop] keys: {action} = {text:?}: {e}")),
            }
        }
        (map, warnings)
    }

    /// The key bound to `action`, spelled as `Settings.desktop.keys` spells it.
    pub fn key(&self, action: &str) -> Option<String> {
        self.chords.get(action).map(Chord::name)
    }

    fn chord(&self, action: &str) -> Chord {
        self.chords[action]
    }
}

/// Appendix D.1 → `Intent` (§2.1).
pub struct DesktopInputPlugin;

impl Plugin for DesktopInputPlugin {
    fn build(&self, app: &mut App) {
        // §1.7 Help: the shortcuts, drawn in ui's Help tab.
        app.init_resource::<DockHooks>()
            .world_mut()
            .resource_mut::<DockHooks>()
            .add_screen_only(Tab::Help, help_keys);
        app.add_message::<Intent>()
            .add_message::<UnloadRequest>()
            .init_resource::<KeyMap>()
            .init_resource::<Clicks>()
            .add_systems(
                PreUpdate,
                (
                    rebuild_keymap,
                    keyboard.run_if(resource_exists::<ButtonInput<KeyCode>>),
                    mouse.run_if(resource_exists::<ButtonInput<MouseButton>>),
                )
                    .chain()
                    .after(InputSystems),
            );
    }
}

fn rebuild_keymap(settings: Option<Res<SettingsRes>>, mut map: ResMut<KeyMap>) {
    let Some(s) = settings.filter(|s| s.is_changed()) else {
        return;
    };
    let (m, warnings) = KeyMap::from_keys(&s.0.desktop.keys);
    for w in warnings {
        warn!("{w}");
    }
    if *map != m {
        *map = m;
    }
}

fn held_mods(keys: &ButtonInput<KeyCode>) -> Mods {
    Mods {
        ctrl: keys.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]),
        shift: keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]),
        alt: keys.any_pressed([KeyCode::AltLeft, KeyCode::AltRight]),
    }
}

/// Whether `held` holds `c`'s modifiers, with only the `loose` ones besides.
fn mods_match(c: Mods, held: Mods, loose: Mods) -> bool {
    let ok = |want: bool, got: bool, extra: bool| (want && got) || (!want && (!got || extra));
    ok(c.ctrl, held.ctrl, loose.ctrl)
        && ok(c.shift, held.shift, loose.shift)
        && ok(c.alt, held.alt, loose.alt)
}

const STRICT: Mods = Mods {
    ctrl: false,
    shift: false,
    alt: false,
};

/// The next render mode of M's cycle: splats, centers, both; hidden goes to centers.
// Q-198
fn next_mode(m: RenderMode, pixel_radius: f32, colormap: Colormap) -> RenderMode {
    match m {
        RenderMode::Splats | RenderMode::Hidden => RenderMode::Centers {
            pixel_radius,
            colormap,
        },
        RenderMode::Centers { .. } => RenderMode::Both {
            pixel_radius,
            colormap,
        },
        RenderMode::Both { .. } => RenderMode::Splats,
    }
}

/// Q-50's colormap spellings; an unknown one is the default DC.
fn colormap(s: &str) -> Colormap {
    match s {
        "time" => Colormap::TimeCenter,
        "speed" => Colormap::Speed,
        "opacity" => Colormap::Opacity,
        _ => Colormap::Dc,
    }
}

#[allow(clippy::too_many_arguments)]
fn keyboard(
    keys: Res<ButtonInput<KeyCode>>,
    map: Res<KeyMap>,
    egui: Option<Res<EguiWantsInput>>,
    selected: Query<(Entity, Option<&SplatMode>), With<Selected>>,
    playback: Option<Res<Playback>>,
    settings: Option<Res<SettingsRes>>,
    mut out: MessageWriter<Intent>,
    mut unload: MessageWriter<UnloadRequest>,
) {
    // Keys egui's text fields consume stay with the UI.
    if egui.is_some_and(|e| e.wants_keyboard_input()) {
        return;
    }
    let held = held_mods(&keys);
    let fired = |action: &str, loose: Mods| -> Option<u8> {
        let c = map.chord(action);
        if !mods_match(c.mods, held, loose) {
            return None;
        }
        match c.key {
            Key::Code(k) => keys.just_pressed(k).then_some(0),
            Key::Digits => (1..=9u8)
                .find(|n| key_code(&format!("Digit{n}")).is_some_and(|k| keys.just_pressed(k))),
        }
    };
    let pressed = |action: &str| -> bool {
        let c = map.chord(action);
        let loose = Mods {
            ctrl: true,
            shift: true,
            alt: false,
        };
        // Q-197: a held key whose chord is another action's binding (Ctrl+Shift+S saves
        // the scene) does not also move.
        let taken = map.chords.iter().any(|(b, x)| {
            *b != action
                && x.key == c.key
                && x.mods != Mods::default()
                && mods_match(x.mods, held, Mods::default())
        });
        !taken
            && matches!(c.key, Key::Code(k) if keys.pressed(k))
            && mods_match(c.mods, held, loose)
    };

    // Fly: held keys, Shift × 4, Ctrl × 0.25 (both: × 1).
    let mut v = Vec3::ZERO;
    for (a, d) in [
        ("move_forward", Vec3::NEG_Z),
        ("move_left", Vec3::NEG_X),
        ("move_back", Vec3::Z),
        ("move_right", Vec3::X),
        ("move_up", Vec3::Y),
        ("move_down", Vec3::NEG_Y),
    ] {
        if pressed(a) {
            v += d;
        }
    }
    if v != Vec3::ZERO {
        let mut k = 1.0;
        if held.shift {
            k *= 4.0;
        }
        if held.ctrl {
            k *= 0.25;
        }
        out.write(Intent::Camera(CameraCommand::Move(v * k)));
    }

    let cam = |c| Intent::Camera(c);
    let simple: [(&str, Intent); 22] = [
        ("toggle_fly_orbit", cam(CameraCommand::ToggleOrbit)),
        ("frame_selected", cam(CameraCommand::Frame)),
        ("reset_view", cam(CameraCommand::ResetView)),
        ("gizmo_translate", Intent::Gizmo(GizmoMode::Translate)),
        ("gizmo_rotate", Intent::Gizmo(GizmoMode::Rotate)),
        ("gizmo_scale", Intent::Gizmo(GizmoMode::Scale)),
        ("open_files", Intent::Ui(UiCommand::OpenFiles)),
        ("save_scene", Intent::Ui(UiCommand::SaveScene)),
        ("load_scene", Intent::Ui(UiCommand::LoadScene)),
        ("play_pause", Intent::Playback(PlaybackCommand::Toggle)),
        (
            "step_back",
            Intent::Playback(PlaybackCommand::StepFrames(-1)),
        ),
        (
            "step_forward",
            Intent::Playback(PlaybackCommand::StepFrames(1)),
        ),
        ("cycle_loop", Intent::Ui(UiCommand::CycleLoop)),
        ("clock_start", Intent::Playback(PlaybackCommand::Seek(0.0))),
        ("toggle_boxes", Intent::Overlay(OverlayToggle::Boxes)),
        (
            "toggle_velocities",
            Intent::Overlay(OverlayToggle::Velocities),
        ),
        ("help", Intent::Ui(UiCommand::Help)),
        ("hide_ui", Intent::Ui(UiCommand::HideUi)),
        ("hud", Intent::Ui(UiCommand::Hud)),
        ("fullscreen", Intent::Ui(UiCommand::Fullscreen)),
        ("screenshot", Intent::Screenshot),
        (
            "clock_end",
            Intent::Playback(PlaybackCommand::Seek(f64::MAX)),
        ),
    ];
    for (a, i) in simple {
        if fired(a, STRICT).is_some() {
            let i = match i {
                // Ctrl+End: the clock's length, which `Clock::seek` clamps to anyway.
                Intent::Playback(PlaybackCommand::Seek(t)) if t == f64::MAX => {
                    let end = playback.as_ref().map_or(0.0, |p| p.clock.length());
                    Intent::Playback(PlaybackCommand::Seek(end))
                }
                i => i,
            };
            out.write(i);
        }
    }
    if let Some(n) = fired("save_bookmark_n", STRICT) {
        out.write(cam(CameraCommand::SaveBookmark(n)));
    }
    if let Some(n) = fired("recall_bookmark_n", STRICT) {
        out.write(cam(CameraCommand::RecallBookmark(n)));
    }
    let speed = playback.as_ref().map_or(1.0, |p| p.clock.speed());
    for (a, up) in [("speed_down", false), ("speed_up", true)] {
        if fired(a, STRICT).is_some() {
            out.write(Intent::Playback(PlaybackCommand::Speed(speed_preset(
                speed, up,
            ))));
        }
    }

    // The selected instance's rows.
    let Some((e, mode)) = selected.iter().next() else {
        return;
    };
    let act = |action| Intent::Instance {
        instance: e,
        action,
    };
    for (a, action) in [
        ("deselect", InstanceAction::Deselect),
        ("fit", InstanceAction::Fit),
        ("save_pose", InstanceAction::SavePose),
        ("reset_pose", InstanceAction::ResetPose),
    ] {
        if fired(a, STRICT).is_some() {
            out.write(act(action));
        }
    }
    // Q-199: Delete asks `ui` to confirm the unload.
    if fired("unload", STRICT).is_some() {
        unload.write(UnloadRequest(e));
    }
    if fired("cycle_mode", STRICT).is_some() {
        let d = settings.map(|s| s.0.display.clone()).unwrap_or_default();
        let m = mode.map_or(RenderMode::Splats, |m| m.0);
        let next = next_mode(m, d.centers_radius, colormap(&d.colormap));
        out.write(act(InstanceAction::SetMode(next)));
    }
    // Nudges in the stage's axes, ×10 with Shift (Q-198).
    let shift = Mods {
        ctrl: false,
        shift: true,
        alt: false,
    };
    let step = if held.shift { 10.0 * NUDGE_M } else { NUDGE_M };
    for (a, d) in [
        ("nudge_left", Vec3::NEG_X),
        ("nudge_right", Vec3::X),
        ("nudge_forward", Vec3::NEG_Z),
        ("nudge_back", Vec3::Z),
        ("nudge_up", Vec3::Y),
        ("nudge_down", Vec3::NEG_Y),
    ] {
        if fired(a, shift).is_some() {
            out.write(Intent::Nudge {
                instance: e,
                delta: d * step,
            });
        }
    }
    // `[` turns left (counter-clockwise seen from above, +y), `]` right.
    for (a, s) in [("turn_left", 1.0), ("turn_right", -1.0)] {
        if fired(a, STRICT).is_some() {
            out.write(Intent::Turn {
                instance: e,
                yaw_deg: s * TURN_DEG,
            });
        }
    }
}

/// The last left press, for the double click.
#[derive(Resource, Default)]
struct Clicks {
    last: Option<(f64, Vec2)>,
}

/// The ray through `cursor` (logical pixels from the top left) of a perspective camera
/// at `camera` with vertical field of view `fov` over a `size` viewport.
fn cursor_ray(camera: &GlobalTransform, fov: f32, size: Vec2, cursor: Vec2) -> (Vec3, Vec3) {
    let ndc = Vec2::new(2.0 * cursor.x / size.x - 1.0, 1.0 - 2.0 * cursor.y / size.y);
    let t = (fov * 0.5).tan();
    let view = Vec3::new(ndc.x * t * size.x / size.y, ndc.y * t, -1.0);
    let dir = camera.affine().transform_vector3(view).normalize();
    (camera.translation(), dir)
}

#[allow(clippy::too_many_arguments)]
fn mouse(
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Option<Res<AccumulatedMouseMotion>>,
    scroll: Option<Res<AccumulatedMouseScroll>>,
    egui: Option<Res<EguiWantsInput>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &RenderTarget, &GlobalTransform, &Projection)>,
    ctl: Option<Res<CameraController>>,
    time: Res<Time<Real>>,
    mut clicks: ResMut<Clicks>,
    mut out: MessageWriter<Intent>,
) {
    let orbit = ctl.is_some_and(|c| c.orbit);
    let ui = egui.is_some_and(|e| e.wants_pointer_input());
    let cursor = windows.single().ok().and_then(|w| {
        let c = w.cursor_position()?;
        Some((c, Vec2::new(w.width(), w.height())))
    });
    // The ray's camera (Q-198): the lowest-order active perspective camera among those
    // that may be the head (Q-180, Q-310).
    // Q-198, Q-310
    let camera = cameras
        .iter()
        .filter(|(c, t, ..)| head_candidate(c, t))
        .filter_map(|(c, _, g, p)| match p {
            Projection::Perspective(p) => Some((c.order, *g, p.fov)),
            _ => None,
        })
        .min_by_key(|(o, ..)| *o);
    if let (Some((c, size)), Some((_, g, fov))) = (cursor, camera)
        && size.x > 0.0
        && size.y > 0.0
    {
        let (origin, dir) = cursor_ray(&g, fov, size, c);
        out.write(Intent::Pointer {
            hand: Hand::Mouse,
            origin,
            dir,
        });
    }
    if buttons.just_released(MouseButton::Left) {
        out.write(Intent::Select {
            hand: Hand::Mouse,
            pressed: false,
        });
    }
    if ui {
        return;
    }
    if buttons.just_pressed(MouseButton::Left) {
        out.write(Intent::Select {
            hand: Hand::Mouse,
            pressed: true,
        });
        let now = time.elapsed_secs_f64();
        let at = cursor.map_or(Vec2::ZERO, |(c, _)| c);
        let double = clicks
            .last
            .is_some_and(|(t, p)| now - t <= DOUBLE_CLICK_S && p.distance(at) <= DOUBLE_CLICK_PX);
        if double {
            out.write(Intent::Camera(CameraCommand::Frame));
            clicks.last = None;
        } else {
            clicks.last = Some((now, at));
        }
    }
    let d = motion.map_or(Vec2::ZERO, |m| m.delta);
    if d != Vec2::ZERO {
        if !orbit && buttons.pressed(MouseButton::Right) {
            out.write(Intent::Camera(CameraCommand::Look(d)));
        }
        if orbit && buttons.pressed(MouseButton::Left) {
            out.write(Intent::Camera(CameraCommand::Orbit(d)));
        }
        if orbit && buttons.pressed(MouseButton::Middle) {
            out.write(Intent::Camera(CameraCommand::Pan(d)));
        }
    }
    if let Some(s) = scroll
        && s.delta.y != 0.0
    {
        let steps = match s.unit {
            MouseScrollUnit::Line => s.delta.y,
            MouseScrollUnit::Pixel => s.delta.y / WHEEL_PX_PER_STEP,
        };
        out.write(Intent::Camera(if orbit {
            CameraCommand::Zoom(steps)
        } else {
            CameraCommand::Speed(steps)
        }));
    }
}

/// The Help tab's shortcut table (§1.7, Appendix D.1): each action and its key.
fn help_keys(world: &mut World, ui: &mut bevy_egui::egui::Ui) {
    let Some(keys) = world.get_resource::<KeyMap>() else {
        return;
    };
    bevy_egui::egui::Grid::new("ftgsxr-keys").show(ui, |ui| {
        for a in KeyMap::actions() {
            ui.label(a);
            ui.label(keys.key(a).unwrap_or_default());
            ui.end_row();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mods_match_is_exact_but_for_the_loose_ones() {
        let shift = Mods {
            shift: true,
            ..STRICT
        };
        assert!(mods_match(STRICT, STRICT, STRICT));
        assert!(!mods_match(STRICT, shift, STRICT));
        assert!(mods_match(STRICT, shift, shift));
        assert!(!mods_match(shift, STRICT, shift));
    }

    #[test]
    fn next_mode_cycles_splats_centers_both() {
        let c = |m| next_mode(m, 3.0, Colormap::Dc);
        let centers = RenderMode::Centers {
            pixel_radius: 3.0,
            colormap: Colormap::Dc,
        };
        assert_eq!(c(RenderMode::Splats), centers);
        assert!(matches!(c(centers), RenderMode::Both { .. }));
        assert_eq!(c(c(centers)), RenderMode::Splats);
        assert_eq!(c(RenderMode::Hidden), centers);
    }

    #[test]
    fn cursor_ray_through_the_centre_is_the_view_axis() {
        let g = GlobalTransform::from(Transform::from_xyz(0.0, 1.0, 2.0));
        let (o, d) = cursor_ray(&g, 1.0, Vec2::new(200.0, 100.0), Vec2::new(100.0, 50.0));
        assert_eq!(o, Vec3::new(0.0, 1.0, 2.0));
        assert!((d - Vec3::NEG_Z).length() < 1e-6);
    }

    /// Q-198 read through Q-180 and Q-310 (#535): the mouse ray comes from the
    /// lowest-order active perspective camera that may be the head, so an active
    /// perspective camera rendering to an image at a lower order than the window
    /// camera is passed over, and the ray starts at the window camera.
    #[test]
    fn mouse_ray_skips_an_image_target_camera_of_a_lower_order() {
        use bevy::camera::ImageRenderTarget;

        let mut app = App::new();
        app.add_message::<Intent>()
            .init_resource::<ButtonInput<MouseButton>>()
            .init_resource::<Time<Real>>()
            .init_resource::<Clicks>()
            .add_systems(Update, mouse);
        let mut window = Window::default();
        window.set_cursor_position(Some(Vec2::new(10.0, 10.0)));
        app.world_mut().spawn((window, PrimaryWindow));
        let perspective = || Projection::Perspective(PerspectiveProjection::default());
        app.world_mut().spawn((
            Camera {
                order: -1,
                ..Default::default()
            },
            RenderTarget::Image(ImageRenderTarget::from(Handle::<Image>::default())),
            perspective(),
            GlobalTransform::from_xyz(5.0, 0.0, 0.0),
        ));
        app.world_mut().spawn((
            Camera::default(),
            RenderTarget::Window(Default::default()),
            perspective(),
            GlobalTransform::from_xyz(0.0, 1.6, 2.0),
        ));
        app.update();
        let messages = app.world().resource::<Messages<Intent>>();
        let origins: Vec<Vec3> = messages
            .get_cursor()
            .read(messages)
            .filter_map(|i| match i {
                Intent::Pointer {
                    hand: Hand::Mouse,
                    origin,
                    ..
                } => Some(*origin),
                _ => None,
            })
            .collect();
        assert_eq!(origins, vec![Vec3::new(0.0, 1.6, 2.0)]);
    }
}
