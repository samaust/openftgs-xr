// SPDX-License-Identifier: Apache-2.0

//! `xr_input`'s systems (§1.8 Rays, Controllers, Locomotion, User interface; PRD #452):
//! `hands_to_intents` each frame from `XrHands`, with each hand's ray tested against the
//! main panel and the wrist menu first (Q-263: a hand on a quad writes `PanelPointer` and
//! no `Pointer`, `Select` or grab of its own); the teleport arc of the `move` stick's
//! hand (Q-269); the rays drawn to the first hit (Q-278's high-contrast colors); and
//! D-44's haptic pulses from what the frame changed (Q-270).

use bevy::app::{App, Plugin, PostUpdate, PreUpdate};
use bevy::asset::Assets;
use bevy::color::Color;
use bevy::ecs::message::{MessageReader, MessageWriter};
use bevy::gizmos::config::GizmoConfigStore;
use bevy::gizmos::gizmos::Gizmos;
use bevy::math::{Isometry3d, UVec2, Vec2, Vec3};
use bevy::prelude::{
    ChildOf, Component, Entity, GlobalTransform, InheritedVisibility, IntoScheduleConfigs, Query,
    Res, ResMut, Resource, Transform, With, Without, resource_exists,
};
use bevy::time::Time;
use bevy_mod_openxr::openxr_session_running;
use bevy_mod_openxr::session::OxrSession;
use ftgsxr_app::{
    Grabbed, Hand, Hovered, Intent, LoadedModel, Locomotion, ManipulateSettings, PanelPointer,
    PlayUiSound, Selected, SettingsRes, UiSound, panel_hit, pick_instance, stick_response,
};
use ftgsxr_core::{Aabb, Pose, ray_hit};
use ftgsxr_render::{Colormap, RenderMode, SplatAsset, SplatMode, SplatModel};

use super::{SIDES, XrInputConfig, XrInputState, aim_ray, hands_to_intents};
use crate::XrError;
use crate::error::log_warning;
use crate::tracking::{Actions, GRAB, HOVER, HandSource, HapticLog, Pulse, TICK, XrHands};

/// A panel quad's size for the ray test (Q-265): declared here, inserted by `panel` on
/// the main panel and the wrist menu, each with `WorldPanel` and a `GlobalTransform`
/// whose translation and rotation are the quad's pose (Q-210).
// Q-265
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub(crate) struct PanelQuad {
    /// The quad's width and height in metres.
    pub(crate) size_m: Vec2,
    /// Its texture's size in pixels.
    pub(crate) size_px: UVec2,
    /// Whether its texture's top `TITLE_FRACTION` is a title bar (the main panel's,
    /// Q-306), where a grab press is the panel's and no click (Q-315).
    pub(crate) title_bar: bool,
}

/// The title bar: the top strip of a quad's texture, as a fraction of its height (48 px
/// of 768), drawn by `ui` as a bar above the tab row, its own copy of the fraction;
/// `panel` grabs by it.
// Q-306
pub(crate) const TITLE_FRACTION: f32 = 1.0 / 16.0;

/// The ray's length without a hit (§1.8 Rays, E10's t\_max).
const RAY_M: f32 = 10.0;
/// The dot at the hit: 1 cm across (PRD #452's geometry).
// Q-316
const DOT_RADIUS_M: f32 = 0.005;
/// The stick's scroll on a panel at full deflection, in pixels per second (Q-304).
// Q-304
const SCROLL_PX_PER_S: f32 = 600.0;
/// The teleport arc's launch speed, gravity and samples (Q-269).
// Q-269
const ARC_SPEED_MPS: f32 = 5.0;
const ARC_GRAVITY: f32 = 9.8;
const ARC_SAMPLES: usize = 64;
/// The move stick's forward value that shows the arc, and the release (Q-269).
// Q-269
const ARC_SHOW: f32 = 0.7;
const ARC_RELEASE: f32 = 0.3;

/// The module's plugin; `XrTail` adds it after `tracking`'s.
pub(crate) struct XrInputPlugin;

impl Plugin for XrInputPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<Intent>()
            .add_message::<PanelPointer>()
            .init_resource::<XrHands>()
            .init_resource::<InputState>()
            .init_resource::<Frame>()
            .init_resource::<HapticMemory>()
            .init_resource::<HapticLog>()
            .add_message::<PanelGrabEdge>()
            .add_message::<PlayUiSound>()
            .add_systems(PreUpdate, write_intents)
            .add_systems(
                PostUpdate,
                (
                    (haptics, apply_pulses.run_if(openxr_session_running)).chain(),
                    draw.run_if(resource_exists::<GizmoConfigStore>),
                ),
            );
    }
}

/// `hands_to_intents`'s state and the teleport's.
#[derive(Resource, Default)]
struct InputState {
    input: XrInputState,
    /// Per side, a select press withheld from a title bar's grab, whose release is
    /// withheld too (Q-315).
    withheld: [bool; 2],
    /// The arc's end while the `move` stick is held forward (Q-269), `None` while not
    /// aiming; `Some(None)` aiming at nothing.
    aiming: Option<Option<Vec3>>,
}

/// What `write_intents` saw this frame, for the drawing: each side's ray, its hit on a
/// panel quad, and the arc.
#[derive(Resource, Default)]
struct Frame {
    rays: [Option<RayDraw>; 2],
    arc: Vec<Vec3>,
}

#[derive(Clone, Copy, Debug)]
struct RayDraw {
    origin: Vec3,
    dir: Vec3,
    /// The distance to the nearest panel quad's hit.
    panel: Option<f32>,
}

/// `XrInputConfig` from D.2's and D.3's thresholds, D-44's dead zone and `Locomotion`
/// (`snap`, `enabled`; its default without the resource).
fn config(loco: Option<&Locomotion>) -> XrInputConfig {
    let d = Locomotion::default();
    let loco = loco.unwrap_or(&d);
    XrInputConfig {
        snap: loco.snap,
        locomotion_enabled: loco.enabled,
        ..XrInputConfig::default()
    }
}

/// The next render mode of Q-198's cycle (splats, centers, both; hidden goes to
/// centers), as the desktop's M, with `[display]`'s radius and Q-50's colormap names.
// Q-198, Q-302
fn next_mode(m: RenderMode, pixel_radius: f32, colormap: &str) -> RenderMode {
    let colormap = match colormap {
        "time" => Colormap::TimeCenter,
        "speed" => Colormap::Speed,
        "opacity" => Colormap::Opacity,
        _ => Colormap::Dc,
    };
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

/// The nearest front hit of a ray on the visible quads: (quad, pixel, distance).
pub(crate) fn nearest_panel(
    origin: Vec3,
    dir: Vec3,
    quads: &[(Entity, Pose, PanelQuad)],
) -> Option<(Entity, Vec2, f32)> {
    let mut best: Option<(Entity, Vec2, f32)> = None;
    for (e, pose, q) in quads {
        let Some(pixel) = panel_hit(origin, dir, *pose, q.size_m, q.size_px) else {
            continue;
        };
        let n = pose.rotation * Vec3::Z;
        let t = (pose.translation - origin).dot(n) / dir.dot(n);
        if best.is_none_or(|(_, _, b)| t < b) {
            best = Some((*e, pixel, t));
        }
    }
    best
}

/// The teleport arc from an aim pose (Q-269): a ballistic curve launched along the aim's
/// −z at 5 m/s under 9.8 m/s², sampled at 64 points from the aim's position to where it
/// meets the floor y = 0, the last point the target; `None` from at or below the floor.
pub(crate) fn teleport_arc(aim: &Pose) -> Option<Vec<Vec3>> {
    let (p0, dir) = aim_ray(aim);
    if p0.y <= 0.0 || !p0.is_finite() || !dir.is_finite() {
        return None;
    }
    let v = dir * ARC_SPEED_MPS;
    // y0 + vy t − g t²/2 = 0, the positive root.
    let t_hit = (v.y + (v.y * v.y + 2.0 * ARC_GRAVITY * p0.y).sqrt()) / ARC_GRAVITY;
    let at = |t: f32| p0 + v * t - Vec3::Y * (0.5 * ARC_GRAVITY * t * t);
    let mut points: Vec<Vec3> = (0..ARC_SAMPLES)
        .map(|k| at(t_hit * k as f32 / (ARC_SAMPLES - 1) as f32))
        .collect();
    if let Some(last) = points.last_mut() {
        last.y = 0.0;
    }
    Some(points)
}

/// Every instance's (entity, stage pose, B_m) for E10, as `manipulate` picks them: its
/// pose relative to the content group followed by the group's (§1.14), the scale from
/// x; an instance whose asset is not loaded has none.
fn instance_boxes(
    instances: &Query<(Entity, &Transform, &SplatModel, Option<&ChildOf>), With<LoadedModel>>,
    groups: &Query<&Transform, Without<LoadedModel>>,
    assets: Option<&Assets<SplatAsset>>,
) -> Vec<(Entity, Pose, Aabb)> {
    let Some(assets) = assets else {
        return Vec::new();
    };
    let pose = |t: &Transform| Pose {
        translation: t.translation,
        rotation: t.rotation,
        scale: t.scale.x,
    };
    instances
        .iter()
        .filter_map(|(e, t, m, parent)| {
            let group = parent
                .and_then(|c| groups.get(c.parent()).ok())
                .map_or(Pose::IDENTITY, pose);
            Some((e, pose(t).then(&group), assets.get(&m.0)?.bounds))
        })
        .collect()
}

/// The hands that hold an instance (a `Grabbed` names them): their rays stay with the
/// instances (Q-304).
fn holding(grabbed: &Query<&Grabbed>) -> Vec<Hand> {
    let mut out = Vec::new();
    for g in grabbed {
        match g {
            Grabbed::One { hand, .. } => out.push(*hand),
            Grabbed::Two { .. } => out.extend([Hand::Left, Hand::Right]),
        }
    }
    out
}

/// One frame of input: the arbitration (Q-263), `hands_to_intents`, the teleport
/// (Q-269), and the frame's record for the drawing.
// Q-263, Q-269, Q-303, Q-304
#[allow(clippy::too_many_arguments)]
fn write_intents(
    hands: Res<XrHands>,
    loco: Option<Res<Locomotion>>,
    settings: Option<Res<SettingsRes>>,
    time: Option<Res<Time>>,
    selected: Query<(Entity, Option<&SplatMode>), With<Selected>>,
    grabbed: Query<&Grabbed>,
    instances: Query<(Entity, &Transform, &SplatModel, Option<&ChildOf>), With<LoadedModel>>,
    groups: Query<&Transform, Without<LoadedModel>>,
    assets: Option<Res<Assets<SplatAsset>>>,
    manipulate: Option<Res<ManipulateSettings>>,
    quads: Query<(
        Entity,
        &PanelQuad,
        &GlobalTransform,
        Option<&InheritedVisibility>,
    )>,
    mut state: ResMut<InputState>,
    mut frame: ResMut<Frame>,
    mut intents: MessageWriter<Intent>,
    mut pointers: MessageWriter<PanelPointer>,
) {
    let cfg = config(loco.as_deref());
    let display = settings.map(|s| s.0.display.clone()).unwrap_or_default();
    let sel = selected.iter().next().map(|(e, m)| {
        let m = m.map_or(RenderMode::Splats, |m| m.0);
        (e, next_mode(m, display.centers_radius, &display.colormap))
    });
    state.input.set_selected(sel);
    let held = holding(&grabbed);
    let quads: Vec<(Entity, Pose, PanelQuad)> = quads
        .iter()
        .filter(|(.., v)| v.is_none_or(|v| v.get()))
        .map(|(e, q, g, _)| {
            let (_, rotation, translation) = g.to_scale_rotation_translation();
            (
                e,
                Pose {
                    translation,
                    rotation,
                    scale: 1.0,
                },
                *q,
            )
        })
        .collect();
    let dt = time.map_or(0.0, |t| t.delta_secs());
    let boxes = instance_boxes(&instances, &groups, assets.as_deref());
    let ray_max_m = manipulate.map_or(RAY_M, |m| m.ray_max_m);

    // The arbitration: a tracked hand that holds no instance and points at a quad nearer
    // than any instance (§1.8 Rays' first hit, Q-317).
    let mut now = *hands;
    let mut on_panel: [Option<(Entity, Vec2, f32)>; 2] = [None, None];
    let mut scroll = [0.0f32; 2];
    frame.rays = [None, None];
    for (i, side) in SIDES.into_iter().enumerate() {
        let slot = if i == 0 {
            &mut now.left
        } else {
            &mut now.right
        };
        let Some(t) = slot.as_mut() else { continue };
        if !t.tracked {
            continue;
        }
        let (origin, dir) = aim_ray(&t.aim);
        let hit = if held.contains(&side) {
            None
        } else {
            // Q-317 (#550): an instance whose box holds the ray's origin (E10's t_in ≤ 0,
            // `ray_hit` 0) is left out, so from inside a box a quad the ray hits wins.
            let outside: Vec<_> = boxes
                .iter()
                .filter(|(_, pose, b)| ray_hit(origin, dir, b, pose, ray_max_m) != Some(0.0))
                .cloned()
                .collect();
            nearest_panel(origin, dir, &quads).filter(|(_, _, d)| {
                pick_instance(origin, dir, &outside, ray_max_m).is_none_or(|(_, t)| *d <= t)
            })
        };
        frame.rays[i] = Some(RayDraw {
            origin,
            dir,
            panel: hit.map(|(_, _, d)| d),
        });
        if hit.is_some() {
            // Q-279: positive moves the content down; the stick pushed up shows what is
            // above (Q-304). The stick scrolls, and moves nothing, while on a quad.
            if t.source == HandSource::Controller {
                scroll[i] = stick_response(t.stick, cfg.stick_deadzone).y * SCROLL_PX_PER_S * dt;
            }
            t.stick = Vec2::ZERO;
        }
        on_panel[i] = hit;
    }

    let mut out = hands_to_intents(&mut state.input, &now, &cfg);

    // Q-263: the quad takes the hand's ray, select and grab presses.
    let mut presses: [(Vec<bool>, Vec<bool>); 2] = Default::default();
    out.retain(|intent| {
        let (hand, select, grab) = match *intent {
            Intent::Pointer { hand, .. } => (hand, None, None),
            Intent::Select { hand, pressed } => (hand, Some(pressed), None),
            Intent::GrabBegin { hand, .. } => (hand, None, Some(true)),
            Intent::GrabMove { hand, .. } | Intent::PushPull { hand, .. } => (hand, None, None),
            // Released to `manipulate` too, which ignores a grab it never began.
            Intent::GrabEnd { hand } => {
                if let Some(i) = SIDES.iter().position(|s| *s == hand)
                    && on_panel[i].is_some()
                {
                    presses[i].1.push(false);
                }
                return true;
            }
            _ => return true,
        };
        let Some(i) = SIDES.iter().position(|s| *s == hand) else {
            return true;
        };
        if on_panel[i].is_none() {
            return true;
        }
        presses[i].0.extend(select);
        presses[i].1.extend(grab);
        false
    });
    // Q-315: a grab press on a title bar is the panel's; the select press that comes
    // with it (a pinch is both, Appendix D.3) is withheld, and so is its release.
    for (i, side) in SIDES.into_iter().enumerate() {
        let on_strip = on_panel[i].is_some_and(|(panel, pixel, _)| {
            quads.iter().any(|(e, _, q)| {
                *e == panel && q.title_bar && pixel.y < q.size_px.y as f32 * TITLE_FRACTION
            })
        });
        let (select, grab) = &mut presses[i];
        if on_strip && grab.contains(&true) && select.contains(&true) {
            select.retain(|p| !*p);
            state.withheld[i] = true;
        }
        if state.withheld[i] && select.contains(&false) {
            select.retain(|p| *p);
            state.withheld[i] = false;
        }
        if out
            .iter()
            .any(|x| matches!(x, Intent::Select { hand, pressed: false } if *hand == side))
        {
            // Released off the quads: nothing is withheld any more.
            state.withheld[i] = false;
        }
    }
    for (i, side) in SIDES.into_iter().enumerate() {
        if let Some((panel, pixel, _)) = on_panel[i] {
            let (select, grab) = std::mem::take(&mut presses[i]);
            pointers.write(PanelPointer {
                hand: side,
                panel,
                pixel,
                select,
                grab,
                scroll_px: scroll[i],
            });
        }
    }

    // Q-269, Q-303: the teleport replaces the smooth move.
    frame.arc.clear();
    let teleport = loco.as_deref().is_some_and(|l| l.teleport) && cfg.locomotion_enabled;
    if teleport {
        for intent in &mut out {
            if let Intent::Locomote { move_xz, .. } = intent {
                *move_xz = Vec2::ZERO;
            }
        }
        out.retain(|i| !matches!(i, Intent::Locomote { move_xz, turn } if *move_xz == Vec2::ZERO && *turn == 0.0));
        let mi = if now.left_handed { 1 } else { 0 };
        let hand = if mi == 0 { hands.left } else { hands.right };
        let usable = hand.filter(|t| {
            t.source == HandSource::Controller
                && t.tracked
                && on_panel[mi].is_none()
                && !held.contains(&SIDES[mi])
        });
        match (usable, state.aiming) {
            (None, _) => state.aiming = None,
            (Some(t), None) => {
                if t.stick.y > ARC_SHOW {
                    let arc = teleport_arc(&t.aim);
                    state.aiming = Some(arc.as_ref().and_then(|a| a.last().copied()));
                    frame.arc = arc.unwrap_or_default();
                }
            }
            (Some(t), Some(target)) => {
                if t.stick.y < ARC_RELEASE {
                    state.aiming = None;
                    if let Some(target) = target {
                        out.push(Intent::Teleport { target });
                    }
                } else {
                    let arc = teleport_arc(&t.aim);
                    state.aiming = Some(arc.as_ref().and_then(|a| a.last().copied()));
                    frame.arc = arc.unwrap_or_default();
                }
            }
        }
    } else {
        state.aiming = None;
    }

    for intent in out {
        intents.write(intent);
    }
}

/// The ray and dot colors (Q-278): a light grey, or yellow under `[display]
/// high_contrast` (Q-304).
fn colors(high_contrast: bool) -> (Color, Color) {
    if high_contrast {
        (Color::srgb(1.0, 1.0, 0.0), Color::srgb(1.0, 1.0, 0.0))
    } else {
        (Color::srgba(0.9, 0.9, 0.9, 0.8), Color::WHITE)
    }
}

/// The rays to their first hit (a panel quad, else the hovered instance's distance,
/// else 10 m) with a dot at a hit, and the teleport arc (§1.8 Rays, PRD #452).
fn draw(
    frame: Res<Frame>,
    hovered: Query<&Hovered>,
    settings: Option<Res<SettingsRes>>,
    mut gizmos: Gizmos,
) {
    let (ray, dot) = colors(settings.is_some_and(|s| s.0.display.high_contrast));
    for (i, side) in SIDES.into_iter().enumerate() {
        let Some(r) = frame.rays[i] else { continue };
        let instance = hovered
            .iter()
            .filter(|h| h.hand == side)
            .map(|h| h.distance)
            .reduce(f32::min);
        let hit = match (r.panel, instance) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let end = r.origin + r.dir * hit.unwrap_or(RAY_M);
        gizmos.line(r.origin, end, ray);
        if hit.is_some() {
            gizmos.sphere(Isometry3d::from_translation(end), DOT_RADIUS_M, dot);
        }
    }
    if frame.arc.len() > 1 {
        gizmos.linestrip(frame.arc.iter().copied(), ray);
    }
}

/// What the haptics compare frame to frame: each hovered instance with its hand, each
/// grabbed instance with the hands holding it.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Touches {
    pub(crate) hovered: Vec<(Entity, Hand)>,
    pub(crate) grabbed: Vec<(Entity, Vec<Hand>)>,
}

/// A start or an end of the main panel's grab by its title bar (§1.8: "Grabbing a UI
/// panel by its title bar moves it the same way"), for D-44's grab pulse: `panel` writes
/// one per hand that joins or leaves its grab, `haptics` reads them.
// Q-313
#[derive(bevy::prelude::Message, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PanelGrabEdge(pub(crate) Hand);

/// The haptics' memory: the touches of the last frame, and the hand whose ray last
/// released `select` on a panel quad with the frames since, which a UI click arriving
/// within `CLICK_FRAMES` after it is credited to (egui clicks on the release, Q-314).
#[derive(Resource, Default)]
struct HapticMemory {
    touches: Touches,
    released: Option<(Hand, u32)>,
}

/// How many frames after a release its click may arrive: egui's pass runs in the
/// release's frame and `audio` raises the sound in the next (Q-314).
const CLICK_FRAMES: u32 = 3;

/// The pulses a frame gives (Q-270, Q-313, Q-314), one per hand at most (the
/// strongest): hover enter when an instance's `Hovered` appears for a hand or switches to
/// it; grab start and end when a hand joins or leaves an instance's `Grabbed` (one hand,
/// two, none), so a squeeze on nothing pulses nothing, and on each of the main panel's
/// grab edges in `panel_grabs`; a snap turn on the turn stick's hand (the right, the left
/// in left-handed mode); a UI click on each hand of `clicks`.
// Q-270, Q-313, Q-314
pub(crate) fn select_pulses(
    prev: &Touches,
    now: &Touches,
    snap_turned: bool,
    clicks: &[Hand],
    panel_grabs: &[Hand],
    left_handed: bool,
) -> Vec<(Hand, Pulse)> {
    let mut out: Vec<(Hand, Pulse)> = Vec::new();
    let mut add = |hand: Hand, p: Pulse| {
        if hand == Hand::Mouse {
            return;
        }
        match out.iter_mut().find(|(h, _)| *h == hand) {
            Some((_, q)) => {
                if (p.amplitude, p.duration_ms) > (q.amplitude, q.duration_ms) {
                    *q = p;
                }
            }
            None => out.push((hand, p)),
        }
    };
    for &(e, hand) in &now.hovered {
        if !prev.hovered.contains(&(e, hand)) {
            add(hand, HOVER);
        }
    }
    let hands_of = |t: &Touches, e: Entity| -> Vec<Hand> {
        t.grabbed
            .iter()
            .find(|(g, _)| *g == e)
            .map(|(_, h)| h.clone())
            .unwrap_or_default()
    };
    for (e, hands) in &now.grabbed {
        let before = hands_of(prev, *e);
        for h in hands {
            if !before.contains(h) {
                add(*h, GRAB);
            }
        }
    }
    for (e, hands) in &prev.grabbed {
        let after = hands_of(now, *e);
        for h in hands {
            if !after.contains(h) {
                add(*h, GRAB);
            }
        }
    }
    for &hand in panel_grabs {
        add(hand, GRAB);
    }
    if snap_turned {
        add(if left_handed { Hand::Left } else { Hand::Right }, TICK);
    }
    for &hand in clicks {
        add(hand, TICK);
    }
    out
}

/// The frame's pulses into `HapticLog.wanted`, kept to the sides a controller owns (a
/// tracked hand has no output, PRD #452's "Haptics"). A UI click is a `PlayUiSound` of
/// the click or the confirmation, which `audio` raises from a widget's click, credited to
/// the hand that last released `select` on a panel quad; a press on empty panel area
/// clicks nothing and pulses nothing (Q-314).
// Q-270, Q-313, Q-314
#[allow(clippy::too_many_arguments)]
fn haptics(
    hands: Res<XrHands>,
    hovered: Query<(Entity, &Hovered)>,
    grabbed: Query<(Entity, &Grabbed)>,
    mut intents: MessageReader<Intent>,
    mut pointers: MessageReader<PanelPointer>,
    mut sounds: MessageReader<PlayUiSound>,
    mut panel_grabs: MessageReader<PanelGrabEdge>,
    mut memory: ResMut<HapticMemory>,
    mut log: ResMut<HapticLog>,
) {
    let now = Touches {
        hovered: hovered.iter().map(|(e, h)| (e, h.hand)).collect(),
        grabbed: grabbed
            .iter()
            .map(|(e, g)| {
                let hands = match g {
                    Grabbed::One { hand, .. } => vec![*hand],
                    Grabbed::Two { .. } => vec![Hand::Left, Hand::Right],
                };
                (e, hands)
            })
            .collect(),
    };
    let snap_turned = intents
        .read()
        .any(|i| matches!(i, Intent::SnapTurn { steps } if *steps != 0));
    let clicked = sounds
        .read()
        .any(|PlayUiSound(s)| matches!(s, UiSound::Click | UiSound::Confirm));
    let mut clicks = Vec::new();
    if clicked && let Some((hand, _)) = memory.released.take() {
        clicks.push(hand);
    }
    memory.released = memory
        .released
        .map(|(h, n)| (h, n + 1))
        .filter(|(_, n)| *n <= CLICK_FRAMES);
    for p in pointers.read() {
        if p.select.contains(&false) {
            memory.released = Some((p.hand, 0));
        }
    }
    let grabs: Vec<Hand> = panel_grabs.read().map(|e| e.0).collect();
    let pulses = select_pulses(
        &memory.touches,
        &now,
        snap_turned,
        &clicks,
        &grabs,
        hands.left_handed,
    );
    memory.touches = now;
    log.wanted.clear();
    for (hand, p) in pulses {
        let side = usize::from(hand == Hand::Right);
        let owner = if side == 0 { hands.left } else { hands.right };
        if owner.is_some_and(|t| t.source == HandSource::Controller) {
            log.wanted.push((side, p));
        }
    }
}

/// Plays the frame's pulses on the controllers while the session runs, recording each
/// call's outcome in `HapticLog`.
fn apply_pulses(
    session: Option<Res<OxrSession>>,
    actions: Option<Res<Actions>>,
    mut log: ResMut<HapticLog>,
) {
    let (Some(session), Some(actions)) = (session, actions) else {
        return;
    };
    let wanted = std::mem::take(&mut log.wanted);
    for (side, p) in wanted {
        let result = actions
            .pulse(&session, side, p.duration_ms, p.amplitude)
            .map_err(|e| e.to_string());
        if let Err(e) = &result {
            log_warning(&XrError::Runtime(format!("a haptic pulse: {e}")));
        }
        log.record(side, p, result);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Quat;

    fn aim_at(x: f32, y: f32, z: f32, rotation: Quat) -> Pose {
        Pose {
            translation: Vec3::new(x, y, z),
            rotation,
            scale: 1.0,
        }
    }

    /// Q-269: launched level from 1.5 m at 5 m/s, the arc meets the floor after
    /// √(2 · 1.5 / 9.8) = 0.553283 s, 2.766417 m ahead (−z); 45° up from 1 m it meets it
    /// after (3.535534 + √(12.5 + 19.6)) / 9.8 = 0.938900 s, 3.319485 m ahead; from below
    /// the floor there is none.
    #[test]
    fn teleport_arc_ends_on_the_floor_along_the_aim() {
        let arc = teleport_arc(&aim_at(0.0, 1.5, 0.0, Quat::IDENTITY)).unwrap();
        assert_eq!(arc.len(), 64);
        assert!(arc[0].distance(Vec3::new(0.0, 1.5, 0.0)) < 1e-6);
        let end = *arc.last().unwrap();
        assert!(end.distance(Vec3::new(0.0, 0.0, -2.766417)) < 1e-4, "{end}");
        assert!(arc.iter().all(|p| p.y >= -1e-5));

        let up = Quat::from_rotation_x(std::f32::consts::FRAC_PI_4);
        let end = *teleport_arc(&aim_at(0.0, 1.0, 0.0, up))
            .unwrap()
            .last()
            .unwrap();
        assert!(end.distance(Vec3::new(0.0, 0.0, -3.319485)) < 1e-4, "{end}");
        // Turned 90° to the left (about +y), the arc lands along −x.
        let left = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let end = *teleport_arc(&aim_at(1.0, 1.5, 2.0, left))
            .unwrap()
            .last()
            .unwrap();
        assert!(
            end.distance(Vec3::new(1.0 - 2.766417, 0.0, 2.0)) < 1e-4,
            "{end}"
        );

        assert!(teleport_arc(&aim_at(0.0, -0.1, 0.0, Quat::IDENTITY)).is_none());
        assert!(teleport_arc(&aim_at(0.0, 0.0, 0.0, Quat::IDENTITY)).is_none());
    }

    fn e(n: u32) -> Entity {
        Entity::from_raw_u32(n).unwrap()
    }

    /// Q-270, D-44: which hand and which pulse each of a frame's events gives.
    #[test]
    fn select_pulses_hover_grab_snap_and_click() {
        let none = Touches::default();
        // Hover enter: a hover appearing, and a hover switching hands.
        let hover_l = Touches {
            hovered: vec![(e(1), Hand::Left)],
            ..Touches::default()
        };
        assert_eq!(
            select_pulses(&none, &hover_l, false, &[], &[], false),
            [(Hand::Left, HOVER)]
        );
        assert!(select_pulses(&hover_l, &hover_l, false, &[], &[], false).is_empty());
        let hover_r = Touches {
            hovered: vec![(e(1), Hand::Right)],
            ..Touches::default()
        };
        assert_eq!(
            select_pulses(&hover_l, &hover_r, false, &[], &[], false),
            [(Hand::Right, HOVER)]
        );
        // The mouse never pulses.
        let mouse = Touches {
            hovered: vec![(e(1), Hand::Mouse)],
            ..Touches::default()
        };
        assert!(select_pulses(&none, &mouse, false, &[], &[], false).is_empty());

        // Grabs: one hand, a second, one released, the last released.
        let one = |h: Hand| Touches {
            grabbed: vec![(e(2), vec![h])],
            ..Touches::default()
        };
        let two = Touches {
            grabbed: vec![(e(2), vec![Hand::Left, Hand::Right])],
            ..Touches::default()
        };
        assert_eq!(
            select_pulses(&none, &one(Hand::Left), false, &[], &[], false),
            [(Hand::Left, GRAB)]
        );
        assert_eq!(
            select_pulses(&one(Hand::Left), &two, false, &[], &[], false),
            [(Hand::Right, GRAB)]
        );
        assert_eq!(
            select_pulses(&two, &one(Hand::Right), false, &[], &[], false),
            [(Hand::Left, GRAB)]
        );
        assert_eq!(
            select_pulses(&one(Hand::Right), &none, false, &[], &[], false),
            [(Hand::Right, GRAB)]
        );
        // A squeeze on nothing changes no `Grabbed`: nothing.
        assert!(select_pulses(&none, &none, false, &[], &[], false).is_empty());

        // A snap turn: the turn stick's hand, the left one in left-handed mode.
        assert_eq!(
            select_pulses(&none, &none, true, &[], &[], false),
            [(Hand::Right, TICK)]
        );
        assert_eq!(
            select_pulses(&none, &none, true, &[], &[], true),
            [(Hand::Left, TICK)]
        );
        // A start or an end of the main panel's grab (Q-313).
        assert_eq!(
            select_pulses(&none, &none, false, &[], &[Hand::Right], false),
            [(Hand::Right, GRAB)]
        );
        // A panel click on the hand that clicked.
        assert_eq!(
            select_pulses(&none, &none, false, &[Hand::Left], &[], false),
            [(Hand::Left, TICK)]
        );
        // One pulse per hand, the strongest: a grab start with a hover enter.
        let both = Touches {
            hovered: vec![(e(2), Hand::Left)],
            grabbed: vec![(e(2), vec![Hand::Left])],
        };
        assert_eq!(
            select_pulses(&none, &both, false, &[Hand::Left], &[], false),
            [(Hand::Left, GRAB)]
        );
    }

    /// The quads' hits: the nearest front hit, with its pixel.
    #[test]
    fn nearest_panel_takes_the_nearer_quad() {
        let q = PanelQuad {
            size_m: Vec2::new(0.56, 0.42),
            size_px: UVec2::new(1024, 768),
            title_bar: true,
        };
        let at = |z: f32| Pose {
            translation: Vec3::new(0.0, 0.0, z),
            ..Pose::IDENTITY
        };
        let quads = [(e(1), at(0.0), q), (e(2), at(1.0), q)];
        let (panel, pixel, d) = nearest_panel(Vec3::new(0.0, 0.0, 3.0), -Vec3::Z, &quads).unwrap();
        assert_eq!(panel, e(2));
        assert!((d - 2.0).abs() < 1e-6);
        assert!(pixel.distance(Vec2::new(512.0, 384.0)) < 1e-3);
        assert!(nearest_panel(Vec3::new(0.0, 0.0, 3.0), Vec3::Z, &quads).is_none());
        assert!(nearest_panel(Vec3::new(1.0, 0.0, 3.0), -Vec3::Z, &quads).is_none());
    }

    /// A platform for the app test: every permission granted, nothing picked.
    struct Stub(ftgsxr_app::AppPaths);

    impl ftgsxr_app::Platform for Stub {
        fn paths(&self) -> &ftgsxr_app::AppPaths {
            &self.0
        }
        fn pick(&self, _: ftgsxr_app::PickKind) -> ftgsxr_app::PickHandle {
            ftgsxr_app::PickHandle(0)
        }
        fn poll_pick(
            &self,
            _: ftgsxr_app::PickHandle,
        ) -> std::task::Poll<Result<Vec<ftgsxr_app::ModelRef>, String>> {
            std::task::Poll::Pending
        }
        fn request_permission(&self, _: ftgsxr_app::Permission) -> ftgsxr_app::PermissionHandle {
            ftgsxr_app::PermissionHandle(0)
        }
        fn poll_permission(&self, _: ftgsxr_app::PermissionHandle) -> std::task::Poll<bool> {
            std::task::Poll::Ready(true)
        }
        fn document_provider(&self) -> Option<std::sync::Arc<dyn ftgsxr_app::DocumentProvider>> {
            None
        }
        fn camera_frames(&self) -> Option<std::sync::Arc<dyn ftgsxr_app::CameraFrames>> {
            None
        }
    }

    #[derive(Resource, Default)]
    struct Seen(Vec<PanelPointer>);

    fn collect(mut r: MessageReader<PanelPointer>, mut seen: ResMut<Seen>) {
        seen.0.extend(r.read().cloned());
    }

    /// The headless app of `ftgsxr-app`'s tests (`MinimalPlugins`, assets, transforms,
    /// `AppPlugin` on a stub platform and the stub audio output) with this module's
    /// plugin, and P1 loaded at the stage origin.
    fn app_with_p1() -> (bevy::app::App, Entity) {
        use bevy::app::{App, PluginsState, Update};
        use ftgsxr_app::{
            AppPlugin, AudioOutputRes, LibraryCommand, LoadedModel, ModelRef, QualityTier,
            StubAudioOutput,
        };
        let root =
            std::env::temp_dir().join(format!("ftgsxr-xr-arbitration-{}", std::process::id()));
        let paths = ftgsxr_app::AppPaths {
            config: root.join("config"),
            cache: root.join("cache"),
            models: root.join("models"),
            scenes: root.join("config/scenes"),
            screenshots: root.join("pictures"),
            logs: root.join("cache/logs"),
        };
        let mut app = App::new();
        app.add_plugins((
            bevy::MinimalPlugins,
            bevy::asset::AssetPlugin::default(),
            bevy::transform::TransformPlugin,
            bevy::input::InputPlugin,
        ))
        .insert_resource(AudioOutputRes(std::sync::Arc::new(StubAudioOutput::new())))
        .add_plugins(AppPlugin {
            platform: std::sync::Arc::new(Stub(paths)),
            tier: QualityTier::conformance(),
            initial: Vec::new(),
            scene: None,
        })
        .add_plugins(XrInputPlugin)
        .init_resource::<Seen>()
        .add_systems(Update, collect);
        while app.plugins_state() == PluginsState::Adding {
            bevy::tasks::tick_global_task_pools_on_main_thread();
        }
        app.finish();
        app.cleanup();
        let path = ftgsxr_testkit::fixtures_dir().join("probes/P1.safetensors");
        app.world_mut()
            .write_message(LibraryCommand::LoadRef(ModelRef::File(path)));
        let mut instance = None;
        for _ in 0..600 {
            app.update();
            let mut q = app
                .world_mut()
                .query_filtered::<Entity, With<LoadedModel>>();
            instance = q.iter(app.world()).next();
            if instance.is_some() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let instance = instance.expect("P1 loads");
        app.update();
        (app, instance)
    }

    /// The right controller 3 m in front of the stage origin, aiming at it through P1,
    /// with its trigger at `trigger`.
    fn aiming(app: &mut bevy::app::App, trigger: f32) {
        let hand = crate::TrackedHand {
            aim: aim_at(0.0, 0.0, 3.0, Quat::IDENTITY),
            grip: Pose::IDENTITY,
            pinch: 0.0,
            pinch_middle: 0.0,
            squeeze: 0.0,
            trigger,
            stick: Vec2::ZERO,
            primary: false,
            secondary: false,
            menu: false,
            source: HandSource::Controller,
            tracked: true,
        };
        app.world_mut().resource_mut::<XrHands>().right = Some(hand);
    }

    /// Q-263 (#460's added test; source of truth §1.8 Rays and User interface): a hand
    /// whose ray hits a panel quad hovers and selects no instance, and its presses go to
    /// the quad; the same hand off the panel hovers and selects the instance.
    #[test]
    fn arbitration_a_ray_on_a_panel_hovers_and_selects_no_instance() {
        use bevy::prelude::Transform;
        let (mut app, instance) = app_with_p1();
        // The main panel 1.5 m in front of the origin, between the hand and P1.
        let panel = app
            .world_mut()
            .spawn((
                PanelQuad {
                    size_m: Vec2::new(0.56, 0.42),
                    size_px: UVec2::new(1024, 768),
                    title_bar: true,
                },
                Transform::from_xyz(0.0, 0.0, 1.5),
                GlobalTransform::from(Transform::from_xyz(0.0, 0.0, 1.5)),
            ))
            .id();
        aiming(&mut app, 0.0);
        app.update();
        aiming(&mut app, 1.0);
        app.update();
        app.update();
        assert!(app.world().get::<Hovered>(instance).is_none(), "no hover");
        assert!(
            app.world().get::<Selected>(instance).is_none(),
            "no selection"
        );
        let seen = &app.world().resource::<Seen>().0;
        assert!(
            seen.iter()
                .all(|p| p.panel == panel && p.hand == Hand::Right)
        );
        assert!(
            seen.iter().any(|p| p.select == [true]),
            "the press went to the panel: {seen:?}"
        );
        assert!(
            seen.iter()
                .all(|p| (p.pixel - Vec2::new(512.0, 384.0)).length() < 1e-2)
        );

        // Off the panel: the same hand hovers and selects the instance.
        app.world_mut().entity_mut(panel).despawn();
        aiming(&mut app, 0.0);
        app.update();
        let hovered = app.world().get::<Hovered>(instance).copied();
        assert_eq!(hovered.map(|h| h.hand), Some(Hand::Right), "hovered");
        aiming(&mut app, 1.0);
        app.update();
        assert!(app.world().get::<Selected>(instance).is_some(), "selected");
    }

    /// Dv-8's pinch clause through this module's own system (kept here, after `tracking`
    /// in §2.1's order, #452): the synthetic 0.8 pinch writes `Select`, and the test
    /// passes with no session (no pulse: no controller).
    #[test]
    fn run_pinch_writes_select_through_xr_input() {
        use crate::tracking::dv8::{Dv8, Phase, run};
        let mut app = App::new();
        app.add_plugins(XrInputPlugin)
            .insert_resource(Dv8::new(Phase::Pulse))
            .add_systems(PostUpdate, run);
        for _ in 0..10 {
            app.update();
        }
        let test = app.world().resource::<Dv8>();
        let (pass, detail) = test.verdict.clone().expect("a verdict");
        assert!(pass, "{detail}");
        assert!(detail.contains("pulse: no controller"), "{detail}");
        assert!(
            detail.contains("wrote Select on the right hand"),
            "{detail}"
        );
    }

    /// Q-314 (D-44: "on a UI click"): the click pulse waits for a widget's click
    /// (`PlayUiSound` of the click), credited to the hand that released `select` on the
    /// panel; a press and release on empty panel area pulses nothing. Q-313: a panel
    /// grab edge gives the grab pulse. Only a controller-owned side gets one.
    #[test]
    fn haptics_pulse_a_ui_click_only_for_a_widget_click() {
        let mut app = App::new();
        app.add_plugins(XrInputPlugin);
        aiming(&mut app, 0.0);
        let pointer = |select: Vec<bool>| PanelPointer {
            hand: Hand::Right,
            panel: e(9),
            pixel: Vec2::ZERO,
            select,
            grab: Vec::new(),
            scroll_px: 0.0,
        };
        let wanted = |app: &mut App| app.world().resource::<HapticLog>().wanted.clone();
        app.update();
        // On empty panel area: a press and a release, and no click.
        app.world_mut().write_message(pointer(vec![true]));
        app.update();
        assert!(wanted(&mut app).is_empty(), "a press is no click");
        app.world_mut().write_message(pointer(vec![false]));
        app.update();
        app.update();
        assert!(wanted(&mut app).is_empty(), "no widget clicked");
        for _ in 0..CLICK_FRAMES {
            app.update();
        }
        app.world_mut().write_message(PlayUiSound(UiSound::Click));
        app.update();
        assert!(
            wanted(&mut app).is_empty(),
            "a late click is not the release's"
        );
        // On a widget: the click's sound a frame after the release.
        app.world_mut().write_message(pointer(vec![true]));
        app.update();
        app.world_mut().write_message(pointer(vec![false]));
        app.update();
        app.world_mut().write_message(PlayUiSound(UiSound::Click));
        app.update();
        assert_eq!(wanted(&mut app), [(1, TICK)]);
        app.update();
        assert!(wanted(&mut app).is_empty(), "one pulse per click");
        // A panel grab edge.
        app.world_mut().write_message(PanelGrabEdge(Hand::Right));
        app.update();
        assert_eq!(wanted(&mut app), [(1, GRAB)]);
        // A tracked hand has no output.
        app.world_mut()
            .resource_mut::<XrHands>()
            .right
            .as_mut()
            .unwrap()
            .source = HandSource::Hand;
        app.world_mut().write_message(PanelGrabEdge(Hand::Right));
        app.update();
        assert!(wanted(&mut app).is_empty());
    }

    /// Dv-8's pulse clause through the haptics (Dv-8: "a grab start sends the 20 ms
    /// pulse"; Q-272): with the right controller active, the clause injects a grab start
    /// on its side, which reaches `haptics` as D-44's grab pulse; with no session nothing
    /// is applied, and the clause says so and fails, while the pinch clause still runs.
    #[test]
    fn run_pulse_drives_a_grab_start_through_the_haptics() {
        use crate::tracking::dv8::{Dv8, Phase, run};
        let mut app = App::new();
        app.add_plugins(XrInputPlugin)
            .insert_resource(Dv8::new(Phase::Pulse))
            .add_systems(PostUpdate, run);
        aiming(&mut app, 0.0);
        for _ in 0..60 {
            app.update();
            if app.world().resource::<Dv8>().verdict.is_some() {
                break;
            }
        }
        let (pass, detail) = app
            .world()
            .resource::<Dv8>()
            .verdict
            .clone()
            .expect("a verdict");
        assert!(!pass, "{detail}");
        assert!(
            detail.contains("a grab start on the right controller reached the haptics"),
            "{detail}"
        );
        assert!(detail.contains("wrote Select"), "{detail}");
        let mut q = app.world_mut().query::<&Grabbed>();
        assert_eq!(q.iter(app.world()).count(), 0, "the injected grab is gone");
    }

    /// §1.8 Rays ("to the first hit"), Q-317: a panel behind the instance does not
    /// take the ray; the instance, the nearer hit, is hovered and selected.
    #[test]
    fn arbitration_an_instance_in_front_of_a_panel_takes_the_ray() {
        use bevy::prelude::Transform;
        let (mut app, instance) = app_with_p1();
        // The main panel 1 m behind P1, seen from the hand through it.
        app.world_mut().spawn((
            PanelQuad {
                size_m: Vec2::new(0.56, 0.42),
                size_px: UVec2::new(1024, 768),
                title_bar: true,
            },
            Transform::from_xyz(0.0, 0.0, -1.0),
            GlobalTransform::from(Transform::from_xyz(0.0, 0.0, -1.0)),
        ));
        aiming(&mut app, 0.0);
        app.update();
        app.update();
        let hovered = app.world().get::<Hovered>(instance).copied();
        assert_eq!(hovered.map(|h| h.hand), Some(Hand::Right), "hovered");
        aiming(&mut app, 1.0);
        app.update();
        assert!(app.world().get::<Selected>(instance).is_some(), "selected");
        assert!(
            app.world().resource::<Seen>().0.is_empty(),
            "nothing went to the panel"
        );
    }

    /// P1 posed around the right hand at (0, 0, 3), scale 50, so its box holds the ray's
    /// origin (Q-317 as changed on #550).
    fn inside_p1() -> (bevy::app::App, Entity) {
        use bevy::prelude::Transform;
        let (mut app, instance) = app_with_p1();
        let around = Transform::from_xyz(0.0, 0.0, 3.0).with_scale(Vec3::splat(50.0));
        app.world_mut().entity_mut(instance).insert(around);
        app.update();
        // The precondition: E10 from the hand gives the instance distance 0.
        let model = app.world().get::<SplatModel>(instance).unwrap().0.clone();
        let bounds = app
            .world()
            .resource::<Assets<SplatAsset>>()
            .get(&model)
            .unwrap()
            .bounds;
        let pose = Pose {
            translation: around.translation,
            rotation: around.rotation,
            scale: 50.0,
        };
        let t = ray_hit(Vec3::new(0.0, 0.0, 3.0), Vec3::NEG_Z, &bounds, &pose, 10.0);
        assert_eq!(t, Some(0.0), "the hand is inside P1's box");
        (app, instance)
    }

    /// The right hand (hand tracking) at (0, 0, 3) aiming along −z, pinching at `pinch`.
    fn pinching(app: &mut App, pinch: f32) {
        let aim = aim_at(0.0, 0.0, 3.0, Quat::IDENTITY);
        app.world_mut().resource_mut::<XrHands>().right = Some(crate::TrackedHand {
            aim,
            grip: aim,
            pinch,
            pinch_middle: 0.0,
            squeeze: 0.0,
            trigger: 0.0,
            stick: Vec2::ZERO,
            primary: false,
            secondary: false,
            menu: false,
            source: HandSource::Hand,
            tracked: true,
        });
    }

    /// Q-317 (#550's added test): from inside P1's box, a pinch on the wrist menu's quad
    /// and on the main panel goes to the quad; the instance is not hovered, selected or
    /// grabbed.
    #[test]
    fn arbitration_a_quad_takes_the_ray_from_inside_an_instance() {
        use bevy::prelude::Transform;
        // The wrist menu's quad 15 cm ahead, then the main panel 0.5 m ahead.
        for (title_bar, ahead, size_m, size_px) in [
            (false, 0.15, Vec2::new(0.12, 0.09), UVec2::new(256, 192)),
            (true, 0.5, Vec2::new(0.56, 0.42), UVec2::new(1024, 768)),
        ] {
            let (mut app, instance) = inside_p1();
            let at = Transform::from_xyz(0.0, 0.0, 3.0 - ahead);
            let quad = app
                .world_mut()
                .spawn((
                    PanelQuad {
                        size_m,
                        size_px,
                        title_bar,
                    },
                    at,
                    GlobalTransform::from(at),
                ))
                .id();
            pinching(&mut app, 0.0);
            app.update();
            pinching(&mut app, 1.0);
            app.update();
            app.update();
            assert!(app.world().get::<Hovered>(instance).is_none(), "no hover");
            assert!(
                app.world().get::<Selected>(instance).is_none(),
                "no selection"
            );
            assert!(app.world().get::<Grabbed>(instance).is_none(), "no grab");
            let seen = &app.world().resource::<Seen>().0;
            assert!(!seen.is_empty(), "the quad got the pointer");
            assert!(
                seen.iter()
                    .all(|p| p.panel == quad && p.hand == Hand::Right)
            );
            assert!(
                seen.iter().any(|p| p.select == [true]),
                "the pinch went to the quad: {seen:?}"
            );
        }
    }

    /// Q-317 (#550's added test; §1.8 Rays): from inside P1's box with no quad hit, the
    /// pinch selects and grabs the enclosing instance.
    #[test]
    fn arbitration_from_inside_an_instance_with_no_quad_the_pinch_grabs_it() {
        let (mut app, instance) = inside_p1();
        pinching(&mut app, 0.0);
        app.update();
        let hovered = app.world().get::<Hovered>(instance).copied();
        assert_eq!(hovered.map(|h| h.hand), Some(Hand::Right), "hovered");
        pinching(&mut app, 1.0);
        app.update();
        app.update();
        assert!(app.world().get::<Selected>(instance).is_some(), "selected");
        assert!(app.world().get::<Grabbed>(instance).is_some(), "grabbed");
        assert!(app.world().resource::<Seen>().0.is_empty(), "no quad");
    }

    /// Q-315 (Q-306's side effect removed): a pinch on the main panel's title bar is
    /// its grab and no click: the panel gets the grab press and release and no select;
    /// a pinch below the strip clicks, from the row just under it, where `ui`'s tab row
    /// begins (Q-306, #542).
    #[test]
    fn a_pinch_on_the_title_bar_grabs_and_does_not_click() {
        use bevy::app::Update;
        use bevy::prelude::Transform;
        let mut app = App::new();
        app.add_plugins(XrInputPlugin)
            .init_resource::<Seen>()
            .add_systems(Update, collect);
        app.world_mut().spawn((
            PanelQuad {
                size_m: Vec2::new(0.56, 0.42),
                size_px: UVec2::new(1024, 768),
                title_bar: true,
            },
            Transform::default(),
            GlobalTransform::default(),
        ));
        // Pixel row 20 (of the 48 px strip) and row 300, from 3 m in front.
        let pinch = |app: &mut App, row: f32, strength: f32| {
            let y = 0.21 - row / 768.0 * 0.42;
            let aim = aim_at(0.0, y, 3.0, Quat::IDENTITY);
            app.world_mut().resource_mut::<XrHands>().right = Some(crate::TrackedHand {
                aim,
                grip: aim,
                pinch: strength,
                pinch_middle: 0.0,
                squeeze: 0.0,
                trigger: 0.0,
                stick: Vec2::ZERO,
                primary: false,
                secondary: false,
                menu: false,
                source: HandSource::Hand,
                tracked: true,
            });
            app.update();
            let seen = std::mem::take(&mut app.world_mut().resource_mut::<Seen>().0);
            let select: Vec<bool> = seen.iter().flat_map(|p| p.select.clone()).collect();
            let grab: Vec<bool> = seen.iter().flat_map(|p| p.grab.clone()).collect();
            (select, grab)
        };
        pinch(&mut app, 20.0, 0.0);
        assert_eq!(pinch(&mut app, 20.0, 0.9), (vec![], vec![true]));
        assert_eq!(pinch(&mut app, 20.0, 0.0), (vec![], vec![false]));
        assert_eq!(pinch(&mut app, 300.0, 0.9), (vec![true], vec![true]));
        assert_eq!(pinch(&mut app, 300.0, 0.0), (vec![false], vec![false]));
        // Either side of the strip's 48 px edge (#542).
        assert_eq!(pinch(&mut app, 47.0, 0.9), (vec![], vec![true]));
        assert_eq!(pinch(&mut app, 47.0, 0.0), (vec![], vec![false]));
        assert_eq!(pinch(&mut app, 49.0, 0.9), (vec![true], vec![true]));
        assert_eq!(pinch(&mut app, 49.0, 0.0), (vec![false], vec![false]));
    }
}
