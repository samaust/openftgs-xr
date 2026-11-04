// SPDX-License-Identifier: Apache-2.0

//! `panel`'s systems (§1.8 User interface, PRD #452's panel mode): with the `Headset`
//! marker, the main panel's and the wrist menu's quads at session creation, the main
//! panel's summon and hide on `Toggle(Feature::Panel)`, its move by a grab on the title
//! bar and its resize by two hands (E11), and the wrist menu on the palm.

use bevy::app::{App, Plugin, Update};
use bevy::asset::{Assets, Handle, RenderAssetUsages};
use bevy::ecs::message::{MessageReader, MessageWriter};
use bevy::image::Image;
use bevy::math::primitives::Rectangle;
use bevy::math::{UVec2, Vec2, Vec3};
use bevy::mesh::{Mesh, Mesh3d};
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::prelude::{
    Commands, Entity, IntoScheduleConfigs, Query, Res, ResMut, Resource, Transform, Visibility,
    With, Without,
};
use bevy::render::render_resource::TextureFormat;
use bevy_mod_xr::session::XrSessionCreated;
use ftgsxr_app::{Feature, Hand, Headset, Intent, PanelPointer, UiSurface, WorldPanel, WristMenu};
use ftgsxr_core::{Grab, Pose, Quat, ScaleLimits, TwoHandGrab};

use super::{MainPanel, PanelFollow, summon_pose, wrist_menu_pose, wrist_menu_visible};
use crate::tracking::{HandSource, TrackedHand, XrHands};
use crate::xr_input::{PanelGrabEdge, PanelQuad};

/// The main panel's default size (§1.8): 0.56 m × 0.42 m at 1024 × 768 px.
const PANEL_M: Vec2 = Vec2::new(0.56, 0.42);
const PANEL_PX: UVec2 = UVec2::new(1024, 768);
/// The texture's resolution after a resize, pixels per metre of width (PRD #452: 1024 px
/// for 0.56 m), and the panel's 4:3 aspect, height over width.
const PX_PER_M: f32 = 1829.0;
const ASPECT: f32 = 0.75;
/// The width a two-hand grab keeps the panel between (§1.8: 0.3–1.2 m).
const WIDTH_LIMITS: ScaleLimits = ScaleLimits { min: 0.3, max: 1.2 };
/// The title bar: the top strip of the texture, as a fraction of its height (48 px of
/// 768).
// Q-306
const TITLE_FRACTION: f32 = 1.0 / 16.0;
/// The wrist menu (Q-265): 0.16 m × 0.05 m at 640 × 200 px.
const WRIST_M: Vec2 = Vec2::new(0.16, 0.05);
const WRIST_PX: UVec2 = UVec2::new(640, 200);

/// The module's plugin; `XrTail` adds it after `xr_input`'s.
pub(crate) struct PanelPlugin;

impl Plugin for PanelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PanelGrab>()
            .init_resource::<XrHands>()
            .add_message::<Intent>()
            .add_message::<PanelPointer>()
            .add_message::<PanelGrabEdge>()
            .add_systems(XrSessionCreated, spawn_quads)
            .add_systems(Update, (toggle, grab, place_wrist).chain());
    }
}

/// The main panel's grab (E11 by the title bar, §1.8): one hand, or two resizing it.
#[derive(Resource, Default)]
pub(crate) enum PanelGrab {
    /// Not held.
    #[default]
    Free,
    /// Held by one hand at its aim pose (Q-302's grab pose).
    One { hand: Hand, grab: Grab },
    /// Held by both: the left hand is E11's a, the right b; the pose's scale is the
    /// width in metres.
    Two { grab: TwoHandGrab },
}

/// A new render-target image of `size_px` (bevy's render-to-texture path, D-21).
fn target(images: &mut Assets<Image>, size_px: UVec2) -> Handle<Image> {
    let mut image =
        Image::new_target_texture(size_px.x, size_px.y, TextureFormat::Rgba8UnormSrgb, None);
    image.asset_usage = RenderAssetUsages::default();
    images.add(image)
}

/// A quad's pieces: a unit rectangle (scaled by the transform to `size_m`), facing +z
/// with its texture's top-left at the quad's top-left (Q-210), and an unlit material
/// showing `image`.
fn quad(
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    image: &Handle<Image>,
) -> (Mesh3d, MeshMaterial3d<StandardMaterial>) {
    (
        Mesh3d(meshes.add(Rectangle::new(1.0, 1.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color_texture: Some(image.clone()),
            unlit: true,
            ..Default::default()
        })),
    )
}

/// The transform of a quad of `size_m` at `pose` (its scale is the size, Q-210's
/// `panel_hit` reading `PanelQuad.size_m` instead).
fn quad_transform(pose: Pose, size_m: Vec2) -> Transform {
    Transform {
        translation: pose.translation,
        rotation: pose.rotation,
        scale: Vec3::new(size_m.x, size_m.y, 1.0),
    }
}

/// At session creation, with the `Headset` marker (Q-266): the main panel's image and
/// quad (`WorldPanel`, `MainPanel { follow: Summon }`, `PanelQuad`), hidden until
/// summoned, and `UiSurface::Panel`; the wrist menu's likewise (`WristMenu`), hidden
/// until the palm faces the head. Once: a re-created session (Q-124) keeps them.
// Q-265, Q-266, Q-307
#[allow(clippy::too_many_arguments)]
pub(crate) fn spawn_quads(
    headset: Option<Res<Headset>>,
    existing: Query<(), With<MainPanel>>,
    images: Option<ResMut<Assets<Image>>>,
    meshes: Option<ResMut<Assets<Mesh>>>,
    materials: Option<ResMut<Assets<StandardMaterial>>>,
    surface: Option<ResMut<UiSurface>>,
    mut commands: Commands,
) {
    if headset.is_none() || !existing.is_empty() {
        return;
    }
    let (Some(mut images), Some(mut meshes), Some(mut materials)) = (images, meshes, materials)
    else {
        return;
    };
    let image = target(&mut images, PANEL_PX);
    let entity = commands
        .spawn((
            WorldPanel {
                image: image.clone(),
            },
            MainPanel {
                follow: PanelFollow::Summon,
            },
            PanelQuad {
                size_m: PANEL_M,
                size_px: PANEL_PX,
            },
            quad(&mut meshes, &mut materials, &image),
            quad_transform(Pose::IDENTITY, PANEL_M),
            Visibility::Hidden,
        ))
        .id();
    let wrist = target(&mut images, WRIST_PX);
    commands.spawn((
        WorldPanel {
            image: wrist.clone(),
        },
        WristMenu,
        PanelQuad {
            size_m: WRIST_M,
            size_px: WRIST_PX,
        },
        quad(&mut meshes, &mut materials, &wrist),
        quad_transform(Pose::IDENTITY, WRIST_M),
        Visibility::Hidden,
    ));
    let surface_now = UiSurface::Panel {
        entity,
        size_px: PANEL_PX,
        size_m: PANEL_M,
    };
    match surface {
        Some(mut s) => *s = surface_now,
        None => commands.insert_resource(surface_now),
    }
}

/// The pose of a quad's transform (its scale is its size, not the pose's).
fn pose_of(t: &Transform) -> Pose {
    Pose {
        translation: t.translation,
        rotation: t.rotation,
        scale: 1.0,
    }
}

/// `Toggle(Feature::Panel)` (§1.8): a hidden panel is shown, at the summon pose with
/// `Summon` (where it was left with `WorldLocked`); a visible one is hidden, dropping its
/// grab. Between summons it stays where it is.
// Q-307
pub(crate) fn toggle(
    mut intents: MessageReader<Intent>,
    hands: Res<XrHands>,
    mut held: ResMut<PanelGrab>,
    mut panels: Query<(&MainPanel, &mut Transform, &mut Visibility)>,
) {
    for intent in intents.read() {
        if *intent != Intent::Toggle(Feature::Panel) {
            continue;
        }
        for (panel, mut transform, mut visibility) in &mut panels {
            if *visibility == Visibility::Hidden {
                if panel.follow == PanelFollow::Summon {
                    let pose = summon_pose(hands.head);
                    transform.translation = pose.translation;
                    transform.rotation = pose.rotation;
                }
                *visibility = Visibility::Inherited;
            } else {
                *visibility = Visibility::Hidden;
                *held = PanelGrab::Free;
            }
        }
    }
}

/// A side's hand in `XrHands`.
fn hand_of(hands: &XrHands, hand: Hand) -> Option<&TrackedHand> {
    match hand {
        Hand::Left => hands.left.as_ref(),
        Hand::Right => hands.right.as_ref(),
        Hand::Mouse => None,
    }
}

/// The other side; none for the mouse.
fn other(hand: Hand) -> Option<Hand> {
    match hand {
        Hand::Left => Some(Hand::Right),
        Hand::Right => Some(Hand::Left),
        Hand::Mouse => None,
    }
}

/// What `grab` reads and changes on the main panel's quad.
type PanelParts = (
    Entity,
    &'static mut Transform,
    &'static mut PanelQuad,
    &'static mut WorldPanel,
    &'static Visibility,
    Option<&'static MeshMaterial3d<StandardMaterial>>,
);

/// The texture's size for a width of `width_m` (PRD #452): 1829 px per metre, rounded,
/// the height following 4:3.
// Q-306
pub(crate) fn resized_px(width_m: f32) -> UVec2 {
    let w = (width_m * PX_PER_M).round().max(1.0);
    UVec2::new(w as u32, (w * ASPECT).round().max(1.0) as u32)
}

/// The main panel's grab (§1.8: "grabbing a UI panel by its title bar moves it the same
/// way (two hands resize it)", E11): a grab press on the title-bar strip begins a
/// one-hand grab at the hand's aim pose; the other hand's grab press anywhere on the
/// panel while it is held makes it a two-hand grab, scaling the width between 0.3 and
/// 1.2 m (yaw only, keep upright); a release (`GrabEnd`, which `xr_input` always passes
/// on) of either returns to the other hand, re-creating the image at the new width
/// (1829 px per metre) and updating `PanelQuad` and `UiSurface::Panel`. The pose follows
/// while the holding hands are tracked. Each hand that joins or leaves the grab sends a
/// `PanelGrabEdge`, D-44's grab pulse (Q-T452.4).
// Q-306, Q-T452.4
#[allow(clippy::too_many_arguments)]
pub(crate) fn grab(
    mut pointers: MessageReader<PanelPointer>,
    mut intents: MessageReader<Intent>,
    mut edges: MessageWriter<PanelGrabEdge>,
    hands: Res<XrHands>,
    mut held: ResMut<PanelGrab>,
    mut surface: Option<ResMut<UiSurface>>,
    images: Option<ResMut<Assets<Image>>>,
    materials: Option<ResMut<Assets<StandardMaterial>>>,
    mut panels: Query<PanelParts, (With<MainPanel>, Without<WristMenu>)>,
) {
    let Some((entity, mut transform, mut quad, mut world_panel, visibility, material)) =
        panels.iter_mut().next()
    else {
        pointers.clear();
        intents.clear();
        return;
    };
    let aim = |hand: Hand| hand_of(&hands, hand).filter(|t| t.tracked).map(|t| t.aim);
    let mut ended_two = false;
    // Releases first: a `GrabEnd` this frame ends the hand's hold.
    for intent in intents.read() {
        let Intent::GrabEnd { hand } = *intent else {
            continue;
        };
        match &*held {
            PanelGrab::One { hand: h, .. } if *h == hand => {
                *held = PanelGrab::Free;
                edges.write(PanelGrabEdge(hand));
            }
            PanelGrab::Two { .. } => {
                let Some(rest) = other(hand) else { continue };
                edges.write(PanelGrabEdge(hand));
                *held = match aim(rest) {
                    Some(a) => PanelGrab::One {
                        hand: rest,
                        grab: Grab::begin(a, pose_of(&transform)),
                    },
                    None => {
                        edges.write(PanelGrabEdge(rest));
                        PanelGrab::Free
                    }
                };
                ended_two = true;
            }
            _ => {}
        }
    }
    for p in pointers.read() {
        if p.panel != entity || !p.grab.contains(&true) || *visibility == Visibility::Hidden {
            continue;
        }
        let Some(a) = aim(p.hand) else { continue };
        match &*held {
            PanelGrab::Free if p.pixel.y < quad.size_px.y as f32 * TITLE_FRACTION => {
                *held = PanelGrab::One {
                    hand: p.hand,
                    grab: Grab::begin(a, pose_of(&transform)),
                };
                edges.write(PanelGrabEdge(p.hand));
            }
            PanelGrab::One { hand, .. } if *hand != p.hand => {
                if let Some(b) = aim(*hand) {
                    let (l, r) = if p.hand == Hand::Left { (a, b) } else { (b, a) };
                    *held = PanelGrab::Two {
                        // The pose's scale is the width (E11's s).
                        grab: TwoHandGrab::begin(
                            l,
                            r,
                            Pose {
                                scale: quad.size_m.x,
                                ..pose_of(&transform)
                            },
                        ),
                    };
                    edges.write(PanelGrabEdge(p.hand));
                }
            }
            _ => {}
        }
    }
    // Follow the tracked hands.
    match &*held {
        PanelGrab::Free => {}
        PanelGrab::One { hand, grab } => {
            if let Some(a) = aim(*hand) {
                let p = grab.update(a);
                transform.translation = p.translation;
                transform.rotation = p.rotation;
            }
        }
        PanelGrab::Two { grab } => {
            if let (Some(a), Some(b)) = (aim(Hand::Left), aim(Hand::Right)) {
                let p = grab.update(a, b, &WIDTH_LIMITS, false);
                let size_m = Vec2::new(p.scale, p.scale * ASPECT);
                *transform = quad_transform(p, size_m);
                quad.size_m = size_m;
            }
        }
    }
    // The end of a resize: the image at the new width.
    let size_px = if ended_two {
        resized_px(quad.size_m.x)
    } else {
        quad.size_px
    };
    if size_px != quad.size_px
        && let Some(mut images) = images
    {
        let image = target(&mut images, size_px);
        if let (Some(mut materials), Some(m)) = (materials, material)
            && let Some(mut mat) = materials.get_mut(&m.0)
        {
            mat.base_color_texture = Some(image.clone());
        }
        world_panel.image = image;
        quad.size_px = size_px;
    }
    let scale = Vec3::new(quad.size_m.x, quad.size_m.y, 1.0);
    if transform.scale != scale {
        transform.scale = scale;
    }
    if let Some(surface) = surface.as_deref_mut()
        && let UiSurface::Panel { entity: e, .. } = *surface
        && e == entity
    {
        let now = UiSurface::Panel {
            entity,
            size_px: quad.size_px,
            size_m: quad.size_m,
        };
        if *surface != now {
            *surface = now;
        }
    }
}

/// The palm frame of a hand's grip (Q-305): +x the palm's outward normal. A
/// controller's grip pose has +x away from the palm on the left hand and into it on the
/// right (source-log 287), so the right one turns 180° about its y; a tracked hand's grip
/// is the wrist joint (Q-262), whose +y points out of the back of the hand on both
/// hands (source-log 290), so it turns −90° about its z, taking +x to its −y.
// Q-305
pub(crate) fn palm_frame(side: Hand, hand: &TrackedHand) -> Pose {
    let turn = match (hand.source, side) {
        (HandSource::Controller, Hand::Left) => Quat::IDENTITY,
        (HandSource::Controller, _) => Quat::from_rotation_y(std::f32::consts::PI),
        (HandSource::Hand, _) => Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2),
    };
    Pose {
        rotation: (hand.grip.rotation * turn).normalize(),
        ..hand.grip
    }
}

/// The wrist menu (§1.8, Comfort and accessibility): on the left hand's grip, the
/// right's in left-handed mode, at `wrist_menu_pose`, shown while that hand is tracked
/// and `wrist_menu_visible`.
// Q-265, Q-305
pub(crate) fn place_wrist(
    hands: Res<XrHands>,
    mut wrists: Query<(&mut Transform, &mut Visibility, &PanelQuad), With<WristMenu>>,
) {
    let side = if hands.left_handed {
        Hand::Right
    } else {
        Hand::Left
    };
    let palm = hand_of(&hands, side)
        .filter(|t| t.tracked)
        .map(|t| palm_frame(side, t));
    for (mut transform, mut visibility, quad) in &mut wrists {
        let shown = palm.filter(|g| wrist_menu_visible(*g, hands.head));
        let want = match shown {
            Some(g) => {
                *transform = quad_transform(wrist_menu_pose(g, hands.head), quad.size_m);
                Visibility::Inherited
            }
            None => Visibility::Hidden,
        };
        if *visibility != want {
            *visibility = want;
        }
    }
}

#[cfg(test)]
mod tests {
    use bevy::ecs::message::Messages;

    use super::*;

    fn at(t: Vec3, rotation: Quat) -> Pose {
        Pose {
            translation: t,
            rotation,
            scale: 1.0,
        }
    }

    /// An app with the plugin, the marker and the asset stores, its quads spawned as at
    /// session creation.
    fn app(head: Pose) -> App {
        let mut app = App::new();
        app.insert_resource(Headset)
            .insert_resource(Assets::<Image>::default())
            .insert_resource(Assets::<Mesh>::default())
            .insert_resource(Assets::<StandardMaterial>::default())
            .init_resource::<UiSurface>()
            .add_plugins(PanelPlugin);
        app.world_mut().resource_mut::<XrHands>().head = head;
        app.world_mut().run_schedule(XrSessionCreated);
        app
    }

    fn main_panel(app: &mut App) -> (Entity, Transform, Visibility) {
        let mut q = app
            .world_mut()
            .query_filtered::<(Entity, &Transform, &Visibility), With<MainPanel>>();
        let v: Vec<_> = q.iter(app.world()).map(|(e, t, v)| (e, *t, *v)).collect();
        assert_eq!(v.len(), 1);
        v[0]
    }

    fn toggle_panel(app: &mut App) {
        app.world_mut()
            .resource_mut::<Messages<Intent>>()
            .write(Intent::Toggle(Feature::Panel));
        app.update();
    }

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).abs().max_element() <= 1e-6
    }

    /// §1.8 User interface: the menu button, the system gesture or the wrist menu's panel
    /// button summon the panel 1 m in front of the head, 0.2 m below eye height, facing
    /// the user; it stays where it is until summoned again; summoning while it is
    /// visible hides it.
    #[test]
    fn toggle_summons_hides_and_leaves_the_panel_where_it_was() {
        let mut app = app(at(Vec3::new(0.0, 1.6, 0.0), Quat::IDENTITY));
        let (entity, _, v) = main_panel(&mut app);
        assert_eq!(v, Visibility::Hidden, "hidden until summoned");
        assert_eq!(
            *app.world().resource::<UiSurface>(),
            UiSurface::Panel {
                entity,
                size_px: UVec2::new(1024, 768),
                size_m: Vec2::new(0.56, 0.42),
            }
        );

        toggle_panel(&mut app);
        let (_, t, v) = main_panel(&mut app);
        assert_ne!(v, Visibility::Hidden);
        assert!(close(t.translation, Vec3::new(0.0, 1.4, -1.0)), "{t:?}");
        assert!(close(t.rotation * Vec3::Z, Vec3::Z), "facing the head");

        // The head walks and turns: the panel stays world-locked.
        app.world_mut().resource_mut::<XrHands>().head = at(
            Vec3::new(2.0, 1.6, 1.0),
            Quat::from_rotation_y(90f32.to_radians()),
        );
        app.update();
        let (_, t2, v) = main_panel(&mut app);
        assert_ne!(v, Visibility::Hidden);
        assert_eq!(t2.translation, t.translation);

        // Summoning while visible hides it, where it is.
        toggle_panel(&mut app);
        let (_, t3, v) = main_panel(&mut app);
        assert_eq!(v, Visibility::Hidden);
        assert_eq!(t3.translation, t.translation);

        // Summoned again: in front of the head as it is now, 1 m along −x.
        toggle_panel(&mut app);
        let (_, t4, v) = main_panel(&mut app);
        assert_ne!(v, Visibility::Hidden);
        assert!(close(t4.translation, Vec3::new(1.0, 1.4, 1.0)), "{t4:?}");
        assert!(close(t4.rotation * Vec3::Z, Vec3::X), "facing the head");
    }

    fn hand(aim: Pose, grip: Pose, source: HandSource) -> TrackedHand {
        TrackedHand {
            aim,
            grip,
            pinch: 0.0,
            pinch_middle: 0.0,
            squeeze: 0.0,
            trigger: 0.0,
            stick: Vec2::ZERO,
            primary: false,
            secondary: false,
            menu: false,
            source,
            tracked: true,
        }
    }

    fn set_hand(app: &mut App, side: Hand, h: Option<TrackedHand>) {
        let mut hands = app.world_mut().resource_mut::<XrHands>();
        match side {
            Hand::Left => hands.left = h,
            _ => hands.right = h,
        }
    }

    fn press(app: &mut App, side: Hand, panel: Entity, pixel: Vec2) {
        app.world_mut()
            .resource_mut::<Messages<PanelPointer>>()
            .write(PanelPointer {
                hand: side,
                panel,
                pixel,
                select: vec![],
                grab: vec![true],
                scroll_px: 0.0,
            });
    }

    fn release(app: &mut App, side: Hand) {
        app.world_mut()
            .resource_mut::<Messages<Intent>>()
            .write(Intent::GrabEnd { hand: side });
    }

    fn quad_of(app: &mut App) -> PanelQuad {
        let mut q = app
            .world_mut()
            .query_filtered::<&PanelQuad, With<MainPanel>>();
        *q.single(app.world()).expect("one panel")
    }

    /// E11's one-hand grab on the title bar (§1.8): the panel keeps its place in the
    /// hand; a grab press below the strip moves nothing.
    #[test]
    fn grab_by_the_title_bar_moves_the_panel_with_the_hand() {
        let mut app = app(at(Vec3::new(0.0, 1.6, 0.0), Quat::IDENTITY));
        toggle_panel(&mut app);
        let (panel, t0, _) = main_panel(&mut app);
        let aim0 = at(Vec3::new(0.2, 1.2, -0.3), Quat::IDENTITY);
        set_hand(
            &mut app,
            Hand::Right,
            Some(hand(aim0, aim0, HandSource::Controller)),
        );
        // Below the 48 px strip: no grab.
        press(&mut app, Hand::Right, panel, Vec2::new(500.0, 60.0));
        app.update();
        let moved = at(Vec3::new(0.5, 1.3, -0.3), Quat::IDENTITY);
        set_hand(
            &mut app,
            Hand::Right,
            Some(hand(moved, moved, HandSource::Controller)),
        );
        app.update();
        assert_eq!(main_panel(&mut app).1.translation, t0.translation);

        // On the strip: the panel follows the hand's 0.3 m and 0.1 m.
        set_hand(
            &mut app,
            Hand::Right,
            Some(hand(aim0, aim0, HandSource::Controller)),
        );
        press(&mut app, Hand::Right, panel, Vec2::new(500.0, 20.0));
        app.update();
        set_hand(
            &mut app,
            Hand::Right,
            Some(hand(moved, moved, HandSource::Controller)),
        );
        app.update();
        let t1 = main_panel(&mut app).1;
        assert!(
            close(t1.translation, t0.translation + Vec3::new(0.3, 0.1, 0.0)),
            "{t1:?}"
        );
        // Released, it stays.
        release(&mut app, Hand::Right);
        app.update();
        set_hand(
            &mut app,
            Hand::Right,
            Some(hand(aim0, aim0, HandSource::Controller)),
        );
        app.update();
        assert_eq!(main_panel(&mut app).1.translation, t1.translation);
    }

    #[derive(Resource, Default)]
    struct Edges(Vec<Hand>);

    fn collect_edges(mut r: MessageReader<PanelGrabEdge>, mut seen: ResMut<Edges>) {
        seen.0.extend(r.read().map(|e| e.0));
    }

    /// D-44 ("at the start and the end of a grab"), Q-T452.4: a title-bar grab and its
    /// release each send the hand's `PanelGrabEdge`; a press below the strip sends none.
    #[test]
    fn a_title_bar_grab_and_its_release_send_grab_edges() {
        let mut app = app(at(Vec3::new(0.0, 1.6, 0.0), Quat::IDENTITY));
        app.init_resource::<Edges>()
            .add_systems(Update, collect_edges.after(grab));
        toggle_panel(&mut app);
        let (panel, _, _) = main_panel(&mut app);
        let aim0 = at(Vec3::new(0.2, 1.2, -0.3), Quat::IDENTITY);
        set_hand(
            &mut app,
            Hand::Right,
            Some(hand(aim0, aim0, HandSource::Controller)),
        );
        press(&mut app, Hand::Right, panel, Vec2::new(500.0, 60.0));
        app.update();
        assert!(
            app.world().resource::<Edges>().0.is_empty(),
            "below the strip"
        );
        release(&mut app, Hand::Right);
        app.update();
        press(&mut app, Hand::Right, panel, Vec2::new(500.0, 20.0));
        app.update();
        assert_eq!(
            app.world().resource::<Edges>().0,
            [Hand::Right],
            "the start"
        );
        release(&mut app, Hand::Right);
        app.update();
        assert_eq!(
            app.world().resource::<Edges>().0,
            [Hand::Right, Hand::Right],
            "the end"
        );
    }

    /// §1.8: two hands resize the panel between 0.3 m and 1.2 m wide (E11's ratio of the
    /// hands' distance), the height following 4:3; at the end of the resize the image is
    /// re-created at 1829 px per metre and `UiSurface::Panel` follows.
    #[test]
    fn grab_with_two_hands_resizes_and_recreates_the_image() {
        let mut app = app(at(Vec3::new(0.0, 1.6, 0.0), Quat::IDENTITY));
        toggle_panel(&mut app);
        let (panel, _, _) = main_panel(&mut app);
        let first_image = app
            .world_mut()
            .query::<&WorldPanel>()
            .get(app.world(), panel)
            .expect("panel")
            .image
            .clone();
        let a = |x: f32| at(Vec3::new(x, 1.4, -0.5), Quat::IDENTITY);
        set_hand(
            &mut app,
            Hand::Left,
            Some(hand(a(-0.1), a(-0.1), HandSource::Controller)),
        );
        set_hand(
            &mut app,
            Hand::Right,
            Some(hand(a(0.1), a(0.1), HandSource::Controller)),
        );
        press(&mut app, Hand::Left, panel, Vec2::new(100.0, 10.0));
        app.update();
        press(&mut app, Hand::Right, panel, Vec2::new(900.0, 400.0));
        app.update();
        // The hands 1.5 × apart: 0.84 m wide.
        set_hand(
            &mut app,
            Hand::Left,
            Some(hand(a(-0.15), a(-0.15), HandSource::Controller)),
        );
        set_hand(
            &mut app,
            Hand::Right,
            Some(hand(a(0.15), a(0.15), HandSource::Controller)),
        );
        app.update();
        let q = quad_of(&mut app);
        assert!((q.size_m.x - 0.84).abs() < 1e-5, "{q:?}");
        assert!((q.size_m.y - 0.63).abs() < 1e-5, "{q:?}");
        assert_eq!(
            q.size_px,
            UVec2::new(1024, 768),
            "the image waits for the release"
        );
        // 10 × apart: clamped at 1.2 m.
        set_hand(
            &mut app,
            Hand::Left,
            Some(hand(a(-1.0), a(-1.0), HandSource::Controller)),
        );
        set_hand(
            &mut app,
            Hand::Right,
            Some(hand(a(1.0), a(1.0), HandSource::Controller)),
        );
        app.update();
        assert!((quad_of(&mut app).size_m.x - 1.2).abs() < 1e-5);
        release(&mut app, Hand::Right);
        app.update();
        let q = quad_of(&mut app);
        // 1.2 × 1829 = 2194.8 → 2195; × 0.75 = 1646.25 → 1646.
        assert_eq!(q.size_px, UVec2::new(2195, 1646));
        let image = app
            .world_mut()
            .query::<&WorldPanel>()
            .get(app.world(), panel)
            .expect("panel")
            .image
            .clone();
        assert_ne!(image, first_image);
        let size = app
            .world()
            .resource::<Assets<Image>>()
            .get(&image)
            .expect("image")
            .size();
        assert_eq!(size, UVec2::new(2195, 1646));
        assert_eq!(
            *app.world().resource::<UiSurface>(),
            UiSurface::Panel {
                entity: panel,
                size_px: q.size_px,
                size_m: q.size_m,
            }
        );
    }

    /// `resized_px` at the default width gives the default texture.
    #[test]
    fn resized_px_keeps_1024_by_768_at_0_56_m() {
        assert_eq!(resized_px(0.56), UVec2::new(1024, 768));
        assert_eq!(resized_px(0.3), UVec2::new(549, 412));
    }

    /// Q-305: the palm frame's +x is the palm's outward normal: the left controller's
    /// grip +x, the right one's −x (source-log 287), a tracked hand's wrist −y on both
    /// sides (source-log 290).
    #[test]
    fn palm_frame_points_x_out_of_the_palm() {
        let grip = at(Vec3::new(1.0, 2.0, 3.0), Quat::from_rotation_x(0.4));
        let out = |side, source| palm_frame(side, &hand(grip, grip, source)).rotation * Vec3::X;
        let gx = grip.rotation * Vec3::X;
        let gy = grip.rotation * Vec3::Y;
        assert!(close(out(Hand::Left, HandSource::Controller), gx));
        assert!(close(out(Hand::Right, HandSource::Controller), -gx));
        assert!(close(out(Hand::Left, HandSource::Hand), -gy));
        assert!(close(out(Hand::Right, HandSource::Hand), -gy));
        assert_eq!(
            palm_frame(Hand::Left, &hand(grip, grip, HandSource::Hand)).translation,
            grip.translation
        );
    }

    fn wrist(app: &mut App) -> (Transform, Visibility) {
        let mut q = app
            .world_mut()
            .query_filtered::<(&Transform, &Visibility), With<WristMenu>>();
        let (t, v) = q.single(app.world()).expect("one wrist menu");
        (*t, *v)
    }

    /// §1.8: the wrist menu rides the left hand, the right in left-handed mode, and
    /// shows while its palm faces the head.
    #[test]
    fn place_wrist_follows_the_menu_hand_and_the_palm() {
        let head = at(Vec3::new(0.0, 1.6, 0.0), Quat::IDENTITY);
        let mut app = app(head);
        // A left controller 0.4 m ahead of the head, its palm (+x) turned toward it.
        let facing = at(
            Vec3::new(0.0, 1.6, -0.4),
            Quat::from_rotation_y(-90f32.to_radians()),
        );
        set_hand(
            &mut app,
            Hand::Left,
            Some(hand(facing, facing, HandSource::Controller)),
        );
        app.update();
        let (t, v) = wrist(&mut app);
        assert_eq!(v, Visibility::Inherited);
        assert!(close(t.translation, Vec3::new(0.0, 1.6, -0.32)), "{t:?}");
        assert!(close(t.scale, Vec3::new(0.16, 0.05, 1.0)));
        // Left-handed: the right hand carries it; the left is ignored.
        app.world_mut().resource_mut::<XrHands>().left_handed = true;
        app.update();
        assert_eq!(wrist(&mut app).1, Visibility::Hidden);
        // The right controller's palm is its −x: turned +90° about y it faces the head.
        let right = at(
            Vec3::new(0.0, 1.6, -0.4),
            Quat::from_rotation_y(90f32.to_radians()),
        );
        set_hand(
            &mut app,
            Hand::Right,
            Some(hand(right, right, HandSource::Controller)),
        );
        app.update();
        assert_eq!(wrist(&mut app).1, Visibility::Inherited);
        // The palm turned away: hidden.
        set_hand(
            &mut app,
            Hand::Right,
            Some(hand(facing, facing, HandSource::Controller)),
        );
        app.update();
        assert_eq!(wrist(&mut app).1, Visibility::Hidden);
    }
}
