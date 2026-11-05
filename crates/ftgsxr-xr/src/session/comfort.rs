// SPDX-License-Identifier: Apache-2.0

//! Locomotion's comfort options on the headset (§1.8 Locomotion, Comfort and
//! accessibility; Q-269): the vignette during smooth movement or turning, the 150 ms fade
//! on `Intent::Teleport`, the fade skipped and the vignette's ramps dropped (it shows and
//! clears at once) under `[locomotion] reduced_motion`, and the seated height offset
//! added to the rig's y.
//!
//! Each eye gets two child quads, the vignette (a generated radial alpha texture) and the
//! fade (plain black), on a render layer only that eye sees, so the left eye's quads never
//! cross the right eye's view. The quads exist where the mesh, material and image stores
//! do (the player's app); the ramps run either way.

use bevy::app::{App, Plugin, PostUpdate, Update};
use bevy::asset::{Assets, Handle, RenderAssetUsages};
use bevy::camera::visibility::{RenderLayers, Visibility};
use bevy::color::{Alpha, Color};
use bevy::ecs::message::MessageReader;
use bevy::image::Image;
use bevy::math::Vec3;
use bevy::math::primitives::Rectangle;
use bevy::mesh::{Mesh, Mesh3d};
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::prelude::{
    Added, AlphaMode, ChildOf, Commands, Component, Entity, IntoScheduleConfigs, Query, Res,
    ResMut, Resource, Transform, TransformSystems, With, Without,
};
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use bevy::time::Time;
use bevy_mod_openxr::render::locate_views;
use bevy_mod_openxr::resources::OxrViews;
use bevy_mod_xr::camera::XrCamera;
use ftgsxr_app::{Headset, Intent, Locomotion, PlayerRig, SettingsRes};

// Q-269
/// The vignette's darkest alpha, at and beyond the edge of the half field of view.
const VIGNETTE_ALPHA: f32 = 0.8;
/// Where the vignette starts, as a fraction of the half field of view: the outer 30 %.
const VIGNETTE_INNER: f32 = 0.7;
/// The vignette's fade-in and fade-out time, seconds.
const VIGNETTE_RAMP_S: f32 = 0.1;
/// The teleport fade's length, seconds.
const FADE_S: f32 = 0.15;

// Q-300
/// The vignette quad's distance ahead of the eye, metres (past the eye's near plane of
/// 0.01 m, Q-87).
const VIGNETTE_DISTANCE_M: f32 = 0.05;
/// The fade quad's distance, metres, in front of the vignette.
const FADE_DISTANCE_M: f32 = 0.04;
/// The radial texture's side, texels.
const VIGNETTE_TEXELS: u32 = 128;

/// The vignette's ramp: its level in [0, 1] moves toward 1 while smooth motion is on and
/// toward 0 otherwise, over `VIGNETTE_RAMP_S`, or at once under reduced motion (§1.8:
/// "removes the animated transitions").
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct VignetteRamp {
    pub(crate) level: f32,
}

impl VignetteRamp {
    /// One frame of `dt` seconds, `instant` under reduced motion; the quad's alpha after
    /// it.
    // Q-269
    pub(crate) fn step(&mut self, moving: bool, dt: f32, instant: bool) -> f32 {
        let d = if instant { 1.0 } else { dt / VIGNETTE_RAMP_S };
        self.level = if moving {
            (self.level + d).min(1.0)
        } else {
            (self.level - d).max(0.0)
        };
        VIGNETTE_ALPHA * self.level
    }
}

/// The teleport fade: black at the teleport, clearing linearly over `FADE_S` (the rig
/// has already jumped in the frame the intent is applied).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Fade {
    pub(crate) remaining_s: f32,
}

impl Fade {
    /// One frame of `dt` seconds, `teleported` when the frame applied a teleport; the
    /// fade quad's alpha for the frame.
    // Q-269, Q-301
    pub(crate) fn step(&mut self, teleported: bool, dt: f32) -> f32 {
        if teleported {
            self.remaining_s = FADE_S;
            return 1.0;
        }
        self.remaining_s = (self.remaining_s - dt).max(0.0);
        self.remaining_s / FADE_S
    }
}

/// The vignette's alpha at a point of the eye's image plane, `r` its radial distance in
/// units of the half field of view: 0 inside the inner 70 %, rising smoothly to 1 at the
/// edge and beyond (the material scales it to `VIGNETTE_ALPHA`).
// Q-269, Q-300
pub(crate) fn vignette_alpha(r: f32) -> f32 {
    let x = ((r - VIGNETTE_INNER) / (1.0 - VIGNETTE_INNER)).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Whether a frame's intents move or turn the rig smoothly (§1.8 Locomotion): a
/// `Locomote` with a move, or with a turn while smooth turn is on; nothing while stick
/// locomotion is off (D-17).
pub(crate) fn smooth_motion<'a>(
    intents: impl IntoIterator<Item = &'a Intent>,
    loco: &Locomotion,
) -> bool {
    loco.enabled
        && intents.into_iter().any(|i| match *i {
            Intent::Locomote { move_xz, turn } => {
                move_xz != bevy::math::Vec2::ZERO || (!loco.snap && turn != 0.0)
            }
            _ => false,
        })
}

/// The height offset the rig should carry: `height_offset_m` in VR mode, none while stick
/// locomotion is off (passthrough mode, where the stage must stay the real room's).
// Q-299
pub(crate) fn height_offset(loco: &Locomotion) -> f32 {
    if loco.enabled {
        loco.height_offset_m
    } else {
        0.0
    }
}

/// The comfort state: the ramps, the offset applied to the rig so far, and the shared
/// materials once made.
#[derive(Resource, Default)]
struct Comfort {
    vignette: VignetteRamp,
    fade: Fade,
    applied_offset_m: f32,
    assets: Option<QuadAssets>,
}

#[derive(Clone)]
struct QuadAssets {
    quad: Handle<Mesh>,
    vignette: Handle<StandardMaterial>,
    fade: Handle<StandardMaterial>,
}

/// One of an eye's comfort quads.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
enum ComfortQuad {
    Vignette,
    Fade,
}

/// `session`'s comfort systems.
pub(crate) struct ComfortPlugin;

impl Plugin for ComfortPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<Intent>()
            .init_resource::<Comfort>()
            .add_systems(Update, (apply_height_offset, run_ramps))
            .add_systems(
                PostUpdate,
                (spawn_quads, place_quads)
                    .chain()
                    .after(locate_views)
                    .before(TransformSystems::Propagate),
            );
    }
}

/// The settings' `reduced_motion` (Q-127's reading of the settings).
fn reduced_motion(settings: Option<&SettingsRes>) -> bool {
    settings.is_some_and(|s| s.0.locomotion.reduced_motion)
}

/// Adds the change of the wanted height offset to the rig's y (§1.8 Locomotion's seated
/// option), so E12's motion, which keeps the rig's height, carries it. Only on the
/// headset (the `Headset` marker, Q-266): the desktop, `play --xr` included, has no
/// offset (Q-299, Q-318).
// Q-299, Q-318
fn apply_height_offset(
    loco: Option<Res<Locomotion>>,
    headset: Option<Res<Headset>>,
    mut comfort: ResMut<Comfort>,
    mut rigs: Query<&mut Transform, With<PlayerRig>>,
) {
    let wanted = match (headset, loco.as_deref()) {
        (Some(_), Some(l)) => height_offset(l),
        _ => 0.0,
    };
    let delta = wanted - comfort.applied_offset_m;
    if delta == 0.0 {
        return;
    }
    let Some(mut rig) = rigs.iter_mut().next() else {
        return;
    };
    rig.translation.y += delta;
    comfort.applied_offset_m = wanted;
}

/// Steps the vignette and the fade from the frame's intents and writes the shared
/// materials' alphas; under reduced motion the fade stays at 0 and the vignette shows
/// and clears without its ramps (Q-269); the vignette stays at 0 while
/// `Locomotion.vignette` is off.
fn run_ramps(
    mut intents: MessageReader<Intent>,
    loco: Option<Res<Locomotion>>,
    settings: Option<Res<SettingsRes>>,
    time: Res<Time>,
    mut comfort: ResMut<Comfort>,
    materials: Option<ResMut<Assets<StandardMaterial>>>,
) {
    let default = Locomotion::default();
    let loco = loco.as_deref().unwrap_or(&default);
    let frame: Vec<Intent> = intents.read().copied().collect();
    let reduced = reduced_motion(settings.as_deref());
    let moving = loco.vignette && smooth_motion(&frame, loco);
    let teleported =
        !reduced && loco.enabled && frame.iter().any(|i| matches!(i, Intent::Teleport { .. }));
    let dt = time.delta_secs();
    let vignette = comfort.vignette.step(moving, dt, reduced);
    let fade = comfort.fade.step(teleported, dt);
    let (Some(mut materials), Some(assets)) = (materials, comfort.assets.clone()) else {
        return;
    };
    for (handle, alpha) in [(&assets.vignette, vignette), (&assets.fade, fade)] {
        if let Some(mut m) = materials.get_mut(handle)
            && m.base_color.alpha() != alpha
        {
            m.base_color.set_alpha(alpha);
        }
    }
}

/// The radial alpha texture: black, its alpha `vignette_alpha` of the distance from the
/// center in units of the half side.
fn vignette_image() -> Image {
    let n = VIGNETTE_TEXELS;
    let mut data = Vec::with_capacity((n * n * 4) as usize);
    for y in 0..n {
        for x in 0..n {
            let c = |i: u32| (i as f32 + 0.5) / n as f32 * 2.0 - 1.0;
            let r = (c(x).powi(2) + c(y).powi(2)).sqrt();
            let a = (vignette_alpha(r) * 255.0).round() as u8;
            data.extend_from_slice(&[0, 0, 0, a]);
        }
    }
    Image::new(
        Extent3d {
            width: n,
            height: n,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    )
}

/// A comfort material: unlit, blended, black with alpha 0 until a ramp raises it.
fn quad_material(texture: Option<Handle<Image>>) -> StandardMaterial {
    StandardMaterial {
        base_color: Color::BLACK.with_alpha(0.0),
        base_color_texture: texture,
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        ..StandardMaterial::default()
    }
}

/// The render layer only eye `index` sees.
fn eye_layer(index: u32) -> usize {
    1 + index as usize
}

/// Gives each new eye its own layer beside the default one and its two quads, once the
/// stores exist; the eyes and their quads go with each session.
#[allow(clippy::too_many_arguments)]
fn spawn_quads(
    eyes: Query<(Entity, &XrCamera), Added<XrCamera>>,
    meshes: Option<ResMut<Assets<Mesh>>>,
    materials: Option<ResMut<Assets<StandardMaterial>>>,
    images: Option<ResMut<Assets<Image>>>,
    mut comfort: ResMut<Comfort>,
    mut commands: Commands,
) {
    if eyes.is_empty() {
        return;
    }
    let (Some(mut meshes), Some(mut materials), Some(mut images)) = (meshes, materials, images)
    else {
        return;
    };
    let assets = comfort
        .assets
        .get_or_insert_with(|| QuadAssets {
            quad: meshes.add(Rectangle::new(1.0, 1.0)),
            vignette: materials.add(quad_material(Some(images.add(vignette_image())))),
            fade: materials.add(quad_material(None)),
        })
        .clone();
    for (eye, index) in &eyes {
        let layer = eye_layer(index.0);
        commands
            .entity(eye)
            .insert(RenderLayers::layer(0).with(layer));
        for (kind, material) in [
            (ComfortQuad::Vignette, assets.vignette.clone()),
            (ComfortQuad::Fade, assets.fade.clone()),
        ] {
            commands.spawn((
                kind,
                Mesh3d(assets.quad.clone()),
                MeshMaterial3d(material),
                Transform::default(),
                Visibility::Hidden,
                RenderLayers::layer(layer),
                ChildOf(eye),
            ));
        }
    }
}

/// The quad's local transform covering the eye's field of view `fov` at `distance`: the
/// unit quad centered on the view axis, its half sides the larger tangent of each axis
/// times the distance, so the radial texture's unit circle reaches the wider edge.
// Q-300
fn quad_transform(fov: &openxr::Fovf, distance: f32) -> Transform {
    let half_x = fov.angle_left.tan().abs().max(fov.angle_right.tan().abs());
    let half_y = fov.angle_down.tan().abs().max(fov.angle_up.tan().abs());
    Transform {
        translation: Vec3::new(0.0, 0.0, -distance),
        rotation: bevy::math::Quat::IDENTITY,
        scale: Vec3::new(2.0 * half_x * distance, 2.0 * half_y * distance, 1.0),
    }
}

/// Sizes each quad to its eye's located field of view and shows it while its alpha is
/// above 0.
fn place_quads(
    eyes: Query<&XrCamera>,
    views: Option<Res<OxrViews>>,
    comfort: Res<Comfort>,
    materials: Option<Res<Assets<StandardMaterial>>>,
    mut quads: Query<(&ComfortQuad, &ChildOf, &mut Transform, &mut Visibility), Without<XrCamera>>,
) {
    let (Some(views), Some(materials), Some(assets)) = (views, materials, comfort.assets.as_ref())
    else {
        return;
    };
    let alpha =
        |h: &Handle<StandardMaterial>| materials.get(h).map_or(0.0, |m| m.base_color.alpha());
    let (vignette, fade) = (alpha(&assets.vignette), alpha(&assets.fade));
    for (kind, parent, mut t, mut vis) in &mut quads {
        let Ok(eye) = eyes.get(parent.parent()) else {
            continue;
        };
        let Some(view) = views.get(eye.0 as usize) else {
            continue;
        };
        let (distance, a) = match kind {
            ComfortQuad::Vignette => (VIGNETTE_DISTANCE_M, vignette),
            ComfortQuad::Fade => (FADE_DISTANCE_M, fade),
        };
        let placed = quad_transform(&view.fov, distance);
        if *t != placed {
            *t = placed;
        }
        let shown = if a > 0.0 {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        if *vis != shown {
            *vis = shown;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::math::Vec2;

    fn loco() -> Locomotion {
        Locomotion::default()
    }

    // Q-269: the vignette ramps to alpha 0.8 in 100 ms and back to 0 in 100 ms.
    #[test]
    fn vignette_ramp_reaches_0_8_in_100_ms_and_clears_in_100_ms() {
        let mut v = VignetteRamp::default();
        assert!((v.step(true, 0.05, false) - 0.4).abs() <= 1e-6);
        assert!((v.step(true, 0.05, false) - 0.8).abs() <= 1e-6);
        assert!(
            (v.step(true, 0.05, false) - 0.8).abs() <= 1e-6,
            "held at 0.8"
        );
        assert!((v.step(false, 0.05, false) - 0.4).abs() <= 1e-6);
        assert_eq!(v.step(false, 0.05, false), 0.0);
        assert_eq!(v.step(false, 0.05, false), 0.0);
    }

    // Q-269 (owner, 2026-11-04): under reduced motion the vignette stays, without its
    // ramps: full at the first frame of smooth motion, clear at the first frame without.
    #[test]
    fn vignette_under_reduced_motion_shows_and_clears_at_once() {
        let mut v = VignetteRamp::default();
        assert_eq!(v.step(true, 0.011, true), 0.8);
        assert_eq!(v.step(true, 0.011, true), 0.8);
        assert_eq!(v.step(false, 0.011, true), 0.0);
    }

    // Q-269, Q-301: black at the teleport, clear 150 ms later, linear between.
    #[test]
    fn fade_is_black_at_the_teleport_and_clear_after_150_ms() {
        let mut f = Fade::default();
        assert_eq!(f.step(false, 0.01), 0.0, "nothing before a teleport");
        assert_eq!(f.step(true, 0.01), 1.0);
        assert!((f.step(false, 0.075) - 0.5).abs() <= 1e-5);
        assert!(f.step(false, 0.075).abs() <= 1e-5);
        assert_eq!(f.step(false, 0.075), 0.0);
    }

    // Q-269: the outer 30 % of the half field of view, to full alpha at the edge.
    #[test]
    fn vignette_alpha_covers_the_outer_30_percent() {
        assert_eq!(vignette_alpha(0.0), 0.0);
        assert_eq!(vignette_alpha(0.7), 0.0);
        assert!((vignette_alpha(0.85) - 0.5).abs() <= 1e-6);
        assert_eq!(vignette_alpha(1.0), 1.0);
        assert_eq!(vignette_alpha(1.4), 1.0, "the corners beyond the edge");
    }

    // §1.8 Locomotion: smooth movement or smooth turning, not a snap turn, and nothing
    // in passthrough mode (D-17).
    #[test]
    fn smooth_motion_is_a_move_or_a_smooth_turn() {
        let mut l = loco();
        let moving = Intent::Locomote {
            move_xz: Vec2::new(0.0, 0.5),
            turn: 0.0,
        };
        let turning = Intent::Locomote {
            move_xz: Vec2::ZERO,
            turn: 0.5,
        };
        let still = Intent::Locomote {
            move_xz: Vec2::ZERO,
            turn: 0.0,
        };
        l.snap = true;
        assert!(smooth_motion(&[moving], &l));
        assert!(!smooth_motion(&[turning], &l), "snap turn");
        assert!(!smooth_motion(&[still], &l));
        assert!(!smooth_motion(&[Intent::SnapTurn { steps: 1 }], &l));
        l.snap = false;
        assert!(smooth_motion(&[turning], &l), "smooth turn");
        l.enabled = false;
        assert!(!smooth_motion(&[moving], &l), "passthrough mode");
    }

    // §1.8 Locomotion, Q-299: the offset in VR mode, none in passthrough mode.
    #[test]
    fn height_offset_only_while_stick_locomotion_is_on() {
        let mut l = loco();
        l.height_offset_m = -0.3;
        assert_eq!(height_offset(&l), -0.3);
        l.enabled = false;
        assert_eq!(height_offset(&l), 0.0);
    }

    // §1.8 Locomotion, Q-299: the offset is added to the rig's y, its change applied
    // once, whatever else moved the rig.
    #[test]
    fn apply_height_offset_adds_the_change_to_the_rigs_y() {
        let mut app = App::new();
        let mut l = loco();
        l.height_offset_m = 0.4;
        app.insert_resource(l)
            .insert_resource(Headset)
            .init_resource::<Comfort>()
            .add_systems(Update, apply_height_offset);
        let rig = app
            .world_mut()
            .spawn((PlayerRig, Transform::from_xyz(1.0, 0.2, -2.0)))
            .id();
        app.update();
        app.update();
        let y = |app: &App| app.world().get::<Transform>(rig).unwrap().translation;
        assert!((y(&app) - Vec3::new(1.0, 0.6, -2.0)).length() <= 1e-6);
        app.world_mut().resource_mut::<Locomotion>().height_offset_m = -0.1;
        app.update();
        assert!((y(&app).y - 0.1).abs() <= 1e-6);
        app.world_mut().resource_mut::<Locomotion>().enabled = false;
        app.update();
        assert!(
            (y(&app).y - 0.2).abs() <= 1e-6,
            "removed in passthrough mode"
        );
    }

    // Q-299 ("the desktop has no offset"), Q-318: without the `Headset` marker
    // (`play --xr`) the rig keeps its height.
    #[test]
    fn apply_height_offset_needs_the_headset_marker() {
        let mut app = App::new();
        let mut l = loco();
        l.height_offset_m = 0.4;
        app.insert_resource(l)
            .init_resource::<Comfort>()
            .add_systems(Update, apply_height_offset);
        let rig = app
            .world_mut()
            .spawn((PlayerRig, Transform::from_xyz(1.0, 0.2, -2.0)))
            .id();
        app.update();
        app.update();
        let t = app.world().get::<Transform>(rig).unwrap().translation;
        assert_eq!(t, Vec3::new(1.0, 0.2, -2.0));
    }

    #[test]
    fn quad_transform_spans_the_wider_tangent_of_each_axis() {
        let fov = openxr::Fovf {
            angle_left: -0.9,
            angle_right: 0.8,
            angle_down: -0.85,
            angle_up: 0.75,
        };
        let t = quad_transform(&fov, 0.05);
        assert_eq!(t.translation, Vec3::new(0.0, 0.0, -0.05));
        assert!((t.scale.x - 2.0 * 0.9f32.tan() * 0.05).abs() <= 1e-6);
        assert!((t.scale.y - 2.0 * 0.85f32.tan() * 0.05).abs() <= 1e-6);
    }
}
