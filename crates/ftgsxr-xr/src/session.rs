// SPDX-License-Identifier: Apache-2.0

//! The session, reference spaces, eye cameras and the rig binding (§1.8 Session and
//! spaces, Locomotion, D-24, §1.14 Frames, §1.15 E-XR-RUNTIME and E-XR-LOST, §2.4 Frame
//! loop hooks).
//!
//! bevy\_mod\_openxr runs the session itself (its init, session and render plugins,
//! composed by `plugin`); this module replaces the crate's reference-space plugin with
//! D-24's chain, makes each eye camera the crate spawns a `SplatCamera`, restarts the
//! session after a loss, makes the tracking root a child of `PlayerRig` (so E12's rig
//! motion carries the eyes and the hands), and re-places the content group on a
//! reference-space change. `comfort` adds the vignette, the teleport fade and the height
//! offset.

use bevy::app::{App, Last, Plugin, PreUpdate, Update};
use bevy::camera::{Camera, Projection};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::ecs::message::{MessageReader, MessageWriter};
use bevy::log::{info, warn};
use bevy::math::{Isometry3d, Mat4};
use bevy::prelude::{
    Added, ChildOf, Commands, Entity, GlobalTransform, IntoScheduleConfigs, Query, Res, ResMut,
    Resource, Transform, With, Without,
};
use bevy::render::RenderApp;
use bevy::render::extract_resource::ExtractResourcePlugin;
use bevy_mod_openxr::exts::OxrEnabledExtensions;
use bevy_mod_openxr::poll_events::{OxrEventHandlerExt, OxrEventIn};
use bevy_mod_openxr::session::OxrSession;
use bevy_mod_xr::camera::{XrCamera, XrProjection};
use bevy_mod_xr::session::{
    XrCreateSessionMessage, XrPreDestroySession, XrSessionCreated, XrSessionDestroyedMessage,
    XrState, XrStateChanged, XrTrackingRoot,
};
use bevy_mod_xr::spaces::{XrPrimaryReferenceSpace, XrReferenceSpace};
use ftgsxr_app::{ContentGroup, PlayerRig, SettingsRes};
use ftgsxr_core::{Pose, Quat, Vec3};
use ftgsxr_io::Settings;
use ftgsxr_render::{OutputSpace, SortMode, SplatCamera, SplatTierState};
use openxr::ReferenceSpaceType;

use crate::XrError;
use crate::error::{log_error, log_runtime, log_warning};

mod comfort;

/// The height D-24's LOCAL fallback assumes the floor below the session's origin, metres.
const LOCAL_FLOOR_DROP_M: f32 = 1.5;

/// The eye projection's near plane, metres: the engine's (Q-87); the crate's is 0.1.
const EYE_NEAR_M: f32 = 0.01;

/// What `session` was configured with by `XrPlugin`.
#[derive(Resource, Clone, Copy, Debug)]
pub(crate) struct SessionConfig {
    /// What each eye's `SplatCamera` gets at session creation (Q-121, Q-399).
    pub(crate) sort: SortMode,
}

/// Set when the session is to be created again once destroyed: after the runtime
/// reported LOSS\_PENDING (E-XR-LOST), or for `display`'s render-scale change (Q-124).
#[derive(Resource, Default)]
pub(crate) struct Recreate(pub(crate) bool);

/// The session's systems; `plugin` adds it after the crate's plugins.
pub(crate) struct SessionPlugin {
    pub(crate) config: SessionConfig,
}

impl Plugin for SessionPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(comfort::ComfortPlugin)
            .insert_resource(self.config)
            .init_resource::<Recreate>()
            .add_plugins(ExtractResourcePlugin::<XrPrimaryReferenceSpace>::default())
            .add_systems(XrSessionCreated, (set_primary_reference_space, bind_rig))
            .add_systems(XrPreDestroySession, cleanup)
            .add_systems(PreUpdate, set_eye_near)
            .add_systems(Last, (make_splat_eyes, follow_sort).chain())
            .add_systems(Update, (watch_states, recreate_after_destroy).chain())
            .add_oxr_event_handler(replace_on_reference_space_change);
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(XrPreDestroySession, cleanup);
        }
    }
}

/// D-24's chain: the first of the settings' spaces (`stage`, `local-floor`, `local`) the
/// runtime enumerates, LOCAL\_FLOOR only with XR\_EXT\_local\_floor enabled; LOCAL when
/// none is, even when the list leaves `local` out (Q-49).
pub(crate) fn chain(
    settings: &[String],
    enumerated: &[ReferenceSpaceType],
    local_floor_enabled: bool,
) -> ReferenceSpaceType {
    for name in settings {
        let ty = match name.as_str() {
            "stage" => ReferenceSpaceType::STAGE,
            "local-floor" if local_floor_enabled => ReferenceSpaceType::LOCAL_FLOOR_EXT,
            "local" => ReferenceSpaceType::LOCAL,
            _ => continue,
        };
        if enumerated.contains(&ty) {
            return ty;
        }
    }
    ReferenceSpaceType::LOCAL
}

/// Creates the primary reference space at session creation (D-24), replacing the crate's
/// STAGE-only plugin; on LOCAL raises the tracking root 1.5 m and warns.
fn set_primary_reference_space(
    session: Option<Res<OxrSession>>,
    exts: Option<Res<OxrEnabledExtensions>>,
    settings: Option<Res<SettingsRes>>,
    mut root: Query<&mut Transform, With<XrTrackingRoot>>,
    mut commands: Commands,
) {
    // The crate runs this schedule even when the session failed: E-XR-RUNTIME (§1.15);
    // the Quest's entry point exits 2 on the missing reference space.
    let Some(session) = session else {
        log_runtime("the session could not be created");
        return;
    };
    let enumerated = match session.enumerate_reference_spaces() {
        Ok(spaces) => spaces,
        Err(e) => {
            log_error(&XrError::Runtime(format!("reference spaces: {e}")));
            Vec::new()
        }
    };
    info!("OpenXR reference spaces: {enumerated:?}");
    let local_floor = exts.is_some_and(|e| e.ext_local_floor);
    // Q-127
    let default = Settings::default();
    let wanted = &settings
        .as_deref()
        .map_or(&default, |s| &s.0)
        .xr
        .reference_space;
    let ty = chain(wanted, &enumerated, local_floor);
    let lift = if ty == ReferenceSpaceType::LOCAL {
        warn!(
            "reference space LOCAL: the floor is assumed {LOCAL_FLOOR_DROP_M} m below the \
             session's origin (D-24)"
        );
        LOCAL_FLOOR_DROP_M
    } else {
        info!("reference space {ty:?} (D-24)");
        0.0
    };
    for mut t in &mut root {
        t.translation.y = lift;
    }
    match session.create_reference_space(ty, Isometry3d::IDENTITY) {
        Ok(space) => {
            commands.insert_resource(XrPrimaryReferenceSpace(space));
            commands.insert_resource(PrimarySpaceType(ty));
        }
        Err(e) => log_error(&XrError::Runtime(format!("reference space {ty:?}: {e}"))),
    }
}

/// The rig binding (§1.8 Locomotion, E12, §2.1 `PlayerRig`): at each session creation the
/// tracking root becomes a child of `PlayerRig`, so moving the rig moves the eyes, whose
/// views the crate composes with the root's global transform, and the hands. The crate
/// spawns the root once and keeps it across sessions, so the binding holds through a
/// re-creation (spike check 4, Q-124); it is checked again at each session. Without a
/// rig (an app without `AppPlugin`) the root stays where it is.
fn bind_rig(
    roots: Query<(Entity, Option<&ChildOf>), With<XrTrackingRoot>>,
    rigs: Query<Entity, With<PlayerRig>>,
    mut commands: Commands,
) {
    let Some(rig) = rigs.iter().next() else {
        info!("no PlayerRig: the tracking root stays unparented");
        return;
    };
    for (root, parent) in &roots {
        if parent.is_none_or(|p| p.parent() != rig) {
            info!("the tracking root is bound to the PlayerRig");
            commands.entity(root).insert(ChildOf(rig));
        }
    }
}

/// Removes the primary reference space before the session is destroyed, in both worlds.
fn cleanup(query: Query<Entity, With<XrReferenceSpace>>, mut commands: Commands) {
    commands.remove_resource::<XrPrimaryReferenceSpace>();
    commands.remove_resource::<PrimarySpaceType>();
    for e in &query {
        commands.entity(e).remove::<XrReferenceSpace>();
    }
}

/// Readies each new eye camera in place: its near plane 0.01 m (Q-87), kept inside the
/// crate's `XrProjection` so the located matrix is never reset to the identity; its
/// tonemapping off (D-6, which `SplatCamera` sets too, once the eye is ready); and its
/// camera order its eye index, so the adapter's first splat camera is the left eye
/// (`XrCamera(0)`), both eyes having order 0 otherwise (Q-125). The crate spawns the eyes
/// at every session and recomputes their matrix from `near` every frame.
fn set_eye_near(
    mut eyes: Query<(Entity, &XrCamera, &mut Camera, &mut Projection), Added<XrCamera>>,
    mut commands: Commands,
) {
    for (eye, index, mut camera, mut projection) in &mut eyes {
        if let Projection::Custom(custom) = projection.as_mut()
            && let Some(p) = custom.get_mut::<XrProjection>()
        {
            p.near = EYE_NEAR_M;
        } else {
            *projection = Projection::custom(XrProjection {
                near: EYE_NEAR_M,
                ..XrProjection::default()
            });
        }
        // Q-125
        camera.order = isize::try_from(index.0).unwrap_or(isize::MAX);
        commands.entity(eye).insert(Tonemapping::None);
    }
}

/// The eyes not yet `SplatCamera`s.
type UnmarkedEyes = (With<XrCamera>, Without<SplatCamera>);

/// Whether a clip mapping is a located eye's: perspective (w_clip = −z_view) at the
/// engine's near plane, as the adapter requires (Q-87, Q-100).
fn is_located(clip_from_view: Mat4) -> bool {
    clip_from_view != Mat4::IDENTITY
        && clip_from_view.z_axis.w == -1.0
        && clip_from_view.w_axis.w == 0.0
        && (clip_from_view.w_axis.z - EYE_NEAR_M).abs() <= EYE_NEAR_M * 1e-3
}

/// Makes each eye a `SplatCamera` over the VR-mode background (Q-49) once the camera's
/// computed clip mapping, which the render world extracts, is the located views' at the
/// near plane 0.01 m: the crate spawns the eyes with the identity, and Bevy computes the
/// camera's matrix from the projection in its own PostUpdate system, so the projection
/// alone can be a frame ahead of what is extracted (#224's "not a perspective
/// projection").
fn make_splat_eyes(
    eyes: Query<(Entity, &Camera), UnmarkedEyes>,
    config: Res<SessionConfig>,
    settings: Option<Res<SettingsRes>>,
    mut commands: Commands,
) {
    // Q-127
    let background = settings.as_deref().map_or_else(
        || Settings::default().environment.background,
        |s| s.0.environment.background,
    );
    for (eye, camera) in &eyes {
        if is_located(camera.clip_from_view()) {
            commands.entity(eye).insert(SplatCamera {
                sort: config.sort,
                output: OutputSpace::LinearOver {
                    background: Some(background),
                },
            });
        }
    }
}

/// Keeps each eye's `SplatCamera.sort` equal to `SplatTierState.sort_mode` once it is a
/// splat camera, so the governor's fourth knob (E16) and a live tier switch (Q-239) reach
/// the eyes; `XrPlugin.sort` is only their mode at session creation. A camera's sort is
/// written only when it differs, since a new sort mode makes new frame buffers.
// Q-399
fn follow_sort(
    tier: Option<Res<SplatTierState>>,
    mut eyes: Query<&mut SplatCamera, With<XrCamera>>,
) {
    let Some(tier) = tier else {
        return;
    };
    for mut camera in &mut eyes {
        if camera.sort != tier.sort_mode {
            camera.sort = tier.sort_mode;
        }
    }
}

/// Logs the session's states (§1.15): LOSS\_PENDING is E-XR-LOST, recoverable, so a
/// warning (§1.13).
fn watch_states(mut changed: MessageReader<XrStateChanged>, mut recreate: ResMut<Recreate>) {
    for XrStateChanged(state) in changed.read() {
        info!("XR session state {state:?}");
        if let XrState::Exiting {
            should_restart: true,
        } = state
        {
            log_warning(&XrError::SessionLost);
            recreate.0 = true;
        }
    }
}

/// E-XR-LOST's recovery and `display`'s re-creation: the crate destroys a session but
/// never creates it again, so this asks for a new one once the old one is gone.
fn recreate_after_destroy(
    mut destroyed: MessageReader<XrSessionDestroyedMessage>,
    mut recreate: ResMut<Recreate>,
    mut create: MessageWriter<XrCreateSessionMessage>,
) {
    if destroyed.read().count() > 0 && recreate.0 {
        recreate.0 = false;
        info!("XR session destroyed: creating it again");
        create.write(XrCreateSessionMessage);
    }
}

/// The reference space `set_primary_reference_space` created, for the change events.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PrimarySpaceType(pub(crate) ReferenceSpaceType);

/// A reference-space change of the primary space (the user recentered, §1.8 Session and
/// spaces) re-places the content group with the event's pose when it is valid
/// (`replaced_group`); the rig is untouched. A change of another space type is only
/// logged; the change is applied when the event arrives, not at its `changeTime`.
// Q-264, Q-298
fn replace_on_reference_space_change(
    event: OxrEventIn,
    primary: Option<Res<PrimarySpaceType>>,
    root: Query<&GlobalTransform, With<XrTrackingRoot>>,
    mut groups: Query<&mut Transform, (With<ContentGroup>, Without<XrTrackingRoot>)>,
) {
    let openxr::Event::ReferenceSpaceChangePending(e) = *event else {
        return;
    };
    let ty = e.reference_space_type();
    if primary.is_none_or(|p| p.0 != ty) {
        info!("reference space {ty:?} changed: not the primary space");
        return;
    }
    let change = e.pose_valid().then(|| {
        let p = e.pose_in_previous_space();
        Pose {
            translation: Vec3::new(p.position.x, p.position.y, p.position.z),
            rotation: Quat::from_xyzw(
                p.orientation.x,
                p.orientation.y,
                p.orientation.z,
                p.orientation.w,
            ),
            scale: 1.0,
        }
    });
    info!(
        "reference space {ty:?} changed (pose valid {}): re-placing the content group",
        change.is_some()
    );
    let root = root.iter().next().map_or(Pose::IDENTITY, |g| {
        let (_, r, t) = g.to_scale_rotation_translation();
        Pose {
            translation: t,
            rotation: r,
            scale: 1.0,
        }
    });
    for mut t in &mut groups {
        let group = Pose {
            translation: t.translation,
            rotation: t.rotation,
            scale: t.scale.x,
        };
        let new = replaced_group(group, root, change);
        if new != group {
            t.translation = new.translation;
            t.rotation = new.rotation;
        }
    }
}

/// The twist about stage +y of `q` (swing-twist decomposition): the yaw-only rotation
/// Q-254 projects onto; the identity for a 180° swing, which has no twist. `anchors`
/// (M-6) projects a located anchor the same way.
// Q-254
pub(crate) fn twist_about_y(q: Quat) -> Quat {
    let twist = Quat::from_xyzw(0.0, q.y, 0.0, q.w);
    if twist.length() < 1e-6 {
        Quat::IDENTITY
    } else {
        twist.normalize()
    }
}

/// The content group's new pose after a reference-space change (§1.8 Session and
/// spaces): with `change` the event's valid pose T of the new space's origin in the
/// previous space (source-log row 285), projected on its yaw (Q-254), and `root` the tracking root's
/// world pose M (the rig and the LOCAL lift), the group becomes M T⁻¹ M⁻¹ composed with
/// its old pose, so every instance keeps its place in the room; Q-264's T⁻¹ ∘ G when the
/// root is the identity. `None` (an invalid pose) changes nothing.
// Q-264, Q-298
pub(crate) fn replaced_group(group: Pose, root: Pose, change: Option<Pose>) -> Pose {
    let Some(t) = change else {
        return group;
    };
    let t = Pose {
        translation: t.translation,
        rotation: twist_about_y(t.rotation),
        scale: 1.0,
    };
    let root = Pose { scale: 1.0, ..root };
    let moved = group.then(&root.inverse()).then(&t.inverse()).then(&root);
    Pose {
        rotation: moved.rotation.normalize(),
        scale: 1.0,
        ..moved
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_owned()).collect()
    }

    /// A rotation of `deg` degrees about stage +y.
    fn yaw(deg: f32) -> ftgsxr_core::Quat {
        ftgsxr_core::Quat::from_rotation_y(deg.to_radians())
    }

    fn pose(t: [f32; 3], q: ftgsxr_core::Quat) -> Pose {
        Pose {
            translation: ftgsxr_core::Vec3::from_array(t),
            rotation: q,
            scale: 1.0,
        }
    }

    fn close(a: ftgsxr_core::Vec3, b: ftgsxr_core::Vec3) -> bool {
        (a - b).length() <= 1e-5
    }

    // §1.8 Session and spaces, Q-264, Q-298: with a valid pose T of the new space's
    // origin in the previous space (source-log row 285), a point of an instance has, in the
    // previous space, T applied to its coordinates in the new space: the instance keeps
    // its place in the room. The tracking root (the rig and the LOCAL lift) maps the
    // spaces into the engine's world.
    #[test]
    fn replaced_group_keeps_every_instance_at_its_room_position() {
        let group = pose([0.5, 0.0, -1.0], yaw(20.0));
        let instance = Pose {
            translation: ftgsxr_core::Vec3::new(0.2, 0.8, -0.3),
            rotation: ftgsxr_core::Quat::from_rotation_x(0.4) * yaw(-35.0),
            scale: 0.7,
        };
        let root = pose([0.0, 1.5, 0.0], ftgsxr_core::Quat::IDENTITY)
            .then(&pose([1.0, 0.0, 2.0], yaw(30.0)));
        let change = pose([0.3, 0.05, -0.7], yaw(40.0));
        let new_group = replaced_group(group, root, Some(change));
        for x in [
            ftgsxr_core::Vec3::ZERO,
            ftgsxr_core::Vec3::new(1.0, -2.0, 0.5),
            ftgsxr_core::Vec3::new(-0.3, 0.4, 3.0),
        ] {
            let before = root
                .inverse()
                .transform_point(group.transform_point(instance.transform_point(x)));
            let after = root
                .inverse()
                .transform_point(new_group.transform_point(instance.transform_point(x)));
            assert!(
                close(change.transform_point(after), before),
                "{x}: {after} in the new space, {before} in the previous one"
            );
        }
        assert!((new_group.scale - 1.0).abs() <= 1e-6);
    }

    // Q-254, Q-264: the change's rotation is projected on its twist about +y, so the
    // group stays yaw-only; a tilt changes nothing.
    #[test]
    fn replaced_group_takes_only_the_changes_yaw() {
        let group = pose([0.5, 0.0, -1.0], yaw(20.0));
        let root = pose([1.0, 0.0, 2.0], yaw(30.0));
        let level = pose([0.3, 0.05, -0.7], yaw(40.0));
        let tilted = pose(
            [0.3, 0.05, -0.7],
            yaw(40.0) * ftgsxr_core::Quat::from_rotation_x(3f32.to_radians()),
        );
        let a = replaced_group(group, root, Some(level));
        let b = replaced_group(group, root, Some(tilted));
        assert!(close(a.translation, b.translation));
        for axis in [ftgsxr_core::Vec3::X, ftgsxr_core::Vec3::Z] {
            assert!(close(a.rotation * axis, b.rotation * axis));
        }
        assert!(close(
            b.rotation * ftgsxr_core::Vec3::Y,
            ftgsxr_core::Vec3::Y
        ));
    }

    // §1.8 Session and spaces, Q-264: an invalid pose changes nothing.
    #[test]
    fn replaced_group_without_a_valid_pose_is_unchanged() {
        let group = pose([0.5, 0.0, -1.0], yaw(20.0));
        let root = pose([1.0, 0.0, 2.0], yaw(30.0));
        assert_eq!(replaced_group(group, root, None), group);
    }

    #[test]
    fn chain_prefers_stage_then_local_floor_then_local() {
        use ReferenceSpaceType as T;
        let all = names(&["stage", "local-floor", "local"]);
        let every = [T::VIEW, T::LOCAL, T::STAGE, T::LOCAL_FLOOR_EXT];
        assert_eq!(chain(&all, &every, true), T::STAGE);
        assert_eq!(
            chain(&all, &[T::LOCAL, T::LOCAL_FLOOR_EXT], true),
            T::LOCAL_FLOOR_EXT
        );
        assert_eq!(
            chain(&all, &[T::LOCAL, T::LOCAL_FLOOR_EXT], false),
            T::LOCAL
        );
        let no_stage = names(&["local-floor", "local"]);
        assert_eq!(chain(&no_stage, &every, true), T::LOCAL_FLOOR_EXT);
        assert_eq!(chain(&no_stage, &every, false), T::LOCAL);
        // The chain ends at LOCAL even when the list leaves it out (Q-49).
        assert_eq!(chain(&names(&["stage"]), &[T::LOCAL], false), T::LOCAL);
    }

    #[test]
    fn is_located_only_for_a_located_eye_at_the_engines_near_plane() {
        use bevy_mod_xr::camera::{Fov, calculate_projection};
        let fov = Fov {
            angle_left: -0.9,
            angle_right: 0.8,
            angle_down: -0.85,
            angle_up: 0.75,
        };
        assert!(!is_located(Mat4::IDENTITY), "the crate's spawn matrix");
        assert!(
            !is_located(calculate_projection(0.1, fov)),
            "the crate's near"
        );
        assert!(is_located(calculate_projection(EYE_NEAR_M, fov)));
    }

    /// The sort mode of the eye cameras of `app`, in `XrCamera` order.
    fn eye_sorts(app: &mut App) -> Vec<SortMode> {
        let mut eyes: Vec<(u32, SortMode)> = app
            .world_mut()
            .query::<(&XrCamera, &SplatCamera)>()
            .iter(app.world())
            .map(|(i, c)| (i.0, c.sort))
            .collect();
        eyes.sort_by_key(|(i, _)| *i);
        eyes.into_iter().map(|(_, s)| s).collect()
    }

    /// Q-399 (#727): after session creation a change of `SplatTierState.sort_mode` (the
    /// governor's fourth knob, a live tier switch) reaches both eyes' `SplatCamera.sort`,
    /// both ways; a splat camera that is not an eye keeps its own.
    #[test]
    fn follow_sort_keeps_both_eyes_on_the_tiers_sort_mode() {
        use ftgsxr_render::{QualityTier, SplatTierState};
        let mut app = App::new();
        app.insert_resource(SplatTierState::of(&QualityTier::quest_balanced()))
            .add_systems(Last, follow_sort);
        let camera = |sort| SplatCamera {
            sort,
            output: OutputSpace::LinearOver { background: None },
        };
        for i in 0..2 {
            app.world_mut()
                .spawn((XrCamera(i), camera(SortMode::SharedCenter)));
        }
        let other = app.world_mut().spawn(camera(SortMode::SharedCenter)).id();
        app.update();
        assert_eq!(eye_sorts(&mut app), [SortMode::SharedCenter; 2]);
        for sort in [SortMode::PerView, SortMode::SharedCenter, SortMode::PerView] {
            app.world_mut().resource_mut::<SplatTierState>().sort_mode = sort;
            app.update();
            assert_eq!(eye_sorts(&mut app), [sort; 2], "both eyes take {sort:?}");
        }
        assert_eq!(
            app.world().get::<SplatCamera>(other).map(|c| c.sort),
            Some(SortMode::SharedCenter),
            "a splat camera that is not an eye keeps its sort"
        );
        // A live tier switch replaces the state with the new tier's own knobs (Q-239).
        app.insert_resource(SplatTierState::of(&QualityTier::quest_low()));
        app.update();
        assert_eq!(eye_sorts(&mut app), [SortMode::SharedCenter; 2]);
    }
}
