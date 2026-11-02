// SPDX-License-Identifier: Apache-2.0

//! The session, reference spaces and eye cameras (§1.8 Session and spaces, D-24, §1.15
//! E-XR-RUNTIME and E-XR-LOST, §2.4 Frame loop hooks). The rig binding is M-4's
//! (`PlayerRig`, `ContentGroup`).
//!
//! bevy\_mod\_openxr runs the session itself (its init, session and render plugins,
//! composed by `plugin`); this module replaces the crate's reference-space plugin with
//! D-24's chain, makes each eye camera the crate spawns a `SplatCamera`, and restarts the
//! session after a loss.

use bevy::app::{App, Last, Plugin, PreUpdate, Update};
use bevy::camera::{Camera, Projection};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::ecs::message::{MessageReader, MessageWriter};
use bevy::log::{error, info, warn};
use bevy::math::{Isometry3d, Mat4};
use bevy::prelude::{
    Added, Commands, Entity, IntoScheduleConfigs, Query, Res, ResMut, Resource, Transform, With,
    Without,
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
use ftgsxr_app::SettingsRes;
use ftgsxr_io::Settings;
use ftgsxr_render::{OutputSpace, SortMode, SplatCamera};
use openxr::ReferenceSpaceType;

use crate::XrError;

/// The height D-24's LOCAL fallback assumes the floor below the session's origin, metres.
const LOCAL_FLOOR_DROP_M: f32 = 1.5;

/// The eye projection's near plane, metres: the engine's (Q-87); the crate's is 0.1.
const EYE_NEAR_M: f32 = 0.01;

/// What `session` was configured with by `XrPlugin`.
#[derive(Resource, Clone, Copy, Debug)]
pub(crate) struct SessionConfig {
    /// What each eye's `SplatCamera` gets (Q-121).
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
        if self.config.sort == SortMode::SharedCenter {
            // Q-121
            warn!("XR sort mode SharedCenter: accepted, but draws nothing until M-8 (R-9)");
        }
        app.insert_resource(self.config)
            .init_resource::<Recreate>()
            .add_plugins(ExtractResourcePlugin::<XrPrimaryReferenceSpace>::default())
            .add_systems(XrSessionCreated, set_primary_reference_space)
            .add_systems(XrPreDestroySession, cleanup)
            .add_systems(PreUpdate, set_eye_near)
            .add_systems(Last, make_splat_eyes)
            .add_systems(Update, (watch_states, recreate_after_destroy).chain())
            .add_oxr_event_handler(log_reference_space_change);
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
    // The crate runs this schedule even when the session failed: E-XR-RUNTIME (§1.15).
    let Some(session) = session else {
        let e = XrError::Runtime("the session could not be created".into());
        error!("{}: {}", e.code(), e.user_text());
        return;
    };
    let enumerated = match session.enumerate_reference_spaces() {
        Ok(spaces) => spaces,
        Err(e) => {
            error!(
                "{}: reference spaces: {e}",
                XrError::Runtime(e.to_string()).code()
            );
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
        Ok(space) => commands.insert_resource(XrPrimaryReferenceSpace(space)),
        Err(e) => error!(
            "{}: reference space {ty:?}: {e}",
            XrError::Runtime(e.to_string()).code()
        ),
    }
}

/// Removes the primary reference space before the session is destroyed, in both worlds.
fn cleanup(query: Query<Entity, With<XrReferenceSpace>>, mut commands: Commands) {
    commands.remove_resource::<XrPrimaryReferenceSpace>();
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

/// Logs the session's states (§1.15): LOSS\_PENDING is E-XR-LOST.
fn watch_states(mut changed: MessageReader<XrStateChanged>, mut lost: ResMut<Recreate>) {
    for XrStateChanged(state) in changed.read() {
        info!("XR session state {state:?}");
        if let XrState::Exiting {
            should_restart: true,
        } = state
        {
            let e = XrError::SessionLost;
            error!("{}: {}", e.code(), e.user_text());
            lost.0 = true;
        }
    }
}

/// E-XR-LOST's recovery and `display`'s re-creation: the crate destroys a session but
/// never creates it again, so this asks for a new one once the old one is gone.
fn recreate_after_destroy(
    mut destroyed: MessageReader<XrSessionDestroyedMessage>,
    mut lost: ResMut<Recreate>,
    mut create: MessageWriter<XrCreateSessionMessage>,
) {
    if destroyed.read().count() > 0 && lost.0 {
        lost.0 = false;
        info!("XR session destroyed: creating it again");
        create.write(XrCreateSessionMessage);
    }
}

/// Logs a reference-space change (the user recentered); the content group's re-placement
/// is M-4's.
fn log_reference_space_change(event: OxrEventIn) {
    if let openxr::Event::ReferenceSpaceChangePending(e) = *event {
        info!(
            "reference space {:?} changed (re-placing the content group is M-4's)",
            e.reference_space_type()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| (*s).to_owned()).collect()
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
}
