// SPDX-License-Identifier: Apache-2.0

//! Spatial anchors (§1.9 Floor, shadows and anchors, D-26): the `AnchorStore` port, the
//! scene anchor's handshake on the XR side (Q-343) and the content group's placement on
//! a located anchor (Q-254).

use std::collections::HashMap;
use std::path::PathBuf;
use std::task::Poll;

use bevy::app::{App, Plugin, PostUpdate};
use bevy::ecs::message::MessageReader;
use bevy::log::{info, warn};
use bevy::prelude::{
    Commands, GlobalTransform, IntoScheduleConfigs, Query, Res, ResMut, Resource, Transform,
    TransformSystems, With, Without,
};
use bevy_mod_openxr::exts::OxrEnabledExtensions;
use bevy_mod_openxr::openxr_session_running;
use bevy_mod_openxr::poll_events::{OxrEventHandlerExt, OxrEventIn};
use bevy_mod_openxr::resources::{OxrFrameState, Pipelined};
use bevy_mod_openxr::session::OxrSession;
use bevy_mod_openxr::spaces::OxrSpaceExt;
use bevy_mod_xr::session::{XrPreDestroySession, XrSessionCreated, XrTrackingRoot};
use bevy_mod_xr::spaces::XrPrimaryReferenceSpace;
use ftgsxr_app::{
    AskPermissionAgain, ContentGroup, DeniedPermissions, FeatureState, HudAnchor, HudData,
    Permission, PersistCommand, PlatformRes, SceneAnchor, Toast, ToastLevel, XrFeatures,
};
use ftgsxr_core::Pose;
use openxr::sys;

use crate::error::{XrError, log_warning};
use crate::session::twist_about_y;
use crate::tracking::hands::{Flow, denied_text};

#[cfg(any(test, target_os = "android"))]
mod dv9;
mod fb;

/// A spatial anchor's id (§2.1): the runtime's UUID as text, lowercase and hyphenated
/// (8-4-4-4-12); while its creation is pending, the request id `request:<n>` (Q-343).
// Q-343, Q-T571.1
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AnchorId(pub String);

/// Spatial anchors (§1.9 Floor, shadows and anchors, D-26): the adapter over
/// XR\_FB\_spatial\_entity with XR\_META\_spatial\_entity\_persistence on the headset,
/// [`NoAnchors`] on the desktop and on a runtime without them. Poses are in the primary
/// reference space. The calls complete through the runtime's events (Q-343): `create`
/// returns the request's id at once, which resolves to the anchor's UUID when the
/// creation completes; `locate` is `None` until the anchor's space exists and is
/// located, and gives the runtime's pose unprojected (Q-254 projects where the content
/// group is placed); `persist` saves asynchronously; `forget` erases the saved anchor.
// Q-343
pub trait AnchorStore: Send + Sync {
    /// Creates an anchor at `pose` (§1.9: the content group's origin).
    fn create(&mut self, pose: Pose) -> Result<AnchorId, XrError>;
    /// The anchor's pose now, `None` while it is not located.
    fn locate(&self, id: &AnchorId) -> Option<Pose>;
    /// Persists the anchor, so a later session finds it by its UUID.
    fn persist(&mut self, id: &AnchorId) -> Result<(), XrError>;
    /// Erases the persisted anchor (§1.9: "Anchors are forgotten").
    fn forget(&mut self, id: &AnchorId) -> Result<(), XrError>;
}

/// The store without anchors (§2.1): the desktop's, and the headset's when the runtime
/// lacks the spatial-entity extensions (E-XR-UNSUPPORTED). Every call errs at once, so a
/// scene save never waits on it (Q-343).
// Q-343
#[derive(Clone, Copy, Debug, Default)]
pub struct NoAnchors;

/// The extension `NoAnchors` reports absent.
const SPATIAL_ENTITY: &str = "XR_FB_spatial_entity";

impl AnchorStore for NoAnchors {
    fn create(&mut self, _pose: Pose) -> Result<AnchorId, XrError> {
        Err(XrError::Unsupported(SPATIAL_ENTITY))
    }

    fn locate(&self, _id: &AnchorId) -> Option<Pose> {
        None
    }

    fn persist(&mut self, _id: &AnchorId) -> Result<(), XrError> {
        Err(XrError::Unsupported(SPATIAL_ENTITY))
    }

    fn forget(&mut self, _id: &AnchorId) -> Result<(), XrError> {
        Err(XrError::Unsupported(SPATIAL_ENTITY))
    }
}

/// What became of an anchor the store was asked about.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Status {
    /// The creation or the query has not completed.
    Pending,
    /// The anchor's space exists (created, or found by a query): its UUID.
    Created(AnchorId),
    /// Created and persisted: its UUID.
    Persisted(AnchorId),
    /// A query completed without finding it.
    NotFound,
    /// The creation, the persist or the query failed.
    Failed(String),
}

/// What the handshake needs beyond §2.1's signatures, which complete through events
/// (Q-343): the status of a creation, a persist or a query, and the query of a saved
/// anchor by its UUID. The adapter and `NoAnchors` implement it; so does the tests' fake.
pub(crate) trait AnchorDriver: AnchorStore {
    /// Whether the runtime offers anchors (false for `NoAnchors`).
    fn available(&self) -> bool;
    /// Whether `create` and `locate` can run now (the adapter: once it has a frame).
    fn ready(&self) -> bool {
        true
    }
    /// Starts loading the saved anchor `uuid`; its `status` then tells.
    fn query(&mut self, uuid: &AnchorId) -> Result<(), XrError>;
    /// What became of `id` (a request id or a UUID).
    fn status(&self, id: &AnchorId) -> Status;
    /// This frame's time and primary reference space, which `create` and `locate` use.
    fn frame(&mut self, _time: sys::Time, _base: sys::Space) {}
    /// An event of the runtime's queue: the completions.
    fn event(&mut self, _event: &openxr::Event<'_>) {}
    /// The result of `forget(uuid)`, once the runtime reported it (Dv-9's cleanup).
    #[cfg_attr(not(target_os = "android"), allow(dead_code))]
    fn erased(&self, _uuid: &str) -> Option<Result<(), String>> {
        None
    }
}

impl AnchorDriver for NoAnchors {
    fn available(&self) -> bool {
        false
    }

    fn query(&mut self, _uuid: &AnchorId) -> Result<(), XrError> {
        Err(XrError::Unsupported(SPATIAL_ENTITY))
    }

    fn status(&self, _id: &AnchorId) -> Status {
        Status::Failed(SPATIAL_ENTITY.to_owned())
    }
}

/// The session's anchor store (PRD #565: a crate-private main-world resource).
#[derive(Resource)]
pub(crate) struct Anchors(pub(crate) Box<dyn AnchorDriver>);

impl Default for Anchors {
    fn default() -> Self {
        Anchors(Box::new(NoAnchors))
    }
}

/// The handshake's job in progress.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
enum Job {
    /// Nothing asked.
    #[default]
    Idle,
    /// A save's anchor: the create's id, whether its persist was issued, and the UUID
    /// the saved file named before (Q-344).
    Creating {
        id: AnchorId,
        persisting: bool,
        old: Option<String>,
        path: Option<PathBuf>,
    },
    /// A load's anchor, queried by its UUID.
    Locating { uuid: String, path: Option<PathBuf> },
}

/// The XR side of the scene anchor's handshake (Q-343): the job, the permission flow
/// (Q-348, kept across sessions), what `Ready` came from (the HUD's `Persisted` or
/// `Located`, Q-345), the UUID each scene path last named (Q-344), the anchors whose
/// save gave up waiting (forgotten once created) and Q-161's toast once per session.
// Q-T571.3
#[derive(Resource, Default)]
pub(crate) struct Handshake {
    job: Job,
    pub(crate) flow: Flow,
    located: bool,
    named: HashMap<PathBuf, String>,
    save_path: Option<PathBuf>,
    load_path: Option<PathBuf>,
    abandoned: Vec<AnchorId>,
    toasted: bool,
    /// Whether the instance enabled the spatial-entity extensions: until the session's
    /// adapter exists and has a frame, a create or a locate waits (Q-T571.2).
    expected: bool,
}

/// The content group's world pose on an anchor located at `located` in the primary
/// reference space, under the tracking root's world pose `root` (Q-254): the twist about
/// stage +y of the anchor's world rotation, its world translation unchanged, scale 1.
// Q-254, Q-T571.1
pub(crate) fn placed_group(located: Pose, root: Pose) -> Pose {
    let root = Pose { scale: 1.0, ..root };
    let located = Pose {
        scale: 1.0,
        ..located
    };
    let world = located.then(&root);
    Pose {
        translation: world.translation,
        rotation: twist_about_y(world.rotation.normalize()),
        scale: 1.0,
    }
}

/// The pose, in the primary reference space, of the anchor created at the content group's
/// origin `group` (its world pose) under the tracking root's world pose `root`: scale
/// dropped on both, so the anchor carries the group's yaw and translation (§1.9).
// Q-T571.1
pub(crate) fn creation_pose(group: Pose, root: Pose) -> Pose {
    let root = Pose { scale: 1.0, ..root };
    let group = Pose {
        scale: 1.0,
        ..group
    };
    let p = group.then(&root.inverse());
    Pose {
        rotation: p.rotation.normalize(),
        ..p
    }
}

/// The anchors' plugin (§1.9, D-26): the store, `NoAnchors` until a session with the
/// spatial-entity extensions exists (E-XR-UNSUPPORTED logged once otherwise, Q-134),
/// the runtime's events into it, and the handshake each frame.
pub(crate) struct AnchorsPlugin;

/// The extensions the adapter needs (Appendix E.1, D-26).
pub(crate) fn missing_extensions(enabled: &openxr::ExtensionSet) -> Vec<&'static str> {
    [
        ("XR_FB_spatial_entity", enabled.fb_spatial_entity),
        (
            "XR_FB_spatial_entity_query",
            enabled.fb_spatial_entity_query,
        ),
        (
            "XR_META_spatial_entity_persistence",
            enabled.meta_spatial_entity_persistence,
        ),
    ]
    .into_iter()
    .filter_map(|(name, on)| (!on).then_some(name))
    .collect()
}

impl Plugin for AnchorsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Anchors>()
            .init_resource::<Handshake>()
            .init_resource::<SceneAnchor>()
            .init_resource::<HudData>()
            .init_resource::<XrFeatures>()
            .add_message::<AskPermissionAgain>()
            .add_message::<PersistCommand>()
            .add_systems(XrSessionCreated, on_session_created)
            .add_systems(XrPreDestroySession, on_session_destroyed)
            .add_oxr_event_handler(on_event)
            .add_systems(
                PostUpdate,
                (feed_frame.run_if(openxr_session_running), handshake)
                    .chain()
                    .before(TransformSystems::Propagate),
            );
        // E-XR-UNSUPPORTED once per absent extension, a warning with no toast (Q-134,
        // Q-161: the default is not an explicit request).
        if let Some(enabled) = app.world().get_resource::<OxrEnabledExtensions>() {
            let missing = missing_extensions(enabled.0.raw());
            for name in &missing {
                log_warning(&XrError::Unsupported(name));
            }
            app.world_mut().resource_mut::<Handshake>().expected = missing.is_empty();
        }
        // Dv-9 (#571): inert until the quest runner's entity names it (Q-308, Q-346).
        #[cfg(target_os = "android")]
        dv9::add(app);
    }
}

/// The session's store: the adapter where the instance enabled the three extensions,
/// else `NoAnchors`; Q-161's toast is once per session.
fn on_session_created(
    exts: Option<Res<OxrEnabledExtensions>>,
    session: Option<Res<OxrSession>>,
    mut hs: ResMut<Handshake>,
    mut commands: Commands,
) {
    hs.toasted = false;
    let full = exts
        .as_deref()
        .is_some_and(|e| missing_extensions(e.0.raw()).is_empty());
    match session {
        Some(s) if full => {
            info!("anchors: XR_FB_spatial_entity with XR_META_spatial_entity_persistence");
            let session: &openxr::Session<openxr::AnyGraphics> = &s;
            commands.insert_resource(Anchors(Box::new(fb::FbAnchors::new(session.clone()))));
        }
        _ => commands.insert_resource(Anchors::default()),
    }
}

/// The anchors' spaces go with their session; the handshake's job fails with it.
fn on_session_destroyed(mut commands: Commands) {
    commands.insert_resource(Anchors::default());
}

/// The runtime's events reach the store (the completions, Q-343).
fn on_event(event: OxrEventIn, store: Option<ResMut<Anchors>>) {
    if let Some(mut store) = store {
        store.0.event(&event);
    }
}

/// The frame's time and primary reference space for `create` and `locate`: the predicted
/// display time plus one period when rendering is pipelined, as `tracking` locates
/// (source-log 273).
fn feed_frame(
    mut store: ResMut<Anchors>,
    space: Option<Res<XrPrimaryReferenceSpace>>,
    frame: Option<Res<OxrFrameState>>,
    pipelined: Option<Res<Pipelined>>,
) {
    let (Some(space), Some(frame)) = (space, frame) else {
        return;
    };
    let time = if pipelined.is_some() {
        openxr::Time::from_nanos(
            frame.predicted_display_time.as_nanos() + frame.predicted_display_period.as_nanos(),
        )
    } else {
        frame.predicted_display_time
    };
    let base = space.0.0.as_raw_openxr_space();
    store.0.frame(time, base);
}

/// The world pose of a global transform, scale dropped.
fn pose_of(g: &GlobalTransform) -> Pose {
    let (_, rotation, translation) = g.to_scale_rotation_translation();
    Pose {
        translation,
        rotation,
        scale: 1.0,
    }
}

/// Asks for `Permission::Anchors` through the platform; without one (a test app without
/// `AppPlugin`) it counts as granted, as `LinuxPlatform` grants every one (Q-276).
// Q-348
fn ask(platform: Option<&PlatformRes>) -> Flow {
    match platform {
        Some(p) => Flow::Asking(p.0.request_permission(Permission::Anchors)),
        None => Flow::Granted,
    }
}

/// The handshake's frame, after `persist`'s commands (Q-343): the permission on the first
/// create or locate (Q-348), a save's anchor created at the content group's origin and
/// persisted, then `Ready(uuid)`, a failure `None` (Q-350); a load's anchor queried and
/// located, the group placed on it by Q-254 and `Ready`, or `Lost(uuid)`; the UUID a
/// re-saved scene named before erased once the new one is persisted (Q-344). Writes
/// `HudData.anchor` (Q-345) and `XrFeatures.anchors` (Q-336).
// Q-343, Q-344, Q-345, Q-348, Q-350, Q-T571.2, Q-T571.3
#[allow(clippy::too_many_arguments)]
pub(crate) fn handshake(
    mut store: ResMut<Anchors>,
    mut hs: ResMut<Handshake>,
    mut scene: ResMut<SceneAnchor>,
    mut hud: ResMut<HudData>,
    mut features: ResMut<XrFeatures>,
    platform: Option<Res<PlatformRes>>,
    mut again: MessageReader<AskPermissionAgain>,
    mut persist: MessageReader<PersistCommand>,
    mut denied: Option<ResMut<DeniedPermissions>>,
    root: Query<&GlobalTransform, With<XrTrackingRoot>>,
    mut groups: Query<&mut Transform, (With<ContentGroup>, Without<XrTrackingRoot>)>,
    mut commands: Commands,
) {
    let hs = &mut *hs;
    let store = &mut *store.0;
    // Q-344: the path of this frame's save or load, the file a UUID belongs to.
    for c in persist.read() {
        match c {
            PersistCommand::SaveScene(p) => hs.save_path = Some(p.clone()),
            PersistCommand::LoadScene(p) => hs.load_path = Some(p.clone()),
            _ => {}
        }
    }
    let root = root.iter().next().map_or(Pose::IDENTITY, pose_of);
    forget_abandoned(store, hs);

    // A job whose state moved on (persist gave up waiting, another load) is dropped; a
    // dropped creation is forgotten once it resolves.
    match (&hs.job, &*scene) {
        (Job::Creating { .. }, SceneAnchor::Creating) => {}
        (Job::Locating { uuid, .. }, SceneAnchor::Locating(u)) if u == uuid => {}
        (Job::Idle, _) => {}
        (Job::Creating { id, .. }, _) => {
            info!(
                "anchor {}: the save stopped waiting; forgotten once created",
                id.0
            );
            let id = id.clone();
            hs.abandoned.push(id);
            hs.job = Job::Idle;
        }
        (Job::Locating { .. }, _) => hs.job = Job::Idle,
    }

    // Q-348: the permission, asked on the first create or locate, again on request.
    let mut listed = |on: bool| {
        if let Some(d) = denied.as_deref_mut() {
            let has = d.0.contains(&Permission::Anchors);
            if on && !has {
                d.0.push(Permission::Anchors);
            } else if !on && has {
                d.0.retain(|p| *p != Permission::Anchors);
            }
        }
    };
    let asked_again = again.read().any(|a| a.0 == Permission::Anchors);
    if asked_again && hs.flow == Flow::Denied {
        hs.flow = ask(platform.as_deref());
        listed(false);
    }
    let available = store.available();
    // Q-T571.2: with the extensions enabled, a request waits for the session's adapter
    // and its first frame rather than failing on the store that stands in before it.
    let waiting = hs.expected && !(available && store.ready());
    let starting = !waiting
        && hs.job == Job::Idle
        && matches!(*scene, SceneAnchor::Creating | SceneAnchor::Locating(_));
    if starting && available && hs.flow == Flow::Idle {
        hs.flow = ask(platform.as_deref());
    }
    if let Flow::Asking(h) = hs.flow
        && let Some(platform) = platform.as_deref()
    {
        match platform.0.poll_permission(h) {
            Poll::Pending => {}
            Poll::Ready(true) => hs.flow = Flow::Granted,
            Poll::Ready(false) => {
                hs.flow = Flow::Denied;
                listed(true);
                // Q-261
                let text = denied_text(Permission::Anchors, "anchors");
                warn!("{text}");
                commands.trigger(Toast {
                    text,
                    level: ToastLevel::Warning,
                });
            }
        }
    }

    if starting && (hs.flow == Flow::Granted || !available) {
        start(store, hs, &mut scene, root, &groups, &mut commands);
    } else if starting && hs.flow == Flow::Denied {
        // The feature stays off until asked again (Q-348): no anchor.
        let next = match &*scene {
            SceneAnchor::Locating(u) => SceneAnchor::Lost(u.clone()),
            _ => SceneAnchor::None,
        };
        *scene = next;
    } else {
        progress(store, hs, &mut scene, root, &mut groups);
    }

    // Q-345: the HUD's anchor state.
    let shown = match &*scene {
        SceneAnchor::None => None,
        SceneAnchor::Creating => Some(HudAnchor::Creating),
        SceneAnchor::Ready(_) if hs.located => Some(HudAnchor::Located),
        SceneAnchor::Ready(_) => Some(HudAnchor::Persisted),
        SceneAnchor::Locating(_) => Some(HudAnchor::Locating),
        SceneAnchor::Lost(_) => Some(HudAnchor::Lost),
    };
    if hud.anchor != shown {
        hud.anchor = shown;
    }
    // Q-336: the feature's state.
    let state = if !available {
        FeatureState::Absent
    } else if hs.flow == Flow::Denied {
        FeatureState::Denied
    } else {
        FeatureState::Available
    };
    if features.anchors != state {
        features.anchors = state;
    }
}

/// Starts the job `scene` asks for.
fn start(
    store: &mut dyn AnchorDriver,
    hs: &mut Handshake,
    scene: &mut SceneAnchor,
    root: Pose,
    groups: &Query<&mut Transform, (With<ContentGroup>, Without<XrTrackingRoot>)>,
    commands: &mut Commands,
) {
    match scene.clone() {
        SceneAnchor::Creating => {
            let group = groups.iter().next().map_or(Pose::IDENTITY, |t| Pose {
                translation: t.translation,
                rotation: t.rotation,
                scale: 1.0,
            });
            match store.create(creation_pose(group, root)) {
                Ok(id) => {
                    let path = hs.save_path.take();
                    let old = path.as_ref().and_then(|p| hs.named.get(p).cloned());
                    hs.job = Job::Creating {
                        id,
                        persisting: false,
                        old,
                        path,
                    };
                }
                Err(e) => {
                    // Q-350: `None` ends persist's wait as a failure.
                    log_warning(&e);
                    *scene = SceneAnchor::None;
                }
            }
        }
        SceneAnchor::Locating(uuid) => match store.query(&AnchorId(uuid.clone())) {
            Ok(()) => {
                hs.job = Job::Locating {
                    uuid,
                    path: hs.load_path.take(),
                };
            }
            Err(e) => {
                log_warning(&e);
                // Q-161: a scene saved with an anchor, reopened on a runtime without the
                // extensions, is an explicit request: one toast per session.
                if matches!(e, XrError::Unsupported(_)) && !hs.toasted {
                    hs.toasted = true;
                    commands.trigger(Toast {
                        text: e.to_string(),
                        level: ToastLevel::Warning,
                    });
                }
                *scene = SceneAnchor::Lost(uuid);
            }
        },
        _ => {}
    }
}

/// Advances the job in progress by what the store reports.
fn progress(
    store: &mut dyn AnchorDriver,
    hs: &mut Handshake,
    scene: &mut SceneAnchor,
    root: Pose,
    groups: &mut Query<&mut Transform, (With<ContentGroup>, Without<XrTrackingRoot>)>,
) {
    match hs.job.clone() {
        Job::Idle => {}
        Job::Creating {
            id,
            persisting,
            old,
            path,
        } => match store.status(&id) {
            Status::Pending => {}
            Status::Created(_) if persisting => {}
            Status::Created(uuid) => match store.persist(&id) {
                Ok(()) => {
                    if let Job::Creating { persisting, .. } = &mut hs.job {
                        *persisting = true;
                    }
                }
                Err(e) => {
                    log_warning(&e);
                    hs.abandoned.push(uuid);
                    hs.job = Job::Idle;
                    *scene = SceneAnchor::None;
                }
            },
            Status::Persisted(uuid) => {
                info!("anchor {} created and persisted", uuid.0);
                // Q-344: the file named another anchor before; it is erased now.
                if let Some(old) = old.filter(|o| *o != uuid.0) {
                    match store.forget(&AnchorId(old.clone())) {
                        Ok(()) => info!("anchor {old}: forgotten, its scene saved again"),
                        Err(e) => warn!("anchor {old}: not forgotten: {e}"),
                    }
                }
                if let Some(p) = path {
                    hs.named.insert(p, uuid.0.clone());
                }
                hs.located = false;
                hs.job = Job::Idle;
                *scene = SceneAnchor::Ready(uuid.0);
            }
            Status::NotFound | Status::Failed(_) => {
                warn!("anchor {}: not created ({:?})", id.0, store.status(&id));
                hs.job = Job::Idle;
                *scene = SceneAnchor::None;
            }
        },
        Job::Locating { uuid, path } => {
            let id = AnchorId(uuid.clone());
            match store.status(&id) {
                Status::Pending => {}
                Status::Created(_) | Status::Persisted(_) => {
                    // Found: placed once the runtime locates it.
                    let Some(at) = store.locate(&id) else { return };
                    let placed = placed_group(at, root);
                    for mut t in groups.iter_mut() {
                        t.translation = placed.translation;
                        t.rotation = placed.rotation;
                    }
                    info!("anchor {uuid} located: the content group placed on it");
                    if let Some(p) = path {
                        hs.named.insert(p, uuid.clone());
                    }
                    hs.located = true;
                    hs.job = Job::Idle;
                    *scene = SceneAnchor::Ready(uuid);
                }
                Status::NotFound | Status::Failed(_) => {
                    warn!("anchor {uuid}: not found ({:?})", store.status(&id));
                    hs.job = Job::Idle;
                    *scene = SceneAnchor::Lost(uuid);
                }
            }
        }
    }
}

/// Forgets the anchors whose save stopped waiting, once their creation resolved.
// Q-T571.2
fn forget_abandoned(store: &mut dyn AnchorDriver, hs: &mut Handshake) {
    hs.abandoned.retain(|id| match store.status(id) {
        Status::Pending => true,
        Status::Created(uuid) | Status::Persisted(uuid) => {
            if let Err(e) = store.forget(&uuid) {
                info!("anchor {}: not forgotten: {e}", uuid.0);
            }
            false
        }
        Status::NotFound | Status::Failed(_) => false,
    });
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use bevy::app::{App, PostUpdate};
    use bevy::transform::TransformPlugin;
    use ftgsxr_core::{Quat, Vec3};

    use super::*;

    fn pose(t: [f32; 3], q: Quat) -> Pose {
        Pose {
            translation: Vec3::from_array(t),
            rotation: q,
            scale: 1.0,
        }
    }

    fn yaw(deg: f32) -> Quat {
        Quat::from_rotation_y(deg.to_radians())
    }

    /// The yaw of a rotation that keeps +y, in radians.
    fn yaw_of(q: Quat) -> f32 {
        2.0 * q.y.atan2(q.w)
    }

    fn close(a: Vec3, b: Vec3, tol: f32) -> bool {
        (a - b).length() <= tol
    }

    /// A rig moved and turned, so the reference space and the world differ.
    fn root() -> Pose {
        pose([1.0, 0.0, 2.0], yaw(10.0))
    }

    /// A store that completes at once: a create resolves to `uuid-<n>`, a persist is
    /// done when issued, a query finds the anchors in `saved`; `located` gives each
    /// UUID's pose.
    #[derive(Default)]
    struct Fake {
        next: u32,
        created: HashMap<AnchorId, (AnchorId, Pose)>,
        persisted: Vec<AnchorId>,
        queried: Vec<AnchorId>,
        saved: HashMap<AnchorId, Pose>,
        forgotten: Arc<Mutex<Vec<AnchorId>>>,
        /// While set, every creation is pending.
        hold: Arc<AtomicBool>,
    }

    impl Fake {
        fn uuid_of(&self, id: &AnchorId) -> Option<AnchorId> {
            self.created.get(id).map(|(u, _)| u.clone()).or_else(|| {
                let known =
                    self.queried.contains(id) || self.created.values().any(|(u, _)| u == id);
                known.then(|| id.clone())
            })
        }
    }

    impl AnchorStore for Fake {
        fn create(&mut self, pose: Pose) -> Result<AnchorId, XrError> {
            self.next += 1;
            let id = AnchorId(format!("request:{}", self.next));
            let uuid = AnchorId(format!("uuid-{}", self.next));
            self.created.insert(id.clone(), (uuid, pose));
            Ok(id)
        }

        fn locate(&self, id: &AnchorId) -> Option<Pose> {
            let uuid = self.uuid_of(id)?;
            self.saved.get(&uuid).copied().or_else(|| {
                self.created
                    .values()
                    .find(|(u, _)| *u == uuid)
                    .map(|(_, p)| *p)
            })
        }

        fn persist(&mut self, id: &AnchorId) -> Result<(), XrError> {
            let uuid = self.uuid_of(id).ok_or(XrError::Runtime("unknown".into()))?;
            self.persisted.push(uuid);
            Ok(())
        }

        fn forget(&mut self, id: &AnchorId) -> Result<(), XrError> {
            self.forgotten.lock().expect("unpoisoned").push(id.clone());
            Ok(())
        }
    }

    impl AnchorDriver for Fake {
        fn available(&self) -> bool {
            true
        }

        fn query(&mut self, uuid: &AnchorId) -> Result<(), XrError> {
            self.queried.push(uuid.clone());
            Ok(())
        }

        fn status(&self, id: &AnchorId) -> Status {
            if let Some((uuid, _)) = self.created.get(id) {
                if self.hold.load(Ordering::SeqCst) {
                    return Status::Pending;
                }
                return if self.persisted.contains(uuid) {
                    Status::Persisted(uuid.clone())
                } else {
                    Status::Created(uuid.clone())
                };
            }
            if self.queried.contains(id) {
                return if self.saved.contains_key(id) {
                    Status::Created(id.clone())
                } else {
                    Status::NotFound
                };
            }
            Status::Failed("unknown".into())
        }
    }

    /// A headless app with the handshake, `store`, a content group at `group` and a
    /// tracking root at `root()`.
    fn app(store: Fake, group: Pose) -> App {
        let mut app = App::new();
        app.add_plugins(TransformPlugin)
            .add_message::<AskPermissionAgain>()
            .add_message::<PersistCommand>()
            .init_resource::<SceneAnchor>()
            .init_resource::<HudData>()
            .init_resource::<XrFeatures>()
            .init_resource::<Handshake>()
            .insert_resource(Anchors(Box::new(store)))
            .add_systems(PostUpdate, handshake);
        let r = root();
        app.world_mut().spawn((
            XrTrackingRoot,
            Transform::from_translation(r.translation).with_rotation(r.rotation),
        ));
        app.world_mut().spawn((
            ContentGroup,
            Transform::from_translation(group.translation).with_rotation(group.rotation),
        ));
        app
    }

    fn group_pose(app: &mut App) -> Pose {
        let t = *app
            .world_mut()
            .query_filtered::<&Transform, With<ContentGroup>>()
            .single(app.world())
            .expect("one group");
        pose(t.translation.to_array(), t.rotation)
    }

    /// Q-343: `Creating` creates an anchor at the content group's origin, persists it and
    /// becomes `Ready` with its UUID; the HUD shows it persisted.
    #[test]
    fn handshake_creating_becomes_ready_with_the_uuid() {
        let group = pose([0.4, 0.0, -1.2], yaw(25.0));
        let mut app = app(Fake::default(), group);
        app.update();
        app.insert_resource(SceneAnchor::Creating);
        for _ in 0..4 {
            app.update();
        }
        assert_eq!(
            *app.world().resource::<SceneAnchor>(),
            SceneAnchor::Ready("uuid-1".into())
        );
        assert_eq!(
            app.world().resource::<HudData>().anchor,
            Some(HudAnchor::Persisted)
        );
        assert_eq!(
            app.world().resource::<XrFeatures>().anchors,
            FeatureState::Available
        );
        // The anchor was created at the group's origin, in the reference space.
        let at = app
            .world()
            .resource::<Anchors>()
            .0
            .locate(&AnchorId("uuid-1".into()))
            .expect("located");
        let world = at.then(&root());
        assert!(close(world.translation, group.translation, 1e-5));
        assert!((yaw_of(world.rotation) - 25f32.to_radians()).abs() <= 1e-5);
    }

    /// Q-343, Q-254: `Locating(uuid)` queries and locates the anchor, places the content
    /// group on it (a 2° tilt dropped, its yaw and translation kept) and becomes `Ready`;
    /// the HUD shows it located.
    #[test]
    fn handshake_locating_becomes_ready_with_the_group_placed() {
        let saved = pose([0.4, 0.0, -1.2], yaw(25.0));
        let created = creation_pose(saved, root());
        let mut store = Fake::default();
        store.saved.insert(
            AnchorId("a1b2".into()),
            Pose {
                rotation: created.rotation * Quat::from_rotation_x(2f32.to_radians()),
                ..created
            },
        );
        // The file's pose left the group elsewhere.
        let mut app = app(store, pose([3.0, 0.0, 1.0], yaw(-70.0)));
        app.update();
        app.insert_resource(SceneAnchor::Locating("a1b2".into()));
        for _ in 0..3 {
            app.update();
        }
        assert_eq!(
            *app.world().resource::<SceneAnchor>(),
            SceneAnchor::Ready("a1b2".into())
        );
        assert_eq!(
            app.world().resource::<HudData>().anchor,
            Some(HudAnchor::Located)
        );
        let group = group_pose(&mut app);
        assert!(
            close(group.translation, saved.translation, 1e-5),
            "{group:?}"
        );
        assert!(close(group.rotation * Vec3::Y, Vec3::Y, 1e-6));
        assert!((yaw_of(group.rotation) - 25f32.to_radians()).abs() <= 1e-5);
    }

    /// Q-343: an anchor the runtime does not find becomes `Lost(uuid)`, the group left as
    /// the file placed it.
    #[test]
    fn handshake_missing_anchor_becomes_lost() {
        let file = pose([3.0, 0.0, 1.0], yaw(-70.0));
        let mut app = app(Fake::default(), file);
        app.update();
        app.insert_resource(SceneAnchor::Locating("gone".into()));
        for _ in 0..3 {
            app.update();
        }
        assert_eq!(
            *app.world().resource::<SceneAnchor>(),
            SceneAnchor::Lost("gone".into())
        );
        assert_eq!(
            app.world().resource::<HudData>().anchor,
            Some(HudAnchor::Lost)
        );
        let group = group_pose(&mut app);
        assert!(close(group.translation, file.translation, 1e-6));
    }

    /// Q-344: a scene saved again under the same path with a new anchor erases the one
    /// it named before, once the new one is persisted; a save under another path does
    /// not.
    #[test]
    fn handshake_resaved_scene_forgets_its_old_anchor() {
        let store = Fake::default();
        let forgotten = store.forgotten.clone();
        let mut app = app(store, pose([0.0, 0.0, -1.0], yaw(0.0)));
        let save = |app: &mut App, path: &str| {
            app.world_mut()
                .write_message(PersistCommand::SaveScene(path.into()));
            app.insert_resource(SceneAnchor::Creating);
            for _ in 0..3 {
                app.update();
            }
        };
        save(&mut app, "demo");
        assert_eq!(
            *app.world().resource::<SceneAnchor>(),
            SceneAnchor::Ready("uuid-1".into())
        );
        save(&mut app, "other");
        assert!(forgotten.lock().expect("unpoisoned").is_empty());
        save(&mut app, "demo");
        assert_eq!(
            *app.world().resource::<SceneAnchor>(),
            SceneAnchor::Ready("uuid-3".into())
        );
        assert_eq!(
            *forgotten.lock().expect("unpoisoned"),
            vec![AnchorId("uuid-1".into())]
        );
    }

    /// Q-343, Q-350: with `NoAnchors` a save's `Creating` becomes `None` at once (no
    /// wait), a load's `Locating` becomes `Lost`, and the feature is absent.
    #[test]
    fn handshake_without_anchors_fails_at_once() {
        let mut app = app(Fake::default(), Pose::IDENTITY);
        app.insert_resource(Anchors::default());
        app.insert_resource(SceneAnchor::Creating);
        app.update();
        assert_eq!(*app.world().resource::<SceneAnchor>(), SceneAnchor::None);
        app.insert_resource(SceneAnchor::Locating("x".into()));
        app.update();
        assert_eq!(
            *app.world().resource::<SceneAnchor>(),
            SceneAnchor::Lost("x".into())
        );
        assert_eq!(
            app.world().resource::<XrFeatures>().anchors,
            FeatureState::Absent
        );
    }

    /// Q-350: a creation still pending when persist stops waiting (the state set to
    /// `None`) is dropped: the state stays `None`, and the anchor is forgotten once it
    /// resolves.
    #[test]
    fn handshake_late_creation_is_dropped_and_forgotten() {
        let store = Fake::default();
        let (forgotten, hold) = (store.forgotten.clone(), store.hold.clone());
        hold.store(true, Ordering::SeqCst);
        let mut app = app(store, Pose::IDENTITY);
        app.insert_resource(SceneAnchor::Creating);
        app.update();
        app.update();
        assert_eq!(
            *app.world().resource::<SceneAnchor>(),
            SceneAnchor::Creating
        );
        app.insert_resource(SceneAnchor::None);
        app.update();
        hold.store(false, Ordering::SeqCst);
        app.update();
        app.update();
        assert_eq!(*app.world().resource::<SceneAnchor>(), SceneAnchor::None);
        assert_eq!(
            *forgotten.lock().expect("unpoisoned"),
            vec![AnchorId("uuid-1".into())]
        );
    }

    /// Q-T571.2: with the extensions enabled on the instance, a `Locating` set before the
    /// session's adapter exists waits (no `Lost`, no toast), and proceeds once it does.
    #[test]
    fn handshake_waits_for_the_sessions_store() {
        let saved = pose([0.4, 0.0, -1.2], yaw(25.0));
        let mut store = Fake::default();
        store
            .saved
            .insert(AnchorId("a1b2".into()), creation_pose(saved, root()));
        let mut app = app(Fake::default(), Pose::IDENTITY);
        app.insert_resource(Anchors::default());
        app.world_mut().resource_mut::<Handshake>().expected = true;
        app.insert_resource(SceneAnchor::Locating("a1b2".into()));
        for _ in 0..3 {
            app.update();
        }
        assert_eq!(
            *app.world().resource::<SceneAnchor>(),
            SceneAnchor::Locating("a1b2".into())
        );
        app.insert_resource(Anchors(Box::new(store)));
        for _ in 0..3 {
            app.update();
        }
        assert_eq!(
            *app.world().resource::<SceneAnchor>(),
            SceneAnchor::Ready("a1b2".into())
        );
    }

    /// A platform that denies every permission.
    struct Denying(ftgsxr_io::AppPaths);

    impl ftgsxr_app::Platform for Denying {
        fn paths(&self) -> &ftgsxr_io::AppPaths {
            &self.0
        }
        fn pick(&self, _: ftgsxr_app::PickKind) -> ftgsxr_app::PickHandle {
            ftgsxr_app::PickHandle(0)
        }
        fn poll_pick(
            &self,
            _: ftgsxr_app::PickHandle,
        ) -> Poll<Result<Vec<ftgsxr_io::ModelRef>, String>> {
            Poll::Ready(Ok(Vec::new()))
        }
        fn request_permission(&self, _: Permission) -> ftgsxr_app::PermissionHandle {
            ftgsxr_app::PermissionHandle(0)
        }
        fn poll_permission(&self, _: ftgsxr_app::PermissionHandle) -> Poll<bool> {
            Poll::Ready(false)
        }
        fn document_provider(&self) -> Option<Arc<dyn ftgsxr_io::DocumentProvider>> {
            None
        }
        fn camera_frames(&self) -> Option<Arc<dyn ftgsxr_app::CameraFrames>> {
            None
        }
    }

    /// Q-348, Q-261: a denied permission toasts naming "anchors", sets the feature
    /// `Denied` and lists the permission; the save gets no anchor.
    #[test]
    fn handshake_denied_permission_toasts_and_marks_the_feature_denied() {
        let mut app = app(Fake::default(), Pose::IDENTITY);
        let dir = PathBuf::from("/nonexistent");
        app.insert_resource(PlatformRes(Arc::new(Denying(ftgsxr_io::AppPaths {
            config: dir.clone(),
            cache: dir.clone(),
            models: dir.clone(),
            scenes: dir.clone(),
            screenshots: dir.clone(),
            logs: dir,
        }))))
        .init_resource::<DeniedPermissions>();
        let toasts = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen = toasts.clone();
        app.add_observer(move |t: bevy::prelude::On<Toast>| {
            seen.lock().expect("unpoisoned").push(t.text.clone());
        });
        app.insert_resource(SceneAnchor::Creating);
        for _ in 0..3 {
            app.update();
        }
        assert_eq!(*app.world().resource::<SceneAnchor>(), SceneAnchor::None);
        assert_eq!(
            app.world().resource::<XrFeatures>().anchors,
            FeatureState::Denied
        );
        assert_eq!(
            app.world().resource::<DeniedPermissions>().0,
            vec![Permission::Anchors]
        );
        let toasts = toasts.lock().expect("unpoisoned");
        assert_eq!(toasts.len(), 1, "{toasts:?}");
        assert!(
            toasts[0].contains("feature anchors unavailable"),
            "{toasts:?}"
        );
    }

    /// Q-254 (a): an anchor located at its creation pose plus a 2° tilt about x places the
    /// group with zero pitch and roll, the anchor's yaw and its translation; an instance
    /// under the group keeps its stage height and footprint.
    #[test]
    fn placed_group_drops_a_two_degree_tilt_and_keeps_yaw_and_translation() {
        let group = pose([0.4, 0.0, -1.2], yaw(25.0));
        let created = creation_pose(group, root());
        let tilted = Pose {
            rotation: created.rotation * Quat::from_rotation_x(2f32.to_radians()),
            ..created
        };
        let placed = placed_group(tilted, root());
        assert!(
            close(placed.rotation * Vec3::Y, Vec3::Y, 1e-6),
            "{placed:?}"
        );
        assert!((yaw_of(placed.rotation) - 25f32.to_radians()).abs() <= 1e-6);
        assert!(close(placed.translation, group.translation, 1e-5));
        assert!((placed.scale - 1.0).abs() <= 1e-6);
        // An instance at 0.5 m above the group's origin and 0.3 m to its side.
        let instance = pose([0.3, 0.5, 0.0], yaw(-40.0));
        let saved = instance.then(&group);
        let now = instance.then(&placed);
        assert!(
            close(saved.translation, now.translation, 1e-5),
            "{saved:?} {now:?}"
        );
        for c in [
            Vec3::new(-0.2, 0.0, -0.2),
            Vec3::new(0.2, 0.0, 0.2),
            Vec3::new(0.2, 0.0, -0.2),
        ] {
            assert!(close(
                saved.transform_point(c),
                now.transform_point(c),
                1e-5
            ));
        }
    }

    /// Q-254 (b): a 30° yaw under a 5° tilt about (1, 0, 1)/√2 gives a yaw of 30° about +y
    /// to 10⁻⁶ rad, whichever order they compose in.
    #[test]
    fn placed_group_takes_the_twist_whichever_order_tilt_and_yaw_compose() {
        let axis = Vec3::new(1.0, 0.0, 1.0).normalize();
        let tilt = Quat::from_axis_angle(axis, 5f32.to_radians());
        for q in [yaw(30.0) * tilt, tilt * yaw(30.0)] {
            let placed = placed_group(pose([0.0, 1.0, -1.0], q), Pose::IDENTITY);
            assert!(
                close(placed.rotation * Vec3::Y, Vec3::Y, 1e-6),
                "{placed:?}"
            );
            let d = (yaw_of(placed.rotation) - 30f32.to_radians()).abs();
            assert!(d <= 1e-6, "{d}");
        }
    }

    /// Q-254 (c): an anchor located 3 cm above the height it was created at places the
    /// group 3 cm higher.
    #[test]
    fn placed_group_follows_a_located_height() {
        let group = pose([0.4, 0.2, -1.2], yaw(-15.0));
        let created = creation_pose(group, root());
        let higher = Pose {
            translation: created.translation + Vec3::new(0.0, 0.03, 0.0),
            ..created
        };
        let placed = placed_group(higher, root());
        assert!(close(
            placed.translation,
            group.translation + Vec3::new(0.0, 0.03, 0.0),
            1e-5
        ));
    }
}
