// SPDX-License-Identifier: Apache-2.0

//! Dv-9 on the device (§3.4, §1.9 Floor, shadows and anchors, D-26), as Q-346 reads it
//! without a person. It lives here, not in `ftgsxr-quest`, because it needs `anchors`'
//! crate-private store; the quest runner starts it by spawning an entity named
//! [`TRIGGER`] (Q-308), and, on the second launch, a second entity named
//! `ftgsxr-anchor: <detail>` carrying the first launch's detail from the intent extra
//! `ftgsxr-anchor` (Q-346). It prints the verdict line
//! `ftgsxr-test: dv9 PASS|FAIL <detail>` itself (§3.4, Q-122).
//!
//! The first launch creates an anchor 1 m ahead of the content group's origin, persists
//! it, and prints `uuid=<uuid> pose=<x>,<y>,<z>,<qx>,<qy>,<qz>,<qw>` (the created pose
//! in the primary reference space) with the frames and milliseconds the creation's and
//! the save's completion events took (PRD #565's spike check 4). The second queries the
//! UUID and locates it (spike check 5: found after a restart, or not), PASS when the
//! located position lies within 5 cm of the created one and the erase succeeds.

use std::task::Poll;
use std::time::{Duration, Instant};

use bevy::app::{App, PostUpdate};
use bevy::log::info;
use bevy::prelude::{
    Commands, GlobalTransform, IntoScheduleConfigs, Name, Query, Res, ResMut, Resource, Transform,
    With, Without, not, resource_exists,
};
use bevy_mod_xr::session::XrTrackingRoot;
use ftgsxr_app::{ContentGroup, Permission, PermissionHandle, PlatformRes};
use ftgsxr_core::{Pose, Quat, Vec3};

use super::{AnchorId, Anchors, Status, creation_pose, pose_of};

/// The name of the entity the quest runner spawns to start the test (Q-308).
// Q-308
pub(crate) const TRIGGER: &str = "ftgsxr-test: dv9";
/// The prefix of the entity carrying the first launch's detail (Q-346).
// Q-346
pub(crate) const PAYLOAD: &str = "ftgsxr-anchor: ";
/// The test's name in the verdict line.
const NAME: &str = "dv9";
/// §3.4 Dv-9's criterion: within 5 cm.
const WITHIN_M: f32 = 0.05;
/// How long the store, the permission and each step may take.
const TIMEOUT: Duration = Duration::from_secs(60);

/// The first launch's anchor: its UUID and created pose.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Created {
    pub(crate) uuid: String,
    pub(crate) pose: Pose,
}

impl Created {
    /// `uuid=<uuid> pose=<x>,<y>,<z>,<qx>,<qy>,<qz>,<qw>`.
    pub(crate) fn detail(&self) -> String {
        let (t, q) = (self.pose.translation, self.pose.rotation);
        format!(
            "uuid={} pose={},{},{},{},{},{},{}",
            self.uuid, t.x, t.y, t.z, q.x, q.y, q.z, q.w
        )
    }

    /// The anchor in `detail`, a first launch's verdict detail (other words ignored).
    pub(crate) fn parse(detail: &str) -> Option<Created> {
        let mut uuid = None;
        let mut pose = None;
        for word in detail.split_whitespace() {
            if let Some(u) = word.strip_prefix("uuid=") {
                uuid = Some(u.trim_end_matches(';').to_owned());
            } else if let Some(p) = word.strip_prefix("pose=") {
                let v: Vec<f32> = p
                    .trim_end_matches(';')
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()
                    .ok()?;
                if v.len() != 7 {
                    return None;
                }
                pose = Some(Pose {
                    translation: Vec3::new(v[0], v[1], v[2]),
                    rotation: Quat::from_xyzw(v[3], v[4], v[5], v[6]),
                    scale: 1.0,
                });
            }
        }
        Some(Created {
            uuid: uuid?,
            pose: pose?,
        })
    }
}

/// Where the test is.
#[derive(Clone, Debug)]
enum Phase {
    /// Waiting for the adapter (a running session with the extensions).
    Store,
    /// Waiting for `Permission::Anchors`.
    Permission(PermissionHandle),
    /// First launch: the create issued.
    Creating {
        id: AnchorId,
        pose: Pose,
    },
    /// First launch: the persist issued.
    Persisting {
        id: AnchorId,
        created: Created,
    },
    /// Second launch: the query issued.
    Querying,
    /// Second launch: found, waiting for a locate.
    Locating,
    /// Second launch: the erase issued.
    Erasing {
        distance: f32,
    },
    Done,
}

/// The test's state.
#[derive(Resource)]
pub(crate) struct Dv9 {
    /// The first launch's anchor, on the second launch.
    second: Option<Created>,
    phase: Phase,
    /// When the current step began, and its frame count.
    since: Instant,
    frames: u32,
    parts: Vec<String>,
    pass: bool,
}

impl Dv9 {
    fn step(&mut self, phase: Phase) {
        self.phase = phase;
        self.since = Instant::now();
        self.frames = 0;
    }

    /// The step's latency, `<n> frames (<ms> ms)`.
    fn latency(&self) -> String {
        format!(
            "{} frames ({} ms)",
            self.frames,
            self.since.elapsed().as_millis()
        )
    }

    fn finish(&mut self, pass: bool, part: String) {
        self.pass &= pass;
        self.parts.push(part);
        let word = if self.pass { "PASS" } else { "FAIL" };
        // §3.4, Q-122: the verdict line, as `ftgsxr-quest`'s runner prints it.
        info!("ftgsxr-test: {NAME} {word} {}", self.parts.join("; "));
        self.phase = Phase::Done;
    }
}

/// Adds the test's systems: inert until an entity named [`TRIGGER`] exists.
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn add(app: &mut App) {
    app.add_systems(
        PostUpdate,
        (
            detect.run_if(not(resource_exists::<Dv9>)),
            run.run_if(resource_exists::<Dv9>).before(super::handshake),
        ),
    );
}

/// Starts the test when the quest runner's entity appears (Q-308), with the first
/// launch's anchor when the payload entity carries one (Q-346).
#[cfg_attr(test, allow(dead_code))]
fn detect(names: Query<&Name>, mut commands: Commands) {
    if !names.iter().any(|n| n.as_str() == TRIGGER) {
        return;
    }
    let payload = names
        .iter()
        .find_map(|n| n.as_str().strip_prefix(PAYLOAD).map(str::to_owned));
    let second = payload.as_deref().and_then(Created::parse);
    info!(
        "dv9: started ({})",
        match &second {
            Some(c) => format!("second launch, anchor {}", c.uuid),
            None => "first launch".to_owned(),
        }
    );
    let mut test = Dv9 {
        second,
        phase: Phase::Store,
        since: Instant::now(),
        frames: 0,
        parts: Vec::new(),
        pass: true,
    };
    if payload.is_some() && test.second.is_none() {
        test.finish(
            false,
            format!("the extra ftgsxr-anchor is unreadable: {payload:?}"),
        );
    }
    commands.insert_resource(test);
}

/// One frame of the test.
#[cfg_attr(test, allow(dead_code))]
fn run(
    mut test: ResMut<Dv9>,
    mut store: ResMut<Anchors>,
    platform: Option<Res<PlatformRes>>,
    root: Query<&GlobalTransform, With<XrTrackingRoot>>,
    groups: Query<&Transform, (With<ContentGroup>, Without<XrTrackingRoot>)>,
) {
    test.frames += 1;
    let timed_out = test.since.elapsed() > TIMEOUT;
    let store = &mut *store.0;
    match test.phase.clone() {
        Phase::Done => {}
        Phase::Store => {
            if store.available() && !root.is_empty() {
                let phase = match platform.as_deref() {
                    Some(p) => Phase::Permission(p.0.request_permission(Permission::Anchors)),
                    None => Phase::Permission(PermissionHandle(u32::MAX)),
                };
                test.step(phase);
            } else if timed_out {
                test.finish(
                    false,
                    format!(
                        "no anchor store after {} s (the spatial-entity extensions absent, or \
                         no session)",
                        TIMEOUT.as_secs()
                    ),
                );
            }
        }
        Phase::Permission(h) => {
            let granted = match platform.as_deref() {
                Some(p) => p.0.poll_permission(h),
                None => Poll::Ready(true),
            };
            match granted {
                Poll::Pending if timed_out => {
                    test.finish(false, "the anchors permission never answered".into());
                }
                Poll::Pending => {}
                Poll::Ready(false) => {
                    test.finish(false, "Permission::Anchors denied".into());
                }
                Poll::Ready(true) => match test.second.clone() {
                    None => {
                        let root = root.iter().next().map_or(Pose::IDENTITY, pose_of);
                        let group = groups.iter().next().map_or(Pose::IDENTITY, |t| Pose {
                            translation: t.translation,
                            rotation: t.rotation,
                            scale: 1.0,
                        });
                        // 1 m ahead of the group's origin, along its forward −z (Q-346).
                        let ahead = Pose {
                            translation: group.transform_point(Vec3::NEG_Z),
                            ..group
                        };
                        let pose = creation_pose(ahead, root);
                        match store.create(pose) {
                            Ok(id) => test.step(Phase::Creating { id, pose }),
                            Err(e) => test.finish(false, format!("create: {e}")),
                        }
                    }
                    Some(c) => match store.query(&AnchorId(c.uuid.clone())) {
                        Ok(()) => test.step(Phase::Querying),
                        Err(e) => test.finish(false, format!("query: {e}")),
                    },
                },
            }
        }
        Phase::Creating { id, pose } => match store.status(&id) {
            Status::Pending if timed_out => test.finish(false, "create: no completion".into()),
            Status::Pending => {}
            Status::Created(uuid) | Status::Persisted(uuid) => {
                // Spike check 4: the creation's completion event.
                let part = format!("created after {}", test.latency());
                test.parts.push(part);
                match store.persist(&id) {
                    Ok(()) => test.step(Phase::Persisting {
                        id,
                        created: Created { uuid: uuid.0, pose },
                    }),
                    Err(e) => test.finish(false, format!("persist: {e}")),
                }
            }
            other => test.finish(false, format!("create: {other:?}")),
        },
        Phase::Persisting { id, created } => match store.status(&id) {
            Status::Persisted(_) => {
                let part = format!("persisted after {}", test.latency());
                test.parts.push(part);
                // The detail the driver passes to the second launch (Q-346).
                test.finish(true, created.detail());
            }
            Status::Created(_) | Status::Pending if timed_out => {
                test.finish(false, "persist: no save result".into());
            }
            Status::Created(_) | Status::Pending => {}
            other => test.finish(false, format!("persist: {other:?}")),
        },
        Phase::Querying => {
            let Some(c) = test.second.clone() else { return };
            match store.status(&AnchorId(c.uuid.clone())) {
                Status::Pending if timed_out => test.finish(false, "query: no completion".into()),
                Status::Pending => {}
                Status::Created(_) | Status::Persisted(_) => {
                    // Spike check 5: the query by UUID found the anchor after a restart.
                    let part = format!("found by its UUID after {}", test.latency());
                    test.parts.push(part);
                    test.step(Phase::Locating);
                }
                Status::NotFound => test.finish(
                    false,
                    "not found by XR_FB_spatial_entity_query after the restart".into(),
                ),
                Status::Failed(e) => test.finish(false, format!("query: {e}")),
            }
        }
        Phase::Locating => {
            let Some(c) = test.second.clone() else { return };
            let id = AnchorId(c.uuid.clone());
            match store.locate(&id) {
                Some(at) => {
                    let distance = (at.translation - c.pose.translation).length();
                    let part = format!(
                        "located after {} at {:.4} m from the created pose (within {WITHIN_M} \
                         m: {})",
                        test.latency(),
                        distance,
                        distance <= WITHIN_M
                    );
                    test.parts.push(part);
                    match store.forget(&id) {
                        Ok(()) => test.step(Phase::Erasing { distance }),
                        Err(e) => test.finish(false, format!("erase: {e}")),
                    }
                }
                None if timed_out => test.finish(false, "never located".into()),
                None => {}
            }
        }
        Phase::Erasing { distance } => {
            let Some(c) = test.second.clone() else { return };
            match store.erased(&c.uuid) {
                Some(Ok(())) => test.finish(distance <= WITHIN_M, "erased".into()),
                Some(Err(e)) => test.finish(false, format!("erase: {e}")),
                None if timed_out => test.finish(false, "erase: no result".into()),
                None => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first launch's detail reaches the second launch intact (Q-346).
    #[test]
    fn created_detail_parses_back() {
        let c = Created {
            uuid: "12345678-9abc-def0-0123-456789abcdef".into(),
            pose: Pose {
                translation: Vec3::new(0.25, 1.5, -2.0),
                rotation: Quat::from_rotation_y(0.3),
                scale: 1.0,
            },
        };
        let detail = format!(
            "created after 3 frames (40 ms); persisted after 9 frames (120 ms); {}",
            c.detail()
        );
        let back = Created::parse(&detail).expect("parsed");
        assert_eq!(back.uuid, c.uuid);
        assert!((back.pose.translation - c.pose.translation).length() < 1e-6);
        assert!(back.pose.rotation.angle_between(c.pose.rotation) < 1e-5);
        assert!(Created::parse("uuid=x pose=1,2,3").is_none());
        assert!(Created::parse("nothing").is_none());
    }
}
