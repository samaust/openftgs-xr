// SPDX-License-Identifier: Apache-2.0

//! "Occlude with tracked hands" (§1.9 Depth occlusion, §1.8 Hands, D-16, #608, spec
//! revision 40): in passthrough mode with `[environment] hand_occluders` on, each tracked
//! hand is drawn, opaque in VR mode's neutral grey (`meshes::grey`), so the drawn hand
//! hides what lies behind it, the splats (which test against the opaque depth, §1.4), the
//! contact shadow and the panels alike, at its tracked pose (the locate's predicted
//! display time, `hands::locate_hands`) instead of at the depth image's capture time.
//!
//! Its shape is VR mode's hand, §1.8's 24 bone capsules with their joints' mean radius
//! (`meshes::BONES`, Q-288), or with `[environment] hand_gauntlets` the gauntlet: the
//! same bones and two palm capsules, from the wrist to the index and to the little
//! finger's metacarpophalangeal joints, each radius plus the margin m = 1.5 cm. A capsule
//! whose two joints are not both located is hidden, so an untracked hand draws nothing.
//! The capsules are children of the tracking root.

use bevy::asset::{Assets, Handle};
use bevy::camera::visibility::NoFrustumCulling;
use bevy::math::Vec3;
use bevy::math::primitives::Capsule3d;
use bevy::mesh::{Mesh, Mesh3d};
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::prelude::{
    ChildOf, Commands, Component, DetectChangesMut, Entity, GlobalTransform, Local, Query, Res,
    ResMut, Resource, Transform, Visibility, With, Without,
};
use bevy_mod_xr::session::XrTrackingRoot;
use openxr::sys;

use super::hands::{JOINTS, LocatedHands, WRIST, located, transform_of};
use super::meshes::{BONES, bone_transform, grey};

/// The shape a tracked hand is drawn in (§1.9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OccluderShape {
    /// VR mode's hand: the bones with their joints' radii.
    Hand,
    /// "Draw gauntlets": the bones and the palm capsules, each radius plus `MARGIN_M`.
    Gauntlet,
}

/// The shape the tracked hands are drawn in, `None` when they are not: in passthrough
/// mode with `[environment] hand_occluders` on (§1.9). `passthrough`, after `tracking` in
/// §2.1's order, sets it each frame before the capsules are placed; `None` until then.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct OccludersDrawn(pub(crate) Option<OccluderShape>);

/// The gauntlet's margin on each capsule's radius, metres (D-16).
pub(crate) const MARGIN_M: f32 = 0.015;

/// XR\_HAND\_JOINT\_INDEX\_PROXIMAL\_EXT and XR\_HAND\_JOINT\_LITTLE\_PROXIMAL\_EXT: a
/// finger's joints are named after the bone that starts at them (source-log 374), so
/// each proximal joint is the finger's metacarpophalangeal joint; their values are the
/// default set's (source-log 276).
const INDEX_MCP: usize = 7;
const LITTLE_MCP: usize = 22;

/// Every capsule a tracked hand may draw, as joint pairs: §1.8's 24 bones, then the
/// gauntlet's two palm capsules (§1.9).
pub(crate) const CAPSULES: [(usize, usize); BONES.len() + 2] = {
    let mut c = [(0, 0); BONES.len() + 2];
    let mut i = 0;
    while i < BONES.len() {
        c[i] = BONES[i];
        i += 1;
    }
    c[BONES.len()] = (WRIST, INDEX_MCP);
    c[BONES.len() + 1] = (WRIST, LITTLE_MCP);
    c
};

/// One capsule: the centres of its two end spheres and its radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Capsule {
    pub(crate) a: Vec3,
    pub(crate) b: Vec3,
    pub(crate) radius: f32,
}

/// One hand's capsules in `shape`, in `CAPSULES`' order: each between its two joints'
/// positions, of their mean radius (Q-288) plus `MARGIN_M` for the gauntlet; `None` where
/// either joint is not located, and for the palm capsules of VR mode's hand.
pub(crate) fn occluder(
    joints: &[sys::HandJointLocationEXT; JOINTS],
    shape: OccluderShape,
) -> Vec<Option<Capsule>> {
    let (count, margin) = match shape {
        OccluderShape::Hand => (BONES.len(), 0.0),
        OccluderShape::Gauntlet => (CAPSULES.len(), MARGIN_M),
    };
    CAPSULES
        .iter()
        .enumerate()
        .map(|(k, &(a, b))| {
            let (ja, jb) = (&joints[a], &joints[b]);
            (k < count && located(ja) && located(jb)).then(|| Capsule {
                a: transform_of(&ja.pose).translation,
                b: transform_of(&jb.pose).translation,
                radius: (ja.radius + jb.radius) * 0.5 + margin,
            })
        })
        .collect()
}

/// One drawn capsule: its side, its index in `CAPSULES`, and the shape its mesh was made
/// for (at the first frame both joints are located, as Q-288's, and again when the shape
/// changes).
#[derive(Component, Clone, Copy)]
pub(crate) struct OccluderCapsule {
    side: usize,
    index: usize,
    meshed: Option<OccluderShape>,
}

/// Spawns the capsules once per tracking root, then each frame places each one between
/// its joints and shows it while the hands are drawn and both joints are located. The
/// capsules' global transforms are set here from the root's, which the propagation has
/// already updated, so this frame's pose is drawn this frame; frustum culling is off since
/// it may run before them. Skipped in an app without meshes and materials.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn draw_occluders(
    drawn: Option<Res<OccludersDrawn>>,
    meshes: Option<ResMut<Assets<Mesh>>>,
    materials: Option<ResMut<Assets<StandardMaterial>>>,
    root: Query<(Entity, &GlobalTransform), With<XrTrackingRoot>>,
    hands: Res<LocatedHands>,
    // Without the root, whose global transform `root` reads (B0001).
    mut capsules: Query<
        (
            Entity,
            &mut OccluderCapsule,
            &mut Transform,
            &mut GlobalTransform,
            &mut Visibility,
            Option<&Mesh3d>,
        ),
        Without<XrTrackingRoot>,
    >,
    mut material: Local<Option<Handle<StandardMaterial>>>,
    mut commands: Commands,
) {
    let (Some(mut meshes), Some(mut materials)) = (meshes, materials) else {
        return;
    };
    let Some((root, root_global)) = root.iter().next() else {
        return;
    };
    if capsules.is_empty() {
        let m = material
            .get_or_insert_with(|| materials.add(grey()))
            .clone();
        for side in 0..2 {
            for index in 0..CAPSULES.len() {
                commands.spawn((
                    OccluderCapsule {
                        side,
                        index,
                        meshed: None,
                    },
                    MeshMaterial3d(m.clone()),
                    Transform::default(),
                    Visibility::Hidden,
                    NoFrustumCulling,
                    ChildOf(root),
                ));
            }
        }
        return;
    }
    let shape = drawn.and_then(|d| d.0);
    let placed: [Option<Vec<Option<Capsule>>>; 2] = [0, 1].map(|side| {
        hands.raw[side]
            .zip(shape)
            .map(|(raw, shape)| occluder(&raw.joints, shape))
    });
    for (e, mut capsule, mut t, mut global, mut vis, mesh) in &mut capsules {
        let at = placed[capsule.side]
            .as_ref()
            .and_then(|c| c[capsule.index])
            .and_then(|c| bone_transform(c.a, c.b).map(|(bt, len)| (bt, len, c.radius)));
        let (Some((bt, len, radius)), Some(shape)) = (at, shape) else {
            vis.set_if_neq(Visibility::Hidden);
            continue;
        };
        if capsule.meshed != Some(shape) {
            capsule.meshed = Some(shape);
            if let Some(old) = mesh {
                meshes.remove(&old.0);
            }
            commands
                .entity(e)
                .insert(Mesh3d(meshes.add(Capsule3d::new(radius, len))));
        }
        *t = bt;
        *global = root_global.mul_transform(bt);
        vis.set_if_neq(Visibility::Visible);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn posef(x: f32, y: f32, z: f32) -> sys::Posef {
        sys::Posef {
            orientation: sys::Quaternionf::IDENTITY,
            position: sys::Vector3f { x, y, z },
        }
    }

    /// Joint i at (i cm, 1 m, 0) with radius (1 + i/10) mm, all located.
    fn joints() -> [sys::HandJointLocationEXT; JOINTS] {
        let mut j = [sys::HandJointLocationEXT::default(); JOINTS];
        for (i, joint) in j.iter_mut().enumerate() {
            joint.location_flags = sys::SpaceLocationFlags::POSITION_VALID
                | sys::SpaceLocationFlags::ORIENTATION_VALID;
            joint.pose = posef(i as f32 * 0.01, 1.0, 0.0);
            joint.radius = 0.001 * (1.0 + i as f32 / 10.0);
        }
        j
    }

    /// §1.9, D-16: the gauntlet is §1.8's 24 bones, then the wrist to the index and to
    /// the little finger's metacarpophalangeal joints; each capsule between its joints, of
    /// their mean radius plus 1.5 cm.
    #[test]
    fn occluder_gauntlet_is_the_bones_and_two_palm_capsules_with_the_margin() {
        assert_eq!(CAPSULES.len(), 26);
        assert_eq!(&CAPSULES[..24], &BONES[..]);
        assert_eq!(CAPSULES[24], (WRIST, 7), "the index's proximal joint");
        assert_eq!(
            CAPSULES[25],
            (WRIST, 22),
            "the little finger's proximal joint"
        );
        let g = occluder(&joints(), OccluderShape::Gauntlet);
        assert_eq!(g.len(), 26);
        for (c, &(a, b)) in g.iter().zip(&CAPSULES) {
            let c = c.expect("every joint is located");
            assert!((c.a - Vec3::new(a as f32 * 0.01, 1.0, 0.0)).length() < 1e-6);
            assert!((c.b - Vec3::new(b as f32 * 0.01, 1.0, 0.0)).length() < 1e-6);
        }
        // Radii worked by hand from the joints' (1 + i/10) mm: the mean of the two joints'
        // plus 15 mm. Wrist (1.1 mm) to the thumb's metacarpal (1.2 mm): 16.15 mm; the
        // index's distal (1.9 mm) to its tip (2.0 mm): 16.95 mm; the palm capsules, wrist
        // to the index's proximal (1.7 mm): 16.4 mm, to the little finger's (3.2 mm):
        // 17.15 mm.
        for (k, want) in [(0, 0.01615), (8, 0.01695), (24, 0.0164), (25, 0.01715)] {
            let r = g[k].expect("located").radius;
            assert!((r - want).abs() < 1e-7, "capsule {k}: {r}");
        }
    }

    /// §1.9 (revision 40): VR mode's hand is the 24 bones alone, of the joints' mean
    /// radius with no margin (Q-288), and no palm capsule.
    #[test]
    fn occluder_hand_is_vr_modes_bones_without_the_margin() {
        let h = occluder(&joints(), OccluderShape::Hand);
        assert_eq!(h.len(), 26);
        assert!(h[..24].iter().all(Option::is_some));
        assert!(h[24].is_none() && h[25].is_none(), "no palm capsule");
        // Wrist (1.1 mm) to the thumb's metacarpal (1.2 mm): 1.15 mm; the index's distal
        // (1.9 mm) to its tip (2.0 mm): 1.95 mm.
        for (k, want) in [(0, 0.00115), (8, 0.00195)] {
            let r = h[k].expect("located").radius;
            assert!((r - want).abs() < 1e-7, "capsule {k}: {r}");
        }
    }

    /// An unlocated joint takes out the capsules that end at it, and an untracked hand
    /// (no joint located) draws nothing.
    #[test]
    fn occluder_leaves_out_capsules_of_unlocated_joints() {
        let mut j = joints();
        j[7].location_flags = sys::SpaceLocationFlags::POSITION_VALID;
        let g = occluder(&j, OccluderShape::Gauntlet);
        for (c, &(a, b)) in g.iter().zip(&CAPSULES) {
            assert_eq!(c.is_none(), a == 7 || b == 7, "{a}-{b}");
        }
        let untracked = [sys::HandJointLocationEXT::default(); JOINTS];
        for shape in [OccluderShape::Hand, OccluderShape::Gauntlet] {
            assert!(occluder(&untracked, shape).iter().all(Option::is_none));
        }
    }

    /// `draw_occluders` in an app (#608: its queries once conflicted, B0001, which only
    /// shows when the system runs): it spawns 2 × 26 hidden capsules under the tracking
    /// root, then shows a tracked hand's 24 bones in VR mode's shape, or its 26 capsules
    /// as a gauntlet (its meshes made again, larger), where the root puts them, with their
    /// global transforms set this frame; nothing while `OccludersDrawn` is `None`.
    #[test]
    fn draw_occluders_places_a_tracked_hands_capsules_under_the_root() {
        use bevy::app::{App, Update};
        use bevy::ecs::world::World;

        use super::super::hands::RawHand;

        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<LocatedHands>()
            .insert_resource(OccludersDrawn(None))
            .add_systems(Update, draw_occluders);
        let lift = Transform::from_xyz(0.0, 0.5, 0.0);
        app.world_mut()
            .spawn((XrTrackingRoot, lift, GlobalTransform::from(lift)));
        app.update();
        let shown = |world: &mut World| {
            let mut q = world.query::<(&OccluderCapsule, &Visibility, &GlobalTransform)>();
            q.iter(world)
                .filter(|(_, v, _)| **v == Visibility::Visible)
                .map(|(c, _, g)| (c.side, c.index, g.translation()))
                .collect::<Vec<_>>()
        };
        let all = app
            .world_mut()
            .query::<&OccluderCapsule>()
            .iter(app.world())
            .count();
        assert_eq!(all, 2 * 26, "spawned hidden");
        assert!(shown(app.world_mut()).is_empty());

        app.world_mut().resource_mut::<LocatedHands>().raw[1] = Some(RawHand {
            joints: joints(),
            aim: None,
        });
        app.update();
        assert!(shown(app.world_mut()).is_empty(), "not drawn");

        // Joint i sits at (i cm, 1 m, 0) and the root lifts it by 0.5 m: a capsule between
        // joints a and b is centred at ((a + b) / 2 cm, 1.5 m, 0).
        let check = |s: Vec<(usize, usize, Vec3)>| {
            for (side, index, at) in s {
                assert_eq!(side, 1);
                let (a, b) = CAPSULES[index];
                let mid = Vec3::new((a + b) as f32 * 0.005, 1.5, 0.0);
                assert!(
                    at.distance(mid) < 1e-5,
                    "capsule {index} at {at}, not {mid}"
                );
            }
        };
        app.insert_resource(OccludersDrawn(Some(OccluderShape::Hand)));
        app.update();
        let s = shown(app.world_mut());
        assert_eq!(s.len(), 24, "VR mode's hand: {s:?}");
        check(s);
        assert_eq!(app.world().resource::<Assets<Mesh>>().len(), 24);

        app.insert_resource(OccludersDrawn(Some(OccluderShape::Gauntlet)));
        app.update();
        let s = shown(app.world_mut());
        assert_eq!(s.len(), 26, "the gauntlet: {s:?}");
        check(s);
        app.update();
        assert_eq!(
            app.world().resource::<Assets<Mesh>>().len(),
            26,
            "the hand's meshes replaced by the gauntlet's"
        );
    }
}
