// SPDX-License-Identifier: Apache-2.0

//! The hands and the controllers drawn in VR mode (§1.8 Controllers and Hands, PRD #452):
//! capsules between the joints with the joint radii, and the player's generated
//! low-polygon controller (a 3 × 3 × 10 cm box with a ring) at the grip pose, unlit grey,
//! children of the tracking root, so their transforms are reference-space poses. Each
//! side shows the device that owns it; nothing in passthrough mode. No binary asset.

use bevy::asset::Assets;
use bevy::color::Color;
use bevy::math::primitives::{Capsule3d, Cuboid, Torus};
use bevy::math::{Quat, Vec3};
use bevy::mesh::{Mesh, Mesh3d};
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::prelude::{
    ChildOf, Commands, Component, Entity, Handle, Local, Query, Res, ResMut, Resource, Transform,
    Visibility, With, Without,
};
use bevy_mod_xr::session::XrTrackingRoot;

use super::HandSpace;
use super::hands::{LocatedHands, Ownership, WRIST, located, transform_of};
use crate::HandSource;

/// Whether the hands and controllers are drawn: in VR mode, not in passthrough mode
/// (§1.8). `passthrough`, after `tracking` in §2.1's order, sets it from `Environment`
/// each frame before the meshes are placed; false until then.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DevicesDrawn(pub(crate) bool);

/// The bones drawn as capsules, as joint pairs of XR\_EXT\_hand\_tracking's default set
/// (source-log 276): the wrist to each metacarpal, then along each finger to its tip
/// (the thumb has no intermediate joint). 24 capsules.
// Q-288
pub(crate) const BONES: [(usize, usize); 24] = [
    (WRIST, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (WRIST, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (WRIST, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (WRIST, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (WRIST, 21),
    (21, 22),
    (22, 23),
    (23, 24),
    (24, 25),
];

/// The controller's box, metres: 3 × 3 cm across, 10 cm along the grip's −z (PRD #452's
/// geometry).
// Q-316
const BOX_M: Vec3 = Vec3::new(0.03, 0.03, 0.10);
/// The ring around the box's front end: its radius and its tube's radius, metres.
// Q-288, Q-316
const RING_M: (f32, f32) = (0.04, 0.004);
/// The neutral grey (§1.8 "a neutral material").
// Q-288
const GREY: f32 = 0.5;

/// The hands' and controllers' material: unlit sRGB grey, shared with the gauntlets
/// "draw gauntlets" draws in passthrough (`occluders`, §1.9).
pub(crate) fn grey() -> StandardMaterial {
    StandardMaterial {
        base_color: Color::srgb(GREY, GREY, GREY),
        unlit: true,
        ..StandardMaterial::default()
    }
}

/// One capsule of a hand: its side and bone; its mesh is made at the first frame both
/// joints are located, from that frame's length and radii (a bone's length is fixed).
#[derive(Component, Clone, Copy)]
pub(crate) struct HandCapsule {
    side: usize,
    bone: usize,
    meshed: bool,
}

/// One side's controller mesh.
#[derive(Component, Clone, Copy)]
pub(crate) struct ControllerMesh {
    side: usize,
}

/// A capsule's place between two joint positions: the midpoint, the capsule's +y along
/// the bone, and its length between the joints' centres (`None` for coincident joints).
pub(crate) fn bone_transform(a: Vec3, b: Vec3) -> Option<(Transform, f32)> {
    let d = b - a;
    let len = d.length();
    if len <= f32::EPSILON {
        return None;
    }
    let t = Transform {
        translation: (a + b) * 0.5,
        rotation: Quat::from_rotation_arc(Vec3::Y, d / len),
        scale: Vec3::ONE,
    };
    Some((t, len))
}

/// The capsules and the two controllers, hidden, under the tracking root.
fn spawn(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    grey: &Handle<StandardMaterial>,
    root: Entity,
) {
    for side in 0..2 {
        for bone in 0..BONES.len() {
            commands.spawn((
                HandCapsule {
                    side,
                    bone,
                    meshed: false,
                },
                MeshMaterial3d(grey.clone()),
                Transform::default(),
                Visibility::Hidden,
                ChildOf(root),
            ));
        }
        let body = meshes.add(Cuboid::new(BOX_M.x, BOX_M.y, BOX_M.z));
        let ring = meshes.add(Torus::new(RING_M.0 - RING_M.1, RING_M.0 + RING_M.1));
        commands
            .spawn((
                ControllerMesh { side },
                Transform::default(),
                Visibility::Hidden,
                ChildOf(root),
            ))
            .with_children(|c| {
                // The box's centre 5 cm ahead of the grip, along −z.
                c.spawn((
                    Mesh3d(body),
                    MeshMaterial3d(grey.clone()),
                    Transform::from_xyz(0.0, 0.0, -BOX_M.z * 0.5),
                ));
                // The ring around the front end, in the plane across the box.
                c.spawn((
                    Mesh3d(ring),
                    MeshMaterial3d(grey.clone()),
                    Transform::from_xyz(0.0, 0.0, -BOX_M.z)
                        .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)),
                ));
            });
    }
}

/// Spawns the meshes once per tracking root, then each frame places them and shows each
/// side's owner, in VR mode only; skipped in an app without meshes and materials.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn draw_hands_and_controllers(
    drawn: Option<Res<DevicesDrawn>>,
    meshes: Option<ResMut<Assets<Mesh>>>,
    materials: Option<ResMut<Assets<StandardMaterial>>>,
    root: Query<Entity, With<XrTrackingRoot>>,
    hands: Res<LocatedHands>,
    ownership: Res<Ownership>,
    grips: Query<(&HandSpace, &Transform), (Without<HandCapsule>, Without<ControllerMesh>)>,
    mut capsules: Query<
        (Entity, &mut HandCapsule, &mut Transform, &mut Visibility),
        Without<ControllerMesh>,
    >,
    mut controllers: Query<
        (&ControllerMesh, &mut Transform, &mut Visibility),
        Without<HandCapsule>,
    >,
    mut material: Local<Option<Handle<StandardMaterial>>>,
    mut commands: Commands,
) {
    let (Some(mut meshes), Some(mut materials)) = (meshes, materials) else {
        return;
    };
    let Some(root) = root.iter().next() else {
        return;
    };
    let grey = material
        .get_or_insert_with(|| materials.add(grey()))
        .clone();
    if controllers.is_empty() {
        spawn(&mut commands, &mut meshes, &grey, root);
        return;
    }
    let vr = drawn.is_some_and(|d| d.0);
    let shown = |side: usize, source: HandSource| vr && ownership.owners[side] == Some(source);
    for (e, mut capsule, mut t, mut vis) in &mut capsules {
        let (a, b) = BONES[capsule.bone];
        let placed = hands.raw[capsule.side]
            .filter(|_| shown(capsule.side, HandSource::Hand))
            .map(|raw| raw.joints)
            .filter(|j| located(&j[a]) && located(&j[b]))
            .and_then(|j| {
                let (ja, jb) = (transform_of(&j[a].pose), transform_of(&j[b].pose));
                bone_transform(ja.translation, jb.translation)
                    .map(|(bt, len)| (bt, len, (j[a].radius + j[b].radius) * 0.5))
            });
        let Some((bt, len, radius)) = placed else {
            *vis = Visibility::Hidden;
            continue;
        };
        if !capsule.meshed {
            capsule.meshed = true;
            // Q-288: the mean of the two joint radii, the cylinder as long as the bone.
            commands
                .entity(e)
                .insert(Mesh3d(meshes.add(Capsule3d::new(radius, len))));
        }
        *t = bt;
        *vis = Visibility::Visible;
    }
    for (c, mut t, mut vis) in &mut controllers {
        let grip = grips
            .iter()
            .find(|(s, _)| s.hand == c.side && !s.aim)
            .map(|(_, t)| *t)
            .filter(|_| shown(c.side, HandSource::Controller));
        match grip {
            Some(g) => {
                *t = g;
                *vis = Visibility::Visible;
            }
            None => *vis = Visibility::Hidden,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::hands::JOINTS;
    use super::*;

    /// A capsule sits at the bone's midpoint, its +y along the bone, as long as the bone.
    #[test]
    fn bone_transform_spans_the_two_joints() {
        let (t, len) = bone_transform(Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 1.0, -0.04))
            .expect("distinct joints");
        assert!((len - 0.04).abs() < 1e-6);
        assert!(t.translation.distance(Vec3::new(0.0, 1.0, -0.02)) < 1e-6);
        assert!((t.rotation * Vec3::Y).distance(Vec3::NEG_Z) < 1e-6);
        assert!(bone_transform(Vec3::ONE, Vec3::ONE).is_none());
    }

    /// The 24 bones: each of the 26 joints but the palm appears, the wrist starts the five
    /// chains, and every finger ends at its tip (source-log 276).
    #[test]
    fn bones_cover_every_joint_but_the_palm() {
        let mut seen = [false; JOINTS];
        for (a, b) in BONES {
            seen[a] = true;
            seen[b] = true;
            assert!(a < b, "bones run from the wrist outward");
        }
        assert!(!seen[0], "the palm (0) is not drawn");
        assert!(seen[1..].iter().all(|&s| s));
        assert_eq!(BONES.iter().filter(|(a, _)| *a == WRIST).count(), 5);
    }
}
