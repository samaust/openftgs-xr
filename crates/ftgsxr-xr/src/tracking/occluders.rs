// SPDX-License-Identifier: Apache-2.0

//! "Occlude with tracked hands" (§1.9 Depth occlusion, §1.8 Hands, D-16, #608, spec
//! revision 41): in passthrough mode with `[environment] hand_occluders` on, each tracked
//! hand is drawn as a gauntlet: one capsule per bone of §1.8's capsules (`meshes::BONES`,
//! the wrist to each fingertip) and two palm capsules, from the wrist to the index and to
//! the little finger's metacarpophalangeal joints, each of its joints' mean radius (as
//! Q-288's) plus the margin m, `[environment] hand_gauntlet_margin_m`.
//!
//! The gauntlet writes depth only and no color, so the real hand stays visible and the
//! gauntlet hides what lies behind it, the splats (which test against the opaque depth,
//! §1.4), the contact shadow and the panels (both drawn in the alpha-mask phase after
//! the opaque one, Q-364, Q-368), at its tracked pose (the locate's predicted display
//! time, `hands::locate_hands`) instead of at the depth image's capture time. With
//! `[environment] hand_gauntlets` ("draw gauntlets") the same gauntlet is drawn opaque in
//! the hands' grey (`meshes::grey`) instead. A capsule whose two joints are not both
//! located is hidden, so an untracked hand draws nothing. The capsules are children of
//! the tracking root.
//!
//! The depth-only material exists where Bevy's `PbrPlugin` does (the player's plugin
//! groups); in an app without it nothing is drawn.

use bevy::app::App;
use bevy::asset::{Asset, Assets, Handle, uuid_handle};
use bevy::camera::visibility::NoFrustumCulling;
use bevy::math::Vec3;
use bevy::math::primitives::Capsule3d;
use bevy::mesh::{Mesh, Mesh3d, MeshVertexBufferLayoutRef};
use bevy::pbr::{
    Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin, MeshMaterial3d, PbrPlugin,
    StandardMaterial,
};
use bevy::prelude::{
    ChildOf, Commands, Component, DetectChangesMut, Entity, GlobalTransform, Local, Query, Res,
    ResMut, Resource, Transform, Visibility, With, Without,
};
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, ColorWrites, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::{Shader, ShaderRef};
use bevy_mod_xr::session::XrTrackingRoot;
use openxr::sys;

use super::hands::{JOINTS, LocatedHands, WRIST, located, transform_of};
use super::meshes::{BONES, bone_transform, grey};

/// How the tracked hands' gauntlets are drawn (§1.9).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Occluders {
    /// "Draw gauntlets": opaque in the hands' grey; otherwise depth only.
    pub(crate) grey: bool,
    /// The margin m over each capsule's joint radius, metres.
    pub(crate) margin_m: f32,
}

/// The tracked hands' gauntlets, `None` when there are none: in passthrough mode with
/// `[environment] hand_occluders` on (§1.9). `passthrough`, after `tracking` in §2.1's
/// order, sets it each frame before the capsules are placed; `None` until then.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct OccludersDrawn(pub(crate) Option<Occluders>);

/// XR\_HAND\_JOINT\_INDEX\_PROXIMAL\_EXT and XR\_HAND\_JOINT\_LITTLE\_PROXIMAL\_EXT: a
/// finger's joints are named after the bone that starts at them (source-log 374), so
/// each proximal joint is the finger's metacarpophalangeal joint; their values are the
/// default set's (source-log 276).
const INDEX_MCP: usize = 7;
const LITTLE_MCP: usize = 22;

/// The gauntlet's capsules as joint pairs: §1.8's 24 bones, then the two palm capsules
/// (§1.9).
const CAPSULES: [(usize, usize); BONES.len() + 2] = {
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
struct Capsule {
    a: Vec3,
    b: Vec3,
    radius: f32,
}

/// One hand's gauntlet, in `CAPSULES`' order: each capsule between its two joints'
/// positions, of their mean radius (Q-288) plus `margin_m`; `None` where either joint is
/// not located.
fn gauntlet(joints: &[sys::HandJointLocationEXT; JOINTS], margin_m: f32) -> Vec<Option<Capsule>> {
    CAPSULES
        .iter()
        .map(|&(a, b)| {
            let (ja, jb) = (&joints[a], &joints[b]);
            (located(ja) && located(jb)).then(|| Capsule {
                a: transform_of(&ja.pose).translation,
                b: transform_of(&jb.pose).translation,
                radius: (ja.radius + jb.radius) * 0.5 + margin_m,
            })
        })
        .collect()
}

/// The depth-only material's fragment shader.
const SHADER: Handle<Shader> = uuid_handle!("2b8e4f61-7a3c-4d95-b0e2-9c1f5a7d3e48");

/// The shader's source, inline as the contact shadow's (Q-354): its output, opaque
/// magenta, is discarded by the pipeline's empty write mask, so only the depth is
/// written and a test sees any color that leaks.
// Q-354
const SHADER_WGSL: &str = r"// SPDX-License-Identifier: Apache-2.0

// The tracked-hand gauntlet (§1.9 Depth occlusion, D-16): depth only; the pipeline
// writes no color. The output is opaque magenta, so a color write would show.

#import bevy_pbr::forward_io::VertexOutput

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}
";

/// The gauntlet's depth-only material: opaque, in the opaque phase, with its color writes
/// off; the contact shadow's and the panels' alpha-mask phase comes after it (Q-364,
/// Q-368), so its depth hides them whatever the bin order.
// Q-364, Q-368
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug, Default)]
pub(crate) struct DepthOnlyMaterial {}

impl Material for DepthOnlyMaterial {
    fn fragment_shader() -> ShaderRef {
        SHADER.into()
    }

    // Drawn in the main opaque pass only, where the splats test against its depth.
    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }

    /// Depth only (§1.9): every color target's write mask is empty.
    fn specialize(
        _: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _: &MeshVertexBufferLayoutRef,
        _: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        if let Some(fragment) = descriptor.fragment.as_mut() {
            for target in fragment.targets.iter_mut().flatten() {
                target.blend = None;
                target.write_mask = ColorWrites::empty();
            }
        }
        Ok(())
    }
}

/// Registers the depth-only material where `PbrPlugin` is.
pub(crate) fn add_material(app: &mut App) {
    if !app.is_plugin_added::<PbrPlugin>() {
        return;
    }
    app.world_mut()
        .resource_mut::<Assets<Shader>>()
        .insert(
            SHADER.id(),
            Shader::from_wgsl(SHADER_WGSL, "ftgsxr_xr/tracking/gauntlet.wgsl"),
        )
        .expect("invariant: the shader's handle is a UUID one, which `insert` accepts");
    app.add_plugins(MaterialPlugin::<DepthOnlyMaterial>::default());
}

/// One capsule of a gauntlet: its side, its index in `CAPSULES`, the margin its mesh was
/// made with (at the first frame both joints are located, as Q-288's, and again when the
/// margin changes) and its material, the grey or the depth-only one, which it keeps: a
/// material of another type on the same entity made Bevy draw the new bind group with
/// the old pipeline, a validation error that quit the app (#608), so each capsule exists
/// once per material and "draw gauntlets" shows one set or the other.
#[derive(Component, Clone, Copy)]
pub(crate) struct GauntletCapsule {
    side: usize,
    index: usize,
    meshed: Option<f32>,
    grey: bool,
}

/// The two materials, made once.
pub(crate) struct Materials {
    grey: Handle<StandardMaterial>,
    depth: Handle<DepthOnlyMaterial>,
}

/// Spawns the capsules once per tracking root, in both materials, then each frame places
/// each one between its joints and shows it while the gauntlets are on, its material is
/// the one "draw gauntlets" asks for and both joints are located. The capsules' global transforms are set
/// here from the root's, which the propagation has already updated, so this frame's pose
/// is drawn this frame; frustum culling is off since it may run before them. Skipped in
/// an app without meshes and the two materials.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn draw_gauntlets(
    drawn: Option<Res<OccludersDrawn>>,
    meshes: Option<ResMut<Assets<Mesh>>>,
    greys: Option<ResMut<Assets<StandardMaterial>>>,
    depths: Option<ResMut<Assets<DepthOnlyMaterial>>>,
    root: Query<(Entity, &GlobalTransform), With<XrTrackingRoot>>,
    hands: Res<LocatedHands>,
    // Without the root, whose global transform `root` reads (B0001).
    mut capsules: Query<
        (
            Entity,
            &mut GauntletCapsule,
            &mut Transform,
            &mut GlobalTransform,
            &mut Visibility,
            Option<&Mesh3d>,
        ),
        Without<XrTrackingRoot>,
    >,
    mut materials: Local<Option<Materials>>,
    mut commands: Commands,
) {
    let (Some(mut meshes), Some(mut greys), Some(mut depths)) = (meshes, greys, depths) else {
        return;
    };
    let Some((root, root_global)) = root.iter().next() else {
        return;
    };
    let m = materials.get_or_insert_with(|| Materials {
        grey: greys.add(grey()),
        depth: depths.add(DepthOnlyMaterial {}),
    });
    if capsules.is_empty() {
        for grey in [false, true] {
            for side in 0..2 {
                for index in 0..CAPSULES.len() {
                    let mut e = commands.spawn((
                        GauntletCapsule {
                            side,
                            index,
                            meshed: None,
                            grey,
                        },
                        Transform::default(),
                        Visibility::Hidden,
                        NoFrustumCulling,
                        ChildOf(root),
                    ));
                    if grey {
                        e.insert(MeshMaterial3d(m.grey.clone()));
                    } else {
                        e.insert(MeshMaterial3d(m.depth.clone()));
                    }
                }
            }
        }
        return;
    }
    let on = drawn.and_then(|d| d.0);
    let placed: [Option<Vec<Option<Capsule>>>; 2] = [0, 1].map(|side| {
        hands.raw[side]
            .zip(on)
            .map(|(raw, o)| gauntlet(&raw.joints, o.margin_m))
    });
    for (e, mut capsule, mut t, mut global, mut vis, mesh) in &mut capsules {
        let at = placed[capsule.side]
            .as_ref()
            .and_then(|c| c[capsule.index])
            .and_then(|c| bone_transform(c.a, c.b).map(|(bt, len)| (bt, len, c.radius)));
        let (Some((bt, len, radius)), Some(o)) = (at, on.filter(|o| o.grey == capsule.grey)) else {
            vis.set_if_neq(Visibility::Hidden);
            continue;
        };
        if capsule.meshed.is_none_or(|m| (m - o.margin_m).abs() > 1e-6) {
            capsule.meshed = Some(o.margin_m);
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

    /// §1.9, D-16: §1.8's 24 bones, then the wrist to the index and to the little
    /// finger's metacarpophalangeal joints; each capsule between its joints, of their mean
    /// radius plus the chosen margin.
    #[test]
    fn gauntlet_is_the_bones_and_two_palm_capsules_with_the_margin() {
        assert_eq!(CAPSULES.len(), 26);
        assert_eq!(&CAPSULES[..24], &BONES[..]);
        assert_eq!(CAPSULES[24], (WRIST, 7), "the index's proximal joint");
        assert_eq!(
            CAPSULES[25],
            (WRIST, 22),
            "the little finger's proximal joint"
        );
        let g = gauntlet(&joints(), 0.005);
        assert_eq!(g.len(), 26);
        for (c, &(a, b)) in g.iter().zip(&CAPSULES) {
            let c = c.expect("every joint is located");
            assert!((c.a - Vec3::new(a as f32 * 0.01, 1.0, 0.0)).length() < 1e-6);
            assert!((c.b - Vec3::new(b as f32 * 0.01, 1.0, 0.0)).length() < 1e-6);
        }
        // Radii worked by hand from the joints' (1 + i/10) mm: the mean of the two joints'
        // plus 5 mm. Wrist (1.1 mm) to the thumb's metacarpal (1.2 mm): 6.15 mm; the
        // index's distal (1.9 mm) to its tip (2.0 mm): 6.95 mm; the palm capsules, wrist
        // to the index's proximal (1.7 mm): 6.4 mm, to the little finger's (3.2 mm):
        // 7.15 mm.
        for (k, want) in [(0, 0.00615), (8, 0.00695), (24, 0.0064), (25, 0.00715)] {
            let r = g[k].expect("located").radius;
            assert!((r - want).abs() < 1e-7, "capsule {k}: {r}");
        }
        // The slider's ends: the joint radii alone, and 2 cm more.
        let r0 = gauntlet(&joints(), 0.0)[0].expect("located").radius;
        let r2 = gauntlet(&joints(), 0.02)[0].expect("located").radius;
        assert!((r0 - 0.00115).abs() < 1e-7 && (r2 - 0.02115).abs() < 1e-7);
    }

    /// An unlocated joint takes out the capsules that end at it, and an untracked hand
    /// (no joint located) draws nothing.
    #[test]
    fn gauntlet_leaves_out_capsules_of_unlocated_joints() {
        let mut j = joints();
        j[7].location_flags = sys::SpaceLocationFlags::POSITION_VALID;
        let g = gauntlet(&j, 0.005);
        for (c, &(a, b)) in g.iter().zip(&CAPSULES) {
            assert_eq!(c.is_none(), a == 7 || b == 7, "{a}-{b}");
        }
        let untracked = [sys::HandJointLocationEXT::default(); JOINTS];
        assert!(gauntlet(&untracked, 0.005).iter().all(Option::is_none));
    }

    /// `draw_gauntlets` in an app (#608: its queries once conflicted, B0001, which only
    /// shows when the system runs): it spawns 2 × 2 × 26 hidden capsules under the tracking
    /// root, each with one material it keeps (#608: a swapped material quit the app), then
    /// shows a tracked hand's 26 depth-only ones where the root puts them, with their
    /// global transforms set this frame; the grey ones instead with "draw gauntlets"; new
    /// meshes when the margin changes; nothing while `OccludersDrawn` is `None`.
    #[test]
    fn draw_gauntlets_places_a_tracked_hands_capsules_under_the_root() {
        use bevy::app::{App, Update};
        use bevy::ecs::world::World;

        use super::super::hands::RawHand;

        let mut app = App::new();
        app.init_resource::<Assets<Mesh>>()
            .init_resource::<Assets<StandardMaterial>>()
            .init_resource::<Assets<DepthOnlyMaterial>>()
            .init_resource::<LocatedHands>()
            .insert_resource(OccludersDrawn(None))
            .add_systems(Update, draw_gauntlets);
        let lift = Transform::from_xyz(0.0, 0.5, 0.0);
        app.world_mut()
            .spawn((XrTrackingRoot, lift, GlobalTransform::from(lift)));
        app.update();
        let shown = |world: &mut World| {
            let mut q = world.query::<(
                &GauntletCapsule,
                &Visibility,
                &GlobalTransform,
                Option<&MeshMaterial3d<StandardMaterial>>,
                Option<&MeshMaterial3d<DepthOnlyMaterial>>,
            )>();
            q.iter(world)
                .filter(|(_, v, ..)| **v == Visibility::Visible)
                .map(|(c, _, g, grey, depth)| {
                    assert!(grey.is_some() != depth.is_some(), "one material");
                    (c.side, c.index, g.translation(), grey.is_some())
                })
                .collect::<Vec<_>>()
        };
        let all = app
            .world_mut()
            .query::<&GauntletCapsule>()
            .iter(app.world())
            .count();
        assert_eq!(all, 2 * 2 * 26, "spawned hidden");
        let materials = |world: &mut World| {
            let mut q = world.query::<(
                Entity,
                &GauntletCapsule,
                Option<&MeshMaterial3d<StandardMaterial>>,
                Option<&MeshMaterial3d<DepthOnlyMaterial>>,
            )>();
            let mut m: Vec<_> = q
                .iter(world)
                .map(|(e, c, g, d)| {
                    assert_eq!(
                        (g.is_some(), d.is_some()),
                        (c.grey, !c.grey),
                        "its material"
                    );
                    (e, c.grey)
                })
                .collect();
            m.sort();
            m
        };
        let spawned = materials(app.world_mut());
        assert!(shown(app.world_mut()).is_empty());

        app.world_mut().resource_mut::<LocatedHands>().raw[1] = Some(RawHand {
            joints: joints(),
            aim: None,
        });
        app.update();
        assert!(shown(app.world_mut()).is_empty(), "not drawn");

        // Joint i sits at (i cm, 1 m, 0) and the root lifts it by 0.5 m: a capsule between
        // joints a and b is centred at ((a + b) / 2 cm, 1.5 m, 0).
        let check = |s: &[(usize, usize, Vec3, bool)], grey: bool| {
            assert_eq!(s.len(), 26, "the tracked hand's capsules: {s:?}");
            for &(side, index, at, g) in s {
                assert_eq!((side, g), (1, grey));
                let (a, b) = CAPSULES[index];
                let mid = Vec3::new((a + b) as f32 * 0.005, 1.5, 0.0);
                assert!(
                    at.distance(mid) < 1e-5,
                    "capsule {index} at {at}, not {mid}"
                );
            }
        };
        let set = |app: &mut App, grey: bool, margin_m: f32| {
            app.insert_resource(OccludersDrawn(Some(Occluders { grey, margin_m })));
            app.update();
            app.update();
        };
        // The grey set's meshes, by capsule.
        let handles = |world: &mut World| {
            let mut q = world.query::<(&GauntletCapsule, &Mesh3d)>();
            let mut h: Vec<_> = q
                .iter(world)
                .filter(|(c, _)| c.grey)
                .map(|(c, m)| (c.index, m.0.id()))
                .collect();
            h.sort_by_key(|(i, _)| *i);
            h
        };
        set(&mut app, false, 0.005);
        check(&shown(app.world_mut()), false);
        assert_eq!(app.world().resource::<Assets<Mesh>>().len(), 26);
        set(&mut app, true, 0.005);
        check(&shown(app.world_mut()), true);
        assert_eq!(app.world().resource::<Assets<Mesh>>().len(), 52);
        let before = handles(app.world_mut());
        // Back and forth: each capsule keeps its material, and its meshes.
        set(&mut app, false, 0.005);
        check(&shown(app.world_mut()), false);
        set(&mut app, true, 0.005);
        check(&shown(app.world_mut()), true);
        assert_eq!(materials(app.world_mut()), spawned, "no material swapped");
        assert_eq!(
            handles(app.world_mut()),
            before,
            "the same meshes for the same margin"
        );
        set(&mut app, true, 0.01);
        check(&shown(app.world_mut()), true);
        let after = handles(app.world_mut());
        assert_eq!(after.len(), 26);
        assert!(
            after
                .iter()
                .zip(&before)
                .all(|(a, b)| a.0 == b.0 && a.1 != b.1),
            "every mesh made again for the new margin"
        );
        assert_eq!(
            app.world().resource::<Assets<Mesh>>().len(),
            52,
            "the old meshes removed"
        );
    }
}
