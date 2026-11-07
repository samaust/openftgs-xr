// SPDX-License-Identifier: Apache-2.0

//! "Occlude with tracked hands" (§1.9 Depth occlusion, §1.8 Hands, D-16, #608):
//! in passthrough mode with `[environment] hand_occluders` on, each tracked hand is drawn
//! as a gauntlet that writes depth only and no color, so the splats, which test against
//! the opaque depth (§1.4 Depth and occlusion), are hidden where the hand is at its
//! tracked pose (the locate's predicted display time, `hands::locate_hands`) instead of
//! at the depth image's capture time.
//!
//! The gauntlet is one capsule per bone of §1.8's VR-mode capsules (`meshes::BONES`, the
//! wrist to each fingertip) and two palm capsules, from the wrist to the index and to the
//! little finger's metacarpophalangeal joints, each with its joints' mean radius (as
//! Q-288's capsules) plus the margin m = 1.5 cm (D-16). A capsule whose two joints are
//! not both located is hidden, so an untracked hand draws nothing. The capsules are
//! children of the tracking root, drawn in Bevy's opaque phase by a material whose color
//! writes are off; the contact shadow is drawn in the alpha-mask phase after it (Q-364),
//! so a gauntlet hides it too.
//!
//! The material exists where Bevy's `PbrPlugin` does (the player's plugin groups); in an
//! app without it nothing is drawn.

use bevy::app::App;
use bevy::asset::{Asset, Assets, Handle, uuid_handle};
use bevy::camera::visibility::NoFrustumCulling;
use bevy::math::Vec3;
use bevy::math::primitives::Capsule3d;
use bevy::mesh::{Mesh, Mesh3d, MeshVertexBufferLayoutRef};
use bevy::pbr::{
    Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin, MeshMaterial3d, PbrPlugin,
};
use bevy::prelude::{
    ChildOf, Commands, Component, DetectChangesMut, Entity, GlobalTransform, Local, Query, Res,
    ResMut, Resource, Transform, Visibility, With,
};
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, ColorWrites, RenderPipelineDescriptor, SpecializedMeshPipelineError,
};
use bevy::shader::{Shader, ShaderRef};
use bevy_mod_xr::session::XrTrackingRoot;
use openxr::sys;

use super::hands::{JOINTS, LocatedHands, WRIST, located, transform_of};
use super::meshes::{BONES, bone_transform};

/// Whether the gauntlets are drawn: in passthrough mode with `[environment]
/// hand_occluders` on (§1.9). `passthrough`, after `tracking` in §2.1's order, sets it
/// each frame before the capsules are placed; false until then.
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GauntletsDrawn(pub(crate) bool);

/// The margin added to each capsule's radius, metres (D-16).
pub(crate) const MARGIN_M: f32 = 0.015;

/// XR\_HAND\_JOINT\_INDEX\_PROXIMAL\_EXT and XR\_HAND\_JOINT\_LITTLE\_PROXIMAL\_EXT: a
/// finger's joints are named after the bone that starts at them (source-log 374), so
/// each proximal joint is the finger's metacarpophalangeal joint; their values are the
/// default set's (source-log 276).
const INDEX_MCP: usize = 7;
const LITTLE_MCP: usize = 22;

/// The gauntlet's capsules as joint pairs: §1.8's 24 bones, then the two palm capsules
/// (§1.9).
pub(crate) const CAPSULES: [(usize, usize); BONES.len() + 2] = {
    let mut c = [(WRIST, INDEX_MCP); BONES.len() + 2];
    let mut i = 0;
    while i < BONES.len() {
        c[i] = BONES[i];
        i += 1;
    }
    c[BONES.len() + 1] = (WRIST, LITTLE_MCP);
    c
};

/// One capsule of a gauntlet: the centres of its two end spheres and its radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Segment {
    pub(crate) a: Vec3,
    pub(crate) b: Vec3,
    pub(crate) radius: f32,
}

/// The gauntlet of one hand's joints, in `CAPSULES`' order: each capsule between its two
/// joints' positions, its radius their mean radius plus `MARGIN_M`; `None` where either
/// joint is not located.
pub(crate) fn gauntlet(joints: &[sys::HandJointLocationEXT; JOINTS]) -> Vec<Option<Segment>> {
    CAPSULES
        .iter()
        .map(|&(a, b)| {
            let (ja, jb) = (&joints[a], &joints[b]);
            (located(ja) && located(jb)).then(|| Segment {
                a: transform_of(&ja.pose).translation,
                b: transform_of(&jb.pose).translation,
                // Q-288's radius, plus D-16's margin.
                radius: (ja.radius + jb.radius) * 0.5 + MARGIN_M,
            })
        })
        .collect()
}

/// The gauntlet's fragment shader.
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

/// The gauntlet's material: opaque, in the opaque phase, with its color writes off.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug, Default)]
pub(crate) struct GauntletMaterial {}

impl Material for GauntletMaterial {
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

/// Registers the material where `PbrPlugin` is; returns whether it did.
pub(crate) fn add_material(app: &mut App) -> bool {
    if !app.is_plugin_added::<PbrPlugin>() {
        return false;
    }
    app.world_mut()
        .resource_mut::<Assets<Shader>>()
        .insert(
            SHADER.id(),
            Shader::from_wgsl(SHADER_WGSL, "ftgsxr_xr/tracking/gauntlet.wgsl"),
        )
        .expect("invariant: the shader's handle is a UUID one, which `insert` accepts");
    app.add_plugins(MaterialPlugin::<GauntletMaterial>::default());
    true
}

/// One capsule of a gauntlet: its side, its index in `CAPSULES`, and its mesh's radius
/// and length once made (at the first frame both joints are located, as Q-288's).
#[derive(Component, Clone, Copy)]
pub(crate) struct GauntletCapsule {
    side: usize,
    index: usize,
    meshed: bool,
}

/// Spawns the capsules once per tracking root, then each frame places each one between
/// its joints and shows it while the gauntlets are drawn and both joints are located.
/// The capsules' global transforms are set here from the root's, which the propagation
/// has already updated, so this frame's pose is drawn this frame; frustum culling is off
/// since it may run before them.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn draw_gauntlets(
    drawn: Option<Res<GauntletsDrawn>>,
    meshes: Option<ResMut<Assets<Mesh>>>,
    materials: Option<ResMut<Assets<GauntletMaterial>>>,
    root: Query<(Entity, &GlobalTransform), With<XrTrackingRoot>>,
    hands: Res<LocatedHands>,
    mut capsules: Query<(
        Entity,
        &mut GauntletCapsule,
        &mut Transform,
        &mut GlobalTransform,
        &mut Visibility,
    )>,
    mut material: Local<Option<Handle<GauntletMaterial>>>,
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
            .get_or_insert_with(|| materials.add(GauntletMaterial {}))
            .clone();
        for side in 0..2 {
            for index in 0..CAPSULES.len() {
                commands.spawn((
                    GauntletCapsule {
                        side,
                        index,
                        meshed: false,
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
    let on = drawn.is_some_and(|d| d.0);
    let segments: [Option<Vec<Option<Segment>>>; 2] = [0, 1].map(|side| {
        hands.raw[side]
            .filter(|_| on)
            .map(|raw| gauntlet(&raw.joints))
    });
    for (e, mut capsule, mut t, mut global, mut vis) in &mut capsules {
        let placed = segments[capsule.side]
            .as_ref()
            .and_then(|s| s[capsule.index])
            .and_then(|s| bone_transform(s.a, s.b).map(|(bt, len)| (bt, len, s.radius)));
        let Some((bt, len, radius)) = placed else {
            vis.set_if_neq(Visibility::Hidden);
            continue;
        };
        if !capsule.meshed {
            capsule.meshed = true;
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
    /// radius plus 1.5 cm.
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
        let j = joints();
        let g = gauntlet(&j);
        assert_eq!(g.len(), 26);
        for (s, &(a, b)) in g.iter().zip(&CAPSULES) {
            let s = s.expect("every joint is located");
            assert!((s.a - Vec3::new(a as f32 * 0.01, 1.0, 0.0)).length() < 1e-6);
            assert!((s.b - Vec3::new(b as f32 * 0.01, 1.0, 0.0)).length() < 1e-6);
            let want = (j[a].radius + j[b].radius) / 2.0 + 0.015;
            assert!((s.radius - want).abs() < 1e-7, "{a}-{b}: {}", s.radius);
        }
    }

    /// An unlocated joint takes out the capsules that end at it, and an untracked hand
    /// (no joint located) draws nothing.
    #[test]
    fn gauntlet_leaves_out_capsules_of_unlocated_joints() {
        let mut j = joints();
        j[7].location_flags = sys::SpaceLocationFlags::POSITION_VALID;
        let g = gauntlet(&j);
        for (s, &(a, b)) in g.iter().zip(&CAPSULES) {
            assert_eq!(s.is_none(), a == 7 || b == 7, "{a}-{b}");
        }
        let untracked = [sys::HandJointLocationEXT::default(); JOINTS];
        assert!(gauntlet(&untracked).iter().all(Option::is_none));
    }
}
