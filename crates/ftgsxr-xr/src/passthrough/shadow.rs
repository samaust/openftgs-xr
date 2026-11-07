// SPDX-License-Identifier: Apache-2.0

//! The contact shadows (§1.9 Floor, shadows and anchors; §1.11 `contact_shadows`; Q-342).
//!
//! For each visible model instance, while `Environment.mode` is passthrough, the `Headset`
//! marker exists and `[environment] contact_shadows` is on: an ellipse at y = 0 under the
//! posed box B_m (its stage-space half extents in x and z), alpha 0.3 inside, falling by
//! smoothstep to 0 over the outer 10 cm of its radius. It is drawn in the main opaque pass
//! with depth (§1.4 Depth and occlusion), in its alpha-mask phase after the opaque phase,
//! so the tracked hands' gauntlets hide it whatever the bin order (Q-364), as
//! premultiplied (0, 0, 0, a) under a blend that keeps
//! the target's color and takes the maximum alpha, so overlapping footprints join; the
//! splats composite over it through the resolve's mesh branch (Q-101) and the compositor
//! blends the view target over the room (D-14). Nothing is drawn in VR mode or on the
//! desktop.
//!
//! The material exists where Bevy's `PbrPlugin` does (the player's plugin groups); in an
//! app without it the plugin draws nothing.

use bevy::app::{App, Plugin, PostUpdate};
use bevy::asset::{Asset, Assets, Handle, uuid_handle};
use bevy::camera::visibility::{InheritedVisibility, Visibility};
use bevy::ecs::prelude::*;
use bevy::material::AlphaMode;
use bevy::math::primitives::Plane3d;
use bevy::math::{Vec2, Vec3, Vec4};
use bevy::mesh::{Mesh, Mesh3d, MeshBuilder, MeshVertexBufferLayoutRef, Meshable};
use bevy::pbr::{
    Material, MaterialPipeline, MaterialPipelineKey, MaterialPlugin, MeshMaterial3d, PbrPlugin,
};
use bevy::reflect::TypePath;
use bevy::render::render_resource::{
    AsBindGroup, BlendComponent, BlendFactor, BlendOperation, BlendState, RenderPipelineDescriptor,
    SpecializedMeshPipelineError,
};
use bevy::shader::{Shader, ShaderRef};
use bevy::transform::TransformSystems;
use bevy::transform::components::{GlobalTransform, Transform};
use ftgsxr_app::{Headset, SettingsRes};
use ftgsxr_io::Settings;
use ftgsxr_render::{SplatBounds, SplatModel};

use super::{Environment, EnvironmentMode};

/// The shadow's fragment shader.
const SHADER: Handle<Shader> = uuid_handle!("6f1d3b0e-8c57-4a62-9d0b-5e2c7a41f893");

/// The shader's source, inline: U-35 (§2.7) has every `.wgsl` file referenced by
/// `ftgsxr-render`, so the shader of a material outside it is no file of its own.
// Q-354
const SHADER_WGSL: &str = r"// SPDX-License-Identifier: Apache-2.0

// The contact shadow (§1.9 Floor, shadows and anchors; Q-342): an ellipse on the floor
// centred at (cx, cz) with half extents (hx, hz), alpha 0.3 inside, falling by smoothstep
// to 0 over the outer 10 cm of its radius (over the whole radius when it is shorter),
// written premultiplied (0, 0, 0, a); the pipeline's blend keeps the target's color and
// takes the maximum alpha. Outside the ellipse nothing is written, depth included.

#import bevy_pbr::forward_io::VertexOutput

// (cx, cz, hx, hz) in stage space, metres.
@group(#{MATERIAL_BIND_GROUP}) @binding(0) var<uniform> shape: vec4<f32>;

const OPACITY: f32 = 0.3;
const FEATHER_M: f32 = 0.1;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let p = in.world_position.xz - shape.xy;
    let half = max(shape.zw, vec2<f32>(1e-6));
    // r: the ellipse's normalized radius, 1 on its edge.
    let r = length(p / half);
    if r >= 1.0 {
        discard;
    }
    var a = OPACITY;
    if r > 0.0 {
        // The ellipse's radius along p's direction, and the feather's width there.
        let radius = length(p) / r;
        let f = min(FEATHER_M, radius);
        a = OPACITY * (1.0 - smoothstep(1.0 - f / radius, 1.0, r));
    }
    return vec4<f32>(0.0, 0.0, 0.0, a);
}
";

/// The contact shadows (§1.9, Q-342): added by `XrPlugin`'s tail, or on its own after
/// Bevy's `PbrPlugin`. It reads `Environment`, `Headset` and `SettingsRes` where they
/// exist (without `SettingsRes`, the default settings: on).
// Q-342
pub struct ContactShadowPlugin;

impl Plugin for ContactShadowPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<PbrPlugin>() {
            return;
        }
        app.world_mut()
            .resource_mut::<Assets<Shader>>()
            .insert(
                SHADER.id(),
                Shader::from_wgsl(SHADER_WGSL, "ftgsxr_xr/passthrough/contact_shadow.wgsl"),
            )
            .expect("invariant: the shader's handle is a UUID one, which `insert` accepts");
        app.add_plugins(MaterialPlugin::<ShadowMaterial>::default())
            .add_systems(PostUpdate, sync_shadows.after(TransformSystems::Propagate));
    }
}

/// The shadow's material: the ellipse as (cx, cz, hx, hz) in stage space, metres.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
struct ShadowMaterial {
    #[uniform(0)]
    shape: Vec4,
}

impl Material for ShadowMaterial {
    fn fragment_shader() -> ShaderRef {
        SHADER.into()
    }

    /// Q-364: the alpha-mask phase, which Bevy's main opaque pass draws after the opaque
    /// phase (source-log 373), so every gauntlet's depth is written first; the blend is
    /// still `specialize`'s and the shader ignores the cutoff.
    // Q-364
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Mask(0.5)
    }

    // The prepass and the shadow maps are not the floor's: the shadow is drawn in the
    // main opaque pass only.
    fn enable_prepass() -> bool {
        false
    }

    fn enable_shadows() -> bool {
        false
    }

    /// Q-342: the target keeps its color (the shadow's premultiplied black adds nothing)
    /// and takes the larger alpha, so overlapping footprints join.
    fn specialize(
        _: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _: &MeshVertexBufferLayoutRef,
        _: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        if let Some(fragment) = descriptor.fragment.as_mut() {
            for target in fragment.targets.iter_mut().flatten() {
                target.blend = Some(BlendState {
                    color: BlendComponent {
                        src_factor: BlendFactor::Zero,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Add,
                    },
                    alpha: BlendComponent {
                        src_factor: BlendFactor::One,
                        dst_factor: BlendFactor::One,
                        operation: BlendOperation::Max,
                    },
                });
            }
        }
        Ok(())
    }
}

/// A shadow entity, with the instance it lies under.
#[derive(Component)]
struct ShadowOf(Entity);

/// The ellipse under an instance: B_m's eight corners posed by the instance's global
/// transform (the content group's included), their x and z extents as (cx, cz, hx, hz).
fn footprint(bounds: &ftgsxr_core::Aabb, pose: &GlobalTransform) -> Vec4 {
    let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
    for k in 0..8 {
        let pick = |bit: usize, a: f32, b: f32| if k & bit == 0 { a } else { b };
        let p = pose.transform_point(Vec3::new(
            pick(1, bounds.min.x, bounds.max.x),
            pick(2, bounds.min.y, bounds.max.y),
            pick(4, bounds.min.z, bounds.max.z),
        ));
        lo = lo.min(Vec2::new(p.x, p.z));
        hi = hi.max(Vec2::new(p.x, p.z));
    }
    let (c, h) = (0.5 * (lo + hi), 0.5 * (hi - lo));
    Vec4::new(c.x, c.y, h.x, h.y)
}

/// The quad covering the ellipse: the unit plane (x, z ∈ [−1, 1]) scaled to the half
/// extents at the center, on the floor.
fn quad_transform(shape: Vec4) -> Transform {
    // A degenerate extent would give a singular transform.
    let half = Vec2::new(shape.z, shape.w).max(Vec2::splat(1e-4));
    Transform {
        translation: Vec3::new(shape.x, 0.0, shape.y),
        scale: Vec3::new(half.x, 1.0, half.y),
        ..Transform::IDENTITY
    }
}

/// Spawns, moves and despawns the shadow entities so that one lies under each visible
/// instance while the shadows are on, and none otherwise. Runs after the transforms'
/// propagation, so the instances' poses are this frame's; the shadows are roots and get
/// their global transform here. `Environment.mode` and the instances' inherited
/// visibility may be a frame old.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn sync_shadows(
    env: Option<Res<Environment>>,
    headset: Option<Res<Headset>>,
    settings: Option<Res<SettingsRes>>,
    // With `SplatModel`, so the shadows' query (`Without<SplatModel>`) stays disjoint.
    instances: Query<
        (
            Entity,
            &SplatBounds,
            &GlobalTransform,
            Option<&InheritedVisibility>,
        ),
        With<SplatModel>,
    >,
    mut shadows: Query<
        (
            Entity,
            &ShadowOf,
            &mut Transform,
            &mut GlobalTransform,
            &MeshMaterial3d<ShadowMaterial>,
        ),
        Without<SplatModel>,
    >,
    mut materials: ResMut<Assets<ShadowMaterial>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut quad: Local<Option<Handle<Mesh>>>,
    mut commands: Commands,
) {
    let on = env.is_some_and(|e| e.mode == EnvironmentMode::Passthrough)
        && headset.is_some()
        && settings.as_deref().map_or_else(
            || Settings::default().environment.contact_shadows,
            |s| s.0.environment.contact_shadows,
        );
    // §1.9 under B_m's footprint, trimmed while a trim is on (E17, §1.5 Poses).
    let mut wanted: Vec<(Entity, Vec4)> = if on {
        instances
            .iter()
            .filter(|(.., vis)| vis.is_none_or(|v| v.get()))
            .map(|(e, bounds, pose, _)| (e, footprint(&bounds.aabb, pose)))
            .collect()
    } else {
        Vec::new()
    };
    for (shadow, of, mut transform, mut global, material) in &mut shadows {
        let Some(at) = wanted.iter().position(|(e, _)| *e == of.0) else {
            commands.entity(shadow).despawn();
            continue;
        };
        let (_, shape) = wanted.swap_remove(at);
        if materials.get(&material.0).is_some_and(|m| m.shape != shape)
            && let Some(mut m) = materials.get_mut(&material.0)
        {
            m.shape = shape;
        }
        let t = quad_transform(shape);
        transform.set_if_neq(t);
        global.set_if_neq(GlobalTransform::from(t));
    }
    if wanted.is_empty() {
        return;
    }
    let quad = quad
        .get_or_insert_with(|| meshes.add(Plane3d::new(Vec3::Y, Vec2::ONE).mesh().build()))
        .clone();
    for (instance, shape) in wanted {
        let t = quad_transform(shape);
        commands.spawn((
            ShadowOf(instance),
            Mesh3d(quad.clone()),
            MeshMaterial3d(materials.add(ShadowMaterial { shape })),
            t,
            GlobalTransform::from(t),
            Visibility::Visible,
        ));
    }
}

#[cfg(test)]
mod tests {
    use ftgsxr_core::Aabb;

    use super::*;

    /// The footprint of a box posed by a yaw of 90° and a scale of 2: x and z swap, the
    /// extents double, the translation moves the center.
    #[test]
    fn footprint_is_the_posed_boxs_extents_in_x_and_z() {
        let b = Aabb {
            min: Vec3::new(-0.5, 0.0, -0.25),
            max: Vec3::new(0.5, 1.0, 0.25),
        };
        let pose = GlobalTransform::from(
            Transform::from_xyz(1.0, 0.3, -2.0)
                .with_rotation(bevy::math::Quat::from_rotation_y(
                    std::f32::consts::FRAC_PI_2,
                ))
                .with_scale(Vec3::splat(2.0)),
        );
        let f = footprint(&b, &pose);
        assert!(
            (f - Vec4::new(1.0, -2.0, 0.5, 1.0)).abs().max_element() < 1e-5,
            "{f}"
        );
    }
}
