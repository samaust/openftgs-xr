// SPDX-License-Identifier: Apache-2.0

//! PRD #275's added adapter tests (#276):
//!
//! - the depth-gated resolve (source of truth: Q-101's decision, §1.4 Depth and
//!   occlusion): a `SplatCamera` with `OutputSpace::LinearOver { background: Some(..) }`
//!   over an opaque quad at y = 0 keeps the quad's pixels where no splat covers them and
//!   shows a splat in front of it over the quad;
//! - the governor's SH cap without a reload (source of truth: E16, §1.2 load option 1):
//!   `SplatTierState.sh_cap` at 1 on a degree-3 probe renders as the probe capped at
//!   degree 1 renders;
//! - a replay's SH limit (source of truth: Q-233, Q-235, E16): `SplatShLimit(Some(1))`
//!   under a governor's cap of 3 renders a degree-3 probe as the probe capped at degree 1
//!   renders, and `SplatShLimit(None)` renders it at its loaded degree again (#388).

use std::sync::{Arc, Mutex};

use bevy::MinimalPlugins;
use bevy::app::{App, PluginsState};
use bevy::asset::{AssetPlugin, Assets, RenderAssetUsages};
use bevy::camera::{
    Camera, Camera3d, CameraPlugin, PerspectiveProjection, Projection, RenderTarget,
};
use bevy::core_pipeline::core_3d::{CORE_3D_DEPTH_FORMAT, main_opaque_pass_3d};
use bevy::core_pipeline::{Core3d, Core3dSystems, CorePipelinePlugin};
use bevy::ecs::prelude::*;
use bevy::image::{Image, ImagePlugin};
use bevy::math::{Mat3, Quat, Vec3, Vec4};
use bevy::mesh::MeshPlugin;
use bevy::render::error_handler::{RenderErrorHandler, RenderErrorPolicy};
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_resource::{StoreOp, TextureFormat, TextureUsages};
use bevy::render::renderer::{RenderContext, RenderDevice, ViewQuery};
use bevy::render::view::{ExtractedView, ViewDepthTexture, ViewTarget};
use bevy::render::{RenderApp, RenderPlugin};
use bevy::transform::TransformPlugin;
use bevy::transform::components::Transform;
use bevy::window::{ExitCondition, WindowPlugin};
use ftgsxr_core::{Camera as FtgsCamera, DMat3, Pose, RenderImage};
use ftgsxr_io::ModelBytes;
use ftgsxr_render::{
    HeadlessRenderer, OutputSpace, QualityTier, SortMode, SplatCamera, SplatEvent, SplatLoad,
    SplatPlugin, SplatShLimit, SplatTierState, SplatTime,
};
use ftgsxr_testkit::{fixtures_dir, layer_messages, metrics, probe};
use wgpu::util::DeviceExt;

/// Frames the app may take to load, upload and draw before the test gives up.
const MAX_FRAMES: usize = 600;
/// Frames rendered after the upload before a read back is taken.
const SETTLE_FRAMES: usize = 4;
/// The quad's color in file space (sRGB-encoded), and the camera's background.
const QUAD: [f32; 3] = [0.3, 0.6, 0.3];
const BACKGROUND: [f32; 3] = [0.2, 0.2, 0.6];
/// The depth-gated resolve's tolerance per channel on the 8-bit target: the rounding of
/// the reference's and the resolve's encodes.
const TOLERANCE: f32 = 2.5 / 255.0;
/// E15's tier.
const MIN_PSNR: f64 = 45.0;
const MAX_SHARE: f64 = 0.005;

#[derive(Resource)]
struct RenderErrored;

#[derive(Resource, Clone, Default)]
struct Seen {
    events: Arc<Mutex<Vec<SplatEvent>>>,
    readback: Arc<Mutex<Option<Vec<u8>>>>,
}

/// The model of a scene: its file, its time and its instance transform.
struct SceneModel {
    file: &'static str,
    tau: f64,
    transform: Transform,
}

/// #291: a scene dropped while a pipeline compiles would let the process exit with a
/// worker in the driver; not while panicking, so a failed test reports.
impl Drop for Scene {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            settle_pipelines(&mut self.app);
        }
    }
}

/// #291 (§2.4 Threads): on bevy's `multi_threaded` pools (unified in by the workspace's
/// build) `PipelineCache` compiles on the async compute pool, so a frame may draw before
/// the splat or tonemapping pipelines exist, and a test returning while one compiles lets
/// the process exit with a worker inside the Vulkan driver. Runs frames until no render
/// pipeline is queued or compiling, at most 10 s of wall time; whether none is.
fn settle_pipelines(app: &mut App) -> bool {
    use bevy::render::render_resource::{CachedPipelineState, PipelineCache};
    let busy = |app: &App| {
        app.sub_app(RenderApp)
            .world()
            .resource::<PipelineCache>()
            .pipelines()
            .filter(|p| {
                matches!(
                    p.state,
                    CachedPipelineState::Queued | CachedPipelineState::Creating(_)
                )
            })
            .count()
    };
    let start = std::time::Instant::now();
    while busy(app) > 0 {
        if start.elapsed() > std::time::Duration::from_secs(10) {
            return false;
        }
        app.update();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    true
}

struct Scene {
    app: App,
    seen: Seen,
}

impl Scene {
    /// One splat camera at `camera` with `projection`, writing `output` into a `w` × `h`
    /// image target of `format`; the model (if any) loaded and uploaded; the opaque quad
    /// at y = 0 drawn before the opaque pass when `quad`.
    fn settled(
        output: OutputSpace,
        format: TextureFormat,
        (w, h): (u32, u32),
        camera: Transform,
        projection: Projection,
        model: Option<SceneModel>,
        quad: bool,
    ) -> Scene {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            TransformPlugin,
            AssetPlugin::default(),
            WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                close_when_requested: false,
                ..WindowPlugin::default()
            },
            RenderPlugin::default(),
            ImagePlugin::default(),
            MeshPlugin,
            CameraPlugin,
            CorePipelinePlugin,
            SplatPlugin {
                tier: QualityTier::conformance(),
            },
        ))
        .insert_resource(RenderErrorHandler(|error, main_world, _| {
            eprintln!("render error: {:?}: {}", error.ty, error.description);
            main_world.insert_resource(RenderErrored);
            RenderErrorPolicy::StopRendering
        }));
        while app.plugins_state() == PluginsState::Adding {
            bevy::tasks::tick_global_task_pools_on_main_thread();
        }
        app.finish();
        app.cleanup();
        if quad {
            app.sub_app_mut(RenderApp).add_systems(
                Core3d,
                quad_node
                    .in_set(Core3dSystems::MainPass)
                    .before(main_opaque_pass_3d),
            );
        }

        let seen = Seen::default();
        app.insert_resource(seen.clone());
        app.add_observer(|event: On<SplatEvent>, seen: Res<Seen>| {
            seen.events.lock().unwrap().push(event.event().clone());
        });

        let mut image = Image::new_target_texture(w, h, format, None);
        image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
        image.asset_usage = RenderAssetUsages::RENDER_WORLD;
        let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);
        app.world_mut().spawn((
            Camera3d::default(),
            Camera::default(),
            RenderTarget::Image(target.clone().into()),
            projection,
            camera,
            SplatCamera {
                sort: SortMode::PerView,
                output,
            },
        ));
        let has_model = model.is_some();
        if let Some(m) = model {
            let bytes = std::fs::read(fixtures_dir().join(m.file)).expect("the model's file");
            app.world_mut().spawn((
                SplatLoad {
                    bytes: ModelBytes::Owned(bytes),
                    kind_hint: None,
                    reduce_to: None,
                },
                SplatTime {
                    tau: m.tau,
                    extrapolating: false,
                },
                m.transform,
            ));
        }
        app.world_mut().spawn(Readback::texture(target)).observe(
            |event: On<ReadbackComplete>, seen: Res<Seen>| {
                *seen.readback.lock().unwrap() = Some(event.data.clone());
            },
        );

        let mut scene = Scene { app, seen };
        let mut uploaded_at = (!has_model).then_some(0);
        for frame in 0..MAX_FRAMES {
            scene.frames(1);
            let events = scene.seen.events.lock().unwrap().clone();
            if let Some(SplatEvent::Failed { error, .. }) = events
                .iter()
                .find(|e| matches!(e, SplatEvent::Failed { .. }))
            {
                panic!("the load failed: {error}");
            }
            if uploaded_at.is_none()
                && events
                    .iter()
                    .any(|e| matches!(e, SplatEvent::Uploaded { .. }))
            {
                uploaded_at = Some(frame);
            }
            if uploaded_at.is_some_and(|f| frame >= f + SETTLE_FRAMES) {
                break;
            }
        }
        assert!(uploaded_at.is_some(), "uploaded within {MAX_FRAMES} frames");
        assert!(
            settle_pipelines(&mut scene.app),
            "the render pipelines compiled within 10 s"
        );
        scene
    }

    fn frames(&mut self, n: usize) {
        for _ in 0..n {
            self.app.update();
            assert!(
                !self.app.world().contains_resource::<RenderErrored>(),
                "Bevy reported a render error"
            );
        }
    }

    /// A read back of the target from a frame rendered after this call.
    fn readback(&mut self) -> Vec<u8> {
        *self.seen.readback.lock().unwrap() = None;
        self.frames(2 + SETTLE_FRAMES);
        self.seen
            .readback
            .lock()
            .unwrap()
            .clone()
            .expect("a read back of the image target")
    }

    fn uploads(&self) -> usize {
        self.seen
            .events
            .lock()
            .unwrap()
            .iter()
            .filter(|e| matches!(e, SplatEvent::Uploaded { .. }))
            .count()
    }
}

/// The opaque quad: |x|, |z| ≤ 100 at y = 0, written in the view target (`QUAD`, decoded
/// so the sRGB target stores it) and the view's depth (reverse-Z, `GreaterEqual`, write
/// on), before the opaque pass, so the splats and the resolve meet it as they would an
/// opaque mesh.
#[derive(Resource)]
struct QuadPipeline(wgpu::RenderPipeline);

const QUAD_WGSL: &str = r"
@vertex
fn vs(@location(0) clip: vec4<f32>) -> @builtin(position) vec4<f32> { return clip; }
@group(0) @binding(0) var<uniform> color: vec4<f32>;
@fragment
fn fs() -> @location(0) vec4<f32> { return color; }
";

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn quad_node(
    view: ViewQuery<(&ExtractedView, &ViewTarget, &ViewDepthTexture)>,
    device: Res<RenderDevice>,
    pipeline: Option<Res<QuadPipeline>>,
    mut commands: Commands,
    mut ctx: RenderContext,
) {
    let device = device.wgpu_device();
    let Some(pipeline) = pipeline else {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test quad"),
            source: wgpu::ShaderSource::Wgsl(QUAD_WGSL.into()),
        });
        let p = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("test quad"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 16,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x4],
                }],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(TextureFormat::Rgba8UnormSrgb.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        commands.insert_resource(QuadPipeline(p));
        // Drawn from the next frame on; the scene settles for several.
        return;
    };
    let (view, target, depth) = view.into_inner();
    let clip_from_world = view.clip_from_view * view.world_from_view.to_matrix().inverse();
    let corner = |x: f32, z: f32| clip_from_world * Vec4::new(x, 0.0, z, 1.0);
    let s = 100.0;
    let verts: Vec<Vec4> = [(-s, -s), (s, -s), (s, s), (-s, -s), (s, s), (-s, s)]
        .into_iter()
        .map(|(x, z)| corner(x, z))
        .collect();
    let vertex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test quad"),
        contents: bytemuck_vec4(&verts).as_slice(),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let color = [
        srgb_to_linear(QUAD[0]),
        srgb_to_linear(QUAD[1]),
        srgb_to_linear(QUAD[2]),
        1.0,
    ];
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test quad"),
        contents: bytemuck_vec4(&[Vec4::from_array(color)]).as_slice(),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test quad"),
        layout: &pipeline.0.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }],
    });
    let color_attachment = [Some(target.get_color_attachment())];
    let mut pass = ctx.begin_tracked_render_pass(wgpu::RenderPassDescriptor {
        label: Some("test quad"),
        color_attachments: &color_attachment,
        depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    let pass = pass.wgpu_pass();
    pass.set_pipeline(&pipeline.0);
    pass.set_bind_group(0, &group, &[]);
    pass.set_vertex_buffer(0, vertex.slice(..));
    pass.draw(0..6, 0..1);
}

fn bytemuck_vec4(v: &[Vec4]) -> Vec<u8> {
    v.iter()
        .flat_map(|q| q.to_array())
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// IEEE binary16 to f32.
fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((bits >> 10) & 0x1f);
    let frac = f32::from(bits & 0x3ff);
    sign * match exp {
        0 => frac * 2f32.powi(-24),
        31 if frac == 0.0 => f32::INFINITY,
        31 => f32::NAN,
        e => (1.0 + frac / 1024.0) * 2f32.powi(e - 15),
    }
}

/// Rows of `w` texels of `texel` bytes, padded to the read back's row.
fn texels(bytes: &[u8], w: u32, h: u32, texel: usize) -> Vec<&[u8]> {
    let row = bytes.len() / h as usize;
    (0..h as usize)
        .flat_map(|y| (0..w as usize).map(move |x| (y, x)))
        .map(|(y, x)| &bytes[y * row + texel * x..][..texel])
        .collect()
}

/// A 16-bit float layer's (C, A) per pixel.
fn layer_of(bytes: &[u8], w: u32, h: u32) -> Vec<[f32; 4]> {
    texels(bytes, w, h, 8)
        .into_iter()
        .map(|t| std::array::from_fn(|i| half(u16::from_le_bytes([t[2 * i], t[2 * i + 1]]))))
        .collect()
}

/// An 8-bit target's stored values per pixel, in [0, 1].
fn target_of(bytes: &[u8], w: u32, h: u32) -> Vec<[f32; 4]> {
    texels(bytes, w, h, 4)
        .into_iter()
        .map(|t| std::array::from_fn(|i| f32::from(t[i]) / 255.0))
        .collect()
}

const SIZE: (u32, u32) = (64, 48);

/// A camera at (0, 1, 3.5) looking at (0, 0.5, 0), with y up: the quad at y = 0 fills the
/// image below the horizon, the background above it, and a model at (0, 0.5, 0) lies
/// in front of the quad's pixels behind it (every ray through it meets y = 0 beyond it).
fn floor_camera() -> (Transform, Projection) {
    let camera = Transform::from_xyz(0.0, 1.0, 3.5).looking_at(Vec3::new(0.0, 0.5, 0.0), Vec3::Y);
    let projection = Projection::Perspective(PerspectiveProjection {
        fov: 2.0 * (24.0f32 / 60.0).atan(),
        aspect_ratio: 64.0 / 48.0,
        near: 0.01,
        ..PerspectiveProjection::default()
    });
    (camera, projection)
}

/// P1 at its first item's time, moved to (0, 0.5, 0) and turned so that its camera's view
/// axis (+y in its world) is the floor camera's (−z).
fn p1_above_the_quad() -> SceneModel {
    let (_, request) = probe("P1");
    SceneModel {
        file: "probes/P1.safetensors",
        tau: request.items[0].timestamp,
        transform: Transform::from_xyz(0.0, 0.5, 0.0)
            .with_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    }
}

#[test]
#[ignore = "gpu"]
fn gpu_bevy_adapter_depth_gated_resolve_keeps_the_quad_and_shows_the_splat_over_it() {
    let _layer = layer_messages(); // R-16, Q-105
    let (w, h) = SIZE;
    let (camera, projection) = floor_camera();
    let over = OutputSpace::LinearOver {
        background: Some(BACKGROUND),
    };
    // The layer alone: the same camera and model without the quad, file-encoded.
    let layer_bytes = Scene::settled(
        OutputSpace::FileEncoded,
        TextureFormat::Rgba16Float,
        SIZE,
        camera,
        projection.clone(),
        Some(p1_above_the_quad()),
        false,
    )
    .readback();
    let layer = layer_of(&layer_bytes, w, h);
    // The quad alone: which pixels it covers, and their stored value.
    let quad_bytes = Scene::settled(
        over,
        TextureFormat::Rgba8UnormSrgb,
        SIZE,
        camera,
        projection.clone(),
        None,
        true,
    )
    .readback();
    let quad_only = target_of(&quad_bytes, w, h);
    let near = |a: f32, b: f32| (a - b).abs() <= TOLERANCE;
    let is_quad = |p: &[f32; 4]| (0..3).all(|c| near(p[c], QUAD[c]));
    let is_bg = |p: &[f32; 4]| (0..3).all(|c| near(p[c], BACKGROUND[c]));
    for (i, p) in quad_only.iter().enumerate() {
        assert!(
            is_quad(p) || is_bg(p),
            "pixel {i} is quad or background: {p:?}"
        );
    }
    // Both together.
    let both = target_of(
        &Scene::settled(
            over,
            TextureFormat::Rgba8UnormSrgb,
            SIZE,
            camera,
            projection,
            Some(p1_above_the_quad()),
            true,
        )
        .readback(),
        w,
        h,
    );
    let (mut over_quad, mut over_bg, mut failures) = (0, 0, Vec::new());
    for i in 0..both.len() {
        let [cr, cg, cb, a] = layer[i];
        let c = [cr, cg, cb];
        let expected: [f32; 3] = if is_quad(&quad_only[i]) {
            // Q-101's mesh branch: lin(C/A)·A over the quad's linear value, encoded.
            if a > 0.1 {
                over_quad += 1;
            }
            std::array::from_fn(|k| {
                let splat = if a > 0.0 {
                    srgb_to_linear(c[k] / a) * a
                } else {
                    0.0
                };
                linear_to_srgb(splat + (1.0 - a) * srgb_to_linear(quad_only[i][k]))
            })
        } else {
            // Q-94's form: E7's C + (1 − A)·c_bg in file space.
            if a > 0.1 {
                over_bg += 1;
            }
            std::array::from_fn(|k| c[k] + (1.0 - a) * BACKGROUND[k])
        };
        let got = both[i];
        if !(0..3).all(|k| near(got[k], expected[k])) || got[3] != 1.0 {
            failures.push(format!("pixel {i}: {got:?} against {expected:?} (A = {a})"));
        }
    }
    eprintln!("splat pixels over the quad {over_quad}, over the background {over_bg}");
    assert!(over_quad > 0, "the splat covers quad pixels");
    assert!(
        quad_only.iter().filter(|p| is_quad(p)).count() > 0
            && quad_only.iter().filter(|p| is_bg(p)).count() > 0,
        "both regions exist"
    );
    assert!(
        failures.is_empty(),
        "{} pixels off:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Q-164: at render scale 0.5 the layer is drawn at its size against the depth copied
/// at that size, and the depth-gated resolve still keeps the quad where no splat covers
/// it and shows the splat over it, opaque everywhere (Q-101, §1.12, §1.7 Window).
#[test]
#[ignore = "gpu"]
fn gpu_bevy_adapter_below_render_scale_1_keeps_the_depth_gated_resolve() {
    let _layer = layer_messages(); // R-16, Q-105
    let (w, h) = SIZE;
    let (camera, projection) = floor_camera();
    let over = OutputSpace::LinearOver {
        background: Some(BACKGROUND),
    };
    // The layer's alpha at scale 1, for where the splat covers pixels.
    let alpha: Vec<f32> = layer_of(
        &Scene::settled(
            OutputSpace::FileEncoded,
            TextureFormat::Rgba16Float,
            SIZE,
            camera,
            projection.clone(),
            Some(p1_above_the_quad()),
            false,
        )
        .readback(),
        w,
        h,
    )
    .iter()
    .map(|p| p[3])
    .collect();
    let quad_only = target_of(
        &Scene::settled(
            over,
            TextureFormat::Rgba8UnormSrgb,
            SIZE,
            camera,
            projection.clone(),
            None,
            true,
        )
        .readback(),
        w,
        h,
    );
    let near = |a: f32, b: f32| (a - b).abs() <= TOLERANCE;
    let is_quad = |p: &[f32; 4]| (0..3).all(|c| near(p[c], QUAD[c]));
    let mut scene = Scene::settled(
        over,
        TextureFormat::Rgba8UnormSrgb,
        SIZE,
        camera,
        projection,
        Some(p1_above_the_quad()),
        true,
    );
    let full = target_of(&scene.readback(), w, h);
    scene
        .app
        .world_mut()
        .resource_mut::<SplatTierState>()
        .render_scale = 0.5;
    let scaled = target_of(&scene.readback(), w, h);

    // The splat's reach at scale 1, widened by 2 pixels for the half-size layer's filter.
    let (wi, hi) = (w as i64, h as i64);
    let covered = |x: i64, y: i64| {
        (-2..=2).any(|dy| {
            (-2..=2).any(|dx| {
                let (u, v) = (x + dx, y + dy);
                (0..wi).contains(&u)
                    && (0..hi).contains(&v)
                    && alpha[(v * wi + u) as usize] >= 1.0 / 255.0
            })
        })
    };
    let (mut kept, mut shown, mut failures) = (0, 0, Vec::new());
    let mut max_change = 0.0f32;
    for i in 0..scaled.len() {
        let (x, y) = ((i as i64) % wi, (i as i64) / wi);
        let p = scaled[i];
        if p[3] != 1.0 {
            failures.push(format!("pixel {i} not opaque: {p:?}"));
        }
        for k in 0..3 {
            max_change = max_change.max((p[k] - full[i][k]).abs());
        }
        if !is_quad(&quad_only[i]) {
            continue;
        }
        if !covered(x, y) {
            kept += 1;
            if !is_quad(&p) {
                failures.push(format!("quad pixel {i} not kept: {p:?}"));
            }
        } else if alpha[i] > 0.3 {
            shown += 1;
            if is_quad(&p) || !(0..3).all(|k| (p[k] - full[i][k]).abs() <= 0.1) {
                failures.push(format!(
                    "splat pixel {i} over the quad: {p:?} against {:?} at scale 1",
                    full[i]
                ));
            }
        }
    }
    eprintln!(
        "quad pixels kept {kept}, splat pixels over the quad {shown}, max change {max_change}"
    );
    assert!(kept > 0 && shown > 0, "both regions exist");
    assert!(
        max_change > TOLERANCE,
        "the half-size layer changes the image, so the scale applies"
    );
    assert!(
        failures.is_empty(),
        "{} pixels off:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Q-164: at render scale 0.5 a splat behind the opaque quad (P1 at y = −0.5, under the
/// floor) is hidden by the depth copied at the layer's size: the image is the quad-only
/// image wherever the quad is, while without the quad the splat covers pixels. A copy
/// that wrote the clear value (or no depth) would let the splat over the quad.
#[test]
#[ignore = "gpu"]
fn gpu_bevy_adapter_below_render_scale_1_hides_a_splat_behind_the_quad() {
    let _layer = layer_messages(); // R-16, Q-105
    let (w, h) = SIZE;
    let (camera, projection) = floor_camera();
    let over = OutputSpace::LinearOver {
        background: Some(BACKGROUND),
    };
    let below = || {
        let mut m = p1_above_the_quad();
        m.transform.translation = Vec3::new(0.0, -0.5, 0.0);
        m
    };
    let scaled = |model: Option<SceneModel>, quad: bool| {
        let mut scene = Scene::settled(
            over,
            TextureFormat::Rgba8UnormSrgb,
            SIZE,
            camera,
            projection.clone(),
            model,
            quad,
        );
        scene
            .app
            .world_mut()
            .resource_mut::<SplatTierState>()
            .render_scale = 0.5;
        target_of(&scene.readback(), w, h)
    };
    let near = |a: f32, b: f32| (a - b).abs() <= TOLERANCE;
    // Not vacuous: without the quad the splat shows at scale 0.5.
    let bare = scaled(Some(below()), false);
    let shown = bare
        .iter()
        .filter(|p| (0..3).any(|k| !near(p[k], BACKGROUND[k])))
        .count();
    assert!(shown > 0, "the splat under the floor is in view");
    let quad_only = scaled(None, true);
    let both = scaled(Some(below()), true);
    let is_quad = |p: &[f32; 4]| (0..3).all(|c| near(p[c], QUAD[c]));
    let mut failures = Vec::new();
    for i in 0..both.len() {
        if is_quad(&quad_only[i]) && !is_quad(&both[i]) {
            failures.push(format!("pixel {i}: {:?} over the quad", both[i]));
        }
    }
    eprintln!("splat pixels without the quad {shown}");
    assert!(
        failures.is_empty(),
        "{} pixels off:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The probes' camera (openftgs's) as a Bevy transform and projection.
fn probe_camera(camera: &FtgsCamera) -> (Transform, Projection) {
    let r = DMat3::from_mat4(camera.world_to_camera);
    let t = camera.world_to_camera.w_axis.truncate();
    let center = -(r.transpose() * t);
    let flip = DMat3::from_diagonal(ftgsxr_core::DVec3::new(1.0, -1.0, -1.0));
    let m = (r.transpose() * flip).as_mat3();
    let transform = Transform {
        translation: center.as_vec3(),
        rotation: Quat::from_mat3(&Mat3::from_cols(m.x_axis, m.y_axis, m.z_axis)),
        scale: Vec3::ONE,
    };
    let (w, h) = (f64::from(camera.width), f64::from(camera.height));
    let projection = Projection::Perspective(PerspectiveProjection {
        fov: (2.0 * (h / 2.0 / camera.k.y_axis.y).atan()) as f32,
        aspect_ratio: (w / h) as f32,
        near: 0.01,
        ..PerspectiveProjection::default()
    });
    (transform, projection)
}

/// The read back premultiplied file-space layer composited over black (E7).
fn image_of(bytes: &[u8], w: u32, h: u32) -> RenderImage {
    let layer = layer_of(bytes, w, h);
    RenderImage {
        width: w,
        height: h,
        rgb: layer.iter().flat_map(|p| [p[0], p[1], p[2]]).collect(),
        alpha: layer.iter().map(|p| p[3]).collect(),
    }
}

#[test]
#[ignore = "gpu"]
fn gpu_bevy_adapter_sh_cap_takes_effect_without_a_reload() {
    let _layer = layer_messages(); // R-16, Q-105
    let (model, request) = probe("P5");
    assert_eq!(model.sh_degree, 3);
    assert_eq!(request.background, [0.0; 3]);
    let item = &request.items[1];
    let (w, h) = (item.camera.width, item.camera.height);
    // The probe loaded at degree 1 by the headless renderer (§1.2 load option 1).
    let mut renderer = HeadlessRenderer::new(QualityTier::conformance()).expect("an adapter");
    let id = renderer.load(&model, Some(1)).expect("P5 loads");
    let capped = renderer
        .render(
            &[(id, Pose::IDENTITY)],
            &item.camera,
            item.timestamp,
            [0.0; 3],
        )
        .expect("the capped render");

    let (camera, projection) = probe_camera(&item.camera);
    let mut scene = Scene::settled(
        OutputSpace::FileEncoded,
        TextureFormat::Rgba16Float,
        (w, h),
        camera,
        projection,
        Some(SceneModel {
            file: "probes/P5.safetensors",
            tau: item.timestamp,
            transform: Transform::IDENTITY,
        }),
        false,
    );
    let full = image_of(&scene.readback(), w, h);
    let m = metrics(&capped, &full);
    assert!(
        !(m.psnr >= MIN_PSNR && m.share <= MAX_SHARE),
        "degree 3 differs from degree 1, or the test is vacuous: {m:?}"
    );
    let uploads = scene.uploads();
    scene
        .app
        .world_mut()
        .resource_mut::<SplatTierState>()
        .sh_cap = 1;
    let ours = image_of(&scene.readback(), w, h);
    assert_eq!(scene.uploads(), uploads, "no reload");
    let m = metrics(&capped, &ours);
    eprintln!(
        "P5 capped at 1 by SplatTierState: PSNR {:.2} dB, s {:.4}, s_alpha {:.4}",
        m.psnr, m.share, m.alpha
    );
    assert!(
        m.psnr >= MIN_PSNR && m.share <= MAX_SHARE && m.alpha <= MAX_SHARE,
        "E15's tier: {m:?}"
    );
}

// Q-233, Q-235 (#388): the extraction applies `SplatShLimit`: with the governor's cap
// at 3, a limit of 1 draws the degree-3 probe at degree 1, and no limit draws it at 3.
#[test]
#[ignore = "gpu"]
fn gpu_bevy_adapter_sh_limit_caps_the_drawn_degree() {
    let _layer = layer_messages(); // R-16, Q-105
    let (model, request) = probe("P5");
    assert_eq!(model.sh_degree, 3);
    let item = &request.items[1];
    let (w, h) = (item.camera.width, item.camera.height);
    let mut renderer = HeadlessRenderer::new(QualityTier::conformance()).expect("an adapter");
    let id = renderer.load(&model, Some(1)).expect("P5 loads");
    let capped = renderer
        .render(
            &[(id, Pose::IDENTITY)],
            &item.camera,
            item.timestamp,
            [0.0; 3],
        )
        .expect("the capped render");

    let (camera, projection) = probe_camera(&item.camera);
    let mut scene = Scene::settled(
        OutputSpace::FileEncoded,
        TextureFormat::Rgba16Float,
        (w, h),
        camera,
        projection,
        Some(SceneModel {
            file: "probes/P5.safetensors",
            tau: item.timestamp,
            transform: Transform::IDENTITY,
        }),
        false,
    );
    scene
        .app
        .world_mut()
        .resource_mut::<SplatTierState>()
        .sh_cap = 3;
    let full = image_of(&scene.readback(), w, h);
    let uploads = scene.uploads();
    let m = metrics(&capped, &full);
    assert!(
        !(m.psnr >= MIN_PSNR && m.share <= MAX_SHARE),
        "degree 3 differs from degree 1, or the test is vacuous: {m:?}"
    );

    *scene.app.world_mut().resource_mut::<SplatShLimit>() = SplatShLimit(Some(1));
    let limited = image_of(&scene.readback(), w, h);
    let m = metrics(&capped, &limited);
    eprintln!(
        "P5 limited to 1 by SplatShLimit: PSNR {:.2} dB, s {:.4}, s_alpha {:.4}",
        m.psnr, m.share, m.alpha
    );
    assert!(
        m.psnr >= MIN_PSNR && m.share <= MAX_SHARE && m.alpha <= MAX_SHARE,
        "E15's tier: {m:?}"
    );
    assert_eq!(
        scene.app.world().resource::<SplatTierState>().sh_cap,
        3,
        "the limit leaves the governor's cap alone"
    );
    assert_eq!(scene.uploads(), uploads, "no reload");

    *scene.app.world_mut().resource_mut::<SplatShLimit>() = SplatShLimit(None);
    let again = image_of(&scene.readback(), w, h);
    let m = metrics(&full, &again);
    assert!(
        m.psnr >= MIN_PSNR && m.share <= MAX_SHARE && m.alpha <= MAX_SHARE,
        "no limit draws the loaded degree again: {m:?}"
    );
}
