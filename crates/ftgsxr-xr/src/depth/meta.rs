// SPDX-License-Identifier: Apache-2.0

//! The `EnvironmentDepth` adapter over XR\_META\_environment\_depth (§1.9 Depth
//! occlusion, E13, D-16, Appendix E.1 and E.2; source-log rows 357 and 358), in the
//! render world.
//!
//! The provider is created at the first `start`, after `Permission::Scene` is granted
//! (the runtime may refuse it before), with its one swapchain, whose images are two-layer
//! `D16_UNORM` arrays (0 left, 1 right) on Vulkan; it lives for the session. Each frame's
//! first `acquire` makes the runtime's one acquire at the frame's predicted display time
//! in the primary reference space (Q-341), imports the acquired swapchain image into
//! wgpu once (Q-339, `ftgsxr_render::import_vulkan_texture`), draws each layer into its
//! view's `DepthFrame.image` through a conversion pass (Q-360), and builds each view's
//! camera through the tracking root; `Views` gives the left then the right frame (Q-349).

use bevy::asset::Handle;
use bevy::image::Image;
use bevy::log::{info, warn};
use bevy::prelude::{GlobalTransform, Quat, Transform, Vec3};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource, BindingType,
    ColorTargetState, ColorWrites, CommandEncoderDescriptor, LoadOp, MultisampleState, Operations,
    PipelineCompilationOptions, PipelineLayoutDescriptor, PollType, PrimitiveState,
    RawFragmentState, RawRenderPipelineDescriptor, RawVertexState, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipeline, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    StoreOp, Texture, TextureAspect, TextureFormat, TextureSampleType, TextureView,
    TextureViewDescriptor, TextureViewDimension,
};
use bevy::render::renderer::RenderDevice;
use ftgsxr_app::Permission;
use ftgsxr_core::{Fov, UVec2};
use ftgsxr_render::{DepthFrame, SplatGpu, import_vulkan_texture};
use openxr::sys::Handle as _;
use openxr::{AnyGraphics, sys};

use super::{Acquired, DepthDriver, EXTENSION, EnvironmentDepth, Views, depth_camera};
use crate::error::XrError;

/// The swapchain images' format on Vulkan (registry, XR\_META\_environment\_depth: the
/// runtime must create `VK_FORMAT_D16_UNORM`; spike check 1 confirmed it on the Quest 3).
const SOURCE_FORMAT: TextureFormat = TextureFormat::Depth16Unorm;
/// The views' `DepthFrame.image` format: the stored depth d unchanged in the red channel,
/// a colour target the conversion pass can draw (Q-360); it binds as an unfilterable
/// float, as `DepthFrame` asks.
pub(crate) const FORMAT: TextureFormat = TextureFormat::R32Float;

/// The conversion pass (Q-360): a full-screen triangle whose fragment loads its own
/// texel of one layer of the runtime's depth image and writes it to red. Inline, as
/// Q-354's shader, so that every `.wgsl` file stays the render crate's (U-35).
// Q-360, Q-354
const CONVERSION_WGSL: &str = r"// SPDX-License-Identifier: Apache-2.0
// Q-360: one layer of the runtime's depth image, texel for texel, into red.
@group(0) @binding(0) var src: texture_depth_2d;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let d = textureLoad(src, vec2<i32>(p.xy), 0);
    return vec4<f32>(d, 0.0, 0.0, 1.0);
}
";

/// The conversion pass's layout and pipeline, made once per session.
struct Pass {
    layout: BindGroupLayout,
    pipeline: RenderPipeline,
}

impl Pass {
    fn new(render: &RenderDevice) -> Pass {
        let module = render.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("ftgsxr depth conversion"),
            source: ShaderSource::Wgsl(CONVERSION_WGSL.into()),
        });
        let layout = render.create_bind_group_layout(
            "ftgsxr depth conversion",
            &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Depth,
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        );
        let pipeline_layout = render.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("ftgsxr depth conversion"),
            bind_group_layouts: &[Some(&*layout)],
            immediate_size: 0,
        });
        let pipeline = render.create_render_pipeline(&RawRenderPipelineDescriptor {
            label: Some("ftgsxr depth conversion"),
            layout: Some(&pipeline_layout),
            vertex: RawVertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(RawFragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format: FORMAT,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Pass { layout, pipeline }
    }
}
/// The swapchain images' array layers: one per view (the registry's "array textures of
/// length 2").
const LAYERS: u32 = 2;

/// What one frame gives the conversion: the time and space of the acquire, the tracking root,
/// the device, and each view's `DepthFrame.image` with its prepared GPU texture.
#[derive(Clone)]
pub(crate) struct FrameCtx {
    pub(crate) time: sys::Time,
    pub(crate) space: sys::Space,
    pub(crate) root: GlobalTransform,
    pub(crate) gpu: Option<SplatGpu>,
    pub(crate) render: Option<RenderDevice>,
    pub(crate) targets: Option<[(Handle<Image>, TextureView); 2]>,
}

/// The adapter.
pub(crate) struct MetaDepth {
    session: openxr::Session<AnyGraphics>,
    provider: sys::EnvironmentDepthProviderMETA,
    swapchain: sys::EnvironmentDepthSwapchainMETA,
    /// The swapchain's `VkImage`s and their size.
    images: Vec<u64>,
    size: Option<UVec2>,
    /// Each swapchain image imported, once (Q-339).
    imported: Vec<Option<Texture>>,
    /// The conversion pass and each image's two layers' bind groups (Q-360).
    pass: Option<Pass>,
    groups: Vec<Option<[BindGroup; 2]>>,
    running: bool,
    views: Views,
    ctx: Option<FrameCtx>,
    /// The last image's capture time, 0 when the runtime does not report it.
    capture: i64,
    /// The calls that failed in this session (Dv-7's "without error").
    failures: u32,
}

/// Whether `r` is a success code.
fn ok(r: sys::Result) -> bool {
    r.into_raw() >= 0
}

/// The error of a failed creation: absent support, the permission, or the runtime's.
fn creation_error(what: &str, r: sys::Result) -> XrError {
    match r {
        sys::Result::ERROR_FEATURE_UNSUPPORTED | sys::Result::ERROR_FUNCTION_UNSUPPORTED => {
            XrError::Unsupported(EXTENSION)
        }
        sys::Result::ERROR_NOT_PERMITTED_PASSTHROUGH_FB => XrError::Permission(Permission::Scene),
        r => XrError::Runtime(format!("{what}: {r:?}")),
    }
}

impl MetaDepth {
    /// The adapter for `session`, whose instance enabled the extension; nothing is created
    /// before the first `start`.
    pub(crate) fn new(session: openxr::Session<AnyGraphics>) -> MetaDepth {
        MetaDepth {
            session,
            provider: sys::EnvironmentDepthProviderMETA::NULL,
            swapchain: sys::EnvironmentDepthSwapchainMETA::NULL,
            images: Vec::new(),
            size: None,
            imported: Vec::new(),
            pass: None,
            groups: Vec::new(),
            running: false,
            views: Views::default(),
            ctx: None,
            capture: 0,
            failures: 0,
        }
    }

    fn table(&self) -> Result<openxr::raw::EnvironmentDepthMETA, XrError> {
        self.session
            .instance()
            .exts()
            .meta_environment_depth
            .ok_or(XrError::Unsupported(EXTENSION))
    }

    /// Creates the provider and its swapchain, reads the swapchain's size and images.
    fn create(&mut self) -> Result<(), XrError> {
        let t = self.table()?;
        let info = sys::EnvironmentDepthProviderCreateInfoMETA {
            ty: sys::EnvironmentDepthProviderCreateInfoMETA::TYPE,
            next: std::ptr::null(),
            create_flags: sys::EnvironmentDepthProviderCreateFlagsMETA::EMPTY,
        };
        let mut provider = sys::EnvironmentDepthProviderMETA::NULL;
        // SAFETY: the table is the instance's (the extension enabled); the session handle
        // is live while `self.session` is held; `info` is a valid typed structure and
        // `provider` an output handle, both alive for the call.
        let r = unsafe {
            (t.create_environment_depth_provider)(self.session.as_raw(), &info, &mut provider)
        };
        if !ok(r) {
            return Err(creation_error("creating the depth provider", r));
        }
        self.provider = provider;
        let info = sys::EnvironmentDepthSwapchainCreateInfoMETA {
            ty: sys::EnvironmentDepthSwapchainCreateInfoMETA::TYPE,
            next: std::ptr::null(),
            create_flags: sys::EnvironmentDepthSwapchainCreateFlagsMETA::EMPTY,
        };
        let mut swapchain = sys::EnvironmentDepthSwapchainMETA::NULL;
        // SAFETY: `provider` was just created on this session; `info` and the output
        // handle are valid for the call.
        let r = unsafe { (t.create_environment_depth_swapchain)(provider, &info, &mut swapchain) };
        if !ok(r) {
            return Err(creation_error("creating the depth swapchain", r));
        }
        self.swapchain = swapchain;
        let mut state = sys::EnvironmentDepthSwapchainStateMETA {
            ty: sys::EnvironmentDepthSwapchainStateMETA::TYPE,
            next: std::ptr::null_mut(),
            width: 0,
            height: 0,
        };
        // SAFETY: `swapchain` is live; `state` is the typed output structure.
        let r = unsafe { (t.get_environment_depth_swapchain_state)(swapchain, &mut state) };
        if !ok(r) {
            return Err(creation_error("reading the depth swapchain's state", r));
        }
        let mut count = 0u32;
        // SAFETY: the two-call idiom's first call: capacity 0 and no array, the count out.
        let r = unsafe {
            (t.enumerate_environment_depth_swapchain_images)(
                swapchain,
                0,
                &mut count,
                std::ptr::null_mut(),
            )
        };
        if !ok(r) {
            return Err(creation_error("counting the depth swapchain's images", r));
        }
        let mut out = vec![
            sys::SwapchainImageVulkanKHR {
                ty: sys::SwapchainImageVulkanKHR::TYPE,
                next: std::ptr::null_mut(),
                image: 0,
            };
            count as usize
        ];
        // SAFETY: the second call: `out` holds `count` typed Vulkan image structures (the
        // session is Vulkan's, §2.4), the base header first in each, as the call reads them.
        let r = unsafe {
            (t.enumerate_environment_depth_swapchain_images)(
                swapchain,
                count,
                &mut count,
                out.as_mut_ptr().cast(),
            )
        };
        if !ok(r) {
            return Err(creation_error(
                "enumerating the depth swapchain's images",
                r,
            ));
        }
        out.truncate(count as usize);
        self.images = out.iter().map(|i| i.image).collect();
        self.imported = vec![None; self.images.len()];
        self.groups = vec![None; self.images.len()];
        self.size = Some(UVec2::new(state.width, state.height));
        info!(
            "depth: provider and swapchain created, {} images of {} x {}",
            self.images.len(),
            state.width,
            state.height
        );
        Ok(())
    }

    /// The runtime's one acquire of the frame, the conversion pass, the two views' frames.
    fn acquire_image(&mut self) -> Acquired {
        let (Some(ctx), Some(size), Ok(t)) = (self.ctx.clone(), self.size, self.table()) else {
            return Acquired::Failed;
        };
        if !self.running {
            return Acquired::Failed;
        }
        let (Some(gpu), Some(targets)) = (ctx.gpu, ctx.targets) else {
            // The images are not prepared yet (the first frames after the first start).
            return Acquired::Same;
        };
        let info = sys::EnvironmentDepthImageAcquireInfoMETA {
            ty: sys::EnvironmentDepthImageAcquireInfoMETA::TYPE,
            next: std::ptr::null(),
            space: ctx.space,
            display_time: ctx.time,
        };
        // The runtime writes `capture_time` (#602): a mutable binding, reached through a
        // pointer made from a mutable place, read back after the call.
        let mut stamp = sys::EnvironmentDepthImageTimestampMETA {
            ty: sys::EnvironmentDepthImageTimestampMETA::TYPE,
            next: std::ptr::null(),
            capture_time: sys::Time::from_nanos(0),
        };
        let view = sys::EnvironmentDepthImageViewMETA {
            ty: sys::EnvironmentDepthImageViewMETA::TYPE,
            next: std::ptr::null(),
            fov: sys::Fovf {
                angle_left: 0.0,
                angle_right: 0.0,
                angle_up: 0.0,
                angle_down: 0.0,
            },
            pose: sys::Posef::IDENTITY,
        };
        let mut image = sys::EnvironmentDepthImageMETA {
            ty: sys::EnvironmentDepthImageMETA::TYPE,
            next: (&raw mut stamp).cast::<std::ffi::c_void>().cast_const(),
            swapchain_index: 0,
            near_z: 0.0,
            far_z: 0.0,
            views: [view; 2],
        };
        // SAFETY: the provider is live and started (`running`), and the call falls
        // between this frame's xrBeginFrame and xrEndFrame (the render world's Prepare,
        // after bevy_mod_openxr's begin, source-log row 361), once per frame (`Views`);
        // `info`, `image` and the timestamp chained on its `next` are valid typed
        // structures alive for the call, which writes `image` through `&mut image` and the
        // timestamp through its `next`, a pointer made from the mutable `stamp` (`&raw
        // mut`), so the write is allowed; nothing else touches `stamp` until it is read
        // after the call (#602).
        let r = unsafe { (t.acquire_environment_depth_image)(self.provider, &info, &mut image) };
        if r == sys::Result::ENVIRONMENT_DEPTH_NOT_AVAILABLE_META {
            return Acquired::Same;
        }
        if !ok(r) {
            self.failures += 1;
            warn!("depth: acquiring the image failed: {r:?}");
            return Acquired::Failed;
        }
        let index = image.swapchain_index as usize;
        let Some(&raw) = self.images.get(index) else {
            self.failures += 1;
            warn!(
                "depth: swapchain index {index} out of {}",
                self.images.len()
            );
            return Acquired::Failed;
        };
        let source = self.imported[index].get_or_insert_with(|| {
            // SAFETY: `raw` is an image of the runtime's depth swapchain, created on the
            // session's VkDevice, which is Bevy's and so `gpu`'s (bevy_mod_openxr creates
            // the device for the session); the registry fixes its format (D16) and layer
            // count (2) and the swapchain state its size; its images live as long as the
            // swapchain, which outlives this texture (`Drop` waits for the GPU and drops
            // the textures before destroying it); it was written by the runtime before
            // this acquire returned. Its usages are sampled and depth-attachment (Dv-7's
            // validation layer, spike check 1), not the transfer-source usage the helper's
            // contract names: the texture is only sampled here (the conversion pass,
            // Q-360), never copied, so no command needs it. Q-356: wgpu tracks it
            // from UNINITIALIZED.
            let texture =
                unsafe { import_vulkan_texture(gpu.device(), raw, size, LAYERS, SOURCE_FORMAT) };
            Texture::from(texture)
        });
        // Q-360: the runtime's images have no transfer-source usage (Dv-7's validation
        // layer, spike check 1), so each layer is drawn into its view's image by the
        // conversion pass, which samples it.
        let Some(render) = ctx.render else {
            return Acquired::Failed;
        };
        let pass = self.pass.get_or_insert_with(|| Pass::new(&render));
        let groups = self.groups[index].get_or_insert_with(|| {
            [0u32, 1].map(|layer| {
                let view = source.create_view(&TextureViewDescriptor {
                    label: Some("ftgsxr depth layer"),
                    dimension: Some(TextureViewDimension::D2),
                    aspect: TextureAspect::DepthOnly,
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..TextureViewDescriptor::default()
                });
                render.create_bind_group(
                    "ftgsxr depth layer",
                    &pass.layout,
                    &[BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&view),
                    }],
                )
            })
        });
        let mut enc = render.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("ftgsxr depth conversion"),
        });
        for (layer, (_, target)) in targets.iter().enumerate() {
            let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("ftgsxr depth conversion"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Load,
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            rp.set_pipeline(&pass.pipeline);
            rp.set_bind_group(0, &*groups[layer], &[]);
            rp.draw(0..3, 0..1);
        }
        // Before the camera passes, which Bevy submits in `Render`.
        gpu.device().queue().submit([enc.finish()]);
        let capture = stamp.capture_time.as_nanos();
        let new = capture == 0 || capture != self.capture;
        self.capture = capture;
        if !new {
            return Acquired::Same;
        }
        let frame = |v: usize| {
            let p = image.views[v].pose;
            let f = image.views[v].fov;
            let view = Transform {
                translation: Vec3::new(p.position.x, p.position.y, p.position.z),
                rotation: Quat::from_xyzw(
                    p.orientation.x,
                    p.orientation.y,
                    p.orientation.z,
                    p.orientation.w,
                )
                .normalize(),
                scale: Vec3::ONE,
            };
            let fov = Fov::Angles {
                left: f.angle_left,
                right: f.angle_right,
                up: f.angle_up,
                down: f.angle_down,
            };
            DepthFrame {
                image: targets[v].0.clone(),
                camera: depth_camera(view, fov, size, &ctx.root),
                near: image.near_z,
                far: image.far_z,
                age_frames: 0,
            }
        };
        Acquired::New(Box::new([frame(0), frame(1)]))
    }

    /// Hand removal on or off (D-16); a system without it keeps the depth with hands.
    fn hand_removal(&mut self, on: bool) -> Result<(), XrError> {
        let t = self.table()?;
        let info = sys::EnvironmentDepthHandRemovalSetInfoMETA {
            ty: sys::EnvironmentDepthHandRemovalSetInfoMETA::TYPE,
            next: std::ptr::null(),
            enabled: on.into(),
        };
        // SAFETY: the provider is live (created before any call to this); `info` is a
        // valid typed structure alive for the call.
        let r = unsafe { (t.set_environment_depth_hand_removal)(self.provider, &info) };
        if ok(r) {
            info!("depth: hand removal {on}");
            Ok(())
        } else {
            Err(creation_error("setting hand removal", r))
        }
    }
}

impl EnvironmentDepth for MetaDepth {
    fn start(&mut self, hand_removal: bool) -> Result<(), XrError> {
        if self.running {
            return Ok(());
        }
        if self.provider == sys::EnvironmentDepthProviderMETA::NULL {
            self.create()?;
        }
        // Q-357: a system without hand removal still gives depth.
        if let Err(e) = self.hand_removal(hand_removal) {
            warn!("depth: {e}");
        }
        let t = self.table()?;
        // SAFETY: the provider is live and stopped (`running` false).
        let r = unsafe { (t.start_environment_depth_provider)(self.provider) };
        if !ok(r) {
            self.failures += 1;
            return Err(creation_error("starting the depth provider", r));
        }
        self.running = true;
        self.views.clear();
        info!("depth: provider started");
        Ok(())
    }

    fn acquire(&mut self) -> Option<DepthFrame> {
        let mut views = std::mem::take(&mut self.views);
        let frame = views.next(|| self.acquire_image());
        self.views = views;
        frame
    }

    fn stop(&mut self) {
        if !self.running {
            return;
        }
        self.running = false;
        self.views.clear();
        let Ok(t) = self.table() else { return };
        // SAFETY: the provider is live and started.
        let r = unsafe { (t.stop_environment_depth_provider)(self.provider) };
        if ok(r) {
            info!("depth: provider stopped");
        } else {
            self.failures += 1;
            warn!("depth: stopping the provider failed: {r:?}");
        }
    }
}

impl DepthDriver for MetaDepth {
    fn swapchain(&self) -> Option<(UVec2, usize)> {
        self.size.map(|s| (s, self.images.len()))
    }

    fn set_hand_removal(&mut self, on: bool) -> Result<(), XrError> {
        if self.provider == sys::EnvironmentDepthProviderMETA::NULL {
            return Ok(());
        }
        self.hand_removal(on)
    }

    fn begin_frame(&mut self, ctx: FrameCtx) {
        self.ctx = Some(ctx);
        self.views.next_frame();
    }

    fn running(&self) -> bool {
        self.running
    }

    fn failures(&self) -> u32 {
        self.failures
    }
}

impl Drop for MetaDepth {
    fn drop(&mut self) {
        self.stop();
        // The swapchain's images may be in flight: the GPU finishes first, then the
        // imported textures go, then the swapchain and the provider (registry: the
        // commands that refer to the swapchain must have completed).
        if let Some(gpu) = self.ctx.as_ref().and_then(|c| c.gpu.as_ref()) {
            let _ = gpu.device().device().poll(PollType::wait_indefinitely());
        }
        self.groups.clear();
        self.imported.clear();
        let Ok(t) = self.table() else { return };
        if self.swapchain != sys::EnvironmentDepthSwapchainMETA::NULL {
            // SAFETY: the swapchain is live, no GPU work refers to it (waited above), and
            // nothing uses it after this.
            unsafe { (t.destroy_environment_depth_swapchain)(self.swapchain) };
        }
        if self.provider != sys::EnvironmentDepthProviderMETA::NULL {
            // SAFETY: the provider is live and stopped, its swapchain destroyed.
            unsafe { (t.destroy_environment_depth_provider)(self.provider) };
        }
    }
}
