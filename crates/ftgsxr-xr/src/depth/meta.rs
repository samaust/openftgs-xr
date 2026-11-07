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
//! wgpu once (Q-339, `ftgsxr_render::import_vulkan_texture`), and on a new capture draws
//! each layer into its view's history ring through a conversion pass (Q-360), then the
//! ring's per-texel median, clamped to the range of the newest capture's 3 × 3
//! neighbourhood, into the view's `DepthFrame.image` (Q-363, #607), and builds each
//! view's camera through the tracking root; `Views` gives the left then the right frame
//! (Q-349).

use bevy::asset::Handle;
use bevy::image::Image;
use bevy::log::{info, warn};
use bevy::prelude::{GlobalTransform, Quat, Transform, Vec3};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource, BindingType,
    ColorTargetState, ColorWrites, CommandEncoder, CommandEncoderDescriptor, Extent3d, LoadOp,
    MultisampleState, Operations, PipelineCompilationOptions, PipelineLayoutDescriptor, PollType,
    PrimitiveState, RawFragmentState, RawRenderPipelineDescriptor, RawVertexState,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, StoreOp, Texture, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
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

/// The conversion pass (Q-360): a full-screen triangle whose fragment loads its
/// texel of one layer of the runtime's depth image, rows flipped, and writes it to red.
/// The runtime stores row 0 at the bottom (Q-362, #600) and `DepthFrame.image` is read
/// top row first (E6's pixel coordinates), so output row y takes source row H − 1 − y.
/// It writes a slot of the view's history ring, an invalid sample (not finite, or d ≥ 1)
/// as d = 1 (Q-363, Q-340). Inline, as Q-354's shader, so that every `.wgsl` file
/// stays the render crate's (U-35).
// Q-360, Q-362, Q-354, Q-363
const CONVERSION_WGSL: &str = r"// SPDX-License-Identifier: Apache-2.0
// Q-360: one layer of the runtime's depth image, texel for texel, into red;
// Q-362: the runtime's row 0 is the bottom one, so the rows are flipped;
// Q-363: an invalid sample (NaN, or d >= 1) is written as d = 1.
@group(0) @binding(0) var src: texture_depth_2d;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let h = i32(textureDimensions(src).y);
    let d = textureLoad(src, vec2<i32>(i32(p.x), h - 1 - i32(p.y)), 0);
    return vec4<f32>(select(d, 1.0, !(d < 1.0)), 0.0, 0.0, 1.0);
}
";

/// The median pass (Q-363): a full-screen triangle whose fragment writes, to red, the
/// median of its texel in three images of the view's history ring (the newest at
/// binding 0, bound three times while the ring holds fewer than 3), clamped to the min
/// and max of the newest capture over the 3 × 3 texels around it, those outside the
/// image left out (#607: the neighbourhood clamp of temporal anti-aliasing). The ring
/// holds invalid samples as d = 1, which take part in the median and the clamp as that
/// value: a texel valid in 2 of 3 captures keeps a valid median unless the newest
/// neighbourhood is all invalid, and d = 1 stays invalid to E13's decode (Q-340). The
/// border's neighbourhood reads clamped indices, which visit only texels inside the
/// image and every one of them, so the min and max are those of the texels inside.
/// Inline, as the conversion pass.
// Q-363, Q-354
const MEDIAN_WGSL: &str = r"// SPDX-License-Identifier: Apache-2.0
// Q-363: the per-texel median of three converted captures, clamped to the range of
// the newest capture's 3 x 3 neighbourhood inside the image (#607).
@group(0) @binding(0) var a: texture_2d<f32>;
@group(0) @binding(1) var b: texture_2d<f32>;
@group(0) @binding(2) var c: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let q = vec2<i32>(p.xy);
    let x = textureLoad(a, q, 0).r;
    let y = textureLoad(b, q, 0).r;
    let z = textureLoad(c, q, 0).r;
    let m = max(min(x, y), min(max(x, y), z));
    let top = vec2<i32>(textureDimensions(a)) - 1;
    var lo = x;
    var hi = x;
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let n = textureLoad(a, clamp(q + vec2<i32>(i, j), vec2<i32>(0), top), 0).r;
            lo = min(lo, n);
            hi = max(hi, n);
        }
    }
    return vec4<f32>(clamp(m, lo, hi), 0.0, 0.0, 1.0);
}
";

/// A full-screen pipeline of `wgsl`'s `vs` and `fs`, writing `FORMAT`, over one bind
/// group of `entries`' fragment textures.
fn full_screen(
    render: &RenderDevice,
    label: &'static str,
    wgsl: &'static str,
    entries: &[TextureSampleType],
) -> (BindGroupLayout, RenderPipeline) {
    let module = render.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(wgsl.into()),
    });
    let entries: Vec<_> = entries
        .iter()
        .enumerate()
        .map(|(i, &sample_type)| BindGroupLayoutEntry {
            binding: i as u32,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture {
                sample_type,
                view_dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        })
        .collect();
    let layout = render.create_bind_group_layout(label, &entries);
    let pipeline_layout = render.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&*layout)],
        immediate_size: 0,
    });
    let pipeline = render.create_render_pipeline(&RawRenderPipelineDescriptor {
        label: Some(label),
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
    (layout, pipeline)
}

/// Draws `pipeline` with `group` over the whole of `target`.
fn draw(
    enc: &mut CommandEncoder,
    pipeline: &RenderPipeline,
    group: &BindGroup,
    target: &TextureView,
) {
    let mut rp = enc.begin_render_pass(&RenderPassDescriptor {
        label: Some("ftgsxr depth"),
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
    rp.set_pipeline(pipeline);
    rp.set_bind_group(0, &**group, &[]);
    rp.draw(0..3, 0..1);
}

/// The conversion and median passes and the two views' history rings (Q-360,
/// Q-363), made once per session, whose swapchain has one size.
struct Filter {
    convert_layout: BindGroupLayout,
    convert: RenderPipeline,
    median: RenderPipeline,
    /// Each view's ring: its 3 slots' views (the textures live as long as their views)…
    slots: [[TextureView; RING]; 2],
    /// …and the median's bind groups: slot i alone (bound three times) at i, the ring's
    /// 3 slots with slot i the newest, at binding 0 where the clamp reads it, at
    /// `RING + i` (#607).
    groups: [[BindGroup; 2 * RING]; 2],
}

impl Filter {
    fn new(render: &RenderDevice, size: UVec2) -> Filter {
        let (convert_layout, convert) = full_screen(
            render,
            "ftgsxr depth conversion",
            CONVERSION_WGSL,
            &[TextureSampleType::Depth],
        );
        let float = TextureSampleType::Float { filterable: false };
        let (median_layout, median) = full_screen(
            render,
            "ftgsxr depth median",
            MEDIAN_WGSL,
            &[float, float, float],
        );
        let slots = [0, 1].map(|_| {
            [0; RING].map(|_| {
                render
                    .create_texture(&TextureDescriptor {
                        label: Some("ftgsxr depth history"),
                        size: Extent3d {
                            width: size.x,
                            height: size.y,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: TextureDimension::D2,
                        format: FORMAT,
                        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    })
                    .create_view(&TextureViewDescriptor::default())
            })
        });
        let group = |views: [&TextureView; 3]| {
            render.create_bind_group(
                "ftgsxr depth median",
                &median_layout,
                &[0, 1, 2].map(|i| BindGroupEntry {
                    binding: i as u32,
                    resource: BindingResource::TextureView(views[i]),
                }),
            )
        };
        let groups = [0, 1].map(|v| {
            let s = &slots[v];
            [
                group([&s[0]; 3]),
                group([&s[1]; 3]),
                group([&s[2]; 3]),
                group([&s[0], &s[1], &s[2]]),
                group([&s[1], &s[2], &s[0]]),
                group([&s[2], &s[0], &s[1]]),
            ]
        });
        Filter {
            convert_layout,
            convert,
            median,
            slots,
            groups,
        }
    }

    /// The conversion pass's bind group of `layer` of `source`, a two-layer depth texture.
    fn layer_group(&self, render: &RenderDevice, source: &Texture, layer: u32) -> BindGroup {
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
            &self.convert_layout,
            &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view),
            }],
        )
    }

    /// A new capture (Q-363): each view's layer (`layers`, left then right) converted
    /// into its step's slot, then the median of what the step reads into the view's
    /// `DepthFrame.image` (`targets`).
    fn encode(
        &self,
        enc: &mut CommandEncoder,
        layers: [&BindGroup; 2],
        steps: [Step; 2],
        targets: [&TextureView; 2],
    ) {
        for v in 0..2 {
            let Step { slot, read } = steps[v];
            draw(enc, &self.convert, layers[v], &self.slots[v][slot]);
            let group = match read {
                Read::Newest => &self.groups[v][slot],
                Read::Median => &self.groups[v][RING + slot],
            };
            draw(enc, &self.median, group, targets[v]);
        }
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

/// Q-363: the captures each view's history ring holds, the last capture's position.
const RING: usize = 3;
/// Q-363: a view's ring restarts when its pose moved more than this (metres)…
const RESET_POSITION_M: f32 = 0.005;
/// …or turned more than this (degrees) since the previous capture.
const RESET_ANGLE_DEG: f32 = 0.2;

/// What a new capture does to one view (Q-363): the conversion pass writes the
/// view's layer into ring slot `slot`, then the median pass reads `read`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Step {
    slot: usize,
    read: Read,
}

/// What the median pass reads (Q-363).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Read {
    /// The newest capture alone (the ring holds fewer than 3).
    Newest,
    /// The median of the ring's 3 captures.
    Median,
}

/// One view's ring (Q-363): how many captures it holds, the newest's slot and the
/// view pose it was taken at.
#[derive(Clone, Copy, Default)]
struct Ring {
    len: usize,
    newest: usize,
    pose: Option<(Vec3, Quat)>,
}

/// The history policy of the temporal median (Q-363), for both views: which acquire
/// gives a new capture, which slot it is converted into and what the median reads, and
/// when a view's ring restarts. Plain Rust, so that the render-world code only follows it.
// Q-363, Q-357
#[derive(Default)]
struct History {
    capture: i64,
    rings: [Ring; 2],
}

/// The angle between two orientations, in degrees, the same for `q` and `−q`.
fn angle_deg(a: Quat, b: Quat) -> f32 {
    let d = a.inverse() * b;
    2.0 * d.xyz().length().atan2(d.w.abs()).to_degrees()
}

impl Ring {
    /// A new capture at the view's raw `pose`: its slot, and what the median reads.
    fn push(&mut self, pose: (Vec3, Quat)) -> Step {
        // Q-363: images from different poses are never combined.
        if let Some((p, q)) = self.pose
            && (p.distance(pose.0) > RESET_POSITION_M || angle_deg(q, pose.1) > RESET_ANGLE_DEG)
        {
            self.len = 0;
        }
        self.newest = if self.len == 0 {
            0
        } else {
            (self.newest + 1) % RING
        };
        self.len = (self.len + 1).min(RING);
        self.pose = Some(pose);
        Step {
            slot: self.newest,
            read: if self.len == RING {
                Read::Median
            } else {
                Read::Newest
            },
        }
    }
}

impl History {
    /// The two views' steps for an acquire of capture time `capture` (0 when the runtime
    /// does not report it) at the views' raw poses; `None` when it is not a new capture
    /// (Q-357: the same capture time), which leaves everything as it is.
    fn next(&mut self, capture: i64, poses: [(Vec3, Quat); 2]) -> Option<[Step; 2]> {
        let new = capture == 0 || capture != self.capture;
        if !new {
            return None;
        }
        self.capture = capture;
        let [l, r] = poses;
        Some([self.rings[0].push(l), self.rings[1].push(r)])
    }

    /// Forgets every capture (the provider started).
    fn clear(&mut self) {
        *self = History::default();
    }
}

/// A view's raw pose from the runtime (before the tracking root).
fn view_pose(p: sys::Posef) -> (Vec3, Quat) {
    (
        Vec3::new(p.position.x, p.position.y, p.position.z),
        Quat::from_xyzw(
            p.orientation.x,
            p.orientation.y,
            p.orientation.z,
            p.orientation.w,
        )
        .normalize(),
    )
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
    /// The conversion and median passes with the views' history rings (Q-360,
    /// Q-363), and each image's two layers' bind groups.
    filter: Option<Filter>,
    groups: Vec<Option<[BindGroup; 2]>>,
    /// Which capture is new and what the median reads (Q-357, Q-363).
    history: History,
    running: bool,
    views: Views,
    ctx: Option<FrameCtx>,
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
            filter: None,
            groups: Vec::new(),
            history: History::default(),
            running: false,
            views: Views::default(),
            ctx: None,
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
        let Some(render) = ctx.render else {
            return Acquired::Failed;
        };
        // Q-357, Q-363: a frame without a new capture leaves the images as they are.
        let poses = [0, 1].map(|v| view_pose(image.views[v].pose));
        let Some(steps) = self.history.next(stamp.capture_time.as_nanos(), poses) else {
            return Acquired::Same;
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
        // layer, spike check 1), so each layer is drawn into its view's history ring by
        // the conversion pass, which samples it; the median pass then writes the view's
        // image (Q-363).
        let filter = self
            .filter
            .get_or_insert_with(|| Filter::new(&render, size));
        let groups = self.groups[index].get_or_insert_with(|| {
            [0u32, 1].map(|layer| filter.layer_group(&render, source, layer))
        });
        let mut enc = render.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("ftgsxr depth conversion"),
        });
        filter.encode(
            &mut enc,
            [&groups[0], &groups[1]],
            steps,
            [&targets[0].1, &targets[1].1],
        );
        // Before the camera passes, which Bevy submits in `Render`.
        gpu.device().queue().submit([enc.finish()]);
        let frame = |v: usize| {
            let (translation, rotation) = poses[v];
            let f = image.views[v].fov;
            let view = Transform {
                translation,
                rotation,
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
        // Q-363: no median across a stop.
        self.history.clear();
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

#[cfg(test)]
mod tests {
    use super::*;

    const STILL: (Vec3, Quat) = (Vec3::ZERO, Quat::IDENTITY);

    fn step(slot: usize, read: Read) -> Step {
        Step { slot, read }
    }

    #[test]
    fn history_uses_the_newest_until_three_captures_then_the_median() {
        let mut h = History::default();
        let reads: Vec<_> = (1..=5)
            .map(|c| h.next(c, [STILL; 2]).expect("a new capture")[0])
            .collect();
        assert_eq!(
            reads,
            [
                step(0, Read::Newest),
                step(1, Read::Newest),
                step(2, Read::Median),
                step(0, Read::Median),
                step(1, Read::Median),
            ]
        );
    }

    #[test]
    fn history_leaves_everything_alone_without_a_new_capture() {
        let mut h = History::default();
        assert!(h.next(10, [STILL; 2]).is_some());
        // The same capture time again (Q-357), even at another pose: no step, no change.
        let moved = (Vec3::X, Quat::IDENTITY);
        assert_eq!(h.next(10, [moved; 2]), None);
        assert_eq!(h.next(10, [STILL; 2]), None);
        assert_eq!(h.next(11, [STILL; 2]), Some([step(1, Read::Newest); 2]));
        assert_eq!(h.next(12, [STILL; 2]), Some([step(2, Read::Median); 2]));
    }

    #[test]
    fn history_counts_an_unreported_capture_time_as_new_every_time() {
        let mut h = History::default();
        assert_eq!(h.next(0, [STILL; 2]), Some([step(0, Read::Newest); 2]));
        assert_eq!(h.next(0, [STILL; 2]), Some([step(1, Read::Newest); 2]));
        assert_eq!(h.next(0, [STILL; 2]), Some([step(2, Read::Median); 2]));
    }

    /// Three captures at `a`, then one at `b`: the fourth capture's step for each view.
    fn after_three(a: [(Vec3, Quat); 2], b: [(Vec3, Quat); 2]) -> [Step; 2] {
        let mut h = History::default();
        for c in 1..=3 {
            h.next(c, a);
        }
        h.next(4, b).expect("a new capture")
    }

    #[test]
    fn history_restarts_a_view_moved_more_than_5_mm() {
        let at = |x: f32| (Vec3::new(x, 1.6, -0.2), Quat::IDENTITY);
        let base = at(0.3);
        // 4.9 mm along x keeps the ring; 5.1 mm restarts it from the new capture.
        assert_eq!(
            after_three([base; 2], [at(0.3049); 2]),
            [step(0, Read::Median); 2]
        );
        assert_eq!(
            after_three([base; 2], [at(0.3051), base]),
            [step(0, Read::Newest), step(0, Read::Median)]
        );
    }

    #[test]
    fn history_restarts_a_view_turned_more_than_0_2_degrees() {
        let turned = |deg: f32| {
            (
                Vec3::ZERO,
                Quat::from_rotation_y(0.4) * Quat::from_rotation_x(deg.to_radians()),
            )
        };
        let base = turned(0.0);
        assert_eq!(
            after_three([base; 2], [turned(0.19); 2]),
            [step(0, Read::Median); 2]
        );
        assert_eq!(
            after_three([base; 2], [base, turned(0.21)]),
            [step(0, Read::Median), step(0, Read::Newest)]
        );
        // The same orientation written as −q is no turn.
        let flipped = (Vec3::ZERO, -base.1);
        assert_eq!(
            after_three([base; 2], [flipped; 2]),
            [step(0, Read::Median); 2]
        );
    }

    #[test]
    fn history_after_a_restart_grows_again_from_the_new_capture() {
        let mut h = History::default();
        let far = (Vec3::new(0.0, 0.0, 0.1), Quat::IDENTITY);
        for c in 1..=4 {
            h.next(c, [STILL; 2]);
        }
        let reads: Vec<_> = (5..=8)
            .map(|c| h.next(c, [far; 2]).expect("a new capture")[0].read)
            .collect();
        assert_eq!(
            reads,
            [Read::Newest, Read::Newest, Read::Median, Read::Median]
        );
    }

    #[test]
    fn history_clear_forgets_every_capture() {
        let mut h = History::default();
        for c in 1..=3 {
            h.next(c, [STILL; 2]);
        }
        h.clear();
        assert_eq!(
            h.next(3, [STILL; 2]).expect("new after a clear")[0].read,
            Read::Newest
        );
    }

    /// Reads back an `R32Float` texture of `w` × `h`, row-major from the top row.
    fn read_back(
        dev: &ftgsxr_render::SplatDevice,
        render: &RenderDevice,
        texture: &Texture,
        (w, h): (u32, u32),
    ) -> Vec<f32> {
        use bevy::render::render_resource::{
            BufferDescriptor, BufferUsages, MapMode, Origin3d, TexelCopyBufferInfo,
            TexelCopyBufferLayout, TexelCopyTextureInfo,
        };
        let row = (4 * w).next_multiple_of(256);
        let buffer = render.wgpu_device().create_buffer(&BufferDescriptor {
            label: None,
            size: u64::from(row * h),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = render.create_command_encoder(&CommandEncoderDescriptor { label: None });
        enc.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(h),
                },
            },
            Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        dev.queue().submit([enc.finish()]);
        buffer.map_async(MapMode::Read, .., |r| r.expect("maps"));
        dev.device()
            .poll(PollType::wait_indefinitely())
            .expect("the device polls");
        let mapped = buffer.get_mapped_range(..);
        (0..h)
            .flat_map(|y| (0..w).map(move |x| (y * row + 4 * x) as usize))
            .map(|at| f32::from_le_bytes(mapped[at..at + 4].try_into().expect("4 bytes")))
            .collect()
    }

    /// Q-363, Q-360, Q-362: synthetic two-layer `Depth16Unorm` captures through the
    /// conversion and median passes. The left layer's texels pin the median's cases (a
    /// texel valid in 2 of 3 captures stays valid, a one-capture flip is rejected, a texel
    /// clamped to the newest capture's 3 × 3 range, #607, the middle of three distinct
    /// values), the newest capture alone before the ring is full and after a restart, and
    /// the row flip; the right layer reaches the right view's image.
    #[test]
    #[ignore = "gpu"]
    fn gpu_median_pass_takes_the_median_of_three_converted_captures() {
        use ftgsxr_testkit::{gpu_context, layer_messages};
        let _layer = layer_messages(); // Q-105
        let dev = gpu_context();
        let render = RenderDevice::from(dev.device().clone());
        let (w, h) = (4u32, 2u32);
        let extent = |layers| Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: layers,
        };
        let source = render.create_texture(&TextureDescriptor {
            label: Some("stand-in depth swapchain image"),
            size: extent(LAYERS),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: SOURCE_FORMAT,
            usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let targets = [0, 1].map(|_| {
            let t = render.create_texture(&TextureDescriptor {
                label: Some("stand-in DepthFrame.image"),
                size: extent(1),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: FORMAT,
                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let v = t.create_view(&TextureViewDescriptor::default());
            (t, v)
        });
        let filter = Filter::new(&render, UVec2::new(w, h));
        let groups = [0, 1].map(|l| filter.layer_group(&render, &source, l));

        // The left layer's captures, by output texel (top row first): row 0 holds the
        // median's cases, row 1 the values 1000 c + x; capture 4 is 42 everywhere.
        let cases = [
            [100, 100, 65535],
            [100, 60000, 100],
            [65535, 65535, 500],
            [300, 200, 100],
        ];
        let left = |c: usize, x: u32, y: u32| -> u16 {
            match (c, y) {
                (4, _) => 42,
                (_, 0) => cases[x as usize][c - 1],
                _ => (1000 * c as u32 + x) as u16,
            }
        };
        let right = |x: u32, y: u32| (7 + x + 4 * y) as u16;
        let d = |v: u16| f32::from(v) / 65535.0;
        let capture = |c: usize, steps: [Step; 2]| -> [Vec<f32>; 2] {
            // The source stores row 0 at the bottom (Q-362): output row y is its H − 1 − y.
            let mut bytes = Vec::new();
            for layer in 0..2 {
                for ys in 0..h {
                    for x in 0..w {
                        let y = h - 1 - ys;
                        let v = if layer == 0 {
                            left(c, x, y)
                        } else {
                            right(x, y)
                        };
                        bytes.extend(v.to_le_bytes());
                    }
                }
            }
            dev.queue().write_texture(
                source.as_image_copy(),
                &bytes,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(2 * w),
                    rows_per_image: Some(h),
                },
                extent(LAYERS),
            );
            let mut enc = render.create_command_encoder(&CommandEncoderDescriptor { label: None });
            filter.encode(
                &mut enc,
                [&groups[0], &groups[1]],
                steps,
                [&targets[0].1, &targets[1].1],
            );
            dev.queue().submit([enc.finish()]);
            [0, 1].map(|v| read_back(&dev, &render, &targets[v].0, (w, h)))
        };
        use bevy::render::render_resource::TexelCopyBufferLayout;
        let texels = |f: &dyn Fn(u32, u32) -> u16| -> Vec<f32> {
            (0..h)
                .flat_map(|y| (0..w).map(move |x| (x, y)))
                .map(|(x, y)| d(f(x, y)))
                .collect()
        };
        let close = |got: &[f32], want: &[f32], what: &str| {
            for (i, (g, e)) in got.iter().zip(want).enumerate() {
                assert!((g - e).abs() <= 1e-6, "{what}, texel {i}: {g} against {e}");
            }
        };
        let both = |slot, read| [step(slot, read); 2];

        let [l, r] = capture(1, both(0, Read::Newest));
        close(
            &l,
            &texels(&|x, y| left(1, x, y)),
            "capture 1 alone, rows flipped",
        );
        close(&r, &texels(&right), "the right layer");
        let [l, _] = capture(2, both(1, Read::Newest));
        close(&l, &texels(&|x, y| left(2, x, y)), "capture 2 alone");
        let [l, r] = capture(3, both(2, Read::Median));
        // Texel (2, 0) is invalid in 2 of 3 captures, but the newest capture is valid
        // over its whole 3 × 3 neighbourhood (100 to 3003), so the clamp (#607) takes
        // the median's d = 1 down to that range's max.
        let median = |x: u32, y: u32| match y {
            0 => [100, 100, 3003, 200][x as usize],
            _ => (2000 + x) as u16,
        };
        close(
            &l,
            &texels(&median),
            "the median of captures 1 to 3, clamped",
        );
        close(&r, &texels(&right), "the right layer's median");
        let [l, _] = capture(4, both(0, Read::Newest));
        close(&l, &texels(&|_, _| 42), "a restart: capture 4 alone");

        drop(groups);
        let messages = dev.take_validation_messages();
        assert!(messages.is_empty(), "validation messages: {messages:?}");
    }

    /// A synthetic layer: `f(x, y)` is output texel (x, y), top row first, as `D16_UNORM`.
    type Layer<'a> = dyn Fn(u32, u32) -> u16 + 'a;

    /// Q-363 (#607): runs `captures`' left layers, each with its step for both views,
    /// through the conversion and median passes on a `w` × `h` image (`f(x, y)` gives
    /// output texel (x, y), top row first, as `D16_UNORM`; the right layer is 7
    /// everywhere), and returns the left view's image after the last.
    fn left_after(w: u32, h: u32, captures: &[(&Layer<'_>, Step)]) -> Vec<f32> {
        use bevy::render::render_resource::TexelCopyBufferLayout;
        use ftgsxr_testkit::{gpu_context, layer_messages};
        let _layer = layer_messages(); // Q-105
        let dev = gpu_context();
        let render = RenderDevice::from(dev.device().clone());
        let extent = |layers| Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: layers,
        };
        let source = render.create_texture(&TextureDescriptor {
            label: Some("stand-in depth swapchain image"),
            size: extent(LAYERS),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: SOURCE_FORMAT,
            usage: TextureUsages::COPY_DST | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let targets = [0, 1].map(|_| {
            let t = render.create_texture(&TextureDescriptor {
                label: Some("stand-in DepthFrame.image"),
                size: extent(1),
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: FORMAT,
                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let v = t.create_view(&TextureViewDescriptor::default());
            (t, v)
        });
        let filter = Filter::new(&render, UVec2::new(w, h));
        let groups = [0, 1].map(|l| filter.layer_group(&render, &source, l));
        for &(f, s) in captures {
            // The source stores row 0 at the bottom (Q-362).
            let mut bytes = Vec::new();
            for layer in 0..2 {
                for ys in 0..h {
                    for x in 0..w {
                        let v = if layer == 0 { f(x, h - 1 - ys) } else { 7 };
                        bytes.extend(v.to_le_bytes());
                    }
                }
            }
            dev.queue().write_texture(
                source.as_image_copy(),
                &bytes,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(2 * w),
                    rows_per_image: Some(h),
                },
                extent(LAYERS),
            );
            let mut enc = render.create_command_encoder(&CommandEncoderDescriptor { label: None });
            filter.encode(
                &mut enc,
                [&groups[0], &groups[1]],
                [s; 2],
                [&targets[0].1, &targets[1].1],
            );
            dev.queue().submit([enc.finish()]);
        }
        let out = read_back(&dev, &render, &targets[0].0, (w, h));
        drop(groups);
        let messages = dev.take_validation_messages();
        assert!(messages.is_empty(), "validation messages: {messages:?}");
        out
    }

    /// Three captures `a`, `b`, `c` (`c` the newest) in the ring, with the newest in
    /// slot 2 and, after a fourth capture, in slot 0: the left image both times.
    fn median_of((w, h): (u32, u32), a: &Layer<'_>, b: &Layer<'_>, c: &Layer<'_>) -> [Vec<f32>; 2] {
        [
            left_after(
                w,
                h,
                &[
                    (a, step(0, Read::Newest)),
                    (b, step(1, Read::Newest)),
                    (c, step(2, Read::Median)),
                ],
            ),
            left_after(
                w,
                h,
                &[
                    (a, step(0, Read::Newest)),
                    (a, step(1, Read::Newest)),
                    (b, step(2, Read::Median)),
                    (c, step(0, Read::Median)),
                ],
            ),
        ]
    }

    /// Asserts that `got` is `want(x, y)` (as `D16_UNORM`) at every texel of `w` × `h`.
    fn assert_image(got: &[f32], (w, h): (u32, u32), want: &dyn Fn(u32, u32) -> u16, what: &str) {
        for y in 0..h {
            for x in 0..w {
                let (g, e) = (got[(y * w + x) as usize], f32::from(want(x, y)) / 65535.0);
                assert!(
                    (g - e).abs() <= 1e-6,
                    "{what}, texel ({x}, {y}): {g} against {e}"
                );
            }
        }
    }

    const FAR: u16 = 40000;
    const NEAR: u16 = 10000;

    /// A 5 × 5 block at x, y in 1..=5 of an 8 × 7 image (a 7 × 7 layer, 98 bytes, would
    /// put the second layer off the 4-byte offset Vulkan asks of a depth upload).
    fn in_block(x: u32, y: u32) -> bool {
        (1..=5).contains(&x) && (1..=5).contains(&y)
    }

    /// Q-363 (#607): a uniform region that changed depth in the newest capture (far, far,
    /// near over a 5 × 5 block): its interior, whose newest 3 × 3 neighbourhood is all
    /// near, takes the near depth at once; the block's edge texels and their outside
    /// neighbours, whose neighbourhood spans both depths, keep the median (far).
    #[test]
    #[ignore = "gpu"]
    fn gpu_median_pass_clamp_takes_a_changed_region_at_once() {
        let size = (8, 7);
        let far = |_, _| FAR;
        let near = |x, y| if in_block(x, y) { NEAR } else { FAR };
        let interior = |x, y| {
            if (2..=4).contains(&x) && (2..=4).contains(&y) {
                NEAR
            } else {
                FAR
            }
        };
        for (i, l) in median_of(size, &far, &far, &near).iter().enumerate() {
            assert_image(l, size, &interior, &format!("run {i}"));
        }
    }

    /// Q-363 (#607): at a silhouette (near left of x = 4, far from it), texels of the near
    /// side's last column flipped to far in the newest capture, and of the far side's
    /// first column flipped to near in the oldest: the newest neighbourhood of a flipped
    /// texel spans both sides, so the median passes and both flips are rejected.
    #[test]
    #[ignore = "gpu"]
    fn gpu_median_pass_clamp_keeps_the_median_at_a_silhouette() {
        let size = (8, 5);
        let steady = |x, _| if x < 4 { NEAR } else { FAR };
        let oldest = |x, y| {
            if x == 4 && y % 2 == 0 {
                NEAR
            } else {
                steady(x, y)
            }
        };
        let newest = |x, y| {
            if x == 3 && y % 2 == 1 {
                FAR
            } else {
                steady(x, y)
            }
        };
        for (i, l) in median_of(size, &oldest, &steady, &newest)
            .iter()
            .enumerate()
        {
            assert_image(l, size, &steady, &format!("run {i}"));
        }
    }

    /// Q-363 (#607), Q-340: a 5 × 5 block that became invalid (d = 1) in the newest
    /// capture: its interior is d = 1 at once, though valid in 2 of 3 captures.
    #[test]
    #[ignore = "gpu"]
    fn gpu_median_pass_clamp_makes_a_newly_invalid_block_invalid_at_once() {
        let size = (8, 7);
        let valid = |_, _| FAR;
        let invalid = |x, y| if in_block(x, y) { u16::MAX } else { FAR };
        for (i, l) in median_of(size, &valid, &valid, &invalid).iter().enumerate() {
            let interior = (3 * 8 + 3) as usize;
            assert_eq!(l[interior], 1.0, "run {i}: the block's centre");
            assert_image(
                l,
                size,
                &|x, y| {
                    if (2..=4).contains(&x) && (2..=4).contains(&y) {
                        u16::MAX
                    } else {
                        FAR
                    }
                },
                &format!("run {i}"),
            );
        }
    }

    /// Q-363 (#607): the neighbourhood of a texel on the image's border holds only the
    /// texels inside the image: with near, near, far everywhere, every texel, the edges
    /// and corners included, takes the far depth (a texel outside read as 0 would lower
    /// the range and let the near median through).
    #[test]
    #[ignore = "gpu"]
    fn gpu_median_pass_clamp_reads_only_texels_inside_the_image_at_its_border() {
        let size = (5, 4);
        let near = |_, _| NEAR;
        let far = |_, _| FAR;
        for (i, l) in median_of(size, &near, &near, &far).iter().enumerate() {
            assert_image(l, size, &far, &format!("run {i}"));
        }
    }
}
