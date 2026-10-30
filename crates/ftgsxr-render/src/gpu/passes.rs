// SPDX-License-Identifier: Apache-2.0

//! `SplatPasses` (§2.1): the pipelines of §1.4's stages, compiled once per (formats, tier).
//! This module holds stages 1–3 (prepare, cull, size) of `encode_prepare`, which then runs
//! stage 4's sort (`sort.rs`), the splat draw of stage 5 in `encode_draw`, and stage 8's
//! `encode_stats`; the centers draw and the resolve join in #116.

use std::sync::atomic::Ordering;

use bevy::render::render_resource::{BufferDescriptor, BufferUsages};
use ftgsxr_core::{Camera, DMat3, DVec3, Mat3, Vec3};

use super::device::SplatDevice;
use super::error::RenderError;
use super::frame::{
    FrameBuffers, RecordMode, SLOT_FREE, SLOT_PENDING, SLOT_READY, STATS_BYTES, TIMESTAMPS,
};
use super::inputs::{ColorTransform, ModelDraw, OcclusionInput, RenderMode, SortPose, ViewParams};
use super::model::MODEL_UNIFORM_BYTES;
use super::sort::{RadixSort, SORT_BLOCK};
use super::tier::QualityTier;

/// `cull.wgsl`: stages 2 and 3 (§1.4).
pub(crate) const CULL_WGSL: &str = include_str!("shaders/cull.wgsl");
/// `splat.wgsl`: stage 5's splat draw (§1.4).
pub(crate) const SPLAT_WGSL: &str = include_str!("shaders/splat.wgsl");
/// One instance's E7 constants in the draw (`splat.wgsl`'s `InstanceConstants`).
const DRAW_CONSTANTS_BYTES: usize = 16;
/// The draw's view block (`splat.wgsl`'s `DrawView`).
const DRAW_VIEW_BYTES: usize = 32;
/// The engine's near plane for the other geometry (§1.7: 0.01 m), which the splats' reverse-Z
/// depth test maps their depth with. // Q-T114.2
const ENGINE_NEAR: f32 = 0.01;

/// The cull's workgroup size (`cull.wgsl`).
const WORKGROUP: u32 = 256;
/// The most workgroups in one dimension of a dispatch (wgpu's default limit).
const MAX_GROUPS_PER_DIMENSION: u32 = 65_535;
/// One instance's uniform block (`cull.wgsl`'s `Instance`).
const INSTANCE_BYTES: usize = 128;
/// Instances sit 256 bytes apart, the largest uniform offset alignment wgpu allows.
const INSTANCE_STRIDE: usize = 256;
/// The frame's uniform block (`cull.wgsl`'s `Frame`).
const FRAME_BYTES: usize = 240;
/// One cull view in the frame block (`cull.wgsl`'s `ViewCam`).
const VIEW_BYTES: usize = 96;
/// The cull reads eight model buffers and writes four frame buffers (§2.4 GPU data layout).
const STORAGE_BUFFERS: u32 = 12;

const FLAG_TRANSFORM: u32 = 1;
const FLAG_COUNT_ACTIVE: u32 = 2;

/// Which cull entry point an instance's dispatch runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Splats,
    Centers,
}

/// The pipelines and layouts of §1.4's passes (§2.1), compiled once per (formats, tier).
pub struct SplatPasses {
    device: wgpu::Device,
    mode: RecordMode,
    depth_format: Option<wgpu::TextureFormat>,
    model_layout: wgpu::BindGroupLayout,
    frame_layout: wgpu::BindGroupLayout,
    instance_layout: wgpu::BindGroupLayout,
    size_layout: wgpu::BindGroupLayout,
    cull_splats: wgpu::ComputePipeline,
    cull_centers: wgpu::ComputePipeline,
    size: wgpu::ComputePipeline,
    sort: RadixSort,
    draw_layout: wgpu::BindGroupLayout,
    splat: wgpu::RenderPipeline,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, dynamic: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: None,
        },
        count: None,
    }
}

impl SplatPasses {
    /// Compiles the passes for `tier`'s record layout (§1.4) and the given layer and depth
    /// formats; a shader that fails to compile, or a device with fewer than the twelve
    /// storage buffers per stage the cull binds, is `RenderError::Shader`.
    pub fn new(
        dev: &SplatDevice,
        layer_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        tier: &QualityTier,
    ) -> Result<SplatPasses, RenderError> {
        let limit = dev.limits().max_storage_buffers_per_shader_stage;
        if limit < STORAGE_BUFFERS {
            return Err(RenderError::Shader(format!(
                "the cull pass binds {STORAGE_BUFFERS} storage buffers per stage; the device allows {limit}"
            )));
        }
        let device = dev.device().clone();
        let mode = RecordMode::of(tier);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ftgsxr cull.wgsl"),
            source: wgpu::ShaderSource::Wgsl(CULL_WGSL.into()),
        });
        let model_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ftgsxr model"),
            entries: &[
                uniform_entry(0, false),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, true),
                storage_entry(5, true),
                storage_entry(6, true),
                storage_entry(7, true),
                storage_entry(8, true),
            ],
        });
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ftgsxr frame"),
            entries: &[
                storage_entry(0, false),
                storage_entry(1, false),
                storage_entry(2, false),
                storage_entry(3, false),
                uniform_entry(4, false),
            ],
        });
        let instance_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ftgsxr instance"),
            entries: &[uniform_entry(0, true)],
        });
        let size_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ftgsxr size"),
            entries: &[
                storage_entry(10, false),
                storage_entry(11, false),
                storage_entry(12, false),
                uniform_entry(13, false),
            ],
        });
        let cull_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ftgsxr cull"),
            bind_group_layouts: &[
                Some(&model_layout),
                Some(&frame_layout),
                Some(&instance_layout),
            ],
            immediate_size: 0,
        });
        let size_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ftgsxr size"),
            bind_group_layouts: &[Some(&size_layout)],
            immediate_size: 0,
        });
        let constants = [
            ("RECORD_MODE", f64::from(mode.shader_value())),
            ("SORT_BLOCK", f64::from(SORT_BLOCK)),
        ];
        let pipeline = |label: &str, layout: &wgpu::PipelineLayout, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            })
        };
        let cull_splats = pipeline("ftgsxr cull splats", &cull_layout, "cull_splats");
        let cull_centers = pipeline("ftgsxr cull centers", &cull_layout, "cull_centers");
        let size = pipeline("ftgsxr size", &size_pipeline_layout, "size");
        let sort = RadixSort::new(&device, tier.sort_key);
        let (draw_layout, splat) = splat_pipeline(&device, layer_format, depth_format);
        if let Some(e) = bevy::tasks::block_on(scope.pop()) {
            return Err(RenderError::Shader(e.to_string()));
        }
        Ok(SplatPasses {
            device,
            mode,
            depth_format,
            model_layout,
            frame_layout,
            instance_layout,
            size_layout,
            cull_splats,
            cull_centers,
            size,
            sort,
            draw_layout,
            splat,
        })
    }

    /// Stages 1–4 of §1.4 for every model at `sort_pose`; `cull_views` holds the view (per view) or both eyes (shared center): a primitive passes when inside either frustum.
    ///
    /// Stage 1 on the CPU: each instance's uniform block (its pose, τ_m − τ₀ as float32,
    /// the SH degree, E14's transform, the mode) and its dispatch ranges from the temporal
    /// index (E3). Stage 2: one cull dispatch per instance and kind, splats before centers,
    /// so the centers region follows the splats (Q-78). Stage 3: one workgroup clamps
    /// the counts to C and writes the sort's dispatch arguments. Stage 4: the radix sort
    /// of the splats' (k̄, slot) pairs, its own compute pass, every dispatch indirect; the
    /// sorted slots end in the first payload buffer (D-33). In per-view mode E4's eye
    /// and E8's depth are the camera's of `cull_views[0]` (`sort_pose` is not read); in
    /// shared-center mode they are `sort_pose`'s (D-4). Nothing is read back (D-34).
    ///
    /// # Panics
    ///
    /// When `cull_views` is empty or holds more than two views, or when `frame` was made
    /// for another tier's record layout.
    pub fn encode_prepare(
        &self,
        enc: &mut wgpu::CommandEncoder,
        frame: &FrameBuffers,
        models: &[ModelDraw],
        sort_pose: &SortPose,
        cull_views: &[ViewParams],
    ) {
        assert!(
            (1..=2).contains(&cull_views.len()),
            "one or two cull views (§1.4 Stereo)"
        );
        assert_eq!(
            frame.mode, self.mode,
            "the frame buffers' record layout is the passes' tier's"
        );
        let set = &frame.set;
        enc.clear_buffer(&set.indirect, 0, None);

        let eye = match self.mode {
            RecordMode::PerView => camera_center(&cull_views[0].camera),
            RecordMode::Shared | RecordMode::SharedHalf => sort_pose.origin,
        };
        let frame_block = frame_uniform(cull_views, frame.capacity(), sort_pose);
        let frame_uniform = self.uniform_buffer("ftgsxr frame uniform", &frame_block);

        // Stage 1: splat dispatches first, then centers (Q-78).
        let mut dispatches = Vec::new();
        for kind in [Kind::Splats, Kind::Centers] {
            for (i, draw) in models.iter().enumerate() {
                let (splats, centers) = match draw.mode {
                    RenderMode::Splats => (true, false),
                    RenderMode::Centers { .. } => (false, true),
                    RenderMode::Both { .. } => (true, true),
                    RenderMode::Hidden => (false, false),
                };
                let wanted = match kind {
                    Kind::Splats => splats,
                    Kind::Centers => centers,
                };
                // Q-79: a row's E3 pass is counted once, by its instance's splat
                // dispatch, or by its centers dispatch for a centers-only instance.
                let count_active = match kind {
                    Kind::Splats => true,
                    Kind::Centers => !splats,
                };
                if wanted {
                    dispatches.push((i, kind, count_active));
                }
            }
        }
        let mut blocks = vec![0u8; dispatches.len().max(1) * INSTANCE_STRIDE];
        let mut groups = Vec::with_capacity(dispatches.len());
        for (k, &(i, kind, count_active)) in dispatches.iter().enumerate() {
            let draw = &models[i];
            let range = draw.model.index.range(draw.tau);
            let rows = range.persistent.len() as u32 + range.transient.len() as u32;
            let block = instance_uniform(draw, i as u32, eye, &range, count_active);
            blocks[k * INSTANCE_STRIDE..k * INSTANCE_STRIDE + INSTANCE_BYTES]
                .copy_from_slice(&block);
            groups.push((kind, i, k, dispatch_groups(rows)));
        }
        let instances = self.uniform_buffer("ftgsxr instance uniforms", &blocks);
        // E7's constants per instance for the draw (D-39), indexed by the records' instance.
        // Q-T114.2
        let mut constants = vec![0u8; models.len().max(1) * DRAW_CONSTANTS_BYTES];
        for (i, draw) in models.iter().enumerate() {
            let c = &draw.model.conventions;
            let mut b = Block {
                bytes: &mut constants[i * DRAW_CONSTANTS_BYTES..(i + 1) * DRAW_CONSTANTS_BYTES],
                at: 0,
            };
            b.f32(c.alpha_max);
            b.f32(c.alpha_skip);
            b.f32(c.pixel_center_offset);
        }
        let constants =
            self.buffer_with("ftgsxr draw constants", &constants, BufferUsages::STORAGE);
        *frame
            .draw_constants
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(constants);
        let instance_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ftgsxr instance"),
            layout: &self.instance_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &instances,
                    offset: 0,
                    size: wgpu::BufferSize::new(INSTANCE_BYTES as u64),
                }),
            }],
        });
        let frame_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ftgsxr frame"),
            layout: &self.frame_layout,
            entries: &[
                buffer_entry(0, &set.records),
                buffer_entry(1, &set.keys[0]),
                buffer_entry(2, &set.payloads[0]),
                buffer_entry(3, &set.indirect),
                buffer_entry(4, &frame_uniform),
            ],
        });
        let size_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ftgsxr size"),
            layout: &self.size_layout,
            entries: &[
                buffer_entry(10, &set.indirect),
                buffer_entry(11, &set.dispatch),
                buffer_entry(12, &set.stats),
                buffer_entry(13, &frame_uniform),
            ],
        });
        let model_groups: Vec<wgpu::BindGroup> = models
            .iter()
            .map(|draw| {
                let b = &draw.model.buffers;
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("ftgsxr model"),
                    layout: &self.model_layout,
                    entries: &[
                        buffer_entry(0, &b.uniform),
                        buffer_entry(1, &b.position),
                        buffer_entry(2, &b.time_center),
                        buffer_entry(3, &b.duration),
                        buffer_entry(4, &b.velocity),
                        buffer_entry(5, &b.scale),
                        buffer_entry(6, &b.rotation),
                        buffer_entry(7, &b.opacity),
                        buffer_entry(8, &b.sh),
                    ],
                })
            })
            .collect();

        let timestamp_writes =
            set.queries
                .as_ref()
                .map(|(queries, _)| wgpu::ComputePassTimestampWrites {
                    query_set: queries,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ftgsxr prepare"),
                timestamp_writes,
            });
            pass.set_bind_group(1, &frame_group, &[]);
            for &(kind, i, k, (x, y)) in &groups {
                if x == 0 {
                    continue;
                }
                pass.set_pipeline(match kind {
                    Kind::Splats => &self.cull_splats,
                    Kind::Centers => &self.cull_centers,
                });
                pass.set_bind_group(0, &model_groups[i], &[]);
                pass.set_bind_group(1, &frame_group, &[]);
                pass.set_bind_group(2, &instance_group, &[(k * INSTANCE_STRIDE) as u32]);
                pass.dispatch_workgroups(x, y, 1);
            }
            // Stage 3.
            pass.set_pipeline(&self.size);
            pass.set_bind_group(0, &size_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        // Stage 4.
        let timestamp_writes =
            set.queries
                .as_ref()
                .map(|(queries, _)| wgpu::ComputePassTimestampWrites {
                    query_set: queries,
                    beginning_of_pass_write_index: Some(2),
                    end_of_pass_write_index: Some(3),
                });
        self.sort.encode(enc, set, timestamp_writes);
        if set.queries.is_some() {
            frame.ring.written.store(4, Ordering::Release);
        }
    }

    /// Stages 5 and 6 into the file-space layer, depth-tested against `depth` when given.
    ///
    /// Stage 5's splats (`splat.wgsl`): one indirect instanced draw of four-vertex strips
    /// from stage 3's arguments, in the sort's order (pair 0's payloads), blended
    /// `One, OneMinusSrcAlpha` on color and alpha into the pass's layer, which the caller
    /// cleared to (0, 0, 0, 0). With the passes' depth format the splats test against the
    /// pass's opaque depth (reverse Z, Q-T114.2) and write none. E13's weight is 1: every
    /// M-2 tier has occlusion off, and `occlusion` is not read until M-6 (R-12). In
    /// shared-center mode nothing is drawn until the per-eye projection of M-8 (R-9,
    /// Q-T114.2). The draw writes no timestamps: the pass is the caller's (Q-T114.2). The
    /// centers of stage 6 join in #116.
    ///
    /// # Panics
    ///
    /// When `frame` was made for another tier's record layout, or `encode_prepare` has not
    /// run on it.
    pub fn encode_draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        frame: &FrameBuffers,
        view: &ViewParams,
        occlusion: Option<&OcclusionInput>,
    ) {
        assert_eq!(
            frame.mode, self.mode,
            "the frame buffers' record layout is the passes' tier's"
        );
        let _ = occlusion; // M-6 (R-12): E13's environment depth.
        if self.mode != RecordMode::PerView {
            // Q-T114.2: the shared-center vertex projection is M-8's (R-9).
            return;
        }
        let set = &frame.set;
        let constants = frame
            .draw_constants
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .expect("encode_prepare runs before encode_draw");
        let block = draw_view_uniform(view, self.depth_format.is_some());
        let view_buffer = self.uniform_buffer("ftgsxr draw view", &block);
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ftgsxr draw"),
            layout: &self.draw_layout,
            entries: &[
                buffer_entry(0, &set.records),
                buffer_entry(1, &set.payloads[0]),
                buffer_entry(2, &constants),
                buffer_entry(3, &view_buffer),
            ],
        });
        pass.set_pipeline(&self.splat);
        pass.set_bind_group(0, &group, &[]);
        pass.draw_indirect(&set.indirect, 0);
    }

    /// Stage 8: the statistics block to the staging buffer; `FrameStats::poll` reads it two frames later.
    ///
    /// The frame's timestamps are resolved into the block first. Each frame takes the next
    /// of three staging buffers, mapped when the encoder's submission completes; a buffer
    /// still waiting for its mapping skips this frame's copy, and one mapped but never
    /// polled is unmapped and reused (Q-81). Nothing waits (P-5).
    pub fn encode_stats(&self, enc: &mut wgpu::CommandEncoder, frame: &FrameBuffers) {
        let set = &frame.set;
        let ring = &frame.ring;
        let written = ring.written.swap(0, Ordering::AcqRel).min(TIMESTAMPS);
        if let Some((queries, resolve)) = &set.queries
            && written > 0
        {
            enc.resolve_query_set(queries, 0..written, resolve, 0);
            enc.copy_buffer_to_buffer(resolve, 0, &set.stats, 16, u64::from(written) * 8);
        }
        let n = ring.encoded.fetch_add(1, Ordering::AcqRel);
        let slot = (n % ring.state.len() as u64) as usize;
        let state = &ring.state[slot];
        match state.load(Ordering::Acquire) {
            SLOT_READY => {
                set.staging[slot].unmap();
                state.store(SLOT_FREE, Ordering::Release);
            }
            SLOT_PENDING => {
                ring.frame[slot].store(u64::MAX, Ordering::Release);
                return;
            }
            _ => {}
        }
        enc.copy_buffer_to_buffer(&set.stats, 0, &set.staging[slot], 0, STATS_BYTES);
        state.store(SLOT_PENDING, Ordering::Release);
        ring.frame[slot].store(n, Ordering::Release);
        let flag = state.clone();
        enc.map_buffer_on_submit(&set.staging[slot], wgpu::MapMode::Read, .., move |r| {
            flag.store(
                if r.is_ok() { SLOT_READY } else { SLOT_FREE },
                Ordering::Release,
            );
        });
    }

    /// A uniform buffer holding `bytes`, written through its mapping at creation, so each
    /// call has its own (no queue write races an earlier, unsubmitted encoder).
    fn uniform_buffer(&self, label: &str, bytes: &[u8]) -> wgpu::Buffer {
        self.buffer_with(label, bytes, BufferUsages::UNIFORM)
    }

    /// A buffer of `usage` holding `bytes`, written through its mapping at creation.
    fn buffer_with(&self, label: &str, bytes: &[u8], usage: BufferUsages) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage,
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(bytes);
        buffer.unmap();
        buffer
    }
}

/// Stage 5's splat pipeline and its bind group layout (`splat.wgsl`): the records, the
/// sorted payloads and the instances' E7 constants as read-only storage, the view block.
fn splat_pipeline(
    device: &wgpu::Device,
    layer_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
) -> (wgpu::BindGroupLayout, wgpu::RenderPipeline) {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ftgsxr splat.wgsl"),
        source: wgpu::ShaderSource::Wgsl(SPLAT_WGSL.into()),
    });
    let stages = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
    let storage = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: stages,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ftgsxr draw"),
        entries: &[
            storage(0),
            storage(1),
            storage(2),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: stages,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ftgsxr splat"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    // §1.4 stage 5: the premultiplied over operator on color and alpha.
    let over = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    };
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ftgsxr splat"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_splat"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            cull_mode: None,
            ..Default::default()
        },
        // The depth test against the pass's opaque depth, no depth write (Q-T114.2).
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_splat"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: layer_format,
                blend: Some(wgpu::BlendState {
                    color: over,
                    alpha: over,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });
    (layout, pipeline)
}

/// The draw's view block (`splat.wgsl`'s `DrawView`): the camera's W × H, the target's
/// size, the engine's near plane when depth-tested (else 0), and E13's switches (off).
fn draw_view_uniform(view: &ViewParams, depth_test: bool) -> [u8; DRAW_VIEW_BYTES] {
    let mut out = [0u8; DRAW_VIEW_BYTES];
    let mut b = Block {
        bytes: &mut out,
        at: 0,
    };
    b.f32(view.camera.width as f32);
    b.f32(view.camera.height as f32);
    b.f32(view.target_size.x as f32);
    b.f32(view.target_size.y as f32);
    b.f32(if depth_test { ENGINE_NEAR } else { 0.0 });
    b.u32(0);
    b.u32(0);
    b.f32(0.0);
    out
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// The workgroups that cover `rows` rows: x up to 65,535, y the rest (`cull.wgsl`'s
/// `row_of` linearizes them).
fn dispatch_groups(rows: u32) -> (u32, u32) {
    let groups = rows.div_ceil(WORKGROUP);
    if groups == 0 {
        return (0, 0);
    }
    let x = groups.min(MAX_GROUPS_PER_DIMENSION);
    (x, groups.div_ceil(x))
}

/// A camera's center in stage space, −R_wᵀ t_w (E4's eye of a view).
fn camera_center(camera: &Camera) -> Vec3 {
    let r = DMat3::from_mat4(camera.world_to_camera);
    let t = camera.world_to_camera.w_axis.truncate();
    (-(r.transpose() * t)).as_vec3()
}

/// Little-endian words into a byte block.
struct Block<'a> {
    bytes: &'a mut [u8],
    at: usize,
}

impl Block<'_> {
    fn f32(&mut self, v: f32) {
        self.bytes[self.at..self.at + 4].copy_from_slice(&v.to_le_bytes());
        self.at += 4;
    }
    fn u32(&mut self, v: u32) {
        self.bytes[self.at..self.at + 4].copy_from_slice(&v.to_le_bytes());
        self.at += 4;
    }
    /// A WGSL `vec3<f32>` followed by one scalar (16 bytes).
    fn vec3(&mut self, v: Vec3) {
        self.f32(v.x);
        self.f32(v.y);
        self.f32(v.z);
    }
    /// A WGSL `mat3x3<f32>`: three columns of 16 bytes.
    fn mat3(&mut self, m: Mat3) {
        for c in [m.x_axis, m.y_axis, m.z_axis] {
            self.vec3(c);
            self.f32(0.0);
        }
    }
}

/// One instance's uniform block (`cull.wgsl`'s `Instance`, §1.4 stage 1): R(q_m), t_m,
/// s_m, E4's eye in model space o_{v,m} = P_m⁻¹(o_v), τ_m − τ₀ in float32 (computed in
/// float64, then rounded, E2), E14's gains and flags, the dispatch range (E3), the SH
/// degree min(requested, L′), the instance's index and the centers' radius.
fn instance_uniform(
    draw: &ModelDraw,
    instance: u32,
    eye: Vec3,
    range: &ftgsxr_core::DispatchRange,
    count_active: bool,
) -> [u8; INSTANCE_BYTES] {
    let pose = draw.pose;
    let rotation = Mat3::from_quat(pose.rotation);
    let eye_model = (pose.rotation.inverse() * (eye - pose.translation)) / pose.scale;
    let transform = draw.transform != ColorTransform::IDENTITY;
    // Q-80: the identity transform leaves the E4 color as it is.
    let flags =
        (u32::from(transform) * FLAG_TRANSFORM) | (u32::from(count_active) * FLAG_COUNT_ACTIVE);
    let pixel_radius = match draw.mode {
        RenderMode::Centers { pixel_radius, .. } | RenderMode::Both { pixel_radius, .. } => {
            pixel_radius
        }
        RenderMode::Splats | RenderMode::Hidden => 0.0,
    };
    let mut out = [0u8; INSTANCE_BYTES];
    let mut b = Block {
        bytes: &mut out,
        at: 0,
    };
    b.mat3(rotation);
    b.vec3(pose.translation);
    b.f32(pose.scale);
    b.vec3(eye_model);
    b.f32((draw.tau - draw.model.tau0) as f32);
    b.vec3(draw.transform.gain);
    b.u32(flags);
    b.u32(range.persistent.end);
    b.u32(range.transient.start);
    b.u32(range.persistent.len() as u32 + range.transient.len() as u32);
    b.u32(u32::from(draw.sh_degree.min(draw.model.sh_degree)));
    b.u32(instance);
    b.f32(pixel_radius);
    debug_assert_eq!(b.at + 8, INSTANCE_BYTES);
    out
}

/// The frame's uniform block (`cull.wgsl`'s `Frame`): up to two cull views as R_w, t_w,
/// W, f, c, H in float32 (E6), their count, C and the shared sort pose (D-4).
fn frame_uniform(views: &[ViewParams], capacity: u32, sort_pose: &SortPose) -> [u8; FRAME_BYTES] {
    let mut out = [0u8; FRAME_BYTES];
    let mut b = Block {
        bytes: &mut out,
        at: 0,
    };
    for i in 0..2 {
        let start = b.at;
        if let Some(view) = views.get(i) {
            let cam = &view.camera;
            let r = DMat3::from_mat4(cam.world_to_camera).as_mat3();
            let t: DVec3 = cam.world_to_camera.w_axis.truncate();
            b.mat3(r);
            b.vec3(t.as_vec3());
            b.f32(cam.width as f32);
            b.f32(cam.k.x_axis.x as f32);
            b.f32(cam.k.y_axis.y as f32);
            b.f32(cam.k.z_axis.x as f32);
            b.f32(cam.k.z_axis.y as f32);
            b.f32(cam.height as f32);
        }
        b.at = start + VIEW_BYTES;
    }
    b.u32(views.len() as u32);
    b.u32(capacity);
    b.at += 8;
    b.vec3(sort_pose.origin);
    b.at += 4;
    b.vec3(sort_pose.forward);
    debug_assert_eq!(b.at + 4, FRAME_BYTES);
    out
}

// The model's uniform block is `model.rs`'s; `cull.wgsl` reads it as 48 bytes.
const _: () = assert!(MODEL_UNIFORM_BYTES == 48);

#[cfg(test)]
mod tests {
    use super::*;
    use ftgsxr_core::DMat4;

    /// `cull.wgsl` parses and validates under naga, with every capability wgpu's Vulkan
    /// backend may offer (the device check is `SplatPasses::new`'s).
    #[test]
    fn cull_wgsl_is_valid_wgsl() {
        let module = naga::front::wgsl::parse_str(CULL_WGSL).expect("cull.wgsl parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("cull.wgsl validates");
        for entry in ["cull_splats", "cull_centers", "size"] {
            assert!(
                module.entry_points.iter().any(|e| e.name == entry),
                "entry point {entry}"
            );
        }
    }

    /// `splat.wgsl` parses and validates under naga, with both entry points.
    #[test]
    fn splat_wgsl_is_valid_wgsl() {
        let module = naga::front::wgsl::parse_str(SPLAT_WGSL).expect("splat.wgsl parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("splat.wgsl validates");
        for entry in ["vs_splat", "fs_splat"] {
            assert!(
                module.entry_points.iter().any(|e| e.name == entry),
                "entry point {entry}"
            );
        }
    }

    #[test]
    fn draw_view_uniform_lays_out_the_sizes_and_the_depth_near() {
        let cam = Camera {
            width: 64,
            height: 48,
            k: DMat3::IDENTITY,
            world_to_camera: DMat4::IDENTITY,
        };
        let view = ViewParams {
            camera: cam,
            target_size: ftgsxr_core::UVec2::new(128, 96),
        };
        let f = |b: &[u8], at: usize| f32::from_le_bytes(b[at..at + 4].try_into().unwrap());
        let b = draw_view_uniform(&view, true);
        assert_eq!(
            [f(&b, 0), f(&b, 4), f(&b, 8), f(&b, 12), f(&b, 16)],
            [64.0, 48.0, 128.0, 96.0, ENGINE_NEAR]
        );
        assert_eq!(&b[20..32], &[0u8; 12]);
        assert_eq!(f(&draw_view_uniform(&view, false), 16), 0.0);
    }

    #[test]
    fn dispatch_groups_cover_the_rows() {
        assert_eq!(dispatch_groups(0), (0, 0));
        assert_eq!(dispatch_groups(1), (1, 1));
        assert_eq!(dispatch_groups(256), (1, 1));
        assert_eq!(dispatch_groups(257), (2, 1));
        let rows = 65_535 * 256 + 1;
        let (x, y) = dispatch_groups(rows);
        assert_eq!((x, y), (65_535, 2));
        assert!(u64::from(x) * u64::from(y) * 256 >= u64::from(rows));
    }

    #[test]
    fn camera_center_is_minus_r_transpose_t() {
        // A camera at (1, 2, 3) looking along +z with OpenCV axes: R_w = I, t_w = −p.
        let mut w2c = DMat4::IDENTITY;
        w2c.w_axis = ftgsxr_core::DVec3::new(-1.0, -2.0, -3.0).extend(1.0);
        let cam = Camera {
            width: 4,
            height: 4,
            k: DMat3::IDENTITY,
            world_to_camera: w2c,
        };
        assert_eq!(camera_center(&cam), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn frame_uniform_lays_out_the_views_and_the_sort_pose() {
        let mut w2c = DMat4::IDENTITY;
        w2c.w_axis = ftgsxr_core::DVec3::new(0.5, -1.0, 2.0).extend(1.0);
        let cam = Camera {
            width: 64,
            height: 48,
            k: DMat3::from_cols(
                DVec3::new(60.0, 0.0, 0.0),
                DVec3::new(0.0, 61.0, 0.0),
                DVec3::new(32.0, 24.0, 1.0),
            ),
            world_to_camera: w2c,
        };
        let view = ViewParams {
            camera: cam,
            target_size: ftgsxr_core::UVec2::new(64, 48),
        };
        let pose = SortPose {
            origin: Vec3::new(7.0, 8.0, 9.0),
            forward: Vec3::new(0.0, 0.0, 1.0),
        };
        let bytes = frame_uniform(&[view], 1000, &pose);
        let f = |at: usize| f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let u = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        // R_w = I as three 16-byte columns.
        assert_eq!([f(0), f(20), f(40)], [1.0, 1.0, 1.0]);
        assert_eq!([f(48), f(52), f(56), f(60)], [0.5, -1.0, 2.0, 64.0]);
        assert_eq!(
            [f(64), f(68), f(72), f(76), f(80)],
            [60.0, 61.0, 32.0, 24.0, 48.0]
        );
        assert_eq!([u(192), u(196)], [1, 1000]);
        assert_eq!([f(208), f(212), f(216)], [7.0, 8.0, 9.0]);
        assert_eq!([f(224), f(228), f(232)], [0.0, 0.0, 1.0]);
    }
}
