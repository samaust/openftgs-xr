// SPDX-License-Identifier: Apache-2.0

//! Shared helpers of the display tests (R-11, R-13, R-14): hand-built models, one frame of
//! §1.4 stages 1–6 into a 16-bit float layer and, when asked, stage 7's resolve onto an
//! sRGB view target, with the layer, the target and the depth read back.

#![allow(dead_code, reason = "each test file uses a part of the helpers")]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ftgsxr_core::{Camera, Model, ModelKind, RenderConventions, TimeSpan, UVec2};
use ftgsxr_render::{FrameBuffers, ModelDraw, QualityTier, SplatDevice, SplatPasses, ViewParams};

/// E4's C₀.
pub const SH_C0: f32 = 0.282_094_8;

/// The probes' camera (§3.2): 64 × 48, focal 60, at (0, −4, 0) looking along +y.
pub fn probe_camera() -> Camera {
    // R_w maps x_c = x, y_c = −z, z_c = y; t_w = (0, 0, 4).
    let r = ftgsxr_core::DMat3::from_cols(
        ftgsxr_core::DVec3::new(1.0, 0.0, 0.0),
        ftgsxr_core::DVec3::new(0.0, 0.0, 1.0),
        ftgsxr_core::DVec3::new(0.0, -1.0, 0.0),
    );
    let mut w2c = ftgsxr_core::DMat4::from_mat3(r);
    w2c.w_axis = ftgsxr_core::DVec3::new(0.0, 0.0, 4.0).extend(1.0);
    Camera {
        width: 64,
        height: 48,
        k: ftgsxr_core::DMat3::from_cols(
            ftgsxr_core::DVec3::new(60.0, 0.0, 0.0),
            ftgsxr_core::DVec3::new(0.0, 60.0, 0.0),
            ftgsxr_core::DVec3::new(32.0, 24.0, 1.0),
        ),
        world_to_camera: w2c,
    }
}

/// One row of a hand-built degree-0 model.
#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub time_center: f64,
    pub duration: f64,
    pub scale: f32,
    pub opacity: f32,
    /// The DC color C₀c₀₀ + 0.5 the row shows in every direction (E4).
    pub color: [f32; 3],
}

/// A degree-0 model of `rows` over τ ∈ [0, 1.9] (the probes' span), isotropic and
/// unrotated, with openftgs's conventions.
pub fn model(rows: &[Row]) -> Model {
    let mut m = Model {
        position: vec![],
        time_center: vec![],
        duration: vec![],
        velocity: vec![],
        scale: vec![],
        rotation: vec![],
        opacity: vec![],
        sh: vec![],
        sh_degree: 0,
        time: TimeSpan {
            tau0: 0.0,
            tau1: 1.9,
            time_unit: "s".to_owned(),
        },
        conventions: RenderConventions::default(),
        producer: String::new(),
        identity: [0; 32],
        kind: ModelKind::ExchangeFile,
    };
    for r in rows {
        m.position.push(r.position);
        m.velocity.push(r.velocity);
        m.time_center.push(r.time_center);
        m.duration.push(r.duration);
        m.scale.push([r.scale; 3]);
        m.rotation.push([1.0, 0.0, 0.0, 0.0]);
        m.opacity.push(r.opacity);
        m.sh.extend(r.color.map(|c| (c - 0.5) / SH_C0));
    }
    m.validate().expect("a valid model");
    m
}

/// The layer's format (§1.4 stage 5).
pub const LAYER: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// The view target's format (Q-91).
pub const TARGET: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// The depth format of the depth-tested runs.
pub const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// What one frame left in its textures, row-major from the top row.
pub struct Frame {
    /// The layer's (C, A), from its half floats.
    pub layer: Vec<[f32; 4]>,
    /// The view target's bytes (sRGB-encoded), when the resolve ran.
    pub target: Option<Vec<[u8; 4]>>,
    /// The depth, when depth-tested.
    pub depth: Option<Vec<f32>>,
}

/// How the frame is drawn.
pub struct Run {
    /// Stage 7 with this background (`Some(None)`: passthrough, no background); `None`
    /// skips the resolve.
    pub resolve: Option<Option<[f32; 3]>>,
    /// Whether the draw pass has a depth attachment (cleared to 0, reverse Z's far).
    pub depth: bool,
}

fn texture(
    dev: &SplatDevice,
    format: wgpu::TextureFormat,
    size: UVec2,
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    dev.device().create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

/// One frame of `draws` from `camera` (per view, C = twice the draws' total N), drawn into a
/// fresh layer and, per `run`, resolved onto a fresh target cleared to (0, 0, 0, 0).
pub fn render(dev: &SplatDevice, draws: &[ModelDraw], camera: &Camera, run: &Run) -> Frame {
    let tier = QualityTier::conformance();
    let passes = SplatPasses::new(dev, LAYER, run.depth.then_some(DEPTH), &tier)
        .expect("the passes compile");
    // Room for every row's splat and center (Q-78: the centers share the C slots).
    let capacity: u32 = 2 * draws.iter().map(|d| d.model.n()).sum::<u32>();
    let frame = FrameBuffers::new(dev, &tier, capacity);
    let size = UVec2::new(camera.width, camera.height);
    let view = ViewParams {
        camera: *camera,
        target_size: size,
    };
    let attach = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC;
    let layer = texture(
        dev,
        LAYER,
        size,
        attach | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    let layer_view = layer.create_view(&Default::default());
    let depth = run.depth.then(|| texture(dev, DEPTH, size, attach));
    let depth_view = depth.as_ref().map(|t| t.create_view(&Default::default()));
    let target = run.resolve.map(|_| texture(dev, TARGET, size, attach));
    let target_view = target.as_ref().map(|t| t.create_view(&Default::default()));

    let mut enc = dev.device().create_command_encoder(&Default::default());
    passes.encode_prepare(&mut enc, &frame, draws, &sort_pose(camera), &[view]);
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test draw"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &layer_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: depth_view.as_ref().map(|v| {
                wgpu::RenderPassDepthStencilAttachment {
                    view: v,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        passes.encode_draw(&mut pass, &frame, &view, None);
    }
    if let (Some(background), Some(target_view)) = (run.resolve, target_view.as_ref()) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("test resolve"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        passes.encode_resolve(&mut pass, &layer_view, background);
    }
    let layer_bytes = copy_out(dev, &mut enc, &layer, 8, size);
    let target_bytes = target.as_ref().map(|t| copy_out(dev, &mut enc, t, 4, size));
    let depth_bytes = depth.as_ref().map(|t| copy_out(dev, &mut enc, t, 4, size));
    dev.queue().submit([enc.finish()]);

    let read = |(buffer, row): (wgpu::Buffer, u32)| -> Vec<u8> {
        let done = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&done);
        buffer.map_async(wgpu::MapMode::Read, .., move |r| {
            flag.store(r.is_ok(), Ordering::Release)
        });
        dev.device()
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("the device polls");
        assert!(done.load(Ordering::Acquire), "the read back maps");
        let bytes = buffer.get_mapped_range(..);
        let texel_row = (bytes.len() as u32 / size.y) as usize;
        let mut out = Vec::new();
        for y in 0..size.y as usize {
            out.extend_from_slice(&bytes[y * texel_row..][..row as usize]);
        }
        out
    };
    let layer = read(layer_bytes)
        .as_chunks::<8>()
        .0
        .iter()
        .map(|t| [0, 1, 2, 3].map(|i| half(u16::from_le_bytes([t[2 * i], t[2 * i + 1]]))))
        .collect();
    let target = target_bytes.map(|b| read(b).as_chunks::<4>().0.to_vec());
    let depth = depth_bytes.map(|b| {
        read(b)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| f32::from_le_bytes(*w))
            .collect()
    });
    let messages = dev.take_validation_messages();
    assert!(messages.is_empty(), "validation messages: {messages:?}");
    Frame {
        layer,
        target,
        depth,
    }
}

/// The camera's center and forward axis (the frame block's sort pose; per-view mode reads
/// the camera itself).
fn sort_pose(camera: &Camera) -> ftgsxr_render::SortPose {
    let r = ftgsxr_core::DMat3::from_mat4(camera.world_to_camera);
    let t = camera.world_to_camera.w_axis.truncate();
    ftgsxr_render::SortPose {
        origin: (-(r.transpose() * t)).as_vec3(),
        forward: (r.transpose() * ftgsxr_core::DVec3::Z).as_vec3(),
    }
}

/// A copy of `texture` into a mappable buffer: the buffer and a row's bytes.
fn copy_out(
    dev: &SplatDevice,
    enc: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    texel: u32,
    size: UVec2,
) -> (wgpu::Buffer, u32) {
    let row = size.x * texel;
    let padded = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = dev.device().create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(padded) * u64::from(size.y),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(size.y),
            },
        },
        wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
    );
    (buffer, row)
}

/// IEEE binary16 to float32.
pub fn half(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let frac = f32::from(h & 0x3ff);
    match exp {
        0 => sign * frac * 2f32.powi(-24),
        31 if frac == 0.0 => sign * f32::INFINITY,
        31 => f32::NAN,
        _ => sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// IEC 61966-2-1's sRGB decoding and encoding, by hand.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

pub fn linear_to_srgb(l: f32) -> f32 {
    if l <= 0.003_130_8 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

/// `model` uploaded under the conformance tier's layout (float32, its full degree).
pub fn upload(dev: &SplatDevice, model: &Model) -> ftgsxr_render::GpuModel {
    let (index, order) = ftgsxr_core::TemporalIndex::build(model);
    let arrays = ftgsxr_core::pack_for_gpu(model, &order, &QualityTier::conformance().layout);
    ftgsxr_render::GpuModel::upload(dev, &arrays, index, &model.conventions).expect("fits")
}

/// A draw of `model` at the identity pose at τ in `mode` with `transform`.
pub fn draw(
    model: &ftgsxr_render::GpuModel,
    tau: f64,
    mode: ftgsxr_render::RenderMode,
    transform: ftgsxr_render::ColorTransform,
) -> ModelDraw<'_> {
    ModelDraw {
        model,
        pose: ftgsxr_core::Pose::IDENTITY,
        tau,
        mode,
        transform,
        sh_degree: 0,
        overlays: ftgsxr_render::Overlays::default(),
    }
}

/// A flat-color probe: two coincident primitives of `color` at the origin, large enough
/// to cover the probe camera's image, of opacity 0.999, so the layer's A at the image
/// center is 1 − 0.001² and its C/A is `color` everywhere it is covered.
pub fn flat_probe(color: [f32; 3]) -> Model {
    let row = Row {
        position: [0.0; 3],
        velocity: [0.0; 3],
        time_center: 0.95,
        duration: 100.0,
        scale: 3.0,
        opacity: 0.999,
        color,
    };
    model(&[row, row])
}

/// The pixels of the 8 × 8 block at the center of a 64 × 48 image.
pub fn center_block() -> impl Iterator<Item = usize> {
    (20..28).flat_map(|y| (28..36).map(move |x| y * 64 + x))
}
