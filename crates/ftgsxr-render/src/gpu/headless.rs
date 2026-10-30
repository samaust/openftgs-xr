// SPDX-License-Identifier: Apache-2.0

//! `HeadlessRenderer` (§2.1, §1.7 `render`): its own wgpu instance, adapter and device; the
//! per-view pipeline of §1.4 stages 1–5 into a 16-bit float layer, read back and composited
//! over the request's background on the CPU, unclamped (E7); E-GPU-LOST's single recovery.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bevy::log::warn;
use bevy::render::render_resource::{BufferDescriptor, BufferUsages};
use ftgsxr_core::{
    Camera, DMat3, DVec3, GpuArrays, Model, Pose, RenderConventions, RenderImage, TemporalIndex,
    UVec2, pack_for_gpu,
};

use super::device::{BudgetSource, MemoryBudget, SplatDevice};
use super::error::RenderError;
use super::frame::{FrameBuffers, TIMESTAMPS};
use super::inputs::{ColorTransform, ModelDraw, Overlays, RenderMode, SortPose, ViewParams};
use super::model::GpuModel;
use super::passes::SplatPasses;
use super::readback::half_to_f32;
use super::tier::{Capacity, DESKTOP_FALLBACK_BUDGET, QualityTier, SortMode};

/// The layer's format (§1.4 stage 5, D-35): 16-bit float RGBA.
const LAYER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// A layer texel's bytes.
const TEXEL_BYTES: u32 = 8;

/// A model loaded into a `HeadlessRenderer`, by the order of the `load` calls (§2.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModelId(pub u32);

/// A loaded model as packed for the GPU, kept so a lost device can be re-created and the
/// models re-uploaded (E-GPU-LOST).
struct Packed {
    arrays: GpuArrays,
    index: TemporalIndex,
    conventions: RenderConventions,
}

/// One device with its passes, the uploaded models and the frame buffers.
struct Gpu {
    dev: SplatDevice,
    passes: SplatPasses,
    models: Vec<GpuModel>,
    frame: Option<FrameBuffers>,
    lost: Arc<AtomicBool>,
}

impl Gpu {
    /// A device of its own (Q-74), the passes of `tier` and `packed` uploaded.
    fn create(tier: &QualityTier, packed: &[Packed]) -> Result<Gpu, RenderError> {
        let mut dev = SplatDevice::headless()?;
        // Q-86: a tier's fixed budget (the Quest's) replaces the device's desktop rule;
        // the desktop-class tiers keep Q-66's 75 % of the heap.
        if tier.memory_budget_bytes != DESKTOP_FALLBACK_BUDGET {
            dev.set_memory_budget(MemoryBudget {
                bytes: tier.memory_budget_bytes,
                source: BudgetSource::Tier,
            });
        }
        let lost = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&lost);
        dev.device()
            .set_device_lost_callback(move |reason, message| {
                warn!("graphics device lost ({reason:?}): {message}");
                flag.store(true, Ordering::Release);
            });
        let passes = SplatPasses::new(&dev, LAYER_FORMAT, None, tier)?;
        let models = packed
            .iter()
            .map(|p| GpuModel::upload(&dev, &p.arrays, p.index.clone(), &p.conventions))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Gpu {
            dev,
            passes,
            models,
            frame: None,
            lost,
        })
    }

    fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }
}

/// The headless adapter (§1.7 `render`): its own wgpu instance, adapter and device.
///
/// It renders one view per call (per-view sort, §1.4 Stereo) into a 16-bit float layer
/// cleared to (0, 0, 0, 0), reads the layer back and composites it over the background on
/// the CPU (E7). A lost device is re-created once, with every loaded model re-uploaded,
/// and the render retried; a second loss is `RenderError::DeviceLost` (E-GPU-LOST).
pub struct HeadlessRenderer {
    tier: QualityTier,
    packed: Vec<Packed>,
    gpu: Gpu,
    /// Whether the single recovery attempt has been spent.
    recovered: bool,
}

impl HeadlessRenderer {
    /// A renderer for `tier` on the high-performance Vulkan adapter (Q-74); E-GPU-ADAPTER
    /// when there is none, E-GPU-SHADER when a pass does not compile.
    ///
    /// A headless render has one view (§1.4 Stereo) and sorts per view whatever the tier
    /// (D-4, Q-109): a shared-center tier keeps its storage, SH cap, key width, capacity
    /// and budget but not its stereo mode, and the render is at the camera's own size, so
    /// the tier's `render_scale` and foveation (the swapchain's) do not apply.
    pub fn new(tier: QualityTier) -> Result<HeadlessRenderer, RenderError> {
        // Q-109, D-4: "per-view sort on the desktop and in headless renders".
        let mut tier = tier;
        tier.sort_mode = SortMode::PerView;
        let gpu = Gpu::create(&tier, &[])?;
        Ok(HeadlessRenderer {
            tier,
            packed: Vec::new(),
            gpu,
            recovered: false,
        })
    }

    /// Loads `model` with the tier's load options (§1.2): the SH cap min(tier's cap,
    /// `sh_degree`), where `None` is the model's L and a degree above L keeps L (the request
    /// check E-REQ-DEGREE is the `render` command's, Q-86); the tier's half-precision
    /// fields; time rebased to τ − τ₀; the memory check against what the loaded models
    /// leave of the budget (E-MEM-BUDGET, Q-74); the temporal ordering (E3). A convention
    /// rendered as openftgs's instead is logged as a warning (E6, D-39, Q-110).
    pub fn load(&mut self, model: &Model, sh_degree: Option<u8>) -> Result<ModelId, RenderError> {
        // D-39, Q-110
        for w in model.conventions.warnings() {
            warn!("{w}");
        }
        // Q-86
        let mut layout = self.tier.layout;
        layout.sh_cap = match (layout.sh_cap, sh_degree) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let (index, order) = TemporalIndex::build(model);
        let packed = Packed {
            arrays: pack_for_gpu(model, &order, &layout),
            index,
            conventions: model.conventions.clone(),
        };
        let gpu_model = GpuModel::upload(
            &self.gpu.dev,
            &packed.arrays,
            packed.index.clone(),
            &packed.conventions,
        )?;
        self.gpu.models.push(gpu_model);
        self.packed.push(packed);
        Ok(ModelId(self.packed.len() as u32 - 1))
    }

    /// E6 and E7 for `camera` at `tau` over `background`; rgb unclamped, alpha = A.
    ///
    /// Each `(id, pose)` is one instance at its pose P_m (E5), evaluated at τ_m = `tau`,
    /// drawn as splats at its loaded degree; all of them share one record buffer, one count
    /// and one sort (D-5), of capacity C from the tier (`SumOfModels`: the instances' total
    /// N). The output is C + (1 − A)·c_bg per channel and A (E7), row-major from the image's
    /// top row; a non-finite value is `RenderError::NonFinite` (E-NUMERIC).
    ///
    /// # Panics
    ///
    /// When an id was not returned by this renderer's `load`.
    pub fn render(
        &mut self,
        models: &[(ModelId, Pose)],
        camera: &Camera,
        tau: f64,
        background: [f32; 3],
    ) -> Result<RenderImage, RenderError> {
        for (id, _) in models {
            assert!(
                (id.0 as usize) < self.packed.len(),
                "model id {} was not loaded by this renderer",
                id.0
            );
        }
        loop {
            let result = if self.gpu.is_lost() {
                Err(RenderError::DeviceLost)
            } else {
                self.render_once(models, camera, tau, background)
            };
            match result {
                Err(RenderError::DeviceLost) if !self.recovered => {
                    // E-GPU-LOST (Q-86): one recovery attempt, the device re-created and the
                    // instances re-uploaded.
                    self.recovered = true;
                    warn!("graphics device lost; re-creating it and re-uploading the models");
                    // The lost device's messages were logged by its collector already.
                    let _ = self.gpu.dev.take_validation_messages();
                    self.gpu = Gpu::create(&self.tier, &self.packed)?;
                }
                other => return other,
            }
        }
    }

    /// The adapter's name, as wgpu reports it.
    pub fn adapter_name(&self) -> &str {
        self.gpu.dev.adapter_name()
    }

    fn render_once(
        &mut self,
        models: &[(ModelId, Pose)],
        camera: &Camera,
        tau: f64,
        background: [f32; 3],
    ) -> Result<RenderImage, RenderError> {
        let capacity = match self.tier.capacity {
            Capacity::Fixed(c) => c,
            // Q-86: the instances' total N.
            Capacity::SumOfModels => models
                .iter()
                .map(|(id, _)| self.gpu.models[id.0 as usize].n())
                .fold(0u32, u32::saturating_add),
        };
        let gpu = &mut self.gpu;
        if gpu.frame.as_ref().is_none_or(|f| f.capacity() != capacity) {
            gpu.frame = Some(FrameBuffers::new(&gpu.dev, &self.tier, capacity));
        }
        let frame = gpu
            .frame
            .as_ref()
            .expect("invariant: the frame buffers were made above");
        let draws: Vec<ModelDraw> = models
            .iter()
            .map(|&(id, pose)| {
                let model = &gpu.models[id.0 as usize];
                ModelDraw {
                    model,
                    pose,
                    tau,
                    mode: RenderMode::Splats,
                    transform: ColorTransform::IDENTITY,
                    sh_degree: model.sh_degree,
                    overlays: Overlays::default(),
                }
            })
            .collect();
        let (w, h) = (camera.width, camera.height);
        let view = ViewParams {
            camera: *camera,
            target_size: UVec2::new(w, h),
        };
        let device = gpu.dev.device();
        let layer = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ftgsxr headless layer"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: LAYER_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let layer_view = layer.create_view(&wgpu::TextureViewDescriptor::default());
        let row_bytes = (w * TEXEL_BYTES).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some("ftgsxr headless read back"),
            size: u64::from(row_bytes) * u64::from(h),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ftgsxr headless render"),
        });
        gpu.passes
            .encode_prepare(&mut enc, frame, &draws, &sort_pose(camera), &[view]);
        {
            // The draw's timestamps, indices 4 and 5 (Q-81): this renderer owns the pass.
            let timestamp_writes =
                frame
                    .set
                    .queries
                    .as_ref()
                    .map(|(queries, _)| wgpu::RenderPassTimestampWrites {
                        query_set: queries,
                        beginning_of_pass_write_index: Some(4),
                        end_of_pass_write_index: Some(5),
                    });
            if timestamp_writes.is_some() {
                frame.ring.written.store(TIMESTAMPS, Ordering::Release);
            }
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ftgsxr headless splats"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &layer_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            gpu.passes.encode_draw(&mut pass, frame, &view, None);
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &layer,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        gpu.dev.queue().submit([enc.finish()]);
        let mapped = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&mapped);
        staging.map_async(wgpu::MapMode::Read, .., move |r| {
            flag.store(r.is_ok(), Ordering::Release);
        });
        let polled = device.poll(wgpu::PollType::wait_indefinitely());
        if polled.is_err() || gpu.is_lost() || !mapped.load(Ordering::Acquire) {
            return Err(RenderError::DeviceLost);
        }
        let image = {
            let bytes = staging.get_mapped_range(..);
            composite(&bytes, w, h, row_bytes, background)
        };
        staging.unmap();
        image
    }
}

/// The camera's center −R_wᵀ t_w and forward axis R_wᵀ ẑ (E4's eye; per-view mode takes the
/// depth from the camera itself, so the pose only fills the frame block).
pub(crate) fn sort_pose(camera: &Camera) -> SortPose {
    let r = DMat3::from_mat4(camera.world_to_camera);
    let t = camera.world_to_camera.w_axis.truncate();
    SortPose {
        origin: (-(r.transpose() * t)).as_vec3(),
        forward: (r.transpose() * DVec3::Z).as_vec3(),
    }
}

/// The layer's half floats (rows of `row_bytes`) composited over `background`: rgb =
/// C + (1 − A)·c_bg, alpha = A, unclamped (E7); any non-finite value is E-NUMERIC.
fn composite(
    bytes: &[u8],
    w: u32,
    h: u32,
    row_bytes: u32,
    background: [f32; 3],
) -> Result<RenderImage, RenderError> {
    let pixels = w as usize * h as usize;
    let mut rgb = Vec::with_capacity(3 * pixels);
    let mut alpha = Vec::with_capacity(pixels);
    for y in 0..h as usize {
        let row = &bytes[y * row_bytes as usize..][..(w * TEXEL_BYTES) as usize];
        for texel in row.as_chunks::<8>().0 {
            let v = |i: usize| half_to_f32(u16::from_le_bytes([texel[2 * i], texel[2 * i + 1]]));
            let a = v(3);
            for (c, bg) in background.iter().enumerate() {
                rgb.push(v(c) + (1.0 - a) * bg);
            }
            alpha.push(a);
        }
    }
    if rgb.iter().chain(&alpha).any(|v| !v.is_finite()) {
        return Err(RenderError::NonFinite);
    }
    Ok(RenderImage {
        width: w,
        height: h,
        rgb,
        alpha,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One texel of halves (r, g, b, a) as little-endian bytes.
    fn texel(h: [u16; 4]) -> Vec<u8> {
        h.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn composite_adds_the_background_by_the_remaining_transmittance() {
        // (C, A) = (0.5, 0.25, 0, 0.5) over (1, 0, 2): (1.0, 0.25, 1.0), alpha 0.5; a row
        // padded to 16 bytes, one texel wide, two rows.
        let mut bytes = texel([0x3800, 0x3400, 0x0000, 0x3800]);
        bytes.extend([0u8; 8]);
        bytes.extend(texel([0, 0, 0, 0]));
        bytes.extend([0u8; 8]);
        let image = composite(&bytes, 1, 2, 16, [1.0, 0.0, 2.0]).unwrap();
        assert_eq!(image.rgb, vec![1.0, 0.25, 1.0, 1.0, 0.0, 2.0]);
        assert_eq!(image.alpha, vec![0.5, 0.0]);
    }

    #[test]
    fn composite_keeps_values_above_one() {
        // C = 1.5 (0x3e00) with A = 0.5 over white: 2.0, unclamped.
        let image = composite(&texel([0x3e00, 0x3e00, 0x3e00, 0x3800]), 1, 1, 8, [1.0; 3]).unwrap();
        assert_eq!(image.rgb, vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn composite_reports_a_non_finite_value() {
        for bad in [0x7e00, 0x7c00, 0xfc00] {
            let r = composite(&texel([bad, 0, 0, 0]), 1, 1, 8, [0.0; 3]);
            assert!(matches!(r, Err(RenderError::NonFinite)), "{bad:#x}");
        }
    }
}
