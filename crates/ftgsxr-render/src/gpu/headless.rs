// SPDX-License-Identifier: Apache-2.0

//! `HeadlessRenderer` (§2.1, §1.7 `render`): its own wgpu instance, adapter and device; the
//! per-view pipeline of §1.4 stages 1–5 into a 16-bit float layer, read back and composited
//! over the request's background on the CPU, unclamped (E7); the two-eye render of R-9 per
//! view or shared-center (D-4, Q-401); E-GPU-LOST's single recovery.

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
use super::frame::FrameBuffers;
use super::inputs::{ColorTransform, ModelDraw, Overlays, RenderMode, SortPose, ViewParams};
use super::model::GpuModel;
use super::passes::SplatPasses;
use super::probe::{LayerSums, PROBE_BYTES, ProbePass};
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

/// The shared-center passes and frame buffers of `render_stereo` (Q-401), made at its
/// first shared-center call.
struct Shared {
    passes: SplatPasses,
    frame: Option<FrameBuffers>,
}

/// One device with its passes, the uploaded models and the frame buffers.
struct Gpu {
    dev: SplatDevice,
    passes: SplatPasses,
    probe: ProbePass,
    models: Vec<GpuModel>,
    frame: Option<FrameBuffers>,
    shared: Option<Shared>,
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
        let probe = ProbePass::new(dev.device())?;
        let models = packed
            .iter()
            .map(|p| GpuModel::upload(&dev, &p.arrays, p.index.clone(), &p.conventions))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Gpu {
            dev,
            passes,
            probe,
            models,
            frame: None,
            shared: None,
            lost,
        })
    }

    fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }
}

/// The headless adapter (§1.7 `render`): its own wgpu instance, adapter and device.
///
/// `render` draws one view per call (per-view sort, §1.4 Stereo), `render_stereo` two
/// (per view or shared-center, Q-401), each into a 16-bit float layer
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
    /// `render` has one view (§1.4 Stereo) and sorts per view whatever the tier (D-4,
    /// Q-109): a shared-center tier keeps its storage, SH cap, key width, capacity and
    /// budget but not its stereo mode, and the render is at the camera's own size, so the
    /// tier's `render_scale` and foveation (the swapchain's) do not apply. `render_stereo`
    /// takes its sort mode per call, and in shared-center mode the tier's `half_records`
    /// picks the record layout (Q-401).
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
        self.with_recovery(models, |r| r.render_once(models, camera, tau, background))
    }

    /// The layer probe of the same render's layer (E14's exposure match, R-21); it blocks,
    /// outside any frame loop.
    ///
    /// The layer is drawn as `render` draws it, then `probe.wgsl` sums each cell of the
    /// 64 × 48 grid over it, (Σ α r_lin, Σ α g_lin, Σ α b_lin, Σ α), and the grid is read
    /// back. A lost device is recovered once, as in `render`.
    ///
    /// # Panics
    ///
    /// When an id was not returned by this renderer's `load`.
    pub fn probe(
        &mut self,
        models: &[(ModelId, Pose)],
        camera: &Camera,
        tau: f64,
    ) -> Result<LayerSums, RenderError> {
        self.with_recovery(models, |r| r.probe_once(models, camera, tau))
    }

    /// E6 and E7 for two eyes at `tau` over `background`, one image per eye in `eyes`'
    /// order, each as `render` makes it (Q-401).
    ///
    /// With `SortMode::PerView` each eye runs §1.4 stages 1–5 with its own camera, as
    /// `render` does. With `SortMode::SharedCenter` (D-4) stages 1–4 run once at the sort
    /// pose, whose origin is the mean of the eyes' centers and whose forward axis the
    /// normalized sum of their forward axes (Q-399), with both eyes as cull views, and
    /// stage 5 runs for each eye, projecting each stage-space record for it (Q-400); the
    /// records are the tier's, the 48-byte half-precision Σ^S layout when it sets
    /// `half_records`. `render`'s per-view rule (Q-109) is unchanged.
    ///
    /// # Panics
    ///
    /// When an id was not returned by this renderer's `load`.
    // Q-401
    pub fn render_stereo(
        &mut self,
        models: &[(ModelId, Pose)],
        eyes: &[Camera; 2],
        tau: f64,
        background: [f32; 3],
        sort: SortMode,
    ) -> Result<[RenderImage; 2], RenderError> {
        self.with_recovery(models, |r| match sort {
            SortMode::PerView => Ok([
                r.render_once(models, &eyes[0], tau, background)?,
                r.render_once(models, &eyes[1], tau, background)?,
            ]),
            SortMode::SharedCenter => r.render_shared_once(models, eyes, tau, background),
        })
    }

    /// `run` with E-GPU-LOST's single recovery, after checking the ids.
    fn with_recovery<T>(
        &mut self,
        models: &[(ModelId, Pose)],
        mut run: impl FnMut(&mut HeadlessRenderer) -> Result<T, RenderError>,
    ) -> Result<T, RenderError> {
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
                run(self)
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
        let (layer, mut enc) = self.draw_layer(models, camera, tau);
        let gpu = &self.gpu;
        let (staging, row_bytes) = copy_layer(gpu.dev.device(), &mut enc, &layer, camera);
        let bytes = gpu.submit_and_read(enc, &[&staging])?;
        composite(
            &bytes[0],
            camera.width,
            camera.height,
            row_bytes,
            background,
        )
    }

    /// `render_stereo`'s shared-center render (Q-401): one prepare, two draws, both layers
    /// read back by one submission.
    fn render_shared_once(
        &mut self,
        models: &[(ModelId, Pose)],
        eyes: &[Camera; 2],
        tau: f64,
        background: [f32; 3],
    ) -> Result<[RenderImage; 2], RenderError> {
        let (layers, mut enc) = self.draw_shared_layers(models, eyes, tau)?;
        let gpu = &self.gpu;
        let device = gpu.dev.device();
        let copies: Vec<(wgpu::Buffer, u32)> = layers
            .iter()
            .zip(eyes)
            .map(|(layer, camera)| copy_layer(device, &mut enc, layer, camera))
            .collect();
        let staging: Vec<&wgpu::Buffer> = copies.iter().map(|(b, _)| b).collect();
        let bytes = gpu.submit_and_read(enc, &staging)?;
        let image = |k: usize| {
            composite(
                &bytes[k],
                eyes[k].width,
                eyes[k].height,
                copies[k].1,
                background,
            )
        };
        Ok([image(0)?, image(1)?])
    }

    /// §1.4 stages 1–4 once at the eyes' sort pose (Q-399) with both eyes as cull views,
    /// then stage 5 into a new layer per eye, cleared to (0, 0, 0, 0): the first eye's pass
    /// writes the draw's begin timestamp and the last eye's its end (Q-403). Encoded on the
    /// returned encoder, not yet submitted. E-GPU-SHADER when the shared-center passes,
    /// made at the first call, do not compile.
    pub(crate) fn draw_shared_layers(
        &mut self,
        models: &[(ModelId, Pose)],
        eyes: &[Camera; 2],
        tau: f64,
    ) -> Result<(Vec<wgpu::Texture>, wgpu::CommandEncoder), RenderError> {
        let capacity = self.capacity(models);
        let tier = QualityTier {
            sort_mode: SortMode::SharedCenter,
            ..self.tier.clone()
        };
        let gpu = &mut self.gpu;
        if gpu.shared.is_none() {
            let passes = SplatPasses::new(&gpu.dev, LAYER_FORMAT, None, &tier)?;
            gpu.shared = Some(Shared {
                passes,
                frame: None,
            });
        }
        let shared = gpu
            .shared
            .as_mut()
            .expect("invariant: the shared-center passes were made above");
        if shared
            .frame
            .as_ref()
            .is_none_or(|f| f.capacity() != capacity)
        {
            shared.frame = Some(FrameBuffers::new(&gpu.dev, &tier, capacity));
        }
        let frame = shared
            .frame
            .as_ref()
            .expect("invariant: the frame buffers were made above");
        let draws = model_draws(&gpu.models, models, tau);
        let views: Vec<ViewParams> = eyes.iter().map(view_of).collect();
        let device = gpu.dev.device();
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ftgsxr headless stereo render"),
        });
        // Q-399, D-4
        shared
            .passes
            .encode_prepare(&mut enc, frame, &draws, &stereo_sort_pose(eyes), &views);
        let layers = views
            .iter()
            .enumerate()
            .map(|(k, view)| {
                let layer = new_layer(device, &eyes[k]);
                // Q-403
                let timestamp_writes = frame.eye_draw_timestamp_writes(k, views.len());
                draw_pass(
                    &mut enc,
                    &shared.passes,
                    frame,
                    &layer,
                    view,
                    timestamp_writes,
                );
                layer
            })
            .collect();
        Ok((layers, enc))
    }

    /// The frame's capacity C for `models` (Q-86: `SumOfModels` is the instances' total N).
    fn capacity(&self, models: &[(ModelId, Pose)]) -> u32 {
        match self.tier.capacity {
            Capacity::Fixed(c) => c,
            Capacity::SumOfModels => models
                .iter()
                .map(|(id, _)| self.gpu.models[id.0 as usize].n())
                .fold(0u32, u32::saturating_add),
        }
    }

    fn probe_once(
        &mut self,
        models: &[(ModelId, Pose)],
        camera: &Camera,
        tau: f64,
    ) -> Result<LayerSums, RenderError> {
        let (layer, mut enc) = self.draw_layer(models, camera, tau);
        let gpu = &self.gpu;
        let device = gpu.dev.device();
        let cells = device.create_buffer(&BufferDescriptor {
            label: Some("ftgsxr headless probe"),
            size: PROBE_BYTES,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&BufferDescriptor {
            label: Some("ftgsxr headless probe read back"),
            size: PROBE_BYTES,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let view = layer.create_view(&wgpu::TextureViewDescriptor::default());
        gpu.probe.encode(device, &mut enc, &view, &cells);
        enc.copy_buffer_to_buffer(&cells, 0, &staging, 0, PROBE_BYTES);
        let bytes = gpu.submit_and_read(enc, &[&staging])?;
        Ok(LayerSums::from_bytes(&bytes[0])
            .expect("invariant: the probe was encoded before the read back"))
    }

    /// §1.4 stages 1–5 for `models` into a new layer cleared to (0, 0, 0, 0), encoded on
    /// the returned encoder, not yet submitted.
    fn draw_layer(
        &mut self,
        models: &[(ModelId, Pose)],
        camera: &Camera,
        tau: f64,
    ) -> (wgpu::Texture, wgpu::CommandEncoder) {
        let capacity = self.capacity(models);
        let gpu = &mut self.gpu;
        if gpu.frame.as_ref().is_none_or(|f| f.capacity() != capacity) {
            gpu.frame = Some(FrameBuffers::new(&gpu.dev, &self.tier, capacity));
        }
        let frame = gpu
            .frame
            .as_ref()
            .expect("invariant: the frame buffers were made above");
        let draws = model_draws(&gpu.models, models, tau);
        let view = view_of(camera);
        let device = gpu.dev.device();
        let layer = new_layer(device, camera);
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ftgsxr headless render"),
        });
        gpu.passes
            .encode_prepare(&mut enc, frame, &draws, &sort_pose(camera), &[view]);
        // The draw's timestamps (Q-81, Q-114): this renderer owns the pass.
        let timestamp_writes = frame.draw_timestamp_writes();
        draw_pass(
            &mut enc,
            &gpu.passes,
            frame,
            &layer,
            &view,
            timestamp_writes,
        );
        (layer, enc)
    }
}

/// Each `(id, pose)` as a splat instance at `tau`, at its loaded degree, untrimmed.
fn model_draws<'a>(
    gpu_models: &'a [GpuModel],
    models: &[(ModelId, Pose)],
    tau: f64,
) -> Vec<ModelDraw<'a>> {
    models
        .iter()
        .map(|&(id, pose)| {
            let model = &gpu_models[id.0 as usize];
            ModelDraw {
                model,
                pose,
                tau,
                mode: RenderMode::Splats,
                transform: ColorTransform::IDENTITY,
                sh_degree: model.sh_degree,
                overlays: Overlays::default(),
                // §2.1: the headless render draws untrimmed (E17).
                trim: ftgsxr_core::Trim::default(),
            }
        })
        .collect()
}

/// `camera`'s view at its own size.
fn view_of(camera: &Camera) -> ViewParams {
    ViewParams {
        camera: *camera,
        target_size: UVec2::new(camera.width, camera.height),
    }
}

/// A new layer of `camera`'s size.
fn new_layer(device: &wgpu::Device, camera: &Camera) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ftgsxr headless layer"),
        size: wgpu::Extent3d {
            width: camera.width,
            height: camera.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: LAYER_FORMAT,
        // The probe reads the layer as a texture (R-21).
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

/// Stage 5 of `view` into `layer`, cleared to (0, 0, 0, 0), in a pass of its own with
/// `timestamp_writes`.
fn draw_pass(
    enc: &mut wgpu::CommandEncoder,
    passes: &SplatPasses,
    frame: &FrameBuffers,
    layer: &wgpu::Texture,
    view: &ViewParams,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
) {
    let layer_view = layer.create_view(&wgpu::TextureViewDescriptor::default());
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
    passes.encode_draw(&mut pass, frame, view, None);
}

/// A copy of `layer` (of `camera`'s size) into a new mappable buffer, encoded on `enc`;
/// the buffer and its row pitch.
fn copy_layer(
    device: &wgpu::Device,
    enc: &mut wgpu::CommandEncoder,
    layer: &wgpu::Texture,
    camera: &Camera,
) -> (wgpu::Buffer, u32) {
    let (w, h) = (camera.width, camera.height);
    let row_bytes = (w * TEXEL_BYTES).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("ftgsxr headless read back"),
        size: u64::from(row_bytes) * u64::from(h),
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: layer,
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
    (staging, row_bytes)
}

impl Gpu {
    /// Submits `enc` and reads each of `staging` back, blocking; E-GPU-LOST when the
    /// device is lost.
    fn submit_and_read(
        &self,
        enc: wgpu::CommandEncoder,
        staging: &[&wgpu::Buffer],
    ) -> Result<Vec<Vec<u8>>, RenderError> {
        self.dev.queue().submit([enc.finish()]);
        let mapped: Vec<Arc<AtomicBool>> = staging
            .iter()
            .map(|buffer| {
                let mapped = Arc::new(AtomicBool::new(false));
                let flag = Arc::clone(&mapped);
                buffer.map_async(wgpu::MapMode::Read, .., move |r| {
                    flag.store(r.is_ok(), Ordering::Release);
                });
                mapped
            })
            .collect();
        let polled = self.dev.device().poll(wgpu::PollType::wait_indefinitely());
        if polled.is_err() || self.is_lost() || mapped.iter().any(|m| !m.load(Ordering::Acquire)) {
            return Err(RenderError::DeviceLost);
        }
        Ok(staging
            .iter()
            .map(|buffer| {
                let bytes = buffer.get_mapped_range(..).to_vec();
                buffer.unmap();
                bytes
            })
            .collect())
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

/// The sort pose of several eyes (Q-399): its origin the mean of the eyes' centers, its
/// forward axis the normalized sum of their forward axes (E4's eye and E8's depth axis in
/// shared-center mode, D-4); one eye's is `sort_pose`'s.
// Q-399
pub(crate) fn stereo_sort_pose(eyes: &[Camera]) -> SortPose {
    let (mut origin, mut forward) = (DVec3::ZERO, DVec3::ZERO);
    for camera in eyes {
        let r = DMat3::from_mat4(camera.world_to_camera);
        let t = camera.world_to_camera.w_axis.truncate();
        origin -= r.transpose() * t;
        forward += r.transpose() * DVec3::Z;
    }
    SortPose {
        origin: (origin / eyes.len() as f64).as_vec3(),
        forward: forward.normalize().as_vec3(),
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
    use ftgsxr_core::Vec3;

    /// One texel of halves (r, g, b, a) as little-endian bytes.
    fn texel(h: [u16; 4]) -> Vec<u8> {
        h.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// A camera at `center` whose OpenCV axes are the rows `right`, `down`, `forward`.
    fn camera_at(center: DVec3, right: DVec3, down: DVec3, forward: DVec3) -> Camera {
        let r = DMat3::from_cols(right, down, forward).transpose();
        let mut w2c = ftgsxr_core::DMat4::from_mat3(r);
        w2c.w_axis = (-(r * center)).extend(1.0);
        Camera {
            width: 4,
            height: 4,
            k: DMat3::IDENTITY,
            world_to_camera: w2c,
        }
    }

    // Q-399: the sort pose of two eyes is at the mean of their positions, along the
    // normalized sum of their forward axes: R-9's parallel eyes at (∓0.032, −1, 0) looking
    // along +y give (0, −1, 0) and +y; eyes toed in by ±45° about y, looking along
    // (±1, 0, 1)/√2, give +z.
    #[test]
    fn stereo_sort_pose_is_the_mean_position_and_the_normalized_sum_of_the_forward_axes() {
        let (x, y, z) = (DVec3::X, DVec3::Y, DVec3::Z);
        let parallel = [
            camera_at(DVec3::new(-0.032, -1.0, 0.0), x, -z, y),
            camera_at(DVec3::new(0.032, -1.0, 0.0), x, -z, y),
        ];
        let pose = stereo_sort_pose(&parallel);
        assert!((pose.origin - Vec3::new(0.0, -1.0, 0.0)).length() < 1e-7);
        assert!((pose.forward - Vec3::Y).length() < 1e-7);
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let toed = [
            camera_at(
                DVec3::new(-0.5, 0.0, 0.0),
                DVec3::new(s, 0.0, -s),
                -y,
                DVec3::new(s, 0.0, s),
            ),
            camera_at(
                DVec3::new(0.5, 0.0, 0.0),
                DVec3::new(s, 0.0, s),
                -y,
                DVec3::new(-s, 0.0, s),
            ),
        ];
        let pose = stereo_sort_pose(&toed);
        assert!(pose.origin.length() < 1e-7);
        assert!((pose.forward - Vec3::Z).length() < 1e-7);
    }

    /// Q-403, Q-114's check across two eyes: a two-eye shared-center render times its
    /// draw from the first eye's pass's begin to the last eye's end, so on a device with
    /// timestamp queries `GpuTimes.draw` > 0 and `total` ≥ `sort` + `draw`. R9_P1's first
    /// eye pair against P1, three frames, the third reading the first's block (Q-81).
    #[test]
    #[ignore = "gpu"]
    fn gpu_shared_center_draw_times_span_both_eyes() {
        let _layer = ftgsxr_testkit::layer_messages(); // R-16, Q-105
        let (model, _) = ftgsxr_testkit::probe("P1");
        let path = ftgsxr_testkit::fixtures_dir().join("probes/R9_P1.request.json");
        let request = ftgsxr_core::read_request(&std::fs::read(path).unwrap()).unwrap();
        let mut renderer = HeadlessRenderer::new(QualityTier::conformance()).unwrap();
        let id = renderer.load(&model, request.sh_degree).unwrap();
        let instances = [(id, Pose::IDENTITY)];
        let eyes = [request.items[0].camera, request.items[1].camera];
        let tau = request.items[0].timestamp;
        let mut stats = None;
        for k in 0..3 {
            let (_layers, mut enc) = renderer.draw_shared_layers(&instances, &eyes, tau).unwrap();
            let gpu = &renderer.gpu;
            let shared = gpu.shared.as_ref().unwrap();
            let frame = shared.frame.as_ref().unwrap();
            shared.passes.encode_stats(&mut enc, frame);
            gpu.dev.queue().submit([enc.finish()]);
            gpu.dev
                .device()
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            stats = super::super::stats::FrameStats::poll(frame, &gpu.dev);
            assert_eq!(stats.is_some(), k == 2, "frame {k}");
        }
        let stats = stats.unwrap();
        assert!(stats.visible_splats > 0);
        let t = stats.gpu_times_ms;
        eprintln!("two-eye shared-center GPU times: {t:?}");
        if renderer
            .gpu
            .dev
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            assert!(t.draw > 0.0, "{t:?}");
            assert!(t.total >= t.sort + t.draw, "{t:?}");
            // The split itself: the first eye of two writes the begin only, the last the
            // end only.
            let shared = renderer.gpu.shared.as_ref().unwrap();
            let frame = shared.frame.as_ref().unwrap();
            let indices = |eye| {
                let w = frame.eye_draw_timestamp_writes(eye, 2).unwrap();
                (w.beginning_of_pass_write_index, w.end_of_pass_write_index)
            };
            assert_eq!(indices(0), (Some(4), None));
            assert_eq!(indices(1), (None, Some(5)));
        }
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
