// SPDX-License-Identifier: Apache-2.0

//! Q-451 (#820): the tightened splat quad of §1.4 stage 5 (revision 61, Q-450) is drawn,
//! not only its fallback to μ′ ± r. Q-448's pixel check passes as well when every splat
//! falls back, so this test counts the splat draw's fragment-shader invocations (wgpu's
//! pipeline-statistics query) for R2\_random's turned eye pairs (Q-448's cameras), per view
//! and in shared-center mode, and compares them in the same run with the pixel centres
//! inside each visible primitive's image-clipped box μ′ ± r from `ftgsxr_core::project`'s
//! radii: at most 0.9 of them, the slack being for helper invocations. On an adapter
//! without the query the test is skipped with a logged reason.

use crate::cull_support::{oracle, sort_pose, upload};
use crate::q448_exact::turned_eyes;

use ftgsxr_core::{Camera, Pose, UVec2};
use ftgsxr_render::{
    ColorTransform, FrameBuffers, ModelDraw, Overlays, QualityTier, RenderMode, SortMode, SortPose,
    SplatDevice, SplatPasses, ViewParams,
};
use ftgsxr_testkit::{layer_messages, probe};

/// Q-451: the invocations may be at most this share of the boxes' pixel centres.
const MAX_SHARE: f64 = 0.9;

/// A device of the test's own with the pipeline-statistics query, or `None` (and why)
/// when the adapter lacks it. The other features are `SplatDevice::headless`'s, where
/// offered.
fn statistics_device() -> Result<SplatDevice, String> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    desc.backends = wgpu::Backends::VULKAN;
    desc.flags = wgpu::InstanceFlags::from_build_config().with_env();
    let instance = wgpu::Instance::new(desc);
    let adapter = bevy::tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .map_err(|e| format!("no Vulkan adapter: {e}"))?;
    let info = adapter.get_info();
    if !adapter
        .features()
        .contains(wgpu::Features::PIPELINE_STATISTICS_QUERY)
    {
        return Err(format!("{} offers no PIPELINE_STATISTICS_QUERY", info.name));
    }
    let wanted = wgpu::Features::PIPELINE_STATISTICS_QUERY
        | wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::INDIRECT_FIRST_INSTANCE;
    let (device, queue) = bevy::tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("ftgsxr q451 device"),
        required_features: adapter.features() & wanted,
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .map_err(|e| format!("{}: no device: {e}", info.name))?;
    Ok(SplatDevice::from_wgpu(device, queue, info))
}

/// The pixel centres (i + ½, j + ½) of a `camera`-sized image inside μ′ ± r.
fn box_centres(mean: [f32; 2], radii: [u32; 2], camera: &Camera) -> u64 {
    let axis = |m: f32, r: u32, n: u32| {
        let (lo, hi) = (f64::from(m) - f64::from(r), f64::from(m) + f64::from(r));
        (0..n)
            .filter(|&i| {
                let c = f64::from(i) + 0.5;
                lo < c && c < hi
            })
            .count() as u64
    };
    axis(mean[0], radii[0], camera.width) * axis(mean[1], radii[1], camera.height)
}

/// The reference of one eye: the pixel centres inside the image-clipped boxes of every
/// primitive `ftgsxr_core::project` keeps in it.
fn reference(model: &ftgsxr_core::Model, tau: f64, camera: &Camera) -> u64 {
    oracle(model, &Pose::IDENTITY, tau, camera)
        .iter()
        .map(|(_, p)| box_centres(p.mean.into(), p.radii, camera))
        .sum()
}

/// The splat draw's fragment-shader invocations for each of `eyes`, after one prepare of
/// `draws` in `sort` (per view: one prepare per eye at its own pose; shared center: one at
/// the eyes' midpoint with both as cull views).
fn invocations(
    dev: &SplatDevice,
    draws: &[ModelDraw],
    eyes: &[Camera; 2],
    sort: SortMode,
) -> [u64; 2] {
    let tier = QualityTier {
        sort_mode: sort,
        ..QualityTier::conformance()
    };
    let format = wgpu::TextureFormat::Rgba16Float;
    let passes = SplatPasses::new(dev, format, None, &tier).expect("the passes compile");
    let capacity = 2 * draws.iter().map(|d| d.model.n()).sum::<u32>();
    let device = dev.device();
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("q451 statistics"),
        ty: wgpu::QueryType::PipelineStatistics(
            wgpu::PipelineStatisticsTypes::FRAGMENT_SHADER_INVOCATIONS,
        ),
        count: 2,
    });
    let resolved = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let views = eyes.map(|camera| ViewParams {
        camera,
        target_size: UVec2::new(camera.width, camera.height),
    });
    let mut enc = device.create_command_encoder(&Default::default());
    // Per view, a frame per eye; shared center, one frame for both.
    let frames: Vec<FrameBuffers> = match sort {
        SortMode::PerView => views
            .iter()
            .map(|v| {
                let frame = FrameBuffers::new(dev, &tier, capacity);
                passes.encode_prepare(
                    &mut enc,
                    &frame,
                    draws,
                    &sort_pose(&v.camera),
                    std::slice::from_ref(v),
                );
                frame
            })
            .collect(),
        SortMode::SharedCenter => {
            let frame = FrameBuffers::new(dev, &tier, capacity);
            let [l, r] = eyes.map(|c| sort_pose(&c));
            let midpoint = SortPose {
                origin: (l.origin + r.origin) / 2.0,
                forward: (l.forward + r.forward).normalize(),
            };
            passes.encode_prepare(&mut enc, &frame, draws, &midpoint, &views);
            vec![frame]
        }
    };
    for (i, view) in views.iter().enumerate() {
        let frame = &frames[i.min(frames.len() - 1)];
        let layer = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: view.target_size.x,
                height: view.target_size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let layer_view = layer.create_view(&Default::default());
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("q451 draw"),
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
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.begin_pipeline_statistics_query(&queries, i as u32);
        passes.encode_draw(&mut pass, frame, view, None);
        pass.end_pipeline_statistics_query();
    }
    enc.resolve_query_set(&queries, 0..2, &resolved, 0);
    enc.copy_buffer_to_buffer(&resolved, 0, &read, 0, 16);
    dev.queue().submit([enc.finish()]);
    read.slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.expect("the counts map"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device finishes");
    let bytes = read.slice(..).get_mapped_range();
    let word = |i: usize| u64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().expect("8 bytes"));
    [word(0), word(1)]
}

/// Q-451 (#820): the splat draw's fragment-shader invocations over R2_random's turned
/// pairs are at most 0.9 of the pixel centres inside their boxes μ′ ± r, per view and in
/// shared-center mode: the rectangle along Σ′'s eigenvectors is drawn (Q-450).
#[test]
#[ignore = "gpu"]
fn gpu_q451_tightened_quads_run_fewer_fragments_than_the_boxes_hold() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = match statistics_device() {
        Ok(dev) => dev,
        Err(why) => {
            eprintln!("skipped: {why} (Q-451)");
            return;
        }
    };
    let (model, request) = probe("R2_random");
    let gpu_model = upload(&dev, &model, &QualityTier::conformance());
    for (name, sort) in [
        ("per view", SortMode::PerView),
        ("shared center", SortMode::SharedCenter),
    ] {
        let (mut counted, mut boxes) = (0u64, 0u64);
        for item in &request.items {
            let eyes = turned_eyes(&item.camera);
            let draw = ModelDraw {
                model: &gpu_model,
                pose: Pose::IDENTITY,
                tau: item.timestamp,
                mode: RenderMode::Splats,
                transform: ColorTransform {
                    gain: ftgsxr_core::Vec3::ONE,
                },
                sh_degree: request.sh_degree.unwrap_or(model.sh_degree),
                overlays: Overlays::default(),
                trim: ftgsxr_core::Trim::default(),
            };
            let counts = invocations(&dev, &[draw], &eyes, sort);
            for (eye, n) in eyes.iter().zip(counts) {
                counted += n;
                boxes += reference(&model, item.timestamp, eye);
            }
        }
        let messages = dev.take_validation_messages();
        assert!(messages.is_empty(), "validation messages: {messages:?}");
        assert!(boxes > 0, "{name}: the turned pairs see the model");
        let share = counted as f64 / boxes as f64;
        eprintln!(
            "{name}: {counted} fragment invocations against {boxes} box pixel centres ({share:.3})"
        );
        assert!(
            share <= MAX_SHARE,
            "{name}: {counted} fragment invocations against {boxes} box pixel centres \
             ({share:.3} > {MAX_SHARE}): the tightened quad is not drawn (Q-450, Q-451)"
        );
    }
}
