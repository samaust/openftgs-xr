// SPDX-License-Identifier: Apache-2.0

//! Q-377 (#642): a model's SH array bound as SH ranges. On a device requested with a
//! lowered `max_storage_buffer_binding_size`, a model whose SH array needs three ranges
//! uploads and draws with no validation message, and its cull records and layer are
//! bit-identical to the same model's on a device with the adapter's limits: the same
//! values read through other bindings (`gpu`).

use ftgsxr_core::pack_for_gpu;
use ftgsxr_render::{RecordView, RenderMode, SplatDevice};

use crate::display_support::{self, Run, probe_camera};

/// The adapter's own device, with the features `SplatDevice::headless` asks for where the
/// adapter offers them, and its largest storage binding lowered to `binding_limit` bytes
/// when given.
fn device(binding_limit: Option<u64>) -> SplatDevice {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    desc.backends = wgpu::Backends::VULKAN;
    desc.flags = wgpu::InstanceFlags::from_build_config().with_env();
    let instance = wgpu::Instance::new(desc);
    let adapter = bevy::tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("an adapter");
    let mut required_limits = adapter.limits();
    if let Some(bytes) = binding_limit {
        required_limits.max_storage_buffer_binding_size = bytes;
    }
    let wanted = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::INDIRECT_FIRST_INSTANCE;
    let (device, queue) = bevy::tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("ftgsxr SH-range test"),
        required_features: adapter.features() & wanted,
        required_limits,
        ..Default::default()
    }))
    .expect("a device");
    SplatDevice::from_wgpu(device, queue, adapter.get_info())
}

/// A record's numbers as their bits, for an exact comparison in a fixed order (the slots'
/// order is the atomics').
fn record_bits(r: &RecordView) -> [u32; 12] {
    [
        r.mean_px.x.to_bits(),
        r.mean_px.y.to_bits(),
        r.conic[0].to_bits(),
        r.conic[1].to_bits(),
        r.conic[2].to_bits(),
        r.half_extent.x.to_bits(),
        r.half_extent.y.to_bits(),
        r.depth.to_bits(),
        r.color[0].to_bits(),
        r.color[1].to_bits(),
        r.color[2].to_bits(),
        r.color[3].to_bits(),
    ]
}

/// One frame of `model` at τ = 0.95 seen by the probe camera at SH degree 3: its sorted
/// record bits, its visible count and its layer's bits.
fn frame(dev: &SplatDevice, model: &ftgsxr_core::Model) -> (Vec<[u32; 12]>, u32, Vec<[u32; 4]>) {
    let gpu = display_support::upload(dev, model);
    let mut draw = display_support::draw(
        &gpu,
        0.95,
        RenderMode::Splats,
        ftgsxr_render::ColorTransform {
            gain: ftgsxr_core::Vec3::ONE,
        },
    );
    draw.sh_degree = 3;
    let run = Run {
        resolve: None,
        depth: false,
    };
    // C = N: under the lowered binding, the records' buffer fits one binding too.
    let (frame, contents) =
        display_support::render_with_records(dev, &[draw], &probe_camera(), &run, gpu.n());
    let mut records: Vec<[u32; 12]> = contents.records.iter().map(record_bits).collect();
    records.sort_unstable();
    let layer = frame.layer.iter().map(|t| t.map(f32::to_bits)).collect();
    (records, contents.visible_splats, layer)
}

/// Q-377 (#642): under a 64 KiB storage binding, 1 000 degree-3 rows in float32 (a
/// 192 000-byte SH buffer: three ranges of 65 536 bytes, the last 60 928) give the same
/// record bits and layer bits as on the adapter's limits, where one binding holds the
/// array, with no validation message on either device.
#[test]
#[ignore = "gpu"]
fn gpu_sh_ranges_give_the_same_records_and_layer() {
    let limit: u64 = 65_536;
    let n = 1000;
    let mut model = ftgsxr_testkit::random_model(642, n, 3);
    // Every row persistent and opaque enough to reach α_skip, so rows in every range are
    // drawn.
    for (d, o) in model.duration.iter_mut().zip(model.opacity.iter_mut()) {
        *d = 8.0;
        *o = o.max(0.5);
    }
    model.validate().expect("a valid model");
    let order: Vec<u32> = ftgsxr_core::TemporalIndex::build(&model).1;
    let sh_bytes = pack_for_gpu(
        &model,
        &order,
        &ftgsxr_render::QualityTier::conformance().layout,
    )
    .sh
    .len() as u64;
    assert_eq!(sh_bytes, 192_000);
    assert_eq!(sh_bytes.div_ceil(limit), 3, "three SH ranges");

    let full = device(None);
    assert!(
        full.device().limits().max_storage_buffer_binding_size >= sh_bytes,
        "one binding holds the SH array on the adapter's limits"
    );
    let low = device(Some(limit));
    assert_eq!(low.device().limits().max_storage_buffer_binding_size, limit);

    let (records, visible, layer) = frame(&full, &model);
    assert_eq!(full.take_validation_messages(), Vec::<String>::new());
    let (low_records, low_visible, low_layer) = frame(&low, &model);
    assert_eq!(low.take_validation_messages(), Vec::<String>::new());

    assert!(visible > 500, "most rows drawn: {visible}");
    assert_eq!(low_visible, visible);
    assert!(low_records == records, "the records' bits differ");
    assert!(
        layer.iter().any(|t| t[3] != 0),
        "the layer is covered somewhere"
    );
    assert!(low_layer == layer, "the layers' bits differ");
}
