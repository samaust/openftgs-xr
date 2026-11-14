// SPDX-License-Identifier: Apache-2.0

//! Q-377 (#642): a model's SH array bound as SH ranges. On a device requested with a
//! lowered `max_storage_buffer_binding_size`, a model whose SH array needs three ranges
//! uploads and draws with no validation message, and its cull records and layer are
//! bit-identical to the same model's on a device with the adapter's limits: the same
//! values read through other bindings (`gpu`).

use ftgsxr_core::{GpuLayoutOptions, HalfFields, Model, TemporalIndex, pack_for_gpu};
use ftgsxr_render::{GpuModel, QualityTier, RecordView, RenderMode, SplatDevice};
use ftgsxr_testkit::layer_messages;

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

/// One frame of `model`, packed under `layout`, at τ = 0.95 seen by the probe camera at SH
/// degree 3 with `capacity` records: its sorted record bits, its visible count and its
/// layer's bits.
fn frame(
    dev: &SplatDevice,
    model: &Model,
    layout: &GpuLayoutOptions,
    capacity: u32,
) -> (Vec<[u32; 12]>, u32, Vec<[u32; 4]>) {
    let (index, order) = TemporalIndex::build(model);
    let arrays = pack_for_gpu(model, &order, layout);
    let gpu = GpuModel::upload(dev, &arrays, index, &model.conventions).expect("fits");
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
    let (frame, contents) =
        display_support::render_with_records(dev, &[draw], &probe_camera(), &run, capacity);
    assert!(!contents.overflow, "every visible row has a slot");
    let mut records: Vec<[u32; 12]> = contents.records.iter().map(record_bits).collect();
    records.sort_unstable();
    let layer = frame.layer.iter().map(|t| t.map(f32::to_bits)).collect();
    (records, contents.visible_splats, layer)
}

/// A random degree-3 model of `n` rows, every row persistent and opaque enough to reach
/// α_skip, so rows in every SH range are drawn.
fn persistent_model(n: usize) -> Model {
    let mut model = ftgsxr_testkit::random_model(642, n, 3);
    for (d, o) in model.duration.iter_mut().zip(model.opacity.iter_mut()) {
        *d = 8.0;
        *o = o.max(0.5);
    }
    model
}

/// `model` under `layout` gives the same record bits and layer bits on a device whose
/// largest storage binding is `limit` as on the adapter's limits, where one binding holds
/// its `sh_bytes`-byte SH array in `ranges` ranges, with no validation message on either
/// device and at least `min_visible` rows drawn.
fn assert_same_frames(
    model: &Model,
    layout: &GpuLayoutOptions,
    capacity: u32,
    limit: u64,
    (sh_bytes, ranges): (u64, u64),
    min_visible: u32,
) {
    model.validate().expect("a valid model");
    let order = TemporalIndex::build(model).1;
    let packed = pack_for_gpu(model, &order, layout).sh.len() as u64;
    assert_eq!(packed, sh_bytes);
    assert_eq!(sh_bytes.div_ceil(limit), ranges);

    let full = device(None);
    assert!(
        full.device().limits().max_storage_buffer_binding_size >= sh_bytes,
        "one binding holds the SH array on the adapter's limits"
    );
    let low = device(Some(limit));
    assert_eq!(low.device().limits().max_storage_buffer_binding_size, limit);

    let (records, visible, layer) = frame(&full, model, layout, capacity);
    assert_eq!(full.take_validation_messages(), Vec::<String>::new());
    let (low_records, low_visible, low_layer) = frame(&low, model, layout, capacity);
    assert_eq!(low.take_validation_messages(), Vec::<String>::new());

    assert!(visible >= min_visible, "rows drawn: {visible}");
    assert_eq!(low_visible, visible);
    assert!(low_records == records, "the records' bits differ");
    assert!(
        layer.iter().any(|t| t[3] != 0),
        "the layer is covered somewhere"
    );
    assert!(low_layer == layer, "the layers' bits differ");
}

/// Q-377 (#642): under a 64 KiB storage binding, 1 000 degree-3 rows in float32 (a
/// 192 000-byte SH buffer: three ranges of 65 536 bytes, the last 60 928, rows straddling
/// two) give the same record bits and layer bits as on the adapter's limits, where one
/// binding holds the array, with no validation message on either device.
#[test]
#[ignore = "gpu"]
fn gpu_sh_ranges_give_the_same_record_and_layer_bits() {
    let _layer = layer_messages(); // R-16, Q-105
    let model = persistent_model(1000);
    let layout = QualityTier::conformance().layout;
    // C = N: under the lowered binding, the records' buffer (48 bytes a slot) fits too.
    assert_same_frames(&model, &layout, 1000, 65_536, (192_000, 3), 500);
}

/// Q-377 (#642): the same in half-precision SH, two values a word: 1 800 degree-3 rows
/// (96 bytes of SH a row, 172 800 bytes: three ranges, rows straddling two) under a 64 KiB
/// binding, two rows in three moved behind the camera so the 1 200 records fit one
/// binding.
#[test]
#[ignore = "gpu"]
fn gpu_sh_ranges_in_half_precision_give_the_same_record_and_layer_bits() {
    let _layer = layer_messages(); // R-16, Q-105
    let mut model = persistent_model(1800);
    for (i, p) in model.position.iter_mut().enumerate() {
        if i % 3 != 0 {
            // The probe camera sits at y = −4 looking along +y.
            *p = [0.0, -10.0, 0.0];
        }
    }
    let layout = GpuLayoutOptions {
        half: HalfFields {
            sh: true,
            ..HalfFields::default()
        },
        ..QualityTier::conformance().layout
    };
    assert_same_frames(&model, &layout, 1200, 65_536, (172_800, 3), 300);
}
