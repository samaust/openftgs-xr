// SPDX-License-Identifier: Apache-2.0

//! The vendored wgpu's fragment density map (D-58, §2.3's note for D-58, #807): the
//! device feature `FRAGMENT_DENSITY_MAP`, offered exactly where the adapter has
//! VK\_EXT\_fragment\_density\_map with `fragmentDensityMap` and
//! `fragmentDensityMapNonSubsampledImages` (as `density_map_support` reads them, Q-461);
//! the texture usage and the pipeline flag rejected on a device without the feature; and,
//! with no map and no flag, a pass that draws as before.

use ftgsxr_render::density_map_support;
use ftgsxr_testkit::layer_messages;

fn vulkan_adapter() -> Option<wgpu::Adapter> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    desc.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(desc);
    match bevy::tasks::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) {
        Ok(a) => Some(a),
        Err(e) => {
            eprintln!("skipped: no Vulkan adapter: {e} (D-58)");
            None
        }
    }
}

/// Runs `f` and returns the validation error it raised on `device`, if any.
fn validation_error(device: &wgpu::Device, f: impl FnOnce()) -> Option<wgpu::Error> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    f();
    bevy::tasks::block_on(scope.pop())
}

#[test]
#[ignore = "gpu"]
fn gpu_density_map_feature_follows_the_extension() {
    let _layer = layer_messages();
    let Some(adapter) = vulkan_adapter() else {
        return;
    };
    let (device, _queue) =
        bevy::tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("a device");
    let s = density_map_support(&adapter, &device).expect("a Vulkan report");
    let offered = adapter
        .features()
        .contains(wgpu::Features::FRAGMENT_DENSITY_MAP);
    assert_eq!(offered, s.extension && s.map && s.non_subsampled, "{s}");
    if !offered {
        eprintln!(
            "the adapter lacks VK_EXT_fragment_density_map with non-subsampled images: the map path is not exercised here (D-58)"
        );
    }
    assert!(
        !device
            .features()
            .contains(wgpu::Features::FRAGMENT_DENSITY_MAP)
    );
    let err = validation_error(&device, || {
        let _map = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("density map without the feature"),
            size: wgpu::Extent3d {
                width: 4,
                height: 4,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg8Unorm,
            usage: wgpu::TextureUsages::FRAGMENT_DENSITY_MAP | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
    });
    assert!(err.is_some(), "the density-map usage needs the feature");
}
