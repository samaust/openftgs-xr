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

const SHADER: &str = "
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = array(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4(p[i], 0.0, 1.0);
}
@fragment fn fs() -> @location(0) vec4<f32> { return vec4(1.0, 0.2, 0.6, 1.0); }
";

fn pipeline(device: &wgpu::Device, fragment_density_map: bool) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fdm test"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("fdm test"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
        }),
        multiview_mask: None,
        fragment_density_map,
        cache: None,
    })
}

#[test]
#[ignore = "gpu"]
fn gpu_no_map_and_no_flag_draws_as_before() {
    let layer = layer_messages();
    let Some(adapter) = vulkan_adapter() else {
        return;
    };
    let (device, queue) =
        bevy::tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("a device");
    // A flagged pipeline needs the feature.
    let err = validation_error(&device, || {
        let _p = pipeline(&device, true);
    });
    assert!(err.is_some(), "the pipeline flag needs the feature");

    // No map and no flag: the triangle covers the target in its color, as without D-58.
    let plain = pipeline(&device, false);
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fdm test target"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fdm test read"),
        size: 64 * 64 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("fdm test pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&plain);
        pass.draw(0..3, 0..1);
    }
    enc.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &read,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(64 * 4),
                rows_per_image: None,
            },
        },
        target.size(),
    );
    queue.submit([enc.finish()]);
    read.slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.expect("mapped"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("polled");
    let bytes = read.slice(..).get_mapped_range().to_vec();
    // 1.0, 0.2, 0.6, 1.0 in Rgba8Unorm: 255, 0.2 × 255 = 51, 0.6 × 255 = 153, 255.
    for px in bytes.as_chunks::<4>().0 {
        assert_eq!(*px, [255, 51, 153, 255]);
    }
    assert_eq!(layer.take(), Vec::<String>::new(), "no validation message");
}
