// SPDX-License-Identifier: Apache-2.0

//! `import_vulkan_texture` (Q-339, §2.6): a Vulkan image made outside wgpu's tracking, as
//! the runtime's depth swapchain images are, becomes a `wgpu::Texture` of the given size,
//! layers and format whose texels are the image's. The test stands in for the runtime
//! with an image of its own device: a two-layer `Depth16Unorm` texture (the format and
//! layer count #570's spike checks on the Quest 3) written with known values, without
//! the transfer-source usage the runtime's images lack (Dv-7, Q-360); its raw handle,
//! imported, gives back the second layer's values through `textureLoad` in a compute
//! pass, the sampled usage alone, as `ftgsxr_xr::depth`'s conversion pass reads it.

use ftgsxr_core::UVec2;
use ftgsxr_render::import_vulkan_texture;
use ftgsxr_testkit::{gpu_context, layer_messages};

/// Loads layer 1 of a depth array texture into `out`, one invocation per texel.
const LOAD_WGSL: &str = r#"
@group(0) @binding(0) var depth: texture_depth_2d_array;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let w = textureDimensions(depth).x;
    out[id.y * w + id.x] = textureLoad(depth, vec2<i32>(id.xy), 1, 0);
}
"#;

#[test]
#[ignore = "gpu"]
fn gpu_import_vulkan_texture_gives_the_images_texels() {
    let _layer = layer_messages(); // R-16, Q-105
    let dev = gpu_context();
    let device = dev.device();
    let (w, h, layers) = (8u32, 4u32, 2u32);
    let format = wgpu::TextureFormat::Depth16Unorm;
    let extent = wgpu::Extent3d {
        width: w,
        height: h,
        depth_or_array_layers: layers,
    };
    let source = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("stand-in swapchain image"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    // Texel (x, y) of layer l holds 1000 l + 16 y + x.
    let values: Vec<u16> = (0..layers * h * w)
        .map(|i| (1000 * (i / (w * h)) + 16 * ((i / w) % h) + i % w) as u16)
        .collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    dev.queue().write_texture(
        source.as_image_copy(),
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(2 * w),
            rows_per_image: Some(h),
        },
        extent,
    );
    dev.queue().submit([]);
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device polls");

    // SAFETY: the test's own Vulkan texture, alive until the end of the test; the raw
    // handle is read, not destroyed. ash's `vk::Image` is `#[repr(transparent)]` over a
    // `u64` (source-log row 337).
    let raw: u64 = unsafe {
        let hal = source
            .as_hal::<wgpu::hal::api::Vulkan>()
            .expect("a Vulkan texture");
        std::mem::transmute(hal.raw_handle())
    };
    // SAFETY: `raw` is a live image of `dev`'s device, of this size, layer count and
    // format, initialized above, and it outlives the import (`source` is dropped last).
    let imported = unsafe { import_vulkan_texture(&dev, raw, UVec2::new(w, h), layers, format) };
    assert_eq!(imported.size(), extent);
    assert_eq!(imported.format(), format);

    // Layer 1's texels, loaded through the sampled usage into a storage buffer.
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test depth load"),
        source: wgpu::ShaderSource::Wgsl(LOAD_WGSL.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("test depth load"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let size = u64::from(4 * w * h);
    let storage = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let view = imported.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: storage.as_entire_binding(),
            },
        ],
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(w, h, 1);
    }
    enc.copy_buffer_to_buffer(&storage, 0, &buffer, 0, size);
    dev.queue().submit([enc.finish()]);
    buffer.map_async(wgpu::MapMode::Read, .., |r| r.expect("maps"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device polls");
    let mapped = buffer.get_mapped_range(..);
    for y in 0..h {
        for x in 0..w {
            let at = (4 * (y * w + x)) as usize;
            let got = f32::from_le_bytes(mapped[at..at + 4].try_into().expect("4 bytes"));
            // Depth16Unorm loads normalized: the stored value is got × 65535.
            let got = (got * 65535.0).round() as u16;
            assert_eq!(got, (1000 + 16 * y + x) as u16, "layer 1, texel ({x}, {y})");
        }
    }
    drop(mapped);
    drop(imported);
    let messages = dev.take_validation_messages();
    assert!(messages.is_empty(), "validation messages: {messages:?}");
    drop(source);
}
