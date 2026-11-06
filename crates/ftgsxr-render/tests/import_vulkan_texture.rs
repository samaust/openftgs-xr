// SPDX-License-Identifier: Apache-2.0

//! `import_vulkan_texture` (Q-339, §2.6): a Vulkan image made outside wgpu's tracking, as
//! the runtime's depth swapchain images are, becomes a `wgpu::Texture` of the given size,
//! layers and format whose texels are the image's. The test stands in for the runtime
//! with an image of its own device: a two-layer `Depth16Unorm` texture (the format and
//! layer count #570's spike checks on the Quest 3) written with known values; its raw
//! handle, imported, gives back the second layer's values through a copy.

use ftgsxr_core::UVec2;
use ftgsxr_render::import_vulkan_texture;
use ftgsxr_testkit::{gpu_context, layer_messages};

#[test]
#[ignore = "gpu"]
fn gpu_import_vulkan_texture_reads_the_images_texels() {
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
        usage: wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
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
    // `u64` (source-log row T566.6).
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

    let row = (2 * w).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(row * h),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &imported,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 1 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    dev.queue().submit([enc.finish()]);
    buffer.map_async(wgpu::MapMode::Read, .., |r| r.expect("maps"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("the device polls");
    let mapped = buffer.get_mapped_range(..);
    for y in 0..h {
        for x in 0..w {
            let at = (y * row + 2 * x) as usize;
            let got = u16::from_le_bytes([mapped[at], mapped[at + 1]]);
            assert_eq!(got, (1000 + 16 * y + x) as u16, "layer 1, texel ({x}, {y})");
        }
    }
    drop(mapped);
    drop(imported);
    let messages = dev.take_validation_messages();
    assert!(messages.is_empty(), "validation messages: {messages:?}");
    drop(source);
}
