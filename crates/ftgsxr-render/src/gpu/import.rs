// SPDX-License-Identifier: Apache-2.0

//! The import of a Vulkan image made outside wgpu, the runtime's environment depth
//! swapchain images (§1.9 Depth occlusion, E13), into a `wgpu::Texture` through
//! `wgpu::hal` (§2.6: `unsafe` in `ftgsxr-render` for wgpu-hal interop; Q-339).

use ftgsxr_core::UVec2;

use super::device::SplatDevice;

/// Imports the Vulkan image `image` (a raw `VkImage` handle) of `size` × `layers` array
/// layers in `format` into a `wgpu::Texture` of `dev`'s device, usable as a copy source
/// and for texture binding; its memory stays the image owner's (Q-339: once per
/// swapchain image, for `ftgsxr_xr::depth`'s conversion pass into `DepthFrame.image`,
/// Q-360).
///
/// # Safety
///
/// `image` must be a live `VkImage` of the `VkDevice` under `dev`'s wgpu device, of
/// exactly this size, layer count, one mip level, one sample and `format`, created with
/// the sampled and transfer-source usages, and initialized; it must outlive the returned
/// texture and every use of it on the GPU, and not be destroyed through it.
///
/// # Panics
///
/// When `dev`'s device is not a Vulkan device (the Quest's and the desktop's always are,
/// Q-74).
// Q-339
pub unsafe fn import_vulkan_texture(
    dev: &SplatDevice,
    image: u64,
    size: UVec2,
    layers: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    let extent = wgpu::Extent3d {
        width: size.x,
        height: size.y,
        depth_or_array_layers: layers,
    };
    let device = dev.device();
    // The pattern of bevy_mod_openxr 0.6.0's `to_wgpu_img` (MIT OR Apache-2.0,
    // source-log row 338): the hal device's guard held for `texture_from_raw` only,
    // then the hal texture handed to wgpu; unlike it, with a drop callback, so that
    // dropping the texture leaves the image alive (wgpu-hal's `destroy_texture`, row 334).
    let hal_texture = {
        // SAFETY: the guard is used only to wrap `image` below and dropped before the
        // device is used through wgpu again; nothing is destroyed through it.
        let hal_device = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
            .expect("invariant: SplatDevice is a Vulkan device (Q-74)");
        // SAFETY: ash's `vk::Image`, the first parameter, is a `#[repr(transparent)]`
        // wrapper of the `u64` handle (ash 0.38 `handle_nondispatchable!`, source-log row
        // 337), so this is its `from_raw`; the caller guarantees the image is live,
        // matches the descriptor and outlives the texture. wgpu-hal destroys an image
        // without a drop callback when its texture drops, so an empty callback is given,
        // and `TextureMemory::External` leaves the memory to its owner too.
        unsafe {
            hal_device.texture_from_raw(
                #[expect(
                    clippy::missing_transmute_annotations,
                    reason = "ash's vk::Image cannot be named without depending on ash (§2.3)"
                )]
                std::mem::transmute(image),
                &wgpu::hal::TextureDescriptor {
                    label: Some("ftgsxr imported image"),
                    size: extent,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_SRC,
                    memory_flags: wgpu::hal::MemoryFlags::empty(),
                    view_formats: vec![],
                },
                Some(Box::new(|| {})),
                wgpu::hal::vulkan::TextureMemory::External,
            )
        }
    };
    // SAFETY: `hal_texture` wraps an image of this device (the caller's guarantee),
    // created with this descriptor's size, format and usages and initialized.
    unsafe {
        device.create_texture_from_hal::<wgpu::hal::api::Vulkan>(
            hal_texture,
            &wgpu::TextureDescriptor {
                label: Some("ftgsxr imported image"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
        )
    }
}
