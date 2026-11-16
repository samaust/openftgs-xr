// SPDX-License-Identifier: Apache-2.0

//! `density_map_support` (Q-461, D-58, Appendix E): the XR device's
//! VK\_EXT\_fragment\_density\_map features and properties, read through wgpu-hal from a
//! Vulkan adapter and its device. On any Vulkan adapter it gives a report; the Vulkan
//! rules it must keep (the extension's specification): no feature without the
//! extension, texel sizes powers of two with the minimum at most the maximum when the
//! extension is present, and the extension never enabled on a device that did not ask
//! for it (wgpu requests no density map).
// Q-461

use ftgsxr_render::density_map_support;
use ftgsxr_testkit::layer_messages;

#[test]
#[ignore = "gpu"]
fn gpu_density_map_support_reports_the_vulkan_adapter() {
    let _layer = layer_messages(); // R-16, Q-105
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    desc.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(desc);
    let adapter = match bevy::tasks::block_on(
        instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
    ) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("skipped: no Vulkan adapter: {e} (Q-461)");
            return;
        }
    };
    let (device, _queue) =
        bevy::tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("a device");
    let s = density_map_support(&adapter, &device).expect("a Vulkan adapter gives a report");
    println!("{s}");
    assert!(!s.enabled, "wgpu enables no density map of its own");
    if !s.extension {
        assert!(!s.map && !s.dynamic && !s.non_subsampled);
    } else {
        for t in [s.min_texel, s.max_texel] {
            assert!(t[0].is_power_of_two() && t[1].is_power_of_two(), "{t:?}");
        }
        assert!(s.min_texel[0] <= s.max_texel[0] && s.min_texel[1] <= s.max_texel[1]);
    }
    let line = s.to_string();
    for word in [
        "VK_EXT_fragment_density_map",
        "fragmentDensityMap ",
        "fragmentDensityMapDynamic",
        "fragmentDensityMapNonSubsampledImages",
        "minFragmentDensityTexelSize",
        "maxFragmentDensityTexelSize",
        "VK_EXT_fragment_density_map2",
        "VK_QCOM_fragment_density_map_offset",
        "VK_EXT_fragment_density_map_offset",
    ] {
        assert!(line.contains(word), "{word} in {line}");
    }
}
