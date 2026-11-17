// SPDX-License-Identifier: Apache-2.0

//! The XR device's density-map support on Monado (`xr` lane; D-58, §2.3's note, Q-461,
//! #847): the device enables VK\_EXT\_fragment\_density\_map exactly where its physical
//! device offers the extension with `fragmentDensityMap` and
//! `fragmentDensityMapNonSubsampledImages`, and `display` reports the support it read
//! from that device once, at info. Prerequisites: `xr_support`'s.

use crate::xr_support;

use bevy::render::renderer::{RenderAdapter, RenderDevice};
use ftgsxr_render::{SortMode, density_map_support};
use ftgsxr_xr::XrPlugin;
use xr_support::{Monado, capture_log, lines, xr_app};

#[test]
#[ignore = "xr"]
fn xr_density_map_is_enabled_exactly_where_the_device_offers_it() {
    capture_log();
    let _monado = Monado::start();
    let mut app = xr_app(
        XrPlugin {
            passthrough: false,
            hands: false,
            sort: SortMode::PerView,
        },
        |_| {},
    );
    for _ in 0..3 {
        app.update();
    }
    let adapter = app.world().resource::<RenderAdapter>();
    let device = app.world().resource::<RenderDevice>();
    let s = density_map_support(adapter, device.wgpu_device()).expect("Monado's device is Vulkan");
    // The lane's record of which device it ran on.
    eprintln!("xr lane device: density map: {s}");
    assert_eq!(s.enabled, s.extension && s.map && s.non_subsampled, "{s}");
    let reported: Vec<_> = lines()
        .into_iter()
        .filter(|l| {
            l.target.starts_with("ftgsxr_xr::display")
                && l.message.starts_with("density map support: ")
        })
        .collect();
    assert_eq!(reported.len(), 1, "one support line: {reported:?}");
    assert_eq!(reported[0].message, format!("density map support: {s}"));
}
