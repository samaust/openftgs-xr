// SPDX-License-Identifier: Apache-2.0

//! The XR device's density-map support (D-58, Q-461; #847): what the device enabled, read
//! once from Bevy's render adapter and device through `density_map_support`, and kept
//! for `display`'s consumers (Q-455's info line, Q-456's `XrDisplay.foveation`). The
//! request itself needs nothing here: bevy\_mod\_openxr asks the device for every feature
//! the adapter exposes, and the vendored wgpu-hal exposes `FRAGMENT_DENSITY_MAP` only
//! where both features are offered, then enables the extension and its features (§2.3's
//! note for D-58).
// Q-463

use bevy::app::{App, Startup};
use bevy::log::info;
use bevy::prelude::{Commands, IntoScheduleConfigs, Res, Resource, not, resource_exists};
use bevy::render::renderer::{RenderAdapter, RenderDevice};
use ftgsxr_render::{DensityMapSupport, density_map_support};

/// The device's support: `None` where there is no render device yet or it is not
/// Vulkan (no report), else Q-461's report, whose `enabled`, `map` and
/// `non_subsampled` together are the signal (Q-456).
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct LayerSupport(pub(crate) Option<DensityMapSupport>);

impl LayerSupport {
    /// Whether the device enabled the extension with both features (§1.12).
    pub(crate) fn enabled(&self) -> bool {
        // Q-456
        self.0
            .as_ref()
            .is_some_and(|s| s.enabled && s.map && s.non_subsampled)
    }
}

pub(crate) fn add(app: &mut App) {
    app.add_systems(Startup, read.run_if(not(resource_exists::<LayerSupport>)));
}

fn read(
    adapter: Option<Res<RenderAdapter>>,
    device: Option<Res<RenderDevice>>,
    mut commands: Commands,
) {
    let support = adapter
        .as_deref()
        .zip(device.as_deref())
        .and_then(|(a, d)| density_map_support(a, d.wgpu_device()));
    match &support {
        Some(s) => info!("density map support: {s}"),
        None => info!("density map support: no Vulkan report"),
    }
    commands.insert_resource(LayerSupport(support));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without a render device there is no report, and the signal is false.
    #[test]
    fn layer_support_without_a_render_device_is_no_report() {
        let mut app = App::new();
        add(&mut app);
        app.update();
        let s = app.world().resource::<LayerSupport>();
        assert_eq!(*s, LayerSupport(None));
        assert!(!s.enabled());
    }

    /// Both features offered without the extension enabled is no signal.
    #[test]
    fn layer_support_without_the_extension_enabled_is_off() {
        let off = DensityMapSupport {
            extension: true,
            map: true,
            non_subsampled: true,
            ..Default::default()
        };
        assert!(!LayerSupport(Some(off)).enabled());
    }

    /// Q-456 (two-axis review, PRD #804): the signal needs the extension enabled with
    /// `fragmentDensityMap` and `fragmentDensityMapNonSubsampledImages`; the extension
    /// enabled without either feature is no signal.
    #[test]
    fn layer_support_enabled_needs_both_features() {
        let all = DensityMapSupport {
            enabled: true,
            extension: true,
            map: true,
            non_subsampled: true,
            ..Default::default()
        };
        assert!(LayerSupport(Some(all.clone())).enabled());
        let no_map = DensityMapSupport {
            map: false,
            ..all.clone()
        };
        assert!(!LayerSupport(Some(no_map)).enabled());
        let no_ns = DensityMapSupport {
            non_subsampled: false,
            ..all
        };
        assert!(!LayerSupport(Some(no_ns)).enabled());
    }
}
