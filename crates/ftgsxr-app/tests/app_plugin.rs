// SPDX-License-Identifier: Apache-2.0

//! `AppPlugin`'s skeleton (§2.1 `ftgsxr_app::plugin`, #277, `cpu` lane, the `gpu_` test
//! on the `gpu` lane): it builds in the headless `App` without `RenderPlugin` and with it
//! (PRD #275's spike check 1), inserts `PlatformRes` and `SplatPlugin`'s resources, and
//! loads the settings into `SettingsRes` from `SettingsPath`, else the platform's
//! `settings.toml`, leaving a caller's `SettingsRes` alone (Q-163).

use crate::support;

use std::fs;

use bevy::render::RenderApp;
use ftgsxr_app::{Headset, HudData, PlatformRes, Settings, SettingsPath, SettingsRes};
use ftgsxr_testkit::layer_messages;
use support::{Setup, TestApp};

#[test]
fn app_plugin_builds_without_a_render_app() {
    let mut t = TestApp::new("no-render");
    assert!(t.app.get_sub_app(RenderApp).is_none());
    t.frames(3);
    assert!(t.app.world().contains_resource::<PlatformRes>());
    let paths = &t.app.world().resource::<PlatformRes>().0.paths().config;
    assert_eq!(paths, &t.dir.join("config"));
}

/// The headset marker is the caller's to insert (Q-266): `AppPlugin` never does, so a
/// desktop or headless app is not "on the headset".
// Q-266
#[test]
fn app_plugin_does_not_insert_the_headset_marker() {
    let mut t = TestApp::new("no-headset");
    t.frames(2);
    assert!(!t.app.world().contains_resource::<Headset>());
}

/// `HudData`'s XR field is the XR crate's to write (Q-274): false without it, and the
/// app's own filling leaves a written value alone.
// Q-274
#[test]
fn hud_data_s_head_tracking_is_left_to_the_xr_crate() {
    let mut t = TestApp::new("hud-xr");
    t.frames(2);
    assert!(!t.app.world().resource::<HudData>().head_tracked);
    assert_eq!(t.app.world().resource::<HudData>().left_hand, None);
    assert_eq!(t.app.world().resource::<HudData>().right_hand, None);
    t.app.world_mut().resource_mut::<HudData>().head_tracked = true;
    t.app.world_mut().resource_mut::<HudData>().right_hand =
        Some((ftgsxr_app::HudHandSource::Hand, true));
    t.frames(2);
    assert!(t.app.world().resource::<HudData>().head_tracked);
    assert_eq!(
        t.app.world().resource::<HudData>().right_hand,
        Some((ftgsxr_app::HudHandSource::Hand, true))
    );
}

/// `HudData`'s depth image and anchor state are the XR crate's to write (Q-345): `None`
/// without it, and the app's own filling leaves written values alone.
// Q-345
#[test]
fn hud_data_s_depth_and_anchor_fields_are_left_to_the_xr_crate() {
    use bevy::math::UVec2;
    use ftgsxr_app::HudAnchor;
    let mut t = TestApp::new("hud-depth-anchor");
    t.frames(2);
    assert_eq!(t.app.world().resource::<HudData>().depth_image, None);
    assert_eq!(t.app.world().resource::<HudData>().anchor, None);
    {
        let mut hud = t.app.world_mut().resource_mut::<HudData>();
        hud.depth_image = Some((UVec2::new(320, 320), 1));
        hud.anchor = Some(HudAnchor::Located);
    }
    t.frames(2);
    let hud = t.app.world().resource::<HudData>();
    assert_eq!(hud.depth_image, Some((UVec2::new(320, 320), 1)));
    assert_eq!(hud.anchor, Some(HudAnchor::Located));
}

#[test]
fn settings_default_to_the_platform_s_file() {
    let t = TestApp::new("default-settings");
    let world = t.app.world();
    assert_eq!(
        world.resource::<SettingsPath>().0,
        t.dir.join("config/settings.toml")
    );
    // No file: the defaults.
    assert_eq!(world.resource::<SettingsRes>().0, Settings::default());
}

#[test]
fn settings_path_overrides_the_file_read() {
    let dir = support::scratch("settings-path-file");
    let file = dir.join("mine.toml");
    fs::write(
        &file,
        "format = \"ftgsxr-settings\"\nversion = 1\n[general]\nlog_level = \"debug\"\n",
    )
    .unwrap();
    let path = file.clone();
    let t = TestApp::with("settings-path", Setup::default(), move |app| {
        app.insert_resource(SettingsPath(path));
    });
    let world = t.app.world();
    assert_eq!(world.resource::<SettingsPath>().0, file);
    assert_eq!(world.resource::<SettingsRes>().0.general.log_level, "debug");
}

#[test]
fn a_caller_s_settings_are_left_alone() {
    let mut mine = Settings::default();
    mine.general.ui_scale = 1.25;
    let given = mine.clone();
    let t = TestApp::with("caller-settings", Setup::default(), move |app| {
        app.insert_resource(SettingsRes(given));
    });
    assert_eq!(t.app.world().resource::<SettingsRes>().0, mine);
}

#[test]
#[ignore = "gpu"]
fn gpu_app_plugin_with_the_render_app_builds() {
    let _layer = layer_messages(); // R-16, Q-105
    let mut t = TestApp::gpu("gpu-render");
    assert!(t.app.get_sub_app(RenderApp).is_some());
    t.frames(3);
    assert!(t.app.world().contains_resource::<PlatformRes>());
}
