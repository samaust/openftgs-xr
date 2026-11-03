// SPDX-License-Identifier: Apache-2.0

//! `AppPlugin`'s skeleton (§2.1 `ftgsxr_app::plugin`, #277, `cpu` lane, the `gpu_` test
//! on the `gpu` lane): it builds in the headless `App` without `RenderPlugin` and with it
//! (PRD #275's spike check 1), inserts `PlatformRes` and `SplatPlugin`'s resources, and
//! loads the settings into `SettingsRes` from `SettingsPath`, else the platform's
//! `settings.toml`, leaving a caller's `SettingsRes` alone (Q-163).

mod support;

use std::fs;

use bevy::render::RenderApp;
use ftgsxr_app::{PlatformRes, Settings, SettingsPath, SettingsRes};
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
fn gpu_app_plugin_builds_with_the_render_app() {
    let _layer = layer_messages(); // R-16, Q-105
    let mut t = TestApp::gpu("gpu-render");
    assert!(t.app.get_sub_app(RenderApp).is_some());
    t.frames(3);
    assert!(t.app.world().contains_resource::<PlatformRes>());
}
