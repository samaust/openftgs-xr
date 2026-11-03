// SPDX-License-Identifier: Apache-2.0

//! I-10 (`cpu`; source of truth §1.11 `settings.toml`, Locations, Versioning rule, Q-51,
//! Q-163) and I-18's persistence clause (§1.10, Q-188): a change is written within one
//! second and at exit, to the `SettingsPath` given; a corrupt file is set aside and the
//! defaults load; the audio values persist in the settings, in percent.

mod support;

use bevy::app::AppExit;
use bevy::prelude::*;
use ftgsxr_app::{
    Feature, Intent, Settings, SettingsPath, SettingsRes, VolumeChannel, load_settings,
};
use support::{Setup, TestApp};

fn read(path: &std::path::Path) -> Settings {
    load_settings(path).expect("the settings read").0
}

#[test]
fn i10_a_change_is_written_within_a_second_and_at_exit() {
    let target = support::scratch("i10_target").join("given/s.toml");
    let given = target.clone();
    let mut t = TestApp::with("i10_write", Setup::default(), move |app| {
        app.insert_resource(SettingsPath(given));
    });
    t.frames(3);
    assert!(!target.exists(), "nothing written without a change");
    t.app
        .world_mut()
        .resource_mut::<SettingsRes>()
        .0
        .display
        .grid = false;
    // Within 1 s (60 frames at 60 Hz).
    let mut written = false;
    for _ in 0..60 {
        t.frames(1);
        if target.exists() && !read(&target).display.grid {
            written = true;
            break;
        }
    }
    assert!(written, "written within a second");
    assert!(
        !t.dir.join("config/settings.toml").exists(),
        "only the given file"
    );
    // A second change at once waits for the second to pass, then is written.
    t.app
        .world_mut()
        .resource_mut::<SettingsRes>()
        .0
        .desktop
        .fov_deg = 75.0;
    t.frames(1);
    assert_eq!(read(&target).desktop.fov_deg, 60.0, "at most once a second");
    t.frames(61);
    assert_eq!(read(&target).desktop.fov_deg, 75.0);
    // At exit, to the given file.
    t.app
        .world_mut()
        .resource_mut::<SettingsRes>()
        .0
        .general
        .ui_scale = 1.5;
    t.app.world_mut().write_message(AppExit::Success);
    t.frames(1);
    assert_eq!(read(&target).general.ui_scale, 1.5);
}

#[test]
fn i10_written_at_exit() {
    let mut t = TestApp::new("i10_exit");
    let path = t.dir.join("config/settings.toml");
    t.frames(2);
    t.app
        .world_mut()
        .resource_mut::<SettingsRes>()
        .0
        .general
        .ui_scale = 1.25;
    t.frames(1);
    t.app
        .world_mut()
        .resource_mut::<SettingsRes>()
        .0
        .general
        .ui_scale = 1.5;
    t.app.world_mut().write_message(AppExit::Success);
    t.frames(1);
    assert_eq!(read(&path).general.ui_scale, 1.5);
}

#[test]
fn i10_a_corrupt_file_is_set_aside_and_defaults_load() {
    let dir = support::scratch("i10_corrupt_files");
    let path = dir.join("settings.toml");
    std::fs::write(&path, "this is = = not toml [").unwrap();
    let given = path.clone();
    let t = TestApp::with("i10_corrupt", Setup::default(), move |app| {
        app.insert_resource(SettingsPath(given));
    });
    assert_eq!(
        t.app.world().resource::<SettingsRes>().0,
        Settings::default()
    );
    let aside: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("settings.toml.unknown-"))
        .collect();
    assert_eq!(aside.len(), 1, "{aside:?}");
}

#[test]
fn i18_the_audio_values_persist_in_the_settings() {
    let mut t = TestApp::new("i18_persist");
    let path = t.dir.join("config/settings.toml");
    t.frames(2);
    let w = t.app.world_mut();
    w.write_message(Intent::Volume {
        channel: VolumeChannel::Master,
        value: 0.4,
    });
    w.write_message(Intent::Volume {
        channel: VolumeChannel::Soundtrack,
        value: 0.25,
    });
    w.write_message(Intent::Toggle(Feature::Sound));
    t.frames(3);
    let audio = read(&path).audio;
    assert!((audio.master - 40.0).abs() < 1e-4, "{}", audio.master);
    assert!((audio.soundtrack - 25.0).abs() < 1e-4);
    assert!((audio.ui - 100.0).abs() < 1e-4);
    assert!(audio.muted);
    // A new player on the same settings starts with these gains.
    let dir = t.dir.clone();
    drop(t);
    let t = TestApp::on("i18_persist_again", support::StubPlatform::new(&dir));
    let mut t = t;
    t.frames(1);
    let a = t.app.world().resource::<ftgsxr_app::AudioState>();
    assert!((a.master - 0.4).abs() < 1e-6 && (a.soundtrack - 0.25).abs() < 1e-6);
    assert!(a.muted);
}
