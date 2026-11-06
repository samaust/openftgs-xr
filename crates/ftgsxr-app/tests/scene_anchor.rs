// SPDX-License-Identifier: Apache-2.0

//! The scene anchor's handshake, the app side (`cpu`, #567; source of truth §1.9 Floor,
//! shadows and anchors, §1.11 Scene files' `anchor`, Q-343, Q-T567.1). The test plays
//! the XR crate by writing `SceneAnchor`: a save under the `Headset` marker in
//! passthrough mode waits in `Creating` for `Ready(uuid)` and writes the uuid; a failure
//! or 2 s without an answer writes a null anchor and toasts; a load of a file with an
//! anchor asks for `Locating(uuid)`; `Lost` toasts "anchor not found; using the saved
//! pose"; without the marker a save writes at once with a null anchor.

use crate::support;

use std::path::{Path, PathBuf};

use ftgsxr_app::{Headset, PersistCommand, SceneAnchor, SettingsRes, ToastLevel, scene_from_world};
use ftgsxr_io::{SceneFile, read_scene, write_scene};
use support::ui::{record_toasts, toasts};
use support::{Setup, TestApp};

const LOST: &str = "anchor not found; using the saved pose";
// Q-T567.1
const NOT_CREATED: &str = "anchor not created; scene saved without one";

#[track_caller]
fn app(name: &str, headset: bool, mode: &str) -> TestApp {
    let mode = mode.to_owned();
    let mut t = TestApp::with(name, Setup::default(), move |app| {
        if headset {
            app.insert_resource(Headset);
        }
        let mut s = ftgsxr_app::Settings::default();
        s.environment.mode = mode;
        app.insert_resource(SettingsRes(s));
    });
    record_toasts(&mut t.app);
    t.frames(1);
    t
}

fn file(t: &TestApp, name: &str) -> PathBuf {
    t.dir
        .join(format!("config/scenes/{name}.ftgsxr-scene.json"))
}

fn read(path: &Path) -> SceneFile {
    read_scene(&std::fs::read(path).expect("the scene file")).expect("a valid scene")
}

fn save(t: &mut TestApp, name: &str) {
    t.app
        .world_mut()
        .write_message(PersistCommand::SaveScene(name.into()));
    t.frames(2);
}

fn anchor(t: &TestApp) -> SceneAnchor {
    t.app.world().resource::<SceneAnchor>().clone()
}

fn set_anchor(t: &mut TestApp, a: SceneAnchor) {
    *t.app.world_mut().resource_mut::<SceneAnchor>() = a;
    t.frames(2);
}

fn texts(t: &TestApp) -> Vec<(String, ToastLevel)> {
    toasts(&t.app)
        .into_iter()
        .map(|x| (x.text, x.level))
        .collect()
}

#[test]
fn a_passthrough_save_on_the_headset_waits_for_ready_and_writes_the_uuid() {
    let mut t = app("anchor_ready", true, "passthrough");
    assert_eq!(anchor(&t), SceneAnchor::None);
    save(&mut t, "a");
    assert_eq!(anchor(&t), SceneAnchor::Creating);
    let path = file(&t, "a");
    t.frames(10);
    assert!(!path.exists(), "the write waits for the anchor");
    set_anchor(&mut t, SceneAnchor::Ready("0123-abcd".into()));
    assert_eq!(read(&path).anchor.as_deref(), Some("0123-abcd"));
    assert!(texts(&t).is_empty(), "{:?}", texts(&t));
}

#[test]
fn a_save_without_the_marker_writes_at_once_with_a_null_anchor() {
    let mut t = app("anchor_desktop", false, "passthrough");
    save(&mut t, "b");
    assert_eq!(anchor(&t), SceneAnchor::None);
    assert_eq!(read(&file(&t, "b")).anchor, None);
}

#[test]
fn a_vr_mode_save_on_the_headset_writes_at_once() {
    let mut t = app("anchor_vr", true, "vr");
    save(&mut t, "c");
    assert_eq!(anchor(&t), SceneAnchor::None);
    assert_eq!(read(&file(&t, "c")).anchor, None);
}

// Q-T567.1
#[test]
fn a_failed_creation_writes_a_null_anchor_and_toasts() {
    let mut t = app("anchor_failed", true, "passthrough");
    save(&mut t, "d");
    assert_eq!(anchor(&t), SceneAnchor::Creating);
    set_anchor(&mut t, SceneAnchor::None);
    assert_eq!(read(&file(&t, "d")).anchor, None);
    assert_eq!(texts(&t), [(NOT_CREATED.to_owned(), ToastLevel::Warning)]);
}

// Q-T567.1: 2 s at the test's 60 Hz frames.
#[test]
fn two_seconds_without_an_answer_write_a_null_anchor_and_toast() {
    let mut t = app("anchor_timeout", true, "passthrough");
    save(&mut t, "e");
    let path = file(&t, "e");
    // 1.9 s.
    t.frames(112);
    assert!(!path.exists(), "still waiting before 2 s");
    assert_eq!(anchor(&t), SceneAnchor::Creating);
    t.frames(10);
    assert_eq!(read(&path).anchor, None);
    assert_eq!(anchor(&t), SceneAnchor::None);
    assert_eq!(texts(&t), [(NOT_CREATED.to_owned(), ToastLevel::Warning)]);
}

#[test]
fn loading_an_anchored_scene_on_the_headset_locates_it_and_lost_toasts() {
    let mut t = app("anchor_load", true, "passthrough");
    let path = file(&t, "f");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut scene = scene_from_world(t.app.world());
    scene.anchor = Some("feed-beef".into());
    write_scene(&path, &scene).unwrap();
    t.app
        .world_mut()
        .write_message(PersistCommand::LoadScene("f".into()));
    t.frames(2);
    assert_eq!(anchor(&t), SceneAnchor::Locating("feed-beef".into()));
    set_anchor(&mut t, SceneAnchor::Lost("feed-beef".into()));
    t.frames(5);
    assert_eq!(texts(&t), [(LOST.to_owned(), ToastLevel::Warning)]);
}

#[test]
fn loading_an_anchored_scene_without_the_marker_leaves_the_anchor_alone() {
    let mut t = app("anchor_load_desktop", false, "vr");
    let path = file(&t, "g");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let mut scene = scene_from_world(t.app.world());
    scene.anchor = Some("feed-beef".into());
    write_scene(&path, &scene).unwrap();
    t.app
        .world_mut()
        .write_message(PersistCommand::LoadScene("g".into()));
    t.frames(2);
    assert_eq!(anchor(&t), SceneAnchor::None);
}
