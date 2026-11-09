// SPDX-License-Identifier: Apache-2.0

//! The scene anchor's handshake, the app side (`cpu`, #567; source of truth §1.9 Floor,
//! shadows and anchors, §1.11 Scene files' `anchor`, Q-343, Q-350). The test plays
//! the XR crate by writing `SceneAnchor`: a save under the `Headset` marker in
//! passthrough mode waits in `Creating` for `Ready(uuid)` and writes the uuid; a failure
//! or 2 s without an answer writes a null anchor and toasts; a load of a file with an
//! anchor asks for `Locating(uuid)`; `Lost` toasts "anchor not found; using the saved
//! pose"; without the marker, or in VR mode, a save writes at once with a null anchor
//! and clears the state, so no earlier save's anchor is written.

use crate::support;

use std::path::{Path, PathBuf};

use ftgsxr_app::{Headset, PersistCommand, SceneAnchor, SettingsRes, ToastLevel, scene_from_world};
use ftgsxr_io::{SceneFile, read_scene, write_scene};
use support::ui::{record_toasts, toasts};
use support::{Setup, TestApp};

const LOST: &str = "anchor not found; using the saved pose";
// Q-350
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

/// §1.9: the anchor is created when a scene is saved in passthrough mode on the headset;
/// a later save in VR mode writes a null anchor, not the earlier save's, and clears it.
// Q-343
#[test]
fn a_vr_mode_save_after_an_anchored_one_writes_a_null_anchor() {
    let mut t = app("anchor_stale", true, "passthrough");
    save(&mut t, "h");
    set_anchor(&mut t, SceneAnchor::Ready("0123-abcd".into()));
    assert_eq!(read(&file(&t, "h")).anchor.as_deref(), Some("0123-abcd"));
    t.app
        .world_mut()
        .resource_mut::<SettingsRes>()
        .0
        .environment
        .mode = "vr".into();
    save(&mut t, "i");
    assert_eq!(read(&file(&t, "i")).anchor, None);
    assert_eq!(anchor(&t), SceneAnchor::None);
}

/// Q-361: a VR-mode save while a passthrough save waits for its anchor writes null at
/// once and leaves the wait alone, so the waiting file still gets its UUID.
// Q-361
#[test]
fn a_vr_mode_save_during_a_wait_keeps_the_wait() {
    let mut t = app("anchor_straddle", true, "passthrough");
    save(&mut t, "j");
    assert_eq!(anchor(&t), SceneAnchor::Creating);
    t.app
        .world_mut()
        .resource_mut::<SettingsRes>()
        .0
        .environment
        .mode = "vr".into();
    save(&mut t, "k");
    assert_eq!(read(&file(&t, "k")).anchor, None);
    assert_eq!(anchor(&t), SceneAnchor::Creating);
    set_anchor(&mut t, SceneAnchor::Ready("0123-abcd".into()));
    assert_eq!(read(&file(&t, "j")).anchor.as_deref(), Some("0123-abcd"));
    assert!(texts(&t).is_empty(), "{:?}", texts(&t));
}

/// Q-350: a second passthrough save while one waits replaces the waiting path and
/// restarts the 2 s wait; the state stays `Creating`, so the XR crate keeps the creation
/// it started at the first save and its `Ready(uuid)` goes to the second file. The
/// first file is never written. Times at the test's 60 Hz frames.
// Q-350
#[test]
fn a_second_passthrough_save_during_a_wait_replaces_the_waiting_path() {
    let mut t = app("anchor_replace", true, "passthrough");
    save(&mut t, "a");
    assert_eq!(anchor(&t), SceneAnchor::Creating);
    // 1.5 s after the first save, the second.
    t.frames(88);
    save(&mut t, "b");
    assert_eq!(anchor(&t), SceneAnchor::Creating);
    // 2.2 s after the first save, 0.7 s after the second: still waiting.
    t.frames(40);
    let (a, b) = (file(&t, "a"), file(&t, "b"));
    assert!(!b.exists(), "the wait restarted at the second save");
    assert_eq!(anchor(&t), SceneAnchor::Creating);
    assert!(texts(&t).is_empty(), "{:?}", texts(&t));
    set_anchor(&mut t, SceneAnchor::Ready("0123-abcd".into()));
    assert_eq!(read(&b).anchor.as_deref(), Some("0123-abcd"));
    assert!(!a.exists(), "the replaced path is never written");
    assert!(texts(&t).is_empty(), "{:?}", texts(&t));
}

// Q-350
#[test]
fn a_failed_creation_writes_a_null_anchor_and_toasts() {
    let mut t = app("anchor_failed", true, "passthrough");
    save(&mut t, "d");
    assert_eq!(anchor(&t), SceneAnchor::Creating);
    set_anchor(&mut t, SceneAnchor::None);
    assert_eq!(read(&file(&t, "d")).anchor, None);
    assert_eq!(texts(&t), [(NOT_CREATED.to_owned(), ToastLevel::Warning)]);
}

// Q-350: 2 s at the test's 60 Hz frames.
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

/// The XR crate's `Ready(uuid)` (written in a frame's `PostUpdate`) and `command` in the
/// next frame (#663).
fn ready_then(t: &mut TestApp, uuid: &str, command: PersistCommand) {
    *t.app.world_mut().resource_mut::<SceneAnchor>() = SceneAnchor::Ready(uuid.into());
    t.app.world_mut().write_message(command);
    t.frames(2);
}

/// Q-343, Q-350, Q-353 (#663): once `Ready(uuid)` has come the wait is over, so a
/// passthrough save of "b" in the frame after it doesn't replace the waiting path: "a"
/// is written with the UUID first, and "b" starts a wait of its own.
// Q-343, Q-350, Q-353
#[test]
fn a_passthrough_save_in_the_frame_after_ready_waits_on_its_own() {
    let mut t = app("anchor_ready_then_save", true, "passthrough");
    save(&mut t, "a");
    assert_eq!(anchor(&t), SceneAnchor::Creating);
    ready_then(&mut t, "0123-abcd", PersistCommand::SaveScene("b".into()));
    assert_eq!(read(&file(&t, "a")).anchor.as_deref(), Some("0123-abcd"));
    assert_eq!(anchor(&t), SceneAnchor::Creating);
    assert!(!file(&t, "b").exists(), "\"b\" waits for its own anchor");
    assert!(texts(&t).is_empty(), "{:?}", texts(&t));
}

/// Q-343, Q-361 (#663): a VR-mode save of "b" in the frame after `Ready(uuid)` writes
/// "b" with a null anchor, and "a", whose wait `Ready` ended, still gets the UUID.
// Q-343, Q-361
#[test]
fn a_vr_mode_save_in_the_frame_after_ready_leaves_the_uuid_to_the_waiting_file() {
    let mut t = app("anchor_ready_then_vr", true, "passthrough");
    save(&mut t, "a");
    assert_eq!(anchor(&t), SceneAnchor::Creating);
    t.app
        .world_mut()
        .resource_mut::<SettingsRes>()
        .0
        .environment
        .mode = "vr".into();
    ready_then(&mut t, "0123-abcd", PersistCommand::SaveScene("b".into()));
    assert_eq!(read(&file(&t, "a")).anchor.as_deref(), Some("0123-abcd"));
    assert_eq!(read(&file(&t, "b")).anchor, None);
    assert!(texts(&t).is_empty(), "{:?}", texts(&t));
}

/// Q-343 (#663): a load in the frame after `Ready(uuid)` comes after the waiting file is
/// written: "a" names the UUID and holds its own scene (here its background), not the
/// loaded one's.
// Q-343
#[test]
fn a_load_in_the_frame_after_ready_comes_after_the_waiting_file() {
    const LOADED: [f32; 3] = [0.25, 0.5, 0.75];
    let mut t = app("anchor_ready_then_load", true, "passthrough");
    let before = scene_from_world(t.app.world()).environment.background;
    assert_ne!(before, LOADED);
    let other = file(&t, "other");
    std::fs::create_dir_all(other.parent().unwrap()).unwrap();
    let mut scene = scene_from_world(t.app.world());
    scene.environment.background = LOADED;
    scene.anchor = None;
    write_scene(&other, &scene).unwrap();
    save(&mut t, "a");
    assert_eq!(anchor(&t), SceneAnchor::Creating);
    ready_then(
        &mut t,
        "0123-abcd",
        PersistCommand::LoadScene("other".into()),
    );
    // The load applied: the world holds the loaded scene now.
    assert_eq!(
        scene_from_world(t.app.world()).environment.background,
        LOADED
    );
    let a = read(&file(&t, "a"));
    assert_eq!(a.anchor.as_deref(), Some("0123-abcd"));
    assert_eq!(a.environment.background, before);
    assert!(texts(&t).is_empty(), "{:?}", texts(&t));
}
