// SPDX-License-Identifier: Apache-2.0

//! I-9 (`cpu`; source of truth §1.5 Scene files, §1.11 Scene files, Q-3, Q-54, Q-245): saving
//! then loading a scene restores the instances, their poses, the clock and the
//! environment settings; a model under the models directory is written relative to it and
//! one outside as given; a scene naming a missing model, or a model whose file is now
//! another model, lists it as missing.

use crate::support;

use bevy::ecs::message::Messages;
use bevy::prelude::*;
use ftgsxr_app::{
    EntryState, Intent, Library, LibraryCommand, LibraryFailure, LoadedModel, ModelRef,
    PersistCommand, Playback, PlaybackCommand, SettingsRes, scene_from_world,
};
use ftgsxr_core::{EndRule, LoopMode, Pose};
use ftgsxr_io::{SceneFile, read_scene};
use ftgsxr_render::{RenderMode, SplatMode};
use ftgsxr_testkit::fixtures_dir;
use support::TestApp;

fn count(t: &mut TestApp) -> usize {
    t.app
        .world_mut()
        .query::<&LoadedModel>()
        .iter(t.app.world())
        .count()
}

fn wait_for(t: &mut TestApp, n: usize) {
    assert!(
        t.until(600, |app| app
            .world_mut()
            .query::<&LoadedModel>()
            .iter(app.world())
            .count()
            == n),
        "{n} instances"
    );
    t.frames(3);
}

fn entity_named(t: &mut TestApp, name: &str) -> Entity {
    let mut q = t.app.world_mut().query::<(Entity, &LoadedModel)>();
    q.iter(t.app.world())
        .find(|(_, m)| m.name == name)
        .map(|(e, _)| e)
        .unwrap()
}

/// The scene with its models in a fixed order.
fn sorted(mut s: SceneFile) -> SceneFile {
    s.models.sort_by(|a, b| a.name.cmp(&b.name));
    s
}

#[test]
fn i09_save_then_load_restores_the_scene() {
    let mut t = TestApp::new("i09_round_trip");
    let models = t.dir.join("models");
    std::fs::create_dir_all(&models).unwrap();
    let inside = models.join("P1.safetensors");
    std::fs::copy(fixtures_dir().join("probes/P1.safetensors"), &inside).unwrap();
    let outside = fixtures_dir().join("probes/P2.safetensors");
    t.command(LibraryCommand::LoadRef(ModelRef::File(inside.clone())));
    t.command(LibraryCommand::LoadRef(ModelRef::File(outside.clone())));
    wait_for(&mut t, 2);
    let p1 = entity_named(&mut t, "P1");
    let p2 = entity_named(&mut t, "P2");
    *t.app.world_mut().get_mut::<Transform>(p1).unwrap() = Transform {
        translation: Vec3::new(0.5, 0.25, -1.0),
        rotation: Quat::from_rotation_y(0.3),
        scale: Vec3::splat(2.0),
    };
    let w = t.app.world_mut();
    w.write_message(Intent::Playback(PlaybackCommand::Offset {
        instance: p2,
        offset_s: 0.125,
    }));
    w.write_message(Intent::Playback(PlaybackCommand::EndRule {
        instance: p2,
        rule: EndRule::Extrapolate,
    }));
    w.write_message(Intent::Playback(PlaybackCommand::Loop(LoopMode::PingPong)));
    w.write_message(Intent::Playback(PlaybackCommand::Speed(0.5)));
    w.write_message(Intent::Playback(PlaybackCommand::Seek(0.25)));
    w.write_message(Intent::Instance {
        instance: p2,
        action: ftgsxr_app::InstanceAction::SetMode(RenderMode::Hidden),
    });
    {
        let mut s = w.resource_mut::<SettingsRes>();
        s.0.environment.background = [0.2, 0.3, 0.4];
        s.0.environment.occlusion_delta_m = 0.07;
    }
    t.frames(2);
    let before = sorted(scene_from_world(t.app.world()));
    t.app
        .world_mut()
        .write_message(PersistCommand::SaveScene("demo".into()));
    t.frames(1);
    let file = t.dir.join("config/scenes/demo.ftgsxr-scene.json");
    let written = read_scene(&std::fs::read(&file).expect("the scene is written")).unwrap();
    // Q-3: the model under `models` relative, the other as given.
    let locations: Vec<String> = sorted(written)
        .models
        .iter()
        .map(|m| m.source.location.clone())
        .collect();
    assert_eq!(
        locations,
        vec![
            "P1.safetensors".to_owned(),
            outside.to_string_lossy().into_owned()
        ]
    );
    assert_eq!(before.models.len(), 2);
    assert_eq!(before.clock.loop_mode, LoopMode::PingPong);

    // Change everything, then load the scene back.
    t.command(LibraryCommand::UnloadAll);
    t.frames(2);
    assert_eq!(count(&mut t), 0);
    t.app
        .world_mut()
        .resource_mut::<SettingsRes>()
        .0
        .environment
        .background = [0.0; 3];
    t.app
        .world_mut()
        .write_message(Intent::Playback(PlaybackCommand::Loop(LoopMode::Once)));
    t.frames(1);
    t.app
        .world_mut()
        .write_message(PersistCommand::LoadScene("demo".into()));
    wait_for(&mut t, 2);
    let after = sorted(scene_from_world(t.app.world()));
    assert_eq!(after.models, before.models);
    assert_eq!(after.clock, before.clock);
    assert_eq!(after.environment, before.environment);
    let p2 = entity_named(&mut t, "P2");
    assert_eq!(
        t.app.world().get::<SplatMode>(p2).map(|m| m.0),
        Some(RenderMode::Hidden)
    );
    let p1 = entity_named(&mut t, "P1");
    let tr = t.app.world().get::<Transform>(p1).unwrap();
    let p = Pose {
        translation: tr.translation,
        rotation: tr.rotation,
        scale: tr.scale.x,
    };
    assert_eq!(p, before.models[0].pose);
}

#[test]
fn i09_a_missing_model_is_listed_as_missing() {
    let mut t = TestApp::new("i09_missing");
    let scenes = t.dir.join("config/scenes");
    std::fs::create_dir_all(&scenes).unwrap();
    let id = "ab".repeat(32);
    let json = format!(
        r#"{{"format": "ftgsxr-scene", "version": 1,
        "models": [{{"source": {{"kind": "dir", "location": "gone.safetensors"}},
            "identity": "{id}", "name": "gone",
            "pose": {{"translation": [0, 0, 0], "rotation": [0, 0, 0, 1], "scale": 1}},
            "visible": true, "mode": "splats", "time_offset_s": 0, "end_rule": "hold",
            "lighting": null, "audio": null}}],
        "clock": {{"t": 0, "speed": 1, "loop": "loop", "range": null}},
        "environment": {{}}, "anchor": null, "rig": null, "camera": null, "bookmarks": []}}"#
    );
    std::fs::write(scenes.join("lost.ftgsxr-scene.json"), json).unwrap();
    t.frames(2);
    t.app
        .world_mut()
        .write_message(PersistCommand::LoadScene("lost".into()));
    t.frames(30);
    let gone = t.dir.join("models/gone.safetensors");
    let lib = t.app.world().resource::<Library>();
    let entry = lib
        .entries
        .iter()
        .find(|e| e.entry.location == gone.to_string_lossy())
        .expect("the missing model is listed");
    assert_eq!(entry.state, EntryState::Missing);
    assert_eq!(entry.identity.as_deref(), Some(id.as_str()));
    assert_eq!(count(&mut t), 0);
}

#[test]
fn i09_the_startup_scene_loads() {
    let dir = support::scratch("i09_startup_files");
    let scene = dir.join("start.ftgsxr-scene.json");
    let model = fixtures_dir().join("probes/P1.safetensors");
    // A scene written by a first player.
    {
        let mut t = TestApp::new("i09_startup_first");
        t.command(LibraryCommand::LoadRef(ModelRef::File(model)));
        wait_for(&mut t, 1);
        t.app
            .world_mut()
            .write_message(PersistCommand::SaveScene(scene.clone()));
        t.frames(1);
    }
    let setup = support::Setup {
        scene: Some(scene),
        ..Default::default()
    };
    let mut t = TestApp::with("i09_startup", setup, |_| {});
    wait_for(&mut t, 1);
}

/// A scene naming P1 at a path that is gone (§1.5 Scene files); its missing entry.
fn p1_missing(t: &mut TestApp, name: &str) -> String {
    let p1 = std::fs::read(fixtures_dir().join("probes/P1.safetensors")).unwrap();
    let id = ftgsxr_core::hex(&ftgsxr_core::sha256(&p1));
    let scenes = t.dir.join("config/scenes");
    std::fs::create_dir_all(&scenes).unwrap();
    let json = format!(
        r#"{{"format": "ftgsxr-scene", "version": 1,
        "models": [{{"source": {{"kind": "dir", "location": "moved.safetensors"}},
            "identity": "{id}", "name": "moved",
            "pose": {{"translation": [0.5, 0, 0], "rotation": [0, 0, 0, 1], "scale": 1}},
            "visible": true, "mode": "splats", "time_offset_s": 0, "end_rule": "hold",
            "lighting": null, "audio": null}}],
        "clock": {{"t": 0, "speed": 1, "loop": "loop", "range": null}},
        "environment": {{}}, "anchor": null, "rig": null, "camera": null, "bookmarks": []}}"#
    );
    std::fs::write(scenes.join(format!("{name}.ftgsxr-scene.json")), json).unwrap();
    t.frames(2);
    t.app
        .world_mut()
        .write_message(PersistCommand::LoadScene(name.into()));
    t.frames(30);
    let lib = t.app.world().resource::<Library>();
    assert!(
        lib.entries.iter().any(|e| e.state == EntryState::Missing),
        "the moved model is missing"
    );
    id
}

// Q-225
#[test]
fn i09_loading_the_missing_models_file_from_elsewhere_clears_the_entry() {
    // PRD #275's Library decision: a file of the missing model's identity is the model,
    // wherever it is; its load clears `EntryState::Missing` and takes the scene's pose.
    let mut t = TestApp::new("i09_locate");
    p1_missing(&mut t, "moved");
    t.command(LibraryCommand::LoadRef(ModelRef::File(
        fixtures_dir().join("probes/P1.safetensors"),
    )));
    wait_for(&mut t, 1);
    t.frames(2);
    let lib = t.app.world().resource::<Library>();
    assert!(
        !lib.entries.iter().any(|e| e.state == EntryState::Missing),
        "located"
    );
    let mut q = t.app.world_mut().query::<(&LoadedModel, &Transform)>();
    let (_, tf) = q.single(t.app.world()).unwrap();
    assert!((tf.translation.x - 0.5).abs() < 1e-6, "the scene's pose");
}

/// Runs `n` frames; every load failure the library sent meanwhile (Q-173).
fn failures_over(t: &mut TestApp, n: usize) -> Vec<LibraryFailure> {
    let mut seen = Vec::new();
    for _ in 0..n {
        t.app.update();
        let msgs = t.app.world().resource::<Messages<LibraryFailure>>();
        seen.extend(
            msgs.iter_current_update_messages()
                .filter(|f| f.reference.is_some())
                .cloned(),
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    seen
}

/// A scene file `name` naming `models` (JSON objects) with the clock at `t` seconds.
fn write_scene_file(t: &TestApp, name: &str, models: &[String], at: f64) {
    let scenes = t.dir.join("config/scenes");
    std::fs::create_dir_all(&scenes).unwrap();
    let json = format!(
        r#"{{"format": "ftgsxr-scene", "version": 1, "models": [{}],
        "clock": {{"t": {at}, "speed": 1, "loop": "loop", "range": null}},
        "environment": {{}}, "anchor": null, "rig": null, "camera": null, "bookmarks": []}}"#,
        models.join(", ")
    );
    std::fs::write(scenes.join(format!("{name}.ftgsxr-scene.json")), json).unwrap();
}

/// A scene model of `identity` at the `dir` location `location`, posed at x = `x`.
fn scene_model(location: &str, identity: &str, name: &str, x: f32) -> String {
    format!(
        r#"{{"source": {{"kind": "dir", "location": "{location}"}},
            "identity": "{identity}", "name": "{name}",
            "pose": {{"translation": [{x}, 0, 0], "rotation": [0, 0, 0, 1], "scale": 1}},
            "visible": true, "mode": "splats", "time_offset_s": 0, "end_rule": "hold",
            "lighting": null, "audio": null}}"#
    )
}

fn identity_of(path: &std::path::Path) -> String {
    ftgsxr_core::hex(&ftgsxr_core::sha256(&std::fs::read(path).unwrap()))
}

// Q-245
#[test]
fn i09_a_model_of_another_identity_is_listed_as_missing() {
    // §1.5 Scene files: a scene names its models by source and identity; the file now at
    // the source is another model, so no source offers the named one (E-SCENE-MISSING).
    let mut t = TestApp::new("i09_other_identity");
    let models = t.dir.join("models");
    std::fs::create_dir_all(&models).unwrap();
    let path = models.join("gone.safetensors");
    std::fs::copy(fixtures_dir().join("probes/P1.safetensors"), &path).unwrap();
    let id = "ab".repeat(32);
    write_scene_file(
        &t,
        "other",
        &[scene_model("gone.safetensors", &id, "gone", 0.0)],
        0.0,
    );
    t.frames(2);
    t.app
        .world_mut()
        .write_message(PersistCommand::LoadScene("other".into()));
    let seen = failures_over(&mut t, 120);
    assert_eq!(seen.len(), 1, "one failure: {seen:?}");
    assert_eq!(seen[0].code, "E-SCENE-MISSING");
    assert_eq!(count(&mut t), 0, "the other model is not kept");
    let missing = |t: &TestApp| {
        t.app
            .world()
            .resource::<Library>()
            .entries
            .iter()
            .filter(|e| e.state == EntryState::Missing)
            .map(|e| (e.entry.location.clone(), e.identity.clone()))
            .collect::<Vec<_>>()
    };
    let want = vec![(path.to_string_lossy().into_owned(), Some(id.clone()))];
    assert_eq!(missing(&t), want);
    // Q-203: kept across relistings until located, beside the present file's entry.
    let n = t.app.world().resource::<Library>().sources.len();
    for i in 0..n {
        t.command(LibraryCommand::Refresh(i));
    }
    t.frames(30);
    assert_eq!(missing(&t), want);
}

// Q-245
#[test]
fn i09_a_scene_with_a_model_of_another_identity_still_restores_its_clock() {
    // I-9: loading restores the clock; §1.5 Scene files: the matching model takes its
    // scene attributes, the other is missing and holds nothing back.
    let mut t = TestApp::new("i09_other_identity_clock");
    let models = t.dir.join("models");
    std::fs::create_dir_all(&models).unwrap();
    let p1 = models.join("P1.safetensors");
    std::fs::copy(fixtures_dir().join("probes/P1.safetensors"), &p1).unwrap();
    std::fs::copy(
        fixtures_dir().join("probes/P2.safetensors"),
        models.join("P2.safetensors"),
    )
    .unwrap();
    let scene = [
        scene_model("P1.safetensors", &identity_of(&p1), "P1", 0.5),
        scene_model("P2.safetensors", &"ab".repeat(32), "P2", 0.0),
    ];
    write_scene_file(&t, "clocked", &scene, 0.75);
    t.frames(2);
    t.app
        .world_mut()
        .write_message(PersistCommand::LoadScene("clocked".into()));
    assert!(
        t.until(600, |app| (app.world().resource::<Playback>().clock.time()
            - 0.75)
            .abs()
            < 1e-9),
        "the scene's T, {}",
        t.app.world().resource::<Playback>().clock.time()
    );
    assert_eq!(count(&mut t), 1);
    let e = entity_named(&mut t, "P1");
    let tf = t.app.world().get::<Transform>(e).unwrap();
    assert!((tf.translation.x - 0.5).abs() < 1e-6, "the scene's pose");
}

// Q-245
#[test]
fn i09_the_right_file_after_another_identity_clears_the_entry() {
    // Q-225: a load of the missing model's identity, from anywhere, is the model: it
    // clears the entry and takes the scene's attributes.
    let mut t = TestApp::new("i09_other_identity_locate");
    let models = t.dir.join("models");
    std::fs::create_dir_all(&models).unwrap();
    std::fs::copy(
        fixtures_dir().join("probes/P2.safetensors"),
        models.join("moved.safetensors"),
    )
    .unwrap();
    let p1 = fixtures_dir().join("probes/P1.safetensors");
    let scene = [scene_model(
        "moved.safetensors",
        &identity_of(&p1),
        "moved",
        0.5,
    )];
    write_scene_file(&t, "swapped", &scene, 0.0);
    t.frames(2);
    t.app
        .world_mut()
        .write_message(PersistCommand::LoadScene("swapped".into()));
    assert!(
        t.until(600, |app| app
            .world()
            .resource::<Library>()
            .entries
            .iter()
            .any(|e| e.state == EntryState::Missing)),
        "listed as missing"
    );
    t.command(LibraryCommand::LoadRef(ModelRef::File(p1)));
    wait_for(&mut t, 1);
    let lib = t.app.world().resource::<Library>();
    assert!(
        !lib.entries.iter().any(|e| e.state == EntryState::Missing),
        "located"
    );
    let mut q = t.app.world_mut().query::<(&LoadedModel, &Transform)>();
    let (m, tf) = q.single(t.app.world()).unwrap();
    assert_eq!(m.name, "moved", "the scene's name");
    assert!((tf.translation.x - 0.5).abs() < 1e-6, "the scene's pose");
}
