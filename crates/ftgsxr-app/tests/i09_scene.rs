// SPDX-License-Identifier: Apache-2.0

//! I-9 (`cpu`; source of truth §1.5 Scene files, §1.11 Scene files, Q-3, Q-54, Q-245): saving
//! then loading a scene restores the instances, their poses, the clock and the
//! environment settings; a model under the models directory is written relative to it and
//! one outside as given; a scene naming a missing model, or a model whose file is now
//! another model, lists it as missing; replacing a missing model with a file of another
//! identity gives that file the scene's attributes, and the next save records the new
//! identity (§1.5 Library, spec revision 26, Q-255).

use crate::support;

use bevy::ecs::message::Messages;
use bevy::prelude::*;
use ftgsxr_app::{
    EntryState, Intent, Library, LibraryCommand, LibraryFailure, LoadedModel, ModelRef,
    PersistCommand, Playback, PlaybackCommand, Poses, ReplaceEntry, SettingsRes, scene_from_world,
};
use ftgsxr_core::{EndRule, LoopMode, Pose, Preset};
use ftgsxr_io::{ManualLighting, SavedPose, SceneFile, read_scene};
use ftgsxr_render::{RenderMode, SplatMode, SplatTransform};
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

/// A one-model scene file's JSON (§1.11): `location` and `lighting` as JSON values.
fn scene_json(location: &str, id: &str, name: &str, x: f32, lighting: &str) -> String {
    format!(
        r#"{{"format": "ftgsxr-scene", "version": 1,
        "models": [{{"source": {{"kind": "dir", "location": {location}}},
            "identity": "{id}", "name": "{name}",
            "pose": {{"translation": [{x}, 0, 0], "rotation": [0, 0, 0, 1], "scale": 1}},
            "visible": true, "mode": "splats", "time_offset_s": 0, "end_rule": "hold",
            "lighting": {lighting}, "audio": null}}],
        "clock": {{"t": 0, "speed": 1, "loop": "loop", "range": null}},
        "environment": {{}}, "anchor": null, "rig": null, "camera": null, "bookmarks": []}}"#
    )
}

#[test]
fn i09_a_missing_model_is_listed_as_missing() {
    let mut t = TestApp::new("i09_missing");
    let scenes = t.dir.join("config/scenes");
    std::fs::create_dir_all(&scenes).unwrap();
    let id = "ab".repeat(32);
    let json = scene_json(r#""gone.safetensors""#, &id, "gone", 0.0, "null");
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
    let json = scene_json(r#""moved.safetensors""#, &id, "moved", 0.5, "null");
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

/// A scene naming P1 at its fixture path with the given `lighting` JSON, loaded.
fn p1_lit(t: &mut TestApp, name: &str, lighting: &str) -> Entity {
    let path = fixtures_dir().join("probes/P1.safetensors");
    let id = ftgsxr_core::hex(&ftgsxr_core::sha256(&std::fs::read(&path).unwrap()));
    let scenes = t.dir.join("config/scenes");
    std::fs::create_dir_all(&scenes).unwrap();
    let location = serde_json::to_string(&path.to_string_lossy()).unwrap();
    let json = scene_json(&location, &id, "P1", 0.0, lighting);
    std::fs::write(scenes.join(format!("{name}.ftgsxr-scene.json")), json).unwrap();
    t.frames(2);
    t.app
        .world_mut()
        .write_message(PersistCommand::LoadScene(name.into()));
    wait_for(t, 1);
    entity_named(t, "P1")
}

// Q-206, Q-218, Q-250
#[test]
fn i09_an_instances_lighting_is_restored_and_saved() {
    // §1.5 Scene files: the scene records each instance's color transform (#423). Its
    // gain by hand from E14's manual form: g = 2^0.5, w = (1.1, 1.2, 0.9) / (3.2 / 3).
    let lighting = ManualLighting {
        exposure_ev: 0.5,
        temperature: 0.25,
        tint: -0.5,
    };
    let g = 2f32.sqrt() * 3.0 / 3.2;
    let want = Vec3::new(1.1 * g, 1.2 * g, 0.9 * g);
    let gain_of = |t: &mut TestApp| {
        let e = entity_named(t, "P1");
        t.app
            .world()
            .get::<SplatTransform>(e)
            .expect("a transform")
            .0
            .gain
    };
    let mut t = TestApp::new("i09_lighting");
    p1_lit(
        &mut t,
        "lit",
        r#"{"exposure_ev": 0.5, "temperature": 0.25, "tint": -0.5}"#,
    );
    let gain = gain_of(&mut t);
    assert!((gain - want).abs().max_element() < 1e-6, "{gain}");
    // Save, unload, load the saved scene back.
    t.app
        .world_mut()
        .write_message(PersistCommand::SaveScene("relit".into()));
    t.frames(1);
    let file = t.dir.join("config/scenes/relit.ftgsxr-scene.json");
    let written = read_scene(&std::fs::read(&file).expect("the scene is written")).unwrap();
    assert_eq!(written.models[0].lighting, Some(lighting.clone()));
    t.command(LibraryCommand::UnloadAll);
    t.frames(2);
    assert_eq!(count(&mut t), 0);
    t.app
        .world_mut()
        .write_message(PersistCommand::LoadScene("relit".into()));
    wait_for(&mut t, 1);
    let gain = gain_of(&mut t);
    assert!((gain - want).abs().max_element() < 1e-6, "{gain}");
    let saved = scene_from_world(t.app.world());
    assert_eq!(saved.models[0].lighting, Some(lighting));
}

// Q-206
#[test]
fn i09_a_null_lighting_loads_at_the_identity() {
    let mut t = TestApp::new("i09_unlit");
    let e = p1_lit(&mut t, "unlit", "null");
    let gain = t
        .app
        .world()
        .get::<SplatTransform>(e)
        .map_or(Vec3::ONE, |c| c.0.gain);
    assert_eq!(gain, Vec3::ONE);
    assert_eq!(scene_from_world(t.app.world()).models[0].lighting, None);
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

/// A scene file `name` naming `models` (JSON objects), with the clock at `at` seconds and
/// the playback range `range`.
fn write_scene_file(
    t: &TestApp,
    name: &str,
    models: &[String],
    at: f64,
    range: Option<(f64, f64)>,
) {
    let range = range.map_or("null".to_owned(), |(a, b)| format!("[{a}, {b}]"));
    let scenes = t.dir.join("config/scenes");
    std::fs::create_dir_all(&scenes).unwrap();
    let json = format!(
        r#"{{"format": "ftgsxr-scene", "version": 1, "models": [{}],
        "clock": {{"t": {at}, "speed": 1, "loop": "loop", "range": {range}}},
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
    // Written after the models directory's first listing, so that its first relisting
    // meets the file and the missing entry together.
    t.frames(30);
    let models = t.dir.join("models");
    std::fs::create_dir_all(&models).unwrap();
    let path = models.join("swapped.safetensors");
    std::fs::copy(fixtures_dir().join("probes/P1.safetensors"), &path).unwrap();
    let id = "ab".repeat(32);
    let scene = [scene_model("swapped.safetensors", &id, "named", 0.0)];
    write_scene_file(&t, "other", &scene, 0.0, None);
    t.frames(2);
    t.app
        .world_mut()
        .write_message(PersistCommand::LoadScene("other".into()));
    let seen = failures_over(&mut t, 120);
    assert_eq!(seen.len(), 1, "one failure: {seen:?}");
    assert_eq!(seen[0].code, "E-SCENE-MISSING");
    // Q-225's text: the scene's model, then the file that is another model.
    assert_eq!(
        seen[0].text,
        "E-SCENE-MISSING: Model named not found; locate it (swapped is another model)"
    );
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
    // The present file is listed as itself, beside the missing entry.
    let lib = t.app.world().resource::<Library>();
    let present: Vec<_> = lib
        .entries
        .iter()
        .filter(|e| e.entry.location == want[0].0 && e.state != EntryState::Missing)
        .collect();
    assert_eq!(present.len(), 1, "the file's own entry");
    assert_ne!(present[0].identity.as_deref(), Some(id.as_str()));
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
    write_scene_file(&t, "clocked", &scene, 0.75, Some((0.25, 1.5)));
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
    let range = scene_from_world(t.app.world()).clock.range;
    assert_eq!(range, Some((0.25, 1.5)), "the scene's range");
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
    write_scene_file(&t, "swapped", &scene, 0.0, None);
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

/// The soundtrack the scene of `p1_missing_with_attributes` names.
fn track(t: &TestApp) -> std::path::PathBuf {
    t.dir.join("track.ogg")
}

/// A scene naming P1 at a gone path as "moved", posed at x = 0.5, with a time offset,
/// end rule, mode, visibility, lighting and soundtrack of its own; loaded, P1 listed as
/// missing. The missing model's identity.
fn p1_missing_with_attributes(t: &mut TestApp, name: &str) -> String {
    let id = identity_of(&fixtures_dir().join("probes/P1.safetensors"));
    let audio = serde_json::to_string(&track(t).to_string_lossy()).unwrap();
    let model = format!(
        r#"{{"source": {{"kind": "dir", "location": "moved.safetensors"}},
            "identity": "{id}", "name": "moved",
            "pose": {{"translation": [0.5, 0, 0], "rotation": [0, 0, 0, 1], "scale": 1}},
            "visible": false, "mode": "centers", "time_offset_s": 0.125,
            "end_rule": "extrapolate",
            "lighting": {{"exposure_ev": 0.5, "temperature": 0.25, "tint": -0.25}},
            "audio": {audio}}}"#
    );
    write_scene_file(t, name, &[model], 0.0, None);
    t.frames(2);
    t.app
        .world_mut()
        .write_message(PersistCommand::LoadScene(name.into()));
    assert!(
        t.until(600, |app| app
            .world()
            .resource::<Library>()
            .entries
            .iter()
            .any(|e| e.state == EntryState::Missing)),
        "listed as missing"
    );
    id
}

fn missing_entries(t: &TestApp) -> usize {
    t.app
        .world()
        .resource::<Library>()
        .entries
        .iter()
        .filter(|e| e.state == EntryState::Missing)
        .count()
}

/// A saved pose for `identity` in the pose store, so that a change to it shows.
fn save_a_pose_for(t: &mut TestApp, identity: &str) {
    t.app.world_mut().resource_mut::<Poses>().0.set(
        identity,
        SavedPose {
            pose: Pose::IDENTITY,
            preset: Preset::ZUp,
            fit_size_m: 1.0,
            saved_at: "2026-11-04T00:00:00Z".into(),
        },
    );
}

// Q-255
#[test]
fn i09_replacing_a_missing_model_gives_the_file_the_scenes_attributes() {
    // §1.5 Library (revision 26): "Replace" takes a file of any identity in the missing
    // model's place, with the scene's name, pose, visibility, mode, time offset and end
    // rule, color transform and audio; the next save records its source and identity;
    // the pose store keeps the old identity's saved pose under that identity.
    let mut t = TestApp::new("i09_replace");
    let old = p1_missing_with_attributes(&mut t, "retrained");
    save_a_pose_for(&mut t, &old);
    let poses = t.app.world().resource::<Poses>().clone();
    let p2 = fixtures_dir().join("probes/P2.safetensors");
    t.app.world_mut().write_message(ReplaceEntry {
        identity: old.clone(),
        file: ModelRef::File(p2.clone()),
    });
    wait_for(&mut t, 1);
    assert_eq!(missing_entries(&t), 0, "the entry is cleared");
    let e = entity_named(&mut t, "moved");
    let w = t.app.world();
    let m = w.get::<LoadedModel>(e).unwrap();
    assert_eq!(m.identity, identity_of(&p2), "the file's own identity");
    assert_eq!(m.time.offset_s, 0.125);
    assert_eq!(m.time.end_rule, EndRule::Extrapolate);
    assert_eq!(
        m.audio,
        Some(ModelRef::File(track(&t))),
        "the scene's soundtrack"
    );
    assert!((w.get::<Transform>(e).unwrap().translation.x - 0.5).abs() < 1e-6);
    assert!(
        matches!(
            w.get::<SplatMode>(e).map(|m| m.0),
            Some(RenderMode::Centers { .. })
        ),
        "the scene's mode"
    );
    assert_eq!(w.get::<Visibility>(e), Some(&Visibility::Hidden));
    assert_eq!(*w.resource::<Poses>(), poses, "the pose store is unchanged");
    let lighting = scene_from_world(w).models[0].lighting.clone();
    assert_eq!(
        lighting,
        Some(ManualLighting {
            exposure_ev: 0.5,
            temperature: 0.25,
            tint: -0.25,
        })
    );
    // The next save records the file's source and identity.
    t.app
        .world_mut()
        .write_message(PersistCommand::SaveScene("retrained".into()));
    t.frames(1);
    let file = t.dir.join("config/scenes/retrained.ftgsxr-scene.json");
    let written = read_scene(&std::fs::read(&file).unwrap()).unwrap();
    assert_eq!(written.models.len(), 1);
    assert_eq!(written.models[0].identity, identity_of(&p2));
    assert_eq!(written.models[0].source.location, p2.to_string_lossy());
    assert_eq!(written.models[0].name, "moved");
}

// Q-255
#[test]
fn i09_a_replacement_naming_no_missing_entry_loads_nothing() {
    let mut t = TestApp::new("i09_replace_none");
    p1_missing_with_attributes(&mut t, "nothing");
    t.app.world_mut().write_message(ReplaceEntry {
        identity: "cd".repeat(32),
        file: ModelRef::File(fixtures_dir().join("probes/P2.safetensors")),
    });
    t.frames(60);
    assert_eq!(count(&mut t), 0);
    assert_eq!(missing_entries(&t), 1);
}

// Q-257
#[test]
fn i09_a_checkpoint_replaces_a_missing_model() {
    let mut t = TestApp::new("i09_replace_checkpoint");
    let old = p1_missing_with_attributes(&mut t, "ckpt");
    t.app.world_mut().write_message(ReplaceEntry {
        identity: old,
        file: ModelRef::File(fixtures_dir().join("checkpoint/tiny")),
    });
    wait_for(&mut t, 1);
    assert_eq!(missing_entries(&t), 0);
    let e = entity_named(&mut t, "moved");
    let tf = t.app.world().get::<Transform>(e).unwrap();
    assert!((tf.translation.x - 0.5).abs() < 1e-6, "the scene's pose");
}

// Q-260
#[test]
fn i09_a_checkpoints_manifest_replaces_a_missing_model() {
    // Q-257, Q-260: a checkpoint picked as its `manifest.json` replaces the model as its
    // directory does: the entry clears, the instance takes the scene's attributes and
    // its source is the directory, which the scene then records (D-51).
    let mut t = TestApp::new("i09_replace_manifest");
    let old = p1_missing_with_attributes(&mut t, "manifest");
    let dir = fixtures_dir().join("checkpoint/tiny");
    t.app.world_mut().write_message(ReplaceEntry {
        identity: old,
        file: ModelRef::File(dir.join("manifest.json")),
    });
    wait_for(&mut t, 1);
    assert_eq!(missing_entries(&t), 0);
    let e = entity_named(&mut t, "moved");
    let tf = t.app.world().get::<Transform>(e).unwrap();
    assert!((tf.translation.x - 0.5).abs() < 1e-6, "the scene's pose");
    assert_eq!(
        t.app.world().get::<LoadedModel>(e).unwrap().source,
        ModelRef::File(dir)
    );
}

// Q-257
#[test]
fn i09_a_ply_replaces_a_missing_model() {
    let mut t = TestApp::new("i09_replace_ply");
    let old = p1_missing_with_attributes(&mut t, "ply");
    t.app.world_mut().write_message(ReplaceEntry {
        identity: old,
        file: ModelRef::File(fixtures_dir().join("checkpoint/tiny.tau0.ply")),
    });
    wait_for(&mut t, 1);
    assert_eq!(missing_entries(&t), 0);
    let e = entity_named(&mut t, "moved");
    let tf = t.app.world().get::<Transform>(e).unwrap();
    assert!((tf.translation.x - 0.5).abs() < 1e-6, "the scene's pose");
}

// Q-258
#[test]
fn i09_replacing_with_the_models_own_file_is_locating_it() {
    let mut t = TestApp::new("i09_replace_own");
    let old = p1_missing_with_attributes(&mut t, "own");
    let p1 = fixtures_dir().join("probes/P1.safetensors");
    t.app.world_mut().write_message(ReplaceEntry {
        identity: old.clone(),
        file: ModelRef::File(p1),
    });
    wait_for(&mut t, 1);
    assert_eq!(missing_entries(&t), 0);
    let e = entity_named(&mut t, "moved");
    assert_eq!(t.app.world().get::<LoadedModel>(e).unwrap().identity, old);
}

// Q-258
#[test]
fn i09_replacing_with_another_missing_models_file_clears_only_the_replaced_entry() {
    // Two missing scene models, P1 as "first" and P2 as "second"; "first" is replaced by
    // P2's file: "second" stays missing, the instance is "first".
    let mut t = TestApp::new("i09_replace_other");
    let p1 = identity_of(&fixtures_dir().join("probes/P1.safetensors"));
    let p2_path = fixtures_dir().join("probes/P2.safetensors");
    let p2 = identity_of(&p2_path);
    let models = [
        scene_model("first.safetensors", &p1, "first", 0.5),
        scene_model("second.safetensors", &p2, "second", 0.75),
    ];
    write_scene_file(&t, "two", &models, 0.0, None);
    t.frames(2);
    t.app
        .world_mut()
        .write_message(PersistCommand::LoadScene("two".into()));
    assert!(t.until(600, |app| {
        app.world()
            .resource::<Library>()
            .entries
            .iter()
            .filter(|e| e.state == EntryState::Missing)
            .count()
            == 2
    }));
    t.app.world_mut().write_message(ReplaceEntry {
        identity: p1,
        file: ModelRef::File(p2_path),
    });
    wait_for(&mut t, 1);
    let lib = t.app.world().resource::<Library>();
    let left: Vec<_> = lib
        .entries
        .iter()
        .filter(|e| e.state == EntryState::Missing)
        .map(|e| e.identity.clone())
        .collect();
    assert_eq!(left, vec![Some(p2)], "the other model stays missing");
    let e = entity_named(&mut t, "first");
    let tf = t.app.world().get::<Transform>(e).unwrap();
    assert!(
        (tf.translation.x - 0.5).abs() < 1e-6,
        "the replaced model's pose"
    );
}

// Q-259
#[test]
fn i09_a_failed_replacement_keeps_the_entry_and_its_attributes() {
    let mut t = TestApp::new("i09_replace_fail");
    let old = p1_missing_with_attributes(&mut t, "broken");
    let bad = t.dir.join("bad.safetensors");
    std::fs::write(&bad, b"not a model").unwrap();
    t.app.world_mut().write_message(ReplaceEntry {
        identity: old.clone(),
        file: ModelRef::File(bad.clone()),
    });
    let failures = failures_over(&mut t, 120);
    assert!(
        failures
            .iter()
            .any(|f| f.reference == Some(ModelRef::File(bad.clone()))),
        "the failure is reported"
    );
    assert_eq!(count(&mut t), 0);
    assert_eq!(missing_entries(&t), 1, "still missing");
    // The scene's attributes are still kept for a later replacement.
    t.app.world_mut().write_message(ReplaceEntry {
        identity: old,
        file: ModelRef::File(fixtures_dir().join("probes/P2.safetensors")),
    });
    wait_for(&mut t, 1);
    let e = entity_named(&mut t, "moved");
    let tf = t.app.world().get::<Transform>(e).unwrap();
    assert!((tf.translation.x - 0.5).abs() < 1e-6, "the scene's pose");
}
