// SPDX-License-Identifier: Apache-2.0

//! I-9 (`cpu`; source of truth §1.5 Scene files, §1.11 Scene files, Q-3, Q-54): saving
//! then loading a scene restores the instances, their poses, the clock and the
//! environment settings; a model under the models directory is written relative to it and
//! one outside as given; a scene naming a missing model lists it as missing.

mod support;

use bevy::prelude::*;
use ftgsxr_app::{
    EntryState, Intent, Library, LibraryCommand, LoadedModel, ModelRef, PersistCommand,
    PlaybackCommand, SettingsRes, scene_from_world,
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
