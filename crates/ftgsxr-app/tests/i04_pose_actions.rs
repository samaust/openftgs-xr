// SPDX-License-Identifier: Apache-2.0

//! The `Instance` actions I-4 drives, on `Poses` (`cpu`, added by #282; source of truth
//! §1.5 Poses, I-4, D-11, Q-18): save, reset to the saved pose, forget then reset to the
//! identity, the fit (the fit box's bottom at 0, its center 1.5 m along the head's yaw,
//! 10⁻⁴; the fit box B_c under `fit_to` content, B_m under whole), snap to floor on the
//! same box, a saved pose applied when the instance appears, and the fit's
//! hints (Q-181). `poses.json` through `PersistCommand` and `Intent::Instance`, and a
//! reload applying the file's pose, are `persist`'s (#285).

use crate::support;

use bevy::prelude::*;
use ftgsxr_app::{
    InstanceAction, Intent, LibraryCommand, LoadedModel, ManipulateSettings, ModelRef,
    PersistCommand, Poses, SettingsRes,
};
use ftgsxr_core::{Aabb, Pose, Preset};
use ftgsxr_io::{PoseStore, SavedPose};
use ftgsxr_render::{SplatAsset, SplatModel};
use ftgsxr_testkit::fixtures_dir;
use support::{Setup, StubPlatform, TestApp};

fn load(t: &mut TestApp, path: std::path::PathBuf) -> Entity {
    t.command(LibraryCommand::LoadRef(ModelRef::File(path)));
    assert!(
        t.until(600, |app| app
            .world_mut()
            .query::<&LoadedModel>()
            .iter(app.world())
            .count()
            == 1),
        "the model loads"
    );
    t.frames(1);
    let mut q = t
        .app
        .world_mut()
        .query_filtered::<Entity, With<LoadedModel>>();
    q.single(t.app.world()).unwrap()
}

fn act(t: &mut TestApp, instance: Entity, action: InstanceAction) {
    t.app
        .world_mut()
        .write_message(Intent::Instance { instance, action });
    t.frames(1);
}

fn pose(t: &TestApp, e: Entity) -> Pose {
    let tr = t.app.world().get::<Transform>(e).unwrap();
    Pose {
        translation: tr.translation,
        rotation: tr.rotation,
        scale: tr.scale.x,
    }
}

fn set(t: &mut TestApp, e: Entity, p: Pose) {
    *t.app.world_mut().get_mut::<Transform>(e).unwrap() = Transform {
        translation: p.translation,
        rotation: p.rotation,
        scale: Vec3::splat(p.scale),
    };
}

fn asset(t: &TestApp, e: Entity) -> &SplatAsset {
    let h = &t.app.world().get::<SplatModel>(e).unwrap().0;
    t.app
        .world()
        .resource::<Assets<SplatAsset>>()
        .get(h)
        .unwrap()
}

/// The fit box under the default `fit_to = "content"`: E10's B_c (§1.5 Poses).
fn bounds(t: &TestApp, e: Entity) -> Aabb {
    asset(t, e).content_bounds
}

fn identity(t: &TestApp, e: Entity) -> String {
    t.app
        .world()
        .get::<LoadedModel>(e)
        .unwrap()
        .identity
        .clone()
}

fn near(a: Pose, b: Pose) -> bool {
    (a.translation - b.translation).length() < 1e-4
        && (a.rotation * Vec3::X - b.rotation * Vec3::X).length() < 1e-4
        && (a.rotation * Vec3::Y - b.rotation * Vec3::Y).length() < 1e-4
        && (a.scale - b.scale).abs() < 1e-4
}

fn posed_corners(b: &Aabb, p: &Pose) -> Vec<Vec3> {
    (0..8)
        .map(|k| {
            p.transform_point(Vec3::new(
                if k & 1 == 0 { b.min.x } else { b.max.x },
                if k & 2 == 0 { b.min.y } else { b.max.y },
                if k & 4 == 0 { b.min.z } else { b.max.z },
            ))
        })
        .collect()
}

#[test]
fn i04_save_reset_forget() {
    let mut t = TestApp::new("i04_actions");
    let e = load(&mut t, fixtures_dir().join("probes/P1.safetensors"));
    let id = identity(&t, e);
    let moved = Pose {
        translation: Vec3::new(0.3, 0.2, -1.0),
        rotation: Quat::from_rotation_y(0.4),
        scale: 1.5,
    };
    set(&mut t, e, moved);
    act(&mut t, e, InstanceAction::SavePose);
    let saved = t
        .app
        .world()
        .resource::<Poses>()
        .0
        .get(&id)
        .cloned()
        .expect("saved");
    assert!(near(saved.pose, moved));
    assert_eq!(saved.preset, Preset::YUp);
    assert_eq!(saved.fit_size_m, 1.0);
    // Reset returns to the saved pose.
    set(&mut t, e, Pose::IDENTITY);
    act(&mut t, e, InstanceAction::ResetPose);
    assert!(near(pose(&t, e), moved));
    // Forget, then reset: the identity.
    act(&mut t, e, InstanceAction::ForgetPose);
    assert!(t.app.world().resource::<Poses>().0.get(&id).is_none());
    act(&mut t, e, InstanceAction::ResetPose);
    assert!(near(pose(&t, e), Pose::IDENTITY));
}

#[test]
fn i04_fit_and_snap_to_floor() {
    let mut t = TestApp::new("i04_fit");
    let e = load(&mut t, fixtures_dir().join("probes/P1.safetensors"));
    let b = bounds(&t, e);
    act(&mut t, e, InstanceAction::Fit);
    let p = pose(&t, e);
    // No camera: the head is the stage origin facing −z (Q-180); Q-18's rotation is
    // the preset's (Y-up) alone.
    let corners = posed_corners(&b, &p);
    let bottom = corners.iter().map(|c| c.y).fold(f32::INFINITY, f32::min);
    assert!(bottom.abs() < 1e-4, "{bottom}");
    let c = p.transform_point(b.center());
    assert!((c.x).abs() < 1e-4 && (c.z + 1.5).abs() < 1e-4, "{c}");
    assert!((p.scale * b.size().max_element() - 1.0).abs() < 1e-4);
    let q = p.rotation;
    let d = (q - Quat::IDENTITY)
        .length()
        .min((q + Quat::IDENTITY).length());
    assert!(d < 1e-4, "{q:?}");
    // Snap to floor after a lift: the bottom back on y = 0, the rest kept.
    let lifted = Pose {
        translation: p.translation + Vec3::new(0.0, 0.7, 0.0),
        ..p
    };
    set(&mut t, e, lifted);
    act(&mut t, e, InstanceAction::SnapToFloor);
    assert!(near(pose(&t, e), p));
}

#[test]
fn i04_a_saved_pose_applies_when_the_instance_appears() {
    let path = fixtures_dir().join("probes/P1.safetensors");
    // P1's identity, from a first load.
    let id = {
        let mut t = TestApp::new("i04_appear_id");
        let e = load(&mut t, path.clone());
        identity(&t, e)
    };
    let saved = Pose {
        translation: Vec3::new(1.0, 0.0, -2.0),
        rotation: Quat::IDENTITY,
        scale: 0.5,
    };
    let mut store = PoseStore::default();
    store.set(
        &id,
        SavedPose {
            pose: saved,
            preset: Preset::ZUp,
            fit_size_m: 2.0,
            saved_at: "2026-11-02T00:00:00.000000000Z".into(),
            trim: None,
        },
    );
    let mut t = TestApp::with("i04_appear", Setup::default(), |app| {
        app.insert_resource(Poses(store));
    });
    let e = load(&mut t, path);
    assert!(near(pose(&t, e), saved));
    // The saved preset and size are the instance's: a save keeps them.
    act(&mut t, e, InstanceAction::SavePose);
    let again = t
        .app
        .world()
        .resource::<Poses>()
        .0
        .get(&id)
        .cloned()
        .unwrap();
    assert_eq!((again.preset, again.fit_size_m), (Preset::ZUp, 2.0));
}

#[test]
fn i04_the_fit_hints_from_a_manifest_and_a_checkpoint() {
    // A checkpoint with the `ftgs-scene` manifest beside it: Z-up, L_fit = 2S.
    let dir = support::scratch("i04_hints_dir");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("tiny")).unwrap();
    let src = fixtures_dir().join("checkpoint");
    for f in std::fs::read_dir(src.join("tiny")).unwrap() {
        let f = f.unwrap().path();
        std::fs::copy(&f, dir.join("tiny").join(f.file_name().unwrap())).unwrap();
    }
    std::fs::copy(src.join("scene/scene.json"), dir.join("scene.json")).unwrap();
    let mut t = TestApp::new("i04_hints");
    let e = load(&mut t, dir.join("tiny"));
    let id = identity(&t, e);
    act(&mut t, e, InstanceAction::SavePose);
    let saved = t
        .app
        .world()
        .resource::<Poses>()
        .0
        .get(&id)
        .cloned()
        .unwrap();
    assert_eq!(saved.preset, Preset::ZUp);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(src.join("tiny/manifest.json")).unwrap()).unwrap();
    let s = manifest["spatial_scale"].as_f64().unwrap();
    assert!((saved.fit_size_m as f64 - (2.0 * s).max(0.25)).abs() < 1e-6);
    // The fit uses them: the longest side 2S, the model's z up.
    let b = bounds(&t, e);
    act(&mut t, e, InstanceAction::Fit);
    let p = pose(&t, e);
    assert!((p.scale * b.size().max_element() - saved.fit_size_m).abs() < 1e-4);
    assert!((p.rotation * Vec3::Z - Vec3::Y).length() < 1e-4);
}

fn poses_json(t: &TestApp) -> PoseStore {
    PoseStore::load(&t.dir.join("config/poses.json")).expect("poses.json reads")
}

#[test]
fn i04_save_writes_poses_json_and_a_reload_applies_it() {
    let path = fixtures_dir().join("probes/P1.safetensors");
    let mut t = TestApp::new("i04_json");
    let e = load(&mut t, path.clone());
    let id = identity(&t, e);
    let moved = Pose {
        translation: Vec3::new(-0.4, 0.1, -1.2),
        rotation: Quat::from_rotation_y(-0.7),
        scale: 0.8,
    };
    set(&mut t, e, moved);
    // Save through the command: `poses.json` holds the pose.
    t.app.world_mut().write_message(PersistCommand::SavePose(e));
    t.frames(3);
    let file = poses_json(&t);
    assert!(near(file.get(&id).expect("saved in the file").pose, moved));
    // Reset returns to the saved pose.
    set(&mut t, e, Pose::IDENTITY);
    act(&mut t, e, InstanceAction::ResetPose);
    assert!(near(pose(&t, e), moved));
    // Forget (the intent) leaves the file without it; reset gives the identity.
    act(&mut t, e, InstanceAction::ForgetPose);
    t.frames(2);
    assert!(poses_json(&t).get(&id).is_none());
    act(&mut t, e, InstanceAction::ResetPose);
    assert!(near(pose(&t, e), Pose::IDENTITY));
    // Save again (the intent), then a new player on the same directories applies it.
    set(&mut t, e, moved);
    act(&mut t, e, InstanceAction::SavePose);
    t.frames(2);
    assert!(near(poses_json(&t).get(&id).unwrap().pose, moved));
    let dir = t.dir.clone();
    drop(t);
    let mut again = TestApp::on("i04_json_reload", StubPlatform::new(&dir));
    let e = load(&mut again, path);
    assert!(near(pose(&again, e), moved));
    // The fit: the box's bottom at 0, its center 1.5 m along the yaw (10⁻⁴).
    let b = bounds(&again, e);
    act(&mut again, e, InstanceAction::Fit);
    let p = pose(&again, e);
    let bottom = posed_corners(&b, &p)
        .iter()
        .map(|c| c.y)
        .fold(f32::INFINITY, f32::min);
    assert!(bottom.abs() < 1e-4, "{bottom}");
    let c = p.transform_point(b.center());
    assert!(c.x.abs() < 1e-4 && (c.z + 1.5).abs() < 1e-4, "{c}");
}

#[test]
fn i04_the_fit_follows_the_heads_yaw() {
    // §1.5: the box's centre 1.5 m ahead of the head along its yaw, its bottom on the
    // floor, the instance turned by the yaw (Q-18, Q-180): a camera at yaw 40°.
    let mut t = TestApp::new("i04_fit_yaw");
    let yaw = 40f32.to_radians();
    let head = Vec3::new(0.7, 1.6, -0.4);
    t.app.world_mut().spawn((
        Camera3d::default(),
        Transform::from_translation(head).with_rotation(Quat::from_rotation_y(yaw)),
    ));
    let e = load(&mut t, fixtures_dir().join("probes/P1.safetensors"));
    let b = bounds(&t, e);
    act(&mut t, e, InstanceAction::Fit);
    let p = pose(&t, e);
    let forward = Vec3::new(-yaw.sin(), 0.0, -yaw.cos());
    let want = head + 1.5 * forward;
    let c = p.transform_point(b.center());
    assert!(
        (c.x - want.x).abs() < 1e-4 && (c.z - want.z).abs() < 1e-4,
        "{c} vs {want}"
    );
    let bottom = posed_corners(&b, &p)
        .iter()
        .map(|c| c.y)
        .fold(f32::INFINITY, f32::min);
    assert!(bottom.abs() < 1e-4, "{bottom}");
    let q = Quat::from_rotation_y(yaw);
    let d = (p.rotation - q).length().min((p.rotation + q).length());
    assert!(d < 1e-4, "{:?}", p.rotation);
}

/// I-4's fit and snap to floor on `b`: the fit puts `b`'s longest side at L_fit = 1 m,
/// its bottom on y = 0 and its center 1.5 m along −z (no camera, Q-180); after a lift,
/// snap to floor puts the bottom back.
fn assert_fit_and_snap_on(t: &mut TestApp, e: Entity, b: &Aabb) {
    act(t, e, InstanceAction::Fit);
    let p = pose(t, e);
    assert!(
        (p.scale * b.size().max_element() - 1.0).abs() < 1e-4,
        "{p:?}"
    );
    let bottom = posed_corners(b, &p)
        .iter()
        .map(|c| c.y)
        .fold(f32::INFINITY, f32::min);
    assert!(bottom.abs() < 1e-4, "{bottom}");
    let c = p.transform_point(b.center());
    assert!(c.x.abs() < 1e-4 && (c.z + 1.5).abs() < 1e-4, "{c}");
    set(
        t,
        e,
        Pose {
            translation: p.translation + Vec3::new(0.0, 0.7, 0.0),
            ..p
        },
    );
    act(t, e, InstanceAction::SnapToFloor);
    assert!(near(pose(t, e), p));
}

#[test]
fn i04_fit_to_content_uses_b_c_and_whole_uses_b_m() {
    // §1.5 Poses, §1.11 `[general] fit_to` (Q-371): eight still rows along x over 1 m,
    // the last of scale 0.5, so B_m (its 1.5 m margin) and B_c (no margin) differ in size
    // and center, and each fit lands its own box.
    let mut t = TestApp::new("i04_fit_to");
    let mut m = ftgsxr_testkit::random_model(5, 8, 0);
    for i in 0..8 {
        m.position[i] = [-0.5 + i as f32 / 7.0, 0.2, 0.0];
        m.velocity[i] = [0.0; 3];
        m.scale[i] = [0.01; 3];
        m.opacity[i] = 0.9;
    }
    m.scale[7] = [0.5; 3];
    let path = t.dir.join("fit_to.safetensors");
    std::fs::write(&path, ftgsxr_testkit::exchange_bytes(&m)).unwrap();
    let e = load(&mut t, path);
    let (whole, content) = (asset(&t, e).bounds, asset(&t, e).content_bounds);
    assert_ne!(whole, content);
    assert_eq!(
        t.app.world().resource::<SettingsRes>().0.general.fit_to,
        "content"
    );
    assert!(
        t.app
            .world()
            .resource::<ManipulateSettings>()
            .fit_to_content
    );
    assert_fit_and_snap_on(&mut t, e, &content);
    t.app
        .world_mut()
        .resource_mut::<SettingsRes>()
        .0
        .general
        .fit_to = "whole".into();
    t.frames(1);
    assert!(
        !t.app
            .world()
            .resource::<ManipulateSettings>()
            .fit_to_content
    );
    assert_fit_and_snap_on(&mut t, e, &whole);
    // The world's value goes back into the settings (Q-371).
    t.app
        .world_mut()
        .resource_mut::<ManipulateSettings>()
        .fit_to_content = true;
    t.frames(1);
    assert_eq!(
        t.app.world().resource::<SettingsRes>().0.general.fit_to,
        "content"
    );
    assert_fit_and_snap_on(&mut t, e, &content);
}
